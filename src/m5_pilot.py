"""m5_pilot.py

명세: spec/rules/scoring.md (파일럿 부분), spec/MILESTONES.md (M5)
역할: 소규모 downstream 파일럿 (I1, K=10, 전 방법, 2B+9B) + 파싱 성공률 게이트 근거 생성
산출물: results/downstream_pilot_{model}_{condition}_K{K}.jsonl, results/pilot_report.json

CLI: python m5_pilot.py --config config.yaml [--force] [--smoke]

목적: 전체(M6) 실행 전, 파이프라인·파서·프롬프트 동일성 검증. 핵심은 모델별 파싱 성공률.
규칙:
  - vLLM 미사용. transformers generate() + apply_chat_template(tools=...). greedy(config.decoding).
  - 전 조건 동일 프롬프트 템플릿·디코딩. 바뀌는 것은 candidate tool 목록(축 A)뿐.
  - seed 고정.
구현: Claude Code.

# DECISION: arg_acc 는 gold 인자 값이 데이터에 없어 null 로 둔다(파일럿은 파싱·방향 확인 목적).
#   func_acc/completeness/miss_type/parse_ok/prompt_tokens 는 gold_tools 로 계산.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402
from utils.qwen_tools import build_prompt, parse_tool_calls, classify_generation, sanitize_name, to_openai_schema  # noqa: E402
from utils.scoring import score_func, score_completeness  # noqa: E402

# 축 A 조건 + 축 B 방법 (retrieved_k 에 적용). fusion 은 M3 에서 oracle prior.
RETRIEVAL_METHODS = ["bm25", "dense_single", "dense_multi", "fusion_add_oracle", "fusion_mult_oracle"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_candidates(condition, query_id, gold_ids, all_ids, retrieved, k, rng):
    """축 A 조건별 candidate tool id 목록."""
    if condition == "full":
        return list(all_ids)
    if condition == "random_k":
        pool = [t for t in all_ids if t not in set(gold_ids)]
        rng2 = random.Random(hash((query_id, "rand")) & 0xFFFFFFFF)
        distract = rng2.sample(pool, max(0, min(k - len(gold_ids), len(pool))))
        cand = list(dict.fromkeys(list(gold_ids) + distract))
        return cand[:max(k, len(gold_ids))]
    if condition == "oracle_tool":
        pool = [t for t in all_ids if t not in set(gold_ids)]
        rng2 = random.Random(hash((query_id, "oracle")) & 0xFFFFFFFF)
        distract = rng2.sample(pool, max(0, min(k - len(gold_ids), len(pool))))
        return list(dict.fromkeys(list(gold_ids) + distract))
    if condition.startswith("retrieved_"):
        method = condition[len("retrieved_"):]
        return retrieved.get(method, {}).get(str(query_id), [])
    raise ValueError(condition)


class QwenRunner:
    """Qwen3.5 downstream 러너 (transformers generate, vLLM 미사용)."""

    def __init__(self, model_id, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hw = cfg["hardware"]
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            str(hw.get("dtype", "bfloat16")), torch.bfloat16)
        # v5: dtype 인자. device_map 로 GPU 배치.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=("auto" if torch.cuda.is_available() else None))
        self.model.eval()
        self.dec = cfg["decoding"]

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=bool(self.dec["do_sample"]),
                max_new_tokens=int(self.dec["max_new_tokens"]),
                temperature=(None if not self.dec["do_sample"] else float(self.dec["temperature"])),
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)

    def prompt_tokens(self, prompt: str) -> int:
        return len(self.tok(prompt).input_ids)


def run_model(runner, model_key, cfg, tools_by_id, all_ids, queries, retrieved, conditions, k,
              system_prompt, out_dir, split, tok_for_schema=None):
    """한 모델에 대해 전 조건×쿼리 실행, 파일 기록, 파싱 상태 집계 반환."""
    status_counter = Counter()
    per_cond_status = {}
    gold_by_q = {str(q["query_id"]): list(q["gold_tools"]) for q in queries}

    for cond in conditions:
        rng = random.Random(cfg["seed"])
        recs = []
        cstat = Counter()
        for q in queries:
            qid = str(q["query_id"])
            gold = gold_by_q[qid]
            cand_ids = build_candidates(cond, qid, gold, all_ids, retrieved, k, rng)
            schemas = [to_openai_schema(tools_by_id[c]) for c in cand_ids if c in tools_by_id]
            prompt = build_prompt(runner.tok, q["query"], schemas, system_prompt=system_prompt)
            gen = runner.generate(prompt)
            calls, parse_ok = parse_tool_calls(gen)
            status = classify_generation(gen)["status"]
            cstat[status] += 1
            status_counter[status] += 1
            # 호출 함수명 → tool id 역매핑 (candidate 내 sanitize_name 기준)
            rev = {sanitize_name(c): c for c in cand_ids}
            called_ids = [rev[c["name"]] for c in calls if c.get("name") in rev]
            fa = score_func(called_ids, gold, split)
            comp, miss = score_completeness(called_ids, gold, cand_ids)
            recs.append({
                "query_id": q["query_id"], "condition": cond, "n_candidates": len(cand_ids),
                "called_tools": called_ids, "parse_ok": parse_ok, "gen_status": status,
                "func_acc": fa, "arg_acc": None, "completeness": comp, "miss_type": miss,
                "prompt_tokens": runner.prompt_tokens(prompt),
            })
        per_cond_status[cond] = dict(cstat)
        out = os.path.join(out_dir, f"downstream_pilot_{model_key}_{cond}_K{k}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(status_counter.values())
    parse_rate = status_counter["ok"] / total if total else 0.0
    return {"model_id": runner.model_id, "n": total, "parse_rate": round(parse_rate, 4),
            "status_counts": dict(status_counter), "per_condition_status": per_cond_status}


def run(config_path, force, runner_factory=None):
    cfg = load_config(config_path)
    split = cfg["pilot"]["split"]
    k = int(cfg["pilot"]["k"])
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    system_prompt = cfg.get("prompt", {}).get("system_instruction")

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tools_by_id = {t["id"]: t for t in tools}
    all_ids = [t["id"] for t in tools]
    queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))

    # M3 retrieved candidates (split, method, K)
    retrieved = {}
    for m in RETRIEVAL_METHODS:
        fname = f"retrieval_{split}_{m}_{k}.jsonl" if not m.endswith("_oracle") else \
                f"retrieval_{split}_{m[:-7]}_oracle_{k}.jsonl"
        path = os.path.join(results_dir, fname)
        if os.path.isfile(path):
            retrieved[m] = {str(r["query_id"]): r["candidate_tools"] for r in _read_jsonl(path)}
        else:
            print(f"[m5] 경고: {fname} 없음 (M3 먼저). retrieved_{m} 건너뜀.")

    conditions = ["full", "random_k", "oracle_tool"] + [f"retrieved_{m}" for m in RETRIEVAL_METHODS if m in retrieved]

    report = {"split": split, "k": k, "conditions": conditions,
              "decoding": cfg["decoding"], "system_prompt": system_prompt, "models": {}}
    factory = runner_factory or (lambda mid: QwenRunner(mid, cfg))
    for model_key in ("weak", "strong"):
        mid = cfg["models"]["downstream"][model_key]
        print(f"[m5] {model_key}={mid} 로드·실행 ...")
        runner = factory(mid)
        report["models"][model_key] = run_model(
            runner, model_key, cfg, tools_by_id, all_ids, queries, retrieved, conditions, k,
            system_prompt, results_dir, split)
        r = report["models"][model_key]
        print(f"[m5] {model_key} 파싱 성공률 {r['parse_rate']} (상태 {r['status_counts']})")

    with open(os.path.join(results_dir, "pilot_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[m5] 완료 → pilot_report.json")


def _smoke():
    """mock 러너로 파이프라인·집계·파일기록 점검 (모델 불필요)."""
    print("[smoke] m5 (mock runner)")
    import tempfile
    import yaml
    d = tempfile.mkdtemp()
    tools = [{"id": f"c__t{i}__a{i}", "name": f"t{i}", "description": "d",
              "params": [{"name": "q", "type": "string", "required": True}],
              "category": "C"} for i in range(8)]
    os.makedirs(f"{d}/out/data"); os.makedirs(f"{d}/out/results")
    open(f"{d}/out/data/tools.jsonl", "w").write("\n".join(map(json.dumps, tools)) + "\n")
    qs = [{"query_id": f"I1_{i}", "query": f"do thing {i}", "gold_tools": [tools[i % 8]["id"]],
           "gold_categories": ["C"]} for i in range(4)]
    open(f"{d}/out/data/queries_I1.jsonl", "w").write("\n".join(map(json.dumps, qs)) + "\n")
    for m in ["bm25", "dense_single", "dense_multi"]:
        rows = [{"query_id": q["query_id"], "candidate_tools": [tools[i % 8]["id"] for i in range(3)]} for q in qs]
        open(f"{d}/out/results/retrieval_I1_{m}_10.jsonl", "w").write("\n".join(map(json.dumps, rows)) + "\n")
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["paths"]["output_dir"] = f"{d}/out"
    cfg["pilot"]["k"] = 10
    yaml.safe_dump(cfg, open(f"{d}/cfg.yaml", "w"))

    class MockRunner:
        def __init__(self, mid):
            self.model_id = mid
            class T:
                def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize):
                    return "PROMPT " + " ".join(t["function"]["name"] for t in tools)
                def __call__(self, text, **kw):
                    class O: input_ids = text.split()
                    return O()
            self.tok = T()

        def generate(self, prompt):
            # 프롬프트의 첫 tool 을 호출하는 유효한 XML (파싱 ok)
            names = prompt.split()[1:]
            name = names[0] if names else "none"
            return f"<tool_call>\n<function={name}>\n<parameter=q>\nhello\n</parameter>\n</function>\n</tool_call>"

        def prompt_tokens(self, prompt):
            return len(prompt.split())

    run(f"{d}/cfg.yaml", force=True, runner_factory=lambda mid: MockRunner(mid))
    rep = json.load(open(f"{d}/out/results/pilot_report.json"))
    assert rep["models"]["weak"]["parse_rate"] == 1.0, rep["models"]["weak"]
    assert rep["models"]["strong"]["n"] == len(rep["conditions"]) * 4
    # func_acc: mock 이 첫 candidate 호출. full 조건에서 첫 tool 이 gold 면 1.
    print(f"[smoke] OK — 조건 {len(rep['conditions'])}, parse_rate {rep['models']['weak']['parse_rate']}")


def main():
    ap = argparse.ArgumentParser(description="M5 downstream 파일럿 (2B+9B 파싱 검증)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config, args.force)


if __name__ == "__main__":
    main()
