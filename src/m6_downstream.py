"""m6_downstream.py

명세: spec/rules/scoring.md + run-matrix.md
역할: 파일럿 통과 조합 전체 downstream 실행
산출물: results/downstream_{split}_{model}_{condition}_K{K}.jsonl, results/m6_missing.json

CLI: python m6_downstream.py --config config.yaml [--force] [--smoke]

실행 조합 (run-matrix.md):
  for split in splits, model in [파일럿 통과], K in k_sweep:
    full(K무관, split·model당 1회), random_k, oracle_tool,
    retrieved_{dense_single, dense_multi, fusion_add_oracle, fusion_mult_oracle,
               fusion_add_real, fusion_mult_real}
  (bm25 는 2026-07 사용자 결정으로 제외. fusion real 은 stage2 — 파일이 있으면 함께 실행.)

규칙:
  - vLLM 미사용, m5 와 동일한 러너·프롬프트·채점 (m5_pilot 재사용 — 조건 간 완전 동일).
  - 재실행 안전: 산출물이 있으면 건너뜀 (--force 로 재생성). 조합 누락은 보간하지 않고
    m6_missing.json 에 사유와 함께 기록 (run-matrix.md "누락은 빈칸+사유").
  - 파일럿 제외 모델은 실행하지 않음 (pilot_exclusions.json).
구현: Claude Code.

# DECISION: full 조건은 K 무관이므로 K0 태그로 1회만 기록한다
#   (downstream_{split}_{model}_full_K0.jsonl). m6_analysis 가 전 K 에 대해 참조.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402
from utils.qwen_tools import build_prompt, to_openai_schema  # noqa: E402
from m5_pilot import (  # noqa: E402
    QwenRunner, RETRIEVAL_METHODS, build_candidates, make_batches, score_generation,
)

BASE_CONDITIONS = ["random_k", "oracle_tool"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_retrieved(results_dir, split, k):
    """K 별 retrieval 후보 로드. 없는 방법은 빠짐 (호출측이 누락 기록)."""
    out = {}
    for m in RETRIEVAL_METHODS:
        p = os.path.join(results_dir, f"retrieval_{split}_{m}_{k}.jsonl")
        if os.path.isfile(p):
            out[m] = {str(r["query_id"]): r["candidate_tools"] for r in _read_jsonl(p)}
    return out


def run_condition(runner, model_key, split, cond, k, queries, tools_by_id, all_ids,
                  retrieved, system_prompt, cfg, out_path):
    """한 (split, model, condition, K) 조합 실행 → jsonl 기록. 파싱 상태 Counter 반환."""
    hw = cfg.get("hardware", {})
    gen_bs = int(hw.get("batch_size_gen", 8))
    gen_btok = int(hw.get("gen_batch_tokens", 40000))

    items = []
    for q in queries:
        qid = str(q["query_id"])
        gold = list(q["gold_tools"])
        cand_ids = build_candidates(cond, qid, gold, all_ids, retrieved, k, None)
        schemas = [to_openai_schema(tools_by_id[c]) for c in cand_ids if c in tools_by_id]
        prompt = build_prompt(runner.tok, q["query"], schemas, system_prompt=system_prompt)
        items.append({"q": q, "gold": gold, "cand_ids": cand_ids, "schemas": schemas,
                      "prompt": prompt, "ptok": runner.prompt_tokens(prompt)})

    recs, cstat, done = [], Counter(), 0
    for batch in make_batches(items, gen_bs, gen_btok):
        prompts = [it["prompt"] for it in batch]
        if hasattr(runner, "generate_batch"):
            gens = runner.generate_batch(prompts)
        else:  # mock 폴백
            gens = [runner.generate(p) for p in prompts]
        for it, gen in zip(batch, gens):
            s = score_generation(gen, it["cand_ids"], it["schemas"], it["gold"], split)
            cstat[s["gen_status"]] += 1
            recs.append({"query_id": it["q"]["query_id"], "condition": cond, "k": k,
                         "n_candidates": len(it["cand_ids"]), **s, "prompt_tokens": it["ptok"]})
        done += len(batch)
        print(f"[m6] {model_key} {split} {cond} K{k}: {done}/{len(items)} (상태 {dict(cstat)})",
              flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return cstat


def run(config_path, force, runner_factory=None):
    cfg = load_config(config_path)
    splits = cfg["experiment"]["splits"]
    k_sweep = [int(k) for k in cfg["experiment"]["k_sweep"]]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    system_prompt = cfg.get("prompt", {}).get("system_instruction")

    excl_path = os.path.join(results_dir, "pilot_exclusions.json")
    if not os.path.isfile(excl_path):
        sys.exit(f"[m6] {excl_path} 없음 — M5 파일럿(verify_m5) 먼저 통과해야 한다.")
    excl = json.load(open(excl_path, encoding="utf-8"))
    kept = [mk for mk in ("weak", "strong") if mk not in excl.get("excluded_models", [])]
    if excl.get("hypothesis_2b_vs_9b_unverifiable"):
        print("[m6] 주의: 2B 제외 상태 — '2B/9B 가설 검증 불가' 플래그가 결론에 반영돼야 함.")
    print(f"[m6] 실행 모델: {kept} (제외 {excl.get('excluded_models', [])})")

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tools_by_id = {t["id"]: t for t in tools}
    all_ids = [t["id"] for t in tools]

    missing = []  # 조합 누락 기록 (보간 금지 — 사유와 함께 남긴다)
    factory = runner_factory or (lambda mid: QwenRunner(mid, cfg))
    for model_key in kept:
        mid = cfg["models"]["downstream"][model_key]
        print(f"[m6] {model_key}={mid} 로드 ...")
        runner = factory(mid)
        for split in splits:
            qpath = os.path.join(data_dir, f"queries_{split}.jsonl")
            if not os.path.isfile(qpath):
                missing.append({"split": split, "model": model_key, "condition": "*", "k": "*",
                                "reason": f"queries_{split}.jsonl 없음 (M1 먼저)"})
                continue
            queries = _read_jsonl(qpath)

            # full: K 무관 → K0 으로 1회
            out = os.path.join(results_dir, f"downstream_{split}_{model_key}_full_K0.jsonl")
            if force or not os.path.isfile(out):
                run_condition(runner, model_key, split, "full", 0, queries, tools_by_id,
                              all_ids, {}, system_prompt, cfg, out)
            else:
                print(f"[m6] 건너뜀(존재): {os.path.basename(out)}")

            for k in k_sweep:
                retrieved = load_retrieved(results_dir, split, k)
                conds = BASE_CONDITIONS + [f"retrieved_{m}" for m in RETRIEVAL_METHODS]
                for cond in conds:
                    if cond.startswith("retrieved_") and cond[len("retrieved_"):] not in retrieved:
                        missing.append({"split": split, "model": model_key, "condition": cond,
                                        "k": k, "reason": "retrieval 파일 없음 (M3/--real-prior 확인)"})
                        continue
                    out = os.path.join(results_dir,
                                       f"downstream_{split}_{model_key}_{cond}_K{k}.jsonl")
                    if not force and os.path.isfile(out):
                        print(f"[m6] 건너뜀(존재): {os.path.basename(out)}")
                        continue
                    run_condition(runner, model_key, split, cond, k, queries, tools_by_id,
                                  all_ids, retrieved, system_prompt, cfg, out)
        # 다음 모델 로드 전 GPU 메모리 반환
        del runner
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    with open(os.path.join(results_dir, "m6_missing.json"), "w", encoding="utf-8") as f:
        json.dump({"missing": missing}, f, ensure_ascii=False, indent=2)
    print(f"[m6] 완료 — 누락 {len(missing)}건 기록 (m6_missing.json). 다음: m6_analysis.py")


def _smoke():
    """mock 러너로 조합 루프·파일 기록·누락 기록 점검 (모델/GPU 불필요)."""
    print("[smoke] m6_downstream (mock runner)")
    import tempfile
    import yaml
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/out/data")
    os.makedirs(f"{d}/out/results")
    tools = [{"id": f"c__t{i}__a{i}", "name": f"t{i}", "description": "d",
              "params": [{"name": "q", "type": "string", "required": True}],
              "category": "C"} for i in range(8)]
    open(f"{d}/out/data/tools.jsonl", "w").write("\n".join(map(json.dumps, tools)) + "\n")
    for s in ("I1", "I2"):
        qs = [{"query_id": f"{s}_{i}", "query": f"do {i}", "gold_tools": [tools[i]["id"]],
               "gold_categories": ["C"]} for i in range(3)]
        open(f"{d}/out/data/queries_{s}.jsonl", "w").write("\n".join(map(json.dumps, qs)) + "\n")
        # dense_multi 만 존재 → 나머지 retrieved 는 누락으로 기록돼야 함
        rows = [{"query_id": q["query_id"], "candidate_tools": [t["id"] for t in tools[:3]]} for q in qs]
        open(f"{d}/out/results/retrieval_{s}_dense_multi_5.jsonl", "w").write(
            "\n".join(map(json.dumps, rows)) + "\n")
    json.dump({"excluded_models": ["weak"], "hypothesis_2b_vs_9b_unverifiable": True},
              open(f"{d}/out/results/pilot_exclusions.json", "w"))
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["paths"]["output_dir"] = f"{d}/out"
    cfg["experiment"]["splits"] = ["I1", "I2"]
    cfg["experiment"]["k_sweep"] = [5]
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
            names = prompt.split()[1:]
            name = names[0] if names else "none"
            return f"<tool_call>\n<function={name}>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>"
        def prompt_tokens(self, prompt):
            return len(prompt.split())

    run(f"{d}/cfg.yaml", force=True, runner_factory=lambda mid: MockRunner(mid))
    rd = f"{d}/out/results"
    # weak 제외 → strong 만. full K0 + (random_k, oracle_tool, dense_multi) × K5 × split 2
    expect = [f"downstream_{s}_strong_{c}_K{k}.jsonl"
              for s in ("I1", "I2")
              for c, k in [("full", 0), ("random_k", 5), ("oracle_tool", 5),
                           ("retrieved_dense_multi", 5)]]
    for fn in expect:
        assert os.path.isfile(os.path.join(rd, fn)), fn
    assert not any("weak" in fn for fn in os.listdir(rd) if fn.startswith("downstream_")), "weak 제외 위반"
    miss = json.load(open(f"{rd}/m6_missing.json"))["missing"]
    missed_conds = {m["condition"] for m in miss}
    assert "retrieved_dense_single" in missed_conds and "retrieved_fusion_add_real" in missed_conds, miss
    rows = _read_jsonl(os.path.join(rd, "downstream_I1_strong_full_K0.jsonl"))
    assert len(rows) == 3 and all("strict_success" in r and r["k"] == 0 for r in rows), rows[0]
    print(f"[smoke] OK — 파일 {len(expect)}개, 누락 기록 {len(miss)}건 (weak 제외 준수)")


def main():
    ap = argparse.ArgumentParser(description="M6 전체 downstream 실행 (파일럿 통과 조합)")
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
