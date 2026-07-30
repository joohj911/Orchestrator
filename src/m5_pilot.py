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

retriever 영향 귀속 (pilot_report.json 에 포함):
  - per_condition_metrics: 조건별 func_acc/exact_match/strict_success/completeness/
      recall_all/args_valid_rate/hallucinated/miss_type 집계.
  - retrieval_effect: retrieved_* 조건마다
      vs_full / vs_random_k — 같은 쿼리 짝 비교 Δ + 도움/해악 쿼리 수 (strict 기준)
      *_given_recall_hit/miss — retriever 성공/실패 시 downstream 조건부 성능.
    (조건 간 프롬프트·디코딩 동일, candidate 만 다르므로 이 짝 비교가 retriever 의 인과 효과.)

성공 기준 계층 (spec func_acc 는 유지, 엄격 보조 지표 추가):
  func_acc(관대: gold 가 호출 목록에 포함되면 성공)
    ⊃ exact_match(정확히 gold 만 호출, 난사·hallucination 실패)
    ⊃ strict_success(exact ∧ 호출이 스키마상 실행 가능 — required 충족·미정의 인자 없음·타입 OK)
  인자 '값'의 의미적 정답(진짜 실행 성공)은 gold 인자가 없어 매칭 불가 — M6 에서
  LLM judge 보강 여부 결정 (# DECISION NEEDED).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402
from utils.qwen_tools import build_prompt, parse_tool_calls, classify_generation, sanitize_name, to_openai_schema  # noqa: E402
from utils.scoring import score_func, score_completeness, recall_all, score_exact, validate_call  # noqa: E402

# 축 A 조건 + 축 B 방법 (retrieved_k 에 적용). fusion 은 oracle prior(M3) + real prior(M4 후
# m3 --real-prior 로 생성). real 파일이 아직 없으면 경고 후 해당 조건만 건너뛴다.
# bm25 는 2026-07 사용자 결정으로 실험에서 제외.
RETRIEVAL_METHODS = ["dense_single", "dense_multi",
                     "fusion_add_oracle", "fusion_mult_oracle",
                     "fusion_add_real", "fusion_mult_real"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _stable_seed(*parts) -> int:
    """쿼리별 결정적 시드. 파이썬 내장 hash() 는 프로세스마다 salt 가 달라(PYTHONHASHSEED)
    실행 간 재현이 깨지므로 md5 기반으로 고정한다."""
    h = hashlib.md5("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_candidates(condition, query_id, gold_ids, all_ids, retrieved, k, rng):
    """축 A 조건별 candidate tool id 목록."""
    if condition == "full":
        return list(all_ids)
    if condition == "random_k":
        # 순수 무작위 K개 — gold 포함 보장 없음 ("신호 없는 좁히기" 하한).
        # 정답을 강제 포함하면 oracle_tool 과 동일 조건이 되어 하한 역할을 못 한다
        # (2026-07 파일럿 실측으로 확인, PLAN.md 축 A 문구도 함께 수정).
        rng2 = random.Random(_stable_seed(query_id, "rand"))
        return rng2.sample(list(all_ids), min(k, len(all_ids)))
    if condition == "oracle_tool":
        pool = [t for t in all_ids if t not in set(gold_ids)]
        rng2 = random.Random(_stable_seed(query_id, "oracle"))
        distract = rng2.sample(pool, max(0, min(k - len(gold_ids), len(pool))))
        return list(dict.fromkeys(list(gold_ids) + distract))
    if condition.startswith("retrieved_"):
        method = condition[len("retrieved_"):]
        return retrieved.get(method, {}).get(str(query_id), [])
    raise ValueError(condition)


def score_generation(gen: str, cand_ids, schemas, gold, split) -> dict:
    """생성 텍스트 1건 채점 (m5/m6 공용). 채점 필드 dict 반환.

    엄격 지표 (spec func_acc 보완): 난사·hallucination·실행 불가 호출을 실패로.
      hallucinated: candidate 에 없는 함수명을 지어낸 호출 (full 조건에서 특히 관찰 대상).
      exact_match : 호출 집합 == gold 집합, hallucination 도 실패.
      args_valid  : gold tool 호출이 스키마상 실행 가능한 비율 (required 충족 등).
      strict_success = exact_match ∧ args_valid=1 — gold 인자 없이 잴 수 있는 최엄격 성공.
    """
    calls, parse_ok = parse_tool_calls(gen)
    status = classify_generation(gen)["status"]
    # 호출 함수명 → tool id 역매핑 (candidate 내 sanitize_name 기준)
    rev = {sanitize_name(c): c for c in cand_ids}
    called_ids = [rev[c["name"]] for c in calls if c.get("name") in rev]
    fa = score_func(called_ids, gold, split)
    comp, miss = score_completeness(called_ids, gold, cand_ids)
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    hallucinated = sum(1 for c in calls if c.get("name") not in rev)
    gold_set = set(gold)
    gold_call_valids = [
        validate_call(c.get("arguments"), schema_by_name[c["name"]])["valid"]
        for c in calls if c.get("name") in rev and rev[c["name"]] in gold_set
    ]
    args_valid = (sum(gold_call_valids) / len(gold_call_valids)) if gold_call_valids else None
    exact = float(score_exact(called_ids, gold) == 1.0 and hallucinated == 0)
    strict = float(exact == 1.0 and args_valid == 1.0)
    return {
        "called_tools": called_ids, "parse_ok": parse_ok, "gen_status": status,
        "func_acc": fa, "arg_acc": None, "completeness": comp, "miss_type": miss,
        "recall_all": recall_all(cand_ids, gold),
        "n_calls": len(calls), "hallucinated_calls": hallucinated,
        "exact_match": exact, "args_valid": args_valid, "strict_success": strict,
        # 파싱 실패(no_call/malformed) 원인 추적용 생성 원문 발췌. ok 면 None (용량 절약).
        "gen_excerpt": (gen[:1500] if status != "ok" else None),
    }


def make_batches(items, gen_bs: int, gen_btok: int):
    """토큰 예산 기반 배치 (m5/m6 공용). items 는 {'ptok': int, ...} dict 리스트."""
    batches, cur, cur_tok = [], [], 0
    for it in items:
        if cur and (len(cur) >= gen_bs or cur_tok + it["ptok"] > gen_btok):
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(it)
        cur_tok += it["ptok"]
    if cur:
        batches.append(cur)
    return batches


class QwenRunner:
    """Qwen3.5 downstream 러너 (transformers generate, vLLM 미사용)."""

    def __init__(self, model_id, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hw = cfg["hardware"]
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # 배치 생성: decoder-only 는 left padding 이어야 생성 결과가 안 깨진다.
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # 턴 종료 토큰 등록: <|im_end|> 가 eos 로 안 걸리면 모델이 가짜 후속 턴을
        # 계속 생성한다 (2026-07 파일럿 실측 — im_end 후 user 턴 이어 생성).
        eos_ids = {self.tok.eos_token_id}
        if "<|im_end|>" in (self.tok.get_vocab() or {}):
            eos_ids.add(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.eos_ids = sorted(t for t in eos_ids if isinstance(t, int) and t >= 0)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            str(hw.get("dtype", "bfloat16")), torch.bfloat16)
        # v5: dtype 인자. device_map 로 GPU 배치.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=("auto" if torch.cuda.is_available() else None))
        self.model.eval()
        self.dec = cfg["decoding"]

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        import torch
        inputs = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=bool(self.dec["do_sample"]),
                max_new_tokens=int(self.dec["max_new_tokens"]),
                temperature=(None if not self.dec["do_sample"] else float(self.dec["temperature"])),
                eos_token_id=(self.eos_ids or None),
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        start = inputs["input_ids"].shape[1]  # left padding → 전 시퀀스 동일 시작점
        return [self.tok.decode(o[start:], skip_special_tokens=True) for o in out]

    def prompt_tokens(self, prompt: str) -> int:
        return len(self.tok(prompt).input_ids)


def _aggregate(recs):
    """조건별 요약: 방향 확인용 (파일럿 게이트는 파싱 성공률이지만, retriever 영향도 리포트)."""
    n = len(recs)
    if not n:
        return {}
    miss_counts = Counter(r["miss_type"] for r in recs if r["miss_type"])
    with_valid = [r["args_valid"] for r in recs if r.get("args_valid") is not None]
    return {
        "n": n,
        "func_acc": round(sum(r["func_acc"] for r in recs) / n, 4),
        "exact_match": round(sum(r["exact_match"] for r in recs) / n, 4),
        "strict_success": round(sum(r["strict_success"] for r in recs) / n, 4),
        "completeness": round(sum(r["completeness"] for r in recs) / n, 4),
        "recall_all": round(sum(r["recall_all"] for r in recs) / n, 4),
        "args_valid_rate": round(sum(with_valid) / len(with_valid), 4) if with_valid else None,
        "mean_n_calls": round(sum(r["n_calls"] for r in recs) / n, 3),
        "mean_hallucinated_calls": round(sum(r["hallucinated_calls"] for r in recs) / n, 3),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in recs) / n, 1),
        "miss_type_counts": dict(miss_counts),
    }


def _retrieval_effect(recs_by_cond):
    """retrieved_* 조건별 retriever 영향 귀속.

    같은 쿼리·같은 모델·같은 프롬프트에서 candidate 목록만 다르므로 baseline 과의
    쿼리 단위 짝 비교가 retriever 의 인과 효과다.
      - vs_full     : retriever 를 껴서 좋아졌나/나빠졌나 (좁히기+랭킹 합산 효과).
      - vs_random_k : 단순 좁히기 대비 retriever 랭킹의 기여.
      - *_given_recall_hit/miss : retriever 가 gold 를 살렸을 때 모델이 잘 쓰는지,
        놓쳤을 때 downstream 이 같이 죽는지 (원인 귀속).
    도움/해악 판정은 strict_success 기준 (gold 포함 호출이면 무조건 성공으로 치는
    관대한 func_acc 가 아니라, 정확히 gold 만 + 실행 가능하게 호출했는지).
    """
    effects = {}
    baselines = {b: {str(r["query_id"]): r for r in recs_by_cond.get(b, [])}
                 for b in ("full", "random_k")}
    for cond, recs in recs_by_cond.items():
        if not cond.startswith("retrieved_") or not recs:
            continue
        hit = [r for r in recs if r["recall_all"] == 1]
        miss = [r for r in recs if r["recall_all"] == 0]

        def _mean(rows, key):
            return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None

        eff = {
            "recall_all": round(len(hit) / len(recs), 4),
            "func_acc_given_recall_hit": _mean(hit, "func_acc"),
            "func_acc_given_recall_miss": _mean(miss, "func_acc"),
            "strict_success_given_recall_hit": _mean(hit, "strict_success"),
            "strict_success_given_recall_miss": _mean(miss, "strict_success"),
        }
        for bname, base in baselines.items():
            if not base:
                continue
            # 도움/해악은 strict_success(난사·실행불가까지 실패로 보는 엄격 기준) 기준.
            # func_acc(spec 지표) 델타도 함께 기록해 관대/엄격 기준 차이를 드러낸다.
            helped = hurt = 0
            d_func, d_strict = [], []
            for r in recs:
                br = base.get(str(r["query_id"]))
                if br is None:
                    continue
                d_func.append(r["func_acc"] - br["func_acc"])
                ds = r["strict_success"] - br["strict_success"]
                d_strict.append(ds)
                if ds > 0:
                    helped += 1
                elif ds < 0:
                    hurt += 1
            if d_strict:
                eff[f"vs_{bname}"] = {
                    "delta_func_acc": round(sum(d_func) / len(d_func), 4),
                    "delta_strict_success": round(sum(d_strict) / len(d_strict), 4),
                    "helped_queries": helped,
                    "hurt_queries": hurt,
                    "unchanged_queries": len(d_strict) - helped - hurt,
                }
        effects[cond] = eff
    return effects


def run_model(runner, model_key, cfg, tools_by_id, all_ids, queries, retrieved, conditions, k,
              system_prompt, out_dir, split, tok_for_schema=None):
    """한 모델에 대해 전 조건×쿼리 실행, 파일 기록, 파싱 상태 집계 반환."""
    status_counter = Counter()
    per_cond_status = {}
    recs_by_cond = {}
    gold_by_q = {str(q["query_id"]): list(q["gold_tools"]) for q in queries}

    hw = cfg.get("hardware", {})
    gen_bs = int(hw.get("batch_size_gen", 8))
    gen_btok = int(hw.get("gen_batch_tokens", 40000))
    tmpl_kwargs = {}
    if cfg.get("prompt", {}).get("enable_thinking") is not None:
        tmpl_kwargs["enable_thinking"] = bool(cfg["prompt"]["enable_thinking"])

    for cond in conditions:
        rng = random.Random(cfg["seed"])
        recs = []
        cstat = Counter()
        # 1) 프롬프트 선구성 (조건 내 전 쿼리)
        items = []
        for q in queries:
            qid = str(q["query_id"])
            gold = gold_by_q[qid]
            cand_ids = build_candidates(cond, qid, gold, all_ids, retrieved, k, rng)
            schemas = [to_openai_schema(tools_by_id[c]) for c in cand_ids if c in tools_by_id]
            prompt = build_prompt(runner.tok, q["query"], schemas, system_prompt=system_prompt,
                                  template_kwargs=tmpl_kwargs or None)
            items.append({"q": q, "gold": gold, "cand_ids": cand_ids, "schemas": schemas,
                          "prompt": prompt, "ptok": runner.prompt_tokens(prompt)})
        # 2) 토큰 예산 기반 배치 (full 조건처럼 프롬프트가 길면 배치가 자동으로 작아짐)
        # 3) 배치 생성 + 채점
        done = 0
        for batch in make_batches(items, gen_bs, gen_btok):
            prompts = [it["prompt"] for it in batch]
            if hasattr(runner, "generate_batch"):
                gens = runner.generate_batch(prompts)
            else:  # mock 등 단건 러너 폴백
                gens = [runner.generate(p) for p in prompts]
            for it, gen in zip(batch, gens):
                s = score_generation(gen, it["cand_ids"], it["schemas"], it["gold"], split)
                cstat[s["gen_status"]] += 1
                status_counter[s["gen_status"]] += 1
                recs.append({
                    "query_id": it["q"]["query_id"], "condition": cond,
                    "n_candidates": len(it["cand_ids"]), **s, "prompt_tokens": it["ptok"],
                })
            done += len(batch)
            print(f"[m5] {model_key} {cond}: {done}/{len(items)} "
                  f"(batch {len(batch)}, 상태 {dict(cstat)})", flush=True)
        per_cond_status[cond] = dict(cstat)
        recs_by_cond[cond] = recs
        out = os.path.join(out_dir, f"downstream_pilot_{model_key}_{cond}_K{k}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(status_counter.values())
    parse_rate = status_counter["ok"] / total if total else 0.0
    attempts = status_counter["ok"] + status_counter["malformed"]
    return {"model_id": runner.model_id, "n": total, "parse_rate": round(parse_rate, 4),
            # 게이트 지표(2026-07 재정의): 파서 무결성. no_call 은 행동 지표로 분리.
            "structural_parse_rate": round(status_counter["ok"] / attempts, 4) if attempts else 0.0,
            "no_call_rate": round(status_counter["no_call"] / total, 4) if total else 0.0,
            "status_counts": dict(status_counter), "per_condition_status": per_cond_status,
            "per_condition_metrics": {c: _aggregate(r) for c, r in recs_by_cond.items()},
            "retrieval_effect": _retrieval_effect(recs_by_cond)}


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
        # 파일명 규약: method 문자열이 prior 태그(_oracle/_real)까지 포함한다.
        fname = f"retrieval_{split}_{m}_{k}.jsonl"
        path = os.path.join(results_dir, fname)
        if os.path.isfile(path):
            retrieved[m] = {str(r["query_id"]): r["candidate_tools"] for r in _read_jsonl(path)}
        else:
            print(f"[m5] 경고: {fname} 없음 (M3 먼저). retrieved_{m} 건너뜀.")

    conditions = ["full", "random_k", "oracle_tool"] + [f"retrieved_{m}" for m in RETRIEVAL_METHODS if m in retrieved]

    report = {"split": split, "k": k, "conditions": conditions,
              "decoding": cfg["decoding"], "system_prompt": system_prompt,
              "enable_thinking": cfg.get("prompt", {}).get("enable_thinking"), "models": {}}
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
        for cond, m in r["per_condition_metrics"].items():
            print(f"[m5]   {cond}: func_acc {m.get('func_acc')} exact {m.get('exact_match')} "
                  f"strict {m.get('strict_success')} recall_all {m.get('recall_all')} "
                  f"halluc {m.get('mean_hallucinated_calls')} miss {m.get('miss_type_counts')}")
        for cond, eff in r["retrieval_effect"].items():
            vf = eff.get("vs_full", {})
            print(f"[m5]   [효과] {cond}: Δstrict(vs full) {vf.get('delta_strict_success')} "
                  f"Δfunc {vf.get('delta_func_acc')} "
                  f"(도움 {vf.get('helped_queries')} / 해악 {vf.get('hurt_queries')}), "
                  f"recall hit/miss 시 strict {eff.get('strict_success_given_recall_hit')}/"
                  f"{eff.get('strict_success_given_recall_miss')}")

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
    for m in ["dense_single", "dense_multi"]:
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
                def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize, **kw):
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
    # retriever 영향 리포트: 조건별 집계 + retrieved_* 의 baseline 짝 비교가 있어야 함.
    pcm = rep["models"]["weak"]["per_condition_metrics"]
    assert set(pcm) == set(rep["conditions"]) and all("recall_all" in m for m in pcm.values()), pcm
    # 엄격 지표: mock 은 항상 첫 candidate 만 (스키마 유효 인자로) 1회 호출.
    #   random_k/oracle 은 gold 가 첫 candidate → exact=strict=1. 인자 q(required) 채움 → args_valid=1.
    assert pcm["oracle_tool"]["strict_success"] == 1.0 and pcm["oracle_tool"]["exact_match"] == 1.0, pcm["oracle_tool"]
    assert pcm["full"]["strict_success"] == 0.25 and pcm["full"]["args_valid_rate"] == 1.0, pcm["full"]
    assert pcm["full"]["mean_hallucinated_calls"] == 0.0, pcm["full"]
    eff = rep["models"]["weak"]["retrieval_effect"]
    assert eff and all(c.startswith("retrieved_") for c in eff), eff
    e0 = next(iter(eff.values()))
    assert "vs_full" in e0 and "vs_random_k" in e0 and "delta_func_acc" in e0["vs_full"], e0
    assert "delta_strict_success" in e0["vs_full"] and "strict_success_given_recall_hit" in e0, e0
    # mock candidate=[t0,t1,t2] 고정: gold=t3 인 쿼리 1개만 recall miss.
    assert e0["recall_all"] == 0.75, e0
    # 레코드에도 새 필드가 기록됐는지.
    rows = [json.loads(l) for l in open(f"{d}/out/results/downstream_pilot_weak_retrieved_dense_single_K10.jsonl")]
    need = {"recall_all", "n_calls", "hallucinated_calls", "exact_match", "args_valid", "strict_success"}
    assert all(need <= set(r) for r in rows), rows[0]
    print(f"[smoke] OK — 조건 {len(rep['conditions'])}, parse_rate {rep['models']['weak']['parse_rate']}, "
          f"retrieval_effect {list(eff)}")


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
