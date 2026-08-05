"""experiment_slm_prior.py — SLM zero-shot category prior (무학습·무로그 후보).

배경: 무학습 clustering 변형(mean/max/시나리오)은 전부 dense_multi 로 수렴 — 같은 임베딩
공간에서는 새 정보가 없음이 확정됨. 남은 무학습 정보원 중 최강 후보 = **서빙 중인 SLM 의
자연어 이해**를 category 선택에 쓰는 것.

설계 원칙 (QFS 계열 — "선택자이지 플래너가 아니다"):
  - 폐쇄 어휘: category 42개 이름을 프롬프트에 제공, **목록에서 그대로(verbatim) 선택만** 허용
  - 요약·추측·설명 금지, 쉼표 구분 이름만 출력, 최대 4개
  - greedy + thinking off → 결정적
파싱은 목록과 exact match(대소문자 무시)만 인정 — 목록 밖 이름은 hallucination 으로 집계.

산출: 쿼리별 이진 category prior → 기존 fusion 프레임(fold 별 grid search, 누출 0)으로
학습 classifier(cat_real)와 Recall_all 직접 비교 + SLM 의 category 예측 P/R/F1 진단.

CLI: python scripts/experiment_slm_prior.py --config config.yaml [--smoke]
GPU 필요 (SLM 추론, 쿼리당 짧은 프롬프트 1회 — 300쿼리×3split 수 분).
산출물: results/slm_prior_ab.json + 콘솔 표
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
from utils.config import load_config  # noqa: E402
from m3_retrieval import cosine_scores_multi, assign_folds  # noqa: E402
from experiment_prior_ablation import recall_with_prior  # noqa: E402
from experiment_cluster_prior import baseline_recalls  # noqa: E402
from m5_pilot import QwenRunner, make_batches  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_category_prompt(tokenizer, query: str, categories: list[str], max_cats: int):
    """폐쇄 선택 프롬프트 (chat template, tools 미사용)."""
    sys_msg = (
        "You are a strict category selector, not a planner. "
        "Choose ONLY from the given list. Output the chosen category names verbatim, "
        "comma-separated, on a single line. No explanations, no extra words."
    )
    user_msg = (
        f"Categories:\n{', '.join(categories)}\n\n"
        f"User request: \"{query}\"\n\n"
        f"Select the 1 to {max_cats} categories needed to fulfill this request."
    )
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)


def parse_categories(gen: str, categories: list[str], max_cats: int):
    """출력에서 목록과 exact match(대소문자 무시)되는 이름만 채택.

    반환: (선택된 category 리스트, hallucinated 토큰 수)
    """
    lower_map = {c.lower(): c for c in categories}
    picked, halluc = [], 0
    first_line = gen.strip().splitlines()[0] if gen.strip() else ""
    for tok in first_line.split(","):
        name = tok.strip().strip('."\'' )
        if not name:
            continue
        hit = lower_map.get(name.lower())
        if hit and hit not in picked:
            picked.append(hit)
        elif not hit:
            halluc += 1
    return picked[:max_cats], halluc


def run(config_path, runner_factory=None):
    cfg = load_config(config_path, make_dirs=False)
    seed = int(cfg["seed"])
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    fcfg = cfg["fusion"]
    sp = cfg.get("slm_prior", {})
    model_key = sp.get("model_key", "weak")
    max_cats = int(sp.get("max_categories", 4))

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    tool_cats = [t["category"] for t in tools]
    categories = sorted(set(tool_cats))  # pool 의 category 목록 (등록 시점에 항상 존재)

    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    mid = cfg["models"]["downstream"][model_key]
    print(f"[slm-prior] category {len(categories)}개, 모델 {model_key}={mid}")
    factory = runner_factory or (lambda m: QwenRunner(m, cfg))
    runner = factory(mid)

    report = {"model": mid, "n_categories": len(categories), "max_categories": max_cats,
              "splits": {}}
    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        goldcats_by_q = [set(q.get("gold_categories", [])) for q in queries]
        q_mat = np.load(os.path.join(emb_dir, f"queries_{split}.npy"))
        sem = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        fold_of = assign_folds([str(q["query_id"]) for q in queries], int(fcfg["n_folds"]), seed)

        # --- SLM 으로 category 선택 (배치 생성) ---
        items = [{"q": q, "prompt": build_category_prompt(runner.tok, q["query"], categories, max_cats),
                  "ptok": 0} for q in queries]
        for it in items:
            it["ptok"] = runner.prompt_tokens(it["prompt"])
        hw = cfg.get("hardware", {})
        picked_by_q, halluc_total, empty = [], 0, 0
        done = 0
        for batch in make_batches(items, int(hw.get("batch_size_gen", 8)),
                                  int(hw.get("gen_batch_tokens", 40000))):
            gens = (runner.generate_batch([it["prompt"] for it in batch])
                    if hasattr(runner, "generate_batch")
                    else [runner.generate(it["prompt"]) for it in batch])
            for it, gen in zip(batch, gens):
                picked, halluc = parse_categories(gen, categories, max_cats)
                picked_by_q.append(picked)
                halluc_total += halluc
                empty += (not picked)
            done += len(batch)
            print(f"[slm-prior] {split}: {done}/{len(items)}", flush=True)

        # --- category 예측 품질 진단 (P/R, 쿼리 평균) ---
        precs, recs = [], []
        for picked, gold in zip(picked_by_q, goldcats_by_q):
            ps = set(picked)
            if ps:
                precs.append(len(ps & gold) / len(ps))
            recs.append(len(ps & gold) / max(1, len(gold)))
        diag = {"precision": round(float(np.mean(precs)) if precs else 0.0, 4),
                "recall": round(float(np.mean(recs)), 4),
                "empty_preds": empty, "hallucinated_names": halluc_total,
                "mean_picked": round(float(np.mean([len(p) for p in picked_by_q])), 2)}
        print(f"[slm-prior] {split} 진단: {diag}")

        # --- 이진 prior → fusion (기존 프레임과 동일 절차) ---
        prior = np.zeros((len(queries), len(tool_ids)), dtype=np.float32)
        for qi, picked in enumerate(picked_by_q):
            pset = set(picked)
            prior[qi] = np.array([1.0 if c in pset else 0.0 for c in tool_cats], dtype=np.float32)

        fused = recall_with_prior(queries, tool_ids, sem, prior, gold_by_q, fold_of, fcfg, ks)
        report["splits"][split] = {
            "diagnostics": diag,
            "fused": fused,
            "baselines": baseline_recalls(results_dir, split, ks),
            "picked": {str(q["query_id"]): p for q, p in zip(queries, picked_by_q)},
        }

    out = os.path.join(results_dir, "slm_prior_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- 비교표 ---
    for split in splits:
        e = report["splits"][split]
        print(f"\n=== {split} : Recall_all@K — 학습 classifier vs SLM zero-shot ===")
        print("variant".ljust(26), *[f"K={k}".rjust(7) for k in ks])
        b = e["baselines"]
        for name in ("dense_multi", "cat_add_real", "cat_mult_real", "cat_add_oracle", "cat_mult_oracle"):
            if name in b:
                print(name.ljust(26), *[f"{b[name].get(k, float('nan')):.2f}".rjust(7) for k in ks])
        for m in ("add", "mult"):
            print(f"slm_prior {m}".ljust(26), *[f"{e['fused'][m][k]:.2f}".rjust(7) for k in ks])
        d = e["diagnostics"]
        print(f"  (SLM category P {d['precision']} / R {d['recall']} / 평균 선택 {d['mean_picked']}개 "
              f"/ 무응답 {d['empty_preds']} / 목록 밖 이름 {d['hallucinated_names']})")
    print(f"\n[slm-prior] 저장: {out}")
    print("[slm-prior] 판정: slm_prior 가 dense_multi 를 넘고 cat_*_real 에 근접하면 "
          "무학습 배포 구성 채택 후보. category R 이 낮으면 프롬프트/max_categories 조정 여지.")


def _smoke():
    """mock 러너로 프롬프트 구성·파싱·prior 조립 점검 (GPU 불필요)."""
    print("[smoke] slm prior 로직")
    cats = ["Sports", "Finance", "Weather", "Music"]
    picked, halluc = parse_categories("Sports, finance, Cooking\n그리고 설명...", cats, 4)
    assert picked == ["Sports", "Finance"] and halluc == 1, (picked, halluc)
    picked, _ = parse_categories("", cats, 4)
    assert picked == []
    picked, _ = parse_categories("Music, Music, Weather", cats, 1)
    assert picked == ["Music"], picked

    class T:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kw):
            assert messages[0]["role"] == "system" and "verbatim" in messages[0]["content"]
            return "PROMPT::" + messages[1]["content"]

    p = build_category_prompt(T(), "check the weather", cats, 4)
    assert "Sports, Finance, Weather, Music" in p and "check the weather" in p
    print("[smoke] OK — 파싱(대소문자/중복/목록 밖/빈 출력)·프롬프트 구성 정상")


def main():
    ap = argparse.ArgumentParser(description="SLM zero-shot category prior (무학습·무로그, GPU 필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
