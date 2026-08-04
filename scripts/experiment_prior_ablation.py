"""experiment_prior_ablation.py — semantic base × prior 2×2 factorial ablation.

목적: 지금까지의 표(desc only / desc+examples / desc+examples+classifier)로는 category
classifier 의 기여가 **예시 발화와 중복인지 상보적인지** 분리되지 않는다. 빠진 칸
"desc only + classifier"(= 기존 방법에 classifier 만 얹은 구성)를 채워 다음을 판정한다.

  prior gain on desc only   vs   prior gain on desc+examples
  → 두 gain 이 비슷하면 상보적(각자 다른 실패를 고침),
    desc only 에서만 크고 desc+ex 에서 작으면 중복 신호(예시가 이미 그 실패를 커버)

전 조합:
  semantic base ∈ {desc only(dense_single), desc+examples(dense_multi)}
  prior ∈ {none, category(real), category(oracle 상한)}
  fusion ∈ {add, mult}   (prior 있는 경우)

GPU/LLM 불필요 — 임베딩 캐시 + m3 의 fusion 기계(fold 별 grid search, 누출 0) 재사용.
각 구성마다 계수를 독립적으로 재탐색한다 (semantic 점수 분포가 base 마다 달라 계수를
공유하면 불공정 비교가 됨).

CLI: python scripts/experiment_prior_ablation.py --config config.yaml [--smoke]
산출물: results/prior_ablation.json + 콘솔 표
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.config import load_config  # noqa: E402
from m3_retrieval import (  # noqa: E402
    FUSION, assign_folds, cosine_scores_multi, cosine_scores_single, fusion_add_scores,
    fusion_mult_scores, grid_search_fusion, oracle_prior_matrix, real_prior_matrix,
    topk_ids, zscore_fit,
)
from utils.scoring import recall_all  # noqa: E402

BASES = ("desc_only", "desc_plus_examples")
PRIORS = ("none", "category_real", "category_oracle")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def recall_no_prior(sem_scores, tool_ids, gold_by_q, ks):
    """prior 없이 semantic 점수만으로 top-K (fusion 미적용)."""
    out = {}
    for k in ks:
        hits = sum(recall_all(topk_ids(sem_scores[i], tool_ids, k), gold_by_q[i])
                   for i in range(len(sem_scores)))
        out[k] = round(hits / max(1, len(sem_scores)), 4)
    return out


def recall_with_prior(queries, tool_ids, sem_scores, prior_mat, gold_by_q, fold_of, fcfg, ks):
    """m3 와 동일한 fold 별 grid search 후 fusion recall@K (add/mult 각각)."""
    eps = float(fcfg["epsilon"])
    fold_info = {}
    for f in sorted({fold_of[str(q["query_id"])] for q in queries}):
        tr = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] != f]
        if not tr:
            continue
        zm, zs = zscore_fit(np.concatenate([sem_scores[i] for i in tr]))
        entry = {"zm": zm, "zs": zs}
        for m in FUSION:
            entry[m] = grid_search_fusion(
                m, [sem_scores[i] for i in tr], [prior_mat[i] for i in tr],
                [set(gold_by_q[i]) for i in tr], tool_ids, fcfg, zm, zs)
        fold_info[f] = entry

    out = {}
    for m in FUSION:
        vals = {}
        for k in ks:
            hits = 0
            for i, q in enumerate(queries):
                info = fold_info[fold_of[str(q["query_id"])]]
                c = info[m]
                if m == "fusion_add":
                    sc = fusion_add_scores(sem_scores[i], prior_mat[i], c["alpha"], c["beta"],
                                           info["zm"], info["zs"])
                else:
                    sc = fusion_mult_scores(sem_scores[i], prior_mat[i], c["lambda"], eps)
                hits += recall_all(topk_ids(sc, tool_ids, k), gold_by_q[i])
            vals[k] = round(hits / max(1, len(queries)), 4)
        out[m.replace("fusion_", "")] = vals
    return out


def run(config_path):
    cfg = load_config(config_path, make_dirs=False)
    seed = int(cfg["seed"])
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    fcfg = cfg["fusion"]

    for p in ("tools_desc.npy", "tools_examples.npy"):
        if not os.path.isfile(os.path.join(emb_dir, p)):
            sys.exit(f"[prior-ablation] 임베딩 캐시 없음: {emb_dir}/{p} (m3 먼저)")
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    tool_cats = [t["category"] for t in tools]

    prior_path = os.path.join(data_dir, "class_prior_real.jsonl")
    if not os.path.isfile(prior_path):
        sys.exit(f"[prior-ablation] {prior_path} 없음 (m4 먼저)")
    prior_by_qid = {str(r["query_id"]): r["prior"] for r in _read_jsonl(prior_path)}

    report = {"bases": list(BASES), "priors": list(PRIORS), "splits": {}}
    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        qids = [str(q["query_id"]) for q in queries]
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        goldcats = [q.get("gold_categories", []) for q in queries]
        q_npy = os.path.join(emb_dir, f"queries_{split}.npy")
        if not os.path.isfile(q_npy):
            sys.exit(f"[prior-ablation] 쿼리 임베딩 캐시 없음: {q_npy} (m3 먼저)")
        q_mat = np.load(q_npy)
        fold_of = assign_folds(qids, int(fcfg["n_folds"]), seed)

        sem = {
            "desc_only": [cosine_scores_single(q_mat[i], desc_mat) for i in range(len(queries))],
            "desc_plus_examples": [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))],
        }
        prior_real, n_missing = real_prior_matrix(prior_by_qid, qids, tool_cats)
        if n_missing:
            print(f"[prior-ablation] 경고: {split} prior 없는 쿼리 {n_missing}/{len(queries)} (0 처리)")
        priors = {"category_real": prior_real,
                  "category_oracle": oracle_prior_matrix(goldcats, tool_cats)}

        entry = {}
        for base in BASES:
            entry[base] = {"none": recall_no_prior(sem[base], tool_ids, gold_by_q, ks)}
            for pname, pmat in priors.items():
                entry[base][pname] = recall_with_prior(
                    queries, tool_ids, sem[base], pmat, gold_by_q, fold_of, fcfg, ks)
            print(f"[prior-ablation] {split} / {base} 완료", flush=True)
        report["splits"][split] = entry

    out = os.path.join(results_dir, "prior_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- 표 출력 ---
    label = {"desc_only": "desc only", "desc_plus_examples": "desc+examples"}
    for split in splits:
        e = report["splits"][split]
        print(f"\n=== {split} : Recall_all@K — semantic base × prior ===")
        print("config".ljust(38), *[f"K={k}".rjust(7) for k in ks])
        for base in BASES:
            print(f"{label[base]} | no prior".ljust(38),
                  *[f"{e[base]['none'][k]:.2f}".rjust(7) for k in ks])
            for pname in ("category_real", "category_oracle"):
                for m in ("add", "mult"):
                    tag = f"{label[base]} | {pname} {m}"
                    print(tag.ljust(38), *[f"{e[base][pname][m][k]:.2f}".rjust(7) for k in ks])

    # --- 핵심 판정: prior gain 비교 (headline K) ---
    hk = int(cfg.get("analysis", {}).get("headline_k", 10))
    print(f"\n=== prior 기여 분해 (K={hk}, real prior, add/mult 중 최선) ===")
    print("split".ljust(8), "desc only".rjust(22), "desc+examples".rjust(22))
    for split in splits:
        e = report["splits"][split]
        row = [split.ljust(8)]
        for base in BASES:
            n = e[base]["none"][hk]
            best = max(e[base]["category_real"][m][hk] for m in ("add", "mult"))
            row.append(f"{n:.2f} → {best:.2f} ({best - n:+.2f})".rjust(22))
        print(*row)
    print("\n해석 가이드: 두 gain 이 비슷하면 예시 발화와 classifier 가 상보적, "
          "desc only 에서만 크면 중복 신호.")
    print(f"[prior-ablation] 저장: {out}")


def _smoke():
    """합성 데이터로 계산 경로 점검."""
    print("[smoke] prior ablation 로직")
    rng = np.random.default_rng(0)
    n_tools, d, nq = 20, 8, 12

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    desc = norm(rng.standard_normal((n_tools, d))).astype(np.float32)
    ex = norm(rng.standard_normal((n_tools, 5, d))).astype(np.float32)
    tool_vecs = np.concatenate([desc[:, None, :], ex], axis=1)
    tool_ids = [f"T{i}" for i in range(n_tools)]
    cats = [["A", "B", "C"][i % 3] for i in range(n_tools)]
    q_mat = norm(rng.standard_normal((nq, d))).astype(np.float32)
    queries = [{"query_id": f"q{i}"} for i in range(nq)]
    gold_by_q = [[tool_ids[i % n_tools]] for i in range(nq)]
    goldcats = [[cats[i % n_tools]] for i in range(nq)]

    sem_single = [cosine_scores_single(q_mat[i], desc) for i in range(nq)]
    sem_multi = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(nq)]
    assert not np.allclose(sem_single[0], sem_multi[0]), "두 base 가 동일하면 비교 의미 없음"

    ks = [5, 10]
    no_prior = recall_no_prior(sem_single, tool_ids, gold_by_q, ks)
    assert set(no_prior) == set(ks) and no_prior[5] <= no_prior[10] + 1e-9

    fcfg = {"alpha_grid": [0.3, 0.7], "beta_grid": [0.1, 0.5], "lambda_grid": [0.1, 0.5],
            "epsilon": 0.05, "n_folds": 3, "norm_method": "zscore"}
    fold_of = assign_folds([q["query_id"] for q in queries], 3, 42)
    op = oracle_prior_matrix(goldcats, cats)
    res = recall_with_prior(queries, tool_ids, sem_single, op, gold_by_q, fold_of, fcfg, ks)
    assert set(res) == {"add", "mult"} and all(0.0 <= res[m][k] <= 1.0 for m in res for k in ks)
    print(f"[smoke] OK — desc only: no prior {no_prior}, oracle add {res['add']}")


def main():
    ap = argparse.ArgumentParser(description="semantic base × prior factorial ablation (GPU 불필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
