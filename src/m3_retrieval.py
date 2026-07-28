"""m3_retrieval.py

명세: spec/rules/retrieval.md
역할: 임베딩 사전계산 + 축B candidate 생성 + fusion 계수 grid search (k-fold CV)
산출물: data/embeddings/, results/retrieval_*.jsonl, fusion_coeffs.json

CLI: python m3_retrieval.py --config config.yaml [--force] [--smoke]

규칙:
  - config에서 모든 파라미터 로드 (하드코딩 금지).
  - seed 고정. 정답 누출 금지 (fusion 계수는 각 쿼리가 참여 안 한 fold 에서만 선택).
구현: Claude Code.

주요 DECISION (근거 명시):
  # DECISION: fusion 계수는 k-fold 교차검증으로 선택(config.fusion.n_folds=5).
  #   각 fold f 의 쿼리는 '나머지 fold(train)'에서 Recall_all@10 최대화로 고른 계수로 채점된다.
  #   → 모든 쿼리가 자신을 안 본 계수로 test 되므로 100개 전부 headline test 로 사용(누출 0).
  #   근거: fusion 검증 쿼리는 gold ⊆ 500-pool 이어야 Recall 이 유효한데, 외부(train/eval) 쿼리는
  #   pool 밖 gold 라 사용 불가. pool 과 짝이 맞는 우리 쿼리를 fold 로 나누는 것이 유일하게 타당.
  #   계수는 fold 별로 다를 수 있고, 대표값으로 뭉개지 않고 fold 별 값을 그대로 쓰고 기록한다.
  # DECISION: s_sem zscore(mean/std)도 각 fold 의 train 분포에서 적합(그 fold 를 안 봄).
  # DECISION: example 벡터는 passage prefix(tool 문서 표현). M3 fusion 은 oracle prior 만
  #   (real 은 M4 후 M6 stage2). fold 계수·zscore 를 M6 real 단계에서 재사용(prior 만 변동).
  # DECISION: 임베딩은 tool_{id}.npy 대신 통합 배열 저장(id 에 파일명 부적합 문자 포함).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402
from utils.scoring import recall_all  # noqa: E402

NON_FUSION = ["bm25", "dense_single", "dense_multi"]
FUSION = ["fusion_add", "fusion_mult"]


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_tool_desc_text(tool: dict) -> str:
    """description 벡터용 텍스트: name + description + params 요약."""
    parts = [str(tool.get("name", "")), str(tool.get("description", ""))]
    pnames = [p.get("name", "") for p in (tool.get("params") or []) if p.get("name")]
    if pnames:
        parts.append("Parameters: " + ", ".join(pnames))
    return ". ".join(p for p in parts if p).strip()


def build_tool_bm25_text(tool: dict, examples: list[str]) -> str:
    return build_tool_desc_text(tool) + " " + " ".join(examples)


def assign_folds(query_ids: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """쿼리를 결정적으로 n_folds 개 fold 에 배정 (거의 균등)."""
    ids = sorted(query_ids, key=str)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {qid: (i % n_folds) for i, qid in enumerate(ids)}


# ----------------------------- 스코어링 -----------------------------
def cosine_scores_single(q_vec, desc_mat):
    return desc_mat @ q_vec


def cosine_scores_multi(q_vec, tool_vecs):
    return (tool_vecs @ q_vec).max(axis=1)


def topk_ids(scores, tool_ids, k):
    k = min(k, len(tool_ids))
    order = np.argsort(-scores, kind="stable")[:k]
    return [tool_ids[i] for i in order]


def oracle_prior_matrix(query_gold_cats, tool_cats):
    out = np.zeros((len(query_gold_cats), len(tool_cats)), dtype=np.float32)
    for i, gcats in enumerate(query_gold_cats):
        gset = set(gcats)
        out[i] = np.array([1.0 if c in gset else 0.0 for c in tool_cats], dtype=np.float32)
    return out


def zscore_fit(values):
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, (std if std > 1e-8 else 1.0)


def fusion_add_scores(s_sem, p_class, alpha, beta, mean, std):
    return alpha * ((s_sem - mean) / std) + beta * p_class


def fusion_mult_scores(s_sem, p_class, lam, eps):
    return s_sem * np.power(np.maximum(p_class, eps), lam)


def _tokenize(text):
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def grid_search_fusion(method, tr_sem, tr_prior, tr_golds, tool_ids, cfg_fusion, mean, std):
    """train 쿼리에서 Recall_all@10 최대화 계수 선택. 동률 시 prior 의존 작은 쪽."""
    eps = float(cfg_fusion["epsilon"])
    combos = ([(a, b) for a in cfg_fusion["alpha_grid"] for b in cfg_fusion["beta_grid"]]
              if method == "fusion_add" else [(l,) for l in cfg_fusion["lambda_grid"]])
    best = None
    for combo in combos:
        hit = 0
        for s_sem, p_cls, gold in zip(tr_sem, tr_prior, tr_golds):
            if method == "fusion_add":
                sc = fusion_add_scores(s_sem, p_cls, combo[0], combo[1], mean, std)
            else:
                sc = fusion_mult_scores(s_sem, p_cls, combo[0], eps)
            hit += recall_all(topk_ids(sc, tool_ids, 10), gold)
        recall = hit / max(1, len(tr_golds))
        tiebreak = combo[-1] if method == "fusion_mult" else combo[1]
        key = (recall, -tiebreak)
        if best is None or key > best[0]:
            coeffs = ({"alpha": combo[0], "beta": combo[1], "lambda": None}
                      if method == "fusion_add" else {"alpha": None, "beta": None, "lambda": combo[0]})
            best = (key, {**coeffs, "recall_all@10_train": round(recall, 4)})
    return best[1]


def run(config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    splits = cfg["experiment"]["splits"]
    k_sweep = cfg["experiment"]["k_sweep"]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    fcfg = cfg["fusion"]
    n_folds = int(fcfg["n_folds"])
    eps = float(fcfg["epsilon"])

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    tool_cats = [t["category"] for t in tools]

    ex_path = cfg["paths"].get("examples_file", "")
    if not (ex_path and os.path.isfile(ex_path)):
        ex_path = os.path.join(data_dir, "tools_examples.jsonl")
    ex_by_tool = {r["tool_id"]: r["examples"] for r in _read_jsonl(ex_path)}

    from utils.embed import embed_passages, embed_queries, PREFIX_QUERY, PREFIX_PASSAGE
    print(f"[m3] e5 prefix 규칙: query='{PREFIX_QUERY}', passage='{PREFIX_PASSAGE}'")

    desc_npy = os.path.join(emb_dir, "tools_desc.npy")
    ex_npy = os.path.join(emb_dir, "tools_examples.npy")
    if not force and os.path.isfile(desc_npy) and os.path.isfile(ex_npy):
        desc_mat, tool_ex_mat = np.load(desc_npy), np.load(ex_npy)
        print(f"[m3] tool 임베딩 캐시 로드: {desc_mat.shape}, {tool_ex_mat.shape}")
    else:
        desc_mat = embed_passages([build_tool_desc_text(t) for t in tools], cfg)
        flat_ex, counts = [], []
        for t in tools:
            exs = ex_by_tool.get(t["id"], [])
            flat_ex.extend(exs)
            counts.append(len(exs))
        ex_vecs = embed_passages(flat_ex, cfg)
        d = desc_mat.shape[1]
        tool_ex_mat = np.zeros((len(tools), 5, d), dtype=np.float32)
        cur = 0
        for i, c in enumerate(counts):
            for j in range(min(c, 5)):
                tool_ex_mat[i, j] = ex_vecs[cur + j]
            cur += c
        np.save(desc_npy, desc_mat)
        np.save(ex_npy, tool_ex_mat)
        print(f"[m3] tool 임베딩 저장: desc {desc_mat.shape}, examples {tool_ex_mat.shape}")

    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)  # (n_tools,6,d)

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([_tokenize(build_tool_bm25_text(t, ex_by_tool.get(t["id"], []))) for t in tools])

    fusion_coeffs: dict[str, Any] = {"n_folds": n_folds, "norm_method": fcfg["norm_method"], "splits": {}}

    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        qids = [str(q["query_id"]) for q in queries]
        fold_of = assign_folds(qids, n_folds, seed)
        print(f"[{split}] queries {len(queries)} → {n_folds}-fold CV "
              f"(fold 크기 {sorted(Counter(fold_of.values()).values())})")

        q_npy = os.path.join(emb_dir, f"queries_{split}.npy")
        if not force and os.path.isfile(q_npy):
            q_mat = np.load(q_npy)
        else:
            q_mat = embed_queries([q["query"] for q in queries], cfg)
            np.save(q_npy, q_mat)

        gold_by_q = [list(q["gold_tools"]) for q in queries]
        goldcats_by_q = [q.get("gold_categories", []) for q in queries]
        prior_mat = oracle_prior_matrix(goldcats_by_q, tool_cats)

        sem_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        single_scores = [cosine_scores_single(q_mat[i], desc_mat) for i in range(len(queries))]
        bm25_scores = [np.asarray(bm25.get_scores(_tokenize(q["query"])), dtype=np.float32) for q in queries]

        # --- fold 별 계수 + zscore (그 fold 를 train 에서 제외) ---
        fold_info: dict[int, dict] = {}
        for f in range(n_folds):
            tr_idx = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] != f]
            if not tr_idx:
                continue
            z_mean, z_std = zscore_fit(np.concatenate([sem_scores[i] for i in tr_idx]))
            tr_sem = [sem_scores[i] for i in tr_idx]
            tr_prior = [prior_mat[i] for i in tr_idx]
            tr_golds = [set(gold_by_q[i]) for i in tr_idx]
            entry = {"zscore_mean": round(z_mean, 6), "zscore_std": round(z_std, 6)}
            for m in FUSION:
                entry[m] = grid_search_fusion(m, tr_sem, tr_prior, tr_golds, tool_ids, fcfg, z_mean, z_std)
            fold_info[f] = entry

        # 계수 분포(정보용, 대표값으로 사용하지 않음)
        dist = {m: dict(Counter((fold_info[f][m]["alpha"], fold_info[f][m]["beta"], fold_info[f][m]["lambda"])
                                for f in fold_info)) for m in FUSION}
        fusion_coeffs["splits"][split] = {
            "fold_assignment": {str(q["query_id"]): fold_of[str(q["query_id"])] for q in queries},
            "folds": {str(f): fold_info[f] for f in fold_info},
            "coeff_distribution_informational": {m: {str(k): v for k, v in dist[m].items()} for m in FUSION},
        }

        # --- candidate 파일 생성 (모든 쿼리 = test; fusion 은 쿼리의 fold 계수 사용) ---
        def write_candidates(method: str, prior_tag: str | None):
            for k in k_sweep:
                fname = (f"retrieval_{split}_{method}_{prior_tag}_{k}.jsonl" if prior_tag
                         else f"retrieval_{split}_{method}_{k}.jsonl")
                with open(os.path.join(results_dir, fname), "w", encoding="utf-8") as fh:
                    for i, q in enumerate(queries):
                        f = fold_of[str(q["query_id"])]
                        if method == "bm25":
                            sc = bm25_scores[i]
                        elif method == "dense_single":
                            sc = single_scores[i]
                        elif method == "dense_multi":
                            sc = sem_scores[i]
                        elif method == "fusion_add":
                            c = fold_info[f]["fusion_add"]
                            sc = fusion_add_scores(sem_scores[i], prior_mat[i], c["alpha"], c["beta"],
                                                   fold_info[f]["zscore_mean"], fold_info[f]["zscore_std"])
                        else:  # fusion_mult
                            c = fold_info[f]["fusion_mult"]
                            sc = fusion_mult_scores(sem_scores[i], prior_mat[i], c["lambda"], eps)
                        cand = topk_ids(sc, tool_ids, k)
                        fh.write(json.dumps({
                            "query_id": q["query_id"], "fold": f, "candidate_tools": cand,
                            "gold_tools": gold_by_q[i], "recall_all": recall_all(cand, gold_by_q[i]),
                        }, ensure_ascii=False) + "\n")

        for m in NON_FUSION:
            write_candidates(m, None)
        for m in FUSION:
            write_candidates(m, "oracle")

    with open(os.path.join(cfg["paths"]["output_dir"], "fusion_coeffs.json"), "w", encoding="utf-8") as f:
        json.dump(fusion_coeffs, f, ensure_ascii=False, indent=2)
    print("[m3] fusion_coeffs.json 저장. 완료 (모든 쿼리 test, fold 별 계수).")


def _smoke() -> None:
    print("[smoke] m3 k-fold 로직 점검 (random 임베딩)")
    rng = np.random.default_rng(0)
    n_tools, d, nq = 12, 8, 15
    def norm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
    tool_vecs = norm(rng.standard_normal((n_tools, 6, d))).astype(np.float32)
    tool_ids = [f"T{i}" for i in range(n_tools)]
    tool_cats = [["A", "B", "C"][i % 3] for i in range(n_tools)]
    q = norm(rng.standard_normal((nq, d))).astype(np.float32)
    golds = [[tool_ids[i % n_tools]] for i in range(nq)]
    goldcats = [[tool_cats[i % n_tools]] for i in range(nq)]
    prior = oracle_prior_matrix(goldcats, tool_cats)
    sem = [cosine_scores_multi(q[i], tool_vecs) for i in range(nq)]

    # fold 배정: 결정적 + 균등 + 전 쿼리 커버
    fo = assign_folds([str(i) for i in range(nq)], 5, 42)
    assert fo == assign_folds([str(i) for i in range(nq)], 5, 42)
    assert set(fo.values()) == set(range(5)) and len(fo) == nq

    # fold 별 계수 선택 (train = 나머지 fold)
    for f in range(5):
        tr = [i for i in range(nq) if fo[str(i)] != f]
        m, s = zscore_fit(np.concatenate([sem[i] for i in tr]))
        fcfg = {"alpha_grid": [0.3, 0.7], "beta_grid": [0.3, 0.7], "lambda_grid": [0.5, 1.0], "epsilon": 0.05}
        ca = grid_search_fusion("fusion_add", [sem[i] for i in tr], [prior[i] for i in tr],
                                [set(golds[i]) for i in tr], tool_ids, fcfg, m, s)
        cm = grid_search_fusion("fusion_mult", [sem[i] for i in tr], [prior[i] for i in tr],
                                [set(golds[i]) for i in tr], tool_ids, fcfg, m, s)
        assert ca["alpha"] in (0.3, 0.7) and cm["lambda"] in (0.5, 1.0)
    # 각 쿼리는 자기 fold(=선택 미참여) 계수로 채점됨을 구조로 보장
    print("[smoke] OK — fold 배정 결정적/균등/전커버, fold별 계수 선택 정상")


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 retrieval + fusion 계수 (k-fold CV)")
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
