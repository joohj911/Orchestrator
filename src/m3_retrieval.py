"""m3_retrieval.py

명세: spec/rules/retrieval.md
역할: 임베딩 사전계산 + 축B candidate 생성 + fusion 계수 grid search
산출물: data/embeddings/, results/retrieval_*.jsonl, fusion_coeffs.json

CLI: python m3_retrieval.py --config config.yaml [--force] [--smoke]

규칙:
  - config에서 모든 파라미터 로드 (하드코딩 금지).
  - seed 고정. 정답 누출 금지 (fusion 계수는 validation 에서만 선택, test 미사용).
  - 산출물 스키마·경로는 명세 준수.
구현: Claude Code.

주요 DECISION (근거 명시):
  # DECISION NEEDED: validation = 각 split 의 100 쿼리를 seed 로 val_frac(0.3)/test 로 분할.
  #   근거: 명세는 ToolBench eval 파티션을 validation 으로 가정하나, 이 환경은 data.zip
  #   (프록시 차단)이 없어 test 서브셋만 확보됨. fusion 계수를 test 로 고르면 게이트 위반이므로,
  #   각 split 을 val/test 로 나눠 계수는 val 에서만 선택하고 headline 지표는 test 에서만 본다.
  # DECISION NEEDED: example 벡터는 passage prefix 로 임베딩(tool 문서 표현의 일부, retrieval.md
  #   가 examples 를 Tool 표현 아래 둠). 대안(query prefix)도 가능하나 명세 분류를 따른다.
  # DECISION NEEDED: M3 fusion 은 oracle prior 만 생성(real 은 M4 classifier 필요 → M6 stage2).
  #   fusion 계수는 oracle prior·validation 에서 1회 선택하고 real 단계에서 재사용(gap 해석 위해
  #   prior 출처만 바뀌도록 계수 고정).
  # DECISION NEEDED: 임베딩은 tool_{id}.npy 대신 통합 배열로 저장(id 에 '/'·공백 등 파일명
  #   부적합 문자 포함). 내용 동일(desc 1 + example 5 벡터).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402
from utils.scoring import recall_all  # noqa: E402

NON_FUSION = ["bm25", "dense_single", "dense_multi"]
FUSION = ["fusion_add", "fusion_mult"]


# ----------------------------- 데이터 로드 -----------------------------
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
    """BM25 코퍼스용 텍스트: desc + example 5개."""
    return build_tool_desc_text(tool) + " " + " ".join(examples)


# ----------------------------- val/test 분할 -----------------------------
def partition_val_test(query_ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """split 쿼리를 결정적으로 val/test 로 분할. val 은 fusion 계수 선택 전용."""
    ids = sorted(query_ids, key=str)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val = set(ids[:n_val])
    return sorted(val, key=str), sorted([i for i in ids if i not in val], key=str)


# ----------------------------- 스코어링 -----------------------------
def cosine_scores_single(q_vec: np.ndarray, desc_mat: np.ndarray) -> np.ndarray:
    """dense_single: query ↔ 각 tool description 벡터 cosine (정규화 가정 → dot)."""
    return desc_mat @ q_vec


def cosine_scores_multi(q_vec: np.ndarray, tool_vecs: np.ndarray) -> np.ndarray:
    """dense_multi: query ↔ tool 6벡터 중 max cosine. tool_vecs: (n_tools, 6, d)."""
    sims = tool_vecs @ q_vec  # (n_tools, 6)
    return sims.max(axis=1)


def topk_ids(scores: np.ndarray, tool_ids: list[str], k: int) -> list[str]:
    """상위 K tool id (점수 내림차순, 동점은 인덱스 순으로 결정적)."""
    k = min(k, len(tool_ids))
    # -score 로 정렬하되 안정 정렬로 동점 결정적.
    order = np.argsort(-scores, kind="stable")[:k]
    return [tool_ids[i] for i in order]


def oracle_prior_matrix(query_gold_cats: list[list[str]], tool_cats: list[str]) -> np.ndarray:
    """oracle p_class: (n_queries, n_tools) — tool.category ∈ query.gold_categories → 1 else 0."""
    tool_cat_arr = np.array(tool_cats)
    out = np.zeros((len(query_gold_cats), len(tool_cats)), dtype=np.float32)
    for i, gcats in enumerate(query_gold_cats):
        gset = set(gcats)
        out[i] = np.array([1.0 if c in gset else 0.0 for c in tool_cat_arr], dtype=np.float32)
    return out


def zscore_fit(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, (std if std > 1e-8 else 1.0)


def fusion_add_scores(s_sem: np.ndarray, p_class: np.ndarray, alpha: float, beta: float,
                      mean: float, std: float) -> np.ndarray:
    """α·zscore(s_sem) + β·p_class. s_sem,(p_class): (n_tools,)."""
    return alpha * ((s_sem - mean) / std) + beta * p_class


def fusion_mult_scores(s_sem: np.ndarray, p_class: np.ndarray, lam: float, eps: float) -> np.ndarray:
    """s_sem · max(p_class, ε)^λ."""
    return s_sem * np.power(np.maximum(p_class, eps), lam)


# ----------------------------- BM25 -----------------------------
def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


# ----------------------------- grid search -----------------------------
def grid_search_fusion(method: str, val_semscores, val_priors, val_golds, tool_ids,
                       cfg_fusion: dict, mean: float, std: float) -> dict:
    """validation 에서 Recall_all@10 최대화하는 계수 선택. 동률 시 prior 의존 작은 쪽.

    val_semscores/val_priors: list over val queries, each (n_tools,) array.
    val_golds: list of gold id sets.
    """
    eps = float(cfg_fusion["epsilon"])
    best = None  # (recall, tiebreak, coeffs)
    if method == "fusion_add":
        combos = [(a, b) for a in cfg_fusion["alpha_grid"] for b in cfg_fusion["beta_grid"]]
    else:
        combos = [(l,) for l in cfg_fusion["lambda_grid"]]

    for combo in combos:
        hit = 0
        for s_sem, p_cls, gold in zip(val_semscores, val_priors, val_golds):
            if method == "fusion_add":
                a, b = combo
                sc = fusion_add_scores(s_sem, p_cls, a, b, mean, std)
            else:
                (l,) = combo
                sc = fusion_mult_scores(s_sem, p_cls, l, eps)
            cand = topk_ids(sc, tool_ids, 10)
            hit += recall_all(cand, gold)
        recall = hit / max(1, len(val_golds))
        # tiebreak: prior 의존 작은 쪽 (add: β 작게, mult: λ 작게)
        tiebreak = combo[-1] if method == "fusion_mult" else combo[1]
        key = (recall, -tiebreak)
        if best is None or key > best[0]:
            if method == "fusion_add":
                coeffs = {"alpha": combo[0], "beta": combo[1], "lambda": None}
            else:
                coeffs = {"alpha": None, "beta": None, "lambda": combo[0]}
            best = (key, {**coeffs, "recall_all@10_val": round(recall, 4)})
    return best[1]


# ----------------------------- 메인 파이프라인 -----------------------------
def run(config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    splits = cfg["experiment"]["splits"]
    k_sweep = cfg["experiment"]["k_sweep"]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    fcfg = cfg["fusion"]
    val_frac = float(fcfg["val_frac"])

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    tool_cats = [t["category"] for t in tools]
    id2idx = {tid: i for i, tid in enumerate(tool_ids)}

    ex_path = cfg["paths"].get("examples_file", "")
    if not (ex_path and os.path.isfile(ex_path)):
        ex_path = os.path.join(data_dir, "tools_examples.jsonl")
    ex_by_tool = {r["tool_id"]: r["examples"] for r in _read_jsonl(ex_path)}

    # --- 임베딩 사전계산 (prefix 규칙 로그) ---
    from utils.embed import embed_passages, embed_queries, PREFIX_QUERY, PREFIX_PASSAGE

    print(f"[m3] e5 prefix 규칙: query='{PREFIX_QUERY}', passage='{PREFIX_PASSAGE}'")
    desc_npy = os.path.join(emb_dir, "tools_desc.npy")
    ex_npy = os.path.join(emb_dir, "tools_examples.npy")
    if not force and os.path.isfile(desc_npy) and os.path.isfile(ex_npy):
        desc_mat = np.load(desc_npy)
        tool_ex_mat = np.load(ex_npy)
        print(f"[m3] tool 임베딩 캐시 로드: {desc_mat.shape}, {tool_ex_mat.shape}")
    else:
        desc_texts = [build_tool_desc_text(t) for t in tools]
        desc_mat = embed_passages(desc_texts, cfg)  # (n_tools, d)
        # examples: passage prefix (tool 문서 표현). 각 tool 5개.
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

    # tool 6벡터 = [desc, ex1..ex5]
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)  # (n_tools, 6, d)

    # BM25 코퍼스
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_tokenize(build_tool_bm25_text(t, ex_by_tool.get(t["id"], []))) for t in tools])

    fusion_coeffs: dict[str, Any] = {"val_frac": val_frac, "norm_method": fcfg["norm_method"], "splits": {}}
    os.makedirs(results_dir, exist_ok=True)

    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        qids = [str(q["query_id"]) for q in queries]
        val_ids, test_ids = partition_val_test(qids, val_frac, seed)
        val_set = set(val_ids)
        print(f"[{split}] queries {len(queries)} → val {len(val_ids)} / test {len(test_ids)}")

        # query 임베딩 (캐시)
        q_npy = os.path.join(emb_dir, f"queries_{split}.npy")
        if not force and os.path.isfile(q_npy):
            q_mat = np.load(q_npy)
        else:
            q_mat = embed_queries([q["query"] for q in queries], cfg)
            np.save(q_npy, q_mat)

        # 사전 계산: 각 query 의 sem score (dense_multi), single score, bm25, oracle prior
        gold_by_q = [[g for g in q["gold_tools"]] for q in queries]
        goldcats_by_q = [q.get("gold_categories", []) for q in queries]
        prior_mat = oracle_prior_matrix(goldcats_by_q, tool_cats)  # (nq, n_tools)

        sem_scores, single_scores, bm25_scores = [], [], []
        for i, q in enumerate(queries):
            sem_scores.append(cosine_scores_multi(q_mat[i], tool_vecs))
            single_scores.append(cosine_scores_single(q_mat[i], desc_mat))
            bm25_scores.append(np.asarray(bm25.get_scores(_tokenize(q["query"])), dtype=np.float32))

        # s_sem zscore: validation 분포 기준 (고정·기록)
        val_idx = [i for i, q in enumerate(queries) if str(q["query_id"]) in val_set]
        val_sem_stack = np.concatenate([sem_scores[i] for i in val_idx]) if val_idx else np.concatenate(sem_scores)
        z_mean, z_std = zscore_fit(val_sem_stack)

        # fusion 계수 grid search (oracle prior, val 만)
        val_golds = [set(gold_by_q[i]) for i in val_idx]
        val_sem = [sem_scores[i] for i in val_idx]
        val_prior = [prior_mat[i] for i in val_idx]
        split_coeffs = {"val_ids": val_ids, "test_ids": test_ids,
                        "zscore_mean": round(z_mean, 6), "zscore_std": round(z_std, 6)}
        for method in FUSION:
            split_coeffs[method] = grid_search_fusion(
                method, val_sem, val_prior, val_golds, tool_ids, fcfg, z_mean, z_std)
        fusion_coeffs["splits"][split] = split_coeffs

        # --- candidate 파일 생성 (전 쿼리, role 태그) ---
        def role(qid: str) -> str:
            return "val" if qid in val_set else "test"

        def write_candidates(method: str, prior_tag: str | None):
            for k in k_sweep:
                fname = (f"retrieval_{split}_{method}_{prior_tag}_{k}.jsonl" if prior_tag
                         else f"retrieval_{split}_{method}_{k}.jsonl")
                out = os.path.join(results_dir, fname)
                with open(out, "w", encoding="utf-8") as f:
                    for i, q in enumerate(queries):
                        if method == "bm25":
                            sc = bm25_scores[i]
                        elif method == "dense_single":
                            sc = single_scores[i]
                        elif method == "dense_multi":
                            sc = sem_scores[i]
                        elif method == "fusion_add":
                            c = split_coeffs["fusion_add"]
                            sc = fusion_add_scores(sem_scores[i], prior_mat[i], c["alpha"], c["beta"], z_mean, z_std)
                        elif method == "fusion_mult":
                            c = split_coeffs["fusion_mult"]
                            sc = fusion_mult_scores(sem_scores[i], prior_mat[i], c["lambda"], float(fcfg["epsilon"]))
                        cand = topk_ids(sc, tool_ids, k)
                        rec = {"query_id": q["query_id"], "role": role(str(q["query_id"])),
                               "candidate_tools": cand, "gold_tools": gold_by_q[i],
                               "recall_all": recall_all(cand, gold_by_q[i])}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for method in NON_FUSION:
            write_candidates(method, None)
        for method in FUSION:
            write_candidates(method, "oracle")  # M3 = oracle prior only

    with open(os.path.join(cfg["paths"]["output_dir"], "fusion_coeffs.json"), "w", encoding="utf-8") as f:
        json.dump(fusion_coeffs, f, ensure_ascii=False, indent=2)
    # fusion_coeffs.json 는 output_dir 루트(README 산출물 트리 기준).
    print(f"[m3] fusion_coeffs.json 저장. 완료.")


def _smoke() -> None:
    """e5 없이 random 벡터로 랭킹/fusion/계수선택/recall 로직 점검."""
    print("[smoke] m3 로직 점검 (random 임베딩)")
    rng = np.random.default_rng(0)
    n_tools, d, nq = 12, 8, 8
    def norm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
    tool_vecs = norm(rng.standard_normal((n_tools, 6, d))).astype(np.float32)
    desc_mat = tool_vecs[:, 0, :]
    tool_ids = [f"T{i}" for i in range(n_tools)]
    tool_cats = [["A", "B", "C"][i % 3] for i in range(n_tools)]
    q = norm(rng.standard_normal((nq, d))).astype(np.float32)
    golds = [[tool_ids[i % n_tools]] for i in range(nq)]
    goldcats = [[tool_cats[i % n_tools]] for i in range(nq)]
    prior = oracle_prior_matrix(goldcats, tool_cats)
    sem = [cosine_scores_multi(q[i], tool_vecs) for i in range(nq)]
    # recall / topk
    cand = topk_ids(sem[0], tool_ids, 5); assert len(cand) == 5
    assert recall_all(tool_ids, golds[0]) == 1
    # zscore + fusion
    m, s = zscore_fit(np.concatenate(sem))
    a = fusion_add_scores(sem[0], prior[0], 0.5, 0.5, m, s); assert a.shape == (n_tools,)
    mu = fusion_mult_scores(sem[0], prior[0], 1.0, 0.05); assert mu.shape == (n_tools,)
    # grid search
    fcfg = {"alpha_grid": [0.3, 0.7], "beta_grid": [0.3, 0.7], "lambda_grid": [0.5, 1.0], "epsilon": 0.05}
    c_add = grid_search_fusion("fusion_add", sem, [prior[i] for i in range(nq)],
                               [set(golds[i]) for i in range(nq)], tool_ids, fcfg, m, s)
    c_mult = grid_search_fusion("fusion_mult", sem, [prior[i] for i in range(nq)],
                                [set(golds[i]) for i in range(nq)], tool_ids, fcfg, m, s)
    assert c_add["alpha"] in (0.3, 0.7) and c_add["lambda"] is None
    assert c_mult["lambda"] in (0.5, 1.0) and c_mult["alpha"] is None
    # partition determinism
    v1, t1 = partition_val_test([str(i) for i in range(10)], 0.3, 42)
    v2, t2 = partition_val_test([str(i) for i in range(10)], 0.3, 42)
    assert v1 == v2 and set(v1).isdisjoint(t1) and len(v1) + len(t1) == 10
    print(f"[smoke] OK — add={c_add}, mult={c_mult}, val={v1}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 retrieval + fusion 계수")
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
