"""experiment_cluster_prior.py — cluster 기반 prior 단위 세분화 A/B (보고서 7절 추가 실험).

가설: category(42개, 평균 12 tool/개, 최대 45)가 너무 굵어 fusion prior 의 상한이 낮다
(perfect classifier 여도 I3 recall@10 = 0.61). tool 임베딩의 k-means cluster 를 prior
단위로 쓰면 단위당 tool 수가 줄어 상한이 올라갈 것.

GPU/LLM 불필요 — 기존 임베딩 캐시와 m3 의 fusion 기계를 재사용한다. 두 변형:
  - oracle_cluster: gold tool 들이 속한 cluster 에 1.0 (진단용 상한 — category 의
    perfect classifier 와 동일한 역할. 이 상한이 category 상한을 못 넘으면 접근 기각)
  - centroid: 쿼리 임베딩 ↔ cluster 중심 유사도의 softmax (학습 없는 zero-shot prior.
    dense 검색과 같은 임베딩 공간이라 '신호 중복' 우려의 조기 신호 — 낮게 나와도
    LoRA 재학습 변형(별도 GPU 실험)의 기각 근거는 아님)

판정 기준 (보고서 7절):
  - 채택 방향: oracle_cluster 가 I3@10 에서 category 상한(0.61)을 유의미하게 상회
    → 다음 단계(cluster 라벨로 classifier 재학습) 진행
  - 기각: oracle_cluster 상한이 category 와 비슷하거나 낮음 → graph 로 직행

CLI: python scripts/experiment_cluster_prior.py --config config.yaml [--smoke]
산출물: results/cluster_prior_ab.json + 콘솔 비교표
구현: Claude Code.

# DECISION: tool 표현 = 벡터 6개(설명 1+예시 5)의 평균 후 정규화 (dense_multi 와 동일
#   재료. 예시가 없는 슬롯은 제외).
# DECISION: fold 배정·grid search 는 m3 와 동일 절차 재사용 (각 쿼리는 자신이 계수
#   선택에 불참한 fold 의 계수로 평가 — 누출 0). k-means 는 seed 고정.
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
    FUSION, assign_folds, cosine_scores_multi, fusion_add_scores, fusion_mult_scores,
    grid_search_fusion, topk_ids, zscore_fit,
)
from utils.scoring import recall_all  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tool_representations(desc_mat, tool_ex_mat):
    """tool 표현: 설명 1 + 예시 5 벡터의 평균 (빈 예시 슬롯 제외) 후 L2 정규화."""
    n, d = desc_mat.shape
    reps = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        vecs = [desc_mat[i]]
        for j in range(tool_ex_mat.shape[1]):
            if np.linalg.norm(tool_ex_mat[i, j]) > 1e-8:
                vecs.append(tool_ex_mat[i, j])
        m = np.mean(vecs, axis=0)
        reps[i] = m / max(np.linalg.norm(m), 1e-8)
    return reps


def cluster_tools(reps, k, seed):
    """k-means → (tool별 cluster 라벨, 정규화된 중심)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(reps)
    cents = km.cluster_centers_
    cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-8)
    return labels, cents


def oracle_cluster_prior(queries, tool_ids, labels, gold_by_q):
    """gold tool 이 속한 cluster 의 모든 tool 에 1.0 (진단용 상한)."""
    idx_of = {t: i for i, t in enumerate(tool_ids)}
    out = np.zeros((len(queries), len(tool_ids)), dtype=np.float32)
    for qi, q in enumerate(queries):
        gclusters = {labels[idx_of[g]] for g in gold_by_q[qi] if g in idx_of}
        out[qi] = np.array([1.0 if labels[i] in gclusters else 0.0
                            for i in range(len(tool_ids))], dtype=np.float32)
    return out


def centroid_prior(q_mat, labels, cents, temperature):
    """쿼리↔cluster 중심 유사도의 softmax 를 소속 tool 에 부여 (zero-shot prior)."""
    sims = q_mat @ cents.T  # (nq, k)
    z = sims / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, labels].astype(np.float32)  # (nq, n_tools)


def fused_recall(queries, tool_ids, sem_scores, prior_mat, gold_by_q, fold_of, fcfg, ks):
    """m3 와 동일한 k-fold refit 절차로 fusion recall@K 계산."""
    eps = float(fcfg["epsilon"])
    n_folds = int(fcfg["n_folds"])
    fold_info = {}
    for f in range(n_folds):
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

    out = {m: {} for m in FUSION}
    for m in FUSION:
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
            out[m][k] = round(hits / max(1, len(queries)), 4)
    return out


def baseline_recalls(results_dir, split, ks):
    """기존 산출물에서 baseline recall 로드 (dense_multi, category fusion oracle/real)."""
    out = {}
    for name, fname in (("dense_multi", "retrieval_{s}_dense_multi_{k}.jsonl"),
                        ("cat_add_oracle", "retrieval_{s}_fusion_add_oracle_{k}.jsonl"),
                        ("cat_mult_oracle", "retrieval_{s}_fusion_mult_oracle_{k}.jsonl"),
                        ("cat_add_real", "retrieval_{s}_fusion_add_real_{k}.jsonl"),
                        ("cat_mult_real", "retrieval_{s}_fusion_mult_real_{k}.jsonl")):
        vals = {}
        for k in ks:
            p = os.path.join(results_dir, fname.format(s=split, k=k))
            if os.path.isfile(p):
                rows = _read_jsonl(p)
                vals[k] = round(sum(r["recall_all"] for r in rows) / max(1, len(rows)), 4)
        if vals:
            out[name] = vals
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
    cp = cfg.get("cluster_prior", {})
    cluster_ks = [int(x) for x in cp.get("ks", [42, 64, 128, 256])]
    temperature = float(cp.get("centroid_temperature", 0.05))

    for p in ("tools_desc.npy", "tools_examples.npy"):
        if not os.path.isfile(os.path.join(emb_dir, p)):
            sys.exit(f"[cluster-ab] 임베딩 캐시 없음: {emb_dir}/{p} (m3 먼저)")
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]

    reps = tool_representations(desc_mat, tool_ex_mat)
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    report = {"config": {"cluster_ks": cluster_ks, "temperature": temperature, "seed": seed},
              "splits": {}}
    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        q_npy = os.path.join(emb_dir, f"queries_{split}.npy")
        if not os.path.isfile(q_npy):
            sys.exit(f"[cluster-ab] 쿼리 임베딩 캐시 없음: {q_npy} (m3 먼저)")
        q_mat = np.load(q_npy)
        sem_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        fold_of = assign_folds([str(q["query_id"]) for q in queries], int(fcfg["n_folds"]), seed)

        entry = {"baselines": baseline_recalls(results_dir, split, ks), "clusters": {}}
        for ck in cluster_ks:
            labels, cents = cluster_tools(reps, ck, seed)
            sizes = np.bincount(labels, minlength=ck)
            variants = {
                "oracle_cluster": oracle_cluster_prior(queries, tool_ids, labels, gold_by_q),
                "centroid": centroid_prior(q_mat, labels, cents, temperature),
            }
            entry["clusters"][ck] = {
                "cluster_size_mean": round(float(sizes.mean()), 2),
                "cluster_size_max": int(sizes.max()),
            }
            for vname, prior in variants.items():
                entry["clusters"][ck][vname] = fused_recall(
                    queries, tool_ids, sem_scores, prior, gold_by_q, fold_of, fcfg, ks)
            print(f"[cluster-ab] {split} k={ck} 완료 (cluster 크기 평균 "
                  f"{entry['clusters'][ck]['cluster_size_mean']}, 최대 {entry['clusters'][ck]['cluster_size_max']})",
                  flush=True)
        report["splits"][split] = entry

    out = os.path.join(results_dir, "cluster_prior_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- 비교표 출력 ---
    for split in splits:
        e = report["splits"][split]
        print(f"\n=== {split} : Recall_all@K — category(기존) vs cluster(신규) ===")
        print("variant".ljust(34), *[f"K={k}".rjust(7) for k in ks])
        b = e["baselines"]
        for name in ("dense_multi", "cat_add_oracle", "cat_mult_oracle", "cat_add_real", "cat_mult_real"):
            if name in b:
                print(name.ljust(34), *[f"{b[name].get(k, float('nan')):.2f}".rjust(7) for k in ks])
        for ck, ce in e["clusters"].items():
            for vname in ("oracle_cluster", "centroid"):
                for m in FUSION:
                    label = f"k={ck} {vname} {m.replace('fusion_', '')}"
                    vals = ce[vname][m]
                    print(label.ljust(34), *[f"{vals[k]:.2f}".rjust(7) for k in ks])
    print(f"\n[cluster-ab] 저장: {out}")
    print("[cluster-ab] 판정: oracle_cluster(상한)가 cat_*_oracle 을 넘는지 → 넘으면 "
          "cluster 라벨 classifier 재학습(GPU) 진행, 아니면 graph 직행.")


def _smoke():
    """합성 데이터로 로직 점검 (sklearn/numpy 만 필요)."""
    print("[smoke] cluster prior A/B 로직")
    rng = np.random.default_rng(0)
    n_tools, d, nq = 24, 8, 12

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    desc = norm(rng.standard_normal((n_tools, d))).astype(np.float32)
    ex = norm(rng.standard_normal((n_tools, 5, d))).astype(np.float32)
    ex[5:, 3:] = 0.0  # 일부 tool 은 예시 슬롯 비움 → 평균에서 제외되는지
    reps = tool_representations(desc, ex)
    assert reps.shape == (n_tools, d)
    assert np.allclose(np.linalg.norm(reps, axis=1), 1.0, atol=1e-5)

    labels, cents = cluster_tools(reps, 6, seed=42)
    assert len(labels) == n_tools and cents.shape == (6, d)
    labels2, _ = cluster_tools(reps, 6, seed=42)
    assert (labels == labels2).all(), "k-means seed 고정 위반"

    tool_ids = [f"T{i}" for i in range(n_tools)]
    queries = [{"query_id": f"q{i}"} for i in range(nq)]
    gold_by_q = [[tool_ids[i % n_tools], tool_ids[(i + 3) % n_tools]] for i in range(nq)]
    op = oracle_cluster_prior(queries, tool_ids, labels, gold_by_q)
    assert op.shape == (nq, n_tools) and set(np.unique(op)) <= {0.0, 1.0}
    # gold tool 자신은 반드시 prior 1
    for qi in range(nq):
        for g in gold_by_q[qi]:
            assert op[qi][tool_ids.index(g)] == 1.0

    q_mat = norm(rng.standard_normal((nq, d))).astype(np.float32)
    cp = centroid_prior(q_mat, labels, cents, temperature=0.05)
    assert cp.shape == (nq, n_tools) and (cp >= 0).all()
    # 같은 cluster 의 tool 은 같은 prior 값
    for i in range(n_tools):
        for j in range(n_tools):
            if labels[i] == labels[j]:
                assert abs(cp[0][i] - cp[0][j]) < 1e-6

    fcfg = {"alpha_grid": [0.3, 0.7], "beta_grid": [0.1, 0.5], "lambda_grid": [0.1, 0.5],
            "epsilon": 0.05, "n_folds": 3, "norm_method": "zscore"}
    tool_vecs = np.concatenate([desc[:, None, :], ex], axis=1)
    sem = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(nq)]
    fold_of = assign_folds([q["query_id"] for q in queries], 3, 42)
    rec = fused_recall(queries, tool_ids, sem, op, gold_by_q, fold_of, fcfg, [5, 10])
    assert set(rec) == {"fusion_add", "fusion_mult"} and 0.0 <= rec["fusion_add"][5] <= 1.0
    print(f"[smoke] OK — oracle_cluster fusion recall 예시 {rec['fusion_add']}")


def main():
    ap = argparse.ArgumentParser(description="cluster prior A/B (category 굵기 문제 검증, GPU 불필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
