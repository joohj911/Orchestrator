"""experiment_rank_prior.py — 순위 기반 cluster prior로 classifier 대체 가능성 재검정.

진단 결과(results/prior_effect_diagnosis.json)가 밝힌 것:
  - 기존 무학습 prior 7종은 정규화 엔트로피 0.979~0.996 (완전 균등 = 1.0). 즉 거의 상수라
    fusion_add 에서 순위가 산술적으로 바뀔 수 없었다. **가설이 검정된 적이 없다.**
    (대조: classifier 0.30, oracle_cluster 0.37~0.66, category_oracle 0.00~0.19)
  - 원인은 softmax(T=0.05)가 e5 의 압축된 코사인 범위에서 평탄해지는 것 — 스케일 의존성.
  - 그러나 방향은 옳다: 강한 결합에서 순위 개선이 악화를 압도 (I3 centroid 32:3, cluster_max 31:3).
  - 용량도 있다: oracle_cluster_k128 이 I2 0.90 / I3 0.93 로 classifier(0.82 / 0.43)를 넘는다.
    → 비어 있는 것은 "쿼리 → cluster" 예측기 하나뿐.

이 실험의 처방: **순위 기반 shaping** — cluster 점수를 순위로 바꾼 뒤 고정 감쇠를 씌운다.
코사인 범위가 얼마나 압축돼 있든 prior 의 동적 범위가 보장되므로 스케일 의존성 자체가 사라진다.

  scorer (쿼리→cluster 점수)   : centroid | cluster_max | topm_mean (크기 정규화)
  shape  (점수→prior 분포)     : softmax(대조) | rank_pow(1/r^p) | rank_exp | topk_mass(하드 선택)

누출 방지: shape 파라미터도 fusion 계수와 **함께 fold train 에서만** 선택한다 (joint grid).
β=0 / λ=0 을 그리드에 포함해 prior 가 해로울 때 끌 수 있게 한다 (I3 에서 classifier 가
0.47→0.43 으로 손해 본 원인이 '끌 수 없음'이었다). 대체 후보는 무해가 최소 조건.

판정: recovery = (변형 이득) / (classifier 이득) @10.  classifier 이득은 I1 +0.10, I2 +0.19.
(I3 는 classifier 가 -0.04 라 비율이 무의미 — 절대 이득으로 읽는다.)

CLI: python scripts/experiment_rank_prior.py --config config.yaml [--smoke]
GPU 불필요 (m3 임베딩 캐시 재사용). 산출물: results/rank_prior_ab.json
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
from utils.config import load_config  # noqa: E402
from utils.scoring import recall_all  # noqa: E402
from m3_retrieval import (  # noqa: E402
    assign_folds, cosine_scores_multi, fusion_add_scores, fusion_mult_scores,
    real_prior_matrix, topk_ids, zscore_fit,
)
from experiment_cluster_prior import (  # noqa: E402
    build_flat_vectors, cluster_tools, oracle_cluster_prior, tool_representations,
)
from diagnose_prior_effect import maxgold_rank, mcnemar_p  # noqa: E402

FUSION = ["fusion_add", "fusion_mult"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- scorer

def cluster_scores(q_mat, flat, vec_cluster, k, mode, topm, cents):
    """(n_q, k) 쿼리-클러스터 점수. centroid 는 중심 코사인, 나머지는 멤버 벡터 집계."""
    if mode == "centroid":
        return np.asarray(q_mat @ cents.T, dtype=np.float32)
    sims = np.asarray(q_mat @ flat.T, dtype=np.float32)  # (n_q, n_vec)
    out = np.full((q_mat.shape[0], k), -1e9, dtype=np.float32)
    for c in range(k):
        idx = np.where(vec_cluster == c)[0]
        if idx.size == 0:
            continue
        s = sims[:, idx]
        if mode == "cluster_max":
            out[:, c] = s.max(axis=1)
        elif mode == "topm_mean":
            m = min(topm, s.shape[1])
            out[:, c] = np.sort(s, axis=1)[:, -m:].mean(axis=1)  # 크기 정규화
        else:
            raise ValueError(f"unknown scorer {mode}")
    return out


# ---------------------------------------------------------------- shape

def _ranks(scores):
    """행별 1-indexed 내림차순 순위."""
    order = np.argsort(-scores, axis=1, kind="stable")
    r = np.empty_like(order)
    seq = np.arange(1, scores.shape[1] + 1)[None, :].repeat(scores.shape[0], axis=0)
    np.put_along_axis(r, order, seq, axis=1)
    return r


def shape_prior(scores, mode, param):
    """cluster 점수 → prior 분포. rank_* 는 점수 스케일과 무관하게 동적 범위를 보장한다."""
    if mode == "softmax":
        z = scores / max(float(param), 1e-6)
        z = z - z.max(axis=1, keepdims=True)
        w = np.exp(z)
    else:
        r = _ranks(scores).astype(np.float64)
        if mode == "rank_pow":
            w = 1.0 / np.power(r, float(param))
        elif mode == "rank_exp":
            w = np.exp(-(r - 1.0) / max(float(param), 1e-6))
        elif mode == "topk_mass":
            w = np.where(r <= int(param), 1.0, 1e-6)
        else:
            raise ValueError(f"unknown shape {mode}")
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def prior_stats(prior_cluster):
    """cluster 축 원 스케일 통계 (category 축으로 접지 않는다 — 진단 스크립트의 한계 보완)."""
    k = prior_cluster.shape[1]
    uni = 1.0 / k
    mx = prior_cluster.max(axis=1)
    ent = []
    for row in prior_cluster:
        nz = row[row > 1e-12]
        ent.append(float(-(nz * np.log(nz)).sum() / math.log(k)))
    return {"k": k, "uniform": round(uni, 5),
            "mean_max": round(float(mx.mean()), 4),
            "max_over_uniform": round(float(mx.mean()) / uni, 1),
            "mean_norm_entropy": round(float(np.mean(ent)), 4)}


# ---------------------------------------------------------------- 누출 없는 평가

def _score(method, sem, prior, c, zm, zs, eps):
    if method == "fusion_add":
        return fusion_add_scores(sem, prior, c["alpha"], c["beta"], zm, zs)
    return fusion_mult_scores(sem, prior, c["lambda"], eps)


def _combos(method, fcfg, include_zero):
    if method == "fusion_add":
        betas = list(fcfg["beta_grid"]) + ([0.0] if include_zero else [])
        return [{"alpha": a, "beta": b, "lambda": None}
                for a in fcfg["alpha_grid"] for b in sorted(set(betas))]
    lams = list(fcfg["lambda_grid"]) + ([0.0] if include_zero else [])
    return [{"alpha": None, "beta": None, "lambda": l} for l in sorted(set(lams))]


def eval_leakfree(queries, tool_ids, sem_scores, prior_by_param, gold_by_q, gold_idx_by_q,
                  fold_of, fcfg, ks, headline_k, include_zero):
    """shape 파라미터 × fusion 계수를 fold train 에서 함께 선택하고 held-out fold 로 채점.

    prior_by_param: {param → (n_q, n_tools) prior 행렬}
    """
    eps = float(fcfg["epsilon"])
    folds = sorted({fold_of[str(q["query_id"])] for q in queries})
    chosen = {m: [] for m in FUSION}
    per_q = {m: np.zeros((len(queries), len(ks)), dtype=np.int8) for m in FUSION}
    mg = {m: np.zeros(len(queries), dtype=np.int32) for m in FUSION}

    for f in folds:
        tr = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] != f]
        te = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] == f]
        if not tr:
            continue
        zm, zs = zscore_fit(np.concatenate([sem_scores[i] for i in tr]))
        for m in FUSION:
            best = None
            for pname, pm in prior_by_param.items():
                for c in _combos(m, fcfg, include_zero):
                    hit = 0
                    for i in tr:
                        sc = _score(m, sem_scores[i], pm[i], c, zm, zs, eps)
                        hit += recall_all(topk_ids(sc, tool_ids, headline_k), gold_by_q[i])
                    rec = hit / len(tr)
                    # 동률이면 prior 의존이 작은 쪽 (m3 관행 유지)
                    dep = c["beta"] if m == "fusion_add" else c["lambda"]
                    key = (rec, -dep)
                    if best is None or key > best[0]:
                        best = (key, pname, c, round(rec, 4))
            _, pname, c, trrec = best
            chosen[m].append({"fold": int(f), "shape_param": pname,
                              "train_recall": trrec, **c})
            pm = prior_by_param[pname]
            for i in te:
                sc = _score(m, sem_scores[i], pm[i], c, zm, zs, eps)
                for ki, k in enumerate(ks):
                    per_q[m][i, ki] = recall_all(topk_ids(sc, tool_ids, k), gold_by_q[i])
                mg[m][i] = maxgold_rank(sc, gold_idx_by_q[i])

    out = {}
    for m in FUSION:
        out[m] = {"recall": {k: round(float(per_q[m][:, ki].mean()), 4)
                             for ki, k in enumerate(ks)},
                  "chosen": chosen[m],
                  "maxgold_mean": round(float(mg[m].mean()), 2)}
    return out, per_q, mg


def sign_vs_anchor(per_q_hit, anchor_hit, mg_treated, mg_anchor):
    imp = int(np.sum((per_q_hit == 1) & (anchor_hit == 0)))
    wor = int(np.sum((per_q_hit == 0) & (anchor_hit == 1)))
    d = mg_anchor - mg_treated
    return {"improved": imp, "worsened": wor, "net": imp - wor,
            "mcnemar_p": round(mcnemar_p(imp, wor), 4),
            "d_maxgold_mean": round(float(d.mean()), 2),
            "rank_improved": int(np.sum(d > 0)), "rank_worsened": int(np.sum(d < 0))}


# ---------------------------------------------------------------- 실행

def run(config_path):
    cfg = load_config(config_path, make_dirs=False)
    seed = int(cfg["seed"])
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    headline_k = int(cfg.get("analysis", {}).get("headline_k", 10))
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    fcfg = dict(cfg["fusion"])
    rp = cfg.get("rank_prior", {})
    cluster_ks = [int(x) for x in rp.get("cluster_ks", [128, 42])]
    scorers = list(rp.get("scorers", ["centroid", "cluster_max", "topm_mean"]))
    topm = int(rp.get("topm", 3))
    shapes = dict(rp.get("shapes", {
        "softmax": [0.01, 0.05], "rank_pow": [0.5, 1.0, 2.0],
        "rank_exp": [2.0, 5.0, 10.0], "topk_mass": [3, 5, 10]}))
    include_zero = bool(rp.get("include_zero", True))

    for p in ("tools_desc.npy", "tools_examples.npy"):
        if not os.path.isfile(os.path.join(emb_dir, p)):
            sys.exit(f"[rank-prior] 임베딩 캐시 없음: {emb_dir}/{p} (m3 먼저)")
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    tool_cats = [t["category"] for t in tools]
    idx_of = {t: i for i, t in enumerate(tool_ids)}

    reps = tool_representations(desc_mat, tool_ex_mat)
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)
    flat, owner = build_flat_vectors(tool_vecs)

    prior_path = os.path.join(data_dir, "class_prior_real.jsonl")
    clf_by_qid = ({str(r["query_id"]): r["prior"] for r in _read_jsonl(prior_path)}
                  if os.path.isfile(prior_path) else None)
    if clf_by_qid is None:
        print(f"[rank-prior] 경고: {prior_path} 없음 — classifier 기준선 생략 (recovery 계산 불가)")

    report = {"config": {"cluster_ks": cluster_ks, "scorers": scorers, "topm": topm,
                         "shapes": shapes, "include_zero": include_zero,
                         "headline_k": headline_k},
              "note": "shape 파라미터도 fold train 에서만 선택 (누출 없음). "
                      "beta=0/lambda=0 포함 — prior 를 끌 수 있어야 '무해'가 성립.",
              "splits": {}}

    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        gold_idx_by_q = [[idx_of[g] for g in gold if g in idx_of] for gold in gold_by_q]
        q_mat = np.load(os.path.join(emb_dir, f"queries_{split}.npy"))
        sem_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        fold_of = assign_folds([str(q["query_id"]) for q in queries], int(fcfg["n_folds"]), seed)

        anchor_hit = np.zeros((len(queries), len(ks)), dtype=np.int8)
        anchor_mg = np.zeros(len(queries), dtype=np.int32)
        for i in range(len(queries)):
            for ki, k in enumerate(ks):
                anchor_hit[i, ki] = recall_all(topk_ids(sem_scores[i], tool_ids, k), gold_by_q[i])
            anchor_mg[i] = maxgold_rank(sem_scores[i], gold_idx_by_q[i])
        anchor = {k: round(float(anchor_hit[:, ki].mean()), 4) for ki, k in enumerate(ks)}
        hk_i = ks.index(headline_k)
        print(f"\n[rank-prior] {split} 앵커 @{headline_k}={anchor[headline_k]:.2f} "
              f"maxgold {anchor_mg.mean():.1f}", flush=True)

        entry = {"anchor": anchor, "anchor_maxgold": round(float(anchor_mg.mean()), 2),
                 "references": {}, "variants": {}}

        # 참조: classifier / oracle_cluster (동일 하네스·동일 그리드)
        if clf_by_qid is not None:
            cm, _ = real_prior_matrix(clf_by_qid, [str(q["query_id"]) for q in queries], tool_cats)
            res, pq, mg = eval_leakfree(queries, tool_ids, sem_scores, {"-": cm}, gold_by_q,
                                        gold_idx_by_q, fold_of, fcfg, ks, headline_k, include_zero)
            best = max(res[m]["recall"][headline_k] for m in FUSION)
            entry["references"]["classifier"] = {
                "recall": {m: res[m]["recall"] for m in FUSION},
                "best@%d" % headline_k: best, "gain": round(best - anchor[headline_k], 4)}
            print(f"[rank-prior]   REF classifier @{headline_k} {best:.2f} "
                  f"({best - anchor[headline_k]:+.2f})", flush=True)
        clf_gain = entry["references"].get("classifier", {}).get("gain", 0.0)

        for ck in cluster_ks:
            labels, cents = cluster_tools(reps, ck, seed)
            vec_cluster = labels[owner]
            oc = oracle_cluster_prior(queries, tool_ids, labels, gold_by_q)
            res, _, _ = eval_leakfree(queries, tool_ids, sem_scores, {"-": oc}, gold_by_q,
                                      gold_idx_by_q, fold_of, fcfg, ks, headline_k, include_zero)
            ob = max(res[m]["recall"][headline_k] for m in FUSION)
            entry["references"][f"oracle_cluster_k{ck}"] = {
                "best@%d" % headline_k: ob, "gain": round(ob - anchor[headline_k], 4)}
            print(f"[rank-prior]   REF oracle_cluster_k{ck} @{headline_k} {ob:.2f}", flush=True)

            for sc_mode in scorers:
                cs = cluster_scores(q_mat, flat, vec_cluster, ck, sc_mode, topm, cents)
                for sh_mode, params in shapes.items():
                    priors, stats = {}, {}
                    for pv in params:
                        pc = shape_prior(cs, sh_mode, pv)
                        stats[str(pv)] = prior_stats(pc)
                        priors[str(pv)] = pc[:, labels].astype(np.float32)
                    res, pq, mg = eval_leakfree(queries, tool_ids, sem_scores, priors, gold_by_q,
                                                gold_idx_by_q, fold_of, fcfg, ks, headline_k,
                                                include_zero)
                    name = f"k{ck}/{sc_mode}/{sh_mode}"
                    best_m = max(FUSION, key=lambda m: res[m]["recall"][headline_k])
                    best = res[best_m]["recall"][headline_k]
                    gain = round(best - anchor[headline_k], 4)
                    entry["variants"][name] = {
                        "prior_stats": stats,
                        "recall": {m: res[m]["recall"] for m in FUSION},
                        "chosen": {m: res[m]["chosen"] for m in FUSION},
                        "best_method": best_m, "best@%d" % headline_k: best, "gain": gain,
                        "recovery_pct": (round(gain / clf_gain * 100, 1)
                                         if clf_gain > 1e-9 else None),
                        "sign_vs_anchor": sign_vs_anchor(
                            pq[best_m][:, hk_i], anchor_hit[:, hk_i], mg[best_m], anchor_mg),
                    }
                    print(f"[rank-prior]   {name:<34} @{headline_k} {best:.2f} ({gain:+.2f})",
                          flush=True)
        report["splits"][split] = entry

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "rank_prior_ab.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_report(report, splits, headline_k)
    print(f"\n[rank-prior] 저장: {out_path}")


def print_report(report, splits, hk):
    print("\n" + "=" * 108)
    print("순위 기반 cluster prior — classifier 대체 가능성 (전부 누출 없는 fold 선택)")
    print("=" * 108)
    for split in splits:
        e = report["splits"][split]
        a = e["anchor"][hk]
        print(f"\n### {split}  dense_multi 앵커 @{hk} = {a:.2f}  (maxgold {e['anchor_maxgold']})")
        for rn, rv in e["references"].items():
            print(f"  [REF] {rn:<26} @{hk} {rv['best@%d' % hk]:.2f}  ({rv['gain']:+.2f})")
        print(f"  {'variant':<34}{'@%d' % hk:>7}{'gain':>8}{'recov':>8}"
              f"{'엔트로피':>9}{'max/균등':>10}  개선/악화  Δmaxgold  선택")
        for name, v in sorted(e["variants"].items(), key=lambda x: -x[1]["gain"]):
            s = v["sign_vs_anchor"]
            first = next(iter(v["prior_stats"].values()))
            rec = "-" if v["recovery_pct"] is None else f"{v['recovery_pct']:.0f}%"
            sel = {}
            for c in v["chosen"][v["best_method"]]:
                sel[c["shape_param"]] = sel.get(c["shape_param"], 0) + 1
            selstr = ",".join(f"{k}×{n}" for k, n in sorted(sel.items(), key=lambda x: -x[1]))
            print(f"  {name:<34}{v['best@%d' % hk]:>7.2f}{v['gain']:>+8.2f}{rec:>8}"
                  f"{first['mean_norm_entropy']:>9.3f}{first['max_over_uniform']:>10.1f}"
                  f"   {s['improved']:>3}/{s['worsened']:<3}"
                  f" {s['d_maxgold_mean']:>+8.2f}  {v['best_method'][7:]}:{selstr}")
    print("\n" + "=" * 108)
    print("판정: recovery 가 I1·I2 에서 유의미(예: ≥50%)해야 classifier 대체 후보.")
    print("      엔트로피가 여전히 0.95+ 면 shaping 이 안 먹은 것 — 파라미터 범위를 넓혀야 함.")
    print("      gain 이 0 이고 선택이 beta=0 이면 '무해하지만 무용' (대체 실패, 손해는 없음).")
    print("=" * 108)


# ---------------------------------------------------------------- smoke

def _smoke():
    print("[smoke] rank prior 로직")
    rng = np.random.default_rng(0)
    k = 8
    sc = rng.standard_normal((5, k)).astype(np.float32)

    r = _ranks(sc)
    assert r.min() == 1 and r.max() == k
    for i in range(5):
        assert sorted(r[i]) == list(range(1, k + 1))
        assert r[i][int(np.argmax(sc[i]))] == 1

    # 순위 기반은 점수 스케일에 불변 — 압축된 코사인에서도 동적 범위 보장
    p1 = shape_prior(sc, "rank_pow", 1.0)
    p2 = shape_prior(sc * 0.001 + 0.9, "rank_pow", 1.0)  # 극단적으로 압축된 범위
    assert np.allclose(p1, p2, atol=1e-6), "rank shaping 이 스케일에 의존함"
    st1 = prior_stats(p1)
    assert st1["max_over_uniform"] > 2.0, st1
    # softmax 는 압축되면 균등으로 붕괴 (기존 실패의 재현)
    st_soft = prior_stats(shape_prior(sc * 0.001 + 0.9, "softmax", 0.05))
    assert st_soft["mean_norm_entropy"] > 0.99, st_soft
    assert st_soft["max_over_uniform"] < 1.05, st_soft

    assert shape_prior(sc, "rank_pow", 2.0)[0].max() > p1[0].max()  # p 클수록 첨예
    pt = shape_prior(sc, "topk_mass", 3)
    assert int(np.sum(pt[0] > 1e-3)) == 3, pt[0]

    # scorer: topm_mean 이 cluster_max 보다 작거나 같아야 (상위 m 평균 ≤ 최대)
    q = rng.standard_normal((4, 6)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    flat = rng.standard_normal((24, 6)).astype(np.float32)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    vc = np.repeat(np.arange(4), 6)
    cents = np.stack([flat[vc == c].mean(axis=0) for c in range(4)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    smax = cluster_scores(q, flat, vc, 4, "cluster_max", 3, cents)
    smean = cluster_scores(q, flat, vc, 4, "topm_mean", 3, cents)
    assert np.all(smean <= smax + 1e-6)
    assert cluster_scores(q, flat, vc, 4, "centroid", 3, cents).shape == (4, 4)

    # beta=0 포함 여부
    fc = {"alpha_grid": [0.3], "beta_grid": [0.1], "lambda_grid": [0.1]}
    assert any(c["beta"] == 0.0 for c in _combos("fusion_add", fc, True))
    assert not any(c["beta"] == 0.0 for c in _combos("fusion_add", fc, False))
    print("[smoke] OK — 순위 불변성·첨예도·하드선택·scorer 관계·beta0 그리드 정상")


def main():
    ap = argparse.ArgumentParser(description="순위 기반 cluster prior 대체 실험 (GPU 불필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
