"""diagnose_prior_effect.py — 무학습 prior null 들의 '순증 0' 정체 규명 (재계측, GPU 불필요).

배경: 무학습 prior 5~6종이 모두 dense_multi 와 같은 수치로 나왔고 이를 "흡수"로 해석했으나,
그 해석은 검증되지 않았다. 코드 확인 결과:
  - fusion_add = α·zscore(sem) + β·prior 이고 β=0 은 그리드에 없다 (config 주석: 의도적).
    동률 시 tiebreak 이 최소 β 를 고르므로 실선택은 (α=0.3, β=0.1) → zscore 기준 0.33σ 부양.
    500-tool 상위권에서 0.33σ 는 큰 교란 → "점수가 안 변했다"는 해석은 성립하지 않는다.
  - grid_search_fusion 의 tiebreak(m3_retrieval.py:138)은 "동률이면 prior 의존 작은 쪽"이다.
    즉 null 은 "정보 없음"이 아니라 "train fold 에서 이득이 확인되지 않음"일 수 있다.
  - recall_with_prior 는 fold 별 선택 계수를 계산만 하고 반환하지 않는다 → 기존 결과 파일에
    증거가 남아 있지 않다.
따라서 Recall_all@K 의 이진 비트만 보던 계측을 다시 한다.

측정 항목 (prior × split):
  1. k-fold 선택 계수 분포 — 최소 β 선택 비율 (tiebreak 발동 여부의 직접 증거)
  2. 강제 β/λ 스윕 — 선택을 끄고 결합 강도별 recall 곡선 (β=0 은 dense_multi 앵커)
  3. 쿼리 단위 부호 분해 — 개선/악화/불변 수 + McNemar 정확검정 ('순증 0'이 0-0 인지 +n-n 인지)
  4. max-gold-rank 이동 — Recall_all@K 가 임계하는 통계량 자체. 평균·중앙·Δ 분포·근접미스
  5. 흡수 직접 진단 — prior 첨예도, top1-sem 이 prior 최상위군에 속하는 비율, corr(prior, sem)

판정: 봉우리(선택 분산 → false negative) / 상쇄(gating 유망) / 사각지대(순위만 이동) /
      유해(단조 감소) / 흡수 확증 중 하나로 자동 분류하고 근거 수치를 함께 출력한다.

주의: 강제 스윕의 최적 β 는 test 에서 사후 선택한 값이므로 **방법 성능이 아니다**(누출).
      상한·기전 진단 전용이며 출력에도 그렇게 표기한다.

CLI: python scripts/diagnose_prior_effect.py --config config.yaml [--priors a,b] [--smoke]
산출물: results/prior_effect_diagnosis.json + 콘솔 표
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
    grid_search_fusion, oracle_prior_matrix, real_prior_matrix, topk_ids, zscore_fit,
)
from experiment_cluster_prior import (  # noqa: E402
    build_flat_vectors, centroid_prior, cluster_max_prior, cluster_tools,
    oracle_cluster_prior, scenario_priors, tool_representations,
)
from experiment_service_boost import build_groups, service_boost_prior, service_key  # noqa: E402

FUSION = ["fusion_add", "fusion_mult"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- 순위 유틸

def ranks_from_scores(scores):
    """topk_ids 와 동일한 정렬(stable, 내림차순) 기준 1-indexed 순위."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    r = np.empty(len(order), dtype=np.int32)
    r[order] = np.arange(1, len(order) + 1)
    return r


def maxgold_rank(scores, gold_idx):
    """gold 중 가장 나쁜 순위. Recall_all@K = 1 ⟺ maxgold_rank ≤ K (정의상 동치)."""
    if not gold_idx:
        return -1
    r = ranks_from_scores(scores)
    return int(max(int(r[g]) for g in gold_idx))


def mcnemar_p(n_improved, n_worsened):
    """불일치 쌍에 대한 이항 정확검정(양측). '순증 0'이 우연 범위인지 판단."""
    n = n_improved + n_worsened
    if n == 0:
        return 1.0
    k = min(n_improved, n_worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return float(min(1.0, 2.0 * tail))


# ---------------------------------------------------------------- prior 진단

def prior_shape_stats(prior_mat, sem_scores, tool_cats):
    """prior 자체의 첨예도와 semantic 과의 중복도 (흡수 가설 직접 검사)."""
    n_q = len(sem_scores)
    max_probs, entropies, corrs, top_agree = [], [], [], 0
    for i in range(n_q):
        p = np.asarray(prior_mat[i], dtype=np.float64)
        s = np.asarray(sem_scores[i], dtype=np.float64)
        # category 축으로 접어서 첨예도 측정 (prior 는 tool 축으로 펼쳐진 상태)
        cat_p = {}
        for c, v in zip(tool_cats, p):
            cat_p[c] = max(cat_p.get(c, 0.0), float(v))
        vals = np.array(list(cat_p.values()), dtype=np.float64)
        tot = vals.sum()
        if tot > 1e-12:
            q = vals / tot
            nz = q[q > 1e-12]
            entropies.append(float(-(nz * np.log(nz)).sum() / math.log(len(vals))))
            max_probs.append(float(q.max()))
        # top-1 semantic tool 이 prior 최상위군에 속하는가 (= prior 가 sem 을 재확인만 하는가)
        if p.max() > 0:
            top1 = int(np.argmax(s))
            top_agree += int(p[top1] >= p.max() - 1e-9)
        if p.std() > 1e-12 and s.std() > 1e-12:
            corrs.append(float(np.corrcoef(p, s)[0, 1]))
    return {
        "mean_max_category_prob": round(float(np.mean(max_probs)), 4) if max_probs else None,
        "mean_norm_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
        "top1_sem_in_prior_argmax_rate": round(top_agree / max(1, n_q), 4),
        "mean_corr_prior_sem": round(float(np.mean(corrs)), 4) if corrs else None,
    }


def selected_coefficients(queries, tool_ids, sem_scores, prior_mat, gold_by_q, fold_of, fcfg):
    """m3 와 동일한 leak-free k-fold 선택을 재현하고 **선택 계수를 기록**한다."""
    out = {m: [] for m in FUSION}
    zparams = {}
    for f in sorted({fold_of[str(q["query_id"])] for q in queries}):
        tr = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] != f]
        if not tr:
            continue
        zm, zs = zscore_fit(np.concatenate([sem_scores[i] for i in tr]))
        zparams[f] = (zm, zs)
        for m in FUSION:
            c = grid_search_fusion(m, [sem_scores[i] for i in tr], [prior_mat[i] for i in tr],
                                   [set(gold_by_q[i]) for i in tr], tool_ids, fcfg, zm, zs)
            out[m].append({"fold": int(f), **c})
    return out, zparams


def selected_recall(queries, tool_ids, sem_scores, prior_mat, gold_by_q, fold_of,
                    coeffs, zparams, eps, k):
    """선택된 계수로 채점 — 기존 보고 수치를 재현해 대조 가능하게 한다."""
    out = {}
    for m in FUSION:
        by_fold = {c["fold"]: c for c in coeffs[m]}
        hits = 0
        for i, q in enumerate(queries):
            f = fold_of[str(q["query_id"])]
            c, (zm, zs) = by_fold[f], zparams[f]
            sc = (fusion_add_scores(sem_scores[i], prior_mat[i], c["alpha"], c["beta"], zm, zs)
                  if m == "fusion_add" else
                  fusion_mult_scores(sem_scores[i], prior_mat[i], c["lambda"], eps))
            hits += recall_all(topk_ids(sc, tool_ids, k), gold_by_q[i])
        out[m] = round(hits / max(1, len(queries)), 4)
    return out


# ---------------------------------------------------------------- 강제 스윕

def sweep(method, queries, tool_ids, sem_scores, prior_mat, gold_by_q, gold_idx_by_q,
          values, ks, headline_k, alpha_fixed, eps):
    """결합 강도를 강제로 밀면서 recall·부호 분해·순위 이동을 전부 기록.

    values[0] 은 반드시 0 (add: β=0, mult: λ=0) — dense_multi 와 동일 랭킹이 되는 앵커.
    """
    n_q = len(queries)
    zm, zs = zscore_fit(np.concatenate(list(sem_scores)))  # 진단용 전역 정규화 (선택 아님)

    def scores_at(i, v):
        if method == "fusion_add":
            return fusion_add_scores(sem_scores[i], prior_mat[i], alpha_fixed, v, zm, zs)
        return fusion_mult_scores(sem_scores[i], prior_mat[i], v, eps)

    rows, base_hit, base_mg = [], None, None
    for vi, v in enumerate(values):
        rec = {k: 0 for k in ks}
        hit_h, mg = np.zeros(n_q, dtype=np.int8), np.zeros(n_q, dtype=np.int32)
        for i in range(n_q):
            sc = scores_at(i, v)
            for k in ks:
                rec[k] += recall_all(topk_ids(sc, tool_ids, k), gold_by_q[i])
            hit_h[i] = recall_all(topk_ids(sc, tool_ids, headline_k), gold_by_q[i])
            mg[i] = maxgold_rank(sc, gold_idx_by_q[i])
        if vi == 0:
            base_hit, base_mg = hit_h.copy(), mg.copy()
            # 자기검증: 정의상 Recall_all@k == (maxgold_rank ≤ k) 여야 한다
            agree = int(np.sum((mg <= headline_k).astype(np.int8) == hit_h))
            if agree != n_q:
                print(f"  [경고] recall/순위 정의 불일치 {n_q - agree}건 — 동점 처리 확인 필요")

        imp = int(np.sum((hit_h == 1) & (base_hit == 0)))
        wor = int(np.sum((hit_h == 0) & (base_hit == 1)))
        d = base_mg - mg  # +면 순위 개선 (숫자가 작아짐)
        rows.append({
            "value": float(v),
            "recall": {k: round(rec[k] / max(1, n_q), 4) for k in ks},
            "improved": imp, "worsened": wor, "unchanged": n_q - imp - wor,
            "net": imp - wor, "mcnemar_p": round(mcnemar_p(imp, wor), 4),
            "maxgold_mean": round(float(mg.mean()), 2),
            "maxgold_median": float(np.median(mg)),
            "d_maxgold_mean": round(float(d.mean()), 2),
            "rank_improved": int(np.sum(d > 0)), "rank_worsened": int(np.sum(d < 0)),
            "rank_same": int(np.sum(d == 0)),
            "near_miss": int(np.sum((mg > headline_k) & (mg <= 2 * headline_k))),
        })
    return rows


def verdict(rows, headline_k, peak_margin):
    """스윕 결과를 다섯 세계 중 하나로 분류하고 근거 수치를 반환."""
    base = rows[0]["recall"][headline_k]
    treated = rows[1:]
    if not treated:
        return {"label": "판정불가", "why": "스윕 값이 앵커뿐"}
    best = max(treated, key=lambda r: r["recall"][headline_k])
    gain = round(best["recall"][headline_k] - base, 4)
    worst = min(treated, key=lambda r: r["recall"][headline_k])
    monotone_down = all(
        treated[i]["recall"][headline_k] <= treated[i - 1]["recall"][headline_k] + 1e-9
        for i in range(1, len(treated)))
    # 부호 분해는 실제 그리드가 도달 가능한 결합(가장 약한 쪽)에서 읽는다
    probe = treated[0]
    if gain >= peak_margin:
        return {"label": "봉우리 — 선택 분산 의심 (false negative 후보)", "gain": gain,
                "at_value": best["value"],
                "why": f"강제 결합 {best['value']} 에서 @{headline_k} {base:.2f}→"
                       f"{best['recall'][headline_k]:.2f} (+{gain:.2f}). "
                       f"k-fold 선택이 이 지점을 못 찾았다면 계수 선택 분산 문제 (정보 문제 아님). "
                       f"※ 이 값은 사후 선택이라 방법 성능이 아니다"}
    if probe["improved"] >= 2 and probe["worsened"] >= 2:
        return {"label": "상쇄 — gating 유망", "gain": gain,
                "why": f"결합 {probe['value']} 에서 개선 {probe['improved']} / 악화 "
                       f"{probe['worsened']} (순증 {probe['net']:+d}, McNemar p="
                       f"{probe['mcnemar_p']}). 악화 쿼리를 끄면 개선분이 남을 수 있다"}
    if probe["d_maxgold_mean"] >= 1.0 or probe["rank_improved"] >= 5:
        return {"label": "사각지대 — 순위는 이동, 이진 비트 미변", "gain": gain,
                "why": f"maxgold rank 평균 {rows[0]['maxgold_mean']}→{probe['maxgold_mean']} "
                       f"(Δ{probe['d_maxgold_mean']:+.2f}), 순위 개선 {probe['rank_improved']} / "
                       f"악화 {probe['rank_worsened']} 쿼리. 근접미스 {probe['near_miss']}건이 "
                       f"@{headline_k} 밖에 대기 중"}
    if monotone_down and worst["recall"][headline_k] < base - 1e-9:
        return {"label": "유해 — 결합 강화가 단조 악화", "gain": gain,
                "why": f"@{headline_k} 가 {base:.2f} 에서 {worst['recall'][headline_k]:.2f} 까지 "
                       f"단조 하락. prior 가 노이즈"}
    return {"label": "흡수 확증 — 점수·순위 모두 실질 변화 없음", "gain": gain,
            "why": f"결합 {probe['value']} 에서 순위 개선 {probe['rank_improved']} / 악화 "
                   f"{probe['rank_worsened']} / 불변 {probe['rank_same']}, "
                   f"Δmaxgold {probe['d_maxgold_mean']:+.2f}. semantic 과 중복"}


# ---------------------------------------------------------------- prior 조립

def build_priors(cfg, split, queries, tool_ids, tool_cats, tool_vecs, reps, q_mat,
                 sem_scores, gold_by_q, gold_idx_by_q, flat, owner, scen_labels, only):
    """진단 대상 prior 행렬들을 (이름 → (n_q, n_tools)) 로 조립."""
    data_dir = cfg["paths"]["data_dir"]
    pd_cfg = cfg.get("prior_diagnostics", {})
    cluster_ks = [int(x) for x in pd_cfg.get("cluster_ks", [42, 128])]
    temperature = float(cfg.get("cluster_prior", {}).get("centroid_temperature", 0.05))
    seed = int(cfg["seed"])
    n_tools = len(tool_ids)
    out = {}

    # 참조: 학습된 classifier (작동하는 prior) / category oracle (상한)
    prior_path = os.path.join(data_dir, "class_prior_real.jsonl")
    if os.path.isfile(prior_path):
        by_qid = {str(r["query_id"]): r["prior"] for r in _read_jsonl(prior_path)}
        mat, missing = real_prior_matrix(by_qid, [str(q["query_id"]) for q in queries], tool_cats)
        if missing:
            print(f"  [주의] {split}: classifier prior 없는 쿼리 {missing}개 (0 행)")
        out["REF_category_real"] = mat
    else:
        print(f"  [주의] {prior_path} 없음 — classifier 참조 생략 (m4 먼저)")
    out["REF_category_oracle"] = oracle_prior_matrix(
        [q.get("gold_categories", []) for q in queries], tool_cats)

    # 대상: 무학습 prior
    for ck in cluster_ks:
        labels, cents = cluster_tools(reps, ck, seed)
        out[f"centroid_k{ck}"] = centroid_prior(q_mat, labels, cents, temperature)
        out[f"cluster_max_k{ck}"] = cluster_max_prior(q_mat, tool_vecs, labels, ck, temperature)
        sc_prior, sc_oracle = scenario_priors(q_mat, flat, owner, scen_labels[ck], ck,
                                              n_tools, gold_idx_by_q, temperature)
        out[f"scenario_max_k{ck}"] = sc_prior
        out[f"REF_oracle_cluster_k{ck}"] = oracle_cluster_prior(queries, tool_ids, labels, gold_by_q)
        del sc_oracle

    groups = build_groups(tool_ids)
    group_of_tool = {i: service_key(t) for i, t in enumerate(tool_ids)}
    out["service_boost"] = service_boost_prior(sem_scores, groups, group_of_tool,
                                               n_tools, temperature)
    if only:
        out = {k: v for k, v in out.items() if k in only}
    return out


# ---------------------------------------------------------------- 실행

def run(config_path, only=None):
    cfg = load_config(config_path, make_dirs=False)
    seed = int(cfg["seed"])
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    headline_k = int(cfg.get("analysis", {}).get("headline_k", 10))
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    fcfg = cfg["fusion"]
    eps = float(fcfg["epsilon"])
    pd_cfg = cfg.get("prior_diagnostics", {})
    beta_sweep = [float(x) for x in pd_cfg.get(
        "beta_sweep", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])]
    lambda_sweep = [float(x) for x in pd_cfg.get(
        "lambda_sweep", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])]
    alpha_fixed = float(pd_cfg.get("alpha_fixed", 1.0))
    peak_margin = float(pd_cfg.get("peak_margin", 0.03))
    cluster_ks = [int(x) for x in pd_cfg.get("cluster_ks", [42, 128])]
    if beta_sweep[0] != 0.0 or lambda_sweep[0] != 0.0:
        sys.exit("[diag] beta_sweep/lambda_sweep 의 첫 값은 0 이어야 한다 (dense_multi 앵커)")

    for p in ("tools_desc.npy", "tools_examples.npy"):
        if not os.path.isfile(os.path.join(emb_dir, p)):
            sys.exit(f"[diag] 임베딩 캐시 없음: {emb_dir}/{p} (m3 먼저 실행)")
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    tool_cats = [t["category"] for t in tools]
    idx_of = {t: i for i, t in enumerate(tool_ids)}

    reps = tool_representations(desc_mat, tool_ex_mat)
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)
    flat, owner = build_flat_vectors(tool_vecs)
    scen_labels = {ck: cluster_tools(flat, ck, seed)[0] for ck in cluster_ks}

    # 실제 그리드가 도달 가능한 결합비 범위 (해석용 참고선)
    a_grid, b_grid = fcfg["alpha_grid"], fcfg["beta_grid"]
    ratio_lo = min(b_grid) / max(a_grid)
    ratio_hi = max(b_grid) / min(a_grid)

    report = {
        "note": "강제 스윕의 최적값은 test 사후 선택 = 누출. 방법 성능이 아니라 기전 진단용.",
        "config": {"beta_sweep": beta_sweep, "lambda_sweep": lambda_sweep,
                   "alpha_fixed": alpha_fixed, "headline_k": headline_k,
                   "peak_margin": peak_margin, "cluster_ks": cluster_ks,
                   "grid_ratio_range": [round(ratio_lo, 3), round(ratio_hi, 3)]},
        "splits": {},
    }

    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        gold_idx_by_q = [[idx_of[g] for g in gold if g in idx_of] for gold in gold_by_q]
        q_npy = os.path.join(emb_dir, f"queries_{split}.npy")
        if not os.path.isfile(q_npy):
            sys.exit(f"[diag] 쿼리 임베딩 캐시 없음: {q_npy} (m3 먼저)")
        q_mat = np.load(q_npy)
        sem_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        fold_of = assign_folds([str(q["query_id"]) for q in queries], int(fcfg["n_folds"]), seed)

        print(f"\n[diag] {split}: prior 조립 중...", flush=True)
        priors = build_priors(cfg, split, queries, tool_ids, tool_cats, tool_vecs, reps,
                              q_mat, sem_scores, gold_by_q, gold_idx_by_q,
                              flat, owner, scen_labels, only)

        # dense_multi 앵커 (모든 prior 공통)
        base_mg = np.array([maxgold_rank(sem_scores[i], gold_idx_by_q[i])
                            for i in range(len(queries))], dtype=np.int32)
        anchor = {k: round(sum(recall_all(topk_ids(sem_scores[i], tool_ids, k), gold_by_q[i])
                               for i in range(len(queries))) / len(queries), 4) for k in ks}
        print(f"[diag] {split} dense_multi 앵커: "
              + " ".join(f"@{k}={anchor[k]:.2f}" for k in ks)
              + f" | maxgold rank 평균 {base_mg.mean():.1f} 중앙 {np.median(base_mg):.0f}")

        entry = {"dense_multi_anchor": anchor,
                 "anchor_maxgold_mean": round(float(base_mg.mean()), 2),
                 "priors": {}}
        for name, prior_mat in priors.items():
            print(f"[diag]   {name} ...", flush=True)
            shape = prior_shape_stats(prior_mat, sem_scores, tool_cats)
            coeffs, zparams = selected_coefficients(
                queries, tool_ids, sem_scores, prior_mat, gold_by_q, fold_of, fcfg)
            sel_rec = selected_recall(queries, tool_ids, sem_scores, prior_mat, gold_by_q,
                                      fold_of, coeffs, zparams, eps, headline_k)
            min_b, min_l = min(b_grid), min(fcfg["lambda_grid"])
            at_floor = {
                "fusion_add": sum(1 for c in coeffs["fusion_add"] if c["beta"] == min_b),
                "fusion_mult": sum(1 for c in coeffs["fusion_mult"] if c["lambda"] == min_l),
                "n_folds": len(coeffs["fusion_add"]),
            }
            add_rows = sweep("fusion_add", queries, tool_ids, sem_scores, prior_mat, gold_by_q,
                             gold_idx_by_q, beta_sweep, ks, headline_k, alpha_fixed, eps)
            mult_rows = sweep("fusion_mult", queries, tool_ids, sem_scores, prior_mat, gold_by_q,
                              gold_idx_by_q, lambda_sweep, ks, headline_k, alpha_fixed, eps)
            entry["priors"][name] = {
                "prior_shape": shape, "selected_coeffs": coeffs,
                "selected_recall@%d" % headline_k: sel_rec, "selected_at_floor": at_floor,
                "sweep_add": add_rows, "sweep_mult": mult_rows,
                "verdict_add": verdict(add_rows, headline_k, peak_margin),
                "verdict_mult": verdict(mult_rows, headline_k, peak_margin),
            }
        report["splits"][split] = entry

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "prior_effect_diagnosis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report, splits, ks, headline_k)
    print(f"\n[diag] 저장: {out_path}")


def print_report(report, splits, ks, headline_k):
    cfgd = report["config"]
    print("\n" + "=" * 100)
    print("prior 효과 재계측 — '순증 0' 의 정체")
    print("=" * 100)
    print(f"강제 스윕은 α={cfgd['alpha_fixed']} 고정, β(또는 λ)=0 이 dense_multi 와 동일 랭킹.")
    print(f"실제 그리드가 도달 가능한 β/α 범위: {cfgd['grid_ratio_range'][0]} ~ "
          f"{cfgd['grid_ratio_range'][1]} (이 밖의 스윕 값은 참고용)")
    print("스윕 최적값은 test 사후 선택 = 누출. 방법 성능이 아니라 기전 진단용.")

    for split in splits:
        e = report["splits"][split]
        print("\n" + "#" * 100)
        print(f"# {split}  |  dense_multi 앵커: "
              + "  ".join(f"@{k}={e['dense_multi_anchor'][k]:.2f}" for k in ks)
              + f"  |  maxgold rank 평균 {e['anchor_maxgold_mean']}")
        print("#" * 100)
        for name, d in e["priors"].items():
            sh = d["prior_shape"]
            fl = d["selected_at_floor"]
            sel = d[f"selected_recall@{headline_k}"]
            print(f"\n--- {name} ---")
            print(f"prior 형태: category 최대확률 {sh['mean_max_category_prob']} | "
                  f"정규화 엔트로피 {sh['mean_norm_entropy']} | "
                  f"top1-sem 이 prior 최상위군 {sh['top1_sem_in_prior_argmax_rate']} | "
                  f"corr(prior,sem) {sh['mean_corr_prior_sem']}")
            print(f"k-fold 선택 계수: add "
                  + ", ".join(f"f{c['fold']}(α{c['alpha']},β{c['beta']})"
                              for c in d["selected_coeffs"]["fusion_add"])
                  + f"  [최소 β 선택 {fl['fusion_add']}/{fl['n_folds']}]")
            print(f"                  mult "
                  + ", ".join(f"f{c['fold']}(λ{c['lambda']})"
                              for c in d["selected_coeffs"]["fusion_mult"])
                  + f"  [최소 λ 선택 {fl['fusion_mult']}/{fl['n_folds']}]")
            print(f"선택 계수 기준 @{headline_k}: add {sel['fusion_add']:.2f} / "
                  f"mult {sel['fusion_mult']:.2f}  "
                  f"(앵커 {e['dense_multi_anchor'][headline_k]:.2f} — 기존 보고와 대조)")

            for label, rows, vd in (("fusion_add (β 스윕)", d["sweep_add"], d["verdict_add"]),
                                    ("fusion_mult (λ 스윕)", d["sweep_mult"], d["verdict_mult"])):
                print(f"\n  [{label}]")
                head = ("   val  " + "".join(f"  R@{k}".rjust(7) for k in ks)
                        + f" | @{headline_k} 개선/악화/불변  순증   p"
                        + " | maxgold 평균(중앙) Δ평균 순위개선/악화 | 근접미스")
                print(head)
                for r in rows:
                    print(f"  {r['value']:>5.2f}  "
                          + "".join(f"{r['recall'][k]:.2f}".rjust(7) for k in ks)
                          + f" |    {r['improved']:>3}/{r['worsened']:>3}/{r['unchanged']:>3}"
                          + f"  {r['net']:+4d}  {r['mcnemar_p']:.2f}"
                          + f" |  {r['maxgold_mean']:>7.1f}({r['maxgold_median']:>4.0f})"
                          + f" {r['d_maxgold_mean']:+6.2f}"
                          + f"  {r['rank_improved']:>3}/{r['rank_worsened']:>3}"
                          + f" |   {r['near_miss']:>3}")
                print(f"  판정: {vd['label']}  (Δ@{headline_k} 최대 {vd.get('gain', 0):+.2f})")
                print(f"        근거: {vd['why']}")

    print("\n" + "=" * 100)
    print("읽는 법: '봉우리'면 계수 선택 분산 문제 → 기존 null 은 false negative.")
    print("        '상쇄'면 gating(쿼리별 on/off)이 직접 처방.")
    print("        '사각지대'면 순위는 개선됐고 K 나 지표를 재고해야 함 (근접미스 수 참고).")
    print("        '흡수 확증'이면 그 prior 라인은 실제로 종결.")
    print("=" * 100)


# ---------------------------------------------------------------- smoke

def _smoke():
    print("[smoke] prior 효과 진단 로직")
    # 순위·recall 정의 동치
    sc = np.array([0.9, 0.5, 0.7, 0.1], dtype=np.float32)
    r = ranks_from_scores(sc)
    assert list(r) == [1, 3, 2, 4], r
    assert maxgold_rank(sc, [0, 2]) == 2 and maxgold_rank(sc, [0, 3]) == 4

    # McNemar
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) > 0.9          # 완전 상쇄 → 우연 범위
    assert mcnemar_p(8, 0) < 0.05         # 한쪽으로만 이동 → 유의

    tool_ids = [f"C{i//5}__svc{i//2}__api{i}" for i in range(20)]
    tool_cats = [f"C{i//5}" for i in range(20)]
    queries = [{"query_id": f"q{i}", "gold_tools": [tool_ids[i % 20]],
                "gold_categories": [tool_cats[i % 20]]} for i in range(20)]
    gold_by_q = [list(q["gold_tools"]) for q in queries]
    gold_idx = [[i % 20] for i in range(20)]
    rng = np.random.default_rng(0)
    sem = [rng.random(20).astype(np.float32) for _ in range(20)]
    ks, hk = [5, 10], 10
    fcfg = {"alpha_grid": [0.3, 0.7], "beta_grid": [0.1, 0.5], "lambda_grid": [0.1, 0.5],
            "epsilon": 0.05, "n_folds": 2}

    # (1) 정보 있는 prior: gold 에 질량 → 강제 결합이 커질수록 recall 상승 (봉우리/상승 검출)
    good = np.zeros((20, 20), dtype=np.float32)
    for i in range(20):
        good[i, gold_idx[i][0]] = 1.0
    rows = sweep("fusion_add", queries, tool_ids, sem, good, gold_by_q, gold_idx,
                 [0.0, 0.5, 2.0], ks, hk, 1.0, 0.05)
    assert rows[0]["recall"][hk] <= rows[-1]["recall"][hk], [r["recall"] for r in rows]
    assert rows[-1]["d_maxgold_mean"] >= 0, rows[-1]
    v = verdict(rows, hk, 0.03)
    assert "봉우리" in v["label"] or rows[0]["recall"][hk] == 1.0, v

    # (2) 상수 prior: 랭킹 불변 → 흡수 확증 + 순위 이동 0
    flat_prior = np.full((20, 20), 0.5, dtype=np.float32)
    rows2 = sweep("fusion_add", queries, tool_ids, sem, flat_prior, gold_by_q, gold_idx,
                  [0.0, 0.5, 2.0], ks, hk, 1.0, 0.05)
    assert all(r["rank_improved"] == 0 and r["rank_worsened"] == 0 for r in rows2), rows2
    assert "흡수" in verdict(rows2, hk, 0.03)["label"]

    # (3) 선택 계수 기록 경로
    fold_of = {f"q{i}": i % 2 for i in range(20)}
    coeffs, zp = selected_coefficients(queries, tool_ids, sem, good, gold_by_q, fold_of, fcfg)
    assert len(coeffs["fusion_add"]) == 2 and "beta" in coeffs["fusion_add"][0]
    rec = selected_recall(queries, tool_ids, sem, good, gold_by_q, fold_of, coeffs, zp, 0.05, hk)
    assert 0.0 <= rec["fusion_add"] <= 1.0

    # (4) prior 형태 통계
    sh = prior_shape_stats(good, sem, tool_cats)
    assert sh["mean_max_category_prob"] is not None and 0 <= sh["top1_sem_in_prior_argmax_rate"] <= 1
    print("[smoke] OK — 순위/recall 동치, McNemar, 스윕 상승·불변 검출, 계수 기록, 형태 통계 정상")


def main():
    ap = argparse.ArgumentParser(
        description="무학습 prior null 재계측 — 부호 분해·순위 이동·강제 스윕 (GPU 불필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--priors", default="", help="쉼표로 구분한 prior 이름만 진단 (기본: 전체)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    only = set(x.strip() for x in args.priors.split(",") if x.strip()) or None
    run(args.config, only)


if __name__ == "__main__":
    main()
