"""experiment_query_split.py — 규칙 기반 멀티 인텐트 분해 + 세그먼트 품질 게이트 (LLM 0).

배경: I3 병목은 멀티 인텐트 희석이다 (dense_multi @50 0.87 vs @10 0.47 — 정답은 찾히는데
저K 에서 밀린다). ToolQP(arXiv 2601.07782) 의 분해 아이디어를 서빙 LLM 없이 검증한다.

1차 실행 결과와 진단 (2026-08-06):
  - 분해율 100/100, 평균 3.17~3.35 세그먼트. **I1(단일 인텐트)조차 3.17개** → 과분해 확정.
    원인: ToolBench 쿼리는 서사형이라 문장 경계 ≠ 인텐트 경계 (맥락 문장·인사말 포함).
  - 그런데도 I2 union_max +0.10 (recovery 53%, 지금까지 무학습 최고), Δmaxgold +6.48.
  - I2 seg_quota 는 @5 에서 +0.08 (union_max +0.02) → 슬롯 배분 축의 첫 증거.
  - I3 는 @10 +0.04 지만 @5 −0.04, seg_quota 는 @5·@10 −0.08 → 패딩 세그먼트가 상위 슬롯 점유.
  - sent vs sent_conj 차이는 세그먼트 4개(100 쿼리) 수준 → **언어별 접속사 규칙 투자 불필요.**

이번 처방 — 세그먼트 품질 게이트 (추가 비용 0):
  신호 m(s) = 세그먼트 단독의 tool 인덱스 최대 유사도. 실제 요청은 특정 API 와 잘 맞아
  m 이 높고, 서사 패딩("파리 여행을 계획 중입니다", "미리 감사합니다")은 대응 API 가 없어
  낮다. 즉 **retriever 자신을 패딩 검출기로** 쓴다.
  임계 형태 3종을 fold train 에서만 선택 (스케일 압축에 데었으므로 절대 임계만 쓰지 않는다):
    abs  : m(s) ≥ τ                      — 단순하나 데이터 의존
    rel  : m(s) ≥ m(전체 쿼리) − δ        — 쿼리별 자기 정규화
    rank : 쿼리 내 m(s) 상위 r개만 유지   — 스케일 무관, n 을 확실히 묶음
  전부 탈락하면 최선 세그먼트 1개로 폴백 (빈 집합 금지).

알려진 위험: 게이트가 retriever 점수를 쓰므로 **문서가 부실한 tool 을 요청하는 세그먼트가
버려질 수 있다** — 우리가 도와야 하는 바로 그 경우다. abs 가 가장 취약하고 rel/rank 는
"전부 낮아도 최선은 남긴다"라 덜하다. 그래서 `gold_only_in_dropped` (버려진 세그먼트만
gold 를 표면화한 건수)를 반드시 집계하고, 세그먼트 전체를 JSONL 로 덤프해 눈으로 검토한다.

CLI: python scripts/experiment_query_split.py --config config.yaml [--smoke]
GPU: 세그먼트 임베딩만. 산출물: results/query_split_ab.json, results/query_split_segments.jsonl
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
from utils.config import load_config  # noqa: E402
from utils.scoring import recall_all  # noqa: E402
from m3_retrieval import assign_folds, cosine_scores_multi, topk_ids  # noqa: E402
from diagnose_prior_effect import maxgold_rank, mcnemar_p  # noqa: E402

# 문장 종결 부호만. 전각은 뒤 공백 없이 분리(CJK), ASCII 는 뒤 공백 요구(소수점 보호).
_SENT_SPLIT = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")
# 영어 전용 — 1차 실행에서 효과가 세그먼트 4개 수준으로 확인됨. 참조 변형으로만 유지.
_CONJ_SPLIT = re.compile(
    r",?\s+(?:and also|also,?|additionally,?|then|as well as|and then|plus,?)\s+",
    re.IGNORECASE)

MODES = ("sent", "sent_conj")
VARIANTS = ("union_max", "seg_only", "seg_quota")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- 분해

def split_query(query: str, min_chars: int, mode: str = "sent") -> list[str]:
    """규칙 기반 분해. 짧은 조각은 앞 조각에 도로 붙인다 (과분해 자동 교정).

    mode="sent"      : 문장 종결 부호만 (다국어 안전, 배포 기본)
    mode="sent_conj" : 문장 부호 + 영어 접속사 (언어 의존 — 참조용)
    """
    parts = []
    for sent in _SENT_SPLIT.split(query.strip()):
        chunks = _CONJ_SPLIT.split(sent) if mode == "sent_conj" else [sent]
        for seg in chunks:
            seg = seg.strip(" ,;、，")
            if seg:
                parts.append(seg)
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < min_chars:
            merged[-1] = merged[-1] + " " + p  # 공백 결합 — 언어 중립
        else:
            merged.append(p)
    return merged if merged else [query.strip() or query]


# ---------------------------------------------------------------- 게이트

def gate_keep(seg_m, full_m, mode, param):
    """유지할 세그먼트 인덱스. 전부 탈락하면 최선 1개 폴백."""
    n = len(seg_m)
    if mode == "none":
        return list(range(n))
    if mode == "abs":
        keep = [i for i, m in enumerate(seg_m) if m >= param]
    elif mode == "rel":
        keep = [i for i, m in enumerate(seg_m) if m >= full_m - param]
    elif mode == "rank":
        order = sorted(range(n), key=lambda i: -seg_m[i])
        keep = sorted(order[:max(1, int(param))])
    else:
        raise ValueError(f"unknown gate mode {mode}")
    return keep if keep else [int(np.argmax(seg_m))]


def gate_grid(gcfg):
    out = [("none", None)]
    for m in ("abs", "rel", "rank"):
        for p in gcfg.get(m, []):
            out.append((m, float(p) if m != "rank" else int(p)))
    return out


# ---------------------------------------------------------------- 변형 채점

def quota_topk(seg_scores, full_score, tool_ids, k):
    """세그먼트별 K/n 슬롯 라운드로빈 → 부족분은 union_max 순위로 채움."""
    n = len(seg_scores)
    if n == 1:
        return topk_ids(np.maximum(full_score, seg_scores[0]), tool_ids, k)
    per = max(1, math.ceil(k / n))
    orders = [np.argsort(-s, kind="stable") for s in seg_scores]
    picked, seen = [], set()
    for r in range(per):
        for o in orders:
            if len(picked) >= k:
                break
            ti = int(o[r])
            if ti not in seen:
                seen.add(ti)
                picked.append(ti)
        if len(picked) >= k:
            break
    if len(picked) < k:
        union = np.maximum.reduce([full_score] + list(seg_scores))
        for ti in np.argsort(-union, kind="stable"):
            if len(picked) >= k:
                break
            if int(ti) not in seen:
                seen.add(int(ti))
                picked.append(int(ti))
    return [tool_ids[i] for i in picked[:k]]


def variant_score_vec(variant, full_score, kept):
    """점수 벡터 (seg_quota 는 집합 선택이라 벡터가 없어 None)."""
    if variant == "union_max":
        return np.maximum.reduce([full_score] + kept)
    if variant == "seg_only":
        return np.maximum.reduce(kept)
    return None


def variant_topk(variant, full_score, kept, tool_ids, k):
    if variant == "seg_quota":
        return quota_topk(kept, full_score, tool_ids, k)
    return topk_ids(variant_score_vec(variant, full_score, kept), tool_ids, k)


# ---------------------------------------------------------------- 누출 없는 평가

def eval_with_gate(variant, queries, tool_ids, gold_by_q, gold_idx_by_q, full_scores,
                   seg_scores_by_q, seg_m_by_q, full_m, grid, fold_of, ks, headline_k):
    """게이트 설정을 fold train 에서만 선택하고 held-out fold 로 채점."""
    n_q = len(queries)
    hit = np.zeros((n_q, len(ks)), dtype=np.int8)
    mg = np.zeros(n_q, dtype=np.int32)
    chosen, keep_by_q = [], [None] * n_q

    def keeps_for(cfg):
        gm, gp = cfg
        return [gate_keep(seg_m_by_q[i], full_m[i], gm, gp) for i in range(n_q)]

    keeps_cache = {cfg: keeps_for(cfg) for cfg in grid}
    for f in sorted(set(fold_of.values())):
        tr = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] != f]
        te = [i for i, q in enumerate(queries) if fold_of[str(q["query_id"])] == f]
        if not tr:
            continue
        best = None
        for cfg in grid:
            keeps = keeps_cache[cfg]
            h = 0
            drop = 0.0
            for i in tr:
                kept = [seg_scores_by_q[i][j] for j in keeps[i]]
                h += recall_all(variant_topk(variant, full_scores[i], kept, tool_ids,
                                             headline_k), gold_by_q[i])
                drop += 1.0 - len(keeps[i]) / len(seg_scores_by_q[i])
            key = (h / len(tr), -drop / len(tr))  # 동률이면 덜 버리는 쪽
            if best is None or key > best[0]:
                best = (key, cfg)
        cfg = best[1]
        chosen.append({"fold": int(f), "gate": cfg[0], "param": cfg[1],
                       "train_recall": round(best[0][0], 4)})
        keeps = keeps_cache[cfg]
        for i in te:
            keep_by_q[i] = keeps[i]
            kept = [seg_scores_by_q[i][j] for j in keeps[i]]
            for ki, k in enumerate(ks):
                hit[i, ki] = recall_all(variant_topk(variant, full_scores[i], kept,
                                                     tool_ids, k), gold_by_q[i])
            sv = variant_score_vec(variant, full_scores[i], kept)
            mg[i] = maxgold_rank(sv, gold_idx_by_q[i]) if sv is not None else -1
    return hit, mg, chosen, keep_by_q


# ---------------------------------------------------------------- 게이트 진단

def gate_diagnostics(cfg_name, keeps, segs, seg_m, full_m, seg_top_gold, gold_by_q):
    """버림 비율·유지/버림 m 분포·gold 손실 건수. cfg_name 은 표기용."""
    n_q = len(keeps)
    n_before = [len(s) for s in segs]
    n_after = [len(k) for k in keeps]
    m_keep, m_drop, lost = [], [], 0
    for i in range(n_q):
        ks_ = set(keeps[i])
        for j in range(len(segs[i])):
            (m_keep if j in ks_ else m_drop).append(seg_m[i][j])
        # 버려진 세그먼트만 표면화한 gold (문서 부실 tool 편향의 실측)
        kept_gold = set().union(*[seg_top_gold[i][j] for j in ks_]) if ks_ else set()
        drop_gold = set().union(*[seg_top_gold[i][j] for j in range(len(segs[i]))
                                  if j not in ks_]) if len(ks_) < len(segs[i]) else set()
        lost += len(drop_gold - kept_gold)
    return {"gate": cfg_name,
            "mean_segments_before": round(float(np.mean(n_before)), 2),
            "mean_segments_after": round(float(np.mean(n_after)), 2),
            "drop_rate": round(1.0 - sum(n_after) / max(1, sum(n_before)), 4),
            "mean_m_kept": round(float(np.mean(m_keep)), 4) if m_keep else None,
            "mean_m_dropped": round(float(np.mean(m_drop)), 4) if m_drop else None,
            "gold_only_in_dropped": int(lost)}


def m_by_position(segs, seg_m, max_pos=5):
    """위치별 평균 m — 서사 패딩이 앞/뒤에 몰리는지 구조적으로 확인."""
    acc = [[] for _ in range(max_pos)]
    last = []
    for i in range(len(segs)):
        for j in range(min(len(segs[i]), max_pos)):
            acc[j].append(seg_m[i][j])
        last.append(seg_m[i][-1])
    return {"by_position": [round(float(np.mean(a)), 4) if a else None for a in acc],
            "last_position": round(float(np.mean(last)), 4)}


# ---------------------------------------------------------------- 실행

def run(config_path, embed_fn=None):
    cfg = load_config(config_path, make_dirs=False)
    seed = int(cfg["seed"])
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    headline_k = int(cfg.get("analysis", {}).get("headline_k", 10))
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    qs = cfg.get("query_split", {})
    min_chars = int(qs.get("min_chars", 12))
    grid = gate_grid(qs.get("gate", {}))
    n_print = int(qs.get("gate", {}).get("examples_to_print", 4))
    n_folds = int(cfg["fusion"]["n_folds"])

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    idx_of = {t: i for i, t in enumerate(tool_ids)}
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    if embed_fn is None:
        from utils.embed import embed_queries
        embed_fn = lambda texts: embed_queries(texts, cfg)  # noqa: E731

    report = {"min_chars": min_chars, "gate_grid": [[m, p] for m, p in grid],
              "note": "배포 기본은 mode=sent (다국어 안전). 게이트 설정은 fold train 에서만 "
                      "선택. gold_only_in_dropped 가 크면 문서 부실 tool 편향 — abs 임계 주의.",
              "splits": {}}
    dump_path = os.path.join(results_dir, "query_split_segments.jsonl")
    os.makedirs(results_dir, exist_ok=True)
    dump = open(dump_path, "w", encoding="utf-8")

    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        gold_idx_by_q = [[idx_of[g] for g in gold if g in idx_of] for gold in gold_by_q]
        q_mat = np.load(os.path.join(emb_dir, f"queries_{split}.npy"))
        n_q = len(queries)
        fold_of = assign_folds([str(q["query_id"]) for q in queries], n_folds, seed)

        segs_by_mode = {m: [split_query(q["query"], min_chars, m) for q in queries]
                        for m in MODES}
        uniq = sorted({s for segs in segs_by_mode.values() for seg in segs for s in seg})
        vecs = np.asarray(embed_fn(uniq), dtype=np.float32)
        vec_of = {t: vecs[i] for i, t in enumerate(uniq)}
        print(f"[query-split] {split}: 고유 세그먼트 {len(uniq)}개 임베딩", flush=True)

        full_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(n_q)]
        full_m = [float(s.max()) for s in full_scores]
        base_hit = np.zeros((n_q, len(ks)), dtype=np.int8)
        base_mg = np.zeros(n_q, dtype=np.int32)
        for i in range(n_q):
            for ki, k in enumerate(ks):
                base_hit[i, ki] = recall_all(topk_ids(full_scores[i], tool_ids, k), gold_by_q[i])
            base_mg[i] = maxgold_rank(full_scores[i], gold_idx_by_q[i])
        base = {k: round(float(base_hit[:, ki].mean()), 4) for ki, k in enumerate(ks)}
        hk = ks.index(headline_k)

        entry = {"base": base, "base_maxgold": round(float(base_mg.mean()), 2), "modes": {}}
        for mode in MODES:
            segs = segs_by_mode[mode]
            seg_scores_by_q = [[cosine_scores_multi(vec_of[s], tool_vecs) for s in segs[i]]
                               for i in range(n_q)]
            seg_m_by_q = [[float(s.max()) for s in ss] for ss in seg_scores_by_q]
            # 세그먼트별로 top-headline_k 안에 든 gold (게이트 손실 측정용)
            seg_top_gold = [[set(topk_ids(s, tool_ids, headline_k)) & set(gold_by_q[i])
                             for s in seg_scores_by_q[i]] for i in range(n_q)]

            mode_entry = {
                "n_multi_segment_queries": sum(1 for s in segs if len(s) > 1),
                "mean_segments": round(float(np.mean([len(s) for s in segs])), 2),
                "max_segments": int(max(len(s) for s in segs)),
                "m_position": m_by_position(segs, seg_m_by_q),
                "gate_sweep": [], "variants": {},
            }
            # 게이트별 진단 (선택과 무관하게 전 설정 기록 — 검토용)
            for gm, gp in grid:
                keeps = [gate_keep(seg_m_by_q[i], full_m[i], gm, gp) for i in range(n_q)]
                d = gate_diagnostics(f"{gm}:{gp}", keeps, segs, seg_m_by_q, full_m,
                                     seg_top_gold, gold_by_q)
                mode_entry["gate_sweep"].append(d)

            for variant in VARIANTS:
                hit, mg, chosen, keep_by_q = eval_with_gate(
                    variant, queries, tool_ids, gold_by_q, gold_idx_by_q, full_scores,
                    seg_scores_by_q, seg_m_by_q, full_m, grid, fold_of, ks, headline_k)
                imp = int(np.sum((hit[:, hk] == 1) & (base_hit[:, hk] == 0)))
                wor = int(np.sum((hit[:, hk] == 0) & (base_hit[:, hk] == 1)))
                valid = mg > 0
                mode_entry["variants"][variant] = {
                    "recall": {k: round(float(hit[:, ki].mean()), 4) for ki, k in enumerate(ks)},
                    "delta": {k: round(float(hit[:, ki].mean()) - base[k], 4)
                              for ki, k in enumerate(ks)},
                    "improved": imp, "worsened": wor, "net": imp - wor,
                    "mcnemar_p": round(mcnemar_p(imp, wor), 4),
                    "d_maxgold_mean": (round(float((base_mg[valid] - mg[valid]).mean()), 2)
                                       if valid.any() else None),
                    "chosen_gate": chosen,
                    "mean_segments_used": round(
                        float(np.mean([len(k) for k in keep_by_q if k is not None])), 2),
                }
                if variant == "union_max" and mode == "sent":
                    _print_examples(split, queries, segs, seg_m_by_q, keep_by_q,
                                    seg_top_gold, n_print)
                    for i in range(n_q):
                        dump.write(json.dumps({
                            "split": split, "query_id": queries[i]["query_id"],
                            "query": queries[i]["query"], "gold_tools": gold_by_q[i],
                            "full_max_sim": round(full_m[i], 4),
                            "segments": [
                                {"text": segs[i][j], "max_sim": round(seg_m_by_q[i][j], 4),
                                 "kept_union_max": (keep_by_q[i] is not None
                                                    and j in keep_by_q[i]),
                                 "gold_in_topk": sorted(seg_top_gold[i][j]),
                                 "top3": topk_ids(seg_scores_by_q[i][j], tool_ids, 3)}
                                for j in range(len(segs[i]))],
                        }, ensure_ascii=False) + "\n")
            entry["modes"][mode] = mode_entry
            print(f"[query-split] {split}/{mode}: 분해 "
                  f"{mode_entry['n_multi_segment_queries']}/{n_q} "
                  f"(평균 {mode_entry['mean_segments']})", flush=True)
        report["splits"][split] = entry

    dump.close()
    out = os.path.join(results_dir, "query_split_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_report(report, splits, ks, headline_k)
    print(f"\n[query-split] 저장: {out}")
    print(f"[query-split] 세그먼트 전수 덤프: {dump_path} (mode=sent, union_max 게이트 결정 포함)")


def _print_examples(split, queries, segs, seg_m, keeps, seg_top_gold, n_print):
    """게이트가 실제로 무엇을 버리는지 원문으로 확인 — 숫자만 보고 넘기지 않기 위해."""
    print(f"\n  --- [{split}] 게이트 판정 예시 (mode=sent, union_max) ---")
    shown = 0
    for i in range(len(queries)):
        if keeps[i] is None or len(segs[i]) < 2:
            continue
        if len(keeps[i]) == len(segs[i]) and shown >= n_print // 2:
            continue  # 아무것도 안 버린 예시는 절반만
        print(f"  Q{queries[i]['query_id']}: \"{queries[i]['query'][:110]}\"")
        for j, s in enumerate(segs[i]):
            mark = "KEEP" if j in keeps[i] else "DROP"
            g = f" gold→{sorted(seg_top_gold[i][j])}" if seg_top_gold[i][j] else ""
            print(f"     [{mark}] m={seg_m[i][j]:.3f}  \"{s[:88]}\"{g}")
        shown += 1
        if shown >= n_print:
            break


def print_report(report, splits, ks, hk):
    print("\n" + "=" * 108)
    print(f"규칙 기반 분해 + 세그먼트 품질 게이트 — Recall_all@K (min_chars={report['min_chars']})")
    print("게이트: m(s)=세그먼트 단독 tool 최대 유사도. abs=절대 / rel=전체쿼리 대비 / rank=상위 r개")
    print("=" * 108)
    for split in splits:
        e = report["splits"][split]
        print(f"\n### {split}   base: " + "  ".join(f"@{k}={e['base'][k]:.2f}" for k in ks)
              + f"   maxgold {e['base_maxgold']}")
        for mode in MODES:
            md = e["modes"][mode]
            pos = md["m_position"]["by_position"]
            print(f"\n  [{mode}] 분해 {md['n_multi_segment_queries']}/100, "
                  f"평균 {md['mean_segments']} · 최대 {md['max_segments']} 세그먼트")
            print(f"    위치별 평균 m: "
                  + " ".join(f"#{i+1}={v:.3f}" for i, v in enumerate(pos) if v is not None)
                  + f" | 마지막={md['m_position']['last_position']:.3f}")
            print("    게이트 진단: " + "gate".ljust(12) + "n전→n후  버림%  m(유지)  m(버림)  gold손실")
            for d in md["gate_sweep"]:
                print(f"                 {d['gate']:<12}"
                      f"{d['mean_segments_before']:.2f}→{d['mean_segments_after']:.2f}"
                      f"  {d['drop_rate']*100:5.1f}%"
                      f"  {(d['mean_m_kept'] or 0):.4f}"
                      f"  {(d['mean_m_dropped'] if d['mean_m_dropped'] is not None else float('nan')):.4f}"
                      f"    {d['gold_only_in_dropped']:>3}")
            print("    " + "variant".ljust(11) + "".join(f"@{k}".rjust(7) for k in ks)
                  + "".join(f"Δ@{k}".rjust(7) for k in ks)
                  + f"  개선/악화 순증    p  Δmaxgold  n사용  선택된 게이트")
            for vn, v in md["variants"].items():
                sel = {}
                for c in v["chosen_gate"]:
                    key = f"{c['gate']}:{c['param']}"
                    sel[key] = sel.get(key, 0) + 1
                selstr = ",".join(f"{k}×{n}" for k, n in sorted(sel.items(), key=lambda x: -x[1]))
                dm = v["d_maxgold_mean"]
                print("    " + vn.ljust(11)
                      + "".join(f"{v['recall'][k]:.2f}".rjust(7) for k in ks)
                      + "".join(f"{v['delta'][k]:+.2f}".rjust(7) for k in ks)
                      + f"   {v['improved']:>3}/{v['worsened']:<3}{v['net']:+4d}"
                      f" {v['mcnemar_p']:>5.2f}"
                      + (f"  {dm:+7.2f}" if dm is not None else "        -")
                      + f"  {v['mean_segments_used']:>5.2f}  {selstr}")
    print("\n" + "=" * 108)
    print("검토 순서: ① 위치별 m — 앞/뒤가 낮으면 서사 패딩 가설 확인 (구조적 근거)")
    print("          ② 게이트 진단의 m(유지) vs m(버림) 격차 — 작으면 신호 자체가 약함")
    print("          ③ gold손실 — 크면 문서 부실 tool 을 버리는 편향 (abs 임계 특히 주의)")
    print("          ④ 예시 원문 KEEP/DROP — 실제로 패딩을 걷어내는지 눈으로 확인")
    print("          ⑤ 그다음에 recall Δ. 게이트 후 seg_quota 가 I3 저K 에서 뒤집히는지가 핵심")
    print("=" * 108)


# ---------------------------------------------------------------- smoke

def _smoke():
    print("[smoke] query split + 게이트 로직")
    # 분해: 영어 문장, 소수점 보호, 파편 병합, 전각(CJK), 접속사 모드
    s = split_query("Find breweries in NY that offer tours. Also, analyze the nutrition "
                    "of a recipe with chicken, broccoli and quinoa.", 12)
    assert len(s) == 2 and "breweries" in s[0] and "nutrition" in s[1], s
    assert split_query("What's the weather in Paris?", 12) == ["What's the weather in Paris?"]
    assert len(split_query("Convert 3.5 kg to pounds.", 12)) == 1
    assert len(split_query("Check the weather. OK?", 12)) == 1
    assert len(split_query("서울 날씨를 알려줘。그리고 환율도 확인해줘。", 12)) == 2
    assert len(split_query("東京の天気を教えて。それから為替も確認して。", 12)) == 2
    assert len(split_query("날씨。응?", 12)) == 1
    q = "Get the top tracks of my favorite artist and then find news about decoration ideas."
    assert len(split_query(q, 12, "sent")) == 1 and len(split_query(q, 12, "sent_conj")) == 2

    # 게이트: 패딩(낮은 m) 이 버려지고 요청(높은 m) 이 남는가
    seg_m, full_m = [0.72, 0.90, 0.68], 0.91
    assert gate_keep(seg_m, full_m, "none", None) == [0, 1, 2]
    assert gate_keep(seg_m, full_m, "abs", 0.85) == [1]
    assert gate_keep(seg_m, full_m, "rel", 0.05) == [1]
    assert gate_keep(seg_m, full_m, "rel", 0.25) == [0, 1, 2]
    assert gate_keep(seg_m, full_m, "rank", 2) == [0, 1]        # 인덱스 순 정렬 유지
    assert gate_keep(seg_m, full_m, "rank", 9) == [0, 1, 2]     # r > n 이면 전체
    assert gate_keep([0.5, 0.6], 0.99, "abs", 0.95) == [1]      # 전부 탈락 → 최선 폴백
    assert ("none", None) in gate_grid({"abs": [0.8], "rank": [2]})

    # 희석 해소 + 슬롯 배분
    rng = np.random.default_rng(0)
    d, n_tools = 8, 6

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    tool_vecs = norm(rng.standard_normal((n_tools, 6, d))).astype(np.float32)
    a, b = tool_vecs[0, 0], tool_vecs[1, 0]
    sa, sb = cosine_scores_multi(a, tool_vecs), cosine_scores_multi(b, tool_vecs)
    sc_full = cosine_scores_multi(norm((a + b) / 2), tool_vecs)
    assert np.maximum(sa, sb)[0] > 0.99 and np.maximum(sa, sb)[1] > 0.99
    ids = [f"T{i}" for i in range(n_tools)]
    assert set(quota_topk([sa, sb], sc_full, ids, 2)) == {"T0", "T1"}
    assert quota_topk([sa], sc_full, ids, 1) == ["T0"]
    g4 = quota_topk([sa, sb], sc_full, ids, 4)
    assert len(g4) == 4 and len(set(g4)) == 4

    # 변형 채점기
    assert variant_score_vec("seg_quota", sc_full, [sa]) is None
    assert np.allclose(variant_score_vec("seg_only", sc_full, [sa, sb]), np.maximum(sa, sb))

    # 게이트 진단: 버려진 세그먼트만 gold 를 표면화하면 손실로 집계
    segs = [["pad", "req"]]
    dd = gate_diagnostics("t", [[1]], segs, [[0.7, 0.9]], [0.9],
                          [[{"G1"}, set()]], [["G1"]])
    assert dd["gold_only_in_dropped"] == 1 and dd["drop_rate"] == 0.5, dd
    dd2 = gate_diagnostics("t", [[0, 1]], segs, [[0.7, 0.9]], [0.9],
                           [[{"G1"}, set()]], [["G1"]])
    assert dd2["gold_only_in_dropped"] == 0 and dd2["drop_rate"] == 0.0, dd2

    mp = m_by_position([["a", "b", "c"]], [[0.6, 0.9, 0.5]])
    assert mp["by_position"][0] == 0.6 and mp["last_position"] == 0.5, mp
    print("[smoke] OK — 분해·게이트 3형태·폴백·슬롯 배분·gold 손실 집계·위치별 m 정상")


def main():
    ap = argparse.ArgumentParser(
        description="규칙 기반 멀티 인텐트 분해 + 세그먼트 품질 게이트 (LLM 0)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
