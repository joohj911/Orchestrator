"""experiment_query_split.py — 규칙 기반 멀티 인텐트 분해 retrieval A/B (LLM 0).

배경: I3 의 병목은 멀티 인텐트 희석이다 — dense_multi 가 @50 은 0.87 인데 @10 은 0.47.
정답은 찾아지는데 저K 에서 밀린다. 요청 2~3개가 한 임베딩에 섞여 각 sub-intent 의 gold
매칭이 약해지는 것이 원인. ToolQP(arXiv 2601.07782) 의 분해 아이디어를 차용하되,
서빙 LLM 금지 제약 하에서 **규칙 기반 분해**로 검증한다.

다국어 대비 설계 (사람 결정): 언어별 단어 구분에 의존하지 않는다.
  - 분리 기준은 **문장 종결 부호만** — ASCII [.!?] + 전각 [。！？] (전각은 뒤 공백 불필요).
    "3.5" 같은 소수점은 ASCII 쪽이 뒤 공백을 요구하므로 안전.
  - 파편 병합은 **단어 수가 아니라 문자 수** (CJK 는 공백 분리가 없어 단어 수가 정의 불가).
  - 영어 접속사 분리는 언어 의존적이므로 **배포 기본에서 제외**하되, 별도 변형으로 함께
    측정한다 — "문장 분리만으로 충분한가 vs 접속사 처리가 필요한가"의 갭을 실측해야
    다국어 확장 시 언어별 규칙에 투자할지 판단할 수 있다.

메커니즘 (서빙 비용: 짧은 세그먼트 2~4개 임베딩 배치 1회 추가, LLM 0):
  base       = 전체 쿼리 1벡터 (dense_multi 앵커)
  union_max  = max(전체 쿼리 점수, 세그먼트별 점수)
  seg_only   = 세그먼트만 (전체 쿼리 벡터의 기여 분리 측정)
  seg_quota  = 세그먼트별로 K/n 슬롯 배분 후 순위 라운드로빈 (슬롯 배분 축의 최소 검정.
               Recall_all 은 집합 지표인데 기존 방식은 tool 을 독립 채점하고 상위만 자른다)

판정: I3 저K(5·10) 개선 + I1 무손실 (I1 은 단일 인텐트라 분해가 거의 없어야 정상 —
분해율이 과분해 진단을 겸한다). 계측은 recall 뿐 아니라 쿼리 단위 부호 분해와
maxgold rank 이동까지 — 이진 비트만 보면 사각지대를 놓친다 (prior 진단에서 확인된 교훈).

CLI: python scripts/experiment_query_split.py --config config.yaml [--smoke]
GPU: 세그먼트 임베딩만 (수백 문장). 산출물: results/query_split_ab.json
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
from m3_retrieval import cosine_scores_multi, topk_ids  # noqa: E402
from diagnose_prior_effect import maxgold_rank, mcnemar_p  # noqa: E402

# 문장 종결 부호만. 전각은 뒤 공백이 없어도 분리(CJK), ASCII 는 뒤 공백을 요구(소수점 보호).
_SENT_SPLIT = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")
# 영어 전용 — 배포 기본에서 제외, 변형으로만 측정.
_CONJ_SPLIT = re.compile(
    r",?\s+(?:and also|also,?|additionally,?|then|as well as|and then|plus,?)\s+",
    re.IGNORECASE)

MODES = ("sent", "sent_conj")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_query(query: str, min_chars: int, mode: str = "sent") -> list[str]:
    """규칙 기반 분해. 짧은 조각은 앞 조각에 도로 붙인다 (과분해 자동 교정).

    mode="sent"      : 문장 종결 부호만 (다국어 안전, 배포 기본)
    mode="sent_conj" : 문장 부호 + 영어 접속사 (언어 의존 — 필요성 측정용)
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


def run(config_path, embed_fn=None):
    cfg = load_config(config_path, make_dirs=False)
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    headline_k = int(cfg.get("analysis", {}).get("headline_k", 10))
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    min_chars = int(cfg.get("query_split", {}).get("min_chars", 12))

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    idx_of = {t: i for i, t in enumerate(tool_ids)}
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    if embed_fn is None:
        from utils.embed import embed_queries
        embed_fn = lambda texts: embed_queries(texts, cfg)  # noqa: E731

    report = {"min_chars": min_chars, "modes": list(MODES),
              "note": "배포 기본은 sent (다국어 안전). sent_conj 는 영어 전용 — 갭이 크면 "
                      "언어별 접속사 규칙에 투자할 근거가 된다.",
              "splits": {}}

    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        gold_idx_by_q = [[idx_of[g] for g in gold if g in idx_of] for gold in gold_by_q]
        q_mat = np.load(os.path.join(emb_dir, f"queries_{split}.npy"))
        n_q = len(queries)

        segs_by_mode = {m: [split_query(q["query"], min_chars, m) for q in queries]
                        for m in MODES}
        uniq = sorted({s for segs in segs_by_mode.values() for seg in segs for s in seg})
        vecs = np.asarray(embed_fn(uniq), dtype=np.float32)
        vec_of = {t: vecs[i] for i, t in enumerate(uniq)}
        print(f"[query-split] {split}: 고유 세그먼트 {len(uniq)}개 임베딩 완료", flush=True)

        full_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(n_q)]
        base_hit = np.zeros((n_q, len(ks)), dtype=np.int8)
        base_mg = np.zeros(n_q, dtype=np.int32)
        for i in range(n_q):
            for ki, k in enumerate(ks):
                base_hit[i, ki] = recall_all(topk_ids(full_scores[i], tool_ids, k), gold_by_q[i])
            base_mg[i] = maxgold_rank(full_scores[i], gold_idx_by_q[i])
        base = {k: round(float(base_hit[:, ki].mean()), 4) for ki, k in enumerate(ks)}

        entry = {"base": base, "base_maxgold": round(float(base_mg.mean()), 2), "modes": {}}
        for m in MODES:
            segs = segs_by_mode[m]
            n_segs = [len(s) for s in segs]
            seg_score_by_q = [[cosine_scores_multi(vec_of[s], tool_vecs) for s in segs[i]]
                              for i in range(n_q)]
            seg_max = [np.max(np.stack(ss), axis=0) for ss in seg_score_by_q]

            variants = {}
            for vname in ("union_max", "seg_only", "seg_quota"):
                hit = np.zeros((n_q, len(ks)), dtype=np.int8)
                mg = np.zeros(n_q, dtype=np.int32)
                for i in range(n_q):
                    if vname == "seg_quota":
                        for ki, k in enumerate(ks):
                            hit[i, ki] = recall_all(
                                quota_topk(seg_score_by_q[i], full_scores[i], tool_ids, k),
                                gold_by_q[i])
                        mg[i] = base_mg[i]  # 슬롯 배분은 점수 랭킹이 아니라 집합 — 순위 비교 제외
                    else:
                        sc = (np.maximum(full_scores[i], seg_max[i]) if vname == "union_max"
                              else seg_max[i])
                        for ki, k in enumerate(ks):
                            hit[i, ki] = recall_all(topk_ids(sc, tool_ids, k), gold_by_q[i])
                        mg[i] = maxgold_rank(sc, gold_idx_by_q[i])
                hk = ks.index(headline_k)
                imp = int(np.sum((hit[:, hk] == 1) & (base_hit[:, hk] == 0)))
                wor = int(np.sum((hit[:, hk] == 0) & (base_hit[:, hk] == 1)))
                variants[vname] = {
                    "recall": {k: round(float(hit[:, ki].mean()), 4) for ki, k in enumerate(ks)},
                    "delta": {k: round(float(hit[:, ki].mean()) - base[k], 4)
                              for ki, k in enumerate(ks)},
                    "improved": imp, "worsened": wor, "net": imp - wor,
                    "mcnemar_p": round(mcnemar_p(imp, wor), 4),
                    "maxgold_mean": round(float(mg.mean()), 2),
                    "d_maxgold_mean": round(float((base_mg - mg).mean()), 2),
                }
            entry["modes"][m] = {
                "n_multi_segment_queries": sum(1 for n in n_segs if n > 1),
                "mean_segments": round(float(np.mean(n_segs)), 2),
                "max_segments": int(max(n_segs)),
                "variants": variants,
            }
            print(f"[query-split] {split}/{m}: 분해 {entry['modes'][m]['n_multi_segment_queries']}"
                  f"/{n_q} (평균 {entry['modes'][m]['mean_segments']})", flush=True)
        report["splits"][split] = entry

    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "query_split_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_report(report, splits, ks, headline_k)
    print(f"\n[query-split] 저장: {out}")


def print_report(report, splits, ks, hk):
    print("\n" + "=" * 100)
    print(f"규칙 기반 멀티 인텐트 분해 — Recall_all@K (min_chars={report['min_chars']})")
    print("sent = 문장 종결 부호만 (다국어 안전, 배포 기본) | sent_conj = + 영어 접속사")
    print("=" * 100)
    for split in splits:
        e = report["splits"][split]
        print(f"\n### {split}   base: "
              + "  ".join(f"@{k}={e['base'][k]:.2f}" for k in ks)
              + f"   maxgold {e['base_maxgold']}")
        for m in MODES:
            md = e["modes"][m]
            print(f"  [{m}] 분해 {md['n_multi_segment_queries']}/100 쿼리, "
                  f"평균 {md['mean_segments']} · 최대 {md['max_segments']} 세그먼트")
            print("    " + "variant".ljust(12) + "".join(f"@{k}".rjust(8) for k in ks)
                  + "".join(f"Δ@{k}".rjust(8) for k in ks)
                  + f"  @{hk} 개선/악화 순증    p   Δmaxgold")
            for vn, v in md["variants"].items():
                print("    " + vn.ljust(12)
                      + "".join(f"{v['recall'][k]:.2f}".rjust(8) for k in ks)
                      + "".join(f"{v['delta'][k]:+.2f}".rjust(8) for k in ks)
                      + f"    {v['improved']:>3}/{v['worsened']:<3} {v['net']:+4d}"
                      f" {v['mcnemar_p']:>5.2f}   {v['d_maxgold_mean']:+6.2f}")
    print("\n" + "=" * 100)
    print("판정: I3 저K(5·10) Δ 가 + 이고 I1 무손실이면 채택 (배포는 sent 변형).")
    print("      sent 와 sent_conj 의 갭 = 언어별 접속사 규칙에 투자할 가치.")
    print("      seg_quota 가 union_max 를 넘으면 슬롯 배분 축이 유효 — 집합 단위로 확장 검토.")
    print("      분해율이 I2·I3 에서 낮으면 규칙 자체가 부족한 것 (개선 전에 규칙 보강).")
    print("=" * 100)


def _smoke():
    print("[smoke] query split 로직")
    # 문장 분리 (영어)
    s = split_query("Find breweries in NY that offer tours. Also, analyze the nutrition "
                    "of a recipe with chicken, broccoli and quinoa.", 12)
    assert len(s) == 2 and "breweries" in s[0] and "nutrition" in s[1], s
    # 단일 인텐트는 분해 없음
    assert split_query("What's the weather in Paris?", 12) == ["What's the weather in Paris?"]
    # 소수점 보호 (뒤 공백 요구)
    assert len(split_query("Convert 3.5 kg to pounds.", 12)) == 1
    # 짧은 파편은 병합 (문자 수 기준)
    assert len(split_query("Check the weather. OK?", 12)) == 1
    # 전각 부호는 뒤 공백 없이 분리 (CJK)
    s = split_query("서울 날씨를 알려줘。그리고 환율도 확인해줘。", 12)
    assert len(s) == 2, s
    s = split_query("東京の天気を教えて。それから為替も確認して。", 12)
    assert len(s) == 2, s
    # 단어 수가 아니라 문자 수 — 공백 없는 언어에서도 병합 판단이 성립
    assert len(split_query("날씨。응?", 12)) == 1

    # 접속사 모드는 한 문장 안을 더 쪼갠다
    q = "Get the top tracks of my favorite artist and then find news about decoration ideas."
    assert len(split_query(q, 12, "sent")) == 1
    assert len(split_query(q, 12, "sent_conj")) == 2

    rng = np.random.default_rng(0)
    d, n_tools = 8, 6

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    tool_vecs = norm(rng.standard_normal((n_tools, 6, d))).astype(np.float32)
    a, b = tool_vecs[0, 0], tool_vecs[1, 0]
    diluted = norm((a + b) / 2)
    sc_full = cosine_scores_multi(diluted, tool_vecs)
    sa, sb = cosine_scores_multi(a, tool_vecs), cosine_scores_multi(b, tool_vecs)
    sc_seg = np.maximum(sa, sb)
    assert sc_seg[0] >= sc_full[0] and sc_seg[1] >= sc_full[1]
    assert sc_seg[0] > 0.99 and sc_seg[1] > 0.99  # 세그먼트가 자기 gold 를 만점 매칭

    # quota: 세그먼트당 슬롯을 나눠 두 gold 를 모두 확보
    ids = [f"T{i}" for i in range(n_tools)]
    got = quota_topk([sa, sb], sc_full, ids, 2)
    assert set(got) == {"T0", "T1"}, got
    assert quota_topk([sa], sc_full, ids, 1) == ["T0"]  # n=1 은 union 과 동일
    got4 = quota_topk([sa, sb], sc_full, ids, 4)
    assert len(got4) == 4 and len(set(got4)) == 4, got4  # 중복 없이 K개 채움
    print("[smoke] OK — 문장/전각 분리·문자 수 병합·접속사 모드·희석 해소·슬롯 배분 정상")


def main():
    ap = argparse.ArgumentParser(
        description="규칙 기반 멀티 인텐트 분해 A/B (LLM 0, GPU: 세그먼트 임베딩)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
