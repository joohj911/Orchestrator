"""experiment_query_split.py — 규칙 기반 멀티 인텐트 분해 retrieval A/B (LLM 0).

배경: 남은 진단된 병목 = I3 멀티 인텐트의 유사도 희석 (요청 2~3개가 한 임베딩에 섞여
각 sub-intent 의 gold 매칭이 약해짐). ToolQP 계열의 분해 아이디어를 차용하되,
latency 제약(서빙 LLM 호출 금지) 하에서 **규칙 기반 분해**로 검증한다 —
ToolBench 복합 쿼리는 대부분 문장 경계·접속사에서 갈라진다.

메커니즘 (서빙 비용: 짧은 세그먼트 2~4개 임베딩 배치 1회 추가):
  쿼리 → 문장부호/접속사 규칙 분해 → 세그먼트별 e5 임베딩
      → tool 점수 = max(전체 쿼리 점수, 세그먼트별 점수)   [union_max]
  비교 변형: seg_only (전체 쿼리 제외, 세그먼트만) — 전체 쿼리 벡터의 기여 분리 측정.

판정: I3 저K recall (dense_multi 0.31@5 / 0.47@10) 개선 + I1 무손실
(I1 은 단일 인텐트라 분해가 거의 발생하지 않아야 정상 — 과분해 진단 겸용).

CLI: python scripts/experiment_query_split.py --config config.yaml [--smoke]
GPU: 세그먼트 임베딩만 (수백 문장, 수십 초). 산출물: results/query_split_ab.json
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import json
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


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# 문장 경계 + 명시적 병렬 접속 표현. 보수적으로 — 과분해보다 미분해가 안전.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONJ_SPLIT = re.compile(
    r",?\s+(?:and also|also,?|additionally,?|then|as well as|and then|plus,?)\s+",
    re.IGNORECASE)


def split_query(query: str, min_words: int) -> list[str]:
    """규칙 기반 분해. 너무 짧은 조각은 앞 조각에 도로 붙인다 (파편화 방지)."""
    parts = []
    for sent in _SENT_SPLIT.split(query.strip()):
        for seg in _CONJ_SPLIT.split(sent):
            seg = seg.strip(" ,;")
            if seg:
                parts.append(seg)
    merged: list[str] = []
    for p in parts:
        if merged and len(p.split()) < min_words:
            merged[-1] = merged[-1] + ", " + p
        else:
            merged.append(p)
    return merged if merged else [query]


def recall_table(score_fn, queries, tool_ids, gold_by_q, ks):
    out = {k: 0 for k in ks}
    for qi in range(len(queries)):
        sc = score_fn(qi)
        for k in ks:
            out[k] += recall_all(topk_ids(sc, tool_ids, k), gold_by_q[qi])
    return {k: round(v / max(1, len(queries)), 4) for k, v in out.items()}


def run(config_path, embed_fn=None):
    cfg = load_config(config_path, make_dirs=False)
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    min_words = int(cfg.get("query_split", {}).get("min_words", 4))

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    if embed_fn is None:
        from utils.embed import embed_queries
        embed_fn = lambda texts: embed_queries(texts, cfg)  # noqa: E731

    report = {"min_words": min_words, "splits": {}}
    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        q_mat = np.load(os.path.join(emb_dir, f"queries_{split}.npy"))

        segs_by_q = [split_query(q["query"], min_words) for q in queries]
        n_segs = [len(s) for s in segs_by_q]
        flat_segs = [s for segs in segs_by_q for s in segs]
        seg_embs = np.asarray(embed_fn(flat_segs), dtype=np.float32)
        offsets = np.cumsum([0] + n_segs)

        full_scores = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        seg_scores = []
        for i in range(len(queries)):
            ss = [cosine_scores_multi(seg_embs[j], tool_vecs)
                  for j in range(offsets[i], offsets[i + 1])]
            seg_scores.append(np.max(np.stack(ss), axis=0))

        base = recall_table(lambda qi: full_scores[qi], queries, tool_ids, gold_by_q, ks)
        union = recall_table(lambda qi: np.maximum(full_scores[qi], seg_scores[qi]),
                             queries, tool_ids, gold_by_q, ks)
        seg_only = recall_table(lambda qi: seg_scores[qi], queries, tool_ids, gold_by_q, ks)

        multi = sum(1 for n in n_segs if n > 1)
        report["splits"][split] = {
            "n_multi_segment_queries": multi,
            "mean_segments": round(float(np.mean(n_segs)), 2),
            "base": base, "union_max": union, "seg_only": seg_only,
            "delta_union": {k: round(union[k] - base[k], 4) for k in ks},
        }
        print(f"[query-split] {split}: 분해된 쿼리 {multi}/{len(queries)} "
              f"(평균 세그먼트 {report['splits'][split]['mean_segments']})", flush=True)

    out = os.path.join(results_dir, "query_split_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for split in splits:
        e = report["splits"][split]
        print(f"\n=== {split} : Recall_all@K — 규칙 기반 쿼리 분해 A/B ===")
        print(f"(분해 {e['n_multi_segment_queries']}/100 쿼리, 평균 {e['mean_segments']} 세그먼트)")
        print("variant".ljust(20), *[f"K={k}".rjust(7) for k in ks])
        for name in ("base", "union_max", "seg_only"):
            print(name.ljust(20), *[f"{e[name][k]:.2f}".rjust(7) for k in ks])
        print("delta(union)".ljust(20), *[f"{e['delta_union'][k]:+.2f}".rjust(7) for k in ks])
    print(f"\n[query-split] 저장: {out}")
    print("[query-split] 판정: I3 저K(5·10) delta + / I1 무손실이면 채택 → "
          "세그먼트별 top-k 배분·adaptive K 와 결합 검토. 분해율이 낮으면 규칙 보강 먼저.")


def _smoke():
    print("[smoke] query split 로직")
    s = split_query("Find breweries in NY that offer tours. Also, analyze the nutrition "
                    "of a recipe with chicken, broccoli and quinoa.", 4)
    assert len(s) == 2 and "breweries" in s[0] and "nutrition" in s[1], s
    s = split_query("What's the weather in Paris?", 4)
    assert s == ["What's the weather in Paris?"], s  # 단일 인텐트는 분해 없음
    s = split_query("Get the top tracks of my favorite artist and then find news "
                    "articles about party decoration ideas.", 4)
    assert len(s) == 2, s
    s = split_query("Check the weather. OK?", 4)
    assert len(s) == 1, s  # 짧은 조각은 병합

    rng = np.random.default_rng(0)
    d, n_tools = 8, 6

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    tool_vecs = norm(rng.standard_normal((n_tools, 6, d))).astype(np.float32)
    # 두 intent 벡터 a,b — tool0 은 a 에, tool1 은 b 에 강함. 희석 쿼리 = (a+b)/2
    a, b = tool_vecs[0, 0], tool_vecs[1, 0]
    diluted = norm((a + b) / 2)
    sc_full = cosine_scores_multi(diluted, tool_vecs)
    sc_seg = np.maximum(cosine_scores_multi(a, tool_vecs), cosine_scores_multi(b, tool_vecs))
    assert sc_seg[0] >= sc_full[0] and sc_seg[1] >= sc_full[1]
    assert sc_seg[0] > 0.99 and sc_seg[1] > 0.99  # 세그먼트가 자기 gold 를 만점 매칭
    print("[smoke] OK — 분해 규칙·병합·희석 해소 경로 정상")


def main():
    ap = argparse.ArgumentParser(description="규칙 기반 멀티 인텐트 분해 A/B (LLM 0, GPU: 세그먼트 임베딩)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
