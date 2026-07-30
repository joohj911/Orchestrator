"""report_retrieval.py — retrieval 층 단독 리포트 (LLM/GPU 불필요).

M3(+ --real-prior)가 생성한 retrieval_{split}_{method}_{K}.jsonl 을 전부 집계해
Recall_all@K 표를 출력하고 results/retrieval_recall.csv 로 저장한다.

Recall_all = "gold tool 전부가 top-K 후보에 포함된 쿼리 비율" (포함 기준, 호출 아님 —
downstream 호출 정확도는 m5/m6 의 completeness/exact_match 가 담당).

CLI: python scripts/report_retrieval.py [--config config.yaml]
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.config import load_config  # noqa: E402

FNAME_RE = re.compile(r"retrieval_(?P<split>I\d)_(?P<method>.+)_(?P<k>\d+)\.jsonl$")

# 표 행 순서 (ablation 사다리): 어휘→단일벡터→멀티벡터→융합(oracle 상한/real 배포치)
METHOD_ORDER = ["bm25", "dense_single", "dense_multi",
                "fusion_add_oracle", "fusion_mult_oracle",
                "fusion_add_real", "fusion_mult_real"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Recall_all@K 집계 (retrieval 층 단독)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, make_dirs=False)
    results_dir = cfg["paths"]["results_dir"]

    agg = {}  # (split, method) -> {k: (recall, n)}
    for p in glob.glob(os.path.join(results_dir, "retrieval_*.jsonl")):
        m = FNAME_RE.search(os.path.basename(p))
        if not m:
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        rec = round(sum(r["recall_all"] for r in rows) / len(rows), 4)
        agg.setdefault((m["split"], m["method"]), {})[int(m["k"])] = (rec, len(rows))

    if not agg:
        sys.exit(f"[report] {results_dir} 에 retrieval_*.jsonl 없음 (m3 먼저)")

    splits = sorted({s for s, _ in agg})
    ks = sorted({k for v in agg.values() for k in v})
    methods = [m for m in METHOD_ORDER if any((s, m) in agg for s in splits)]
    methods += sorted({m for _, m in agg} - set(methods))  # 목록 밖 방법도 표시

    csv_rows = []
    for split in splits:
        n = next((v[k][1] for (s, m2), v in agg.items() if s == split for k in v), "?")
        print(f"\n=== {split} : Recall_all@K (gold 전부 top-K 포함 비율, n={n}) ===")
        print("method".ljust(22), *[f"K={k}".rjust(8) for k in ks])
        for meth in methods:
            v = agg.get((split, meth), {})
            print(meth.ljust(22), *[f"{v[k][0]:.2f}".rjust(8) if k in v else "-".rjust(8) for k in ks])
            for k in v:
                csv_rows.append({"split": split, "method": meth, "k": k,
                                 "recall_all": v[k][0], "n": v[k][1]})

    out = os.path.join(results_dir, "retrieval_recall.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("split,method,k,recall_all,n\n")
        for r in sorted(csv_rows, key=lambda r: (r["split"], r["method"], r["k"])):
            f.write(f"{r['split']},{r['method']},{r['k']},{r['recall_all']},{r['n']}\n")
    print(f"\n[report] 저장: {out}")
    print("[report] 주의: *_oracle 은 gold category prior 를 쓴 진단용 상한 — 배포치는 *_real.")


if __name__ == "__main__":
    main()
