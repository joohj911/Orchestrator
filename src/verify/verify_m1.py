"""verify_m1.py — GATE: M1 데이터 무결성

검사 항목 (모두 통과해야 exit 0, 하나라도 실패 시 exit 1):
  전 쿼리 gold_tools ⊆ tools.jsonl(누락 0), description 결측 0,
  각 split 정확히 n_queries, multi-tool split gold≥2, category 분포 로그.

CLI: python verify_m1.py --config config.yaml
동작:
  - 산출물을 읽어 위 항목을 검사.
  - 실패 시 무엇이 왜 실패했는지 stderr에 명확히 출력하고 sys.exit(1).
  - 통과 시 요약을 stdout에 출력하고 sys.exit(0).
  - run_all.sh가 exit code로 파이프라인 중단을 판단하므로 exit code 정확히.
구현: Claude Code. 상세 spec/MILESTONES.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402

# m1 과 동일한 벤치마크 사실 (single-tool split).
_SINGLE_TOOL_SPLITS = {"I1"}


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fail(msg: str) -> None:
    print(f"[verify_m1] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, make_dirs=False)
    exp = cfg["experiment"]
    splits = exp["splits"]
    n_expected = exp["n_queries_per_split"]
    data_dir = cfg["paths"]["data_dir"]

    tools_path = os.path.join(data_dir, "tools.jsonl")
    errors: list[str] = []

    # --- tools.jsonl ---
    if not os.path.isfile(tools_path):
        _fail(f"tools.jsonl 없음: {tools_path}")
        sys.exit(1)
    tools = _read_jsonl(tools_path)
    tool_ids = {t["id"] for t in tools}

    # description 결측 0.
    missing_desc = [t["id"] for t in tools if not str(t.get("description", "")).strip()]
    if missing_desc:
        errors.append(
            f"description 결측 {len(missing_desc)}개 (예: {missing_desc[:3]}). 결측 0이어야 함."
        )

    # id 중복 검사 (pool 무결성).
    if len(tool_ids) != len(tools):
        dup = [i for i, c in Counter(t["id"] for t in tools).items() if c > 1]
        errors.append(f"tools.jsonl id 중복 {len(dup)}개 (예: {dup[:3]}).")

    # --- 각 split ---
    split_summ = {}
    for split in splits:
        qpath = os.path.join(data_dir, f"queries_{split}.jsonl")
        if not os.path.isfile(qpath):
            errors.append(f"[{split}] queries 파일 없음: {qpath}")
            continue
        queries = _read_jsonl(qpath)

        # 정확히 n_expected 개.
        if len(queries) != n_expected:
            errors.append(f"[{split}] 쿼리 수 {len(queries)} ≠ 요구 {n_expected}.")

        # gold_tools ⊆ tools.jsonl (누락 0) — 게이트 핵심.
        missing_gold = set()
        for q in queries:
            for g in q.get("gold_tools", []):
                if g not in tool_ids:
                    missing_gold.add(g)
        if missing_gold:
            errors.append(
                f"[{split}] gold_tools 중 pool 누락 {len(missing_gold)}개 "
                f"(예: {list(missing_gold)[:3]}). Recall_all 상한 붕괴 → 실험 무효."
            )

        # multi-tool split gold ≥ 2.
        if split not in _SINGLE_TOOL_SPLITS:
            bad = [q["query_id"] for q in queries if len(set(q.get("gold_tools", []))) < 2]
            if bad:
                errors.append(
                    f"[{split}] multi-tool 인데 gold<2 인 쿼리 {len(bad)}개 (예: {bad[:3]})."
                )

        # gold_categories 일관성(비어있지 않음).
        empty_cat = [q["query_id"] for q in queries if not q.get("gold_categories")]
        if empty_cat:
            errors.append(f"[{split}] gold_categories 비어있는 쿼리 {len(empty_cat)}개.")

        split_summ[split] = {
            "n": len(queries),
            "cat_dist": Counter(c for q in queries for c in q.get("gold_categories", [])),
        }

    # --- category 분포 로그 (통과/실패 무관 출력) ---
    print("=== M1 category 분포 ===")
    pool_dist = Counter(t["category"] for t in tools)
    print(f"[pool] {len(tools)} tools, {len(pool_dist)} categories. 상위 10:")
    for cat, n in pool_dist.most_common(10):
        print(f"   {cat}: {n}")
    for split, s in split_summ.items():
        print(f"[{split}] {s['n']} queries, gold category 상위 8: {s['cat_dist'].most_common(8)}")

    # --- 판정 ---
    if errors:
        print("\n[verify_m1] 게이트 실패:", file=sys.stderr)
        for e in errors:
            _fail(e)
        print(
            "\n재현: python src/m1_data_prep.py --config <config> --force 후 재검증.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[verify_m1] PASS — gold 누락 0, description 결측 0, split 크기·multi-tool 조건 충족.")
    sys.exit(0)


if __name__ == "__main__":
    main()
