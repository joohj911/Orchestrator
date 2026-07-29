"""verify_m6.py — GATE: M6 전체 실행 + 분석

검사 항목 (spec/MILESTONES.md — 모두 통과해야 exit 0):
  - summary.csv 존재 + 전 조합 행 존재 (누락 조합은 빈칸+reason — 행 자체가 없으면 FAIL)
  - completeness 원인 분리(retrieval_miss/generation_miss)가 집계돼 있음
  - 표1·2·3, 그림1, conclusions.md 생성 + 결론 5항목 (a)~(e) 존재
  - arg_acc 전부 N/A 확인 (gold 인자 없음 — 값이 있으면 출처 의심)
  - oracle 을 배포치로 제시 금지 → conclusions 에 '상한' 명기 확인

CLI: python verify_m6.py --config config.yaml
구현: Claude Code. 실패 시 사유를 stderr 에 출력하고 exit 1 (run_all.sh 중단 근거).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402
from m6_analysis import expected_combos  # noqa: E402


def _fail(msg):
    print(f"[verify_m6] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, make_dirs=False)
    results_dir = cfg["paths"]["results_dir"]
    splits = cfg["experiment"]["splits"]
    k_sweep = [int(k) for k in cfg["experiment"]["k_sweep"]]

    excl_path = os.path.join(results_dir, "pilot_exclusions.json")
    if not os.path.isfile(excl_path):
        _fail("pilot_exclusions.json 없음 (verify_m5 먼저)")
        sys.exit(1)
    excl = json.load(open(excl_path, encoding="utf-8"))
    models = [mk for mk in ("weak", "strong") if mk not in excl.get("excluded_models", [])]

    # 산출물 존재 (없으면 즉시 실패 — 이후 검사가 의미 없음)
    missing_files = [fn for fn in
                     ("summary.csv", "table1.csv", "table2.csv", "table3.csv", "fig1.png", "conclusions.md")
                     if not os.path.isfile(os.path.join(results_dir, fn))]
    if missing_files:
        for fn in missing_files:
            _fail(f"산출물 없음: {fn} (m6_analysis 먼저)")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(os.path.join(results_dir, "summary.csv"))
    print(f"=== M6 검증 === summary {len(df)}행, 모델 {models}")

    errors = []

    # 전 조합 행 존재 + 빈칸엔 reason 필수 (보간 금지)
    have = {(r["split"], r["model"], r["condition"], int(r["k"])) for _, r in df.iterrows()}
    expected = list(expected_combos(splits, models, k_sweep))
    absent = [c for c in expected if c not in have]
    if absent:
        errors.append(f"summary.csv 에 조합 행 자체가 없음 {len(absent)}건 (예: {absent[:3]})")
    blank = df[df["func_acc"].isna()]
    bad_blank = blank[blank["reason"].isna() | (blank["reason"].astype(str).str.strip() == "")]
    if len(bad_blank):
        errors.append(f"빈 조합에 reason 누락 {len(bad_blank)}건")
    print(f"  조합 커버: 기대 {len(expected)}, 실측치 행 {len(df) - len(blank)}, 빈칸+사유 {len(blank)}")

    # completeness 원인 분리 집계
    if not {"retrieval_miss", "generation_miss"} <= set(df.columns):
        errors.append("retrieval_miss/generation_miss 열 없음 (원인 분리 미집계)")
    else:
        rm, gm = int(df["retrieval_miss"].sum()), int(df["generation_miss"].sum())
        print(f"  completeness 원인 분리: retrieval_miss {rm} / generation_miss {gm}")

    # arg_acc 전부 N/A (gold 인자 없음)
    if "arg_acc" in df.columns and df["arg_acc"].notna().any():
        errors.append("arg_acc 에 값 존재 — gold 인자 없는 설계와 모순 (출처 확인)")
    else:
        print("  arg_acc: 전부 N/A (범위 제한 명시)")

    # 결론 5항목 + oracle 상한 명기 + 2B 제외 플래그 반영
    text = open(os.path.join(results_dir, "conclusions.md"), encoding="utf-8").read()
    miss_items = [c for c in "abcde" if f"({c})" not in text]
    if miss_items:
        errors.append(f"conclusions.md 결론 항목 누락: {miss_items}")
    else:
        print("  결론 5항목 (a)~(e) 존재")
    if "oracle" in text.lower() and ("상한" not in text and "upper bound" not in text.lower()):
        errors.append("conclusions.md 에 oracle '상한' 명기 없음 (배포치 제시 금지)")
    if excl.get("hypothesis_2b_vs_9b_unverifiable") and "검증 불가" not in text:
        errors.append("2B 제외 상태인데 conclusions.md 에 '검증 불가' 미반영")

    if errors:
        print("\n[verify_m6] 게이트 실패:", file=sys.stderr)
        for e in errors:
            _fail(e)
        sys.exit(1)
    print("\n[verify_m6] PASS — 전 조합 커버(누락은 사유 기록), 원인 분리·표·그림·결론 확인.")
    sys.exit(0)


if __name__ == "__main__":
    main()
