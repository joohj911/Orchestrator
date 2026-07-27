"""verify_m2.py — GATE: M2 example 누출

검사 항목 (모두 통과해야 exit 0, 하나라도 실패 시 exit 1):
  전 example max_leak_sim ≤ config.leak_sim_threshold(0.9), tool당 정확히 5개(미달 로그),
  tools.jsonl 전 tool 커버(누락 0).

CLI: python verify_m2.py --config config.yaml
동작:
  - m2_examples.py 산출물(tools_examples_checked.jsonl)을 읽어 검사.
  - 실패 시 무엇이 왜 실패했는지 stderr에 명확히 출력하고 sys.exit(1).
  - 통과 시 요약을 stdout에 출력하고 sys.exit(0).
구현: Claude Code. 상세 spec/MILESTONES.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fail(msg: str) -> None:
    print(f"[verify_m2] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, make_dirs=False)
    data_dir = cfg["paths"]["data_dir"]
    threshold = float(cfg["experiment"]["leak_sim_threshold"])
    n_expected = int(cfg["experiment"]["n_examples_per_tool"])

    tools_path = os.path.join(data_dir, "tools.jsonl")
    checked_path = os.path.join(data_dir, "tools_examples_checked.jsonl")

    for p in (tools_path, checked_path):
        if not os.path.isfile(p):
            _fail(f"필요 파일 없음: {p} (M1/M2 먼저 실행)")
            sys.exit(1)

    tools = _read_jsonl(tools_path)
    tool_ids = {t["id"] for t in tools}
    checked = _read_jsonl(checked_path)
    checked_ids = {c["tool_id"] for c in checked}

    errors: list[str] = []

    # 1) 커버리지: 전 tool 커버 + 정확히 n_expected 개.
    missing = sorted(tool_ids - checked_ids)
    if missing:
        errors.append(f"tools.jsonl 중 example 없는 tool {len(missing)}개 (예: {missing[:3]}).")
    extra = sorted(checked_ids - tool_ids)
    if extra:
        errors.append(f"tools.jsonl 에 없는 tool_id 의 example {len(extra)}개 (예: {extra[:3]}).")

    wrong_count = [(c["tool_id"], len(c.get("examples", []))) for c in checked
                   if len(c.get("examples", [])) != n_expected]
    if wrong_count:
        errors.append(
            f"example 수 != {n_expected} 인 tool {len(wrong_count)}개 (예: {wrong_count[:3]})."
        )

    # 2) 누출 컷: 전 example max_leak_sim ≤ threshold.
    violations = []
    max_seen = 0.0
    for c in checked:
        sims = c.get("max_leak_sim", [])
        for i, s in enumerate(sims):
            max_seen = max(max_seen, s)
            if s > threshold:
                violations.append((c["tool_id"], i, round(s, 4)))
    if violations:
        errors.append(
            f"max_leak_sim > {threshold} 인 example {len(violations)}개 "
            f"(예: {violations[:5]}). 누출 → 해당 example 재작성 후 재검증."
        )

    print(f"=== M2 요약 ===")
    print(f"  tool 수(example): {len(checked)} / pool {len(tools)}")
    print(f"  example 총계: {sum(len(c.get('examples', [])) for c in checked)}")
    print(f"  최대 leak_sim: {round(max_seen, 4)} (threshold {threshold})")

    if errors:
        print("\n[verify_m2] 게이트 실패:", file=sys.stderr)
        for e in errors:
            _fail(e)
        sys.exit(1)

    print(f"\n[verify_m2] PASS — 전 tool {n_expected}개 example, 누출 max_leak_sim ≤ {threshold}.")
    sys.exit(0)


if __name__ == "__main__":
    main()
