"""verify_m5.py — GATE: M5 파일럿

검사 항목:
  - 모델별 파싱 성공률 < config.pilot.parse_success_threshold(0.90) → 제외(2B 우선 기록).
  - 9B(strong)도 미달이면 FAIL(중단).
  - 프롬프트 템플릿·디코딩이 조건 간 동일함을 로그(단일 config 기반이라 구조적으로 동일).
  - 제외 결정을 results/pilot_exclusions.json 에 기록(M6 가 읽어 제외 모델 결정 + 가설 플래그).

CLI: python verify_m5.py --config config.yaml
구현: Claude Code. 상세 spec/MILESTONES.md, spec/rules/scoring.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402


def _fail(msg):
    print(f"[verify_m5] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, make_dirs=False)
    results_dir = cfg["paths"]["results_dir"]
    threshold = float(cfg["pilot"]["parse_success_threshold"])

    report_path = os.path.join(results_dir, "pilot_report.json")
    if not os.path.isfile(report_path):
        _fail(f"pilot_report.json 없음: {report_path} (m5 먼저)")
        sys.exit(1)
    rep = json.load(open(report_path, encoding="utf-8"))
    models = rep.get("models", {})

    print(f"=== M5 파일럿 검증 === threshold {threshold}")
    print(f"  decoding={rep.get('decoding')} | system_prompt={'있음' if rep.get('system_prompt') else '없음'} "
          f"(조건 간 동일)")

    weak = models.get("weak", {})
    strong = models.get("strong", {})
    for k, m in models.items():
        print(f"  {k}={m.get('model_id')} 파싱성공률 {m.get('parse_rate')} 상태 {m.get('status_counts')}")

    excluded, hyp_unverifiable = [], False
    # 우선순위: 2B(weak) 미달 → 제외 + 가설 검증불가 플래그.
    if weak and weak.get("parse_rate", 0) < threshold:
        excluded.append("weak")
        hyp_unverifiable = True
        print(f"  → weak(2B) 파싱성공률 {weak.get('parse_rate')} < {threshold} → 제외. "
              f"'2B/9B 가설 검증 불가' 플래그.")

    errors = []
    # 9B(strong) 미달이면 중단.
    if strong and strong.get("parse_rate", 0) < threshold:
        errors.append(f"strong(9B) 파싱성공률 {strong.get('parse_rate')} < {threshold} → 파서/프롬프트 점검 필요(중단).")

    # 제외 결정 기록 (M6 참조).
    excl = {"threshold": threshold, "excluded_models": excluded,
            "hypothesis_2b_vs_9b_unverifiable": hyp_unverifiable,
            "parse_rates": {k: m.get("parse_rate") for k, m in models.items()}}
    with open(os.path.join(results_dir, "pilot_exclusions.json"), "w", encoding="utf-8") as f:
        json.dump(excl, f, ensure_ascii=False, indent=2)

    if errors:
        print("\n[verify_m5] 게이트 실패:", file=sys.stderr)
        for e in errors:
            _fail(e)
        sys.exit(1)
    kept = [k for k in models if k not in excluded]
    print(f"\n[verify_m5] PASS — 유지 모델 {kept}, 제외 {excluded}. pilot_exclusions.json 기록.")
    sys.exit(0)


if __name__ == "__main__":
    main()
