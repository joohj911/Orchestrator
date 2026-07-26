"""verify_m4.py — GATE: M4 classifier 누출

검사 항목 (모두 통과해야 exit 0, 하나라도 실패 시 exit 1):
  test id ∉ train/val(누출 0), classifier_eval.json per-category AUPRC 포함, class_prior_real.jsonl 전 test 커버

CLI: python verify_m4.py --config config.yaml
동작:
  - 산출물을 읽어 위 항목을 검사.
  - 실패 시 무엇이 왜 실패했는지 stderr에 명확히 출력하고 sys.exit(1).
  - 통과 시 요약을 stdout에 출력하고 sys.exit(0).
  - run_all.sh가 exit code로 파이프라인 중단을 판단하므로 exit code 정확히.
구현: Claude Code. 상세 spec/MILESTONES.md.
"""
import sys
# TODO(Claude Code): 검사 구현. 실패 시 sys.exit(1), 통과 시 sys.exit(0).
if __name__ == "__main__":
    # TODO: 실제 검사로 교체
    print("verify_m4: NOT IMPLEMENTED")
    sys.exit(1)
