"""verify_m2.py — GATE: M2 example 누출

검사 항목 (모두 통과해야 exit 0, 하나라도 실패 시 exit 1):
  전 example max_leak_sim ≤ config.leak_sim_threshold(0.9), tool당 5개(미달 로그)

CLI: python verify_m2.py --config config.yaml
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
    print("verify_m2: NOT IMPLEMENTED")
    sys.exit(1)
