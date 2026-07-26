"""verify_m6.py — GATE: M6 전체+분석

검사 항목 (모두 통과해야 exit 0, 하나라도 실패 시 exit 1):
  summary.csv 조합 누락 0(빈칸은 사유 있어야), completeness 원인 분리 존재, 표1·2·3+그림1 존재, 결론 5항목

CLI: python verify_m6.py --config config.yaml
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
    print("verify_m6: NOT IMPLEMENTED")
    sys.exit(1)
