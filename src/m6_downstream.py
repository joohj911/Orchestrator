"""m6_downstream.py

명세: spec/rules/scoring.md + run-matrix.md
역할: 파일럿 통과 조합 전체 downstream 실행
산출물: results/downstream_*.jsonl

CLI: python m6_downstream.py --config config.yaml [--force] [--smoke]
  --config: config.yaml 경로
  --force : 산출물이 있어도 재생성
  --smoke : dummy 소량 입력으로 로직만 점검 (데이터/GPU 없이, 선택)

규칙:
  - config에서 모든 파라미터 로드 (하드코딩 금지).
  - seed=42, greedy 디코딩 고정.
  - 정답 누출 금지 (해당 단계에 test가 새어들지 않게).
  - 산출물 스키마·경로는 명세 준수.
  - 애매하면 임의 결정 말고 '# DECISION NEEDED:' 표시 + 근거.
구현: Claude Code (spec 문서 참조).
"""
# TODO(Claude Code): spec/rules/scoring.md + run-matrix.md 명세대로 구현
