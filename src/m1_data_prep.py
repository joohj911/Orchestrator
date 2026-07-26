"""m1_data_prep.py

명세: spec/rules/data-prep.md
역할: ToolBench 로드, subset 샘플링, tool pool 500 구성
산출물: data/tools.jsonl, data/queries_{I1,I2,I3}.jsonl

CLI: python m1_data_prep.py --config config.yaml [--force] [--smoke]
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
# TODO(Claude Code): spec/rules/data-prep.md 명세대로 구현
