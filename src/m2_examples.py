"""m2_examples.py

명세: spec/rules/example-generation.md
역할: **검증 전용** (생성 아님).
  - example(data/tools_examples.jsonl)은 Claude Code가 코드 작성 단계에서 직접 작성해 레포에 커밋한다.
  - 이 스크립트는 서버 런타임에서 그 example을 로드하여 누출 컷만 검사한다.

동작:
  1. data/tools.jsonl, data/tools_examples.jsonl 로드.
  2. 각 example을 e5로 임베딩 (utils/embed.embed_queries).
  3. 그 tool을 gold로 갖는 모든 test 쿼리와 cosine 유사도 계산.
  4. 각 example의 max_leak_sim 기록. > config.leak_sim_threshold(0.9)면 위반으로 표시.
  5. 커버리지 검사: 모든 tool이 정확히 5개 example을 갖는지.
  6. 결과를 data/tools_examples_checked.jsonl (max_leak_sim 채워서)로 저장.
     실제 통과/실패 판정은 verify_m2.py가 수행.

CLI: python m2_examples.py --config config.yaml
규칙:
  - 여기서 example을 생성하지 않는다. 없으면 에러로 중단하고 "Claude Code가 작성해야 함" 안내.
  - seed=42.
구현: Claude Code.
"""
# TODO(Claude Code): 위 동작 구현. example 생성 로직은 넣지 말 것 (검증만).
