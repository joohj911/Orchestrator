# rules/example-generation.md — example 생성(작성 단계) + 검증(런타임)

## 생성 주체 — Claude Code (코드 작성 단계)
example은 서버 런타임이 아니라 **Claude Code가 직접 작성**해 레포에 커밋한다. local LLM 생성기 없음.

### 선행 조건
- tool 목록(data/tools.jsonl)이 있어야 함 → M1(m1_data_prep.py)을 먼저 실행해 tools.jsonl 생성.
  - Claude Code가 ToolBench 데이터에 접근 가능하면 M1을 실제 실행해 tools.jsonl을 만든 뒤 example 작성.
  - tools.jsonl의 500개 tool 각각에 대해 example 5개 작성.

### 작성 규칙 (tool당 5개)
- 각 tool의 name/description/params를 보고 "이 tool을 호출하게 만드는 자연스러운 사용자 발화" 5개.
- **API명·파라미터명을 노출하지 말 것** (사용자 목표 표현). 예: get_weather_by_city → "내일 서울 날씨 알려줘"(O) / "get_weather_by_city 호출"(X).
- **영어**로 작성 (test set 언어 일치).
- 5개가 서로 다른 표현·구조·길이·구체성.
- **누출 주의**: test 쿼리를 그대로 베끼지 말 것. (Claude Code는 test 쿼리를 참조하지 않고 tool 정보만으로 작성 → 누출 위험 원천 차단.)

### 저장 (Claude Code가 레포에 커밋)
- `data/tools_examples.jsonl`
- 스키마: {tool_id, examples: [str x5]}
  - max_leak_sim은 서버 검증(m2) 후 채워짐. 작성 시엔 생략 가능.

## 검증 주체 — 서버 런타임 (m2_examples.py + verify_m2.py)
- m2_examples.py: 저장된 example을 e5 임베딩 → 그 tool을 gold로 갖는 test 쿼리와 유사도 → max_leak_sim 기록.
- verify_m2.py (게이트):
  - 전 example max_leak_sim ≤ config.leak_sim_threshold(0.92) — 초과 시 FAIL(중단).
  - 전 tool 정확히 5개 — 미달 시 FAIL.
- 누출 컷에 걸리면: 해당 example을 사람이 수정(또는 Claude Code에 재작성 요청) 후 재실행.

## 요약
- 생성: Claude Code (작성 단계, tool 정보만 보고).
- 저장: 레포에 tools_examples.jsonl 커밋.
- 검증: 서버 런타임에서 누출 컷·커버리지 게이트.
- example은 한 번 작성 후 고정 (실험 전체 공통).
