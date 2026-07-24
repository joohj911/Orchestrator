# rules/example-generation.md — 발화 example 생성

## 목적
각 tool의 multi-vector 표현에 쓸 representative user utterance example을 tool당 5개 생성한다. 이 example은 1·2단계 공통으로 고정 사용되는 자산이다.

## 핵심 제약 — 정답 누출 방지 (위반 시 전체 실험 무효)
생성한 example이 test 쿼리와 유사하면, multi-vector가 정답을 "외운" 것이 되어 결과가 무효가 된다. 반드시:

1. 생성된 각 example을 임베딩(`intfloat/multilingual-e5-large`)으로 변환.
2. 해당 tool을 gold로 갖는 **모든 test 쿼리**와 cosine 유사도 계산.
3. 최대 유사도 > **0.9**이면 그 example 폐기하고 재생성.
4. 5개를 확보할 때까지 반복. 재생성 10회 초과 시 그 tool을 로그에 남기고 확보된 개수만 사용(원인 조사용).

## 생성 절차

### 입력
- `data/tools.jsonl`의 각 tool (name, description, params, category)

### 프롬프트 규칙
- 생성 LLM에 tool의 name/description/params를 주고, "이 tool을 호출하게 만드는 자연스러운 사용자 발화 5개"를 요청.
- 발화는 **사용자 관점의 목표 표현**이어야 함 (API 이름·파라미터명을 그대로 노출하지 말 것). 예: `get_weather_by_city` → "내일 서울 날씨 알려줘" (O) / "get_weather_by_city 호출해줘" (X).
- 영어 test set이므로 example도 **영어**로 생성 (임베딩 모델은 다국어지만 분포 일치를 위해 언어 통일).
- 표현 다양성: 5개가 서로 다른 표현·구조가 되도록 (문체·길이·구체성 변주).

### 저장
- `data/tools_examples.jsonl`
- 스키마: `{tool_id, examples: [str x5], regeneration_count, max_leak_sim}`
  - `max_leak_sim`: 최종 채택된 example들의 test 쿼리 대비 최대 유사도 (누출 검증 기록)

## 완료 조건
- 전 tool에 대해 example 생성 완료.
- 전 example의 `max_leak_sim` ≤ 0.9.
- 누출 컷 초과로 5개 미달인 tool 목록 로그 출력 (0개가 이상적).

## 주의
- 이 단계는 downstream 모델(Qwen)이 아니라 별도 생성 LLM으로 수행 가능. 어떤 모델을 썼는지 기록.
- Example은 한 번 생성 후 **고정**. 1단계·2단계에서 재생성하지 말 것.
