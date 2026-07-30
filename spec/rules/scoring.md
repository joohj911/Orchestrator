# rules/scoring.md — M5 파일럿 / M6 downstream + 채점 (src/m5_pilot.py, src/m6_downstream.py)

## Downstream 실행 (vLLM 미사용)
- 모델: Qwen3.5-9B, 2B. transformers AutoModelForCausalLM.
- 프롬프트: tokenizer.apply_chat_template(messages, tools=candidate_schemas, add_generation_prompt=True).
  - **템플릿 전 조건 동일. 바뀌는 것은 candidate tool 목록(축 A)뿐.**
  - tool schema는 OpenAI function-calling 포맷.
- 디코딩: config.decoding (greedy, temperature 0, max_new_tokens). 전 조건 동일.
- 출력에서 tool_call 파싱: utils/qwen_tools.py. **파싱 성공/실패 명시 집계.**

## 채점 (BFCL 스타일, 매칭 기반, 실행 아님)
- **func_acc**: 호출 함수명이 gold와 일치.
  - I1: 정답 tool 호출=1.
  - I2/I3: precision(호출 중 정답 비율)으로 기록. set 일치는 completeness로 별도.
- **arg_acc**: 올바른 함수 호출 시, 필수 파라미터 값이 gold와 일치. 정확 일치 우선, 자유 텍스트는 정규화(소문자/공백) 후 비교, 애매하면 LLM judge(사용 여부 기록). = 정확히 채운 필수 인자 / 필수 인자, 쿼리 평균.
- **completeness (I2/I3)**: gold 전부 호출=1, 쿼리 평균.
  - **원인 분리**: retrieval_miss(candidate에 gold 없음, recall_all=0) vs generation_miss(candidate엔 있으나 미호출). 비율 기록.

## 진단
- multi-tool subset Recall_all (I2/I3).
- 저유사도 필수 tool 누락률: gold 중 query-dense_single 유사도 하위 20% tool의 candidate 누락 비율.

## M5 파일럿 (src/m5_pilot.py)
- 범위: I1, K=10, 전 방법, 2B+9B.
- **구조적 파싱 성공률 측정** = ok/(ok+malformed). 모델별 성공률 <
  config.pilot.parse_success_threshold(0.90)이면 자동 제외.
  - no_call(호출 없이 답변)은 게이트가 아니라 행동 지표 — 채점(0점)에 이미 반영,
    모델별·조건별 비율을 보고 (재정의 2026-07-30, 근거는 verify_m5 docstring).
  - 우선순위: 2B 제외. 9B도 미달이면 중단·보고.
  - 2B 제외 시 "2B/9B 가설 검증 불가" 플래그 기록.

## 산출물
- results/downstream_{split}_{model}_{condition}_{K}.jsonl: {query_id, called_tools, called_args, parse_ok, func_acc, arg_acc, completeness, miss_type, prompt_tokens}
- 집계 → results/summary.csv

## 완료 조건
- (M5) 파일럿 파싱 성공률 리포트, 제외 모델 기록.
- (M6) 전 조건×모델×split×K 실행, completeness 원인 분리 포함.

## 금지
- 조건마다 프롬프트/디코딩 변경 금지. 실제 API 실행 채점 아님(매칭만).
