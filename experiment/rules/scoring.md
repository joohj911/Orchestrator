# rules/scoring.md — Downstream 실행 및 채점

## 목적
candidate tool 목록을 프롬프트에 넣고 Qwen이 tool-call을 생성하게 한 뒤, 정확도를 채점한다.

## Downstream 실행
- 모델: `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-2B` (각각 별도 실행).
- 서빙: vLLM, `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- 디코딩: `temperature=0`, 고정 seed. 전 조건 동일.
- 프롬프트: 시스템 프롬프트 + candidate tool schema(축 A가 정한 집합) + 사용자 query.
  - **템플릿은 전 조건 동일.** 바뀌는 것은 tool schema 목록뿐.
  - Tool schema는 OpenAI function-calling 포맷으로 통일.
- 출력: 모델이 생성한 tool-call 목록 (함수명 + 인자).

## 채점 (BFCL 스타일, 실행 아닌 매칭 기반)

### func_acc (함수명 정확도)
- 모델이 호출한 함수명이 gold_tools와 일치하는가.
- single-tool(I1): 정답 tool을 호출했으면 1, 아니면 0.
- multi-tool(I2/I3): 정밀도/재현율 대신 **set 일치** 기준은 completeness로 별도 측정. func_acc는 "호출한 것 중 정답 비율"(precision)로 정의하고 기록.

### arg_acc (인자 정확도)
- 올바른 함수를 호출한 경우에 한해, 필수 파라미터의 값이 gold와 일치하는가.
- 매칭 규칙:
  - 정확 일치 우선.
  - 자유 텍스트 인자(문자열 값)는 정규화(소문자, 공백 정리) 후 비교. 그래도 애매하면 LLM judge로 판정하되, judge 사용 여부를 인자별로 기록.
- arg_acc = (정확히 채운 필수 인자 수) / (필수 인자 수), 쿼리 평균.

### completeness (multi-tool 전용, I2/I3)
- gold_tools 전부를 호출했으면 1, 아니면 0. 쿼리 평균.
- **원인 분리 (중요)**: completeness 실패를 두 유형으로 태깅.
  - `retrieval_miss`: gold tool이 candidate에 애초에 없었음 (recall_all=0).
  - `generation_miss`: candidate엔 있었으나 모델이 호출 안 함.
  - 이 분리로 "retrieval 탓 vs 생성 탓"을 구분. summary에 비율 기록.

## 진단 지표
- **multi-tool subset Recall_all**: I2/I3에서 방법별 Recall_all (retrieval.md 산출물에서 집계).
- **저유사도 필수 tool 누락률**: gold tool 중 query와 dense_single 유사도가 하위 20%인 tool의 candidate 누락 비율. (dependency/prerequisite 문제 크기 진단 — §후속 확장 판단 근거.)

## 산출물
- `results/downstream_{split}_{model}_{condition}_{K}.jsonl`
  - `condition` = candidate_scope × retrieval_method × prior_source 조합 식별자.
  - 스키마: `{query_id, called_tools, called_args, func_acc, arg_acc, completeness, miss_type, prompt_tokens}`
- 집계 → `results/summary.csv` (PLAN §6 스키마).

## 완료 조건
- 전 조건 × 2 모델 × 3 split × K sweep 실행 완료.
- summary.csv 생성, completeness 원인 분리 비율 포함.

## 금지
- 조건마다 프롬프트 템플릿·디코딩 파라미터를 바꾸지 말 것.
- 실행 채점(실제 API 호출)은 이 선행 실험 범위 아님 — 매칭 기반만.
