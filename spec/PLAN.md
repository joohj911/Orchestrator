# PLAN — 실험 설계 명세

이 문서는 실험 설계의 단일 출처다. 코드는 이 명세를 따른다. 값의 확정본은 `config.yaml`.

## 목적
1. Candidate 좁히기가 downstream tool-call 정확도에 유의미한가. 작은 모델일수록 효과가 큰가.
2. 좁히는 방법(dense_single / dense_multi / class prior fusion) 간 우열.

핵심 가설: 작은 모델(2B)일수록 좁히기 효과가 크다.

## 고정 자원 (config.yaml, 변경 금지)
- 벤치마크: ToolBench I1/I2/I3
- Downstream: Qwen/Qwen3.5-9B, Qwen/Qwen3.5-2B (transformers generate, vLLM 미사용)
- 임베딩: intfloat/multilingual-e5-large
- seed 42, split당 300쿼리, tool pool 500, tool당 example 5, K∈{5,10,20,50}

## 실험 축 (3개 독립)

### 축 A — candidate 범위 (프롬프트에 넣는 tool 집합)
- full: pool 500 전부 (하한, 좁히기 안 함)
- random_k: 무작위 K (정답 강제 포함) — 하한
- retrieved_k: retrieval로 좁힌 K — 측정 대상
- oracle_tool: 정답 tool + distractor로 K — 상한 (완벽한 retrieval)

### 축 B — retrieval 방법 (retrieved_k에만)
- bm25: 참고 baseline
- dense_single: description-only single-vector
- dense_multi: multi-vector(desc+example5) max aggregation
- fusion_add: dense_multi + class prior, additive
- fusion_mult: dense_multi + class prior, multiplicative

### 축 C — class prior 출처 (fusion_*에만)
- oracle: gold category 기반 이상적 prior (상한)
- real: 학습된 multi-label classifier 예측
- **둘 다 무조건 측정.** gap(oracle−real) = classifier 개선 여지. oracle조차 무효면 "prior 방향 무효"로 결론.

## 지표 (상세 spec/rules/scoring.md)
- Retrieval 층: Recall_all@K (정답 tool 전부 candidate 포함 비율). 주 지표.
- Downstream 층: func_acc, arg_acc, (multi) completeness.
- 진단: multi-tool Recall_all, 저유사도 필수 tool 누락률, completeness 원인 분리(retrieval_miss/generation_miss).

## 산출물 (README 참조)
summary.csv 스키마: split, model, candidate_scope, retrieval_method, prior_source, K, recall_all, func_acc, arg_acc, completeness, mean_prompt_tokens, n_queries

## 무결성 규칙
1. 정답 누출 금지 (example/classifier/fusion 계수에 test 미사용).
2. 조건 격리: candidate 목록 외 모든 것 동일 (프롬프트·파서·디코딩·seed).
3. seed=42, greedy.
4. oracle은 상한 표기, 배포치 아님.
5. 애매하면 중단·표시.
