# rules/retrieval.md — Retrieval 및 candidate 생성

## 목적
축 B의 각 retrieval 방법으로 쿼리별 candidate tool 목록을 생성한다. Fusion 계수는 grid search로 결정한다.

## 임베딩 사전계산
- 모델: `intfloat/multilingual-e5-large`. (e5 계열은 입력 prefix 규칙이 있으니 준수: query에는 `"query: "`, 문서에는 `"passage: "` prefix. 준수 여부 기록.)
- Tool 표현:
  - **Description vector**: `name + description + params 요약` 텍스트 1개 → 1 벡터.
  - **Example vectors**: `data/tools_examples.jsonl`의 example 5개 → 각 1 벡터.
  - 저장: `data/embeddings/tool_{id}.npy` (desc 1 + example 5 = 6 벡터).
- Query embedding: 각 test 쿼리 1 벡터. 저장.

## 축 B 방법별 candidate 생성

### bm25
- Tool 텍스트(desc + example)에 BM25. Query로 top-K.

### dense_single
- Query ↔ **description vector만** cosine. top-K.

### dense_multi
- Query ↔ tool의 6개 벡터 전체 중 **max** cosine = tool score. top-K.
- 이것이 §2.1 multi-vector max aggregation.

### fusion_add / fusion_mult
- Base score = dense_multi의 tool score (`s_sem`).
- Class prior = 해당 tool category의 확률 (`p_class`, 출처는 축 C).
- **전처리 (필수)**:
  - `s_sem` 정규화: validation split의 score 분포로 min-max 또는 z-score. 방법 택1하고 고정, 기록.
  - `p_class` calibration: real 단계는 temperature scaling. oracle 단계는 이미 0/1이므로 생략(soft oracle을 쓸 경우만 적용).
- **additive**: `score = α * norm(s_sem) + β * p_class`
- **multiplicative**: `score = s_sem * max(p_class, ε)^λ`, `ε = 0.05` 고정.
- top-K.

## Class prior 출처 (축 C)

### oracle (1단계)
- 쿼리의 `gold_categories`에 속하면 `p_class = 1.0`, 아니면 `0.0`.
- (선택) soft oracle: 정답 category 0.9, 비정답 0.1 — 둘 다 돌려 비교해도 됨. 기본은 hard.

### real (2단계)
- 별도 multi-label classifier를 학습:
  - 입력: query embedding (동일 e5 모델).
  - 출력: 49 category에 대한 sigmoid 확률.
  - label: 쿼리의 `gold_categories` (multi-hot).
  - train/val: 각 split의 train 부분(ToolBench 원 split) 사용. test 쿼리는 학습에 절대 미포함(누출 금지).
  - 모델: 2-layer MLP 또는 로지스틱. 경량이면 무방. 구조·하이퍼파라미터 기록.
- 예측 확률을 `p_class`로 사용, temperature scaling으로 calibration (val ECE 기록).

## Fusion 계수 grid search
- 탐색: `α ∈ {0.3, 0.5, 0.7}`, `β ∈ {0.3, 0.5, 0.7}`, `λ ∈ {0.5, 1.0}`.
- 대상: fusion_add는 (α, β), fusion_mult는 λ.
- 선택 기준: **validation split에서 Recall_all@10 최대화**. 동률이면 mean candidate 수가 작은 쪽.
- validation split: 각 split의 train/eval 중 eval 부분 사용 (test와 분리).
- 결과 저장: `fusion_coeffs.json` — 방법별·split별 선택된 계수와 그때의 Recall_all@10.

## 산출물
- `results/retrieval_{split}_{method}_{K}.jsonl`
- 스키마: `{query_id, candidate_tools: [id...], gold_tools: [id...], recall_all: 0/1}`
  - `recall_all` = 1 if gold_tools ⊆ candidate_tools else 0.

## 완료 조건
- 전 split × 전 방법 × 전 K에 대해 candidate 파일 생성.
- fusion_coeffs.json 존재, 전 방법·split 커버.
- e5 prefix 규칙 준수 로그.

## 금지
- test 쿼리를 classifier 학습·계수 탐색에 사용 금지.
- fusion 계수를 test에서 튜닝 금지 (validation에서만).
