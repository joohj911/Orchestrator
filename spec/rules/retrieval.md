# rules/retrieval.md — M3 retrieval + fusion 계수 (src/m3_retrieval.py)

## 임베딩 사전계산
- e5(intfloat/multilingual-e5-large). **prefix 규칙**: query="query: ", 문서="passage: ". mean pooling + L2 normalize.
- Tool 표현:
  - description vector: name+description+params 요약 → 1 벡터
  - example vectors: examples 5개 → 5 벡터
  - 저장: data/embeddings/tool_{id}.npy (6 벡터)
- Query embedding: test 쿼리별 1 벡터 저장. (Router 재사용 가정이나 이 실험은 retrieval까지.)

## 축 B 방법
- **bm25**: tool 텍스트(desc+example) BM25, top-K.
- **dense_single**: query ↔ description vector cosine, top-K.
- **dense_multi**: query ↔ 6벡터 max cosine = tool score, top-K.
- **fusion_add**: score = α·norm(s_sem) + β·p_class. s_sem=dense_multi score.
- **fusion_mult**: score = s_sem · max(p_class, ε)^λ. ε=config.fusion.epsilon(0.05).

## 전처리 (fusion 필수)
- s_sem 정규화: config.fusion.norm_method(zscore), validation 분포 기준. 고정·기록.
- p_class calibration: real 단계 temperature scaling. oracle(0/1)은 생략.

## class prior 출처 (축 C)
- **oracle**: gold_categories ∈ → 1.0, else 0.0.
- **real**: m4 classifier 예측(class_prior_real.jsonl) 사용.

## fusion 계수 grid search
- α,β ∈ config.fusion.alpha_grid/beta_grid, λ ∈ lambda_grid.
- 선택: **validation split에서 Recall_all@10 최대화**. 동률 시 mean candidate 수 작은 쪽.
- validation: 각 split의 eval 부분 (test 분리).
- 저장: fusion_coeffs.json (방법·split별 계수 + Recall_all@10).

## 산출물
- `results/retrieval_{split}_{method}_{K}.jsonl`: {query_id, candidate_tools:[id], gold_tools:[id], recall_all:0/1}
  - recall_all = 1 if gold ⊆ candidate

## 완료 조건 (verify_m3)
- 전 split×방법×K candidate 파일 생성
- fusion_coeffs.json 존재, 계수 test 미사용 검증
- e5 prefix 로그

## 금지
- test를 계수 탐색에 사용 금지. fusion 계수 test 튜닝 금지.
