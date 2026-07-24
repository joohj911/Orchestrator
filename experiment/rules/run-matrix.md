# rules/run-matrix.md — 전체 실행 조합과 결과표

## 목적
전체 조건 조합을 빠짐없이 실행하고, 최종 결과표를 생성한다.

## 실행 조합

### 1단계 (class prior = oracle)
```
for split in [I1, I2, I3]:
  for model in [Qwen3.5-2B, Qwen3.5-9B]:
    for K in [5, 10, 20, 50]:
      # candidate 범위
      full          (K 무관, pool 전체 — K 루프 1회만)
      random_k      (K)
      oracle_tool   (K)
      # retrieval 방법 (retrieved_k 범위)
      bm25          (K)
      dense_single  (K)
      dense_multi   (K)
      fusion_add    (K, prior=oracle)
      fusion_mult   (K, prior=oracle)
```

### 2단계 (class prior = real) — 무조건 실행
```
for split, model, K:
  fusion_add    (K, prior=real)
  fusion_mult   (K, prior=real)
```
- Classifier 실험은 항상 수행한다. oracle과 real을 **둘 다 측정**하고, 게이트로 스킵하지 않는다.
- oracle과 real의 관계는 게이트가 아니라 **해석 축**이다:
  - gap(oracle − real) = classifier가 상한에 못 미친 정도 = classifier 개선 여지.
  - oracle조차 dense_multi 대비 효과 없으면 → "class prior 방향 자체가 무효"라는 **결론**으로 기록 (스킵 사유 아님).
- Classifier 학습 명세는 `rules/classifier.md`.

## 비용 절감 규칙
- `full`은 K와 무관 → split·model당 1회만 실행.
- retrieval 산출물(candidate 목록)은 downstream과 분리 캐싱 → 같은 candidate에 대해 모델만 바꿔 재사용.
- 임베딩은 1회 계산 후 재사용.

## 최종 결과표 (summary.csv → 분석용 피벗)

### 표 1 — 좁히기 유의미성 (candidate 범위 대조)
행: {full, random_k, oracle_tool, best_retrieved} × split × model
열: func_acc, arg_acc, completeness, mean_prompt_tokens
→ full vs oracle_tool 격차 = 좁히기 상한 효과. 2B vs 9B 격차 = 가설 검증.

### 표 2 — 방법 비교 (retrieval 방법별)
행: {bm25, dense_single, dense_multi, fusion_add, fusion_mult} × split
열: Recall_all@K (K별), downstream func_acc/completeness
→ multi-vector·class prior의 기여 분리.

### 그림 1 — K sweep 곡선
x: K (또는 mean_prompt_tokens), y: downstream 정확도
곡선: 각 retrieval 방법 + full/oracle_tool 수평선. split·model별 subplot.

### 표 3 — 진단
- multi-tool Recall_all (I2/I3, 방법별)
- completeness 원인 분리 (retrieval_miss vs generation_miss 비율)
- 저유사도 필수 tool 누락률

## 완료 조건
- summary.csv 전 조합 채움 (2단계 생략 시 그 사유 기록).
- 표 1·2·3, 그림 1 생성.
- 결론 요약: (a) 좁히기 유의미한가, (b) 어느 방법이 최선인가, (c) 2B/9B 가설 성립하는가, (d) class prior oracle→real gap, (e) dependency 진단 결과 → 후속 확장 필요 여부.

## 금지
- 조합 누락 후 보간 금지. 실행 못 한 조합은 빈칸으로 두고 사유 기록.
- oracle 조건을 배포 가능 결과처럼 제시 금지 (상한임을 명기).
