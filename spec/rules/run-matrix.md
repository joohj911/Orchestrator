# rules/run-matrix.md — M6 전체 조합 + 분석 (src/m6_downstream.py, src/m6_analysis.py)

## 실행 조합
### 1단계 (prior=oracle)
```
for split in [I1,I2,I3]:
 for model in [파일럿 통과 모델]:
  for K in [5,10,20,50]:
   full(K무관 1회), random_k, oracle_tool,
   bm25, dense_single, dense_multi,
   fusion_add(prior=oracle), fusion_mult(prior=oracle)
```
### 2단계 (prior=real) — 무조건 실행
```
for split,model,K: fusion_add(real), fusion_mult(real)
```
- classifier 실험 항상 수행. oracle/real 둘 다 측정. gap 해석(§classifier.md).

## 비용 절감
- full은 K무관 → split·model당 1회.
- candidate 목록 캐싱 → 모델만 바꿔 재사용.
- 임베딩 1회 계산 재사용.

## 결과표 (src/m6_analysis.py)
- **표1 좁히기 유의미**: 행 {full, random_k, oracle_tool, best_retrieved}×split×model, 열 func_acc/arg_acc/completeness/mean_prompt_tokens. full vs oracle_tool 격차 + 2B vs 9B 격차.
- **표2 방법 비교**: 행 {bm25,dense_single,dense_multi,fusion_add,fusion_mult}×split, 열 Recall_all@K(K별)/func_acc/completeness.
- **그림1 K sweep**: x=K(또는 mean_prompt_tokens), y=정확도, 곡선=방법 + full/oracle_tool 수평선. split·model subplot.
- **표3 진단**: multi-tool Recall_all, completeness 원인 분리, 저유사도 필수 tool 누락률.

## 완료 조건 (verify_m6)
- summary.csv 전 조합 (누락은 빈칸+사유, 보간 금지)
- 표1·2·3, 그림1 생성
- 결론 5항목 기록: (a)좁히기 유의미 (b)최선 방법 (c)2B/9B 가설 (d)oracle→real gap (e)dependency 진단→후속 확장 판단

## 금지
- 조합 누락 보간 금지. oracle을 배포치로 제시 금지.
