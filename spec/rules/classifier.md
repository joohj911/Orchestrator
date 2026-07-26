# rules/classifier.md — M4 multi-label classifier (src/m4_classifier.py)

## 목적
발화→필요 category 예측. 출력이 fusion의 real prior. **무조건 수행.**

## 설계 (확정)
- 입력: query embedding (e5, frozen).
- 출력: 49 category sigmoid (multi-label).
- label: gold_categories multi-hot.
- 모델: frozen e5 위 2-layer MLP. 구조/hidden/dropout 기록.
- loss: BCE. 불균형 크면 class weight/focal.

## Train 데이터 — 통합 (확정)
- I1/I2/I3 train split **합쳐** 하나의 classifier.
- 근거: 배포 시 split 미상. multi-label 학습에 여러 category 동시 예시(I2/I3) 필요.
- **test 300×3은 학습·val 미포함** (누출 금지, 위반 시 무효).
- train/val 분리(seed). val은 calibration·early stop·계수에만.

## 통합 필수 진단
1. train category 빈도 로그 (편차).
2. 극단 불균형(예 최다:최소>50:1) 시 class weight, 방식 기록.
3. per-category AUPRC (test). 전반 저조 vs 특정 category 저조 구분.

## Calibration
- temperature scaling(val). ECE 전/후 기록.

## 산출물
- models/class_classifier.pt + 하이퍼파라미터 json
- results/classifier_eval.json: micro/macro F1, mean+per-category AUPRC, ECE(전/후), T, 분포/보정 내역
- data/class_prior_real.jsonl: test 쿼리별 49-dim 확률

## 완료 조건 (verify_m4)
- test 누출 0 (test id ∉ train/val, 명시 검증+로그)
- classifier_eval.json per-category 포함
- class_prior_real.jsonl 전 test 커버

## 금지
- test를 학습·val·calibration·계수에 사용 금지. split별 개별 classifier 금지.
