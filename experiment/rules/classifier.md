# rules/classifier.md — Multi-label Class Classifier 학습

## 목적
발화로부터 필요한 tool category를 예측하는 multi-label classifier를 학습한다. 이 classifier의 출력이 fusion 단계의 `p_class` (prior_source=real)가 된다. Classifier 실험은 **무조건 수행**한다 (oracle과 real을 항상 둘 다 측정).

## 설계 (확정)

- **입력**: query embedding — `intfloat/multilingual-e5-large` (실험 전체와 동일 모델, query prefix 규칙 준수). 추가 인코더 없음, e5는 frozen.
- **출력**: 49개 RapidAPI category에 대한 sigmoid (multi-label).
- **label**: 쿼리의 `gold_categories` multi-hot 벡터.
- **모델**: frozen e5 위 2-layer MLP (hidden 1개). 구조·hidden dim·dropout 기록.
- **loss**: binary cross-entropy. category 불균형이 크면 class weight 또는 focal loss로 보정 (아래 진단 결과에 따라).

## Train 데이터 — 통합 (확정)

- I1/I2/I3의 **train split을 모두 합쳐** 하나의 classifier 학습.
- 근거: 배포 시 발화가 어느 split인지 알 수 없음. Multi-label을 학습하려면 여러 category가 동시에 켜진 예시(I2/I3)가 train에 필요.
- **test 300×3 쿼리는 학습·검증에 절대 미포함** (누출 금지, 위반 시 전체 실험 무효).
- train/val 분리: ToolBench 원본 train을 다시 train/val로 나눔 (seed=42). val은 calibration과 early stopping·계수 선택에만 사용.

## 통합의 필수 진단 (category 편향)

1. **분포 로그**: 통합 train set의 category별 빈도 출력. 상위/하위 편차 기록.
2. **불균형 보정**: 극단적 불균형(예: 최다:최소 > 50:1) 시 class weight 적용. 적용 여부·방식 기록.
3. **per-category 성능**: test에서 category별 AUPRC 측정. 전반 저조 vs 특정 category 저조를 구분.
   - 이 진단이 있어야 real classifier가 oracle에 못 미칠 때 원인(어느 category가 무너졌나)을 짚을 수 있음.

## Calibration

- Temperature scaling (val에서 T 학습).
- val ECE 기록 (calibration 전/후).
- calibration된 확률을 `p_class`로 사용.

## 산출물

- `models/class_classifier.pt` (또는 동등) + 구조·하이퍼파라미터 json.
- `results/classifier_eval.json`:
  - 전체: multi-label F1 (micro/macro), mean AUPRC
  - per-category AUPRC (49개)
  - ECE (전/후), temperature T
  - train category 분포, 불균형 보정 내역
- `data/class_prior_real.jsonl`: test 쿼리별 49-dim 예측 확률 (fusion에서 사용).

## 완료 조건

- test 누출 0 (test 쿼리 id가 train/val에 없음을 명시적으로 검증하고 로그).
- classifier_eval.json 생성 (per-category 포함).
- class_prior_real.jsonl이 전 test 쿼리 커버.

## Oracle과의 관계 (해석용, 게이트 아님)

- prior_source=oracle: gold_categories 기반 이상적 prior (상한).
- prior_source=real: 이 classifier 예측.
- **gap(oracle − real)** = classifier가 상한에 못 미친 정도 = classifier 개선 여지.
- oracle조차 효과 없으면 → "prior 방향 무효"라는 결론 (스킵 사유 아님, 결과임).

## 금지
- test 쿼리를 학습·val·calibration·계수 선택 어디에도 사용 금지.
- split별 개별 classifier 학습 금지 (통합 확정).
