# MILESTONES.md — 단계별 게이트

이 실험은 앞 단계 오류가 뒤로 전파된다 (누출 example → 전 retrieval 무효, gold 누락 → Recall 상한 붕괴). 따라서 각 milestone마다 **자동 검증을 통과하고 사람 승인을 받은 뒤에만** 다음으로 진행한다. 통과 못 하면 중단하고 원인을 보고한다. 다음 단계로 임의 진행 금지.

각 milestone 완료 시: (1) 자동 검증 스크립트 실행 → 결과 출력, (2) 사람 확인용 샘플·요약 제시, (3) 명시적 승인 대기.

---

## M1 — 데이터 준비
- **산출물**: `data/tools.jsonl`, `data/queries_{I1,I2,I3}.jsonl`
- **자동 검증 (전부 통과 필수)**:
  - 전 쿼리의 gold_tools ⊆ tools.jsonl (누락 0)
  - tools.jsonl description 결측 0
  - 각 split 정확히 300개, multi-tool split은 gold ≥ 2
  - category 분포 로그 출력 (pool / split별)
- **사람 확인**: 무작위 쿼리 10개의 (query, gold_tools, gold_categories) 육안 검토
- **게이트**: gold 누락이 하나라도 있으면 중단 (Recall_all 상한이 1 미만이 되어 실험 무효)

## M2 — Example 생성
- **산출물**: `data/tools_examples.jsonl`
- **자동 검증**:
  - 전 example의 max_leak_sim ≤ 0.9 (누출 컷)
  - tool당 5개 (미달 tool 목록 출력, 0개가 이상적)
- **사람 확인**: 무작위 tool 10개의 example 5개씩 품질·다양성 검토, API명/파라미터명 노출 여부
- **게이트**: max_leak_sim > 0.9인 example이 남아 있으면 중단

## M3 — Retrieval + Fusion 계수
- **산출물**: `data/embeddings/`, `results/retrieval_*`, `fusion_coeffs.json`
- **자동 검증**:
  - Recall_all 계산 정확성 (gold ⊆ candidate 검산)
  - fusion 계수가 **validation에서만** 선택됨 (test 미사용 검증)
  - e5 prefix 규칙 준수 로그
- **사람 확인**: oracle prior의 Recall_all 곡선이 타당한가 (oracle_tool ≈ 1.0, random_k 낮음, retrieved 중간). dense_multi > dense_single 방향 확인
- **게이트**: fusion 계수가 test에서 선택된 흔적이 있으면 중단

## M4 — Classifier 학습
- **산출물**: classifier, `results/classifier_eval.json`, `data/class_prior_real.jsonl`
- **자동 검증**:
  - test 쿼리 id가 train/val에 **없음** (누출 0, 명시 검증)
  - classifier_eval.json에 per-category AUPRC 포함
  - class_prior_real.jsonl이 전 test 쿼리 커버
- **사람 확인**: multi-label F1·ECE가 납득 범위인가, per-category 중 심하게 무너진 category 파악
- **게이트**: test 누출 발견 시 중단

## M5 — Downstream 파일럿 (전체 실행 전 필수)
- **목적**: 전체 수백 조합 실행 전, 파이프라인·파서 검증
- **범위**: I1 split, K=10, 전 retrieval 방법, 2B & 9B만 (소규모)
- **자동 검증**:
  - **2B 파싱 성공률 측정** — tool-call 포맷을 얼마나 안정적으로 따르는가. 낮으면(예: <90%) scoring.md에 파싱 실패 별도 집계 규칙 추가 후 재확인
  - 9B 파싱 성공률
  - 프롬프트 템플릿·디코딩 파라미터가 조건 간 동일함을 로그로 확인
- **사람 확인**: 파일럿 결과 방향이 상식적인가 (full ≤ oracle_tool, 2B가 9B보다 좁히기 효과 큼 조짐)
- **게이트**: 2B 파싱이 심각하게 불안정하면(예: 파싱 성공률 <90%) 아래 우선순위로 대응하고 진행:
  1. **2B 제외** (우선) — 2B를 실험에서 빼고 9B 단독으로 진행. "2B/9B 가설(작은 모델일수록 좁히기 효과 큼)"은 검증 불가로 결론에 명기하고, 9B 결과만으로 좁히기 유의미성·방법 비교를 수행.
  2. (2B 결과가 꼭 필요하다고 사람이 판단할 때만) 파서 교체 또는 프롬프트 조정 후 재확인.
  - 기본 동작은 1번. 2·3번은 사람의 명시적 요청이 있을 때만.

## M6 — 전체 실행 + 분석
- **산출물**: `results/summary.csv`, 표 1·2·3, 그림 1
- **자동 검증**:
  - 조합 누락 0 (누락은 빈칸 + 사유, 보간 금지)
  - completeness 원인 분리(retrieval_miss / generation_miss) 집계됨
- **사람 확인**: 결론 5개 항목 검토
  - (a) 좁히기 유의미한가 (full vs oracle_tool)
  - (b) 어느 retrieval 방법이 최선인가
  - (c) 2B/9B 가설 성립하는가
  - (d) class prior oracle→real gap
  - (e) dependency 진단 (저유사도 필수 tool 누락률) → 후속 확장 필요 여부
- **게이트**: 없음 (최종 단계). 결과 해석에서 oracle을 배포 가능치로 제시하지 말 것.

---

## 전역 규칙
- 각 게이트에서 중단 시: 무엇이 왜 실패했는지 + 재현 방법을 보고. 임의 우회 금지.
- Milestone 산출물은 삭제·덮어쓰지 말고 버전 유지 (M5 파일럿 결과 포함).
- 애매한 판단은 진행 말고 질문.
