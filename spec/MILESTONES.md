# MILESTONES — 단계 게이트 (run_all.sh가 자동 강제)

각 단계 뒤 `verify_m*.py`가 돌고, 실패 시 파이프라인이 exit 1로 중단된다. 게이트 = verify 스크립트의 검사 항목.

## M1 데이터 (verify_m1.py)
- 전 쿼리 gold_tools ⊆ tools.jsonl (누락 0) — 실패 시 중단 (Recall 상한 붕괴)
- description 결측 0
- 각 split 정확히 300, multi-tool split gold ≥ 2
- category 분포 로그 출력

## M2 example (verify_m2.py)
- example은 Claude Code가 작성해 레포에 커밋 (런타임 생성 아님). m2는 검증만.
- 전 example max_leak_sim ≤ config.leak_sim_threshold(0.92) — 실패 시 중단 (누출)
- tool당 정확히 5개 (미달 목록 출력)

## M3 retrieval (verify_m3.py)
- Recall_all 검산 (gold ⊆ candidate 재확인)
- fusion 계수가 validation에서만 선택됨 (test 미사용 흔적 검증)
- e5 prefix 규칙 준수 로그

## M4 classifier (verify_m4.py)
- test 쿼리 id가 train/val에 없음 (누출 0) — 실패 시 중단
- classifier_eval.json에 per-category AUPRC 포함
- class_prior_real.jsonl 전 test 쿼리 커버

## M5 파일럿 (verify_m5.py)
- 소규모(I1, K=10, 전 방법, 2B+9B) 실행 후 파싱 성공률 측정
- 파싱 성공률 < config.pilot.parse_success_threshold(0.90)인 모델은 **자동 제외**하고 기록
  - 우선순위: 2B(weak) 제외. 9B도 미달이면 중단·보고.
  - 2B 제외 시 "2B/9B 가설 검증 불가"를 결론에 명기하도록 플래그 기록.
- 프롬프트 템플릿·디코딩이 조건 간 동일함을 로그로 확인

## M6 전체+분석 (verify_m6.py)
- 조합 누락 0 (누락은 빈칸+사유, 보간 금지)
- completeness 원인 분리(retrieval_miss/generation_miss) 집계됨
- 표 1·2·3, 그림 1, summary.csv 생성
- 결론 5항목: (a)좁히기 유의미 (b)최선 방법 (c)2B/9B 가설 (d)oracle→real gap (e)dependency 진단

## 전역
- 게이트 실패 시 원인+재현법 보고, 임의 우회 금지.
- oracle을 배포치로 제시 금지.
