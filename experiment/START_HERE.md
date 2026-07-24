# START HERE — Claude Code 실행 지침

## 이 실험을 시작하는 방법

이 디렉토리는 Candidate Tool Retrieval 선행 실험의 명세다. **문서를 읽고 그대로 실행**하되, 아래 규칙을 최우선으로 지킨다.

### 진입점과 읽는 순서
1. **`MILESTONES.md`** — 이것이 상위 통제 문서다. 실행은 M1 → M6 순서로, 각 milestone의 게이트를 통과하고 사람 승인을 받은 뒤에만 다음으로 넘어간다.
2. **`PLAN.md`** — 실험 목적, 고정 자원(모델·임베딩·seed·규모), 3개 실험 축, 산출물 규격.
3. **`rules/*.md`** — 각 단계 실행 규칙. 해당 milestone 시작 전에 반드시 읽는다.

### 절대 규칙 (위반 시 실험 무효)
- 각 milestone 게이트를 통과하기 전에 다음 단계로 **진행 금지**. 게이트 실패 시 중단하고 원인 보고.
- **정답 누출 금지**: test 쿼리가 example 생성·classifier 학습·fusion 계수 선택에 새어들면 안 됨.
- 모호한 지점은 임의로 채우지 말고 **중단하고 질문**.
- 모든 무작위성에 `seed=42`, LLM 디코딩 `temperature=0`.
- oracle 조건은 상한 측정용 — 배포 가능치처럼 제시 금지.

### 한 번에 하나의 milestone
전체를 한 번에 실행하지 말 것. 사람이 "M1 진행"이라고 지시하면 M1만 수행하고 게이트에서 멈춰 검증 결과와 사람 확인용 샘플을 제시한 뒤 승인을 대기한다.

## 환경 준비 (M1 전에 확인)
- Python 환경, 다음 접근 확인:
  - ToolBench 데이터셋 (OpenBMB/ToolBench, G1/G2/G3) — 다운로드 경로·RapidAPI 키 필요 여부 확인
  - 임베딩 모델 `intfloat/multilingual-e5-large`
  - vLLM + `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-2B` (post-trained, Base 아님)
- 접근 불가한 자원이 있으면 M1 시작 전에 보고.

## 파일 목록
```
START_HERE.md            ← 지금 이 파일
PLAN.md                  실험 전체 명세
MILESTONES.md            단계별 게이트 (실행 통제)
rules/
  data-prep.md           M1
  example-generation.md  M2
  retrieval.md           M3
  classifier.md          M4
  scoring.md             M5, M6
  run-matrix.md          M6
```
