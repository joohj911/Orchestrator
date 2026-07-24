# rules/data-prep.md — 데이터 준비

## 목적
ToolBench에서 실험용 subset을 구성한다. 산출물: tool pool, split별 쿼리, tool 메타데이터.

## 입력
- ToolBench 데이터셋 (OpenBMB/ToolBench, G1/G2/G3 = I1/I2/I3)
- RapidAPI 계층 정보 (category → tool → API)

## 절차

### 1. 원본 로드
- G1(single-tool)=I1, G2(intra-category)=I2, G3(intra-collection)=I3 의 test split을 로드.
- 각 인스턴스에서 추출: `query`(사용자 발화), `gold_apis`(정답 API 목록), 각 API의 `tool`·`category`.
- 용어 통일: 이 실험에서 "tool" = ToolBench의 **API 단위**(호출 단위). category = RapidAPI 49개 대분류.

### 2. 쿼리 샘플링
- 각 split에서 seed=42로 300개 무작위 추출.
- 조건: gold_apis가 tool pool(아래) 안에 전부 포함 가능한 것만. 불가능하면 재추출.
- multi-tool split(I2/I3)은 gold_apis 길이 ≥ 2인 것만.

### 3. Tool pool 구성 (500개)
- 세 split 300개씩의 gold API를 모두 합집합 → 필수 포함(P_gold).
- |P_gold| < 500이면 distractor로 500까지 채움:
  - Distractor 선택: gold category 분포와 **유사한 비율**로 다른 API를 샘플링(현실적 혼동 유발). category당 상한을 두어 한 category 독점 방지.
  - seed=42.
- |P_gold| ≥ 500이면 500 초과분을 줄이지 말고 pool을 |P_gold|로 설정하고 PLAN의 500을 실제값으로 갱신 기록.

### 4. Tool 메타데이터 정규화
각 tool에 대해 저장:
- `id`: 고유 식별자 (category__tool__api 형태로 충돌 방지)
- `name`: API 이름
- `description`: API description (없으면 tool description으로 대체, 대체 사실 기록)
- `params`: 파라미터 목록 (name, type, description, required)
- `category`: RapidAPI category

### 5. 산출물 저장
- `data/tools.jsonl` — tool 500개, 위 스키마
- `data/queries_I1.jsonl`, `_I2.jsonl`, `_I3.jsonl` — 각 300개
  - 스키마: `{query, gold_tools: [id...], gold_categories: [category...]}`
  - `gold_categories`는 gold_tools의 category 합집합 (class prior oracle·real 학습에 공통 사용)

## 완료 조건
- 3개 split 각 300개, 전 쿼리의 gold_tools가 tools.jsonl에 존재.
- tools.jsonl에 description 결측 tool 0개 (결측은 대체 후 기록).
- category 분포 로그 출력 (pool과 각 split).

## 금지
- gold API가 pool에 없는 쿼리를 남기지 말 것 (Recall_all 상한이 1 미만이 되어 실험 무효).
- 원본 query 텍스트 수정·번역 금지 (영어 원문 유지).
