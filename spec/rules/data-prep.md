# rules/data-prep.md — M1 데이터 준비 (src/m1_data_prep.py)

## 목적
ToolBench에서 실험 subset 구성. 산출물: tool pool, split별 쿼리.

## 절차
1. **로드**: G1=I1(single), G2=I2(intra-category), G3=I3(intra-collection)의 test split. 각 인스턴스에서 query, gold_apis(정답), 각 API의 tool/category 추출.
   - 용어: 이 실험의 "tool" = ToolBench API 단위(호출 단위). category = RapidAPI 49 대분류.
2. **쿼리 샘플링**: split당 seed로 300개. gold가 tool pool에 전부 포함 가능한 것만. multi-tool split은 gold≥2.
3. **tool pool(500)**: 세 split gold API 합집합=필수포함. 500 미달 시 distractor로 채움(gold category 분포 비율 유사, category당 상한). gold가 500 초과면 pool=|gold|로 두고 실제값 기록.
4. **메타데이터 정규화**: id(category__tool__api), name, description(결측 시 tool description 대체+기록), params(name/type/description/required), category.

## 산출물
- `data/tools.jsonl`: {id, name, description, params, category}
- `data/queries_{I1,I2,I3}.jsonl`: {query_id, query, gold_tools:[id], gold_categories:[category]}
  - gold_categories = gold_tools의 category 합집합

## 완료 조건 (verify_m1)
- 전 쿼리 gold_tools ⊆ tools.jsonl (누락 0)
- description 결측 0, 각 split 300, multi-tool gold≥2
- category 분포 로그

## 금지
- gold가 pool에 없는 쿼리 잔존 금지 (Recall 상한 붕괴).
- query 텍스트 수정·번역 금지.
