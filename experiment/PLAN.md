# 실험 계획서 — Candidate Tool Retrieval 선행 실험

이 문서는 **실행 에이전트(Claude Code)가 따라 수행하는 명세**다. 사람용 설계 설명이 아니라 실행 규칙이다. 모호한 지점은 임의로 채우지 말고, 해당 `rules/*.md`를 확인하고 그래도 불명확하면 중단하고 질문한다.

## 0. 실험 목적

두 가지를 측정한다.

1. **Candidate 좁히기가 downstream tool-call 정확도에 유의미한가** — 전체 tool을 다 주는 것 대비, 좁힌 candidate만 주는 것이 정확도를 올리는가. On-device 급(작은 모델)에서 효과가 더 큰가.
2. **좁히는 방법 간 우열** — dense single-vector / multi-vector / class prior fusion 중 무엇이 정답 tool을 candidate에 더 잘 담고(retrieval recall), downstream 정확도를 더 올리는가.

핵심 가설: **작은 모델일수록 candidate 좁히기의 효과가 크다.** (2B에서 9B보다 개선폭이 크게 나오면 "on-device일수록 retrieval이 중요하다"가 데이터로 증명됨.)

## 1. 고정 자원 (전 실험 공통, 변경 금지)

| 항목 | 값 |
|---|---|
| 벤치마크 | ToolBench (ToolLLM, OpenBMB) — I1 / I2 / I3 |
| Downstream 모델 (강) | `Qwen/Qwen3.5-9B` (post-trained, Base 아님) |
| Downstream 모델 (약) | `Qwen/Qwen3.5-2B` (post-trained, Base 아님) |
| Tool-call 서빙 | vLLM, `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| 임베딩 모델 | `intfloat/multilingual-e5-large` |
| 난수 seed | `42` (모든 샘플링·셔플에 고정) |
| Split당 쿼리 수 | 300 (I1, I2, I3 각각) |
| Tool pool 크기 | 500 (정답 tool 전부 + distractor로 500 채움) |
| Tool당 example 수 | 5 (생성, 규칙은 `rules/example-generation.md`) |

- `Base` 접미사 모델은 tool-call 불가 → 절대 사용 금지.
- 모든 조건에서 **동일 프롬프트 템플릿·동일 파서** 사용. 조건마다 바뀌는 것은 "프롬프트에 넣는 tool 목록"뿐이다.

## 2. 실험 축 (3개 독립 축)

### 축 A — candidate 범위 (프롬프트에 넣는 tool 집합)
| 조건 | 내용 | 역할 |
|---|---|---|
| `full` | tool pool 500개 전부 | 하한 기준선 (좁히기 안 함) |
| `random_k` | 무작위 K개 (정답 tool 강제 포함) | 하한 (아무렇게나 좁힘) |
| `retrieved_k` | retrieval로 좁힌 K개 | **측정 대상** |
| `oracle_tool` | 정답 tool만 (+ 소수 distractor로 K 맞춤) | 상한 (완벽한 retrieval) |

### 축 B — retrieval 방법 (`retrieved_k`일 때만 적용)
| 조건 | 내용 | 검증 대상 |
|---|---|---|
| `bm25` | BM25 sparse | 참고 baseline |
| `dense_single` | description-only single-vector | dense baseline |
| `dense_multi` | multi-vector (desc + example 5개), max aggregation | multi-vector 효과 |
| `fusion_add` | multi-vector + class prior, additive | class prior 효과 |
| `fusion_mult` | multi-vector + class prior, multiplicative | class prior 효과 |

### 축 C — class prior 출처 (fusion_* 조건에만 적용)
| 조건 | 내용 |
|---|---|
| `oracle` | 정답 tool의 category를 안다고 가정한 prior (1단계) |
| `real` | 실제 학습된 multi-label classifier 예측 (2단계) |

**단계 순서**: 1단계 = class prior `oracle`로 상한 측정, 2단계 = `real` classifier 학습 후 측정. **둘 다 무조건 수행** (classifier 실험은 게이트 없이 항상 함). oracle−real gap은 classifier 개선 여지의 해석 축이다. Example·retrieval·downstream·채점은 두 단계 공통 고정 (example은 단계 무관 고정 자산). Classifier 학습 명세는 `rules/classifier.md`.

## 3. K sweep

`K ∈ {5, 10, 20, 50}`. 각 K에서 축 A·B 전체를 측정. 주 비교 그림의 x축.

## 4. 측정 지표

`rules/scoring.md`에 정의. 요약:
- **Retrieval 층** (LLM 불필요): `Recall_all@K` (정답 tool 전부가 candidate에 포함된 쿼리 비율) — 주 지표. multi-tool split(I2/I3)에서 특히 중요.
- **Downstream 층**: `func_acc` (함수명 정확도), `arg_acc` (인자 정확도), multi-tool은 `completeness` (정답 tool set을 모두 호출한 비율).
- **진단**: multi-tool subset의 Recall_all, 저유사도 필수 tool 누락률.

## 5. 실행 순서 (milestone 게이트 기반)

**`MILESTONES.md`가 실행의 상위 통제 문서다.** 각 milestone은 자동 검증 + 사람 승인을 통과해야 다음으로 진행한다. 통과 못 하면 중단·보고. 임의 진행 금지.

| Milestone | rules 파일 | 내용 |
|---|---|---|
| M1 데이터 | `rules/data-prep.md` | ToolBench 로드, subset 샘플링, tool pool·split 구성 |
| M2 example | `rules/example-generation.md` | tool별 example 5개 생성·검증 (누출 방지) |
| M3 retrieval | `rules/retrieval.md` | 임베딩 사전계산, 축 B candidate 생성, fusion 계수 grid search |
| M4 classifier | `rules/classifier.md` | multi-label classifier 학습 (통합 train), real prior 생성 |
| M5 파일럿 | `rules/scoring.md` | 소규모(I1·K=10) downstream, 2B 파싱 검증 |
| M6 전체+분석 | `rules/scoring.md`, `rules/run-matrix.md` | 전체 조합 실행, 채점, 결과표·그림 |

각 단계는 시작 전 해당 rules 파일과 `MILESTONES.md`의 그 milestone 게이트를 읽고, 게이트 통과 후 진행한다.

## 6. 산출물 규격

```
experiment/
  PLAN.md
  MILESTONES.md            # 실행 게이트 (상위 통제)
  rules/*.md
  data/
    tools.jsonl            # tool pool 500개: id, name, description, params, category
    tools_examples.jsonl   # tool별 생성 example 5개
    queries_{I1,I2,I3}.jsonl  # 쿼리 300개씩: query, gold_tools, gold_categories
    embeddings/            # 사전계산 임베딩 (.npy)
    class_prior_real.jsonl # classifier 예측 (test 쿼리별 49-dim)
  models/
    class_classifier.pt    # 학습된 classifier + 하이퍼파라미터 json
  results/
    retrieval_{split}_{method}_{K}.jsonl   # candidate 목록 + Recall_all
    classifier_eval.json   # F1, per-category AUPRC, ECE, 분포 진단
    downstream_{split}_{model}_{condition}_{K}.jsonl  # tool-call 결과 + 채점
    summary.csv            # 최종 결과표 (아래 스키마)
  fusion_coeffs.json       # grid search로 선택된 α, β, λ, threshold
```

`summary.csv` 스키마: `split, model, candidate_scope, retrieval_method, prior_source, K, recall_all, func_acc, arg_acc, completeness, mean_prompt_tokens, n_queries`

## 7. 실험 무결성 규칙 (위반 시 결과 무효)

1. **정답 누출 금지**: 생성 example이 test 쿼리와 겹치면 안 됨 (`rules/example-generation.md`의 유사도 컷).
2. **조건 격리**: candidate 범위 외 모든 것(프롬프트, 파서, 디코딩 파라미터, seed)은 조건 간 동일.
3. **재현성**: 모든 무작위성에 seed=42. LLM 디코딩은 `temperature=0`.
4. **oracle 명시**: oracle 조건(oracle_tool, prior=oracle)은 상한 측정용이며 배포 불가임을 결과에 명기.
5. 애매하면 채우지 말고 중단·질문.
