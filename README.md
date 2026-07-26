# Candidate Tool Retrieval 선행 실험

Candidate tool 좁히기가 downstream tool-call 정확도에 유의미한지, 그리고 좁히는 방법(dense/multi-vector/class prior fusion) 간 우열을 측정하는 선행 실험.

## 실험 목적
1. **좁히기 유의미성**: 전체 tool을 다 주는 것 대비 좁힌 candidate만 주는 것이 정확도를 올리는가. 작은 모델(2B)에서 효과가 더 큰가.
2. **방법 비교**: dense_single / dense_multi / class prior fusion 중 무엇이 정답 tool을 candidate에 더 잘 담고 downstream 정확도를 올리는가.

핵심 가설: **작은 모델일수록 좁히기 효과가 크다.**

## 설계 요약
- 벤치마크: ToolBench (I1 single / I2 intra-category / I3 intra-collection)
- Downstream: `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-2B` (transformers `generate()`, vLLM 미사용)
- 임베딩: `intfloat/multilingual-e5-large`
- 상세 명세: `spec/` 디렉토리 (PLAN, MILESTONES, rules)

## 환경 준비

```bash
# Python 3.10+ 권장
pip install -r requirements.txt
```

서버 환경 고정 버전: torch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1 / transformers 5.5.0.
**transformers v5는 v4와 breaking change가 있음** — `spec/CODING_NOTES.md` 반드시 확인.

### 사전 요건
- ToolBench 데이터셋(G1/G2/G3)을 `config.yaml`의 `paths.toolbench_root`에 배치.
  (다운로드: OpenBMB/ToolBench. RapidAPI 키 필요 여부 확인.)
- GPU (9B 추론 가능한 메모리). `config.yaml`의 `hardware`에서 dtype/batch 조정.

## 실행

```bash
# config.yaml에서 paths.* 를 서버 환경에 맞게 수정한 뒤:
bash run_all.sh
```

- 파이프라인은 M1→M6 순서로 실행되며, **각 단계 뒤 검증(gate)이 실패하면 즉시 중단**된다.
- 중단 시: 출력된 원인을 확인 → 수정 → `bash run_all.sh` 재실행 (완료된 단계는 산출물 존재 시 건너뜀).

## 단계 개요 (상세는 spec/MILESTONES.md)

| 단계 | 스크립트 | 게이트 |
|---|---|---|
| M1 데이터 | `src/m1_data_prep.py` | gold 누락 0, description 결측 0 |
| M2 example | `src/m2_examples.py` | 누출 컷 (max_leak_sim ≤ 0.9) |
| M3 retrieval | `src/m3_retrieval.py` | Recall_all 검산, 계수 test 미사용 |
| M4 classifier | `src/m4_classifier.py` | test 누출 0 |
| M5 파일럿 | `src/m5_pilot.py` | 파싱 성공률 (미달 모델 자동 제외) |
| M6 전체+분석 | `src/m6_downstream.py`, `src/m6_analysis.py` | 조합 누락 0 |

## 산출물
```
outputs/
  data/       tools.jsonl, tools_examples.jsonl, queries_*.jsonl, embeddings/, class_prior_real.jsonl
  models/     class_classifier.pt
  results/    retrieval_*, classifier_eval.json, downstream_*, summary.csv, 표/그림
  fusion_coeffs.json
```

`results/summary.csv`가 최종 결과표. 컬럼: split, model, candidate_scope, retrieval_method, prior_source, K, recall_all, func_acc, arg_acc, completeness, mean_prompt_tokens, n_queries.

## 무결성 규칙 (위반 시 결과 무효)
- 정답 누출 금지: test 쿼리가 example 생성/classifier 학습/fusion 계수 선택에 새어들면 안 됨.
- seed=42, greedy 디코딩 고정.
- oracle 조건은 상한 측정용 — 배포 가능치로 해석 금지.
- 애매한 지점은 임의 진행 말고 중단·보고.
