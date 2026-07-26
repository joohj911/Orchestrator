# CLAUDE.md — 코드 작성 지침 (Claude Code 전용)

너의 역할: 이 레포의 **실행 코드를 작성**하는 것. 실험을 직접 돌리는 게 아니다 (데이터·GPU는 별도 서버에 있고, 서버에서 `bash run_all.sh`로 실행된다).

## 작성 대상
아래 스텁 파일들을 **실제로 동작하는 Python 코드로 채운다**. 구현 명세는 각 스텁 상단 docstring과 `spec/rules/*.md`에 있다.

```
src/
  m1_data_prep.py        ← spec/rules/data-prep.md
  m2_examples.py         ← spec/rules/example-generation.md (검증 전용, 생성 아님)
  m3_retrieval.py        ← spec/rules/retrieval.md
  m4_classifier.py       ← spec/rules/classifier.md
  m5_pilot.py            ← spec/rules/scoring.md (파일럿 부분)
  m6_downstream.py       ← spec/rules/scoring.md (전체)
  m6_analysis.py         ← spec/rules/run-matrix.md
  verify/verify_m{1..6}.py   ← 각 milestone 게이트 (spec/MILESTONES.md)
  utils/*.py             ← 공용 (임베딩, 파싱, 채점, config 로더)
```

## 추가 작업 — example 직접 작성 (코드 아님, 데이터 산출물)
`data/tools_examples.jsonl`은 **네가 직접 작성**해 레포에 커밋한다 (런타임 LLM 생성 아님).
1. 먼저 `m1_data_prep.py`를 구현하고 **실제로 실행**해 `data/tools.jsonl`(500개 tool)을 생성한다.
   (ToolBench 데이터 접근이 되는 환경 전제. 안 되면 중단·보고.)
2. 각 tool의 name/description/params를 보고 자연스러운 사용자 발화 5개를 작성한다.
   - 규칙은 `spec/rules/example-generation.md` (API명 노출 금지, 영어, 다양성, **test 쿼리 미참조**).
3. `data/tools_examples.jsonl`로 저장하고 커밋한다.
- 서버 런타임의 `m2_examples.py`는 이 파일을 **검증만** 한다 (누출 컷·커버리지). 생성 로직을 m2에 넣지 마라.

## 절대 규칙
1. **transformers 5.5.0 (v5)** 로 작성. v4 관용구 금지. `spec/CODING_NOTES.md` 먼저 읽어라.
2. **vLLM 사용 금지.** downstream 추론은 transformers `generate()` + `apply_chat_template(tools=...)`.
3. **정답 누출 금지**: test 쿼리가 example 생성/classifier 학습/fusion 계수 선택에 들어가면 안 됨. 코드에서 명시적으로 분리·검증.
4. **config.yaml에서 모든 파라미터를 읽는다.** 하드코딩 금지 (경로·모델·seed·K 전부 config).
5. **seed=42, greedy 디코딩** 고정.
6. 각 스크립트는 `--config` 인자를 받고, 산출물을 `spec` 명세의 스키마·경로대로 저장한다.
7. **검증 스크립트(verify_m*)는 실패 시 exit code 1**로 종료 (run_all.sh가 이걸로 중단).
8. 애매하거나 명세에 없는 결정이 필요하면 **임의로 정하지 말고 주석 `# DECISION NEEDED:`로 표시**하고 합리적 기본값 + 근거를 남긴다.

## 구현 순서 (권장)
1. `utils/` 먼저 — config 로더, 임베딩 래퍼, Qwen chat template + tool-call 파서, 채점 함수. 나머지가 여기 의존.
2. m1 → verify_m1 → m2 → ... 순서로, 각 단계와 그 검증을 짝지어 작성.
3. `spec/rules/*.md`의 "완료 조건"이 곧 verify_m*의 검사 항목이다.

## 특히 주의 (실패 잦은 지점)
- **Qwen tool-call 파싱**: vLLM 파서가 없으므로 chat template로 tools를 주입하고, 생성 텍스트에서 tool_call을 직접 파싱해야 한다. `utils/qwen_tools.py`에 집중 구현. 파싱 성공/실패를 반드시 별도 집계 (M5 게이트 근거).
- **e5 prefix**: query엔 `"query: "`, 문서엔 `"passage: "` prefix. 빠뜨리면 성능 급락.
- **transformers v5 로딩 API**: 모델 로드 방식이 v4와 다를 수 있음. CODING_NOTES 확인.
- **재실행 안전성**: 산출물이 이미 있으면 재계산 건너뛰거나 `--force`로 덮어쓰기. 임베딩·생성은 비싸니 캐싱.

## 테스트
- 데이터·GPU 없이 로직을 검증할 수 있게, 각 스크립트에 소량 dummy 입력으로 도는 `--smoke` 모드나 단위 테스트를 넣으면 좋다 (선택).
- 실제 실행 검증은 서버에서 이뤄지므로, 코드가 서버에서 처음 돌 때 깨지지 않도록 방어적으로 작성 (에러 메시지 명확히).
