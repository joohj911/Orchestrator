# CODING_NOTES.md — transformers 5.5.0 (v5) 작성 주의

서버 환경: `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`, `transformers==5.5.0`.
transformers v5는 v4와 **breaking change**가 있다. v4 예제를 그대로 복붙하면 깨진다.

## v5 주요 변경 (확인된 것)
- **huggingface_hub >= 1.0.0 pin.** HTTP 백엔드가 requests → httpx로 교체됨.
  - `requests.HTTPError`를 잡던 코드는 `httpx.HTTPError`로 바꿔야 함.
  - 스크립트에서 proxy 설정 불가 → 필요 시 `HTTP_PROXY`/`HTTPS_PROXY` 환경변수 사용.
- **weight loading API 변경**: 새 로딩 API 도입. `from_pretrained` 사용법이 대부분 유지되지만,
  device_map/dtype 인자나 저수준 로딩을 쓸 경우 v5 문서 기준으로 확인.
- `hf_transfer`/`HF_HUB_ENABLE_HF_TRANSFER` 제거, `hf_xet`로 대체 (대부분 투명).
- `typer-slim`이 필수 의존성으로 추가됨 (CLI용, 실험 코드엔 영향 적음).

## 이 실험에서 쓰는 API (v5 기준으로 검증할 것)
- 모델 로드: `AutoModelForCausalLM.from_pretrained(model_id, dtype=..., device_map=...)`.
  - **주의**: v5에서 `torch_dtype` 인자명이 `dtype`으로 바뀌었을 수 있음 → 실제 서명 확인 후 사용.
- 토크나이저: `AutoTokenizer.from_pretrained(model_id)`.
- **chat template + tools**: `tokenizer.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=True, tokenize=...)`.
  - Qwen3.5의 tool-call 포맷은 tokenizer의 chat_template에 내장. 이걸 따른다.
- 생성: `model.generate(..., do_sample=False, max_new_tokens=...)`. greedy 고정.
- 임베딩(e5): sentence-transformers 또는 transformers 직접. mean pooling + L2 normalize, prefix 규칙 준수.

## 확인 절차 (코드 작성 시)
1. 위 API의 **실제 시그니처를 설치된 transformers 5.5.0에서 확인**하고 쓴다 (추측 금지).
2. 인자명이 불확실하면 `# DECISION NEEDED:` 주석 + v5 문서 링크 근거를 남긴다.
3. Qwen3.5 tool-call 포맷은 모델카드/chat_template.jinja를 근거로 파싱 로직을 짠다.

## 파싱 (vLLM 파서 없음)
- vLLM의 `--tool-call-parser qwen3_coder`를 쓰지 않으므로, 생성 텍스트에서 tool_call을 **직접 파싱**한다.
- Qwen3.5 chat_template이 정의하는 tool_call 마커/포맷(예: `<tool_call>...</tool_call>` JSON)을 파싱.
- 파싱 실패(포맷 위반, JSON 깨짐)를 예외로 흘리지 말고 **명시적으로 집계** (M5 파싱 성공률 게이트의 근거).
- 구현 위치: `src/utils/qwen_tools.py`.
