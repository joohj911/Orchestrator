"""Qwen3.5 tool-call 프롬프트 구성 + 파싱 (vLLM 파서 미사용).

계약:
  build_prompt(tokenizer, query, tool_schemas) -> str/inputs
    # tokenizer.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=True)
  parse_tool_calls(generated_text) -> (calls: list[{name, arguments}], parse_ok: bool)
    # Qwen3.5 chat_template의 tool_call 포맷(예: <tool_call>{json}</tool_call>) 파싱.
    # 파싱 실패(포맷 위반/JSON 깨짐)는 예외로 흘리지 말고 parse_ok=False로 반환.

주의:
  - vLLM의 --tool-call-parser qwen3_coder 없음 → 직접 파싱.
  - 파싱 성공/실패는 M5 게이트 근거이므로 반드시 명시 반환.
  - Qwen3.5 chat_template.jinja / 모델카드 포맷을 근거로 구현.
구현: Claude Code
"""
# TODO(Claude Code)
