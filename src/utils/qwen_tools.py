"""Qwen3.5 tool-call 프롬프트 구성 + 파싱 (vLLM 파서 미사용).

계약:
  build_prompt(tokenizer, query, tool_schemas, system_prompt=None) -> str
    # tokenizer.apply_chat_template(messages, tools=tool_schemas,
    #                               add_generation_prompt=True, tokenize=False)
    # system_prompt: config.prompt.system_instruction (전 조건·전 모델 공통 고정).
  parse_tool_calls(generated_text) -> (calls: list[{name, arguments}], parse_ok: bool)
    # Qwen3.5 chat_template 의 tool_call 포맷을 직접 파싱.
    # 파싱 실패(포맷 위반/JSON 깨짐)는 예외로 흘리지 말고 parse_ok=False 로 반환.

주의:
  - vLLM 의 --tool-call-parser qwen3_coder 없음 → 직접 파싱.
  - 파싱 성공/실패는 M5 게이트 근거이므로 반드시 명시 반환.
  - Qwen3.5 chat_template.jinja / 모델카드 포맷을 근거로 구현.
구현: Claude Code

# 포맷 확인 (2026-07, 웹 확인 완료):
#   Qwen3.5 는 Qwen3(Hermes JSON)와 tool-call 포맷이 다르다. Qwen3.5 는 Qwen3-Coder
#   XML 포맷으로 학습됐다 (parser: qwen3_coder / qwen3_xml). 실제 생성 문자열:
#       <tool_call>
#       <function=func_name>
#       <parameter=key>
#       value
#       </parameter>
#       </function>
#       </tool_call>
#   따라서 이 실험(Qwen3.5-9B/2B)의 1차 포맷은 XML 이다. CODING_NOTES 의 JSON 예시는
#   Qwen3 기준이었다.
#   - 파서는 XML 을 우선 시도하고, Hermes JSON 을 호환 폴백으로 함께 처리한다.
#     (긴 컨텍스트(>~65K)에서 XML/JSON 이 섞여 나오는 알려진 이슈가 있어 이중 파싱이
#      단순 방어가 아니라 견고성 이득이다.)
#   - 최종 근거는 설치된 Qwen3.5 tokenizer 의 chat_template.jinja 이며, 서버에서
#     M5 파싱 성공률로 실측 검증된다.
#   근거: QwenLM/Qwen3-Coder tool-call 포맷 문서, vLLM qwen3_coder/qwen3_xml 파서,
#         Qwen3.5 chat-template-fix 논의.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Qwen tool_call 블록 마커. chat_template 이 생성 텍스트에 삽입하는 형태.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# 폴백(qwen3_coder XML): <function=name> ... <parameter=key>value</parameter> ... </function>
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def build_prompt(
    tokenizer,
    query: str,
    tool_schemas: list[dict[str, Any]],
    system_prompt: str | None = None,
):
    """chat template 로 프롬프트 문자열을 만든다 (tokenize=False).

    조건 간 동일성(무결성 규칙 2)을 위해 바뀌는 것은 tool_schemas 뿐이다.

    system_prompt:
      config.prompt.system_instruction 값을 그대로 넘긴다. 전 조건(full/random/
      retrieved/oracle)·전 모델(2B/9B)에 **동일하게** 적용되는 고정 문구여야 한다.
      호출측(m5/m6)이 조건마다 다르게 주면 조건 격리가 깨지므로 금지.
      None/빈 문자열이면 system 메시지를 넣지 않고 chat_template 의 기본 tool 지시만
      사용한다.
      (tool 사용 지시 자체는 chat_template 이 tools= 로부터도 생성하지만, 특히 약한
       2B 가 호출 대신 직접 답하는 no_call 을 줄이려면 명시적 중립 지시가 도움이 된다.
       M5 파싱/no_call 리포트가 이 문구의 효과를 실측 검증한다.)

    반환: 렌더된 프롬프트 문자열. 배치/토크나이즈는 호출측(m5/m6)이 담당해
    padding·디코딩 파라미터를 조건 간 동일하게 통제한다.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    return tokenizer.apply_chat_template(
        messages,
        tools=tool_schemas,
        add_generation_prompt=True,
        tokenize=False,
    )


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """arguments 필드를 dict 로 정규화. 문자열이면 JSON 파싱 시도."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except json.JSONDecodeError:
            return {"_value": raw}
    if raw is None:
        return {}
    return {"_value": raw}


def _parse_json_block(block: str) -> dict[str, Any] | None:
    """<tool_call> 안의 JSON 블록을 {name, arguments} 로 파싱. 실패 시 None."""
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return {"name": obj["name"], "arguments": _coerce_arguments(obj.get("arguments", {}))}


def _parse_xml_block(block: str) -> dict[str, Any] | None:
    """폴백: <function=..><parameter=..> 포맷 파싱. 실패 시 None."""
    fmatch = _FUNCTION_RE.search(block)
    if not fmatch:
        return None
    name = fmatch.group(1)
    args: dict[str, Any] = {}
    for pname, pval in _PARAMETER_RE.findall(fmatch.group(2)):
        args[pname] = pval.strip()
    return {"name": name, "arguments": args}


def classify_generation(generated_text: str) -> dict[str, Any]:
    """생성 텍스트를 파싱해 상태를 분류한다 (parse_tool_calls 의 내부 구현).

    반환 status:
      'ok'        : tool_call 마커가 있고 전부 정상 파싱됨.
      'malformed' : tool_call 마커는 있으나 하나 이상 파싱 실패(JSON 깨짐/포맷 위반).
      'no_call'   : tool_call 마커가 없음(모델이 호출을 내지 않음).

    M5 게이트가 세 상태를 구분해 집계할 수 있게 별도 함수로 노출한다.
    """
    blocks = _TOOL_CALL_RE.findall(generated_text)

    if not blocks:
        # 마커가 아예 없음 → 폴백 XML 이 tool_call 래핑 없이 나온 경우도 확인.
        if _FUNCTION_RE.search(generated_text):
            call = _parse_xml_block(generated_text)
            if call is not None:
                return {"status": "ok", "calls": [call]}
            return {"status": "malformed", "calls": []}
        return {"status": "no_call", "calls": []}

    calls: list[dict[str, Any]] = []
    any_malformed = False
    for block in blocks:
        # Qwen3.5 = Qwen3-Coder XML 이 학습 포맷 → 우선 시도.
        call = _parse_xml_block(block)
        if call is None:
            call = _parse_json_block(block)  # Hermes JSON 호환 폴백.
        if call is None:
            any_malformed = True
            continue
        calls.append(call)

    if any_malformed or not calls:
        # 마커는 있으나 하나라도 깨졌거나, 전부 깨져 유효 호출이 없음.
        return {"status": "malformed", "calls": calls}
    return {"status": "ok", "calls": calls}


def parse_tool_calls(generated_text: str) -> tuple[list[dict[str, Any]], bool]:
    """계약 함수. (calls, parse_ok) 반환.

    # DECISION NEEDED: parse_ok = (status == 'ok').
    #   근거: M5 '파싱 성공률' 게이트는 "모델이 tool-call 포맷을 안정적으로 따르는가"
    #   를 측정한다. 따라서 유효한 구조적 tool_call 을 얻은 경우만 성공(True)으로 본다.
    #   - malformed(포맷 있으나 깨짐)와 no_call(포맷 없음) 모두 parse_ok=False.
    #   - 둘의 구분이 필요하면 classify_generation() 의 status 를 별도 집계한다.
    """
    result = classify_generation(generated_text)
    return result["calls"], result["status"] == "ok"


def sanitize_name(tool_id: str) -> str:
    """tool id 를 OpenAI function-name 규칙(^[A-Za-z0-9_-]+$)에 맞게 정규화.

    downstream 채점은 함수명 매칭이므로, 스키마 생성과 gold 매핑에서 동일 변환을
    쓰도록 여기 한 곳에 둔다.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", tool_id)


def to_openai_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """tools.jsonl 의 tool 레코드를 OpenAI function-calling 스키마로 변환.

    apply_chat_template(tools=...) 에 그대로 넘길 수 있는 형태.
    함수명은 sanitize_name(tool['id']) 로 고정해 채점과 일관되게 매핑한다.

    tool params 스키마: [{name, type, description, required}, ...] (data-prep.md).
    """
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in tool.get("params", []) or []:
        pname = p.get("name")
        if not pname:
            continue
        # JSON Schema type 매핑. 알 수 없으면 string 으로.
        ptype = str(p.get("type", "string")).lower()
        json_type = {
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "double": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
        }.get(ptype, "string")
        props[pname] = {"type": json_type, "description": p.get("description", "") or ""}
        if p.get("required"):
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": sanitize_name(tool["id"]),
            "description": tool.get("description", "") or "",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }
