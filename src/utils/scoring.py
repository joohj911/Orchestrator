"""채점 함수 (BFCL 스타일, 매칭 기반).

계약:
  score_func(called, gold, split) -> func_acc
  score_args(called, gold) -> arg_acc         # 정확일치→정규화→(옵션)LLM judge
  score_completeness(called_set, gold_set, candidate_set) -> (completeness, miss_type)
    # miss_type: 'retrieval_miss'(gold∉candidate) | 'generation_miss'(gold∈candidate,미호출) | None
  recall_all(candidate_ids, gold_ids) -> 0/1   # gold ⊆ candidate
  score_exact(called, gold) -> 0/1             # 호출 집합 == gold 집합 (여분 호출도 실패)
  validate_call(arguments, function_schema) -> dict  # 스키마 수준 실행 가능성 (BFCL AST 체크류)

구현: Claude Code. 상세 규칙 spec/rules/scoring.md.

식별자 규약: 모든 함수는 '함수명(문자열)' 집합/리스트를 비교한다. 호출측(m5/m6)은
gold tool id 와 모델이 호출한 함수명을 동일 변환(utils.qwen_tools.sanitize_name)으로
맞춰 넘겨야 한다. 이 모듈은 넘어온 문자열을 그대로 매칭한다.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

# multi-tool split (I2/I3). I1 은 single-tool.
_MULTI_TOOL_SPLITS = {"I2", "I3"}


def _names(calls: Iterable[Any]) -> list[str]:
    """호출/gold 입력을 함수명 리스트로 정규화.

    입력은 문자열 리스트 또는 {name, arguments} dict 리스트 둘 다 허용.
    """
    out: list[str] = []
    for c in calls:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict) and "name" in c:
            out.append(c["name"])
    return out


def recall_all(candidate_ids: Iterable[str], gold_ids: Iterable[str]) -> int:
    """gold ⊆ candidate 이면 1, 아니면 0 (Recall_all@K 의 쿼리 단위 값)."""
    return int(set(gold_ids).issubset(set(candidate_ids)))


def score_func(called: Iterable[Any], gold: Iterable[Any], split: str) -> float:
    """함수명 정확도.

    - I1 (single): gold tool 이 호출됐으면 1.0, 아니면 0.0.
    - I2/I3 (multi): precision = |호출∩gold| / |호출| (호출 중 정답 비율).
      호출이 없으면 0.0. set 일치 여부는 completeness 로 별도 측정.
    """
    called_names = _names(called)
    gold_names = set(_names(gold))

    if split not in _MULTI_TOOL_SPLITS:  # I1
        return float(any(n in gold_names for n in called_names))

    if not called_names:
        return 0.0
    correct = sum(1 for n in called_names if n in gold_names)
    return correct / len(called_names)


def score_exact(called: Iterable[Any], gold: Iterable[Any]) -> float:
    """엄격 함수 정확도: 호출 집합이 gold 집합과 정확히 일치해야 1.0.

    func_acc(I1: any-포함)와 달리 여분 호출(gold 외 tool 난사)도 실패로 본다.
    "후보가 적어 고르기 쉬워진다"는 효과를 엄격 기준으로 재는 보조 지표.
    호출측이 hallucinated call(후보에 없는 이름)을 별도 집계한다면 그 존재 시
    exact 도 0 으로 처리하는 것은 호출측 책임 (이 함수는 넘어온 집합만 비교).
    """
    called_names = set(_names(called))
    gold_names = set(_names(gold))
    return float(bool(called_names) and called_names == gold_names)


def _value_coercible(value: Any, json_type: str) -> bool:
    """값이 선언된 JSON Schema type 으로 해석 가능한가.

    Qwen XML 파서는 모든 인자 값을 문자열로 돌려주므로, 문자열이면 타입 강제
    (coercion) 가능 여부로 판정한다 ("5" 는 integer 로 유효).
    """
    if json_type == "string":
        return True
    if isinstance(value, str):
        s = value.strip()
        try:
            if json_type == "integer":
                int(s)
                return True
            if json_type == "number":
                float(s)
                return True
            if json_type == "boolean":
                return s.lower() in ("true", "false")
            if json_type == "array":
                return isinstance(json.loads(s), list)
            if json_type == "object":
                return isinstance(json.loads(s), dict)
        except (ValueError, json.JSONDecodeError):
            return False
        return True  # 알 수 없는 타입명 → 관대하게 통과
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


def validate_call(arguments: dict[str, Any] | None, function_schema: dict[str, Any]) -> dict[str, Any]:
    """스키마 수준 실행 가능성 검증 (gold 인자 값 불필요, BFCL AST 체크와 같은 발상).

    모델에게 보여준 OpenAI function 스키마 그대로에 대해:
      - required 파라미터가 전부 채워졌는가
      - 스키마에 없는 파라미터를 지어내지 않았는가
      - 값이 선언 타입으로 해석 가능한가
    를 판정한다. 값이 '의미적으로 정답'인지는 판정하지 않는다 (gold 인자 없음 —
    그건 arg_acc 의 영역, M6 결정 사항).

    반환: {valid, missing_required, unknown_params, bad_types}
    """
    fn = function_schema.get("function", function_schema)
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = params.get("required", []) or []
    args = arguments or {}

    missing = [r for r in required if r not in args]
    unknown = [k for k in args if k not in props]
    bad_types = [
        k for k, v in args.items()
        if k in props and not _value_coercible(v, str(props[k].get("type", "string")))
    ]
    return {
        "valid": not (missing or unknown or bad_types),
        "missing_required": missing,
        "unknown_params": unknown,
        "bad_types": bad_types,
    }


def _normalize_value(v: Any) -> str:
    """자유 텍스트 인자 정규화: 문자열화 → 소문자 → 공백 정리."""
    return " ".join(str(v).lower().split())


def _args_match(called_val: Any, gold_val: Any) -> bool:
    """단일 인자 값 일치 판정. 정확 일치 우선, 실패 시 정규화 후 비교."""
    if called_val == gold_val:
        return True
    return _normalize_value(called_val) == _normalize_value(gold_val)


def score_args(
    called: Iterable[dict[str, Any]],
    gold: Iterable[dict[str, Any]],
    use_llm_judge: bool = False,
) -> float:
    """인자 정확도.

    올바른 함수 호출(gold 함수명과 일치)에 대해, **스키마상 필수(required) 인자** 값이
    gold 와 일치하는 비율. = (정확히 채운 필수 인자) / (필수 인자), gold 호출 평균.
    (scoring.md: "필수 파라미터 값이 gold와 일치".)

    입력 구조:
      called: [{name, arguments: {k: v}}, ...]  (모델이 실제로 호출한 것)
      gold:   [{name, arguments: {k: v}, required: [param, ...]}, ...]
        - arguments: 정답 인자 값(옵션 인자 포함 가능).
        - required : 그 tool 의 스키마상 필수 파라미터명 목록. 채점 분모.
          m5/m6 가 tools.jsonl 의 params[].required 로부터 채워 넘긴다.

    채점 기준 (확정됨):
      - 분모 = required ∩ gold.arguments.keys()
        (스키마상 필수이면서 gold 가 정답 값을 준 인자만. gold 는 실행 가능한 정답이라
         보통 필수를 모두 담지만, 참조 값이 없는 필수는 검증 불가라 제외한다.)
      - 옵션 인자는 gold 가 값을 줬어도 채점하지 않는다(실행에 강제되지 않으므로).
      - required 키가 gold 에 없으면(= 호환용 폴백) gold.arguments 전체를 분모로 쓴다.
        이 경우 옵션까지 포함되니, 정식 실행에서는 required 를 반드시 넘길 것.

    # DECISION NEEDED (확정): 분모=스키마 required (사람 확인 완료, 옵션 제외).
    # DECISION NEEDED: LLM judge 는 기본 미사용(use_llm_judge=False), 사용 시 결과에 기록.
    #   채점 유틸은 외부 모델 의존을 두지 않는다(조건 격리). 애매한 자유텍스트 인자는
    #   정규화 비교까지만 수행한다.
    """
    if use_llm_judge:
        # 채점 유틸 자체는 LLM 을 호출하지 않는다. 필요 시 호출측에서 판정을 주입하고
        # 그 사용 사실을 산출물에 기록해야 한다.
        raise NotImplementedError(
            "LLM judge 는 이 유틸에서 직접 수행하지 않는다. 호출측에서 판정 후 기록할 것."
        )

    called_by_name: dict[str, dict[str, Any]] = {}
    for c in called:
        if isinstance(c, dict) and "name" in c:
            # 같은 함수가 여러 번 호출되면 첫 호출을 채점 대상으로 둔다.
            called_by_name.setdefault(c["name"], c.get("arguments", {}) or {})

    per_call_scores: list[float] = []
    for g in gold:
        gname = g["name"]
        gold_args = g.get("arguments", {}) or {}
        required = g.get("required")
        if required is None:
            # 폴백: required 정보 미제공 시 gold 인자 전체를 채점(스모크/구버전 호출).
            req_keys = list(gold_args.keys())
        else:
            # (b) 스키마 필수 ∩ gold 가 참조 값을 준 인자.
            req_keys = [k for k in required if k in gold_args]
        if not req_keys:
            # 채점할 필수 인자가 없음(분모 0) → 이 gold 호출은 arg 채점 대상 아님.
            continue
        called_args = called_by_name.get(gname)
        if called_args is None:
            # 해당 gold 함수를 호출하지 않음 → 필수 인자 전부 미충족(0점).
            per_call_scores.append(0.0)
            continue
        correct = sum(
            1 for k in req_keys if k in called_args and _args_match(called_args[k], gold_args[k])
        )
        per_call_scores.append(correct / len(req_keys))

    if not per_call_scores:
        return 0.0
    return sum(per_call_scores) / len(per_call_scores)


def score_completeness(
    called_set: Iterable[str],
    gold_set: Iterable[str],
    candidate_set: Iterable[str],
) -> tuple[float, str | None]:
    """multi-tool completeness + 미완 원인 분리.

    completeness = 1.0 if gold ⊆ called else 0.0.
    miss_type (완성 실패 시):
      - 'retrieval_miss'  : gold 중 candidate 에 없는 것이 있음 (recall_all=0).
      - 'generation_miss' : gold 는 candidate 에 있으나 호출되지 않음.
      완성(1.0)이면 None.

    원인 우선순위: candidate 에 gold 가 빠졌으면 retrieval_miss 를 우선한다
    (retrieval 이 상한을 이미 깎았으므로 generation 을 탓할 수 없다).
    """
    called = set(called_set)
    gold = set(gold_set)
    candidate = set(candidate_set)

    if gold.issubset(called):
        return 1.0, None
    if not gold.issubset(candidate):
        return 0.0, "retrieval_miss"
    return 0.0, "generation_miss"
