"""채점 함수 (BFCL 스타일, 매칭 기반).

계약:
  score_func(called, gold, split) -> func_acc
  score_args(called, gold) -> arg_acc         # 정확일치→정규화→(옵션)LLM judge
  score_completeness(called_set, gold_set, candidate_set) -> (completeness, miss_type)
    # miss_type: 'retrieval_miss'(gold∉candidate) | 'generation_miss'(gold∈candidate,미호출) | None
  recall_all(candidate_ids, gold_ids) -> 0/1   # gold ⊆ candidate

구현: Claude Code. 상세 규칙 spec/rules/scoring.md.

식별자 규약: 모든 함수는 '함수명(문자열)' 집합/리스트를 비교한다. 호출측(m5/m6)은
gold tool id 와 모델이 호출한 함수명을 동일 변환(utils.qwen_tools.sanitize_name)으로
맞춰 넘겨야 한다. 이 모듈은 넘어온 문자열을 그대로 매칭한다.
"""
from __future__ import annotations

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

    올바른 함수 호출(gold 함수명과 일치)에 대해, gold 가 명시한 필수 인자 값이
    호출 인자와 일치하는 비율. = (정확히 채운 필수 인자) / (필수 인자), gold 호출 평균.

    입력: called/gold 는 [{name, arguments: {k: v}}, ...].
    gold 의 arguments 키를 '필수 인자'로 본다 (ToolBench gold_api 의 정답 인자).

    # DECISION NEEDED: gold.arguments 의 키 전체를 필수 인자로 간주한다.
    #   근거: ToolBench gold 는 정답 호출의 인자 값을 담는다. 별도 required 플래그가
    #   gold 레코드에 없으므로 "정답이 값을 준 인자 = 채워야 할 필수 인자"로 본다.
    #   tools.jsonl 의 스키마 required 와 다를 수 있어, 채점 기준은 gold 값 기준으로 고정.
    #
    # DECISION NEEDED: LLM judge 는 기본 미사용(use_llm_judge=False)이고, 사용 시
    #   그 사실을 결과에 기록한다. 채점 유틸은 외부 모델 의존을 두지 않는다(조건 격리).
    #   애매한 자유텍스트 인자는 정규화 비교까지만 수행한다.
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
        if not gold_args:
            # 필수 인자가 없으면 인자 채점 대상이 아님(분모 0) → 건너뜀.
            continue
        called_args = called_by_name.get(gname)
        if called_args is None:
            # 해당 gold 함수를 호출하지 않음 → 필수 인자 전부 미충족(0점).
            per_call_scores.append(0.0)
            continue
        correct = sum(
            1 for k, gv in gold_args.items() if k in called_args and _args_match(called_args[k], gv)
        )
        per_call_scores.append(correct / len(gold_args))

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
