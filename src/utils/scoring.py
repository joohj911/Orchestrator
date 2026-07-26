"""채점 함수 (BFCL 스타일, 매칭 기반).

계약:
  score_func(called, gold, split) -> func_acc
  score_args(called, gold) -> arg_acc         # 정확일치→정규화→(옵션)LLM judge
  score_completeness(called_set, gold_set, candidate_set) -> (completeness, miss_type)
    # miss_type: 'retrieval_miss'(gold∉candidate) | 'generation_miss'(gold∈candidate,미호출) | None
  recall_all(candidate_ids, gold_ids) -> 0/1   # gold ⊆ candidate

구현: Claude Code. 상세 규칙 spec/rules/scoring.md.
"""
# TODO(Claude Code)
