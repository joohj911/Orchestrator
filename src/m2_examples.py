"""m2_examples.py

명세: spec/rules/example-generation.md
역할: **검증 전용** (생성 아님).
  - example(data/tools_examples.jsonl)은 Claude Code가 코드 작성 단계에서 직접 작성해 레포에 커밋한다.
  - 이 스크립트는 서버 런타임에서 그 example을 로드하여 누출 컷만 검사한다.

동작:
  1. data/tools.jsonl, data/tools_examples.jsonl 로드.
  2. 각 example을 e5로 임베딩 (utils/embed.embed_queries).
  3. 그 tool을 gold로 갖는 모든 test 쿼리와 cosine 유사도 계산.
  4. 각 example의 max_leak_sim 기록. > config.leak_sim_threshold(0.9)면 위반으로 표시.
  5. 커버리지 검사: 모든 tool이 정확히 5개 example을 갖는지.
  6. 결과를 data/tools_examples_checked.jsonl (max_leak_sim 채워서)로 저장.
     실제 통과/실패 판정은 verify_m2.py가 수행.

CLI: python m2_examples.py --config config.yaml
규칙:
  - 여기서 example을 생성하지 않는다. 없으면 에러로 중단하고 "Claude Code가 작성해야 함" 안내.
  - seed=42.
구현: Claude Code.

주: example 은 사용자 발화(=query)이므로 test 쿼리와 동일하게 "query: " prefix 로 임베딩한다
    (utils.embed.embed_queries). 누출 = "내가 쓴 example 이 실제 test 쿼리와 과도하게
    유사한가"를 그 tool 을 gold 로 갖는 test 쿼리에 한해 측정한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402

N_EXAMPLES_EXPECTED_KEY = ("experiment", "n_examples_per_tool")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_max_leak_sims(
    example_vecs, testq_vecs, ex_index: list[int], gold_rows_per_example: list[list[int]]
):
    """각 example 의 max cosine 유사도를 그 tool 의 gold test 쿼리 집합에 대해 계산.

    벡터는 L2 정규화돼 있으므로 cosine = dot product.
    Args:
      example_vecs: (E, d) 전체 example 임베딩.
      testq_vecs:   (Q, d) 고유 test 쿼리 임베딩.
      ex_index:     길이 E, 각 example 의 example_vecs 행 인덱스(= 0..E-1, 항등).
      gold_rows_per_example: 길이 E, 각 example 에 대해 비교할 test 쿼리 행 인덱스 목록.
    Returns:
      길이 E 의 float 리스트 (해당 gold test 쿼리 없으면 0.0).
    """
    import numpy as np

    out: list[float] = []
    for e in range(len(ex_index)):
        rows = gold_rows_per_example[e]
        if not rows:
            out.append(0.0)
            continue
        sims = testq_vecs[rows] @ example_vecs[ex_index[e]]
        out.append(float(np.max(sims)))
    return out


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    data_dir = cfg["paths"]["data_dir"]
    splits = cfg["experiment"]["splits"]

    tools_path = os.path.join(data_dir, "tools.jsonl")
    out_path = os.path.join(data_dir, "tools_examples_checked.jsonl")

    # example 은 커밋된 자산(paths.examples_file, 기본 ./data/tools_examples.jsonl).
    # 하위호환: 지정 경로에 없으면 data_dir 에서도 찾아본다.
    examples_path = cfg["paths"].get("examples_file", "")
    if not (examples_path and os.path.isfile(examples_path)):
        fallback = os.path.join(data_dir, "tools_examples.jsonl")
        examples_path = fallback if os.path.isfile(fallback) else examples_path

    if not os.path.isfile(tools_path):
        print(f"[m2] tools.jsonl 없음: {tools_path}. 먼저 M1 실행.", file=sys.stderr)
        sys.exit(1)
    if not (examples_path and os.path.isfile(examples_path)):
        print(
            f"[m2] tools_examples.jsonl 없음: {examples_path}.\n"
            f"     이 파일은 런타임 생성물이 아니라 Claude Code 가 작성해 레포에 커밋해야 함.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[m2] example 로드: {examples_path}")

    tools = _read_jsonl(tools_path)
    tool_ids = {t["id"] for t in tools}
    examples = _read_jsonl(examples_path)

    # tool_id -> examples 매핑 (커버리지/순서 확인은 verify 가 담당, 여기선 계산).
    ex_by_tool: dict[str, list[str]] = {e["tool_id"]: list(e.get("examples", [])) for e in examples}

    # 그 tool 을 gold 로 갖는 test 쿼리 텍스트 수집.
    tool_to_queries: dict[str, list[str]] = defaultdict(list)
    for split in splits:
        qpath = os.path.join(data_dir, f"queries_{split}.jsonl")
        if not os.path.isfile(qpath):
            print(f"[m2] {qpath} 없음. M1 산출물 확인.", file=sys.stderr)
            sys.exit(1)
        for q in _read_jsonl(qpath):
            for gt in q.get("gold_tools", []):
                tool_to_queries[gt].append(q["query"])

    # 고유 test 쿼리 목록 + 인덱스.
    uniq_queries: list[str] = []
    q_index: dict[str, int] = {}
    for qs in tool_to_queries.values():
        for qtext in qs:
            if qtext not in q_index:
                q_index[qtext] = len(uniq_queries)
                uniq_queries.append(qtext)

    # 전체 example 평탄화 + 각 example 의 비교 대상 test 쿼리 행.
    flat_examples: list[str] = []
    flat_tool: list[str] = []
    gold_rows_per_example: list[list[int]] = []
    for e in examples:
        tid = e["tool_id"]
        for ex in e.get("examples", []):
            flat_examples.append(ex)
            flat_tool.append(tid)
            gold_rows_per_example.append([q_index[q] for q in tool_to_queries.get(tid, [])])

    # 임베딩 (query prefix). example 이 없으면 빈 처리.
    from utils.embed import embed_queries

    print(f"[m2] 임베딩: example {len(flat_examples)}개, 고유 test 쿼리 {len(uniq_queries)}개")
    ex_vecs = embed_queries(flat_examples, cfg) if flat_examples else None
    testq_vecs = embed_queries(uniq_queries, cfg) if uniq_queries else None

    if testq_vecs is None:
        max_sims = [0.0] * len(flat_examples)
    else:
        max_sims = compute_max_leak_sims(
            ex_vecs, testq_vecs, list(range(len(flat_examples))), gold_rows_per_example
        )

    # example 단위 max_leak_sim 을 tool 단위로 재조립.
    threshold = float(cfg["experiment"]["leak_sim_threshold"])
    checked: list[dict[str, Any]] = []
    cursor = 0
    n_violations = 0
    for e in examples:
        tid = e["tool_id"]
        exs = list(e.get("examples", []))
        sims = max_sims[cursor : cursor + len(exs)]
        cursor += len(exs)
        viol = [s > threshold for s in sims]
        n_violations += sum(viol)
        checked.append(
            {
                "tool_id": tid,
                "examples": exs,
                "max_leak_sim": [round(s, 4) for s in sims],
                "leak_ok": not any(viol),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in checked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 요약 (판정은 verify_m2).
    missing = sorted(tool_ids - set(ex_by_tool))
    n_expected = cfg["experiment"]["n_examples_per_tool"]
    wrong_count = [t for t, exs in ex_by_tool.items() if len(exs) != n_expected]
    print(f"[m2] 완료 → {out_path}")
    print(f"     example 총 {len(flat_examples)}개, 누출 위반(>{threshold}) {n_violations}개")
    print(f"     커버리지 미달 tool(!= {n_expected}개): {len(wrong_count)}, tools.jsonl 누락: {len(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 example 누출 검증 (생성 아님)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
