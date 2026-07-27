"""prepare_toolbench_hf.py — HuggingFace 미러에서 ToolBench test 데이터 준비.

배경: 서버가 사내 DLP 프록시 뒤라 Google Drive/Tsinghua(클라우드 스토리지)가 막히고,
HuggingFace/GitHub 는 열려 있다. ToolBench 원본 test_instruction 과 동일한 내용이
HF 데이터셋 `tuandunghcmut/toolbench-v1` 의 `benchmark` config 에 있다.

이 스크립트는 그 config 를 받아 ToolBench `test_instruction/G{1,2,3}_instruction.json`
포맷으로 변환·저장한다. 그러면 m1_data_prep.py 가 그대로 읽는다.

HF benchmark splits (확인됨):
  g1_instruction(200) g1_category(200) g1_tool(200)
  g2_instruction(200) g2_category(200)
  g3_instruction(100)
각 row: {query_id, query, api_list(JSON 문자열), relevant_apis(JSON 문자열)}
  api_list 항목: category_name/tool_name/api_name/api_description/
                 required_parameters/optional_parameters
  relevant_apis: [[tool_name, api_name], ...]  (= gold)

기본 매핑: I{1,2,3} ← g{1,2,3}_instruction (ToolBench 표준 test 서브셋).
  --combine: 그룹 내 모든 서브셋을 합쳐 쿼리 수를 늘림(I1=600, I2=400, I3=100).
    (I3 는 서브셋이 하나라 100 이 상한. split 간 균형을 위해 보통 --combine 없이
     n_queries_per_split=100 로 맞추는 것을 권장.)

사용:
  python scripts/prepare_toolbench_hf.py [--dest ./data/toolbench] [--combine]
  이후: python src/m1_data_prep.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

HF_REPO = "tuandunghcmut/toolbench-v1"
HF_CONFIG = "benchmark"

# 기본: 표준 instruction 서브셋만.
INSTRUCTION_MAP = {
    "G1": ["g1_instruction"],
    "G2": ["g2_instruction"],
    "G3": ["g3_instruction"],
}
# --combine: 그룹 내 전 서브셋 합침.
COMBINE_MAP = {
    "G1": ["g1_instruction", "g1_category", "g1_tool"],
    "G2": ["g2_instruction", "g2_category"],
    "G3": ["g3_instruction"],
}


def _parse_json_field(v: Any) -> Any:
    """api_list/relevant_apis 가 JSON 문자열이면 파싱, 이미 객체면 그대로."""
    if isinstance(v, str):
        return json.loads(v)
    return v


def convert_row(row: dict[str, Any], subset: str) -> dict[str, Any]:
    """HF row → ToolBench test_instruction 인스턴스.

    query_id 는 서브셋을 접두어로 붙여 그룹 내 전역 고유성 보장(합칠 때 충돌 방지).
    """
    return {
        "query_id": f'{subset}::{row["query_id"]}',
        "query": row["query"],
        "api_list": _parse_json_field(row["api_list"]),
        "relevant APIs": _parse_json_field(row["relevant_apis"]),
    }


def distinct_gold_count(instance: dict[str, Any]) -> int:
    """gold (tool, api) 쌍의 distinct 개수 (multi-tool 판정용)."""
    pairs = set()
    for p in instance.get("relevant APIs", []):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pairs.add((p[0], p[1]))
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description="HF → ToolBench test_instruction 변환")
    ap.add_argument("--dest", default="./data/toolbench", help="toolbench_root (config 와 일치)")
    ap.add_argument("--combine", action="store_true", help="그룹 내 전 서브셋 합침")
    args = ap.parse_args()

    from datasets import load_dataset  # 지연 임포트 (테스트 시 datasets 불필요)

    print(f"[load] {HF_REPO} config={HF_CONFIG}")
    ds = load_dataset(HF_REPO, HF_CONFIG)

    out_dir = os.path.join(args.dest, "data", "test_instruction")
    os.makedirs(out_dir, exist_ok=True)

    mapping = COMBINE_MAP if args.combine else INSTRUCTION_MAP
    print(f"[mode] {'combine (그룹 내 전 서브셋)' if args.combine else 'instruction-only (표준)'}")
    print("--- 그룹별 변환 결과 (multi-tool = distinct gold ≥ 2) ---")
    summary = {}
    for group, subsets in mapping.items():
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for sub in subsets:
            if sub not in ds:
                print(f"  [경고] split '{sub}' 없음 — 건너뜀")
                continue
            for r in ds[sub]:
                o = convert_row(r, sub)
                if o["query_id"] in seen:
                    continue
                seen.add(o["query_id"])
                rows.append(o)
        path = os.path.join(out_dir, f"{group}_instruction.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        multi = sum(1 for o in rows if distinct_gold_count(o) >= 2)
        summary[group] = (len(rows), multi)
        print(f"  {group}: total {len(rows)}, multi-tool(gold≥2) {multi}  -> {path}")

    print("\n--- n_queries_per_split 설정 참고 ---")
    print("  I1(G1) single-tool: 전체 사용 가능")
    print("  I2(G2)/I3(G3) multi-tool: 위 multi-tool 개수가 상한")
    i2_multi = summary.get("G2", (0, 0))[1]
    i3_multi = summary.get("G3", (0, 0))[1]
    max_uniform = min(summary.get("G1", (0, 0))[0], i2_multi, i3_multi)
    print(f"  → 세 split 균형(uniform) 상한 ≈ {max_uniform}")
    print(f"    config.yaml 의 experiment.n_queries_per_split 를 이 값 이하로 설정 후 M1 실행.")


if __name__ == "__main__":
    main()
