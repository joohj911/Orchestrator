"""flag_tool_quality.py — pool 품질 진단 (DIAGNOSTIC ONLY).

목적: ToolBench pool 에 섞인 test/junk RapidAPI 항목을 휴리스틱으로 표시해 **인지·기록**한다.
      결과는 M6 한계(limitations)·dependency 진단 보고에만 쓴다.

!!! 중요: 이 산출물(tool_quality.jsonl)은 실험 파이프라인(m3 retrieval / m4 classifier /
    m5·m6 downstream·scoring)에서 **절대 사용하지 않는다.** pool 500 은 M1 에서 확정된
    고정 자산이며, 여기서 tool 을 제외하거나 가중치를 바꾸지 않는다. 순수 보고용 라벨이다.

휴리스틱(과/소 포함될 수 있음, 참고용):
  - clearly_junk: 이름 패턴(test/asdf/hello world/petstore/reqres/swagger/…), hex 이름,
    gibberish description → 명백한 테스트/placeholder API.
  - missing_meta: desc_fallback(원본 description 결측) 또는 매우 짧은 description.
    (주의: missing_meta 는 '쓰레기'가 아닐 수 있음 — 진짜 API 인데 메타만 없을 수 있음.)

사용: python scripts/flag_tool_quality.py --config config.yaml
출력: {output_dir}/data/tool_quality.jsonl  {tool_id, clearly_junk, missing_meta, reasons}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.config import load_config  # noqa: E402

_JUNK_NAME = re.compile(
    r"(?i)(^|[^a-z])(test|asdf|asd|qwerty|foo|bar|baz|hello ?world|petstore|reqres|swagger|"
    r"demo|sample|example|dont ?change|should ?be ?free|untitled|placeholder|tmp|xxx|todo|"
    r"myapi|new ?api|route-precedence)([^a-z]|$)"
)
_HEX_TOOL = re.compile(r"^[0-9a-f]{20,}$")


def _tool_segment(tool_id: str) -> str:
    parts = tool_id.split("__")
    return parts[1] if len(parts) >= 3 else tool_id


def classify(rec: dict) -> dict:
    tool = _tool_segment(rec["id"])
    name = str(rec.get("name", ""))
    desc = str(rec.get("description", "")).strip()
    reasons: list[str] = []

    if _JUNK_NAME.search(f"{tool} {name}"):
        reasons.append("name_pattern")
    if _HEX_TOOL.match(tool.replace(" ", "")):
        reasons.append("hex_tool")
    if desc and re.fullmatch(r"[a-zA-Z]{3,8}", desc) and not re.search(r"(?i)[aeiou].*[aeiou]", desc):
        reasons.append("gibberish_desc")

    missing = []
    if rec.get("desc_fallback"):
        missing.append("desc_fallback")
    if len(desc) < 12:
        missing.append("short_desc")

    clearly_junk = bool(reasons)
    return {
        "tool_id": rec["id"],
        "clearly_junk": clearly_junk,
        "missing_meta": bool(missing) and not clearly_junk,
        "reasons": reasons + missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="pool 품질 진단 (실험 미사용, 보고용)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, make_dirs=False)
    data_dir = cfg["paths"]["data_dir"]
    tools_path = os.path.join(data_dir, "tools.jsonl")
    if not os.path.isfile(tools_path):
        print(f"tools.jsonl 없음: {tools_path} (M1 먼저)", file=sys.stderr)
        sys.exit(1)

    with open(tools_path, "r", encoding="utf-8") as f:
        tools = [json.loads(l) for l in f if l.strip()]

    out = [classify(t) for t in tools]
    out_path = os.path.join(data_dir, "tool_quality.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_junk = sum(1 for r in out if r["clearly_junk"])
    n_miss = sum(1 for r in out if r["missing_meta"])
    reason_counts = Counter(x for r in out for x in r["reasons"])
    print(f"[flag_tool_quality] pool {len(tools)} tools → {out_path}")
    print(f"  clearly_junk (test/placeholder): {n_junk}")
    print(f"  missing_meta only (진짜 API 포함 가능): {n_miss}")
    print(f"  사유별: {dict(reason_counts)}")
    print("  주의: 이 라벨은 실험에 미사용. M6 한계 보고용.")


if __name__ == "__main__":
    main()
