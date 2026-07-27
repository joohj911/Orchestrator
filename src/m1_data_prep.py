"""m1_data_prep.py

명세: spec/rules/data-prep.md
역할: ToolBench 로드, subset 샘플링, tool pool 500 구성
산출물: data/tools.jsonl, data/queries_{I1,I2,I3}.jsonl

CLI: python m1_data_prep.py --config config.yaml [--force] [--smoke]
  --config: config.yaml 경로
  --force : 산출물이 있어도 재생성
  --smoke : 합성 소량 입력으로 로직만 점검 (데이터/GPU 없이)

규칙:
  - config에서 모든 파라미터 로드 (하드코딩 금지).
  - seed 고정(config.seed). 정답 누출과 무관(순수 데이터 구성 단계).
  - 산출물 스키마·경로는 명세 준수.
  - 애매하면 임의 결정 말고 '# DECISION NEEDED:' 표시 + 근거.
구현: Claude Code.

데이터 출처 판단 (data-prep.md 절차 준수):
  ToolBench test_instruction 의 각 쿼리 인스턴스는 `api_list`(후보 API 전체 메타:
  category_name/tool_name/api_name/description/required_parameters/optional_parameters)
  와 `relevant APIs`(gold = [tool_name, api_name] 목록)를 담는다. 따라서 tool 메타와
  gold 를 이 파일들만으로 구성할 수 있어 RapidAPI 키·toolenv 전량 스캔이 불필요하다.
  (이 실험 채점은 실행이 아니라 매칭 기반 — scoring.md.)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any

# 프로젝트 루트를 import 경로에 추가 (src 를 패키지로 사용).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402

# --- 벤치마크 사실 상수 (튜닝 파라미터 아님, data-prep.md 정의) ---
# split → ToolBench 그룹. I1=G1(single), I2=G2(intra-category), I3=G3(intra-collection).
_SPLIT_TO_GROUP = {"I1": "G1", "I2": "G2", "I3": "G3"}
# single-tool split. 나머지(I2,I3)는 multi-tool → gold ≥ 2 요구.
_SINGLE_TOOL_SPLITS = {"I1"}

# DECISION NEEDED: distractor 의 category 당 상한 비율.
#   근거: data-prep.md "category당 상한을 두어 한 category 독점 방지". config 에 없는
#   값이라 기본 0.20 (한 category 가 distractor 의 20% 초과 못 하게)으로 둔다. gold
#   category 분포가 극단적으로 쏠려도 pool 이 한 category 로 지배되지 않게 하는 완충.
_DISTRACTOR_CAP_FRAC = 0.20


def _norm_name(s: str) -> str:
    """gold ↔ api_list 매칭용 정규화 (소문자, 영숫자만)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _make_id(category: str, tool: str, api: str) -> str:
    """충돌 방지용 tool id: category__tool__api (data-prep.md 스키마)."""
    return f"{category}__{tool}__{api}"


def _extract_params(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """api_list 항목에서 params 정규화. required/optional 를 required 플래그로 통합."""
    params: list[dict[str, Any]] = []
    for p in entry.get("required_parameters", []) or []:
        params.append(
            {
                "name": p.get("name", ""),
                "type": p.get("type", ""),
                "description": p.get("description", "") or "",
                "required": True,
            }
        )
    for p in entry.get("optional_parameters", []) or []:
        params.append(
            {
                "name": p.get("name", ""),
                "type": p.get("type", ""),
                "description": p.get("description", "") or "",
                "required": False,
            }
        )
    return params


def _entry_to_meta(entry: dict[str, Any]) -> dict[str, Any] | None:
    """api_list 항목 → tool 메타 레코드. 필수 키 없으면 None."""
    category = entry.get("category_name")
    tool = entry.get("tool_name")
    api = entry.get("api_name")
    if not (category and tool and api):
        return None
    tool_id = _make_id(category, tool, api)

    # description 결측 대체 (verify_m1: 결측 0 요구).
    desc = entry.get("api_description") or entry.get("description") or ""
    desc_fallback = False
    if not desc.strip():
        desc = entry.get("tool_description") or ""
        desc_fallback = True
    if not desc.strip():
        # 최후 대체: 이름 기반 합성. 결측을 남기지 않되 대체 사실 기록.
        desc = f"API '{api}' of tool '{tool}' in category '{category}'."
        desc_fallback = True

    return {
        "id": tool_id,
        "name": api,
        "description": desc.strip(),
        "params": _extract_params(entry),
        "category": category,
        "desc_fallback": desc_fallback,
    }


def find_test_file(root: str, group: str) -> str:
    """toolbench_root 아래에서 {group}_instruction.json 후보 경로를 탐색."""
    candidates = [
        os.path.join(root, "test_instruction", f"{group}_instruction.json"),
        os.path.join(root, "data", "test_instruction", f"{group}_instruction.json"),
        os.path.join(root, f"{group}_instruction.json"),
        os.path.join(root, "test_instruction", f"{group}_query.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"[{group}] test_instruction 파일을 찾지 못했습니다. 시도한 경로:\n  "
        + "\n  ".join(candidates)
        + f"\nconfig.paths.toolbench_root='{root}' 아래에 ToolBench 데이터를 배치하세요."
    )


def parse_queries(
    raw_list: list[dict[str, Any]], meta: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """원본 쿼리 목록 → 파싱된 쿼리 목록. meta(id→레코드)를 채운다.

    반환: (parsed, drop_stats). parsed 항목: {query_id, query, gold_ids, candidate_ids}.
    gold 를 api_list 로 해소 못 하면 그 쿼리를 드롭하고 사유 집계.
    """
    parsed: list[dict[str, Any]] = []
    drops = Counter()
    for raw in raw_list:
        query = raw.get("query")
        qid = raw.get("query_id")
        api_list = raw.get("api_list") or []
        relevant = raw.get("relevant APIs") or raw.get("relevant_apis") or []
        if not query or qid is None:
            drops["no_query_or_id"] += 1
            continue
        if not relevant:
            drops["no_relevant_apis"] += 1
            continue

        # 이 쿼리의 (tool,api) → 메타. gold category 해소에 사용.
        local: dict[tuple[str, str], dict[str, Any]] = {}
        local_norm: dict[tuple[str, str], dict[str, Any]] = {}
        candidate_ids: list[str] = []
        for entry in api_list:
            m = _entry_to_meta(entry)
            if m is None:
                continue
            meta[m["id"]] = m  # 전역 메타/universe 축적
            candidate_ids.append(m["id"])
            key = (entry.get("tool_name"), entry.get("api_name"))
            local[key] = m
            local_norm[(_norm_name(key[0]), _norm_name(key[1]))] = m

        # gold 해소: relevant APIs = [[tool_name, api_name], ...]
        gold_ids: list[str] = []
        ok = True
        for pair in relevant:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                ok = False
                break
            t, a = pair[0], pair[1]
            m = local.get((t, a)) or local_norm.get((_norm_name(t), _norm_name(a)))
            if m is None:
                ok = False
                break
            gold_ids.append(m["id"])
        if not ok or not gold_ids:
            drops["gold_unresolved"] += 1
            continue

        gold_ids = list(dict.fromkeys(gold_ids))  # 중복 제거, 순서 유지
        parsed.append(
            {
                "query_id": qid,
                "query": query,
                "gold_ids": gold_ids,
                "candidate_ids": list(dict.fromkeys(candidate_ids)),
            }
        )
    return parsed, dict(drops)


def sample_split(
    parsed: list[dict[str, Any]], split: str, n_queries: int, seed: int
) -> list[dict[str, Any]]:
    """split 규칙(multi-tool gold≥2)에 맞는 쿼리에서 seed 로 n_queries 샘플."""
    if split in _SINGLE_TOOL_SPLITS:
        pool = list(parsed)
    else:
        pool = [q for q in parsed if len(set(q["gold_ids"])) >= 2]
    if len(pool) < n_queries:
        raise ValueError(
            f"[{split}] 조건 만족 쿼리 {len(pool)}개 < 요구 {n_queries}개. "
            f"(multi-tool={split not in _SINGLE_TOOL_SPLITS}) 데이터를 확인하세요."
        )
    # 결정적 샘플: query_id 로 정렬 후 seed 샘플.
    pool_sorted = sorted(pool, key=lambda q: str(q["query_id"]))
    rng = random.Random(seed)
    return rng.sample(pool_sorted, n_queries)


def fill_distractors(
    p_gold: set[str],
    meta: dict[str, dict[str, Any]],
    gold_cat_counts: dict[str, int],
    pool_size: int,
    seed: int,
) -> list[str]:
    """gold category 분포에 유사한 비율로 distractor 를 채워 pool_size 를 맞춘다."""
    target = pool_size - len(p_gold)
    if target <= 0:
        return []
    universe = [aid for aid in meta if aid not in p_gold]
    by_cat: dict[str, list[str]] = defaultdict(list)
    for aid in universe:
        by_cat[meta[aid]["category"]].append(aid)
    rng = random.Random(seed)
    for c in sorted(by_cat):  # 결정적: category 정렬 후 셔플
        by_cat[c].sort()
        rng.shuffle(by_cat[c])

    total_gold = sum(gold_cat_counts.values()) or 1
    cap = max(1, math.ceil(target * _DISTRACTOR_CAP_FRAC))
    chosen: list[str] = []
    # 1차: gold category 별 비례 할당(상한 적용).
    for c in sorted(gold_cat_counts):
        q = round(target * gold_cat_counts[c] / total_gold)
        q = min(q, cap, len(by_cat.get(c, [])))
        take = by_cat.get(c, [])[:q]
        chosen.extend(take)
        by_cat[c] = by_cat.get(c, [])[q:]
    chosen = chosen[:target]
    # 2차: 부족분을 남은 universe 에서 보충(gold category 우선, 그다음 기타).
    if len(chosen) < target:
        leftover: list[str] = []
        for c in sorted(gold_cat_counts):
            leftover.extend(by_cat.get(c, []))
        for c in sorted(by_cat):
            if c not in gold_cat_counts:
                leftover.extend(by_cat[c])
        rng.shuffle(leftover)
        chosen.extend(leftover[: target - len(chosen)])
    return chosen[:target]


def build_dataset(
    parsed_by_split: dict[str, list[dict[str, Any]]],
    meta: dict[str, dict[str, Any]],
    n_queries: int,
    pool_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """샘플링 + pool 구성 + 쿼리 산출물 레코드 생성.

    반환: (tools_pool_records, queries_by_split_records, log_info).
    """
    # 1) split 별 300 샘플.
    sampled: dict[str, list[dict[str, Any]]] = {}
    for split, parsed in parsed_by_split.items():
        sampled[split] = sample_split(parsed, split, n_queries, seed)

    # 2) P_gold = 샘플된 gold 의 합집합 (필수 포함).
    p_gold: set[str] = set()
    for split in sampled:
        for q in sampled[split]:
            p_gold.update(q["gold_ids"])

    # 3) pool 구성.
    if len(p_gold) >= pool_size:
        # gold 가 목표보다 많으면 줄이지 않고 실제값으로 pool 을 설정(data-prep.md).
        pool_ids = sorted(p_gold)
        effective_size = len(pool_ids)
        distractor_ids: list[str] = []
    else:
        gold_cat_counts = Counter(meta[aid]["category"] for aid in p_gold)
        distractor_ids = fill_distractors(p_gold, meta, dict(gold_cat_counts), pool_size, seed)
        pool_ids = sorted(p_gold) + distractor_ids
        effective_size = len(pool_ids)

    # 4) tool 레코드 (스키마: id, name, description, params, category (+desc_fallback)).
    tools_records = [meta[aid] for aid in pool_ids]

    # 5) 쿼리 산출물 레코드.
    queries_records: dict[str, list[dict[str, Any]]] = {}
    for split, qs in sampled.items():
        recs = []
        for q in qs:
            gold_cats = sorted({meta[g]["category"] for g in q["gold_ids"]})
            recs.append(
                {
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "gold_tools": q["gold_ids"],
                    "gold_categories": gold_cats,
                }
            )
        queries_records[split] = recs

    log_info = {
        "n_gold": len(p_gold),
        "n_distractor": len(distractor_ids),
        "pool_size_requested": pool_size,
        "pool_size_effective": effective_size,
        "pool_category_dist": dict(Counter(r["category"] for r in tools_records)),
        "split_category_dist": {
            split: dict(Counter(c for r in recs for c in r["gold_categories"]))
            for split, recs in queries_records.items()
        },
        "desc_fallback_count": sum(1 for r in tools_records if r.get("desc_fallback")),
    }
    return tools_records, queries_records, log_info


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _print_log(log_info: dict[str, Any]) -> None:
    print("--- M1 데이터 요약 ---")
    print(f"  gold(필수포함): {log_info['n_gold']}, distractor: {log_info['n_distractor']}")
    print(
        f"  pool 크기: 요청 {log_info['pool_size_requested']} → "
        f"실제 {log_info['pool_size_effective']}"
    )
    print(f"  description 대체 tool 수: {log_info['desc_fallback_count']}")
    print("  [pool category 분포] (상위 15)")
    for cat, n in sorted(log_info["pool_category_dist"].items(), key=lambda x: -x[1])[:15]:
        print(f"    {cat}: {n}")
    for split, dist in log_info["split_category_dist"].items():
        top = sorted(dist.items(), key=lambda x: -x[1])[:10]
        print(f"  [{split} gold category 분포] (상위 10) {top}")


def run(config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    exp = cfg["experiment"]
    splits = exp["splits"]
    n_queries = exp["n_queries_per_split"]
    pool_size = exp["tool_pool_size"]
    data_dir = cfg["paths"]["data_dir"]
    root = cfg["paths"]["toolbench_root"]

    tools_path = os.path.join(data_dir, "tools.jsonl")
    query_paths = {s: os.path.join(data_dir, f"queries_{s}.jsonl") for s in splits}

    # 재실행 안전성: 전 산출물 존재 & not force → skip.
    if not force and os.path.isfile(tools_path) and all(
        os.path.isfile(p) for p in query_paths.values()
    ):
        print(f"[skip] 산출물이 이미 존재합니다 (--force 로 재생성). {data_dir}")
        return

    # 로드 + 파싱.
    meta: dict[str, dict[str, Any]] = {}
    parsed_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        group = _SPLIT_TO_GROUP[split]
        path = find_test_file(root, group)
        with open(path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
        parsed, drops = parse_queries(raw_list, meta)
        parsed_by_split[split] = parsed
        print(f"[{split}] {path}: 원본 {len(raw_list)} → 유효 {len(parsed)} (drop: {drops})")

    tools_records, queries_records, log_info = build_dataset(
        parsed_by_split, meta, n_queries, pool_size, seed
    )

    # 저장.
    _write_jsonl(tools_path, tools_records)
    for split, recs in queries_records.items():
        _write_jsonl(query_paths[split], recs)
    print(f"[write] {tools_path} ({len(tools_records)} tools)")
    for split, recs in queries_records.items():
        print(f"[write] {query_paths[split]} ({len(recs)} queries)")
    _print_log(log_info)


def _smoke() -> None:
    """합성 데이터로 로직만 점검 (데이터/GPU 없이)."""
    print("[smoke] 합성 ToolBench 유사 데이터로 로직 점검")
    rng = random.Random(0)
    cats = [f"Cat{i}" for i in range(4)]

    def make_entry(ci, ti, ai):
        c = cats[ci]
        return {
            "category_name": c,
            "tool_name": f"tool{ti}",
            "api_name": f"api{ai}",
            "api_description": f"does {ai}",
            "required_parameters": [{"name": "q", "type": "string", "description": "query"}],
            "optional_parameters": [{"name": "limit", "type": "number", "description": "n"}],
        }

    # 각 split 용 합성 쿼리 (universe 를 넉넉히 만들어 distractor 채움 확인).
    def make_queries(n, multi):
        out = []
        for i in range(n):
            entries = [make_entry(rng.randrange(4), rng.randrange(6), rng.randrange(6)) for _ in range(5)]
            gold = [[entries[0]["tool_name"], entries[0]["api_name"]]]
            if multi:
                gold.append([entries[1]["tool_name"], entries[1]["api_name"]])
            out.append({"query_id": i, "query": f"q{i}", "api_list": entries, "relevant APIs": gold})
        return out

    meta: dict[str, dict[str, Any]] = {}
    parsed_by_split = {}
    for split, multi in [("I1", False), ("I2", True), ("I3", True)]:
        parsed, drops = parse_queries(make_queries(12, multi), meta)
        parsed_by_split[split] = parsed
        assert not multi or all(len(set(q["gold_ids"])) >= 2 for q in parsed)
    # (A) distractor 채움 경로: pool_size 가 |P_gold| 보다 크게.
    tools, queries, log = build_dataset(parsed_by_split, meta, n_queries=5, pool_size=40, seed=42)
    assert all(len(v) == 5 for v in queries.values()), "split 크기 불일치"
    pool_ids = {t["id"] for t in tools}
    for split, recs in queries.items():
        for r in recs:
            assert set(r["gold_tools"]).issubset(pool_ids), "gold ⊄ pool"
            if split != "I1":
                assert len(set(r["gold_tools"])) >= 2, "multi gold<2"
    assert all(t["description"].strip() for t in tools), "description 결측"
    assert len(pool_ids) == len(tools), "pool id 중복"
    assert log["pool_size_effective"] == 40, "distractor 채움으로 40 이어야 함"
    assert log["n_distractor"] == 40 - log["n_gold"], "distractor 수 불일치"
    _print_log(log)

    # (B) gold 과다 경로: pool_size 를 |P_gold| 보다 작게 → pool=|P_gold|, 축소 금지.
    tools2, _, log2 = build_dataset(parsed_by_split, meta, n_queries=5, pool_size=3, seed=42)
    assert log2["pool_size_effective"] == log2["n_gold"] >= 3, "gold 과다 시 pool=|P_gold|"
    assert log2["n_distractor"] == 0
    pool2 = {t["id"] for t in tools2}
    assert set().union(*[set(r["gold_tools"]) for recs in queries.values() for r in recs]).issubset(
        pool2
    ), "overflow 경로에서도 gold ⊆ pool"
    print("[smoke] OK — distractor 채움/gold 과다 두 경로 및 불변식 통과")


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 데이터 준비 (ToolBench subset)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="산출물이 있어도 재생성")
    ap.add_argument("--smoke", action="store_true", help="합성 입력으로 로직만 점검")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config, args.force)


if __name__ == "__main__":
    main()
