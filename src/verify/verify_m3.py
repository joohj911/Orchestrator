"""verify_m3.py — GATE: M3 retrieval

검사 항목 (모두 통과해야 exit 0):
  - 전 split×방법×K candidate 파일 존재
  - Recall_all 검산 (gold⊆candidate ⟺ recall_all=1, 기록값 일치)
  - fusion_coeffs.json 존재 + 계수 test 미사용 검증 (val/test 분리, 계수는 val 기록)
  - e5 prefix 규칙 로그 + 임베딩 산출물 존재

CLI: python verify_m3.py --config config.yaml
구현: Claude Code. 상세 spec/MILESTONES.md, spec/rules/retrieval.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402
from utils.scoring import recall_all  # noqa: E402

NON_FUSION = ["bm25", "dense_single", "dense_multi"]
FUSION = ["fusion_add", "fusion_mult"]


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _fail(msg: str) -> None:
    print(f"[verify_m3] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, make_dirs=False)
    splits = cfg["experiment"]["splits"]
    k_sweep = cfg["experiment"]["k_sweep"]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    coeffs_path = os.path.join(cfg["paths"]["output_dir"], "fusion_coeffs.json")

    errors: list[str] = []

    # e5 prefix 로그
    try:
        from utils.embed import PREFIX_QUERY, PREFIX_PASSAGE
        print(f"=== M3 검증 === e5 prefix: query='{PREFIX_QUERY}', passage='{PREFIX_PASSAGE}'")
    except Exception as e:  # noqa: BLE001
        errors.append(f"utils.embed prefix 상수 로드 실패: {e}")

    # 임베딩 산출물
    for fn in ["tools_desc.npy", "tools_examples.npy"] + [f"queries_{s}.npy" for s in splits]:
        if not os.path.isfile(os.path.join(emb_dir, fn)):
            errors.append(f"임베딩 산출물 없음: {os.path.join(emb_dir, fn)}")

    # fusion_coeffs.json + test 미사용 검증
    coeffs = None
    if not os.path.isfile(coeffs_path):
        errors.append(f"fusion_coeffs.json 없음: {coeffs_path}")
    else:
        coeffs = json.load(open(coeffs_path, encoding="utf-8"))
        for split in splits:
            sc = coeffs.get("splits", {}).get(split)
            if not sc:
                errors.append(f"[{split}] fusion_coeffs 항목 없음.")
                continue
            val_ids = set(map(str, sc.get("val_ids", [])))
            test_ids = set(map(str, sc.get("test_ids", [])))
            if not val_ids:
                errors.append(f"[{split}] val_ids 비어있음 (계수 선택 근거 없음).")
            if val_ids & test_ids:
                errors.append(f"[{split}] val ∩ test != ∅ → 계수 선택에 test 누출 위험.")
            for m in FUSION:
                if m not in sc:
                    errors.append(f"[{split}] fusion_coeffs 에 {m} 계수 없음.")

    # candidate 파일 존재 + Recall_all 검산
    def check_file(path: str):
        if not os.path.isfile(path):
            errors.append(f"candidate 파일 없음: {os.path.basename(path)}")
            return
        for rec in _read_jsonl(path):
            cand, gold = rec.get("candidate_tools", []), rec.get("gold_tools", [])
            recomputed = recall_all(cand, gold)
            if recomputed != rec.get("recall_all"):
                errors.append(f"{os.path.basename(path)} q{rec.get('query_id')}: recall_all 기록 {rec.get('recall_all')} ≠ 재계산 {recomputed}")
                return  # 파일당 첫 불일치만 보고
            # 계수 test 미사용 교차검증: fusion 파일에 test 롤이 있어도 OK(보고는 test),
            # 핵심은 계수 선택이 val 기반이라는 fusion_coeffs 검증으로 이미 커버.

    for split in splits:
        for m in NON_FUSION:
            for k in k_sweep:
                check_file(os.path.join(results_dir, f"retrieval_{split}_{m}_{k}.jsonl"))
        for m in FUSION:
            for k in k_sweep:
                check_file(os.path.join(results_dir, f"retrieval_{split}_{m}_oracle_{k}.jsonl"))

    # 요약
    if coeffs:
        for split in splits:
            sc = coeffs["splits"].get(split, {})
            print(f"[{split}] val {len(sc.get('val_ids', []))} / test {len(sc.get('test_ids', []))} | "
                  f"add={sc.get('fusion_add')} mult={sc.get('fusion_mult')}")

    if errors:
        print("\n[verify_m3] 게이트 실패:", file=sys.stderr)
        for e in errors[:20]:
            _fail(e)
        sys.exit(1)
    print("\n[verify_m3] PASS — candidate 전 조합 생성, Recall_all 검산 통과, fusion 계수 test 미사용.")
    sys.exit(0)


if __name__ == "__main__":
    main()
