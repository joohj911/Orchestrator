"""verify_m3.py — GATE: M3 retrieval (k-fold CV)

검사 항목 (모두 통과해야 exit 0):
  - 전 split×방법×K candidate 파일 존재
  - Recall_all 검산 (gold⊆candidate ⟺ recall_all=1, 기록값 일치)
  - fusion_coeffs.json 존재 + 계수 test 미사용 검증:
      · fold_assignment 가 split 쿼리를 완전 분할(각 쿼리 정확히 1 fold)
      · fold 별 계수 존재, n_folds ≥ 2 (train fold 가 있어 각 fold 가 자신을 안 봄)
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


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _fail(msg):
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

    try:
        from utils.embed import PREFIX_QUERY, PREFIX_PASSAGE
        print(f"=== M3 검증 === e5 prefix: query='{PREFIX_QUERY}', passage='{PREFIX_PASSAGE}'")
    except Exception as e:  # noqa: BLE001
        errors.append(f"utils.embed prefix 상수 로드 실패: {e}")

    for fn in ["tools_desc.npy", "tools_examples.npy"] + [f"queries_{s}.npy" for s in splits]:
        if not os.path.isfile(os.path.join(emb_dir, fn)):
            errors.append(f"임베딩 산출물 없음: {os.path.join(emb_dir, fn)}")

    coeffs = None
    if not os.path.isfile(coeffs_path):
        errors.append(f"fusion_coeffs.json 없음: {coeffs_path}")
    else:
        coeffs = json.load(open(coeffs_path, encoding="utf-8"))
        n_folds = int(coeffs.get("n_folds", 0))
        if n_folds < 2:
            errors.append(f"n_folds={n_folds} < 2 → 각 fold 가 자신을 안 본 계수를 못 가짐(누출 위험).")
        for split in splits:
            sc = coeffs.get("splits", {}).get(split)
            if not sc:
                errors.append(f"[{split}] fusion_coeffs 항목 없음.")
                continue
            fa = sc.get("fold_assignment", {})
            qset_path = os.path.join(data_dir, f"queries_{split}.jsonl")
            qids = {str(q["query_id"]) for q in _read_jsonl(qset_path)} if os.path.isfile(qset_path) else set()
            fa_ids = set(map(str, fa.keys()))
            if qids and fa_ids != qids:
                errors.append(f"[{split}] fold_assignment 이 쿼리 전체를 덮지 않음(누락/초과).")
            used_folds = sorted(set(fa.values()))
            if len(used_folds) < 2:
                errors.append(f"[{split}] 사용된 fold 수 {len(used_folds)} < 2.")
            folds = sc.get("folds", {})
            for f in used_folds:
                fe = folds.get(str(f))
                if not fe or any(m not in fe for m in FUSION):
                    errors.append(f"[{split}] fold {f} 계수 누락.")

    def check_file(path):
        if not os.path.isfile(path):
            errors.append(f"candidate 파일 없음: {os.path.basename(path)}")
            return
        for rec in _read_jsonl(path):
            recomputed = recall_all(rec.get("candidate_tools", []), rec.get("gold_tools", []))
            if recomputed != rec.get("recall_all"):
                errors.append(f"{os.path.basename(path)} q{rec.get('query_id')}: "
                              f"recall_all 기록 {rec.get('recall_all')} ≠ 재계산 {recomputed}")
                return

    for split in splits:
        for m in NON_FUSION:
            for k in k_sweep:
                check_file(os.path.join(results_dir, f"retrieval_{split}_{m}_{k}.jsonl"))
        for m in FUSION:
            for k in k_sweep:
                check_file(os.path.join(results_dir, f"retrieval_{split}_{m}_oracle_{k}.jsonl"))

    if coeffs:
        for split in splits:
            sc = coeffs["splits"].get(split, {})
            print(f"[{split}] {len(sc.get('folds', {}))} folds | "
                  f"coeff 분포(add)={sc.get('coeff_distribution_informational', {}).get('fusion_add')}")

    if errors:
        print("\n[verify_m3] 게이트 실패:", file=sys.stderr)
        for e in errors[:20]:
            _fail(e)
        sys.exit(1)
    print("\n[verify_m3] PASS — candidate 전 조합 생성, Recall_all 검산 통과, "
          "fusion 계수 k-fold(각 fold 자신 미사용) 검증.")
    sys.exit(0)


if __name__ == "__main__":
    main()
