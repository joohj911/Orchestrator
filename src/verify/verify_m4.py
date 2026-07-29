"""verify_m4.py — GATE: M4 classifier

검사 항목 (모두 통과해야 exit 0):
  - test 누출 0: test query_id ∉ classifier_train (val 은 train 부분집합이라 함께 커버)
  - classifier_eval.json 존재 + per_category_auprc 포함
  - class_prior_real.jsonl 이 전 test 쿼리 커버 + prior 가 pool category 전부 포함(fusion 조회용)

CLI: python verify_m4.py --config config.yaml
구현: Claude Code. 상세 spec/MILESTONES.md, spec/rules/classifier.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _fail(msg):
    print(f"[verify_m4] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, make_dirs=False)
    splits = cfg["experiment"]["splits"]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    errors: list[str] = []

    test_ids = set()
    for s in splits:
        p = os.path.join(data_dir, f"queries_{s}.jsonl")
        if os.path.isfile(p):
            test_ids |= {str(q["query_id"]) for q in _read_jsonl(p)}
    train_path = cfg["paths"].get("classifier_train", "")
    train_ids = set()
    if train_path and os.path.isfile(train_path):
        train_ids = {str(r["query_id"]) for r in _read_jsonl(train_path)}
    else:
        errors.append(f"classifier_train 없음: {train_path}")

    # 1) 누출 0 (게이트 핵심)
    leak = train_ids & test_ids
    if leak:
        errors.append(f"train 에 test query_id 누출 {len(leak)}개: {list(leak)[:3]} → 실험 무효.")

    # 2) classifier_eval.json + per-category AUPRC
    eval_path = os.path.join(results_dir, "classifier_eval.json")
    ev = None
    if not os.path.isfile(eval_path):
        errors.append(f"classifier_eval.json 없음: {eval_path}")
    else:
        ev = json.load(open(eval_path, encoding="utf-8"))
        if not ev.get("per_category_auprc"):
            errors.append("classifier_eval.json 에 per_category_auprc 없음/비어있음.")
        for key in ("micro_f1", "macro_f1", "ece_val_before", "ece_val_after", "temperature"):
            if key not in ev:
                errors.append(f"classifier_eval.json 에 '{key}' 없음.")

    # 3) class_prior_real.jsonl 커버리지 + pool category 포함
    prior_path = os.path.join(data_dir, "class_prior_real.jsonl")
    if not os.path.isfile(prior_path):
        errors.append(f"class_prior_real.jsonl 없음: {prior_path}")
    else:
        priors = _read_jsonl(prior_path)
        covered = {str(r["query_id"]) for r in priors}
        missing = test_ids - covered
        if missing:
            errors.append(f"class_prior_real 이 test 쿼리 {len(missing)}개 미커버: {list(missing)[:3]}")
        tools_p = os.path.join(data_dir, "tools.jsonl")
        if os.path.isfile(tools_p) and priors:
            pool_cats = {t["category"] for t in _read_jsonl(tools_p)}
            prior_cats = set(priors[0].get("prior", {}).keys())
            miss_cat = pool_cats - prior_cats
            if miss_cat:
                errors.append(f"prior 가 pool category {len(miss_cat)}개 누락(fusion 조회 불가): {list(miss_cat)[:3]}")

    if ev:
        print(f"=== M4 요약 === micro-F1 {ev.get('micro_f1')}, macro-F1 {ev.get('macro_f1')}, "
              f"mean AUPRC {ev.get('mean_auprc')}, ECE {ev.get('ece_val_before')}→{ev.get('ece_val_after')}")
        cov = ev.get("coverage", {})
        print(f"  vocab {cov.get('vocab_size')} | test category 미커버 "
              f"{len(cov.get('test_cats_missing_in_train', []))}개 | 불균형비 {cov.get('imbalance_ratio')}")

    if errors:
        print("\n[verify_m4] 게이트 실패:", file=sys.stderr)
        for e in errors[:20]:
            _fail(e)
        sys.exit(1)
    print("\n[verify_m4] PASS — test 누출 0, per-category AUPRC 포함, class_prior_real 전 test 커버.")
    sys.exit(0)


if __name__ == "__main__":
    main()
