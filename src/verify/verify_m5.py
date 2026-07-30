"""verify_m5.py — GATE: M5 파일럿

검사 항목:
  - 모델별 **구조적 파싱 성공률** = ok / (ok + malformed) < threshold(0.90) → 제외(2B 우선).
    9B(strong)도 미달이면 FAIL(중단).
  - no_call 은 게이트가 아니라 **행동 지표**로 별도 보고 (조건별 분포 포함).
  - 프롬프트 템플릿·디코딩이 조건 간 동일함을 로그.
  - 제외 결정 + 파싱/no_call 지표를 results/pilot_exclusions.json 에 기록 (M6 참조).

# DECISION (2026-07-30, 사람 확인 근거): 게이트 재정의 — 기존 정의(ok/전체)는 no_call 을
#   파싱 실패로 세었으나, 실측 결과 no_call 은 파서 문제가 아니라 모델 행동이었다:
#   (1) 원문(gen_excerpt) 전수 확인 — 정보 부족 쿼리에 대한 되물음/설명 (malformed 0).
#   (2) no_call 의 절반이 random_k(무관 후보만 제공되는 조건)에서 발생 — 부적절한 후보를
#       거부하는 것은 올바른 행동이며 게이트가 벌점화할 대상이 아니다.
#   no_call 은 이미 func/strict 0점으로 채점에 정당히 반영되며, 모델별 비율은 결론에서
#   행동 특성(9B 신중/2B 직행)으로 기술한다. 게이트의 본래 목적(파서가 출력을 채점
#   가능하게 잡는가)은 ok/(ok+malformed) 가 정확히 측정한다.

CLI: python verify_m5.py --config config.yaml
구현: Claude Code. 상세 spec/MILESTONES.md, spec/rules/scoring.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config  # noqa: E402


def _fail(msg):
    print(f"[verify_m5] FAIL: {msg}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, make_dirs=False)
    results_dir = cfg["paths"]["results_dir"]
    threshold = float(cfg["pilot"]["parse_success_threshold"])

    report_path = os.path.join(results_dir, "pilot_report.json")
    if not os.path.isfile(report_path):
        _fail(f"pilot_report.json 없음: {report_path} (m5 먼저)")
        sys.exit(1)
    rep = json.load(open(report_path, encoding="utf-8"))
    models = rep.get("models", {})

    print(f"=== M5 파일럿 검증 === threshold {threshold} (게이트: 구조적 파싱 = ok/(ok+malformed))")
    print(f"  decoding={rep.get('decoding')} | system_prompt={'있음' if rep.get('system_prompt') else '없음'} "
          f"| enable_thinking={rep.get('enable_thinking')} (조건 간 동일)")

    weak = models.get("weak", {})
    strong = models.get("strong", {})
    for k, m in models.items():
        sc = m.get("status_counts", {}) or {}
        ok, mal, nc = sc.get("ok", 0), sc.get("malformed", 0), sc.get("no_call", 0)
        total = ok + mal + nc
        m["_structural"] = round(ok / (ok + mal), 4) if (ok + mal) else 0.0
        m["_no_call_rate"] = round(nc / total, 4) if total else 0.0
        print(f"  {k}={m.get('model_id')} 구조적파싱 {m['_structural']} | "
              f"no_call {m['_no_call_rate']} (행동 지표, 게이트 아님) | 상태 {sc}")
        nc_by_cond = {c: v.get("no_call", 0) for c, v in (m.get("per_condition_status") or {}).items()
                      if v.get("no_call")}
        if nc_by_cond:
            print(f"    no_call 조건별 분포: {nc_by_cond}")
        # 사람 확인용(게이트 아님): 방향 상식성 — full ≤ oracle_tool 인지, retriever 가 도움/해악인지.
        # strict_success = 정확히 gold 만 + 스키마상 실행 가능하게 호출 (관대한 func_acc 와 구분).
        for cond, cm in (m.get("per_condition_metrics") or {}).items():
            print(f"    {cond}: func_acc {cm.get('func_acc')} exact {cm.get('exact_match')} "
                  f"strict {cm.get('strict_success')} recall_all {cm.get('recall_all')} "
                  f"halluc {cm.get('mean_hallucinated_calls')} miss {cm.get('miss_type_counts')}")
        for cond, eff in (m.get("retrieval_effect") or {}).items():
            vf, vr = eff.get("vs_full", {}), eff.get("vs_random_k", {})
            print(f"    [효과] {cond}: Δstrict(vs full) {vf.get('delta_strict_success')} "
                  f"(도움 {vf.get('helped_queries')}/해악 {vf.get('hurt_queries')}) | "
                  f"Δstrict(vs random_k) {vr.get('delta_strict_success')} | "
                  f"recall hit/miss 시 strict {eff.get('strict_success_given_recall_hit')}/"
                  f"{eff.get('strict_success_given_recall_miss')}")

    excluded, hyp_unverifiable = [], False
    # 우선순위: 2B(weak) 미달 → 제외 + 가설 검증불가 플래그.
    if weak and weak.get("_structural", 0) < threshold:
        excluded.append("weak")
        hyp_unverifiable = True
        print(f"  → weak(2B) 구조적파싱 {weak.get('_structural')} < {threshold} → 제외. "
              f"'2B/9B 가설 검증 불가' 플래그.")

    errors = []
    # 9B(strong) 미달이면 중단.
    if strong and strong.get("_structural", 0) < threshold:
        errors.append(f"strong(9B) 구조적파싱 {strong.get('_structural')} < {threshold} → 파서 점검 필요(중단).")

    # 제외 결정 + 지표 기록 (M6 참조 — no_call 은 결론에서 행동 특성으로 기술).
    excl = {"threshold": threshold, "gate_metric": "structural_parse_rate = ok/(ok+malformed)",
            "excluded_models": excluded,
            "hypothesis_2b_vs_9b_unverifiable": hyp_unverifiable,
            "structural_parse_rates": {k: m.get("_structural") for k, m in models.items()},
            "no_call_rates": {k: m.get("_no_call_rate") for k, m in models.items()},
            "raw_parse_rates_incl_no_call": {k: m.get("parse_rate") for k, m in models.items()}}
    with open(os.path.join(results_dir, "pilot_exclusions.json"), "w", encoding="utf-8") as f:
        json.dump(excl, f, ensure_ascii=False, indent=2)

    if errors:
        print("\n[verify_m5] 게이트 실패:", file=sys.stderr)
        for e in errors:
            _fail(e)
        sys.exit(1)
    kept = [k for k in models if k not in excluded]
    print(f"\n[verify_m5] PASS — 유지 모델 {kept}, 제외 {excluded}. pilot_exclusions.json 기록.")
    sys.exit(0)


if __name__ == "__main__":
    main()
