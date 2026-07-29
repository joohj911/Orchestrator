"""m6_analysis.py

명세: spec/rules/run-matrix.md
역할: summary.csv 집계 + 표1·2·3 + 그림1 + 결론 5항목 초안
산출물: results/summary.csv, results/table1.csv, table2.csv, table3.csv,
        results/fig1.png, results/conclusions.md

CLI: python m6_analysis.py --config config.yaml [--force] [--smoke]

규칙:
  - 조합 누락은 보간하지 않는다 — summary.csv 에 빈칸 + reason 열로 남긴다.
  - oracle 수치를 배포치로 제시하지 않는다 (표·결론에 '상한' 명기).
구현: Claude Code.

# DECISION: 표1 의 best_retrieved 는 headline_k(config.analysis) 에서 func_acc 최대
#   (동률 시 strict_success)로 선택하고 어떤 방법인지 표에 기록한다.
# DECISION: 그림1 의 spec 문구 "full/oracle_tool 수평선" 중 full 만 수평선으로 그린다
#   (full 은 K 무관). oracle_tool 은 K 에 의존하므로 점선 곡선으로 그린다 — 수평선으로
#   뭉개면 K별 상한이 왜곡된다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402
from m5_pilot import RETRIEVAL_METHODS  # noqa: E402 (조합 열거 — bm25 제외 목록과 동기)

FNAME_RE = re.compile(r"downstream_(?P<split>I\d)_(?P<model>weak|strong)_(?P<cond>.+)_K(?P<k>\d+)\.jsonl$")

METRIC_COLS = ["func_acc", "exact_match", "strict_success", "completeness", "recall_all",
               "parse_rate", "arg_acc", "mean_prompt_tokens", "mean_n_calls",
               "mean_hallucinated_calls", "retrieval_miss", "generation_miss", "n"]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def aggregate_records(recs):
    """downstream jsonl 레코드 → 요약 dict (summary.csv 한 행)."""
    n = len(recs)
    if not n:
        return {}
    miss = Counter(r["miss_type"] for r in recs if r.get("miss_type"))
    return {
        "n": n,
        "func_acc": round(sum(r["func_acc"] for r in recs) / n, 4),
        "exact_match": round(sum(r["exact_match"] for r in recs) / n, 4),
        "strict_success": round(sum(r["strict_success"] for r in recs) / n, 4),
        "completeness": round(sum(r["completeness"] for r in recs) / n, 4),
        "recall_all": round(sum(r["recall_all"] for r in recs) / n, 4),
        "parse_rate": round(sum(1 for r in recs if r["parse_ok"]) / n, 4),
        "arg_acc": None,  # gold 인자 없음 (scoring.md, M6 결정 사항 — N/A 명기)
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in recs) / n, 1),
        "mean_n_calls": round(sum(r["n_calls"] for r in recs) / n, 3),
        "mean_hallucinated_calls": round(sum(r["hallucinated_calls"] for r in recs) / n, 3),
        "retrieval_miss": miss.get("retrieval_miss", 0),
        "generation_miss": miss.get("generation_miss", 0),
    }


def expected_combos(splits, models, k_sweep):
    """전 조합 열거 (full 은 K0 1회)."""
    conds_k = ["random_k", "oracle_tool"] + [f"retrieved_{m}" for m in RETRIEVAL_METHODS]
    for s in splits:
        for m in models:
            yield (s, m, "full", 0)
            for k in k_sweep:
                for c in conds_k:
                    yield (s, m, c, k)


def load_all(results_dir):
    """downstream_*.jsonl 전부 → {(split, model, cond, k): 요약}."""
    out = {}
    for p in glob.glob(os.path.join(results_dir, "downstream_*.jsonl")):
        mt = FNAME_RE.search(os.path.basename(p))
        if not mt:
            continue  # 파일럿 산출물(downstream_pilot_*) 등은 M6 집계에서 제외
        key = (mt["split"], mt["model"], mt["cond"], int(mt["k"]))
        out[key] = aggregate_records(_read_jsonl(p))
    return out


def build_summary(cfg, results_dir, splits, models, k_sweep):
    """summary.csv 행 목록 (누락은 빈칸 + reason)."""
    agg = load_all(results_dir)
    reasons = {}
    mp = os.path.join(results_dir, "m6_missing.json")
    if os.path.isfile(mp):
        for m in json.load(open(mp, encoding="utf-8")).get("missing", []):
            reasons[(m["split"], m["model"], m["condition"], m["k"])] = m["reason"]
    rows = []
    for key in expected_combos(splits, models, k_sweep):
        s, m, c, k = key
        base = {"split": s, "model": m, "condition": c, "k": k, "reason": ""}
        if key in agg and agg[key]:
            rows.append({**base, **agg[key]})
        else:
            reason = reasons.get(key) or reasons.get((s, m, "*", "*")) or "산출물 없음 (m6_downstream 미실행/실패)"
            rows.append({**base, **{c2: None for c2 in METRIC_COLS}, "reason": reason})
    return rows


def _get(rows, **cond):
    for r in rows:
        if all(r.get(k2) == v for k2, v in cond.items()):
            return r
    return None


def build_table1(rows, splits, models, headline_k):
    """표1 좁히기 유의미: full / random_k / oracle_tool / best_retrieved."""
    out = []
    for s in splits:
        for m in models:
            picks = [("full", _get(rows, split=s, model=m, condition="full", k=0)),
                     ("random_k", _get(rows, split=s, model=m, condition="random_k", k=headline_k)),
                     ("oracle_tool", _get(rows, split=s, model=m, condition="oracle_tool", k=headline_k))]
            # best_retrieved: headline K 의 retrieved_* 중 func_acc 최대 (동률 시 strict)
            cands = [r for r in rows
                     if r["split"] == s and r["model"] == m and r["k"] == headline_k
                     and str(r["condition"]).startswith("retrieved_") and r.get("func_acc") is not None]
            best = max(cands, key=lambda r: (r["func_acc"], r["strict_success"]), default=None)
            picks.append((f"best_retrieved({best['condition'] if best else 'N/A'})", best))
            for label, r in picks:
                out.append({
                    "split": s, "model": m, "condition": label,
                    "k": (r or {}).get("k"),
                    "func_acc": (r or {}).get("func_acc"),
                    "arg_acc": None,
                    "completeness": (r or {}).get("completeness"),
                    "exact_match": (r or {}).get("exact_match"),
                    "strict_success": (r or {}).get("strict_success"),
                    "mean_prompt_tokens": (r or {}).get("mean_prompt_tokens"),
                })
    return out


def build_table2(rows, splits, models, k_sweep, headline_k):
    """표2 방법 비교: Recall_all@K(전 K) + downstream(headline K)."""
    out = []
    for method in RETRIEVAL_METHODS:
        cond = f"retrieved_{method}"
        for s in splits:
            for m in models:
                row = {"method": method, "split": s, "model": m}
                for k in k_sweep:
                    r = _get(rows, split=s, model=m, condition=cond, k=k)
                    row[f"recall_all@{k}"] = (r or {}).get("recall_all")
                r = _get(rows, split=s, model=m, condition=cond, k=headline_k)
                row["func_acc"] = (r or {}).get("func_acc")
                row["completeness"] = (r or {}).get("completeness")
                row["strict_success"] = (r or {}).get("strict_success")
                out.append(row)
    return out


def low_sim_miss_rate(cfg, split, method, k):
    """표3 진단: gold 중 query-dense_single 유사도 하위 p% tool 의 candidate 누락 비율.

    임베딩 캐시가 없으면 None (사유는 표에 기록).
    """
    import numpy as np
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    q_npy = os.path.join(emb_dir, f"queries_{split}.npy")
    d_npy = os.path.join(emb_dir, "tools_desc.npy")
    rpath = os.path.join(results_dir, f"retrieval_{split}_{method}_{k}.jsonl")
    if not (os.path.isfile(q_npy) and os.path.isfile(d_npy) and os.path.isfile(rpath)):
        return None
    queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_idx = {t["id"]: i for i, t in enumerate(tools)}
    q_mat, d_mat = np.load(q_npy), np.load(d_npy)
    cand_by_q = {str(r["query_id"]): set(r["candidate_tools"]) for r in _read_jsonl(rpath)}

    pairs = []  # (유사도, 누락 여부)
    for i, q in enumerate(queries):
        cand = cand_by_q.get(str(q["query_id"]))
        if cand is None:
            continue
        for g in q["gold_tools"]:
            if g not in tool_idx:
                continue
            sim = float(d_mat[tool_idx[g]] @ q_mat[i])
            pairs.append((sim, g not in cand))
    if not pairs:
        return None
    pct = float(cfg.get("analysis", {}).get("low_sim_percentile", 20))
    sims = sorted(p[0] for p in pairs)
    thr = sims[max(0, int(len(sims) * pct / 100) - 1)]
    low = [miss for sim, miss in pairs if sim <= thr]
    return round(sum(low) / len(low), 4) if low else None


def build_table3(cfg, rows, splits, models, headline_k):
    """표3 진단: multi-tool Recall_all, miss 원인 분리, 저유사도 필수 tool 누락률."""
    out = []
    for method in RETRIEVAL_METHODS:
        cond = f"retrieved_{method}"
        for s in splits:
            for m in models:
                r = _get(rows, split=s, model=m, condition=cond, k=headline_k)
                if r is None:
                    continue
                rm, gm = r.get("retrieval_miss"), r.get("generation_miss")
                tot = (rm or 0) + (gm or 0)
                out.append({
                    "method": method, "split": s, "model": m, "k": headline_k,
                    "recall_all": r.get("recall_all"),
                    "retrieval_miss": rm, "generation_miss": gm,
                    "retrieval_miss_ratio": round(rm / tot, 4) if tot else None,
                    "low_sim_gold_miss_rate": low_sim_miss_rate(cfg, s, method, headline_k),
                })
    return out


def draw_fig1(rows, splits, models, k_sweep, out_path):
    """그림1 K sweep: x=K, y=func_acc. 곡선=retrieved 방법 + oracle_tool(점선), full 수평선."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(splits), len(models),
                             figsize=(5.5 * len(models), 3.6 * len(splits)),
                             squeeze=False)
    for si, s in enumerate(splits):
        for mi, m in enumerate(models):
            ax = axes[si][mi]
            for method in RETRIEVAL_METHODS:
                ys = [(_get(rows, split=s, model=m, condition=f"retrieved_{method}", k=k) or {}).get("func_acc")
                      for k in k_sweep]
                if any(y is not None for y in ys):
                    ax.plot(k_sweep, ys, marker="o", label=method)
            ys = [(_get(rows, split=s, model=m, condition="oracle_tool", k=k) or {}).get("func_acc")
                  for k in k_sweep]
            if any(y is not None for y in ys):
                # 범례는 영문 고정 (서버 기본 폰트에 한글 글리프 없음)
                ax.plot(k_sweep, ys, linestyle="--", color="black", label="oracle_tool (upper bound)")
            full = (_get(rows, split=s, model=m, condition="full", k=0) or {}).get("func_acc")
            if full is not None:
                ax.axhline(full, linestyle=":", color="gray", label="full (no narrowing)")
            ax.set_title(f"{s} / {m}")
            ax.set_xlabel("K")
            ax.set_ylabel("func_acc")
            ax.set_xticks(k_sweep)
            if si == 0 and mi == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draft_conclusions(rows, splits, models, headline_k, hyp_flag):
    """결론 5항목 초안 — 수치는 자동 채움, 해석은 사람 확인용 표시."""
    def v(cond, s, m, k, col="func_acc"):
        r = _get(rows, split=s, model=m, condition=cond, k=k)
        return (r or {}).get(col)

    lines = ["# M6 결론 (초안 — 수치 자동 집계, 해석은 사람 확인 후 확정)", ""]
    lines.append("주의: oracle_* 수치는 상한(진단용)이며 배포 성능이 아니다.\n")
    # (a) 좁히기 유의미
    lines.append("## (a) 좁히기 유의미성")
    for s in splits:
        for m in models:
            f0, ot = v("full", s, m, 0), v("oracle_tool", s, m, headline_k)
            if f0 is not None and ot is not None:
                lines.append(f"- {s}/{m}: full {f0} → oracle_tool@{headline_k} {ot} (Δ {round(ot - f0, 4)})")
    # (b) 최선 방법
    lines.append("\n## (b) 최선 retrieval 방법 (headline K 기준 func_acc)")
    for s in splits:
        for m in models:
            cands = [(r["condition"], r["func_acc"]) for r in rows
                     if r["split"] == s and r["model"] == m and r["k"] == headline_k
                     and str(r["condition"]).startswith("retrieved_") and r.get("func_acc") is not None]
            if cands:
                best = max(cands, key=lambda x: x[1])
                lines.append(f"- {s}/{m}: {best[0]} ({best[1]})")
    # (c) 2B vs 9B
    lines.append("\n## (c) 2B/9B 가설 (작은 모델일수록 좁히기 효과 큰가)")
    if hyp_flag:
        lines.append("- 검증 불가: 파일럿에서 2B(weak) 제외됨 (pilot_exclusions.json).")
    else:
        for s in splits:
            deltas = {}
            for m in models:
                f0, ot = v("full", s, m, 0), v("oracle_tool", s, m, headline_k)
                if f0 is not None and ot is not None:
                    deltas[m] = round(ot - f0, 4)
            if len(deltas) == 2:
                lines.append(f"- {s}: 좁히기 이득(oracle−full) weak {deltas.get('weak')} vs "
                             f"strong {deltas.get('strong')}")
    # (d) oracle→real gap
    lines.append("\n## (d) prior oracle→real gap (fusion)")
    for fm in ("fusion_add", "fusion_mult"):
        for s in splits:
            for m in models:
                o = v(f"retrieved_{fm}_oracle", s, m, headline_k)
                r = v(f"retrieved_{fm}_real", s, m, headline_k)
                if o is not None and r is not None:
                    lines.append(f"- {fm} {s}/{m}: oracle {o} vs real {r} (gap {round(o - r, 4)})")
    # (e) dependency 진단
    lines.append("\n## (e) 진단 → 후속 확장 판단")
    for s in splits:
        for m in models:
            rows_sm = [r for r in rows if r["split"] == s and r["model"] == m
                       and str(r["condition"]).startswith("retrieved_") and r["k"] == headline_k
                       and r.get("retrieval_miss") is not None]
            rm = sum(r["retrieval_miss"] for r in rows_sm)
            gm = sum(r["generation_miss"] for r in rows_sm)
            if rm + gm:
                lines.append(f"- {s}/{m}: 실패 원인 retrieval {rm} vs generation {gm} "
                             f"(retrieval 비중 {round(rm / (rm + gm), 3)})")
    lines.append("\n<!-- 사람 확인: 위 수치에 근거해 (a)~(e) 서술 결론을 확정할 것 -->")
    return "\n".join(lines) + "\n"


def run(config_path, force):
    import pandas as pd
    cfg = load_config(config_path)
    splits = cfg["experiment"]["splits"]
    k_sweep = [int(k) for k in cfg["experiment"]["k_sweep"]]
    results_dir = cfg["paths"]["results_dir"]
    headline_k = int(cfg.get("analysis", {}).get("headline_k", 10))

    excl_path = os.path.join(results_dir, "pilot_exclusions.json")
    excl = json.load(open(excl_path, encoding="utf-8")) if os.path.isfile(excl_path) else {}
    models = [mk for mk in ("weak", "strong") if mk not in excl.get("excluded_models", [])]

    rows = build_summary(cfg, results_dir, splits, models, k_sweep)
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    n_missing = sum(1 for r in rows if r["reason"])
    print(f"[m6a] summary.csv {len(rows)}행 (누락 {n_missing} — 빈칸+사유, 보간 없음)")

    pd.DataFrame(build_table1(rows, splits, models, headline_k)).to_csv(
        os.path.join(results_dir, "table1.csv"), index=False)
    pd.DataFrame(build_table2(rows, splits, models, k_sweep, headline_k)).to_csv(
        os.path.join(results_dir, "table2.csv"), index=False)
    pd.DataFrame(build_table3(cfg, rows, splits, models, headline_k)).to_csv(
        os.path.join(results_dir, "table3.csv"), index=False)
    draw_fig1(rows, splits, models, k_sweep, os.path.join(results_dir, "fig1.png"))
    with open(os.path.join(results_dir, "conclusions.md"), "w", encoding="utf-8") as f:
        f.write(draft_conclusions(rows, splits, models, headline_k,
                                  excl.get("hypothesis_2b_vs_9b_unverifiable", False)))
    print(f"[m6a] table1/2/3.csv, fig1.png, conclusions.md 생성 (headline K={headline_k})")


def _smoke():
    """합성 downstream 파일로 집계·표·그림·결론 생성 점검."""
    print("[smoke] m6_analysis (합성 결과)")
    import random
    import tempfile
    import yaml
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/out/data")
    rd = f"{d}/out/results"
    os.makedirs(rd)
    rng = random.Random(0)
    splits, models, ks = ["I1", "I2"], ["weak", "strong"], [5, 10]

    def write(split, model, cond, k, base):
        recs = []
        for i in range(4):
            fa = min(1.0, max(0.0, base + rng.uniform(-0.1, 0.1)))
            recs.append({"query_id": f"{split}_{i}", "condition": cond, "k": k, "n_candidates": k or 500,
                         "called_tools": [], "parse_ok": True, "gen_status": "ok",
                         "func_acc": round(fa, 2), "arg_acc": None,
                         "completeness": round(fa * 0.6, 2), "miss_type": ("generation_miss" if fa < 0.9 else None),
                         "recall_all": 1 if cond != "random_k" else 0,
                         "n_calls": 2, "hallucinated_calls": 0, "exact_match": round(fa * 0.5, 2),
                         "args_valid": 1.0, "strict_success": round(fa * 0.5, 2), "prompt_tokens": 900})
        with open(f"{rd}/downstream_{split}_{model}_{cond}_K{k}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    for s in splits:
        for m in models:
            write(s, m, "full", 0, 0.5)
            for k in ks:
                write(s, m, "random_k", k, 0.05)
                write(s, m, "oracle_tool", k, 0.9)
                for meth in RETRIEVAL_METHODS[:3]:  # 일부 방법만 → 나머지는 누락 처리 검증
                    write(s, m, f"retrieved_{meth}", k, 0.7)
    json.dump({"excluded_models": [], "hypothesis_2b_vs_9b_unverifiable": False},
              open(f"{rd}/pilot_exclusions.json", "w"))

    cfg = yaml.safe_load(open("config.yaml"))
    cfg["paths"]["output_dir"] = f"{d}/out"
    cfg["experiment"]["splits"] = splits
    cfg["experiment"]["k_sweep"] = ks
    cfg["analysis"]["headline_k"] = 10
    yaml.safe_dump(cfg, open(f"{d}/cfg.yaml", "w"))
    run(f"{d}/cfg.yaml", force=True)

    import pandas as pd
    df = pd.read_csv(f"{rd}/summary.csv")
    n_cond = 1 + len(ks) * (2 + len(RETRIEVAL_METHODS))
    assert len(df) == len(splits) * len(models) * n_cond, len(df)
    missing = df[df["reason"].notna() & (df["reason"] != "")]
    assert set(missing["condition"]) == {f"retrieved_{m}" for m in RETRIEVAL_METHODS[3:]}, set(missing["condition"])
    t1 = pd.read_csv(f"{rd}/table1.csv")
    assert any(str(c).startswith("best_retrieved(") for c in t1["condition"]), t1["condition"].tolist()
    for fn in ("table2.csv", "table3.csv", "fig1.png", "conclusions.md"):
        assert os.path.isfile(f"{rd}/{fn}"), fn
    text = open(f"{rd}/conclusions.md").read()
    assert all(f"({c})" in text for c in "abcde"), "결론 5항목 누락"
    print(f"[smoke] OK — summary {len(df)}행(누락 {len(missing)}), 표3종+그림+결론 생성")


def main():
    ap = argparse.ArgumentParser(description="M6 집계·표·그림·결론 (run-matrix.md)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config, args.force)


if __name__ == "__main__":
    main()
