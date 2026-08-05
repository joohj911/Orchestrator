"""experiment_contrastive_ab.py — contrastive 예시 병합 + 누출 게이트 + recall A/B.

배경: 무학습 prior 5종 전부 null → 남은 갭(형제 변별)은 인덱스에 **새 변별 텍스트**를
넣어야 닫힘 (Representation Sharpening 계열). data/tools_examples_contrastive.jsonl 은
혼동 밀집 상위 tool 에 대해 "이 tool 에는 맞고 혼동 이웃에는 안 맞는" 발화를 오프라인
작성해 커밋한 것 (작성 규칙: API명 금지·서비스명 허용·test 쿼리 미참조).

이 스크립트 (GPU 필요 — 신규 예시 임베딩):
  1. 신규 예시를 e5 passage 로 임베딩
  2. **누출 게이트**: 전 test 쿼리와의 최대 유사도 > threshold(0.92) 인 예시는 제외하고
     상세 보고 (재작성 대상 목록)
  3. 통과 예시를 tool 벡터 집합에 추가해 dense_multi(max) recall 을 A/B
     - A = 기존(설명 1 + 예시 5) / B = A + contrastive
  4. 판정 기준: I2·저K recall 개선 (형제 변별이 병목인 구역)

CLI: python scripts/experiment_contrastive_ab.py --config config.yaml [--smoke]
산출물: results/contrastive_ab.json + 콘솔 표
구현: Claude Code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
from utils.config import load_config  # noqa: E402
from utils.scoring import recall_all  # noqa: E402
from m3_retrieval import topk_ids  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def leak_gate(new_recs, new_embs, query_embs_by_split, queries_by_split, threshold):
    """신규 예시 × 전 test 쿼리 유사도 전수 검사. (통과 마스크, 위반 상세) 반환."""
    ok = np.ones(len(new_embs), dtype=bool)
    violations = []
    for split, q_mat in query_embs_by_split.items():
        sims = new_embs @ q_mat.T  # (n_new, n_q)
        for i in range(len(new_embs)):
            j = int(np.argmax(sims[i]))
            if sims[i][j] > threshold:
                ok[i] = False
                violations.append({
                    "tool_id": new_recs[i]["tool_id"], "example": new_recs[i]["example"],
                    "split": split, "max_sim": round(float(sims[i][j]), 4),
                    "closest_query": queries_by_split[split][j]["query"][:120],
                })
    return ok, violations


def dense_multi_recall(q_mat, vec_list_by_tool, tool_ids, gold_by_q, ks):
    """tool 별 가변 개수 벡터에 대한 max-cosine 랭킹 recall."""
    flat, owner = [], []
    for ti, vecs in enumerate(vec_list_by_tool):
        for v in vecs:
            flat.append(v)
            owner.append(ti)
    flat = np.stack(flat)
    owner = np.array(owner)
    n_tools = len(tool_ids)

    out = {k: 0 for k in ks}
    for qi in range(q_mat.shape[0]):
        sims = flat @ q_mat[qi]
        scores = np.full(n_tools, -1e9, dtype=np.float32)
        np.maximum.at(scores, owner, sims)
        for k in ks:
            out[k] += recall_all(topk_ids(scores, tool_ids, k), gold_by_q[qi])
    return {k: round(v / max(1, q_mat.shape[0]), 4) for k, v in out.items()}


def run(config_path, embed_fn=None):
    cfg = load_config(config_path, make_dirs=False)
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    threshold = float(cfg["experiment"]["leak_sim_threshold"])
    c_path = cfg.get("contrastive", {}).get("examples_file", "") or "./data/tools_examples_contrastive.jsonl"

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    idx_of = {t: i for i, t in enumerate(tool_ids)}

    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))

    # 신규 예시 로드 → 평탄화
    if not os.path.isfile(c_path):
        sys.exit(f"[contrastive-ab] {c_path} 없음 (contrastive 예시 커밋 먼저)")
    new_recs = []
    unknown = 0
    for r in _read_jsonl(c_path):
        if r["tool_id"] not in idx_of:
            unknown += 1
            continue
        for ex in r["examples"]:
            new_recs.append({"tool_id": r["tool_id"], "example": ex})
    print(f"[contrastive-ab] 신규 예시 {len(new_recs)}개 "
          f"({len(set(r['tool_id'] for r in new_recs))} tools, pool 밖 tool {unknown})")

    # 임베딩 (기존 예시와 동일하게 passage prefix)
    if embed_fn is None:
        from utils.embed import embed_passages
        embed_fn = lambda texts: embed_passages(texts, cfg)  # noqa: E731
    new_embs = embed_fn([r["example"] for r in new_recs])
    new_embs = np.asarray(new_embs, dtype=np.float32)

    queries_by_split = {s: _read_jsonl(os.path.join(data_dir, f"queries_{s}.jsonl")) for s in splits}
    query_embs = {s: np.load(os.path.join(emb_dir, f"queries_{s}.npy")) for s in splits}

    ok, violations = leak_gate(new_recs, new_embs, query_embs, queries_by_split, threshold)
    print(f"[contrastive-ab] 누출 게이트: 통과 {int(ok.sum())}/{len(ok)} (threshold {threshold})")
    for v in violations:
        print(f"  [위반] {v['tool_id']} sim={v['max_sim']} vs {v['split']} "
              f"\"{v['closest_query']}\"\n         예시: \"{v['example']}\"")

    # A/B 벡터 집합 구성
    base_vecs = []
    for ti in range(len(tool_ids)):
        vecs = [desc_mat[ti]]
        for j in range(tool_ex_mat.shape[1]):
            if np.linalg.norm(tool_ex_mat[ti, j]) > 1e-8:
                vecs.append(tool_ex_mat[ti, j])
        base_vecs.append(vecs)
    plus_vecs = [list(v) for v in base_vecs]
    for i, r in enumerate(new_recs):
        if ok[i]:
            plus_vecs[idx_of[r["tool_id"]]].append(new_embs[i])

    report = {"n_new": len(new_recs), "n_passed": int(ok.sum()),
              "violations": violations, "splits": {}}
    for split in splits:
        gold_by_q = [list(q["gold_tools"]) for q in queries_by_split[split]]
        q_mat = query_embs[split]
        a = dense_multi_recall(q_mat, base_vecs, tool_ids, gold_by_q, ks)
        b = dense_multi_recall(q_mat, plus_vecs, tool_ids, gold_by_q, ks)
        report["splits"][split] = {"base": a, "plus_contrastive": b,
                                   "delta": {k: round(b[k] - a[k], 4) for k in ks}}
        print(f"[contrastive-ab] {split} 완료", flush=True)

    out = os.path.join(results_dir, "contrastive_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for split in splits:
        e = report["splits"][split]
        print(f"\n=== {split} : Recall_all@K — contrastive 예시 A/B (dense_multi) ===")
        print("variant".ljust(22), *[f"K={k}".rjust(7) for k in ks])
        print("base (예시 5)".ljust(22), *[f"{e['base'][k]:.2f}".rjust(7) for k in ks])
        print("+ contrastive".ljust(22), *[f"{e['plus_contrastive'][k]:.2f}".rjust(7) for k in ks])
        print("delta".ljust(22), *[f"{e['delta'][k]:+.2f}".rjust(7) for k in ks])
    print(f"\n[contrastive-ab] 저장: {out}")
    print("[contrastive-ab] 판정: I2 저K(5·10)에서 +면 형제 변별 주입 유효 → 나머지 100개 "
          "tool 로 확대. 위반 예시는 재작성 대상.")


def _smoke():
    """합성 임베딩 주입으로 게이트·병합·A/B 경로 점검 (GPU 불필요)."""
    print("[smoke] contrastive A/B 로직")
    rng = np.random.default_rng(0)
    d = 8

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    # leak_gate: 위반 검출
    q = norm(rng.standard_normal((3, d))).astype(np.float32)
    new = norm(np.concatenate([q[0:1] + 1e-3, rng.standard_normal((2, d))])).astype(np.float32)
    recs = [{"tool_id": f"T{i}", "example": f"ex{i}"} for i in range(3)]
    ok, vio = leak_gate(recs, new, {"I1": q}, {"I1": [{"query": "dup"}] * 3}, 0.92)
    assert not ok[0] and ok[1] and ok[2] and len(vio) == 1, (ok, vio)

    # dense_multi_recall: 추가 벡터가 정확히 그 tool 만 끌어올리는지
    tool_ids = ["A", "B", "C"]
    qv = norm(rng.standard_normal((1, d))).astype(np.float32)
    far = norm(rng.standard_normal((3, d))).astype(np.float32)
    base = [[far[0]], [far[1]], [far[2]]]
    r_base = dense_multi_recall(qv, base, tool_ids, [["B"]], [1])
    plus = [list(v) for v in base]
    plus[1].append(qv[0])  # B 에 쿼리와 동일한 벡터 추가 → top-1 보장
    r_plus = dense_multi_recall(qv, plus, tool_ids, [["B"]], [1])
    assert r_plus[1] == 1.0 and r_plus[1] >= r_base[1], (r_base, r_plus)
    print("[smoke] OK — 게이트 위반 검출·가변 벡터 병합·recall 경로 정상")


def main():
    ap = argparse.ArgumentParser(description="contrastive 예시 병합·게이트·recall A/B (GPU: 신규 예시 임베딩)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
