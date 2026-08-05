"""prepare_contrastive_targets.py — contrastive 예시 작성 대상 선정 (Representation Sharpening 계열).

배경: perfect classifier 로도 남는 갭 = category/이웃 내부의 형제 변별력 (semantic 몫).
현재 예시 5개/tool 은 각 tool 문서만 보고 독립 생성돼 형제 구분 신호가 없음.
개선: **clustering/kNN 은 "누가 누구와 헷갈리는지" 식별에만** 쓰고, 변별 정보 자체는
"A 에는 해당하지만 이웃 B, C 에는 해당하지 않는" contrastive 예시(오프라인 작성)로 주입.
(Representation Sharpening, EACL 2026 — zero-shot, 서빙 추가 비용 0)

이 스크립트의 역할 (서버, CPU, LLM 불필요):
  1. tool 표현(설명+예시 평균) 공간에서 각 tool 의 최근접 이웃 탐색
  2. 혼동 밀집도(상위 이웃 평균 유사도)로 작성 대상 top-N 선정
  3. 작성 작업 지시서(contrastive_targets.jsonl) 출력 — tool 정보 + 기존 예시 +
     혼동 이웃들의 정보 포함 (작성자가 이 파일만 보고 contrastive 예시를 쓸 수 있게)

이후 워크플로 (파이프라인 규약 준수):
  targets 파일 → contrastive 예시를 **오프라인으로 작성해 레포에 커밋**
  (data/tools_examples_contrastive.jsonl) → 병합·재임베딩·recall A/B 는 별도 스크립트.
  신규 예시도 M2 누출 게이트(전 test 쿼리 유사도 ≤ 0.92)를 재통과해야 함.

CLI: python scripts/prepare_contrastive_targets.py --config config.yaml [--smoke]
산출물: results/contrastive_targets.jsonl + 콘솔 요약
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
from experiment_cluster_prior import tool_representations  # noqa: E402
from experiment_service_boost import service_key  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def neighbor_table(reps, n_neighbors, exclude_same_service, tool_ids):
    """각 tool 의 최근접 이웃 인덱스·유사도. 같은 서비스 소속은 제외 옵션
    (같은 서비스 API 는 원래 함께 쓰이는 관계라 '혼동'이 아니라 '동료' — 변별 대상 아님)."""
    sims = reps @ reps.T
    np.fill_diagonal(sims, -1.0)
    n = len(tool_ids)
    if exclude_same_service:
        svc = [service_key(t) for t in tool_ids]
        for i in range(n):
            for j in range(n):
                if i != j and svc[i] == svc[j]:
                    sims[i, j] = -1.0
    order = np.argsort(-sims, axis=1)[:, :n_neighbors]
    return order, np.take_along_axis(sims, order, axis=1)


def run(config_path):
    cfg = load_config(config_path, make_dirs=False)
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    cc = cfg.get("contrastive", {})
    n_targets = int(cc.get("n_targets", 150))
    n_neighbors = int(cc.get("n_neighbors", 5))
    per_tool = int(cc.get("examples_per_tool", 3))

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    by_id = {t["id"]: t for t in tools}

    ex_path = cfg["paths"].get("examples_file", "") or os.path.join(data_dir, "tools_examples.jsonl")
    ex_by_tool = {r["tool_id"]: r["examples"] for r in _read_jsonl(ex_path)}

    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    reps = tool_representations(desc_mat, tool_ex_mat)

    order, sims = neighbor_table(reps, n_neighbors, exclude_same_service=True, tool_ids=tool_ids)
    confusion = sims[:, :3].mean(axis=1)  # 상위 3 이웃 평균 유사도 = 혼동 밀집도
    targets = np.argsort(-confusion)[:n_targets]

    out_path = os.path.join(results_dir, "contrastive_targets.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in targets:
            t = tools[i]
            rec = {
                "tool_id": t["id"],
                "confusion_score": round(float(confusion[i]), 4),
                "n_new_examples": per_tool,
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "params": [p.get("name") for p in (t.get("params") or [])],
                "category": t.get("category", ""),
                "existing_examples": ex_by_tool.get(t["id"], []),
                "confusable_neighbors": [
                    {"tool_id": tool_ids[j], "similarity": round(float(s), 4),
                     "name": by_id[tool_ids[j]].get("name", ""),
                     "description": by_id[tool_ids[j]].get("description", "")[:300]}
                    for j, s in zip(order[i], sims[i]) if s > -1.0
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cat_counts: dict[str, int] = {}
    for i in targets:
        c = tools[i].get("category", "?")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"[contrastive] 대상 {len(targets)}개 선정 (혼동 밀집도 상위, 같은 서비스 이웃 제외)")
    print(f"[contrastive] 혼동 점수 범위: {confusion[targets].max():.3f} ~ {confusion[targets].min():.3f}")
    print(f"[contrastive] category 분포 상위: {sorted(cat_counts.items(), key=lambda x: -x[1])[:8]}")
    print(f"[contrastive] 저장: {out_path}")
    print("[contrastive] 다음: 이 파일을 전달 → contrastive 예시 오프라인 작성·커밋 → 병합 A/B")


def _smoke():
    print("[smoke] contrastive target 선정 로직")
    rng = np.random.default_rng(0)
    n, d = 12, 8

    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    reps = norm(rng.standard_normal((n, d))).astype(np.float32)
    # tool 0/1 을 거의 동일하게 (혼동 쌍), 서로 다른 서비스로
    reps[1] = norm(reps[0] + 0.01 * rng.standard_normal(d))[None].astype(np.float32)
    ids = [f"C__svc{i}__api{i}" for i in range(n)]
    order, sims = neighbor_table(reps, 3, True, ids)
    assert order[0][0] == 1 and order[1][0] == 0, "최근접 혼동 쌍 식별 실패"
    assert sims[0][0] > 0.99
    # 같은 서비스 이웃 제외 확인
    ids2 = ["C__svcX__a", "C__svcX__b"] + ids[2:]
    order2, sims2 = neighbor_table(reps, 3, True, ids2)
    assert order2[0][0] != 1 or sims2[0][0] == -1.0, "같은 서비스 이웃이 제외되지 않음"
    print("[smoke] OK — 이웃 식별·서비스 제외 정상")


def main():
    ap = argparse.ArgumentParser(description="contrastive 예시 작성 대상 선정 (CPU, LLM 불필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
