"""experiment_service_boost.py — 문서 메타데이터 기반 집합 boost (무학습·무로그·서빙 LLM 0).

배경: 무학습 clustering 3변형이 전부 null — 같은 임베딩 공간의 기하에는 새 정보가 없음.
탈출 조건 = "기하 밖 정보원". 가장 싼 후보가 **tool 메타데이터**: 같은 서비스(ToolBench
tool)에 속한 API 들은 함께 쓰일 개연성이 높지만, 임베딩상 서로 멀 수 있다
(예: 한 서비스의 Checkhealth 와 Projects). cluster_max 와 기계는 같아도 집합의 출처가
임베딩이 아니라 메타데이터라는 점이 결정적 차이.

메커니즘 (서빙 비용: 행렬곱 1회 수준, LLM 0):
  group(t) = t 가 속한 서비스의 API 집합 (tool id 의 category__tool 접두로 유도)
  group_score(q, g) = max_{t∈g} dense_multi(q, t)   ← "서비스의 최고 API 점수"
  prior(q, t) = softmax_g(group_score)[group(t)]    ← 동료 API 전체를 끌어올림
기대: I1(정답 = 한 서비스의 API 여러 개)에서 직접 효과. I2/I3 는 정답이 서비스를 넘나들어
효과 제한 예상 — 그 한계도 실측으로 기록.

검증 안전장치: 서비스 그룹핑이 올바른지 I1 gold 로 진단 (I1 정답 집합이 단일 그룹에
포함되는 비율 — 정의상 ~1.0 이어야 함. 낮으면 id 파싱이 틀린 것이므로 중단).

CLI: python scripts/experiment_service_boost.py --config config.yaml [--smoke]
GPU 불필요. 산출물: results/service_boost_ab.json + 콘솔 표
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
from m3_retrieval import assign_folds, cosine_scores_multi  # noqa: E402
from experiment_prior_ablation import recall_with_prior  # noqa: E402
from experiment_cluster_prior import baseline_recalls, _softmax_rows  # noqa: E402


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def service_key(tool_id: str) -> str:
    """tool id (category__tool__api) → 서비스 키 (category__tool).

    api 부분에 '__' 가 포함될 수 있어 완벽하진 않으나, 진단(아래)이 오파싱을 잡아낸다.
    """
    parts = tool_id.split("__")
    return "__".join(parts[:2]) if len(parts) >= 3 else tool_id


def build_groups(tool_ids):
    """서비스 키 → 멤버 tool 인덱스 배열."""
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tool_ids):
        groups.setdefault(service_key(t), []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


def grouping_diagnostic(queries_i1, tool_ids):
    """I1 정답 집합이 단일 서비스 그룹에 들어가는 비율 (그룹핑 정합성 검증)."""
    ok = 0
    for q in queries_i1:
        keys = {service_key(g) for g in q["gold_tools"]}
        ok += (len(keys) == 1)
    return ok / max(1, len(queries_i1))


def service_boost_prior(sem_scores, groups, group_of_tool, n_tools, temperature):
    """서비스 그룹의 최고 semantic 점수를 softmax 후 멤버 전체에 부여."""
    keys = list(groups.keys())
    key_idx = {k: i for i, k in enumerate(keys)}
    nq = len(sem_scores)
    gs = np.full((nq, len(keys)), -1e9, dtype=np.float32)
    for gi, k in enumerate(keys):
        idx = groups[k]
        for qi in range(nq):
            gs[qi, gi] = sem_scores[qi][idx].max()
    probs = _softmax_rows(gs, temperature)
    tool_group_idx = np.array([key_idx[group_of_tool[t]] for t in range(n_tools)])
    return probs[:, tool_group_idx].astype(np.float32)


def run(config_path):
    cfg = load_config(config_path, make_dirs=False)
    seed = int(cfg["seed"])
    splits = cfg["experiment"]["splits"]
    ks = [int(k) for k in cfg["experiment"]["k_sweep"]]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    fcfg = cfg["fusion"]
    temperature = float(cfg.get("cluster_prior", {}).get("centroid_temperature", 0.05))

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    tool_ids = [t["id"] for t in tools]
    groups = build_groups(tool_ids)
    group_of_tool = {i: service_key(t) for i, t in enumerate(tool_ids)}
    sizes = np.array([len(v) for v in groups.values()])
    print(f"[svc-boost] 서비스 그룹 {len(groups)}개 (API 수 평균 {sizes.mean():.2f}, 최대 {sizes.max()})")

    q_i1 = _read_jsonl(os.path.join(data_dir, "queries_I1.jsonl"))
    diag = grouping_diagnostic(q_i1, tool_ids)
    print(f"[svc-boost] 그룹핑 진단: I1 gold 가 단일 서비스에 담기는 비율 = {diag:.2f}")
    if diag < 0.9:
        sys.exit("[svc-boost] 진단 실패 — service_key 파싱이 데이터와 안 맞음. 중단.")

    desc_mat = np.load(os.path.join(emb_dir, "tools_desc.npy"))
    tool_ex_mat = np.load(os.path.join(emb_dir, "tools_examples.npy"))
    tool_vecs = np.concatenate([desc_mat[:, None, :], tool_ex_mat], axis=1)

    report = {"n_groups": len(groups), "group_size_mean": round(float(sizes.mean()), 2),
              "group_size_max": int(sizes.max()), "i1_single_service_rate": round(diag, 4),
              "splits": {}}
    for split in splits:
        queries = _read_jsonl(os.path.join(data_dir, f"queries_{split}.jsonl"))
        gold_by_q = [list(q["gold_tools"]) for q in queries]
        q_mat = np.load(os.path.join(emb_dir, f"queries_{split}.npy"))
        sem = [cosine_scores_multi(q_mat[i], tool_vecs) for i in range(len(queries))]
        fold_of = assign_folds([str(q["query_id"]) for q in queries], int(fcfg["n_folds"]), seed)

        prior = service_boost_prior(sem, groups, group_of_tool, len(tool_ids), temperature)
        fused = recall_with_prior(queries, tool_ids, sem, prior, gold_by_q, fold_of, fcfg, ks)
        report["splits"][split] = {"fused": fused,
                                   "baselines": baseline_recalls(results_dir, split, ks)}
        print(f"[svc-boost] {split} 완료", flush=True)

    out = os.path.join(results_dir, "service_boost_ab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for split in splits:
        e = report["splits"][split]
        print(f"\n=== {split} : Recall_all@K — 서비스 집합 boost ===")
        print("variant".ljust(26), *[f"K={k}".rjust(7) for k in ks])
        b = e["baselines"]
        for name in ("dense_multi", "cat_add_real", "cat_mult_real"):
            if name in b:
                print(name.ljust(26), *[f"{b[name].get(k, float('nan')):.2f}".rjust(7) for k in ks])
        for m in ("add", "mult"):
            print(f"service_boost {m}".ljust(26), *[f"{e['fused'][m][k]:.2f}".rjust(7) for k in ks])
    print(f"\n[svc-boost] 저장: {out}")
    print("[svc-boost] 판정: I1 저K 에서 dense_multi 를 넘으면 '메타데이터 집합 신호' 유효 — "
          "합성 시나리오 집합(멀티툴)으로 확장 근거.")


def _smoke():
    print("[smoke] service boost 로직")
    tool_ids = [f"C__svc{i // 3}__api{i}" for i in range(9)]  # 서비스 3개 × API 3개
    groups = build_groups(tool_ids)
    assert len(groups) == 3 and all(len(v) == 3 for v in groups.values())
    got = grouping_diagnostic([{"gold_tools": [tool_ids[0], tool_ids[1]]},
                               {"gold_tools": [tool_ids[0], tool_ids[3]]}], tool_ids)
    assert abs(got - 0.5) < 1e-9, got

    rng = np.random.default_rng(0)
    nq, n_tools = 4, 9
    sem = [rng.random(n_tools).astype(np.float32) for _ in range(nq)]
    gof = {i: service_key(t) for i, t in enumerate(tool_ids)}
    prior = service_boost_prior(sem, groups, gof, n_tools, 0.05)
    assert prior.shape == (nq, n_tools)
    # 같은 서비스 멤버는 같은 prior, 최고점 멤버가 낮은 점수 멤버를 끌어올림
    for qi in range(nq):
        assert abs(prior[qi][0] - prior[qi][2]) < 1e-6
    print("[smoke] OK — 그룹핑·진단·prior 조립 정상")


def main():
    ap = argparse.ArgumentParser(description="문서 메타데이터(서비스 소속) 집합 boost A/B (GPU 불필요)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    run(args.config)


if __name__ == "__main__":
    main()
