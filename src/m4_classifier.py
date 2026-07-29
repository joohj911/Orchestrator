"""m4_classifier.py

명세: spec/rules/classifier.md
역할: 통합 multi-label category classifier 학습 + real prior 생성
산출물: models/class_classifier.pt(+hparams), results/classifier_eval.json, data/class_prior_real.jsonl

CLI: python m4_classifier.py --config config.yaml [--force] [--smoke]

설계 (classifier.md):
  - 입력: query embedding (e5, frozen). 출력: category sigmoid (multi-label).
  - label: gold_categories multi-hot. 모델: frozen e5 위 2-layer MLP. loss: BCE(+불균형 pos_weight).
  - train: I1/I2/I3 train 을 합친 셋(여기선 미사용 benchmark 서브셋 classifier_train.jsonl).
    test 300×3 은 학습·val 미포함 (누출 금지). train/val 분리(seed) — val 은 early stop·calibration.
  - calibration: temperature scaling(val). ECE 전/후 기록.
구현: Claude Code.

주요 DECISION:
  # DECISION: category 라벨 공간 = pool tool categories ∪ train ∪ test gold_categories 합집합(정렬).
  #   근거: fusion real prior 는 pool tool 의 category 확률이 필요 → pool category 를 반드시 포함.
  #   명목 49 와 다를 수 있어 실제 |vocab| 을 기록. train 에 없는 category 는 예측 불가 → 커버리지로 보고.
  # DECISION: test query 임베딩은 m3 캐시(embeddings/queries_{split}.npy) 재사용, 없으면 재임베딩.
  # DECISION: temperature scaling 은 val BCE 최소화 T 를 그리드 탐색(외부 최적화 의존 회피).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import load_config  # noqa: E402


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------- 순수 함수 (torch 불필요, 테스트 가능) -----------------------------
def build_vocab(pool_cats, train_cats, test_cats) -> list[str]:
    """category 라벨 공간: 세 출처의 합집합, 정렬(결정적)."""
    return sorted(set(pool_cats) | set(train_cats) | set(test_cats))


def multihot(cats, vocab_index: dict[str, int], n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    for c in cats:
        if c in vocab_index:
            v[vocab_index[c]] = 1.0
    return v


def pos_weight_from_labels(Y: np.ndarray) -> np.ndarray:
    """BCEWithLogits 용 pos_weight = (neg/pos) per category (pos=0 이면 1)."""
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    w = np.ones_like(pos, dtype=np.float32)
    nz = pos > 0
    w[nz] = (neg[nz] / np.maximum(pos[nz], 1.0)).astype(np.float32)
    return w


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """multi-label ECE: 전 (query,category) 확률을 이진 예측으로 보고 bin 별 |conf-acc| 가중평균."""
    p = probs.reshape(-1)
    y = labels.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / total) * abs(conf - acc)
    return float(ece)


def coverage_report(train_cat_freq: Counter, test_cats: set, vocab: list[str]) -> dict:
    """train 이 test category 를 얼마나 덮는가."""
    covered = [c for c in test_cats if train_cat_freq.get(c, 0) > 0]
    missing = sorted(c for c in test_cats if train_cat_freq.get(c, 0) == 0)
    freqs = [train_cat_freq.get(c, 0) for c in vocab]
    nz = [f for f in freqs if f > 0]
    return {
        "vocab_size": len(vocab),
        "test_categories": len(test_cats),
        "test_cats_covered_by_train": len(covered),
        "test_cats_missing_in_train": missing,
        "train_max_freq": max(nz) if nz else 0,
        "train_min_nonzero_freq": min(nz) if nz else 0,
        "imbalance_ratio": round(max(nz) / min(nz), 1) if nz else 0.0,
    }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """val BCE 최소화하는 스칼라 T 를 그리드 탐색 (coarse→fine)."""
    def bce_at(T):
        p = np.clip(_sigmoid(val_logits / T), 1e-6, 1 - 1e-6)
        return float(-np.mean(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p)))
    grid = np.linspace(0.5, 5.0, 46)
    best = min(grid, key=bce_at)
    fine = np.linspace(max(0.3, best - 0.2), best + 0.2, 41)
    best = min(fine, key=bce_at)
    return float(best)


def per_category_auprc(y_true: np.ndarray, y_score: np.ndarray, vocab: list[str]) -> dict:
    """category 별 AUPRC (test positive 있는 것만). sklearn."""
    from sklearn.metrics import average_precision_score
    out = {}
    for c in range(len(vocab)):
        if y_true[:, c].sum() > 0:
            out[vocab[c]] = round(float(average_precision_score(y_true[:, c], y_score[:, c])), 4)
    return out


# ----------------------------- 학습 (torch) -----------------------------
def train_mlp(X_tr, Y_tr, X_val, Y_val, cfg_cls, seed, use_pos_weight):
    """frozen e5 위 2-layer MLP 학습. (logits_fn, model_state, hparams) 반환."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    d, n_cat = X_tr.shape[1], Y_tr.shape[1]
    hidden = int(cfg_cls["hidden_dim"])

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(float(cfg_cls["dropout"])),
                nn.Linear(hidden, n_cat),
            )

        def forward(self, x):
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    pw = None
    if use_pos_weight:
        pw = torch.tensor(pos_weight_from_labels(Y_tr), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg_cls["lr"]),
                           weight_decay=float(cfg_cls["weight_decay"]))

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_val, dtype=torch.float32, device=device)
    bs = int(cfg_cls["batch_size"])
    n = Xt.shape[0]
    best_val, best_state, patience = float("inf"), None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(int(cfg_cls["epochs"])):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), Yv).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, patience = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= int(cfg_cls["early_stop_patience"]):
                break
    model.load_state_dict(best_state)
    model.eval()

    def logits_fn(X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

    hparams = {"hidden_dim": hidden, "dropout": float(cfg_cls["dropout"]), "lr": float(cfg_cls["lr"]),
               "weight_decay": float(cfg_cls["weight_decay"]), "batch_size": bs,
               "best_val_bce": round(best_val, 6), "pos_weight_used": bool(use_pos_weight),
               "input_dim": d, "n_categories": n_cat}
    return logits_fn, best_state, hparams


def run(config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    splits = cfg["experiment"]["splits"]
    ccls = cfg["classifier"]
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    models_dir = cfg["paths"]["models_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    os.makedirs(models_dir, exist_ok=True)

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    pool_cats = {t["category"] for t in tools}

    train_path = cfg["paths"].get("classifier_train", "")
    if not (train_path and os.path.isfile(train_path)):
        print(f"[m4] classifier_train 없음: {train_path}. prepare_toolbench_hf.py 먼저 실행.", file=sys.stderr)
        sys.exit(1)
    train_rows_raw = _read_jsonl(train_path)

    # test 쿼리 + gold_categories
    test_rows: dict[str, list[dict]] = {}
    test_cats: set[str] = set()
    for s in splits:
        rows = _read_jsonl(os.path.join(data_dir, f"queries_{s}.jsonl"))
        test_rows[s] = rows
        for q in rows:
            test_cats.update(q.get("gold_categories", []))
    all_test_ids = {str(q["query_id"]) for s in splits for q in test_rows[s]}

    # classifier_train 은 benchmark superset → 실제 test 로 샘플된 query_id 제외(누출 0).
    train_rows = [r for r in train_rows_raw if str(r["query_id"]) not in all_test_ids]
    n_removed = len(train_rows_raw) - len(train_rows)
    print(f"[m4] classifier_train {len(train_rows_raw)} 후보 중 test 중복 {n_removed}개 제외 → 학습 {len(train_rows)}")
    # 실제 학습셋을 감사 가능하게 기록 (verify_m4 가 이 파일로 누출 검사).
    used_path = os.path.join(data_dir, "classifier_train_used.jsonl")
    with open(used_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cat_freq = Counter(c for r in train_rows for c in r.get("gold_categories", []))
    vocab = build_vocab(pool_cats, set(train_cat_freq), test_cats)
    vindex = {c: i for i, c in enumerate(vocab)}
    cov = coverage_report(train_cat_freq, test_cats, vocab)
    print(f"[m4] vocab {len(vocab)} categories | train {len(train_rows)} queries")
    print(f"[m4] 커버리지: test category {cov['test_categories']}개 중 "
          f"train 이 {cov['test_cats_covered_by_train']}개 덮음, 미커버 {len(cov['test_cats_missing_in_train'])}개")
    if cov["test_cats_missing_in_train"]:
        print(f"     미커버(예): {cov['test_cats_missing_in_train'][:8]}")
    print(f"[m4] train 불균형비(최다:최소) ≈ {cov['imbalance_ratio']}")

    # --- 누출 안전 확인: 필터 후 train 에 test id 가 없어야 함 (verify_m4 가 최종 검증) ---
    leak = {str(r["query_id"]) for r in train_rows} & all_test_ids
    if leak:
        print(f"[m4] 치명적: 필터 후에도 test query_id 잔존 {len(leak)}: {list(leak)[:3]}", file=sys.stderr)
        sys.exit(1)

    # --- 임베딩 (e5 frozen) ---
    from utils.embed import embed_queries
    X_train = embed_queries([r["query"] for r in train_rows], cfg)
    Y_train = np.stack([multihot(r.get("gold_categories", []), vindex, len(vocab)) for r in train_rows])

    # test 임베딩: m3 캐시 재사용
    test_emb: dict[str, np.ndarray] = {}
    for s in splits:
        cache = os.path.join(emb_dir, f"queries_{s}.npy")
        if not force and os.path.isfile(cache):
            test_emb[s] = np.load(cache)
        else:
            test_emb[s] = embed_queries([q["query"] for q in test_rows[s]], cfg)

    # train/val 분리 (seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_rows))
    n_val = max(1, int(round(len(train_rows) * float(ccls["val_frac"]))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Ytr = X_train[tr_idx], Y_train[tr_idx]
    Xval, Yval = X_train[val_idx], Y_train[val_idx]

    use_pw = cov["imbalance_ratio"] > float(ccls["imbalance_ratio_threshold"])
    print(f"[m4] pos_weight(class weight) {'적용' if use_pw else '미적용'} "
          f"(불균형비 {cov['imbalance_ratio']} vs 임계 {ccls['imbalance_ratio_threshold']})")

    logits_fn, state, hparams = train_mlp(Xtr, Ytr, Xval, Yval, ccls, seed, use_pw)

    # --- calibration (val) ---
    val_logits = logits_fn(Xval)
    ece_before = expected_calibration_error(_sigmoid(val_logits), Yval)
    T = fit_temperature(val_logits, Yval)
    ece_after = expected_calibration_error(_sigmoid(val_logits / T), Yval)
    print(f"[m4] temperature T={round(T,3)} | ECE {round(ece_before,4)} → {round(ece_after,4)}")

    # --- test 평가 + class_prior_real ---
    from sklearn.metrics import f1_score
    y_true_all, y_prob_all = [], []
    prior_out = []
    for s in splits:
        logits = logits_fn(test_emb[s])
        probs = _sigmoid(logits / T)
        for i, q in enumerate(test_rows[s]):
            yt = multihot(q.get("gold_categories", []), vindex, len(vocab))
            y_true_all.append(yt)
            y_prob_all.append(probs[i])
            prior_out.append({"query_id": q["query_id"], "split": s,
                              "prior": {vocab[c]: round(float(probs[i][c]), 6) for c in range(len(vocab))}})
    Yt = np.stack(y_true_all)
    Yp = np.stack(y_prob_all)
    thr = float(ccls["pred_threshold"])
    pred = (Yp >= thr).astype(int)
    micro_f1 = float(f1_score(Yt, pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(Yt, pred, average="macro", zero_division=0))
    auprc = per_category_auprc(Yt, Yp, vocab)
    mean_auprc = round(float(np.mean(list(auprc.values()))), 4) if auprc else 0.0
    test_ece = expected_calibration_error(Yp, Yt)

    # --- 저장 ---
    eval_json = {
        "vocab": vocab, "coverage": cov,
        "micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
        "mean_auprc": mean_auprc, "per_category_auprc": auprc,
        "ece_val_before": round(ece_before, 4), "ece_val_after": round(ece_after, 4),
        "ece_test": round(test_ece, 4), "temperature": round(T, 4),
        "train_category_freq": dict(train_cat_freq.most_common()),
        "hparams": hparams, "n_train": len(tr_idx), "n_val": len(val_idx),
        "n_test": len(prior_out), "pred_threshold": thr,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "classifier_eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, "class_prior_real.jsonl"), "w", encoding="utf-8") as f:
        for r in prior_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import torch
    torch.save(state, os.path.join(models_dir, "class_classifier.pt"))
    with open(os.path.join(models_dir, "class_classifier_hparams.json"), "w", encoding="utf-8") as f:
        json.dump({**hparams, "vocab": vocab, "temperature": T, "seed": seed}, f, ensure_ascii=False, indent=2)

    print(f"[m4] micro-F1 {round(micro_f1,4)}, macro-F1 {round(macro_f1,4)}, mean AUPRC {mean_auprc}")
    print(f"[m4] 완료: classifier_eval.json, class_prior_real.jsonl ({len(prior_out)} test), 모델 저장.")


def _smoke() -> None:
    """torch 있으면 소량 학습까지, 없으면 순수 함수만 점검."""
    print("[smoke] m4 순수 함수 점검")
    vocab = build_vocab({"A", "B"}, {"B", "C"}, {"A", "C", "D"})
    assert vocab == ["A", "B", "C", "D"]
    vi = {c: i for i, c in enumerate(vocab)}
    assert list(multihot(["A", "C"], vi, 4)) == [1, 0, 1, 0]
    Y = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    pw = pos_weight_from_labels(Y)
    assert pw.shape == (4,) and pw[0] == 0.5 and pw[3] == 1.0  # cat0: neg1/pos2; cat3: no pos → 1
    ece = expected_calibration_error(np.array([[0.9, 0.1]]), np.array([[1.0, 0.0]]))
    assert 0.0 <= ece <= 1.0
    T = fit_temperature(np.array([[2.0, -2.0], [3.0, -1.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert 0.3 <= T <= 5.0
    cov = coverage_report(Counter({"A": 5, "B": 1}), {"A", "C"}, vocab)
    assert cov["test_cats_missing_in_train"] == ["C"] and cov["imbalance_ratio"] == 5.0
    print("[smoke] OK — vocab/multihot/pos_weight/ECE/temperature/coverage 정상")
    try:
        import torch  # noqa: F401
    except Exception:
        print("[smoke] torch 미설치 → 학습 경로는 서버에서 검증")
        return
    rng = np.random.default_rng(0)
    d, n = 16, 60
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = (rng.random((n, 4)) > 0.6).astype(np.float32)
    lf, st, hp = train_mlp(X[:48], Y[:48], X[48:], Y[48:],
                           {"hidden_dim": 8, "dropout": 0.1, "lr": 0.01, "weight_decay": 0.0,
                            "epochs": 5, "batch_size": 16, "early_stop_patience": 3}, 42, True)
    assert lf(X[:2]).shape == (2, 4)
    print("[smoke] OK — torch 학습 경로 정상")


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 multi-label category classifier")
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
