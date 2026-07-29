"""m4_classifier.py

명세: spec/rules/classifier.md
역할: 통합 multi-label category classifier 학습 + real prior 생성
산출물: models/class_classifier.pt(+hparams), results/classifier_eval.json, data/class_prior_real.jsonl

CLI: python m4_classifier.py --config config.yaml [--force] [--smoke]

설계:
  - 입력: 질의. 출력: category sigmoid (multi-label). label: gold_categories multi-hot.
  - method(config.classifier.method):
      · "lora": e5 인코더를 LoRA 로 미세조정 + head. (표현 한계 대응, 권장)
      · "frozen_mlp": frozen e5 임베딩 위 2-layer MLP. (spec 원안, 비교/폴백)
  - train: benchmark superset 에서 test query_id 제외(누출 0). train/val 분리(seed).
  - calibration: temperature scaling(val). ECE 전/후 기록.
구현: Claude Code.

주요 DECISION:
  # DECISION: 진단(고빈도 category 저AUPRC, 예: Tools freq 235 → AUPRC 0.27)상 frozen e5 표현이
  #   catch-all category 를 못 가른다 → method=lora 를 기본. spec 의 frozen+2layer 는 owner 결정으로
  #   확장(frozen_mlp 로 여전히 선택·비교 가능).
  # DECISION: category vocab = pool ∪ train ∪ test gold_categories (fusion 이 pool category 필요).
  # DECISION: temperature scaling 은 val BCE 최소 T 그리드 탐색.
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
from utils.embed import PREFIX_QUERY  # noqa: E402


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------- 순수 함수 (torch 불필요) -----------------------------
def build_vocab(pool_cats, train_cats, test_cats) -> list[str]:
    return sorted(set(pool_cats) | set(train_cats) | set(test_cats))


def multihot(cats, vocab_index, n) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    for c in cats:
        if c in vocab_index:
            v[vocab_index[c]] = 1.0
    return v


def pos_weight_from_labels(Y) -> np.ndarray:
    pos = Y.sum(axis=0)
    neg = Y.shape[0] - pos
    w = np.ones_like(pos, dtype=np.float32)
    nz = pos > 0
    w[nz] = (neg[nz] / np.maximum(pos[nz], 1.0)).astype(np.float32)
    return w


def expected_calibration_error(probs, labels, n_bins=10) -> float:
    p = probs.reshape(-1)
    y = labels.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def coverage_report(train_cat_freq, test_cats, vocab) -> dict:
    covered = [c for c in test_cats if train_cat_freq.get(c, 0) > 0]
    missing = sorted(c for c in test_cats if train_cat_freq.get(c, 0) == 0)
    nz = [train_cat_freq.get(c, 0) for c in vocab if train_cat_freq.get(c, 0) > 0]
    return {"vocab_size": len(vocab), "test_categories": len(test_cats),
            "test_cats_covered_by_train": len(covered), "test_cats_missing_in_train": missing,
            "train_max_freq": max(nz) if nz else 0, "train_min_nonzero_freq": min(nz) if nz else 0,
            "imbalance_ratio": round(max(nz) / min(nz), 1) if nz else 0.0}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(val_logits, val_labels) -> float:
    def bce_at(T):
        p = np.clip(_sigmoid(val_logits / T), 1e-6, 1 - 1e-6)
        return float(-np.mean(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p)))
    grid = np.linspace(0.5, 5.0, 46)
    best = min(grid, key=bce_at)
    best = min(np.linspace(max(0.3, best - 0.2), best + 0.2, 41), key=bce_at)
    return float(best)


def per_category_auprc(y_true, y_score, vocab) -> dict:
    from sklearn.metrics import average_precision_score
    return {vocab[c]: round(float(average_precision_score(y_true[:, c], y_score[:, c])), 4)
            for c in range(len(vocab)) if y_true[:, c].sum() > 0}


def _average_pool(last_hidden, attention_mask):
    import torch
    mask = attention_mask[..., None].bool()
    summed = last_hidden.masked_fill(~mask, 0.0).sum(dim=1)
    counts = attention_mask.sum(dim=1)[..., None].clamp(min=1)
    return summed / counts


# ----------------------------- frozen_mlp 학습 -----------------------------
def train_mlp(X_tr, Y_tr, X_val, Y_val, cfg_cls, seed, use_pos_weight):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    d, n_cat = X_tr.shape[1], Y_tr.shape[1]
    hidden = int(cfg_cls["hidden_dim"])

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                     nn.Dropout(float(cfg_cls["dropout"])), nn.Linear(hidden, n_cat))

        def forward(self, x):
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    pw = torch.tensor(pos_weight_from_labels(Y_tr), dtype=torch.float32, device=device) if use_pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg_cls["lr"]), weight_decay=float(cfg_cls["weight_decay"]))
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device); Yt = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device); Yv = torch.tensor(Y_val, dtype=torch.float32, device=device)
    bs, n = int(cfg_cls["batch_size"]), Xt.shape[0]
    best_val, best_state, patience = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(cfg_cls["epochs"])):
        model.train()
        for i in range(0, n, bs):
            idx = torch.randperm(n, generator=g)[i:i + bs].to(device)
            opt.zero_grad(); loss_fn(model(Xt[idx]), Yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), Yv).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, patience = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= int(cfg_cls["early_stop_patience"]):
                break
    model.load_state_dict(best_state); model.eval()

    def logits_of(X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

    hp = {"method": "frozen_mlp", "hidden_dim": hidden, "dropout": float(cfg_cls["dropout"]),
          "lr": float(cfg_cls["lr"]), "best_val_bce": round(best_val, 6),
          "pos_weight_used": bool(use_pos_weight), "input_dim": d, "n_categories": n_cat}
    return logits_of, {"model": best_state}, hp


# ----------------------------- lora 학습 -----------------------------
def train_lora(texts_tr, Y_tr, texts_val, Y_val, test_texts_by_split, model_id, cfg_cls, seed, use_pos_weight):
    """e5 인코더를 LoRA 로 미세조정 + linear head. (val_logits, test_logits_by_split, save, hparams)."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    try:
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
    except Exception as e:  # noqa: BLE001
        print(f"[m4] peft 임포트 실패: {e}. classifier.method=frozen_mlp 로 바꾸거나 peft 설치.", file=sys.stderr)
        raise

    lc = cfg_cls["lora"]
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_cat = Y_tr.shape[1]
    max_len = int(lc["max_length"])

    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModel.from_pretrained(model_id)
    peft_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=int(lc["r"]), lora_alpha=int(lc["alpha"]),
                          lora_dropout=float(lc["dropout"]), target_modules=list(lc["target_modules"]),
                          bias="none")
    enc = get_peft_model(base, peft_cfg).to(device)
    head = nn.Linear(base.config.hidden_size, n_cat).to(device)

    def encode(texts, train_mode):
        enc.train(train_mode); head.train(train_mode)
        enc_in = tok([PREFIX_QUERY + t for t in texts], max_length=max_len, padding=True,
                     truncation=True, return_tensors="pt").to(device)
        out = enc(**enc_in)
        pooled = _average_pool(out.last_hidden_state, enc_in["attention_mask"])
        return head(pooled)

    def logits_np(texts, bs=32):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                outs.append(encode(texts[i:i + bs], False).cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, n_cat), dtype=np.float32)

    trainable = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=float(lc["lr"]))
    pw = torch.tensor(pos_weight_from_labels(Y_tr), dtype=torch.float32, device=device) if use_pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_val, dtype=torch.float32, device=device)
    bs, n = int(lc["batch_size"]), len(texts_tr)
    best_val, best_state, patience = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(int(lc["epochs"])):
        order = rng.permutation(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            logits = encode([texts_tr[j] for j in idx], True)
            loss_fn(logits, Yt[idx]).backward()
            opt.step()
        with torch.no_grad():
            vloss = loss_fn(torch.tensor(logits_np(texts_val), device=device), Yv).item()
        print(f"[m4/lora] epoch {epoch+1} val_bce {round(vloss,4)}")
        if vloss < best_val - 1e-5:
            best_val, patience = vloss, 0
            best_state = {"lora": {k: v.cpu().clone() for k, v in get_peft_model_state_dict(enc).items()},
                          "head": {k: v.cpu().clone() for k, v in head.state_dict().items()}}
        else:
            patience += 1
            if patience >= int(lc["early_stop_patience"]):
                print(f"[m4/lora] early stop @ epoch {epoch+1}")
                break

    val_logits = logits_np(texts_val)
    test_logits = {s: logits_np(txts) for s, txts in test_texts_by_split.items()}
    hp = {"method": "lora", "r": int(lc["r"]), "alpha": int(lc["alpha"]), "lora_dropout": float(lc["dropout"]),
          "target_modules": list(lc["target_modules"]), "lr": float(lc["lr"]), "max_length": max_len,
          "best_val_bce": round(best_val, 6), "pos_weight_used": bool(use_pos_weight), "n_categories": n_cat}
    return val_logits, test_logits, best_state, hp


def run(config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    splits = cfg["experiment"]["splits"]
    ccls = cfg["classifier"]
    method = ccls.get("method", "lora")
    data_dir = cfg["paths"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    models_dir = cfg["paths"]["models_dir"]
    emb_dir = os.path.join(data_dir, "embeddings")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    model_id = cfg["models"]["embedder"]

    tools = _read_jsonl(os.path.join(data_dir, "tools.jsonl"))
    pool_cats = {t["category"] for t in tools}

    train_path = cfg["paths"].get("classifier_train", "")
    if not (train_path and os.path.isfile(train_path)):
        print(f"[m4] classifier_train 없음: {train_path}. prepare_toolbench_hf.py 먼저 실행.", file=sys.stderr)
        sys.exit(1)
    train_rows_raw = _read_jsonl(train_path)

    test_rows, test_cats = {}, set()
    for s in splits:
        rows = _read_jsonl(os.path.join(data_dir, f"queries_{s}.jsonl"))
        test_rows[s] = rows
        for q in rows:
            test_cats.update(q.get("gold_categories", []))
    all_test_ids = {str(q["query_id"]) for s in splits for q in test_rows[s]}

    train_rows = [r for r in train_rows_raw if str(r["query_id"]) not in all_test_ids]
    print(f"[m4] method={method} | classifier_train {len(train_rows_raw)} 후보 중 test 중복 "
          f"{len(train_rows_raw)-len(train_rows)}개 제외 → 학습 {len(train_rows)}")

    # --- tool example 증강 (config.classifier.augment_tool_examples) ---
    # 배경: oracle→real fusion recall gap 이 커서(2026-07 실측) classifier 가 병목.
    # tools_examples.jsonl(500 tool × 5 발화)은 pool 49 category 를 전부 덮는 in-domain
    # 데이터이고, M2 게이트가 test 유사도 누출(max_leak_sim ≤ 0.92)을 이미 검증했다.
    # 라벨은 해당 tool 의 category 1개 (single-label 이지만 category 인식 신호로 유효).
    # val/calibration 에는 넣지 않는다 (benchmark 분포 유지) — 아래 split 후 train 쪽에만 결합.
    aug_base: list[dict] = []
    aug_rows: list[dict] = []
    if ccls.get("augment_tool_examples"):
        ex_path = cfg["paths"].get("examples_file", "") or ""
        if not (ex_path and os.path.isfile(ex_path)):
            ex_path = os.path.join(data_dir, "tools_examples.jsonl")
        cat_of = {t["id"]: t["category"] for t in tools}
        for r in _read_jsonl(ex_path):
            cat = cat_of.get(r["tool_id"])
            if cat is None:
                continue
            for j, ex in enumerate(r.get("examples", [])):
                aug_base.append({"query_id": f"augex_{r['tool_id']}_{j}", "query": ex,
                                 "gold_categories": [cat], "source": "tool_example_aug"})
        repeat = int(ccls.get("augment_repeat", 1))
        aug_rows = aug_base * repeat
        print(f"[m4] tool example 증강: {len(aug_base)}건 × repeat {repeat} = +{len(aug_rows)} "
              f"(val 미포함, M2 누출 게이트 통과분)")

    with open(os.path.join(data_dir, "classifier_train_used.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows + aug_base:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_cat_freq = Counter(c for r in train_rows + aug_rows for c in r.get("gold_categories", []))
    vocab = build_vocab(pool_cats, set(train_cat_freq), test_cats)
    vindex = {c: i for i, c in enumerate(vocab)}
    cov = coverage_report(train_cat_freq, test_cats, vocab)
    print(f"[m4] vocab {len(vocab)} | 커버리지: test {cov['test_categories']}개 중 "
          f"{cov['test_cats_covered_by_train']}개 덮음, 미커버 {len(cov['test_cats_missing_in_train'])} "
          f"{cov['test_cats_missing_in_train'][:6]} | 불균형비 {cov['imbalance_ratio']}")

    leak = {str(r["query_id"]) for r in train_rows} & all_test_ids
    if leak:
        print(f"[m4] 치명적: 필터 후에도 test id 잔존 {len(leak)}", file=sys.stderr)
        sys.exit(1)

    Y_train = np.stack([multihot(r.get("gold_categories", []), vindex, len(vocab)) for r in train_rows])
    Y_aug = (np.stack([multihot(r["gold_categories"], vindex, len(vocab)) for r in aug_rows])
             if aug_rows else np.zeros((0, len(vocab)), dtype=np.float32))
    use_pw = cov["imbalance_ratio"] > float(ccls["imbalance_ratio_threshold"])
    print(f"[m4] pos_weight {'적용' if use_pw else '미적용'} (불균형비 {cov['imbalance_ratio']})")

    # train/val 분리 (val 은 benchmark 쿼리만 — 증강 rows 는 아래에서 train 쪽에만 결합)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_rows))
    n_val = max(1, int(round(len(train_rows) * float(ccls["val_frac"]))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Yval = Y_train[val_idx]
    Ytr = np.concatenate([Y_train[tr_idx], Y_aug]) if len(Y_aug) else Y_train[tr_idx]

    # --- method 별 학습 → val_logits, test_logits_by_split, save ---
    if method == "lora":
        texts = [r["query"] for r in train_rows]
        aug_texts = [r["query"] for r in aug_rows]
        test_texts = {s: [q["query"] for q in test_rows[s]] for s in splits}
        val_logits, test_logits, save_obj, hparams = train_lora(
            [texts[i] for i in tr_idx] + aug_texts, Ytr, [texts[i] for i in val_idx], Yval,
            test_texts, model_id, ccls, seed, use_pw)
    elif method == "frozen_mlp":
        from utils.embed import embed_queries
        X_train = embed_queries([r["query"] for r in train_rows], cfg)
        X_aug = embed_queries([r["query"] for r in aug_rows], cfg) if aug_rows else None
        Xtr = np.concatenate([X_train[tr_idx], X_aug]) if X_aug is not None else X_train[tr_idx]
        test_emb = {}
        for s in splits:
            cache = os.path.join(emb_dir, f"queries_{s}.npy")
            test_emb[s] = np.load(cache) if (not force and os.path.isfile(cache)) else embed_queries([q["query"] for q in test_rows[s]], cfg)
        logits_of, save_obj, hparams = train_mlp(Xtr, Ytr, X_train[val_idx], Yval, ccls, seed, use_pw)
        val_logits = logits_of(X_train[val_idx])
        test_logits = {s: logits_of(test_emb[s]) for s in splits}
    else:
        print(f"[m4] 알 수 없는 method: {method}", file=sys.stderr)
        sys.exit(1)

    # --- calibration (val) ---
    ece_before = expected_calibration_error(_sigmoid(val_logits), Yval)
    T = fit_temperature(val_logits, Yval)
    ece_after = expected_calibration_error(_sigmoid(val_logits / T), Yval)
    print(f"[m4] T={round(T,3)} | ECE {round(ece_before,4)} → {round(ece_after,4)}")

    # --- test 평가 + class_prior_real ---
    from sklearn.metrics import f1_score
    y_true_all, y_prob_all, prior_out = [], [], []
    for s in splits:
        probs = _sigmoid(test_logits[s] / T)
        for i, q in enumerate(test_rows[s]):
            y_true_all.append(multihot(q.get("gold_categories", []), vindex, len(vocab)))
            y_prob_all.append(probs[i])
            prior_out.append({"query_id": q["query_id"], "split": s,
                              "prior": {vocab[c]: round(float(probs[i][c]), 6) for c in range(len(vocab))}})
    Yt, Yp = np.stack(y_true_all), np.stack(y_prob_all)
    thr = float(ccls["pred_threshold"])
    pred = (Yp >= thr).astype(int)
    micro_f1 = float(f1_score(Yt, pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(Yt, pred, average="macro", zero_division=0))
    auprc = per_category_auprc(Yt, Yp, vocab)
    mean_auprc = round(float(np.mean(list(auprc.values()))), 4) if auprc else 0.0
    test_ece = expected_calibration_error(Yp, Yt)

    eval_json = {"method": method, "vocab": vocab, "coverage": cov,
                 "micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
                 "mean_auprc": mean_auprc, "per_category_auprc": auprc,
                 "ece_val_before": round(ece_before, 4), "ece_val_after": round(ece_after, 4),
                 "ece_test": round(test_ece, 4), "temperature": round(T, 4),
                 "train_category_freq": dict(train_cat_freq.most_common()),
                 "hparams": hparams, "n_train": len(tr_idx), "n_aug": len(aug_rows),
                 "n_val": len(val_idx), "n_test": len(prior_out)}
    with open(os.path.join(results_dir, "classifier_eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, "class_prior_real.jsonl"), "w", encoding="utf-8") as f:
        for r in prior_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import torch
    torch.save(save_obj, os.path.join(models_dir, "class_classifier.pt"))
    with open(os.path.join(models_dir, "class_classifier_hparams.json"), "w", encoding="utf-8") as f:
        json.dump({**hparams, "vocab": vocab, "temperature": T, "seed": seed}, f, ensure_ascii=False, indent=2)

    print(f"[m4] micro-F1 {round(micro_f1,4)}, macro-F1 {round(macro_f1,4)}, mean AUPRC {mean_auprc}")
    print(f"[m4] 완료: classifier_eval.json, class_prior_real.jsonl ({len(prior_out)} test), 모델 저장.")


def _smoke() -> None:
    print("[smoke] m4 순수 함수 점검")
    vocab = build_vocab({"A", "B"}, {"B", "C"}, {"A", "C", "D"})
    assert vocab == ["A", "B", "C", "D"]
    vi = {c: i for i, c in enumerate(vocab)}
    assert list(multihot(["A", "C"], vi, 4)) == [1, 0, 1, 0]
    Y = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    pw = pos_weight_from_labels(Y)
    assert pw[0] == 0.5 and pw[3] == 1.0
    assert 0.0 <= expected_calibration_error(np.array([[0.9, 0.1]]), np.array([[1.0, 0.0]])) <= 1.0
    assert 0.3 <= fit_temperature(np.array([[2.0, -2.0]]), np.array([[1.0, 0.0]])) <= 5.0
    cov = coverage_report(Counter({"A": 5, "B": 1}), {"A", "C"}, vocab)
    assert cov["test_cats_missing_in_train"] == ["C"] and cov["imbalance_ratio"] == 5.0
    print("[smoke] OK — 순수 함수 정상 (lora/frozen 학습은 서버에서)")


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 multi-label category classifier (lora | frozen_mlp)")
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
