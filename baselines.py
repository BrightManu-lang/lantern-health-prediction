from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score)

from utils import ensure_dir, brier_score, expected_calibration_error, save_json


# Metric helpers
def _safe_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def _safe_ap(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(average_precision_score(y_true_bin, y_score))


def _binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def _binary_ece(y_true_bin: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    p2 = np.vstack([1.0 - y_prob, y_prob]).T  # [N,2]
    return float(expected_calibration_error(y_true_bin, p2, n_bins=n_bins))


# risk stratification helpers
def _risk_band_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    if N == 0:
        return {"n_bins": int(n_bins), "overall_event_rate": float("nan"), "bins": []}

    overall_rate = float(y_true.mean())
    order = np.argsort(-y_prob)  # high -> low
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    edges = np.linspace(0, N, n_bins + 1).round().astype(int)
    edges[-1] = N

    bins = []
    events_total = int(y_true.sum())
    cum_events = 0

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        y_b = y_sorted[lo:hi]
        p_b = p_sorted[lo:hi]

        n_b = int(hi - lo)
        ev_b = int(y_b.sum())
        cum_events += ev_b

        obs = float(ev_b / n_b) if n_b > 0 else float("nan")
        pred = float(np.mean(p_b)) if n_b > 0 else float("nan")
        oe = float(obs / pred) if (pred is not None and pred > 0) else float("nan")
        lift = float(obs / overall_rate) if overall_rate > 0 else float("nan")
        capture = float(cum_events / events_total) if events_total > 0 else float("nan")

        bins.append({
            "bin": int(b + 1),  # 1 = highest risk
            "n": n_b,
            "events": ev_b,
            "event_rate": obs,
            "mean_pred": pred,
            "OE": oe,
            "lift_vs_overall": lift,
            "cum_event_capture": capture,
            "min_score": float(np.min(p_b)),
            "max_score": float(np.max(p_b)),
        })

    return {
        "n_bins": int(n_bins),
        "overall_event_rate": overall_rate,
        "total_events": int(events_total),
        "bins": bins,
    }


def _topk_capture(y_true: np.ndarray, y_prob: np.ndarray, top_fracs=(0.01, 0.02, 0.05, 0.10)) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    events_total = int(y_true.sum())
    out = {}

    if N == 0 or events_total == 0:
        for f in top_fracs:
            out[f"top_{int(f*100)}pct_capture"] = float("nan")
            out[f"top_{int(f*100)}pct_flag_rate"] = float("nan")
        return out

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    for f in top_fracs:
        k = max(1, int(round(f * N)))
        out[f"top_{int(f*100)}pct_capture"] = float(y_sorted[:k].sum() / events_total)
        out[f"top_{int(f*100)}pct_flag_rate"] = float(k / N)
    return out


def _lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    if N == 0:
        return float("nan")
    overall = float(y_true.mean())
    if overall <= 0:
        return float("nan")
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    k = max(1, int(round(top_frac * N)))
    top_rate = float(y_sorted[:k].mean())
    return float(top_rate / overall)


def _risk_pricing_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "risk_bands": _risk_band_table(y_true, y_prob, n_bins=n_bins),
        "topk_capture": _topk_capture(y_true, y_prob),
        "lift_at_10pct": _lift_at_k(y_true, y_prob, top_frac=0.10),
        "lift_at_5pct": _lift_at_k(y_true, y_prob, top_frac=0.05),
    }

# Operational thresholding (FLAG-RATE)
def pick_threshold_by_flag_rate(y_prob: np.ndarray, flag_rate: float) -> dict:
    """
    Choose threshold so that approx flag_rate of examples are flagged (>= threshold).
    Uses quantile on VAL.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.size == 0:
        return {"threshold": float("inf"), "flag_rate": 0.0}

    flag_rate = float(flag_rate)
    if flag_rate <= 0.0:
        return {"threshold": float("inf"), "flag_rate": 0.0}
    if flag_rate >= 1.0:
        return {"threshold": float("-inf"), "flag_rate": 1.0}

    thr = float(np.quantile(y_prob, 1.0 - flag_rate, method="linear"))
    achieved = float((y_prob >= thr).mean())
    return {"threshold": thr, "flag_rate": achieved}


def threshold_operating_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    y_hat = (y_prob >= thr).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (rec + spec)

    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "balanced_accuracy": float(bal_acc),
        "flag_rate": float(y_hat.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

# baseline metrics helper
def compute_baseline_metrics(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, *, 
                             y_true_val: Optional[np.ndarray] = None, y_prob_val: Optional[np.ndarray] = None, 
                             include_pricing: bool = False, include_operational: bool = False,
                             death_flag_rate: float = 0.01, severe_flag_rate: float = 0.10):
    
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    # Binary endpoints (2=Severe, 3=Death)
    yD = (y_true == 3).astype(int)
    yS = (y_true == 2).astype(int)
    pD = y_prob[:, 3]
    pS = y_prob[:, 2]

    risk_endpoints = {
        "Death": {
            "prevalence": float(yD.mean()),
            "AUROC": _safe_auc(yD, pD),
            "PRAUC": _safe_ap(yD, pD),
            "Brier": _binary_brier(yD, pD),
            "ECE": _binary_ece(yD, pD, n_bins=10),
        },
        "Severe": {
            "prevalence": float(yS.mean()),
            "AUROC": _safe_auc(yS, pS),
            "PRAUC": _safe_ap(yS, pS),
            "Brier": _binary_brier(yS, pS),
            "ECE": _binary_ece(yS, pS, n_bins=10),
        },
    }

    out = {
        "Model": model_name,
        "Multiclass": {
            "Brier": float(brier),
            "ECE": float(ece),
        },
        "RiskEndpoints": risk_endpoints,
        # Backward-compat
        "Endpoints": {
            "Death": {
                "AUROC": risk_endpoints["Death"]["AUROC"],
                "PRAUC": risk_endpoints["Death"]["PRAUC"],
                "prevalence": risk_endpoints["Death"]["prevalence"],
            },
            "Severe": {
                "AUROC": risk_endpoints["Severe"]["AUROC"],
                "PRAUC": risk_endpoints["Severe"]["PRAUC"],
                "prevalence": risk_endpoints["Severe"]["prevalence"],
            },
        },
    }

    if include_pricing:
        out["PricingStratification"] = {
            "Death": _risk_pricing_summary(yD, pD, n_bins=10),
            "Severe": _risk_pricing_summary(yS, pS, n_bins=10),
        }

    if include_operational:
        if (y_true_val is None) or (y_prob_val is None):
            raise ValueError("include_operational=True requires y_true_val and y_prob_val.")

        y_true_val = np.asarray(y_true_val, dtype=int)
        y_prob_val = np.asarray(y_prob_val, dtype=float)

        yD_val = (y_true_val == 3).astype(int)
        yS_val = (y_true_val == 2).astype(int)
        pD_val = y_prob_val[:, 3]
        pS_val = y_prob_val[:, 2]

        death_sel = pick_threshold_by_flag_rate(pD_val, flag_rate=death_flag_rate)
        severe_sel = pick_threshold_by_flag_rate(pS_val, flag_rate=severe_flag_rate)

        out["Operational"] = {
            "death_flag_rate_val_target": float(death_flag_rate),
            "severe_flag_rate_val_target": float(severe_flag_rate),
            "Death_threshold_selection_val": death_sel,
            "Severe_threshold_selection_val": severe_sel,
            "Death_operating_test": threshold_operating_metrics(yD, pD, death_sel["threshold"]),
            "Severe_operating_test": threshold_operating_metrics(yS, pS, severe_sel["threshold"]),
        }

    return out


# LSP baseline
def _heuristic_probs_for_split(df_split: pd.DataFrame, class_prior: np.ndarray, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    "Last state persists":
      - first record per patient: train marginal probs
      - subsequent: one-hot(previous state's LABEL), where "state" here is df[target_col]
    """
    df_work = df_split.reset_index(drop=True).copy()
    df_work["_orig_idx"] = np.arange(len(df_work), dtype=np.int64)
    df_sorted = df_work.sort_values(["patient_idx", "time"], kind="mergesort").copy()

    y_true_sorted = df_sorted[target_col].to_numpy(dtype=int)
    K = int(class_prior.shape[0])
    y_prob_sorted = np.zeros((len(df_sorted), K), dtype=float)

    idx = 0
    for _, g in df_sorted.groupby("patient_idx", sort=False):
        states = g[target_col].to_numpy(dtype=int)

        # first record: prior
        y_prob_sorted[idx] = class_prior
        idx += 1

        # subsequent: previous state's one-hot
        for t in range(1, len(states)):
            prev_state = int(states[t - 1])
            one_hot = np.zeros(K, dtype=float)
            if 0 <= prev_state < K:
                one_hot[prev_state] = 1.0
            y_prob_sorted[idx] = one_hot
            idx += 1

    # restore original row order
    orig_idx = df_sorted["_orig_idx"].to_numpy()
    y_prob = np.zeros((len(df_work), K), dtype=float)
    y_true = np.zeros((len(df_work),), dtype=int)
    y_prob[orig_idx] = y_prob_sorted
    y_true[orig_idx] = y_true_sorted
    return y_true, y_prob


def run_heuristic_baselines(cfg, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, outdir: Path):
    outdir = ensure_dir(outdir / "baselines_heuristics")

    class_counts = df_train[cfg.target_col].value_counts().sort_index()
    K = int(class_counts.index.max()) + 1
    class_prior = np.zeros(K, dtype=float)
    class_prior[class_counts.index.values] = class_counts.values / class_counts.values.sum()

    # probs for VAL + TEST (needed for flag-rate operational thresholding)
    y_true_val, y_prob_val = _heuristic_probs_for_split(df_val, class_prior, cfg.target_col)
    y_true_test, y_prob_test = _heuristic_probs_for_split(df_test, class_prior, cfg.target_col)

    np.savez(outdir / "preds_test.npz", y_true=y_true_test, y_prob=y_prob_test)

    metrics = compute_baseline_metrics(y_true=y_true_test, y_prob=y_prob_test, model_name="Heuristic_LastStatePersists",
        y_true_val=y_true_val, y_prob_val=y_prob_val, include_pricing=True, include_operational=True,
        death_flag_rate=float(getattr(cfg, "death_flag_rate", 0.01)),
        severe_flag_rate=float(getattr(cfg, "severe_flag_rate", 0.10)))
    
    save_json(metrics, outdir / "heuristic_metrics.json")   


# Logistic baseline
def _make_safe_logreg(max_iter: int = 2000) -> LogisticRegression:
    return LogisticRegression(solver="saga", penalty="l2", max_iter=max_iter, n_jobs=1, verbose=0)

def run_logistic_baselines(cfg, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                           feature_cols: List[str], outdir: Path):
    outdir = ensure_dir(outdir / "baselines_logistic")

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = df_train[cfg.target_col].to_numpy(dtype=np.int64, copy=True)

    X_val = df_val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_val = df_val[cfg.target_col].to_numpy(dtype=np.int64, copy=True)

    X_test = df_test[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_test = df_test[cfg.target_col].to_numpy(dtype=np.int64, copy=True)
    
    # Impute categorical missing
    X_train = np.nan_to_num(X_train, nan=-1.0)
    X_val   = np.nan_to_num(X_val,   nan=-1.0)
    X_test  = np.nan_to_num(X_test,  nan=-1.0)

    logreg = _make_safe_logreg(max_iter=2000)
    logreg.fit(X_train, y_train)

    y_prob_val = logreg.predict_proba(X_val)
    y_prob_test = logreg.predict_proba(X_test)
    np.savez(outdir / "preds_test.npz", y_true=y_test, y_prob=y_prob_test)

    metrics = compute_baseline_metrics(y_true=y_test, y_prob=y_prob_test, model_name="LogisticRegression_saga",
                                       y_true_val=y_val, y_prob_val=y_prob_val, include_pricing=True, include_operational=True,
                                       death_flag_rate=float(getattr(cfg, "death_flag_rate", 0.01)),
                                       severe_flag_rate=float(getattr(cfg, "severe_flag_rate", 0.10)))
    
    save_json(metrics, outdir / "logistic_full_metrics.json")    
    

def run_lightgbm_baseline(cfg, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                          feature_cols: List[str], outdir: Path):
    outdir = ensure_dir(outdir / "baselines_lightgbm")

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = df_train[cfg.target_col].to_numpy(dtype=np.int64, copy=True)

    X_val = df_val[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_val = df_val[cfg.target_col].to_numpy(dtype=np.int64, copy=True)

    X_test = df_test[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_test = df_test[cfg.target_col].to_numpy(dtype=np.int64, copy=True)

    num_classes = int(np.max(y_train)) + 1
    assert num_classes == 4, f"Expected 4 classes, got {num_classes}"

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "seed": int(getattr(cfg, "seed", 42)),
        "verbosity": -1,
        }

    booster = lgb.train(params=params, train_set=train_set, valid_sets=[val_set], valid_names=["val"],
                        num_boost_round=200, callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

    y_prob_val  = np.asarray(booster.predict(X_val,  num_iteration=booster.best_iteration), dtype=float)
    y_prob_test = np.asarray(booster.predict(X_test, num_iteration=booster.best_iteration), dtype=float)

    if y_prob_test.ndim != 2 or y_prob_test.shape[1] != num_classes:
        raise ValueError(f"Unexpected proba shape: {y_prob_test.shape}")

    np.savez(outdir / "preds_test.npz", y_true=y_test, y_prob=y_prob_test)

    metrics = compute_baseline_metrics(y_true=y_test, y_prob=y_prob_test, model_name="LightGBM_default",
                                       y_true_val=y_val, y_prob_val=y_prob_val, include_pricing=True, include_operational=True,
                                       death_flag_rate=float(getattr(cfg, "death_flag_rate", 0.01)),
                                       severe_flag_rate=float(getattr(cfg, "severe_flag_rate", 0.10)))
    
    save_json(metrics, outdir / "lightgbm_metrics.json")    


# GRU baseline (sequence -> next-step per visit)
class GRUNextStepBaseline(nn.Module):
    def __init__(self, in_feats: int, hidden_dim: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size=in_feats, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B,T,H]
        logits = self.fc(out)                                       # [B,T,C]
        return logits

class NextStepDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        self.samples = []

        if "_orig_idx" not in df.columns:
            df = df.reset_index(drop=True).copy()
            df["_orig_idx"] = np.arange(len(df), dtype=np.int64)

        df_sorted = df.sort_values(["patient_idx", "time"], kind="mergesort")

        for _, g in df_sorted.groupby("patient_idx", sort=False):
            if len(g) < 2:
                continue

            X_full = g[feature_cols].to_numpy(dtype=np.float32)   # [T,F]
            y_full = g[target_col].to_numpy(dtype=np.int64)       # [T]
            rows   = g["_orig_idx"].to_numpy(dtype=np.int64)      # [T]

            X = torch.tensor(X_full[:-1], dtype=torch.float32)    # [T-1,F]
            y = torch.tensor(y_full[1:], dtype=torch.long)        # [T-1]
            orig_idx = torch.tensor(rows[1:], dtype=torch.long)   # [T-1] targets

            self.samples.append((X, y, X.size(0), orig_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def nextstep_collate_fn(batch):
    X_list, y_list, len_list, idx_list = zip(*batch)
    lengths = torch.tensor(len_list, dtype=torch.long)

    X_padded = pad_sequence(X_list, batch_first=True)  # [B,Tmax,F]
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=-100)  # [B,Tmax]
    idx_padded = pad_sequence(idx_list, batch_first=True, padding_value=-1)  # [B,Tmax]

    return X_padded, y_padded, lengths, idx_padded


def _masked_ce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, T, C = logits.shape
    return nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(B * T, C), y.reshape(B * T))

def _gru_predict_full(model: GRUNextStepBaseline, loader: DataLoader, device: torch.device, N_rows: int,
    class_prior: np.ndarray) -> np.ndarray:
    
    C = model.num_classes
    y_prob_full = np.zeros((N_rows, C), dtype=float)
    y_prob_full[:] = class_prior[None, :]

    model.eval()
    with torch.no_grad():
        for X, _, lengths, orig_idx in loader:
            X = X.to(device)
            lengths = lengths.to(device)

            logits = model(X, lengths)            # [B,T,C]
            probs = torch.softmax(logits, dim=-1) # [B,T,C]

            B = X.size(0)
            for i in range(B):
                L = int(lengths[i].item())
                if L <= 0:
                    continue
                rows = orig_idx[i, :L].cpu().numpy()
                p = probs[i, :L].cpu().numpy()
                valid = rows >= 0
                y_prob_full[rows[valid]] = p[valid]

    return y_prob_full


def run_gru_baseline(cfg, device: torch.device, df_train: pd.DataFrame, df_val: pd.DataFrame,
                     df_test: pd.DataFrame, feature_cols: List[str], outdir: Path):
    outdir = ensure_dir(outdir / "baselines_gru_nextstep")

    d_x = len(feature_cols)
    model = GRUNextStepBaseline(in_feats=d_x, hidden_dim=cfg.hidden_dim, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    df_train = df_train.reset_index(drop=True).copy()
    df_val   = df_val.reset_index(drop=True).copy()
    df_test  = df_test.reset_index(drop=True).copy()
    df_train["_orig_idx"] = np.arange(len(df_train), dtype=np.int64)
    df_val["_orig_idx"]   = np.arange(len(df_val), dtype=np.int64)
    df_test["_orig_idx"]  = np.arange(len(df_test), dtype=np.int64)
    
    for _df in (df_train, df_val, df_test):
        _df[feature_cols] = _df[feature_cols].fillna(-1)

    train_ds = NextStepDataset(df_train, feature_cols, cfg.target_col)
    val_ds   = NextStepDataset(df_val,   feature_cols, cfg.target_col)
    test_ds  = NextStepDataset(df_test,  feature_cols, cfg.target_col)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=nextstep_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=nextstep_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=nextstep_collate_fn)

    best_val = float("inf")
    best_state = None
    stale = 0
    patience = 4
    max_epochs = 15

    print("\n[GRU baseline] training...")
    for epoch in range(1, max_epochs + 1):
        # ---- Train ----
        model.train()
        tr_sum, tr_n = 0.0, 0
        for X, y, lengths, _ in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X, lengths)
            loss = _masked_ce_loss(logits, y)
            loss.backward()
            optimizer.step()
            tr_sum += float(loss.item()) * int(lengths.sum().item())
            tr_n += int(lengths.sum().item())
        tr_loss = tr_sum / max(1, tr_n)

        # ---- Val ----
        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for X, y, lengths, _ in val_loader:
                X, y, lengths = X.to(device), y.to(device), lengths.to(device)
                logits = model(X, lengths)
                loss = _masked_ce_loss(logits, y)
                va_sum += float(loss.item()) * int(lengths.sum().item())
                va_n += int(lengths.sum().item())
        va_loss = va_sum / max(1, va_n)

        print(f"[Epoch {epoch:02d}] Train={tr_loss:.4f}  Val={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print("[GRU baseline] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- class prior for rows w/out GRU preds ----
    C = model.num_classes
    class_counts = df_train[cfg.target_col].value_counts().sort_index()
    class_prior = np.zeros(C, dtype=float)
    class_prior[class_counts.index.to_numpy()] = (class_counts.to_numpy(dtype=float) / class_counts.sum())

    # ---- Full aligned VAL + TEST probs ----
    y_true_val_full = df_val[cfg.target_col].to_numpy(dtype=int)
    y_true_test_full = df_test[cfg.target_col].to_numpy(dtype=int)

    y_prob_val_full = _gru_predict_full(model, val_loader, device, N_rows=len(df_val), class_prior=class_prior)
    y_prob_test_full = _gru_predict_full(model, test_loader, device, N_rows=len(df_test), class_prior=class_prior)

    # Save TEST for overlays
    np.savez(outdir / "preds_test.npz", y_true=y_true_test_full, y_prob=y_prob_test_full)

    metrics = compute_baseline_metrics(y_true=y_true_test_full, y_prob=y_prob_test_full, model_name="GRU_nextstep_per_visit_full",
                                       y_true_val=y_true_val_full, y_prob_val=y_prob_val_full, include_pricing=True, include_operational=True,
                                       death_flag_rate=float(getattr(cfg, "death_flag_rate", 0.01)), severe_flag_rate=float(getattr(cfg, "severe_flag_rate", 0.10)))
    
    save_json(metrics, outdir / "gru_nextstep_metrics.json")