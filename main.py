# imports
import os
import time
import math
import argparse
from pathlib import Path
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score)
from sklearn.linear_model import LogisticRegression

from utils import (seed_everything, count_params, ensure_dir, brier_score, expected_calibration_error, save_json)
from model import LANTERN
from baselines import run_heuristic_baselines, run_logistic_baselines, run_gru_baseline, run_lightgbm_baseline
from plots import (plot_losses, plot_roc_pr_curves, plot_roc_pr_joint, plot_calibration_curve_overall,
                    plot_mean_sd_by_group, plot_binary_roc_pr_overlay, dataset_summary)


# Args
@dataclass
class Config:
    csv_path: str = "Final_Preprocessed_RAND_LTCI_LONG.csv"
    id_col: str = "HHIDPN"
    time_col: str = "WAVE"
    target_col: str = "Y_STATE"
    seed: int = 42

    hidden_dim: int = 128
    t2v_dim: int = 8
    attn_heads: int = 4
    dropout: float = 0.0

    lr: float = 0.003
    weight_decay: float = 1e-6
    epochs: int = 50
    patience: int = 6
    grad_clip: float = 1.0
    use_amp: bool = True

    output_dir: str = "LANTERN_outputs"
    skip_baselines: bool = False

    tune_thresholds: bool = True
    death_flag_rate: float = 0.01
    severe_flag_rate: float = 0.10

    # Ablations
    ablate_time2vec: bool = False
    ablate_attr_attention: bool = False
    
    # data split
    split_path: str = ""
    save_split_path: str = ""


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=str, default=Config.csv_path)
    ap.add_argument("--id-col", type=str, default=Config.id_col)
    ap.add_argument("--time-col", type=str, default=Config.time_col)
    ap.add_argument("--target-col", type=str, default=Config.target_col)
    ap.add_argument("--seed", type=int, default=Config.seed)

    ap.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    ap.add_argument("--t2v-dim", type=int, default=Config.t2v_dim)
    ap.add_argument("--attn-heads", type=int, default=Config.attn_heads)
    ap.add_argument("--dropout", type=float, default=Config.dropout)

    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--patience", type=int, default=Config.patience)
    ap.add_argument("--grad-clip", type=float, default=Config.grad_clip)
    ap.add_argument("--no-amp", action="store_true")

    ap.add_argument("--output-dir", type=str, default=Config.output_dir)
    ap.add_argument("--skip-baselines", action="store_true")

    ap.add_argument("--no-threshold-tuning", action="store_true")
    ap.add_argument("--death-flag-rate", type=float, default=Config.death_flag_rate)
    ap.add_argument("--severe-flag-rate", type=float, default=Config.severe_flag_rate)

    ap.add_argument("--ablate-time2vec", action="store_true")
    ap.add_argument("--ablate-attr-attention", action="store_true")
    
    ap.add_argument("--split-path", type=str, default="")
    ap.add_argument("--save-split-path", type=str, default="")

    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt-path", type=str, default="")

    args = ap.parse_args() if os.getenv("PYTHONINSPECT") is None else ap.parse_args([])

    cfg = Config(csv_path=args.csv_path, id_col=args.id_col, time_col=args.time_col, target_col=args.target_col, seed=args.seed,
                 hidden_dim=args.hidden_dim, t2v_dim=args.t2v_dim, attn_heads=args.attn_heads, dropout=args.dropout, lr=args.lr,
                 weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience, grad_clip=args.grad_clip, use_amp=not args.no_amp,
                 output_dir=args.output_dir, skip_baselines=args.skip_baselines, tune_thresholds=not args.no_threshold_tuning,
                 death_flag_rate=float(args.death_flag_rate), severe_flag_rate=float(args.severe_flag_rate), ablate_time2vec=bool(args.ablate_time2vec),
                 ablate_attr_attention=bool(args.ablate_attr_attention), split_path=str(args.split_path), save_split_path=str(args.save_split_path))
    cfg._do_train = args.train
    cfg._do_eval = args.eval
    cfg._ckpt_path = args.ckpt_path
    return cfg

# define variables
STATE_OH_PREFIX = "STATE_OH_"
STATE_COL = "STATE"
NEXT_STATE_COL = "NEXT_STATE"
AGEGROUP_COL = "AGE_GROUP"
ORIG_IDX_COL = "_orig_idx"

# Attributes
DEFAULT_ATTR_COLS = ["REGION", "MARITAL_STATUS", "RAGENDER", "RARACEM", "RAHISPAN", "RAEDUC"]

UNK_TOKEN = "__UNK__"
MISS_TOKEN = "__MISSING__"

def drop_state_oh(feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if not c.startswith(STATE_OH_PREFIX)]

def _to_str(x):
    if pd.isna(x):
        return MISS_TOKEN
    return str(x)

def save_rng_state() -> dict:
    state = {"python_random": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])

def build_attr_vocabs(df_train: pd.DataFrame, attr_cols: List[str]) -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    for col in attr_cols:
        vals = df_train[col].map(_to_str).astype(str).unique().tolist() if col in df_train.columns else []
        vals = sorted(set(vals))
        vocab = {UNK_TOKEN: 0, MISS_TOKEN: 1}
        for v in vals:
            if v in (UNK_TOKEN, MISS_TOKEN):
                continue
            vocab[v] = len(vocab)
        vocabs[col] = vocab
    return vocabs

def encode_attrs(df: pd.DataFrame, attr_cols: List[str], vocabs: Dict[str, Dict[str, int]]) -> np.ndarray:
    A = len(attr_cols)
    out = np.zeros((len(df), A), dtype=np.int64)
    for j, col in enumerate(attr_cols):
        if col not in df.columns:
            out[:, j] = 0
            continue
        vocab = vocabs[col]
        s = df[col].map(_to_str).astype(str)
        out[:, j] = s.map(lambda v: vocab.get(v, vocab[UNK_TOKEN])).values
    return out


# Data loading / split / scaling
def load_and_split_wave_batch_age_dt(cfg: Config, attr_cols: Optional[List[str]] = None, split_pids: Optional[Dict[str, np.ndarray]] = None):
    df = pd.read_csv(cfg.csv_path)
    df = df.dropna(subset=[cfg.target_col]).reset_index(drop=True)
    df[cfg.target_col] = df[cfg.target_col].astype(int)

    df["patient_idx"] = df[cfg.id_col].astype("category").cat.codes
    num_patients = int(df["patient_idx"].nunique())

    if cfg.time_col not in df.columns:
        raise ValueError(f"Expected '{cfg.time_col}' column for batching/ordering.")
    df["time"] = df[cfg.time_col].astype(float)
    print("[Info] Using Wave for batching/ordering (df['time']=Wave).")

    if "AGE" in df.columns and df["AGE"].notna().any():
        df["age_time"] = df["AGE"].astype(float)
        print("[Info] Using Age for irregular dt (df['age_time']=AGE).")
    else:
        raise ValueError("Age column not found (or all missing). Needed to model irregular dt using Age.")

    if attr_cols is None:
        attr_cols = [c for c in DEFAULT_ATTR_COLS if c in df.columns]

    exclude_cols = {cfg.id_col, "patient_idx", cfg.time_col, "time", "age_time", cfg.target_col, STATE_COL, NEXT_STATE_COL,
                    AGEGROUP_COL, "DEAD", "STATE_ID", "RADAGE_Y", "Y_DEATH", "Y_STATE3", "WAVES_SO_FAR", "DAGE", "AGE_C", "AGE2"}
    for a in attr_cols:
        exclude_cols.add(a)
        exclude_cols.add(f"{a}_MISSING")

    feature_cols_all = [c for c in df.columns if c not in exclude_cols]
    feature_cols = drop_state_oh(feature_cols_all)
    df[feature_cols] = df[feature_cols].fillna(0)

    # patient-level split
    if split_pids is None:
        rng = np.random.RandomState(cfg.seed)
        pids = rng.permutation(df["patient_idx"].unique())
        cut1, cut2 = int(0.7 * len(pids)), int(0.85 * len(pids))
        train_p, val_p, test_p = np.split(pids, [cut1, cut2])
        split_pids = {"train": train_p, "val": val_p, "test": test_p}
    else:
        train_p = split_pids["train"]
        val_p   = split_pids["val"]
        test_p  = split_pids["test"]

    df_train = df[df["patient_idx"].isin(train_p)].copy().reset_index(drop=True)
    df_val   = df[df["patient_idx"].isin(val_p)].copy().reset_index(drop=True)
    df_test  = df[df["patient_idx"].isin(test_p)].copy().reset_index(drop=True)

    # stable per-split row id for alignment
    df_train[ORIG_IDX_COL] = np.arange(len(df_train), dtype=np.int64)
    df_val[ORIG_IDX_COL]   = np.arange(len(df_val), dtype=np.int64)
    df_test[ORIG_IDX_COL]  = np.arange(len(df_test), dtype=np.int64)

    # build attr vocabs on TRAIN only
    vocabs = build_attr_vocabs(df_train, attr_cols)
    df_train["_attr_ids"] = list(encode_attrs(df_train, attr_cols, vocabs))
    df_val["_attr_ids"]   = list(encode_attrs(df_val,   attr_cols, vocabs))
    df_test["_attr_ids"]  = list(encode_attrs(df_test,  attr_cols, vocabs))

    # scale numeric features on TRAIN only
    if len(feature_cols) > 0:
        scaler = StandardScaler().fit(df_train[feature_cols])
        for split in (df_train, df_val, df_test):
            split[feature_cols] = scaler.transform(split[feature_cols])
    else:
        scaler = None

    return df_train, df_val, df_test, feature_cols, attr_cols, vocabs, num_patients, scaler, split_pids


def build_event_batches(df_split: pd.DataFrame, feature_cols: List[str], target_col: str):
    if ORIG_IDX_COL not in df_split.columns:
        raise ValueError(f"Missing {ORIG_IDX_COL} in df_split.")

    df_sorted = df_split.sort_values(["time", "patient_idx"], kind="mergesort").reset_index(drop=True)
    batches = []
    for wave, g in df_sorted.groupby("time", sort=False):
        idxs = g["patient_idx"].to_numpy(np.int64)
        age  = g["age_time"].to_numpy(np.float32)

        X    = g[feature_cols].to_numpy(np.float32) if feature_cols else np.zeros((len(g), 0), np.float32)
        y    = g[target_col].to_numpy(np.int64)
        attr = np.vstack(g["_attr_ids"].values).astype(np.int64)
        orig = g[ORIG_IDX_COL].to_numpy(np.int64)

        batches.append((
            torch.tensor(np.full(len(g), float(wave), dtype=np.float32)),
            torch.tensor(age,  dtype=torch.float32),
            torch.tensor(idxs, dtype=torch.long),
            torch.tensor(X,    dtype=torch.float32),
            torch.tensor(attr, dtype=torch.long),
            torch.tensor(y,    dtype=torch.long),
            torch.tensor(orig, dtype=torch.long)))
    return batches


# Risk metric helpers
def _binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

def endpoint_metrics_from_probs(y_true, prob4):
    y_true = np.asarray(y_true, dtype=int)
    prob4 = np.asarray(prob4, dtype=float)
    yD = (y_true == 3).astype(int)
    yS = (y_true == 2).astype(int)
    return {
        "death_auc": _safe_auc(yD, prob4[:, 3]),
        "death_pr":  _safe_ap(yD, prob4[:, 3]),
        "severe_auc": _safe_auc(yS, prob4[:, 2]),
        "severe_pr":  _safe_ap(yS, prob4[:, 2]),
    }

def paired_bootstrap_by_patient(patient_ids, y_true, probs_A, probs_B, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true)
    probs_A = np.asarray(probs_A)
    probs_B = np.asarray(probs_B)

    unique_p = np.unique(patient_ids)
    rows_by_pid = {pid: np.where(patient_ids == pid)[0] for pid in unique_p}

    deltas = {k: [] for k in ["death_auc","death_pr","severe_auc","severe_pr"]}

    for _ in range(n_boot):
        samp = rng.choice(unique_p, size=len(unique_p), replace=True)
        idx = np.concatenate([rows_by_pid[pid] for pid in samp], axis=0)

        mA = endpoint_metrics_from_probs(y_true[idx], probs_A[idx])
        mB = endpoint_metrics_from_probs(y_true[idx], probs_B[idx])

        for k in deltas:
            if np.isfinite(mA[k]) and np.isfinite(mB[k]):
                deltas[k].append(mA[k] - mB[k])

    out = {}
    for k, vals in deltas.items():
        vals = np.asarray(vals, dtype=float)
        out[k] = {
            "mean_delta": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
    return out

def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y_true)
    return {"calibration_intercept": float(lr.intercept_[0]), "calibration_slope": float(lr.coef_[0][0])}


# flagrate-based thresholding
def pick_threshold_by_flag_rate(y_prob: np.ndarray, flag_rate: float) -> Dict[str, float]:
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

def threshold_operating_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
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


# Risk stratification helpers
def _risk_band_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    N = len(y_true)
    if N == 0:
        return {"n_bins": n_bins, "overall_event_rate": float("nan"), "bins": []}
    overall_rate = float(y_true.mean())
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    bins = []
    events_total = int(y_true.sum())
    cum_events = 0
    edges = np.linspace(0, N, n_bins + 1).round().astype(int)
    edges[-1] = N
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
            "bin": int(b + 1),
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
    return {"n_bins": int(n_bins), "overall_event_rate": overall_rate, "total_events": int(events_total), "bins": bins}

def _topk_capture(y_true: np.ndarray, y_prob: np.ndarray, top_fracs=(0.01, 0.02, 0.05, 0.10)) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    N = len(y_true)
    events_total = int(y_true.sum())
    if N == 0 or events_total == 0:
        return {f"top_{int(f*100)}pct_capture": float("nan") for f in top_fracs}
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    out = {}
    for f in top_fracs:
        k = max(1, int(round(f * N)))
        out[f"top_{int(f*100)}pct_capture"] = float(y_sorted[:k].sum() / events_total)
        out[f"top_{int(f*100)}pct_flag_rate"] = float(k / N)
    return out

def _lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10) -> float:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
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

def risk_pricing_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    return {
        "risk_bands": _risk_band_table(y_true, y_prob, n_bins=n_bins),
        "topk_capture": _topk_capture(y_true, y_prob),
        "lift_at_10pct": _lift_at_k(y_true, y_prob, top_frac=0.10),
        "lift_at_5pct": _lift_at_k(y_true, y_prob, top_frac=0.05),
    }

# Training helpers
def make_memories(num_patients: int, d_h: int, d_x: int, device: torch.device):
    mem = torch.zeros(num_patients, d_h, device=device)
    last_age = torch.zeros(num_patients, device=device)
    last_feat = torch.zeros(num_patients, d_x, device=device)
    visit_count = torch.zeros(num_patients, dtype=torch.long, device=device)
    return mem, last_age, last_feat, visit_count


def _build_inputs_with_or_without_irregularity(cfg: Config, age: torch.Tensor, prev_age: torch.Tensor,
                                               is_first: torch.Tensor, visit_count: torch.Tensor):
    B = age.size(0)

    if cfg.ablate_time2vec:
        dt_raw = torch.zeros_like(age)
        dt = dt_raw.unsqueeze(1)
        log_dt = torch.zeros((B, 1), device=age.device)
        is_first_f = torch.zeros((B, 1), device=age.device)
        vc_f = torch.zeros((B, 1), device=age.device)
        return dt, log_dt, is_first_f, vc_f

    dt_raw = (age - prev_age).clamp(min=0.0)
    dt_raw[is_first] = 0.0
    dt = dt_raw.unsqueeze(1)
    log_dt = torch.log1p(dt_raw).unsqueeze(1)
    is_first_f = is_first.float().unsqueeze(1)
    vc_f = visit_count.float().unsqueeze(1)
    return dt, log_dt, is_first_f, vc_f


def run_epoch(cfg: Config, model: LANTERN, batches, mem: torch.Tensor, last_age: torch.Tensor, last_feat: torch.Tensor, visit_count: torch.Tensor,
              bce: nn.Module, ce_alive: nn.Module, device: torch.device, train: bool, optimizer: Optional[optim.Optimizer] = None,
              scaler: Optional[torch.amp.GradScaler] = None, grad_clip: float = 1.0):
    running = 0.0
    model.train() if train else model.eval()

    for _, age_cpu, idxs_cpu, X_num_cpu, attr_cpu, y_true_cpu, _ in batches:
        idxs = idxs_cpu.to(device)
        age  = age_cpu.to(device)
        X_num = X_num_cpu.to(device)
        attr_ids = attr_cpu.to(device)
        y_true = y_true_cpu.to(device)

        m_prev = mem[idxs]
        prev_age = last_age[idxs]
        is_first = (visit_count[idxs] == 0)

        dt, log_dt, is_first_f, vc_f = _build_inputs_with_or_without_irregularity(
            cfg=cfg,
            age=age,
            prev_age=prev_age,
            is_first=is_first,
            visit_count=visit_count[idxs],
        )

        X_num2 = torch.cat([X_num, log_dt, is_first_f, vc_f], dim=1)

        use_amp = (scaler is not None) and (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            m_new, probs4, z_death, logits_alive = model.forward_one_batch(m_prev, X_num2, dt, attr_ids)

            y_death = (y_true == 3).float()
            alive_mask = (y_true != 3)

            loss_death = bce(z_death, y_death)
            if alive_mask.any():
                y_alive = y_true[alive_mask]
                loss_alive = ce_alive(logits_alive[alive_mask], y_alive)
            else:
                loss_alive = 0.0 * loss_death

            loss = loss_death + loss_alive

        if train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        mem[idxs] = m_new.detach()
        last_age[idxs] = age
        last_feat[idxs] = X_num2.detach()
        visit_count[idxs] += 1

        running += float(loss.item())

    return running / max(1, len(batches))


def build_model(cfg, d_x, device, attr_cols, vocabs):
    return LANTERN(in_feats=d_x, hidden_dim=cfg.hidden_dim, attr_cols=attr_cols, attr_vocabs=vocabs, t2v_dim=cfg.t2v_dim,
                   attn_heads=cfg.attn_heads, dropout=cfg.dropout, ablate_time2vec=cfg.ablate_time2vec,
                   ablate_attr_attention=cfg.ablate_attr_attention).to(device)

def train_only(cfg, device, df_train, df_val, feature_cols, attr_cols, vocabs, num_patients, outdir: Path):
    train_batches = build_event_batches(df_train, feature_cols, cfg.target_col)
    val_batches   = build_event_batches(df_val,   feature_cols, cfg.target_col)

    d_x = len(feature_cols) + 3
    model = build_model(cfg, d_x, device, attr_cols, vocabs)
    print(f"[Model] params={count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    bce = nn.BCEWithLogitsLoss()
    ce_alive = nn.CrossEntropyLoss()

    use_amp = cfg.use_amp and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val = float("inf")
    best_path = outdir / "best_model.pt"
    last_path = outdir / "last_model.pt"
    stale = 0

    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        mem_tr, last_age_tr, feat_tr, vc_tr = make_memories(num_patients, cfg.hidden_dim, d_x, device)
        mem_va, last_age_va, feat_va, vc_va = make_memories(num_patients, cfg.hidden_dim, d_x, device)

        t0 = time.time()
        tr_loss = run_epoch(
            cfg, model, train_batches, mem_tr, last_age_tr, feat_tr, vc_tr,
            bce=bce, ce_alive=ce_alive, device=device, train=True,
            optimizer=optimizer, scaler=amp_scaler, grad_clip=cfg.grad_clip
        )
        va_loss = run_epoch(
            cfg, model, val_batches, mem_va, last_age_va, feat_va, vc_va,
            bce=bce, ce_alive=ce_alive, device=device, train=False,
            optimizer=None, scaler=None, grad_clip=cfg.grad_clip
        )

        scheduler.step(va_loss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        torch.save({"model_state": model.state_dict(), "feature_cols": feature_cols, "attr_cols": attr_cols,
                    "attr_vocabs": vocabs, "config": asdict(cfg)}, last_path)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}  Train {tr_loss:.4f}  Val {va_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.1f}s)")

        if math.isfinite(va_loss) and va_loss < best_val:
            best_val = va_loss
            stale = 0
            torch.save({"model_state": model.state_dict(), "feature_cols": feature_cols, "attr_cols": attr_cols,
                        "attr_vocabs": vocabs, "config": asdict(cfg)}, best_path)
        else:
            stale += 1
            if stale >= cfg.patience:
                print("Early stopping.")
                break

    plot_losses(train_losses, val_losses, outdir)

    return best_path if best_path.exists() else last_path


# Evaluation helpers
def _run_inference_probs(cfg: Config, model: LANTERN, device: torch.device, df_split: pd.DataFrame, feature_cols: List[str],
                         attr_cols: List[str], target_col: str, num_patients: int, hidden_dim: int) -> Dict[str, np.ndarray]:
    d_x = len(feature_cols) + 3
    batches = build_event_batches(df_split, feature_cols, target_col)
    mem, last_age, last_feat, vc = make_memories(num_patients, hidden_dim, d_x, device)

    probs_chunks, preds_chunks, orig_chunks = [], [], []
    A = len(attr_cols)
    attn_chunks = []

    model.eval()
    with torch.no_grad():
        for _, age_cpu, idxs_cpu, X_num_cpu, attr_cpu, y_true_cpu, orig_cpu in batches:
            idxs     = idxs_cpu.to(device)
            age      = age_cpu.to(device)
            X_num    = X_num_cpu.to(device)
            attr_ids = attr_cpu.to(device)

            m_prev = mem[idxs]
            prev_age = last_age[idxs]
            is_first = (vc[idxs] == 0)

            dt, log_dt, is_first_f, vc_f = _build_inputs_with_or_without_irregularity(cfg=cfg, age=age, prev_age=prev_age, is_first=is_first, visit_count=vc[idxs])
            X_num2 = torch.cat([X_num, log_dt, is_first_f, vc_f], dim=1)

            m_new, probs4, _, _ = model.forward_one_batch(m_prev, X_num2, dt, attr_ids)

            attn = model.last_attn_attr
            if attn is None:
                attn = torch.zeros((probs4.size(0), A), device=probs4.device)
            attn_chunks.append(attn.cpu().numpy())

            mem[idxs] = m_new.detach()
            last_age[idxs] = age
            last_feat[idxs] = X_num2.detach()
            vc[idxs] += 1

            preds = probs4.argmax(dim=1)

            probs_chunks.append(probs4.cpu().numpy())
            preds_chunks.append(preds.cpu().numpy())
            orig_chunks.append(orig_cpu.cpu().numpy())

    N = len(df_split)
    probs_full = np.zeros((N, 4), dtype=float)
    preds_full = np.zeros((N,), dtype=int)
    attn_full = np.zeros((N, A), dtype=float)

    if probs_chunks:
        probs_cat = np.concatenate(probs_chunks, axis=0)
        preds_cat = np.concatenate(preds_chunks, axis=0)
        orig_cat  = np.concatenate(orig_chunks, axis=0)

        probs_full[orig_cat] = probs_cat
        preds_full[orig_cat] = preds_cat

        attn_cat = np.concatenate(attn_chunks, axis=0) if len(attn_chunks) else None
        if attn_cat is not None:
            attn_full[orig_cat] = attn_cat

    truths_full = df_split[target_col].to_numpy(dtype=int, copy=True)

    return {"probs": probs_full, "preds": preds_full, "truths": truths_full, "attn_attr": attn_full}


def eval_only(cfg, device, df_test, feature_cols, attr_cols, vocabs, num_patients, outdir, ckpt_path: Path):
    d_x = len(feature_cols) + 3
    model = build_model(cfg, d_x, device, attr_cols, vocabs)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_test = _run_inference_probs(cfg=cfg, model=model, device=device, df_split=df_test, feature_cols=feature_cols,
                                    attr_cols=attr_cols, target_col=cfg.target_col, num_patients=num_patients, hidden_dim=cfg.hidden_dim)

    probs = out_test["probs"]
    preds = out_test["preds"]
    truths = out_test["truths"]
    attn_attr = out_test.get("attn_attr", None)

    # Multiclass metrics
    if len(preds):
        brier_mc = brier_score(truths, probs)
        ece_mc = expected_calibration_error(truths, probs, n_bins=10)
    else:
        brier_mc = ece_mc = float("nan")

    print(f"TEST  Brier (MC):       {brier_mc:.4f}")
    print(f"TEST  ECE (MC):         {ece_mc:.4f}")

    # Endpoints
    pricing_death = {}
    pricing_severe = {}
    death_metrics = {}
    severe_metrics = {}
    operational = {}

    if probs.shape[0] > 0:
        yD = (truths == 3).astype(int)
        yS = (truths == 2).astype(int)
        pD = probs[:, 3]
        pS = probs[:, 2]

        death_metrics = {
            "prevalence": float(yD.mean()),
            "AUROC": _safe_auc(yD, pD),
            "PRAUC": _safe_ap(yD, pD),
            "Brier": _binary_brier(yD, pD),
            "ECE": float(expected_calibration_error(yD, np.vstack([1 - pD, pD]).T, n_bins=10)),
            **calibration_slope_intercept(yD, pD),
        }
        severe_metrics = {
            "prevalence": float(yS.mean()),
            "AUROC": _safe_auc(yS, pS),
            "PRAUC": _safe_ap(yS, pS),
            "Brier": _binary_brier(yS, pS),
            "ECE": float(expected_calibration_error(yS, np.vstack([1 - pS, pS]).T, n_bins=10)),
            **calibration_slope_intercept(yS, pS),
        }

        pricing_death = risk_pricing_summary(yD, pD, n_bins=10)
        pricing_severe = risk_pricing_summary(yS, pS, n_bins=10)

        print("\n[Risk endpoints]")
        print(f"Death  prev={death_metrics['prevalence']:.4f}  AUROC={death_metrics['AUROC']:.4f}  PRAUC={death_metrics['PRAUC']:.4f}  "
              f"Brier={death_metrics['Brier']:.4f}  ECE={death_metrics['ECE']:.6f}")
        print(f"Severe prev={severe_metrics['prevalence']:.4f}  AUROC={severe_metrics['AUROC']:.4f}  PRAUC={severe_metrics['PRAUC']:.4f}  "
              f"Brier={severe_metrics['Brier']:.4f}  ECE={severe_metrics['ECE']:.6f}")

    save_json(
        {
            "Multiclass": {"Brier": float(brier_mc), "ECE": float(ece_mc)},
            "RiskEndpoints": {"Death": death_metrics, "Severe": severe_metrics},
            "Operational": operational,
            "PricingStratification": {"Death": pricing_death, "Severe": pricing_severe},
        },
        Path(outdir) / "test_metrics.json"
    )

    lantern_dir = ensure_dir(Path(outdir) / "lantern_preds")
    np.savez(lantern_dir / "preds_test.npz", y_true=truths, y_prob=probs)
    
    # Plot folders
    main_dir = ensure_dir(outdir / "main_plots")
    supp_dir = ensure_dir(outdir / "supp_plots")

    class_names = ["Healthy", "Mild", "Severe", "Death"]

    if probs.shape[0] == 0:
        pd.DataFrame().to_csv(outdir / "predictions_test_annotated.csv", index=False)
        return

    # risk stratification plots (main)
    save_json(pricing_severe["risk_bands"], Path(outdir) / "riskbands_severe.json")
    save_json(pricing_death["risk_bands"],  Path(outdir) / "riskbands_death.json")

    # PR/ROC joint for Severe & Death (main)
    plot_roc_pr_joint(truths=truths, probs=probs, outdir=main_dir, class_indices=[2, 3], class_names=class_names, filename_prefix="main_severe_death")

    # Supplementary: all classes
    plot_roc_pr_curves( truths, probs, outdir=supp_dir, class_names=class_names, class_indices=[0, 1, 2, 3], prefix="supp_")
    plot_roc_pr_joint(truths=truths, probs=probs, outdir=supp_dir, class_indices=[0, 1, 2, 3], class_names=class_names, filename_prefix="supp_all_states")

    # Calibration curves (main endpoints)
    y_true_D = (truths == 3).astype(int)
    p_D = probs[:, 3]
    plot_calibration_curve_overall(y_true_binary=y_true_D, y_prob=p_D, outpath=main_dir / "calibration_overall_death.png",
                                   n_bins=10, title="Calibration curve – P(Death)")

    y_true_S = (truths == 2).astype(int)
    p_S = probs[:, 2]
    plot_calibration_curve_overall(y_true_binary=y_true_S, y_prob=p_S, outpath=main_dir / "calibration_overall_severe.png",
                                   n_bins=10, title="Calibration curve – P(Severe disability)")

    # Calibration all (supp)
    class_tags = {0: "H", 1: "M", 2: "S", 3: "D"}
    for cls_idx in range(4):
        y_true_cls = (truths == cls_idx).astype(int)
        p_cls = probs[:, cls_idx]
        plot_calibration_curve_overall(y_true_binary=y_true_cls, y_prob=p_cls, outpath=supp_dir / f"calibration_overall_{class_tags[cls_idx]}.png", n_bins=10)

    # Build annotated predictions directly in df_test row order
    pred_df_annot = df_test[[ORIG_IDX_COL, "patient_idx", "time"]].copy()
    pred_df_annot["y_true"] = truths
    pred_df_annot["y_pred"] = preds
    pred_df_annot["p_H"] = probs[:, 0]
    pred_df_annot["p_M"] = probs[:, 1]
    pred_df_annot["p_S"] = probs[:, 2]
    pred_df_annot["p_D"] = probs[:, 3]

    for col in [STATE_COL, NEXT_STATE_COL, AGEGROUP_COL]:
        if col in df_test.columns:
            pred_df_annot[col] = df_test[col].values
        else:
            print(f"[INFO] Column '{col}' not found in df_test; skipping annotation.")
    
    if attn_attr is not None and attn_attr.shape[0] == len(pred_df_annot):
        for j, col in enumerate(attr_cols):
            pred_df_annot[f"attn_{col}"] = attn_attr[:, j]

    pred_df_annot = pred_df_annot.sort_values(["time", "patient_idx"], kind="mergesort")
    pred_df_annot.to_csv(outdir / "predictions_test_annotated.csv", index=False)

    # flags for group plots
    pred_df_annot["is_H"] = (pred_df_annot["y_true"] == 0).astype(int)
    pred_df_annot["is_M"] = (pred_df_annot["y_true"] == 1).astype(int)
    pred_df_annot["is_S"] = (pred_df_annot["y_true"] == 2).astype(int)
    pred_df_annot["is_D"] = (pred_df_annot["y_true"] == 3).astype(int)

    if AGEGROUP_COL in pred_df_annot.columns:
        plot_mean_sd_by_group(pred_df_annot, group_col=AGEGROUP_COL, outpath=main_dir / "mean_sd_by_agegroup_death.png",
                              true_col="is_D", pred_col="p_D", title_prefix="P(Death) (Observed vs Predicted)", endpoint_class=3)
        plot_mean_sd_by_group(pred_df_annot, group_col=AGEGROUP_COL, outpath=main_dir / "mean_sd_by_agegroup_severe.png",
                              true_col="is_S", pred_col="p_S", title_prefix="P(Severe disability) (Observed vs Predicted)", endpoint_class=2)

    # Overlay plots: Model vs baselines
    overlay_dir = ensure_dir(outdir / "overlays")
    base_dir = outdir / "baselines"

    baseline_paths = {
        "Logistic":  base_dir / "baselines_logistic" / "preds_test.npz",
        "GRU":       base_dir / "baselines_gru_nextstep" / "preds_test.npz",
        "Heuristic": base_dir / "baselines_heuristics" / "preds_test.npz",
        "LightGBM": base_dir / "baselines_lightgbm" / "preds_test.npz",
    }

    baseline_probs = {}
    for name, pth in baseline_paths.items():
        if pth.exists():
            d = np.load(pth)
            baseline_probs[name] = d["y_prob"]
        else:
            print(f"[overlay] Missing baseline file: {pth}")

    baseline_probs["LANTERN"] = probs

    N = len(truths)
    for name in list(baseline_probs.keys()):
        if baseline_probs[name].shape[0] != N:
            print(f"[overlay] Dropping {name}: length {baseline_probs[name].shape[0]} != {N}")
            baseline_probs.pop(name)

    # Death endpoint
    y_death = (truths == 3).astype(int)
    death_model_probs = {name: arr[:, 3] for name, arr in baseline_probs.items()}
    plot_binary_roc_pr_overlay(y_death, death_model_probs, overlay_dir, prefix="death")

    # Severe endpoint
    y_severe = (truths == 2).astype(int)
    severe_model_probs = {name: arr[:, 2] for name, arr in baseline_probs.items()}
    plot_binary_roc_pr_overlay(y_severe, severe_model_probs, overlay_dir, prefix="severe")
    
    # patient ids aligned with truths/probs
    patient_ids = df_test["patient_idx"].to_numpy()
    
    def run_bootstrap_vs(baseline_name: str, outstem: str):
        if baseline_name not in baseline_probs:
            print(f"[Bootstrap] {baseline_name} missing; skipping.")
            return None
    
        deltas = paired_bootstrap_by_patient(patient_ids=patient_ids, y_true=truths, probs_A=baseline_probs["LANTERN"],
                                             probs_B=baseline_probs[baseline_name], n_boot=1000, seed=cfg.seed)
    
        out_json = Path(outdir) / f"ci_deltas_vs_{outstem}.json"
        save_json(deltas, out_json)
    
        print(f"[Bootstrap vs {baseline_name}] 95% CI deltas:")
        for k, v in deltas.items():
            print(f"  {k}: {v['mean_delta']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    
        return deltas
    
    # Logistic
    run_bootstrap_vs("Logistic", "logistic")
    
    # LightGBM
    run_bootstrap_vs("LightGBM", "lightgbm")

def save_split_npz(path: str, split_pids: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=split_pids["train"].astype(np.int64), val=split_pids["val"].astype(np.int64), test=split_pids["test"].astype(np.int64))

def load_split_npz(path: str) -> dict:
    z = np.load(str(path), allow_pickle=False)
    return {"train": z["train"].astype(np.int64), "val":   z["val"].astype(np.int64), "test":  z["test"].astype(np.int64)}

# Main
def main():
    cfg = parse_args()

    outdir = ensure_dir(cfg.output_dir)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(outdir) / run_tag / "wave_batch_age_dt_hybrid")

    seed_everything(cfg.seed, deterministic=True, single_thread=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Ablations] ablate_time2vec={cfg.ablate_time2vec}  ablate_attr_attention={cfg.ablate_attr_attention}")

    # load if split data exist, otherwise create and save
    split_pids_in = None
    if cfg.split_path:
        split_pids_in = load_split_npz(cfg.split_path)
        print(f"[Split] Loaded patient split from: {cfg.split_path}")
    
    df_train, df_val, df_test, fcols, attr_cols, vocabs, num_patients, scaler, split_pids = \
        load_and_split_wave_batch_age_dt(cfg, split_pids=split_pids_in)
    
    if cfg.save_split_path:
        save_split_npz(cfg.save_split_path, split_pids)
        print(f"[Split] Saved patient split to: {cfg.save_split_path}")
    
    fcols_baseline = list(fcols) + list(attr_cols)

    save_json({"run_tag": "wave_batch_age_dt_hybrid", "n_train_rows": int(len(df_train)), "n_val_rows": int(len(df_val)), "n_test_rows": int(len(df_test)),
               "n_patients_total": int(num_patients), "feature_cols_numeric": fcols, "attr_cols": attr_cols, "attr_vocab_sizes": {k: len(v) for k, v in vocabs.items()},
               "split_sizes": {k: int(len(v)) for k, v in split_pids.items()}, "config": asdict(cfg)}, run_dir / "meta.json")

    dataset_summary(df_train, cfg.target_col, run_dir / "data_summary_train")
    dataset_summary(df_val,   cfg.target_col, run_dir / "data_summary_val")
    dataset_summary(df_test,  cfg.target_col, run_dir / "data_summary_test")

    if not cfg.skip_baselines:
        rng_state_before_baselines = save_rng_state()
        base_dir = ensure_dir(run_dir / "baselines")
        run_heuristic_baselines(cfg, df_train, df_val, df_test, base_dir)
        run_logistic_baselines(cfg, df_train, df_val, df_test, fcols_baseline, base_dir)
        run_gru_baseline(cfg, device, df_train, df_val, df_test, fcols_baseline, base_dir)
        run_lightgbm_baseline(cfg, df_train, df_val, df_test, fcols_baseline, base_dir)
        restore_rng_state(rng_state_before_baselines)

    best_path = run_dir / "best_model.pt"
    if cfg._do_train or (not cfg._do_train and not cfg._do_eval):
        best_path = train_only(cfg, device, df_train, df_val, fcols, attr_cols, vocabs, num_patients, run_dir)

    if cfg._do_eval:
        ckpt_path = Path(cfg._ckpt_path) if cfg._ckpt_path else best_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        eval_only(cfg, device, df_test, fcols, attr_cols, vocabs, num_patients, run_dir, ckpt_path)

    print(f"\nArtifacts saved to: {run_dir.resolve()}")

if __name__ == "__main__":
    main()