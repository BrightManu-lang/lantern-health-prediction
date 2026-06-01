from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (auc, average_precision_score, roc_curve, roc_auc_score, precision_recall_curve)
from utils import save_fig

def _save_current_fig(path: Path, dpi: int = 600):
    """Match utils_new.save_fig signature (fig, path)."""
    fig = plt.gcf()
    save_fig(fig, path, dpi=dpi)

CLASS_COLORS = {
    0: "#4C72B0",  # Healthy
    1: "#55A868",  # Mild
    2: "#C44E52",  # Severe
    3: "#8172B3",  # Death
}

# Overlay styling (Model vs baselines)
MODEL_STYLES = {
    "LANTERN":   {"color": "#dd1c77", "linestyle": "-", "linewidth": 1.9, "zorder": 4},  # magenta
    "Logistic":  {"color": "#e6550d", "linestyle": "-", "linewidth": 1.9, "zorder": 3},  # orange
    "LightGBM":  {"color": "#2ca25f", "linestyle": "-", "linewidth": 1.9, "zorder": 3},  # midgreen
    "GRU":       {"color": "#f1c40f", "linestyle": "-", "linewidth": 1.9, "zorder": 3},  # gold
    "Heuristic": {"color": "#00cfe3", "linestyle": "-", "linewidth": 1.9, "zorder": 2},  # cyan
    "Chance":    {"color": "#cfcfcf", "linestyle": "-", "linewidth": 1.2, "zorder": 1},  # light gray
}

# legend order
OVERLAY_LEGEND_ORDER = ["LANTERN", "Logistic", "LightGBM", "GRU", "Heuristic", "Chance"]

def _style_for(name: str) -> dict:
    return MODEL_STYLES[name]

def _apply_ordered_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return

    def base_name(lab: str) -> str:
        return lab.split(" (", 1)[0].strip()

    ordered_idx = []
    used = set()

    for key in OVERLAY_LEGEND_ORDER:
        for i, lab in enumerate(labels):
            if i in used:
                continue
            if base_name(lab) == key:
                ordered_idx.append(i)
                used.add(i)
                break

    # Append anything unexpected
    for i in range(len(labels)):
        if i not in used:
            ordered_idx.append(i)

    ax.legend([handles[i] for i in ordered_idx], [labels[i] for i in ordered_idx], frameon=False)

AGEGROUP_COL = "AGE_GROUP"


# Plotting functions
def plot_losses(train_losses, val_losses, outdir: Path):
    outdir = Path(outdir)
    fig = plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LANTERN Loss Curves")
    plt.legend()
    save_fig(fig, outdir / "loss_curves.png", dpi=200)
    plt.close(fig)


def _ordered_groups(values):
    # keep AgeGroup in natural order
    order = ["<60", "60–69", "70–79", "80+"]
    uniq = [g for g in pd.Series(values).dropna().unique()]
    if all(g in order for g in uniq):
        return [g for g in order if g in uniq]
    return sorted(uniq, key=lambda x: str(x))


def plot_mean_sd_by_group(df: pd.DataFrame, group_col: str, outpath: Path, true_col: str = "y_true",
                          pred_col: str = "y_pred", title_prefix: str = "Mean ± SD",
                          endpoint_class: int = 2):   # 2 = Severe, 3 = Death

    use = df.dropna(subset=[group_col, true_col, pred_col]).copy()
    if use.empty:
        return

    groups = _ordered_groups(use[group_col])
    stats = use.groupby(group_col).agg(true_mean=(true_col, "mean"), true_sd=(true_col, "std"),
            pred_mean=(pred_col, "mean"), pred_sd=(pred_col, "std")).loc[groups]

    x = np.arange(len(groups))
    plt.figure()

    base_color = CLASS_COLORS.get(endpoint_class, "#333333")

    plt.errorbar(x, stats["true_mean"], yerr=stats["true_sd"], marker="o", linestyle="-",
        linewidth=1.2, color=base_color, ecolor=base_color, elinewidth=1.0, alpha=0.9,
        label="Observed")
    
    plt.errorbar(x, stats["pred_mean"], yerr=stats["pred_sd"], marker="s", linestyle="--",
        linewidth=1.8, color=base_color, ecolor=base_color, elinewidth=1.0, alpha=0.7,
        label="Predicted")

    plt.xticks(x, groups, rotation=0)
    plt.ylim(bottom=0)
    plt.ylabel("Probability")
    plt.xlabel("Age Group")
    plt.title(f"{title_prefix} by {group_col}")
    plt.legend()
    _save_current_fig(Path(outpath))

def plot_roc_pr_curves(truths: np.ndarray, probs: np.ndarray, outdir: Path, class_names=None,
    class_indices=None, prefix: str = ""):
    
    outdir = Path(outdir)
    num_classes = probs.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    if class_indices is None:
        class_indices = list(range(num_classes))

    for c in class_indices:
        y_true_c = (truths == c).astype(int)
        y_score_c = probs[:, c]

        # If class never appears, skip
        if y_true_c.sum() == 0:
            continue

        color = CLASS_COLORS.get(c, None)

        # ROC
        fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
        auc = roc_auc_score(y_true_c, y_score_c)

        plt.figure()
        plt.plot(fpr, tpr, color=color, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC – {class_names[c]}")
        plt.legend()
        _save_current_fig(outdir / f"{prefix}roc_class_{c}.png")

        # PR
        prec, rec, _ = precision_recall_curve(y_true_c, y_score_c)
        plt.figure()
        plt.plot(rec, prec, color=color)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall – {class_names[c]}")
        _save_current_fig(outdir / f"{prefix}pr_class_{c}.png")


def plot_roc_pr_joint(truths: np.ndarray, probs: np.ndarray, outdir: Path, class_indices: List[int], class_names: List[str],
    filename_prefix: str = "joint"):
    outdir = Path(outdir)

    # ROC joint
    plt.figure()
    any_plotted = False
    for c in class_indices:
        y_true_c = (truths == c).astype(int)
        y_score_c = probs[:, c]
        if y_true_c.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
        auc = roc_auc_score(y_true_c, y_score_c)
        color = CLASS_COLORS.get(c, None)

        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})", color=color)
        any_plotted = True

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (one-vs-rest)")
    plt.legend()
    if any_plotted:
        _save_current_fig(outdir / f"{filename_prefix}_roc_joint.png")
    else:
        plt.close()

    # PR joint
    plt.figure()
    any_plotted = False
    for c in class_indices:
        y_true_c = (truths == c).astype(int)
        y_score_c = probs[:, c]
        if y_true_c.sum() == 0:
            continue

        prec, rec, _ = precision_recall_curve(y_true_c, y_score_c)
        color = CLASS_COLORS.get(c, None)
        plt.plot(rec, prec, label=f"{class_names[c]}", color=color)
        any_plotted = True

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves (one-vs-rest)")
    plt.legend()
    if any_plotted:
        _save_current_fig(outdir / f"{filename_prefix}_pr_joint.png")
    else:
        plt.close()


def plot_calibration_curve_overall(y_true_binary: np.ndarray, y_prob: np.ndarray, outpath: Path,
    n_bins: int = 10, color: Optional[str] = None, title: str = "Calibration curve"):
    
    y_true_binary = np.asarray(y_true_binary)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    frac_pos = []
    mean_pred = []

    for i in range(n_bins):
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if in_bin.sum() == 0:
            continue
        frac_pos.append(y_true_binary[in_bin].mean())
        mean_pred.append(y_prob[in_bin].mean())

    if not mean_pred:
        return

    if color is None:
        title_lower = title.lower()
        if "death" in title_lower:
            color = CLASS_COLORS[3]
        elif "severe" in title_lower:
            color = CLASS_COLORS[2]
        else:
            color = "C0"

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(mean_pred, frac_pos, marker="o", linestyle="-", color=color, label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    _save_current_fig(Path(outpath))

def plot_binary_roc_pr_overlay(y_true_binary: np.ndarray, model_probs: dict, outdir: Path, prefix: str):

    outdir = Path(outdir)
    y_true_binary = np.asarray(y_true_binary).astype(int)

    # ROC
    fig = plt.figure()
    ax = plt.gca()

    metrics_lines = []
    for name, p in model_probs.items():
        p = np.asarray(p, dtype=float)
        mask = np.isfinite(p)
        if mask.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true_binary[mask], p[mask])
        roc_auc = auc(fpr, tpr)

        st = _style_for(name)
        label = f"{name} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, label=label, **st)

    st = _style_for("Chance")
    ax.plot([0, 1], [0, 1], label="Chance", **st)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{prefix}: ROC overlay")
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _apply_ordered_legend(ax)

    # metrics annotation (top-left)
    if metrics_lines:
        ax.text(
            0.02, 0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85", alpha=0.9),
        )

    _save_current_fig(outdir / f"{prefix}_roc_overlay.png")
    plt.close(fig)

    # PR
    fig = plt.figure()
    ax = plt.gca()

    metrics_lines = []
    for name, p in model_probs.items():
        p = np.asarray(p, dtype=float)
        mask = np.isfinite(p)
        if mask.sum() == 0:
            continue

        prec, rec, _ = precision_recall_curve(y_true_binary[mask], p[mask])
        ap = average_precision_score(y_true_binary[mask], p[mask])

        st = _style_for(name)
        label = f"{name} (AP={ap:.3f})"
        ax.plot(rec, prec, label=label, **st)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix}: PR overlay")
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _apply_ordered_legend(ax)

    if metrics_lines:
        ax.text(
            0.02, 0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85", alpha=0.9),
        )

    _save_current_fig(outdir / f"{prefix}_pr_overlay.png")
    plt.close(fig)

# Dataset summary
def dataset_summary(df: pd.DataFrame, target_col: str, outdir: Path):
    """
    dataset description: patients, visits, wave gaps, age gaps (if available), class balance.

    Assumptions:
      - df contains 'patient_idx'
      - df contains 'time' (Wave)
      - df contain 'age_time' (Age) for irregular gap summary
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    visits_per_patient = df.groupby("patient_idx").size()

    df_sorted = df.sort_values(["patient_idx", "time"]).copy()
    df_sorted["dt_wave"] = df_sorted.groupby("patient_idx")["time"].diff()

    summary = {
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_idx"].nunique()),
        "mean_visits_per_patient": float(visits_per_patient.mean()),
        "std_visits_per_patient": float(visits_per_patient.std()),
        "mean_dt_wave": float(df_sorted["dt_wave"].dropna().mean()) if df_sorted["dt_wave"].notna().any() else None,
        "std_dt_wave": float(df_sorted["dt_wave"].dropna().std()) if df_sorted["dt_wave"].notna().any() else None,
        "class_counts": df[target_col].value_counts().sort_index().to_dict(),
    }

    if "age_time" in df_sorted.columns:
        df_sorted["dt_age"] = df_sorted.groupby("patient_idx")["age_time"].diff()
        summary.update({
            "mean_dt_age": float(df_sorted["dt_age"].dropna().mean()) if df_sorted["dt_age"].notna().any() else None,
            "std_dt_age": float(df_sorted["dt_age"].dropna().std()) if df_sorted["dt_age"].notna().any() else None,
        })

    save_json(summary, outdir / "dataset_summary.json")

    plt.figure()
    plt.hist(visits_per_patient, bins=20)
    plt.xlabel("# visits per patient")
    plt.ylabel("# patients")
    plt.title("Visits per patient")
    _save_current_fig(outdir / "hist_visits_per_patient.png")

    plt.figure()
    plt.hist(df_sorted["dt_wave"].dropna(), bins=20)
    plt.xlabel("ΔWave between visits")
    plt.ylabel("# intervals")
    plt.title("Wave gaps between visits")
    _save_current_fig(outdir / "hist_dt_wave.png")

    if "age_time" in df_sorted.columns:
        plt.figure()
        plt.hist(df_sorted["dt_age"].dropna(), bins=20)
        plt.xlabel("ΔAge between visits")
        plt.ylabel("# intervals")
        plt.title("Age gaps between visits")
        _save_current_fig(outdir / "hist_dt_age.png")