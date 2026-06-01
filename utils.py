import json
import os
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42, deterministic: bool = False, single_thread: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if single_thread:
        # helps reproducibility on CPU for some ops
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    if deterministic:
        # CUDNN determinism knobs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # older torch or unsupported op; best-effort
            pass


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def ensure_dir(path: Union[str, Path]) -> Path:
    path = str(path)
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Multiclass Brier score:
      mean_i sum_k (p_ik - 1[y_i=k])^2
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2:
        raise ValueError("brier_score expects y_prob shape [N,K].")

    n, k = y_prob.shape
    if n == 0:
        return float("nan")

    y_onehot = np.zeros((n, k), dtype=float)
    # safety: ignore labels outside range
    mask = (y_true >= 0) & (y_true < k)
    y_onehot[np.arange(n)[mask], y_true[mask]] = 1.0

    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Multiclass ECE using confidence=max(prob) and accuracy within confidence bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2:
        raise ValueError("expected_calibration_error expects y_prob shape [N,K].")

    n = len(y_true)
    if n == 0:
        return float("nan")

    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # include right edge only on last bin
        if i == n_bins - 1:
            idx = (conf >= lo) & (conf <= hi)
        else:
            idx = (conf >= lo) & (conf < hi)

        m = int(np.sum(idx))
        if m == 0:
            continue

        bin_acc = float(np.mean(acc[idx]))
        bin_conf = float(np.mean(conf[idx]))
        ece += (m / n) * abs(bin_acc - bin_conf)

    return float(ece)


def save_fig(fig: plt.Figure, path: Union[str, Path], dpi: int = 200) -> None:
    path = str(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    def to_py(x: Any) -> Any:
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer, np.int32, np.int64)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, dict):
            return {str(k): to_py(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_py(v) for v in x]
        return x

    path = str(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_py(obj), f, indent=indent, sort_keys=False)