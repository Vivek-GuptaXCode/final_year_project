from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_rsu_hash(rsu_id: str, modulo: int = 997) -> float:
    digest = hashlib.md5(rsu_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return float(int(digest[:8], 16) % max(2, modulo))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute a simple ECE estimate for binary probabilities."""
    if y_true.size == 0:
        return float("nan")

    y_true = y_true.astype(float)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, max(2, n_bins + 1))

    ece = 0.0
    n = float(y_true.size)
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (count / n) * abs(avg_conf - avg_acc)

    return float(ece)


def rolling_expanding_splits(
    n_samples: int,
    n_splits: int,
    test_size: int,
    gap: int,
    min_train_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window splits without randomization."""
    if n_samples <= 0:
        return []

    test_size = max(1, int(test_size))
    gap = max(0, int(gap))
    min_train_size = max(1, int(min_train_size))
    n_splits = max(1, int(n_splits))

    max_possible = (n_samples - min_train_size - gap) // test_size
    if max_possible <= 0:
        return []

    n_folds = min(n_splits, max_possible)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_folds):
        remaining_folds = n_folds - i - 1
        test_end = n_samples - remaining_folds * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        if train_end < min_train_size:
            continue

        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        splits.append((train_idx, test_idx))

    return splits


def safe_mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))
