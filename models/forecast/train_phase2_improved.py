"""Phase 2 improved trainer: LightGBM, XGBoost-v2, PyTorch MLP, soft-voting ensemble.

Uses feature_builder_v2 (31 features) for richer temporal/congestion context.
Exports a v2 artifact compatible with inference_v2.

Usage:
    python models/forecast/train_phase2_improved.py \\
        --processed-glob 'data/processed/phase2_selected_passed_2x/*/rsu_horizon_labels.csv' \\
        --profile local
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── sklearn ──────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import (  # type: ignore
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    HistGradientBoostingClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore

# ── XGBoost ──────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore

# ── LightGBM ─────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:
    HAS_LGB = False
    lgb = None  # type: ignore

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from models.forecast.common import (
    clamp01,
    compute_expected_calibration_error,
    ensure_dir,
    now_utc_iso,
    rolling_expanding_splits,
    safe_mean,
)
from models.forecast.feature_builder_v2 import (
    FEATURE_NAMES_V2,
    FeatureStateV2,
    build_training_features_from_row_v2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    model_kind: str
    fold_metrics: list[dict[str, Any]]
    summary: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 improved trainer.")
    p.add_argument(
        "--processed-glob",
        default="data/processed/phase2_selected_passed_2x/*/rsu_horizon_labels.csv",
    )
    p.add_argument("--target-column", default="label_congestion_60s")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=int, default=360)
    p.add_argument("--gap", type=int, default=0)
    p.add_argument("--min-train-size", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="models/forecast/artifacts")
    p.add_argument("--report-path", default="docs/reports/phase2_forecast_report.md")
    p.add_argument("--mlp-epochs", type=int, default=40, help="Epochs per CV fold for MLP.")
    p.add_argument("--mlp-final-epochs", type=int, default=80, help="Epochs for final MLP retrain.")
    p.add_argument("--skip-mlp", action="store_true", help="Skip PyTorch MLP (faster run).")
    p.add_argument("--profile", default="local")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset(
    paths: list[Path], target_col: str, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        run_id = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if target_col not in row:
                    continue
                r = dict(row)
                r["run_id"] = run_id
                all_rows.append(r)

    def _key(r: dict[str, Any]) -> tuple[str, float, int, str]:
        try:
            ts = float(r.get("timestamp_s", 0))
        except (TypeError, ValueError):
            ts = 0.0
        try:
            fi = int(float(r.get("frame_idx", 0)))
        except (TypeError, ValueError):
            fi = 0
        return (str(r.get("run_id", "")), ts, fi, str(r.get("rsu_node", "")))

    all_rows.sort(key=_key)

    if max_rows and len(all_rows) > max_rows:
        grouped: dict[str, list] = {}
        for r in all_rows:
            grouped.setdefault(str(r.get("run_id", "unk")), []).append(r)
        run_ids = sorted(grouped)
        per = max_rows // max(1, len(run_ids))
        rem = max_rows % max(1, len(run_ids))
        limited: list[dict[str, Any]] = []
        for i, rid in enumerate(run_ids):
            take = per + (1 if i < rem else 0)
            limited.extend(grouped[rid][:take])
        limited.sort(key=_key)
        all_rows = limited

    states: dict[tuple[str, str], FeatureStateV2] = {}
    xs, ys, kept = [], [], []
    for row in all_rows:
        try:
            y = int(float(row.get(target_col, 0)))
        except (TypeError, ValueError):
            continue
        y = 1 if y >= 1 else 0
        xs.append(build_training_features_from_row_v2(row, states))
        ys.append(y)
        kept.append(row)

    if not xs:
        return np.zeros((0, len(FEATURE_NAMES_V2)), dtype=float), np.zeros((0,), dtype=int), []

    return np.vstack(xs).astype(float), np.array(ys, dtype=int), kept


# ─────────────────────────────────────────────────────────────────────────────
# Rolling CV splits (label-aware, same logic as original trainer)
# ─────────────────────────────────────────────────────────────────────────────

def _window_ok(y: np.ndarray, start: int, size: int) -> bool:
    if start < 0 or start + size > len(y):
        return False
    w = y[start : start + size]
    return w.size > 1 and int(w.min()) != int(w.max())


def _find_label_aware_start(
    y: np.ndarray, *, preferred: int, low: int, high: int, size: int
) -> int | None:
    if low > high:
        return None
    candidates = sorted(range(low, high + 1), key=lambda s: (abs(s - preferred), s))
    for s in candidates:
        if _window_ok(y, s, size):
            return s
    return None


def _build_splits(
    y: np.ndarray, n_splits: int, test_size: int, gap: int, min_train: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    base = rolling_expanding_splits(len(y), n_splits, test_size, gap, min_train)
    if not base:
        return []

    preferred_starts = [int(ti[0]) for _, ti in base]
    max_start = len(y) - test_size
    low_global = min_train + gap

    # Build from the end backwards to avoid overlapping test windows
    selected_rev: list[int] = []
    next_bound = len(y)
    for pref in reversed(preferred_starts):
        hi = min(max_start, next_bound - test_size)
        lo = max(0, low_global)
        chosen = _find_label_aware_start(y, preferred=pref, low=lo, high=hi, size=test_size)
        if chosen is None:
            continue
        selected_rev.append(chosen)
        next_bound = chosen

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for start in reversed(selected_rev):
        tr_end = start - gap
        if tr_end < min_train:
            continue
        te = np.arange(start, start + test_size, dtype=int)
        if not _window_ok(y, start, test_size):
            continue
        splits.append((np.arange(0, tr_end, dtype=int), te))
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, p_pred: np.ndarray, latency_ms: float) -> dict[str, float]:
    p = np.clip(p_pred.astype(float), 1e-6, 1 - 1e-6)
    y = y_true.astype(int)
    m: dict[str, float] = {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": float(compute_expected_calibration_error(y, p, n_bins=10)),
        "latency_ms_per_sample": latency_ms,
        "positive_rate": float(y.mean()) if y.size else float("nan"),
    }
    if len(np.unique(y)) >= 2:
        m["roc_auc"] = float(roc_auc_score(y, p))
    else:
        m["roc_auc"] = float("nan")
    if int(y.sum()) > 0:
        m["pr_auc"] = float(average_precision_score(y, p))
    else:
        m["pr_auc"] = float("nan")
    return m


def _summarize(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["brier", "ece", "log_loss", "roc_auc", "pr_auc", "latency_ms_per_sample"]
    summary: dict[str, Any] = {"fold_count": len(fold_metrics)}
    for k in keys:
        vals = []
        for fm in fold_metrics:
            try:
                vals.append(float(fm.get(k, float("nan"))))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        summary[f"{k}_mean"] = safe_mean(vals)
        finite = [v for v in vals if not math.isnan(v)]
        summary[f"{k}_std"] = float(np.std(finite)) if len(finite) >= 2 else float("nan")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Persistence baseline
# ─────────────────────────────────────────────────────────────────────────────

def _persistence_proba(x: np.ndarray) -> np.ndarray:
    idx_c = FEATURE_NAMES_V2.index("connected_vehicle_count")
    idx_l = FEATURE_NAMES_V2.index("congested_local")
    count = np.clip(x[:, idx_c] / 25.0, 0, 1)
    local = np.clip(x[:, idx_l], 0, 1)
    return np.clip(0.10 + 0.55 * local + 0.35 * count, 0, 1)


def _eval_persistence(x: np.ndarray, y: np.ndarray, splits: list) -> ModelResult:
    fms = []
    for i, (_, ti) in enumerate(splits, 1):
        t0 = time.perf_counter()
        p = _persistence_proba(x[ti])
        lat = (time.perf_counter() - t0) * 1000 / max(1, len(ti))
        fm = {"fold": i, "test_size": int(ti.size), "status": "ok"}
        fm.update(_metrics(y[ti], p, lat))
        fms.append(fm)
    return ModelResult("persistence_v1", "rule", fms, _summarize(fms))


# ─────────────────────────────────────────────────────────────────────────────
# Gradient boosting models
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgb(seed: int) -> Any:
    if not HAS_LGB:
        raise RuntimeError("lightgbm unavailable")
    return lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _make_lgb_dart(seed: int) -> Any:
    """LightGBM DART with higher capacity — slower but more accurate."""
    if not HAS_LGB:
        raise RuntimeError("lightgbm unavailable")
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.04,
        boosting_type="dart",
        num_leaves=255,
        max_depth=10,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.0,
        drop_rate=0.15,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _make_xgb(seed: int, pos_weight: float = 1.0) -> Any:
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost unavailable")
    return XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        device="cpu",
    )


def _make_xgb_gpu(seed: int, pos_weight: float = 1.0) -> Any:
    """XGBoost with GPU acceleration and higher capacity."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost unavailable")
    return XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.3,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        device="cuda",
    )


def _make_histgb(seed: int) -> Any:
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn unavailable")
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=seed,
    )


def _eval_gbm(
    *,
    name: str,
    kind: str,
    factory: Callable[[int], Any],
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
    pos_weight_aware: bool = False,
) -> ModelResult:
    fms = []
    for i, (tri, ti) in enumerate(splits, 1):
        yt, yv = y[tri], y[ti]
        row: dict[str, Any] = {
            "fold": i,
            "train_size": int(tri.size),
            "test_size": int(ti.size),
            "train_positive_rate": float(yt.mean()) if yt.size else float("nan"),
            "status": "ok",
        }
        if len(np.unique(yt)) < 2:
            p = np.full(ti.size, clamp01(float(yt.mean()) if yt.size else 0.5))
            row["status"] = "fallback_single_class"
            row.update(_metrics(yv, p, 0.0))
            fms.append(row)
            continue

        pw = float((yt == 0).sum()) / max(1, int((yt == 1).sum())) if pos_weight_aware else 1.0
        est = factory(seed + i) if not pos_weight_aware else _make_xgb(seed + i, pw)
        t0 = time.perf_counter()
        est.fit(x[tri], yt)
        p = est.predict_proba(x[ti])[:, 1]
        lat = (time.perf_counter() - t0) * 1000 / max(1, ti.size)
        row.update(_metrics(yv, p, lat))
        fms.append(row)

    return ModelResult(name, kind, fms, _summarize(fms))


# ─────────────────────────────────────────────────────────────────────────────
# Soft-voting ensemble
# ─────────────────────────────────────────────────────────────────────────────

class SoftEnsemble:
    """Average predict_proba of two sklearn-compatible models."""

    def __init__(self, m1: Any, m2: Any) -> None:
        self.m1 = m1
        self.m2 = m2

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = self.m1.predict_proba(x)[:, 1]
        p2 = self.m2.predict_proba(x)[:, 1]
        avg = (p1 + p2) / 2.0
        return np.column_stack([1 - avg, avg])


def _eval_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
    name: str = "ensemble_lgb_xgb_v1",
    use_dart: bool = False,
    use_gpu: bool = False,
) -> ModelResult:
    fms = []
    for i, (tri, ti) in enumerate(splits, 1):
        yt, yv = y[tri], y[ti]
        row: dict[str, Any] = {
            "fold": i,
            "train_size": int(tri.size),
            "test_size": int(ti.size),
            "status": "ok",
        }
        if len(np.unique(yt)) < 2:
            p = np.full(ti.size, 0.5)
            row["status"] = "fallback_single_class"
            row.update(_metrics(yv, p, 0.0))
            fms.append(row)
            continue

        pw = float((yt == 0).sum()) / max(1, int((yt == 1).sum()))
        lgb_m = _make_lgb_dart(seed + i) if use_dart else _make_lgb(seed + i)
        xgb_m = _make_xgb_gpu(seed + i, pw) if use_gpu else _make_xgb(seed + i, pw)
        t0 = time.perf_counter()
        lgb_m.fit(x[tri], yt)
        xgb_m.fit(x[tri], yt)
        ens = SoftEnsemble(lgb_m, xgb_m)
        p = ens.predict_proba(x[ti])[:, 1]
        lat = (time.perf_counter() - t0) * 1000 / max(1, ti.size)
        row.update(_metrics(yv, p, lat))
        fms.append(row)

    return ModelResult(name, "ensemble", fms, _summarize(fms))


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch MLP
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class TabularMLP(nn.Module):
        def __init__(self, n_in: int, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.3) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.BatchNorm1d(n_in)]
            prev = n_in
            for i, h in enumerate(hidden):
                layers.append(nn.Linear(prev, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if i < len(hidden) - 1:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.net(x)).squeeze(1)


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    epochs: int,
    seed: int,
    batch_size: int = 512,
) -> tuple[np.ndarray, float]:
    if not HAS_TORCH:
        raise RuntimeError("torch unavailable")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    xt = scaler.fit_transform(x_train).astype(np.float32)
    xv = scaler.transform(x_test).astype(np.float32)

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    pos_w = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

    ds = TensorDataset(
        torch.tensor(xt, device=device),
        torch.tensor(y_train.astype(np.float32), device=device),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TabularMLP(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction="none")

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            w = torch.where(yb == 1, pos_w.expand_as(yb), torch.ones_like(yb))
            loss = (criterion(pred, yb) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    xv_t = torch.tensor(xv, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        p = model(xv_t).cpu().numpy()
    lat = (time.perf_counter() - t0) * 1000 / max(1, len(xv))
    return p, lat


def _eval_mlp(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
    epochs: int,
) -> ModelResult:
    fms = []
    for i, (tri, ti) in enumerate(splits, 1):
        yt, yv = y[tri], y[ti]
        row: dict[str, Any] = {
            "fold": i,
            "train_size": int(tri.size),
            "test_size": int(ti.size),
            "status": "ok",
        }
        if len(np.unique(yt)) < 2:
            p = np.full(ti.size, 0.5)
            row["status"] = "fallback_single_class"
            row.update(_metrics(yv, p, 0.0))
            fms.append(row)
            continue

        try:
            p, lat = _train_mlp(x[tri], yt, x[ti], epochs=epochs, seed=seed + i)
            row.update(_metrics(yv, p, lat))
        except Exception as exc:
            row["status"] = f"error: {exc}"
            row.update({"brier": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan"), "ece": float("nan")})
        fms.append(row)
    return ModelResult("mlp_tabular_v1", "pytorch", fms, _summarize(fms))


# ─────────────────────────────────────────────────────────────────────────────
# Best model selection
# ─────────────────────────────────────────────────────────────────────────────

def _score(r: ModelResult) -> tuple[float, float, float]:
    brier = float(r.summary.get("brier_mean", float("nan")))
    ece = float(r.summary.get("ece_mean", float("nan")))
    roc = float(r.summary.get("roc_auc_mean", float("nan")))
    return (
        brier if not math.isnan(brier) else 1e6,
        ece if not math.isnan(ece) else 1e6,
        -roc if not math.isnan(roc) else 1e6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Final model training (full dataset)
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_final(
    best: ModelResult, x: np.ndarray, y: np.ndarray, seed: int, mlp_epochs: int
) -> tuple[Any, str, Any]:
    """Returns (model_obj, model_filename, scaler_or_None)."""
    if best.name == "persistence_v1":
        return None, "none", None

    if len(np.unique(y)) < 2:
        return None, "none", None

    pw = float((y == 0).sum()) / max(1, int((y == 1).sum()))

    if best.name == "lightgbm_v1":
        m = _make_lgb(seed)
        m.fit(x, y)
        return m, "model.pkl", None

    if best.name == "xgboost_v2":
        m = _make_xgb(seed, pw)
        m.fit(x, y)
        return m, "model.json", None

    if best.name == "hist_gradient_boosting_v2":
        m = _make_histgb(seed)
        m.fit(x, y)
        return m, "model.pkl", None

    if best.name in ("ensemble_lgb_xgb_v1", "ensemble_dart_xgb_v1"):
        m1 = _make_lgb_dart(seed) if "dart" in best.name else _make_lgb(seed)
        m2 = _make_xgb_gpu(seed, pw) if "xgb_gpu" in best.name else _make_xgb(seed, pw)
        m1.fit(x, y)
        m2.fit(x, y)
        ens = SoftEnsemble(m1, m2)
        return ens, "model_ensemble.pkl", None

    if best.name == "lightgbm_dart_v1":
        m = _make_lgb_dart(seed)
        m.fit(x, y)
        return m, "model.pkl", None

    if best.name == "xgboost_gpu_v1":
        m = _make_xgb_gpu(seed, pw)
        m.fit(x, y)
        return m, "model.json", None

    if best.name == "mlp_tabular_v1":
        if not HAS_TORCH:
            return None, "none", None
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sc = StandardScaler()
        xt = sc.fit_transform(x).astype(np.float32)
        yt = y.astype(np.float32)
        pos = int(yt.sum())
        neg = len(yt) - pos
        pos_w = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
        ds = TensorDataset(
            torch.tensor(xt, device=device),
            torch.tensor(yt, device=device),
        )
        dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
        model = TabularMLP(x.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCELoss(reduction="none")
        for _ in range(mlp_epochs):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                pred = model(xb)
                w = torch.where(yb == 1, pos_w.expand_as(yb), torch.ones_like(yb))
                loss = (criterion(pred, yb) * w).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        return model, "model_mlp.pt", sc

    return None, "none", None


# ─────────────────────────────────────────────────────────────────────────────
# Artifact export
# ─────────────────────────────────────────────────────────────────────────────

def _write_artifact(
    run_dir: Path,
    best: ModelResult,
    model_obj: Any,
    model_filename: str,
    scaler: Any,
    all_results: list[ModelResult],
    dataset_meta: dict[str, Any],
    cv_cfg: dict[str, Any],
    target_col: str,
) -> Path:
    model_file_path = None
    if model_obj is not None and model_filename != "none":
        model_file_path = run_dir / model_filename
        if model_filename.endswith(".json"):
            model_obj.save_model(str(model_file_path))
        elif model_filename.endswith(".pt"):
            torch.save(model_obj.state_dict(), str(model_file_path))
        else:
            with model_file_path.open("wb") as fh:
                pickle.dump(model_obj, fh)

    if scaler is not None:
        with (run_dir / "scaler.pkl").open("wb") as fh:
            pickle.dump(scaler, fh)

    summaries = []
    for r in all_results:
        row = {"name": r.name, "model_kind": r.model_kind}
        row.update(r.summary)
        summaries.append(row)

    payload: dict[str, Any] = {
        "artifact_version": "phase2_forecast_artifact_v2",
        "generated_utc": now_utc_iso(),
        "target_column": target_col,
        "model": {
            "name": best.name,
            "kind": best.model_kind,
            "model_file": model_filename if model_file_path else None,
            "scaler_file": "scaler.pkl" if scaler else None,
            "hidden_dims": [256, 128, 64] if best.model_kind == "pytorch" else None,
            "trusted_local_only": True,
        },
        "feature_contract": {
            "version": "v2",
            "feature_names": FEATURE_NAMES_V2,
            "source": "models.forecast.feature_builder_v2",
        },
        "dataset": dataset_meta,
        "cv_config": cv_cfg,
        "baseline_summaries": summaries,
        "selected_model": {"name": best.name, "summary": best.summary},
        "fold_details": {r.name: r.fold_metrics for r in all_results},
        "inference_output_contract": {
            "p_congestion": "float[0,1]",
            "confidence": "float[0,1]",
            "uncertainty": "float[0,1]",
        },
    }

    artifact_path = run_dir / "forecast_artifact.json"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 2 Improved Forecast Report", ""]
    lines += [
        f"- Generated UTC: {payload['generated_utc']}",
        f"- Target: {payload['target_column']}",
        f"- Processed files: {payload['dataset']['file_count']}",
        f"- Total rows: {payload['dataset']['row_count']}",
        f"- Positive rate: {payload['dataset']['positive_rate']:.4f}",
        f"- Rolling folds: {payload['cv_config']['n_splits_actual']}",
        f"- Feature set: v2 ({len(FEATURE_NAMES_V2)} features)",
        f"- Best model: {payload['selected_model']['name']}",
        "",
    ]
    lines += ["## Baseline Comparison (CV means)", ""]
    lines += ["| model | brier | ece | roc_auc | pr_auc | latency_ms/sample |"]
    lines += ["|---|---:|---:|---:|---:|---:|"]
    for row in payload["baseline_summaries"]:
        def _fmt(k: str) -> str:
            v = row.get(k, float("nan"))
            try:
                f = float(v)
                return f"{f:.6f}" if not math.isnan(f) else "nan"
            except (TypeError, ValueError):
                return "nan"
        lines.append(
            f"| {row['name']} | {_fmt('brier_mean')} | {_fmt('ece_mean')} "
            f"| {_fmt('roc_auc_mean')} | {_fmt('pr_auc_mean')} | {_fmt('latency_ms_per_sample_mean')} |"
        )
    lines += ["", "## Notes", ""]
    lines += [
        "- Feature set v2 uses 31 features (lags t-1..t-5, rolling stats, congestion duration).",
        "- LightGBM and XGBoost use class_weight/scale_pos_weight for imbalance handling.",
        "- MLP uses BatchNorm + Dropout + weighted BCE loss on GTX 1650.",
        "- Ensemble is soft voting (average probabilities) of LightGBM + XGBoost.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    processed_paths = sorted(Path(p) for p in glob.glob(args.processed_glob))
    if not processed_paths:
        print("[P2-IMPROVED] No files matched glob.")
        return 2

    print(f"[P2-IMPROVED] Loading {len(processed_paths)} files …")
    x, y, kept_rows = _load_dataset(processed_paths, args.target_column, args.max_rows)
    if x.shape[0] == 0:
        print("[P2-IMPROVED] Dataset empty.")
        return 2

    print(f"[P2-IMPROVED] Dataset: {x.shape[0]} rows, {x.shape[1]} features, "
          f"positive_rate={float(y.mean()):.4f}")

    splits = _build_splits(y, args.n_splits, args.test_size, args.gap, args.min_train_size)
    if not splits:
        print("[P2-IMPROVED] Could not build splits.")
        return 2
    print(f"[P2-IMPROVED] CV splits: {len(splits)}")

    results: list[ModelResult] = []

    # Persistence baseline
    print("[P2-IMPROVED] Evaluating persistence_v1 …")
    results.append(_eval_persistence(x, y, splits))
    print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
          f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")

    # LightGBM
    if HAS_LGB:
        print("[P2-IMPROVED] Evaluating lightgbm_v1 …")
        results.append(_eval_gbm(
            name="lightgbm_v1", kind="lightgbm",
            factory=_make_lgb, x=x, y=y, splits=splits, seed=args.seed,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")
    else:
        print("[P2-IMPROVED] lightgbm unavailable; skipping.")

    # XGBoost v2
    if HAS_XGBOOST:
        print("[P2-IMPROVED] Evaluating xgboost_v2 …")
        results.append(_eval_gbm(
            name="xgboost_v2", kind="xgboost",
            factory=_make_xgb, x=x, y=y, splits=splits, seed=args.seed,
            pos_weight_aware=True,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")
    else:
        print("[P2-IMPROVED] xgboost unavailable; skipping.")

    # HistGB v2
    if HAS_SKLEARN:
        print("[P2-IMPROVED] Evaluating hist_gradient_boosting_v2 …")
        results.append(_eval_gbm(
            name="hist_gradient_boosting_v2", kind="sklearn",
            factory=_make_histgb, x=x, y=y, splits=splits, seed=args.seed,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")

    # Ensemble (LGB + XGB standard)
    if HAS_LGB and HAS_XGBOOST:
        print("[P2-IMPROVED] Evaluating ensemble_lgb_xgb_v1 …")
        results.append(_eval_ensemble(x, y, splits, args.seed))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")

    # XGBoost GPU high capacity
    if HAS_XGBOOST:
        print("[P2-IMPROVED] Evaluating xgboost_gpu_v1 (CUDA) …")
        results.append(_eval_gbm(
            name="xgboost_gpu_v1", kind="xgboost",
            factory=_make_xgb_gpu, x=x, y=y, splits=splits, seed=args.seed,
            pos_weight_aware=True,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")

    # LightGBM DART (slower — run last)
    if HAS_LGB:
        print("[P2-IMPROVED] Evaluating lightgbm_dart_v1 (DART, slower) …")
        results.append(_eval_gbm(
            name="lightgbm_dart_v1", kind="lightgbm",
            factory=_make_lgb_dart, x=x, y=y, splits=splits, seed=args.seed,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")

    # Ensemble DART + XGB-GPU
    if HAS_LGB and HAS_XGBOOST:
        print("[P2-IMPROVED] Evaluating ensemble_dart_xgb_v1 …")
        results.append(_eval_ensemble(
            x, y, splits, args.seed,
            name="ensemble_dart_xgb_v1", use_dart=True, use_gpu=True,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")

    # MLP
    if HAS_TORCH and HAS_SKLEARN and not args.skip_mlp:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[P2-IMPROVED] Evaluating mlp_tabular_v1 (device={device_str}, epochs={args.mlp_epochs}) …")
        results.append(_eval_mlp(x, y, splits, args.seed, args.mlp_epochs))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"brier={results[-1].summary.get('brier_mean', 'nan'):.4f}")
    elif not HAS_TORCH:
        print("[P2-IMPROVED] torch unavailable; skipping MLP.")

    best = sorted(results, key=_score)[0]
    print(f"[P2-IMPROVED] Best model: {best.name} "
          f"(roc_auc={best.summary.get('roc_auc_mean', 'nan'):.4f}, "
          f"brier={best.summary.get('brier_mean', 'nan'):.4f})")

    print(f"[P2-IMPROVED] Retraining {best.name} on full dataset …")
    model_obj, model_filename, scaler = _retrain_final(best, x, y, args.seed, args.mlp_final_epochs)

    run_id = f"phase2_improved_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = ensure_dir(Path(args.output_dir) / run_id)

    run_ids = sorted({str(r.get("run_id", "unknown")) for r in kept_rows})
    dataset_meta = {
        "file_count": len(processed_paths),
        "rows_used": int(x.shape[0]),
        "positive_rate": float(y.mean()),
        "run_ids": run_ids,
        "feature_version": "v2",
    }
    cv_cfg = {
        "n_splits_actual": len(splits),
        "test_size": args.test_size,
        "gap": args.gap,
        "min_train_size": args.min_train_size,
    }

    artifact_path = _write_artifact(
        run_dir=run_dir,
        best=best,
        model_obj=model_obj,
        model_filename=model_filename,
        scaler=scaler,
        all_results=results,
        dataset_meta=dataset_meta,
        cv_cfg=cv_cfg,
        target_col=args.target_column,
    )

    # Update latest/
    latest_dir = ensure_dir(Path(args.output_dir) / "latest")
    for stale in latest_dir.glob("model*"):
        try:
            stale.unlink()
        except OSError:
            pass
    shutil.copy2(artifact_path, latest_dir / "forecast_artifact.json")
    if model_obj is not None and model_filename != "none":
        shutil.copy2(run_dir / model_filename, latest_dir / model_filename)
    if scaler is not None:
        shutil.copy2(run_dir / "scaler.pkl", latest_dir / "scaler.pkl")

    report_payload = {
        "generated_utc": now_utc_iso(),
        "target_column": args.target_column,
        "dataset": {"file_count": len(processed_paths), "row_count": int(x.shape[0]),
                    "positive_rate": float(y.mean())},
        "cv_config": cv_cfg,
        "selected_model": {"name": best.name, "summary": best.summary},
        "baseline_summaries": [{"name": r.name, "model_kind": r.model_kind, **r.summary} for r in results],
    }
    _write_report(Path(args.report_path), report_payload)

    print(f"[P2-IMPROVED] Artifact: {artifact_path}")
    print(f"[P2-IMPROVED] Latest:   {latest_dir}")
    print(f"[P2-IMPROVED] Report:   {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
