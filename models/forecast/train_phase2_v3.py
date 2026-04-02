"""Phase 2 V3 trainer: Enhanced 45-feature set for improved congestion forecasting.

Uses feature_builder_v3 with:
- Cross-RSU spatial features
- Enhanced time encoding (hour sin/cos, peak hour)
- Advanced rolling windows (median, range, EMA)
- Velocity/acceleration ratio

Target: 87.3% → 95% accuracy

Usage:
    python models/forecast/train_phase2_v3.py \\
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── sklearn ──────────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ── LightGBM ─────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False
    lgb = None

# ── XGBoost ──────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None

from models.forecast.common import (
    clamp01,
    compute_expected_calibration_error,
    ensure_dir,
    now_utc_iso,
    rolling_expanding_splits,
    safe_mean,
)
from models.forecast.feature_builder_v3 import (
    FEATURE_NAMES_V3,
    FeatureStateV3,
    RSUNeighborhood,
    build_training_features_from_row_v3,
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
    p = argparse.ArgumentParser(description="Train Phase 2 V3 forecast model")
    p.add_argument(
        "--processed-glob",
        default="data/processed/*/rsu_horizon_labels.csv",
        help="Glob pattern for input CSVs",
    )
    p.add_argument(
        "--output-dir",
        default="models/forecast/artifacts",
        help="Output directory for artifacts",
    )
    p.add_argument("--target-column", default="label_congestion_60s", help="Target column")
    p.add_argument("--max-rows", type=int, default=500_000, help="Max rows to load")
    p.add_argument("--n-splits", type=int, default=5, help="CV folds")
    p.add_argument("--test-size", type=int, default=360, help="Test size per fold")
    p.add_argument("--gap", type=int, default=30, help="Gap between train and test")
    p.add_argument("--min-train-size", type=int, default=1200, help="Min training size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--profile", default="local", help="Profile name")
    p.add_argument("--calibrate", action="store_true", help="Apply Platt scaling calibration")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _build_rsu_neighborhood_from_data(
    rows: list[dict[str, Any]],
) -> RSUNeighborhood:
    """Build RSU neighbor relationships from spatial proximity in data."""
    neighborhood = RSUNeighborhood()

    # Group RSUs by run
    runs = defaultdict(set)
    for row in rows:
        run_id = str(row.get("run_id", "unknown"))
        rsu_id = str(row.get("rsu_node", "RSU_UNKNOWN"))
        runs[run_id].add(rsu_id)

    # For each run, assume all RSUs are neighbors (simplified)
    # In production, this would use actual road network topology
    for run_id, rsus in runs.items():
        rsu_list = list(rsus)
        for i, rsu1 in enumerate(rsu_list):
            for rsu2 in rsu_list[i + 1 :]:
                neighborhood.register_neighbor(rsu1, rsu2)

    return neighborhood


def _load_dataset(
    paths: list[Path],
    target_col: str,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load dataset with V3 features and neighbor relationships."""
    all_rows: list[dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                if len(all_rows) >= max_rows:
                    break
        if len(all_rows) >= max_rows:
            break

    if not all_rows:
        return np.empty((0, len(FEATURE_NAMES_V3))), np.empty((0,)), []

    # Sort by run_id and timestamp for temporal coherence
    all_rows.sort(
        key=lambda r: (
            str(r.get("run_id", "")),
            float(r.get("timestamp_s", r.get("frame_idx", 0))),
        )
    )

    # Build neighborhood from data
    neighborhood = _build_rsu_neighborhood_from_data(all_rows)

    # Build features
    state_by_stream: dict[tuple[str, str], FeatureStateV3] = {}
    features: list[np.ndarray] = []
    labels: list[float] = []
    kept_rows: list[dict[str, Any]] = []

    for row in all_rows:
        try:
            label = float(row.get(target_col, 0))
        except (TypeError, ValueError):
            continue

        feat = build_training_features_from_row_v3(row, state_by_stream, neighborhood)
        features.append(feat)
        labels.append(label)
        kept_rows.append(row)

    x = np.vstack(features) if features else np.empty((0, len(FEATURE_NAMES_V3)))
    y = np.array(labels, dtype=float)
    return x, y, kept_rows


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation splits
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
    y_pred = (p >= 0.5).astype(int)

    m: dict[str, float] = {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": float(compute_expected_calibration_error(y, p, n_bins=10)),
        "latency_ms_per_sample": latency_ms,
        "positive_rate": float(y.mean()) if y.size else float("nan"),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
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
    keys = ["brier", "ece", "log_loss", "roc_auc", "pr_auc", "latency_ms_per_sample",
            "accuracy", "precision", "recall", "f1"]
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
# Models
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgb_v3(seed: int) -> Any:
    """Enhanced LightGBM for V3 features."""
    if not HAS_LGB:
        raise RuntimeError("lightgbm unavailable")
    return lgb.LGBMClassifier(
        n_estimators=800,          # More trees for richer features
        learning_rate=0.04,        # Slightly slower learning
        num_leaves=255,            # More leaves for 45 features
        max_depth=10,              # Deeper trees
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.85,     # Use more features
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _make_lgb_dart_v3(seed: int) -> Any:
    """LightGBM DART with higher capacity for V3."""
    if not HAS_LGB:
        raise RuntimeError("lightgbm unavailable")
    return lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        boosting_type="dart",
        num_leaves=511,
        max_depth=12,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=1.5,
        drop_rate=0.12,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _make_xgb_v3(seed: int, pos_weight: float = 1.0) -> Any:
    """Enhanced XGBoost for V3 features."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost unavailable")
    return XGBClassifier(
        n_estimators=700,
        max_depth=9,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        device="cpu",
    )


def _eval_model(
    *,
    name: str,
    kind: str,
    factory: Callable[[int], Any],
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
    pos_weight_aware: bool = False,
    calibrate: bool = False,
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

        pw = float((yt == 0).sum() / max(1, (yt == 1).sum()))
        if pos_weight_aware:
            model = factory(seed + i)
            if hasattr(model, "set_params"):
                try:
                    model.set_params(scale_pos_weight=pw)
                except Exception:
                    pass
        else:
            model = factory(seed + i)

        t0 = time.perf_counter()
        model.fit(x[tri], yt)

        if calibrate and HAS_SKLEARN:
            # Apply Platt scaling calibration
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated.fit(x[tri], yt)
            p = calibrated.predict_proba(x[ti])[:, 1]
        else:
            p = model.predict_proba(x[ti])[:, 1]

        lat = (time.perf_counter() - t0) * 1000 / max(1, ti.size)
        row.update(_metrics(yv, p, lat))
        fms.append(row)

    return ModelResult(name, kind, fms, _summarize(fms))


def _eval_ensemble_v3(
    x: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
    calibrate: bool = False,
) -> ModelResult:
    """Weighted ensemble of LGB + XGB + DART."""
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

        t0 = time.perf_counter()

        # Train all models
        lgb_model = _make_lgb_v3(seed + i)
        lgb_model.fit(x[tri], yt)
        p_lgb = lgb_model.predict_proba(x[ti])[:, 1]

        xgb_model = _make_xgb_v3(seed + i)
        xgb_model.fit(x[tri], yt)
        p_xgb = xgb_model.predict_proba(x[ti])[:, 1]

        # Weighted average: LGB 0.5, XGB 0.3, DART 0.2
        if HAS_LGB:
            dart_model = _make_lgb_dart_v3(seed + i)
            dart_model.fit(x[tri], yt)
            p_dart = dart_model.predict_proba(x[ti])[:, 1]
            p = 0.5 * p_lgb + 0.3 * p_xgb + 0.2 * p_dart
        else:
            p = 0.6 * p_lgb + 0.4 * p_xgb

        lat = (time.perf_counter() - t0) * 1000 / max(1, ti.size)
        row.update(_metrics(yv, p, lat))
        fms.append(row)

    return ModelResult("ensemble_v3", "ensemble", fms, _summarize(fms))


def _score(r: ModelResult) -> tuple[float, float]:
    """Score for sorting: lower is better."""
    auc = r.summary.get("roc_auc_mean", 0.0)
    brier = r.summary.get("brier_mean", 1.0)
    if math.isnan(auc):
        auc = 0.0
    if math.isnan(brier):
        brier = 1.0
    return (-auc, brier)


def _retrain_final(
    best: ModelResult,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    calibrate: bool = False,
) -> tuple[Any, str, Any]:
    """Retrain best model on full dataset."""
    if best.model_kind == "lightgbm" and HAS_LGB:
        if "dart" in best.name:
            model = _make_lgb_dart_v3(seed)
        else:
            model = _make_lgb_v3(seed)
        model.fit(x, y)
        return model, "model.pkl", None
    elif best.model_kind == "xgboost" and HAS_XGBOOST:
        model = _make_xgb_v3(seed)
        model.fit(x, y)
        return model, "model.json", None
    elif best.model_kind == "ensemble":
        # For ensemble, save the primary LGB model
        model = _make_lgb_v3(seed)
        model.fit(x, y)
        return model, "model.pkl", None
    return None, "none", None


# ─────────────────────────────────────────────────────────────────────────────
# Artifact export
# ─────────────────────────────────────────────────────────────────────────────

def _write_artifact(
    run_dir: Path,
    best: ModelResult,
    model_obj: Any,
    model_filename: str,
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
        else:
            with model_file_path.open("wb") as fh:
                pickle.dump(model_obj, fh)

    summaries = []
    for r in all_results:
        row = {"name": r.name, "model_kind": r.model_kind}
        row.update(r.summary)
        summaries.append(row)

    payload: dict[str, Any] = {
        "artifact_version": "phase2_forecast_artifact_v3",
        "generated_utc": now_utc_iso(),
        "target_column": target_col,
        "model": {
            "name": best.name,
            "kind": best.model_kind,
            "model_file": model_filename if model_file_path else None,
            "trusted_local_only": True,
        },
        "feature_contract": {
            "version": "v3",
            "feature_count": len(FEATURE_NAMES_V3),
            "feature_names": FEATURE_NAMES_V3,
            "source": "models.forecast.feature_builder_v3",
            "improvements": [
                "cross_rsu_spatial_features",
                "enhanced_time_encoding",
                "advanced_rolling_windows",
                "velocity_acceleration_ratio",
            ],
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    processed_paths = sorted(Path(p) for p in glob.glob(args.processed_glob))
    if not processed_paths:
        print("[P2-V3] No files matched glob.")
        return 2

    print(f"[P2-V3] Loading {len(processed_paths)} files with V3 features (45 dims)...")
    x, y, kept_rows = _load_dataset(processed_paths, args.target_column, args.max_rows)
    if x.shape[0] == 0:
        print("[P2-V3] Dataset empty.")
        return 2

    print(f"[P2-V3] Dataset: {x.shape[0]} rows, {x.shape[1]} features, "
          f"positive_rate={float(y.mean()):.4f}")

    splits = _build_splits(y, args.n_splits, args.test_size, args.gap, args.min_train_size)
    if not splits:
        print("[P2-V3] Could not build splits.")
        return 2
    print(f"[P2-V3] CV splits: {len(splits)}")

    results: list[ModelResult] = []

    # LightGBM V3
    if HAS_LGB:
        print("[P2-V3] Evaluating lightgbm_v3...")
        results.append(_eval_model(
            name="lightgbm_v3", kind="lightgbm",
            factory=_make_lgb_v3, x=x, y=y, splits=splits, seed=args.seed,
            calibrate=args.calibrate,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"accuracy={results[-1].summary.get('accuracy_mean', 'nan'):.4f}  "
              f"f1={results[-1].summary.get('f1_mean', 'nan'):.4f}")

    # XGBoost V3
    if HAS_XGBOOST:
        print("[P2-V3] Evaluating xgboost_v3...")
        results.append(_eval_model(
            name="xgboost_v3", kind="xgboost",
            factory=_make_xgb_v3, x=x, y=y, splits=splits, seed=args.seed,
            pos_weight_aware=True, calibrate=args.calibrate,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"accuracy={results[-1].summary.get('accuracy_mean', 'nan'):.4f}  "
              f"f1={results[-1].summary.get('f1_mean', 'nan'):.4f}")

    # LightGBM DART V3
    if HAS_LGB:
        print("[P2-V3] Evaluating lightgbm_dart_v3 (slower)...")
        results.append(_eval_model(
            name="lightgbm_dart_v3", kind="lightgbm",
            factory=_make_lgb_dart_v3, x=x, y=y, splits=splits, seed=args.seed,
            calibrate=args.calibrate,
        ))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"accuracy={results[-1].summary.get('accuracy_mean', 'nan'):.4f}  "
              f"f1={results[-1].summary.get('f1_mean', 'nan'):.4f}")

    # Ensemble V3
    if HAS_LGB and HAS_XGBOOST:
        print("[P2-V3] Evaluating ensemble_v3...")
        results.append(_eval_ensemble_v3(x, y, splits, args.seed, args.calibrate))
        print(f"  roc_auc={results[-1].summary.get('roc_auc_mean', 'nan'):.4f}  "
              f"accuracy={results[-1].summary.get('accuracy_mean', 'nan'):.4f}  "
              f"f1={results[-1].summary.get('f1_mean', 'nan'):.4f}")

    best = sorted(results, key=_score)[0]
    print(f"\n[P2-V3] Best model: {best.name}")
    print(f"  ROC-AUC: {best.summary.get('roc_auc_mean', 'nan'):.4f} ± {best.summary.get('roc_auc_std', 'nan'):.4f}")
    print(f"  Accuracy: {best.summary.get('accuracy_mean', 'nan'):.4f} ± {best.summary.get('accuracy_std', 'nan'):.4f}")
    print(f"  F1 Score: {best.summary.get('f1_mean', 'nan'):.4f} ± {best.summary.get('f1_std', 'nan'):.4f}")
    print(f"  Brier: {best.summary.get('brier_mean', 'nan'):.4f}")

    print(f"\n[P2-V3] Retraining {best.name} on full dataset...")
    model_obj, model_filename, _ = _retrain_final(best, x, y, args.seed, args.calibrate)

    run_id = f"phase2_v3_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = ensure_dir(Path(args.output_dir) / run_id)

    dataset_meta = {
        "file_count": len(processed_paths),
        "rows_used": int(x.shape[0]),
        "row_count": int(x.shape[0]),
        "positive_rate": float(y.mean()),
        "feature_version": "v3",
        "feature_count": len(FEATURE_NAMES_V3),
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
        all_results=results,
        dataset_meta=dataset_meta,
        cv_cfg=cv_cfg,
        target_col=args.target_column,
    )

    # Update latest symlink
    latest_dir = Path(args.output_dir) / "latest"
    if latest_dir.is_symlink():
        latest_dir.unlink()
    elif latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.symlink_to(run_dir.name)

    print(f"\n[P2-V3] Artifact saved: {artifact_path}")
    print(f"[P2-V3] Latest symlink updated: {latest_dir}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("PHASE 2 V3 FORECAST TRAINING COMPLETE")
    print("=" * 70)
    print(f"Feature set: V3 ({len(FEATURE_NAMES_V3)} features)")
    print("Improvements over V2:")
    print("  - Cross-RSU spatial features (neighbor congestion)")
    print("  - Enhanced time encoding (hour sin/cos, peak hour)")
    print("  - Advanced rolling windows (median, range, EMA)")
    print("  - Velocity/acceleration ratio")
    print(f"\nBest model: {best.name}")
    print(f"  ROC-AUC: {best.summary.get('roc_auc_mean', 0):.4f}")
    print(f"  Accuracy: {best.summary.get('accuracy_mean', 0):.4f}")
    print(f"  F1: {best.summary.get('f1_mean', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
