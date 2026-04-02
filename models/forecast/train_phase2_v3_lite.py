"""Train Phase 2 V3 Lite: 38 features without noisy spatial features.

Usage:
    python models/forecast/train_phase2_v3_lite.py \\
        --processed-glob 'data/processed/*/rsu_horizon_labels.csv' \\
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
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    lgb = None

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
from models.forecast.feature_builder_v3_lite import (
    FEATURE_NAMES_V3_LITE,
    FeatureStateV3Lite,
    build_training_features_from_row_v3_lite,
)


@dataclass
class ModelResult:
    name: str
    model_kind: str
    fold_metrics: list[dict[str, Any]]
    summary: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase 2 V3 Lite forecast model")
    p.add_argument("--processed-glob", default="data/processed/*/rsu_horizon_labels.csv")
    p.add_argument("--output-dir", default="models/forecast/artifacts")
    p.add_argument("--target-column", default="label_congestion_60s")
    p.add_argument("--max-rows", type=int, default=500_000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=int, default=360)
    p.add_argument("--gap", type=int, default=30)
    p.add_argument("--min-train-size", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", default="local")
    return p.parse_args()


def _load_dataset(
    paths: list[Path],
    target_col: str,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
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
        return np.empty((0, len(FEATURE_NAMES_V3_LITE))), np.empty((0,)), []

    all_rows.sort(
        key=lambda r: (
            str(r.get("run_id", "")),
            float(r.get("timestamp_s", r.get("frame_idx", 0))),
        )
    )

    state_by_stream: dict[tuple[str, str], FeatureStateV3Lite] = {}
    features: list[np.ndarray] = []
    labels: list[float] = []
    kept_rows: list[dict[str, Any]] = []

    for row in all_rows:
        try:
            label = float(row.get(target_col, 0))
        except (TypeError, ValueError):
            continue

        feat = build_training_features_from_row_v3_lite(row, state_by_stream)
        features.append(feat)
        labels.append(label)
        kept_rows.append(row)

    x = np.vstack(features) if features else np.empty((0, len(FEATURE_NAMES_V3_LITE)))
    y = np.array(labels, dtype=float)
    return x, y, kept_rows


def _window_ok(y: np.ndarray, start: int, size: int) -> bool:
    if start < 0 or start + size > len(y):
        return False
    w = y[start : start + size]
    return w.size > 1 and int(w.min()) != int(w.max())


def _find_label_aware_start(y, *, preferred, low, high, size):
    if low > high:
        return None
    candidates = sorted(range(low, high + 1), key=lambda s: (abs(s - preferred), s))
    for s in candidates:
        if _window_ok(y, s, size):
            return s
    return None


def _build_splits(y, n_splits, test_size, gap, min_train):
    base = rolling_expanding_splits(len(y), n_splits, test_size, gap, min_train)
    if not base:
        return []

    preferred_starts = [int(ti[0]) for _, ti in base]
    max_start = len(y) - test_size
    low_global = min_train + gap

    selected_rev = []
    next_bound = len(y)
    for pref in reversed(preferred_starts):
        hi = min(max_start, next_bound - test_size)
        lo = max(0, low_global)
        chosen = _find_label_aware_start(y, preferred=pref, low=lo, high=hi, size=test_size)
        if chosen is None:
            continue
        selected_rev.append(chosen)
        next_bound = chosen

    splits = []
    for start in reversed(selected_rev):
        tr_end = start - gap
        if tr_end < min_train:
            continue
        te = np.arange(start, start + test_size, dtype=int)
        if not _window_ok(y, start, test_size):
            continue
        splits.append((np.arange(0, tr_end, dtype=int), te))
    return splits


def _metrics(y_true, p_pred, latency_ms):
    p = np.clip(p_pred.astype(float), 1e-6, 1 - 1e-6)
    y = y_true.astype(int)
    y_pred = (p >= 0.5).astype(int)

    m = {
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


def _summarize(fold_metrics):
    keys = ["brier", "ece", "log_loss", "roc_auc", "pr_auc", "latency_ms_per_sample",
            "accuracy", "precision", "recall", "f1"]
    summary = {"fold_count": len(fold_metrics)}
    for k in keys:
        vals = [float(fm.get(k, float("nan"))) for fm in fold_metrics]
        summary[f"{k}_mean"] = safe_mean(vals)
        finite = [v for v in vals if not math.isnan(v)]
        summary[f"{k}_std"] = float(np.std(finite)) if len(finite) >= 2 else float("nan")
    return summary


def _make_lgb_v3_lite(seed):
    return lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=255,
        max_depth=10,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _make_lgb_dart(seed):
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.035,
        boosting_type="dart",
        num_leaves=255,
        max_depth=10,
        min_child_samples=12,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_alpha=0.25,
        reg_lambda=1.2,
        drop_rate=0.1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _eval_model(*, name, kind, factory, x, y, splits, seed):
    fms = []
    for i, (tri, ti) in enumerate(splits, 1):
        yt, yv = y[tri], y[ti]
        row = {
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

        model = factory(seed + i)
        t0 = time.perf_counter()
        model.fit(x[tri], yt)
        p = model.predict_proba(x[ti])[:, 1]
        lat = (time.perf_counter() - t0) * 1000 / max(1, ti.size)
        row.update(_metrics(yv, p, lat))
        fms.append(row)

    return ModelResult(name, kind, fms, _summarize(fms))


def _score(r):
    auc = r.summary.get("roc_auc_mean", 0.0)
    brier = r.summary.get("brier_mean", 1.0)
    if math.isnan(auc):
        auc = 0.0
    if math.isnan(brier):
        brier = 1.0
    return (-auc, brier)


def _write_artifact(run_dir, best, model_obj, all_results, dataset_meta, cv_cfg, target_col):
    model_file = run_dir / "model.pkl"
    with model_file.open("wb") as fh:
        pickle.dump(model_obj, fh)

    summaries = [{"name": r.name, "model_kind": r.model_kind, **r.summary} for r in all_results]

    payload = {
        "artifact_version": "phase2_forecast_artifact_v3_lite",
        "generated_utc": now_utc_iso(),
        "target_column": target_col,
        "model": {
            "name": best.name,
            "kind": best.model_kind,
            "model_file": "model.pkl",
            "trusted_local_only": True,
        },
        "feature_contract": {
            "version": "v3_lite",
            "feature_count": len(FEATURE_NAMES_V3_LITE),
            "feature_names": FEATURE_NAMES_V3_LITE,
            "source": "models.forecast.feature_builder_v3_lite",
            "improvements": [
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
    }

    artifact_path = run_dir / "forecast_artifact.json"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def main():
    args = _parse_args()

    processed_paths = sorted(Path(p) for p in glob.glob(args.processed_glob))
    if not processed_paths:
        print("[V3-LITE] No files matched glob.")
        return 2

    print(f"[V3-LITE] Loading {len(processed_paths)} files with 38-feature V3 Lite...")
    x, y, kept_rows = _load_dataset(processed_paths, args.target_column, args.max_rows)
    if x.shape[0] == 0:
        print("[V3-LITE] Dataset empty.")
        return 2

    print(f"[V3-LITE] Dataset: {x.shape[0]} rows, {x.shape[1]} features, pos_rate={y.mean():.4f}")

    splits = _build_splits(y, args.n_splits, args.test_size, args.gap, args.min_train_size)
    if not splits:
        print("[V3-LITE] Could not build splits.")
        return 2
    print(f"[V3-LITE] CV splits: {len(splits)}")

    results = []

    if HAS_LGB:
        print("[V3-LITE] Evaluating lightgbm_v3_lite...")
        results.append(_eval_model(
            name="lightgbm_v3_lite", kind="lightgbm",
            factory=_make_lgb_v3_lite, x=x, y=y, splits=splits, seed=args.seed,
        ))
        s = results[-1].summary
        print(f"  roc_auc={s.get('roc_auc_mean', 'nan'):.4f}  "
              f"accuracy={s.get('accuracy_mean', 'nan'):.4f}  f1={s.get('f1_mean', 'nan'):.4f}")

        print("[V3-LITE] Evaluating lightgbm_dart_v3_lite...")
        results.append(_eval_model(
            name="lightgbm_dart_v3_lite", kind="lightgbm",
            factory=_make_lgb_dart, x=x, y=y, splits=splits, seed=args.seed,
        ))
        s = results[-1].summary
        print(f"  roc_auc={s.get('roc_auc_mean', 'nan'):.4f}  "
              f"accuracy={s.get('accuracy_mean', 'nan'):.4f}  f1={s.get('f1_mean', 'nan'):.4f}")

    best = sorted(results, key=_score)[0]
    print(f"\n[V3-LITE] Best: {best.name}")
    print(f"  ROC-AUC: {best.summary.get('roc_auc_mean', 0):.4f} ± {best.summary.get('roc_auc_std', 0):.4f}")
    print(f"  Accuracy: {best.summary.get('accuracy_mean', 0):.4f}")
    print(f"  F1: {best.summary.get('f1_mean', 0):.4f}")

    print(f"\n[V3-LITE] Retraining {best.name} on full dataset...")
    if "dart" in best.name:
        model = _make_lgb_dart(args.seed)
    else:
        model = _make_lgb_v3_lite(args.seed)
    model.fit(x, y)

    run_id = f"phase2_v3_lite_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = ensure_dir(Path(args.output_dir) / run_id)

    dataset_meta = {
        "file_count": len(processed_paths),
        "row_count": int(x.shape[0]),
        "positive_rate": float(y.mean()),
        "feature_version": "v3_lite",
        "feature_count": len(FEATURE_NAMES_V3_LITE),
    }
    cv_cfg = {
        "n_splits_actual": len(splits),
        "test_size": args.test_size,
        "gap": args.gap,
        "min_train_size": args.min_train_size,
    }

    artifact_path = _write_artifact(run_dir, best, model, results, dataset_meta, cv_cfg, args.target_column)

    # Update latest
    latest_dir = Path(args.output_dir) / "latest"
    if latest_dir.is_symlink():
        latest_dir.unlink()
    elif latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.symlink_to(run_dir.name)

    print(f"\n[V3-LITE] Artifact: {artifact_path}")
    print("=" * 60)
    print("V3 LITE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Features: 38 (V2 31 + 7 temporal/rolling)")
    print(f"Best: {best.name}")
    print(f"  ROC-AUC: {best.summary.get('roc_auc_mean', 0):.4f}")
    print(f"  Accuracy: {best.summary.get('accuracy_mean', 0):.4f}")
    print(f"  F1: {best.summary.get('f1_mean', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
