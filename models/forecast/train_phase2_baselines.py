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
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score  # type: ignore

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    HistGradientBoostingClassifier = None  # type: ignore
    LogisticRegression = None  # type: ignore

    def average_precision_score(*args, **kwargs):  # type: ignore
        raise RuntimeError("sklearn unavailable")

    def brier_score_loss(*args, **kwargs):  # type: ignore
        raise RuntimeError("sklearn unavailable")

    def log_loss(*args, **kwargs):  # type: ignore
        raise RuntimeError("sklearn unavailable")

    def roc_auc_score(*args, **kwargs):  # type: ignore
        raise RuntimeError("sklearn unavailable")

from models.forecast.common import (
    clamp01,
    compute_expected_calibration_error,
    ensure_dir,
    now_utc_iso,
    rolling_expanding_splits,
    safe_mean,
)
from models.forecast.feature_builder import FEATURE_NAMES, build_training_features_from_row

try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore


@dataclass
class BaselineResult:
    name: str
    model_kind: str
    fold_metrics: list[dict[str, Any]]
    summary: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 baseline ladder trainer with rolling-window CV and uncertainty metrics."
    )
    parser.add_argument(
        "--processed-glob",
        default="data/processed/smoke_phase1_logger*/rsu_horizon_labels.csv",
        help="Glob for processed horizon-labeled CSV files.",
    )
    parser.add_argument(
        "--target-column",
        default="label_congestion_60s",
        help="Binary target column to train/evaluate.",
    )
    parser.add_argument(
        "--profile-config",
        default="experiments/training_profiles.json",
        help="Training profile config path.",
    )
    parser.add_argument(
        "--profile",
        default="local_smoke",
        help="Profile key from profile config.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max row cap after sorting.")
    parser.add_argument("--n-splits", type=int, default=5, help="Rolling CV split count.")
    parser.add_argument("--test-size", type=int, default=360, help="Rolling CV test window size.")
    parser.add_argument("--gap", type=int, default=0, help="Gap rows between train and test windows.")
    parser.add_argument("--min-train-size", type=int, default=1200, help="Minimum rows in first train window.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        default="models/forecast/artifacts",
        help="Output root for model artifacts and reports.",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/phase2_forecast_report.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def _load_profile_settings(profile_config: Path, profile_name: str) -> dict[str, Any]:
    if not profile_config.exists():
        return {}
    try:
        config = json.loads(profile_config.read_text(encoding="utf-8"))
    except Exception:
        return {}

    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(profile_name)
    return profile if isinstance(profile, dict) else {}


def _iter_rows_from_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _load_dataset(paths: list[Path], target_column: str, max_rows: int | None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        run_id = path.parent.name
        for row in _iter_rows_from_csv(path):
            if target_column not in row:
                continue
            merged = dict(row)
            merged["run_id"] = run_id
            all_rows.append(merged)

    def _sort_key(row: dict[str, Any]) -> tuple[str, float, int, str]:
        run_id = str(row.get("run_id", ""))
        rsu_node = str(row.get("rsu_node", ""))
        try:
            timestamp = float(row.get("timestamp_s", 0.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        try:
            frame_idx = int(float(row.get("frame_idx", 0)))
        except (TypeError, ValueError):
            frame_idx = 0
        return (run_id, timestamp, frame_idx, rsu_node)

    all_rows.sort(key=_sort_key)
    if max_rows is not None and max_rows > 0 and len(all_rows) > max_rows:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in all_rows:
            grouped.setdefault(str(row.get("run_id", "unknown")), []).append(row)

        run_ids = sorted(grouped.keys())
        per_run = max_rows // max(1, len(run_ids))
        remainder = max_rows % max(1, len(run_ids))

        limited_rows: list[dict[str, Any]] = []
        for idx, run_id in enumerate(run_ids):
            take = per_run + (1 if idx < remainder else 0)
            limited_rows.extend(grouped[run_id][:take])

        limited_rows.sort(key=_sort_key)
        all_rows = limited_rows

    state_by_stream: dict[tuple[str, str], Any] = {}
    x_rows: list[np.ndarray] = []
    y_values: list[int] = []
    kept_rows: list[dict[str, Any]] = []

    for row in all_rows:
        try:
            y = int(float(row.get(target_column, 0)))
        except (TypeError, ValueError):
            continue
        y = 1 if y >= 1 else 0

        features = build_training_features_from_row(row, state_by_stream)
        x_rows.append(features)
        y_values.append(y)
        kept_rows.append(row)

    if not x_rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float), np.zeros((0,), dtype=int), []

    x = np.vstack(x_rows).astype(float)
    y = np.array(y_values, dtype=int)
    return x, y, kept_rows


def _persistence_probability(x: np.ndarray) -> np.ndarray:
    idx_count = FEATURE_NAMES.index("connected_vehicle_count")
    idx_local = FEATURE_NAMES.index("congested_local")

    count_term = np.clip(x[:, idx_count] / 25.0, 0.0, 1.0)
    local_term = np.clip(x[:, idx_local], 0.0, 1.0)
    probs = 0.10 + 0.55 * local_term + 0.35 * count_term
    return np.clip(probs, 0.0, 1.0)


def _fit_xgboost(seed: int) -> Any:
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost unavailable")
    if XGBClassifier is None:
        raise RuntimeError("xgboost unavailable")
    return XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def _fit_histgb(seed: int) -> Any:
    if not HAS_SKLEARN or HistGradientBoostingClassifier is None:
        raise RuntimeError("sklearn unavailable")
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=seed,
    )


def _fit_spatiotemporal_proxy(seed: int) -> Any:
    if not HAS_SKLEARN or LogisticRegression is None:
        raise RuntimeError("sklearn unavailable")
    return LogisticRegression(
        random_state=seed,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
    )


def _compute_fold_metrics(y_true: np.ndarray, p_pred: np.ndarray, latency_ms_per_sample: float) -> dict[str, float]:
    p_safe = np.clip(p_pred.astype(float), 1e-6, 1.0 - 1e-6)
    y_true = y_true.astype(int)

    metrics: dict[str, float] = {
        "brier": float(brier_score_loss(y_true, p_safe)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - p_safe, p_safe]), labels=[0, 1])),
        "ece": float(compute_expected_calibration_error(y_true, p_safe, n_bins=10)),
        "latency_ms_per_sample": float(latency_ms_per_sample),
        "positive_rate": float(y_true.mean()) if y_true.size else float("nan"),
    }

    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_safe))
    else:
        metrics["roc_auc"] = float("nan")

    if int(y_true.sum()) > 0:
        metrics["pr_auc"] = float(average_precision_score(y_true, p_safe))
    else:
        metrics["pr_auc"] = float("nan")

    return metrics


def _window_has_both_classes(y: np.ndarray, start: int, test_size: int) -> bool:
    if start < 0:
        return False
    end = start + test_size
    if end > y.shape[0]:
        return False
    window = y[start:end]
    if window.size <= 1:
        return False
    return int(window.min()) != int(window.max())


def _find_label_aware_start(
    y: np.ndarray,
    *,
    preferred_start: int,
    low_start: int,
    high_start: int,
    test_size: int,
) -> int | None:
    if low_start > high_start:
        return None

    candidates = list(range(low_start, high_start + 1))
    candidates.sort(key=lambda s: (abs(s - preferred_start), s))

    for start in candidates:
        if _window_has_both_classes(y, start, test_size):
            return start
    return None


def _rolling_label_aware_splits(
    *,
    y: np.ndarray,
    n_splits: int,
    test_size: int,
    gap: int,
    min_train_size: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    """Build expanding-window splits and adjust test windows to include both classes.

    Returns (splits, requested_folds, dropped_folds).
    """
    base_splits = rolling_expanding_splits(
        n_samples=y.shape[0],
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        min_train_size=min_train_size,
    )
    if not base_splits:
        return [], 0, 0

    requested_folds = len(base_splits)
    preferred_starts = [int(test_idx[0]) for _train_idx, test_idx in base_splits]

    max_start = y.shape[0] - test_size
    low_global = min_train_size + gap

    selected_reverse: list[int] = []
    next_start_bound = y.shape[0]

    for preferred_start in reversed(preferred_starts):
        high_start = min(max_start, next_start_bound - test_size)
        low_start = max(0, low_global)

        chosen_start = _find_label_aware_start(
            y,
            preferred_start=preferred_start,
            low_start=low_start,
            high_start=high_start,
            test_size=test_size,
        )
        if chosen_start is None:
            continue

        selected_reverse.append(chosen_start)
        next_start_bound = chosen_start

    selected_starts = list(reversed(selected_reverse))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for start in selected_starts:
        train_end = start - gap
        if train_end < min_train_size:
            continue

        test_end = start + test_size
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(start, test_end, dtype=int)
        if train_idx.size < min_train_size:
            continue
        if not _window_has_both_classes(y, int(test_idx[0]), int(test_idx.size)):
            continue
        splits.append((train_idx, test_idx))

    dropped_folds = max(0, requested_folds - len(splits))
    return splits, requested_folds, dropped_folds


def _evaluate_learned_baseline(
    *,
    name: str,
    model_kind: str,
    estimator_factory: Callable[[int], Any],
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> BaselineResult:
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]

        row: dict[str, Any] = {
            "fold": fold_idx,
            "train_size": int(train_idx.size),
            "test_size": int(test_idx.size),
            "train_positive_rate": float(y_train.mean()) if y_train.size else float("nan"),
            "test_positive_rate": float(y_test.mean()) if y_test.size else float("nan"),
            "status": "ok",
        }

        if len(np.unique(y_train)) < 2:
            const_prob = float(y_train.mean()) if y_train.size else 0.5
            p_pred = np.full(test_idx.size, clamp01(const_prob), dtype=float)
            row["status"] = "train_single_class_fallback"
            row["model_note"] = "constant_probability"
            metrics = _compute_fold_metrics(y_test, p_pred, latency_ms_per_sample=0.0)
            row.update(metrics)
            fold_metrics.append(row)
            continue

        estimator = estimator_factory(seed + fold_idx)
        start = time.perf_counter()
        estimator.fit(x[train_idx], y_train)
        if hasattr(estimator, "predict_proba"):
            p_pred = estimator.predict_proba(x[test_idx])[:, 1]
        else:
            scores = estimator.decision_function(x[test_idx])
            p_pred = 1.0 / (1.0 + np.exp(-scores))
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed * 1000.0) / max(1, test_idx.size)

        metrics = _compute_fold_metrics(y_test, p_pred, latency_ms_per_sample=latency_ms)
        row.update(metrics)
        fold_metrics.append(row)

    summary = _summarize_fold_metrics(fold_metrics)
    return BaselineResult(name=name, model_kind=model_kind, fold_metrics=fold_metrics, summary=summary)


def _evaluate_persistence_baseline(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> BaselineResult:
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (_, test_idx) in enumerate(splits, start=1):
        y_test = y[test_idx]
        start = time.perf_counter()
        p_pred = _persistence_probability(x[test_idx])
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed * 1000.0) / max(1, test_idx.size)

        row: dict[str, Any] = {
            "fold": fold_idx,
            "train_size": None,
            "test_size": int(test_idx.size),
            "status": "ok",
            "model_note": "rule_based_persistence",
        }
        row.update(_compute_fold_metrics(y_test, p_pred, latency_ms_per_sample=latency_ms))
        fold_metrics.append(row)

    summary = _summarize_fold_metrics(fold_metrics)
    return BaselineResult(
        name="persistence_v1",
        model_kind="rule",
        fold_metrics=fold_metrics,
        summary=summary,
    )


def _summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["brier", "ece", "log_loss", "roc_auc", "pr_auc", "latency_ms_per_sample"]
    summary: dict[str, Any] = {
        "fold_count": len(fold_metrics),
        "status_counts": {},
    }

    status_counts: dict[str, int] = {}
    for row in fold_metrics:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary["status_counts"] = status_counts

    for metric in metric_names:
        values: list[float] = []
        for row in fold_metrics:
            value = row.get(metric, float("nan"))
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                fvalue = float("nan")
            values.append(fvalue)
        summary[f"{metric}_mean"] = safe_mean(values)

    return summary


def _select_best_baseline(results: list[BaselineResult]) -> BaselineResult:
    def score(item: BaselineResult) -> tuple[float, float, float]:
        brier = float(item.summary.get("brier_mean", float("nan")))
        ece = float(item.summary.get("ece_mean", float("nan")))
        roc = float(item.summary.get("roc_auc_mean", float("nan")))

        if math.isnan(brier):
            brier = 1e6
        if math.isnan(ece):
            ece = 1e6
        if math.isnan(roc):
            roc = -1e6

        return (brier, ece, -roc)

    return sorted(results, key=score)[0]


def _train_final_model(best: BaselineResult, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, str]:
    if best.name == "persistence_v1":
        return None, "none"

    if best.name == "xgboost_binary_classifier_v1":
        estimator = _fit_xgboost(seed)
    elif best.name == "hist_gradient_boosting_v1":
        estimator = _fit_histgb(seed)
    else:
        estimator = _fit_spatiotemporal_proxy(seed)

    if len(np.unique(y)) < 2:
        return None, "none"

    estimator.fit(x, y)
    if best.name == "xgboost_binary_classifier_v1":
        return estimator, "model.json"
    return estimator, "model.pkl"


def _write_phase2_report(report_path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Phase 2 Forecast Report")
    lines.append("")
    lines.append(f"- Generated UTC: {payload['generated_utc']}")
    lines.append(f"- Target: {payload['target_column']}")
    lines.append(f"- Processed files: {payload['dataset']['file_count']}")
    lines.append(f"- Total rows: {payload['dataset']['row_count']}")
    lines.append(f"- Positive rate: {payload['dataset']['positive_rate']:.6f}")
    lines.append(f"- Rolling folds: {payload['cv_config']['n_splits_actual']}")
    lines.append(f"- Best baseline: {payload['selected_model']['name']}")
    lines.append("")
    lines.append("## Baseline Summary")
    lines.append("")
    lines.append("| baseline | brier | ece | roc_auc | pr_auc | latency_ms/sample |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload["baseline_summaries"]:
        lines.append(
            "| {name} | {brier:.6f} | {ece:.6f} | {roc} | {pr} | {latency:.6f} |".format(
                name=row["name"],
                brier=float(row.get("brier_mean", float("nan"))),
                ece=float(row.get("ece_mean", float("nan"))),
                roc=(
                    f"{float(row['roc_auc_mean']):.6f}"
                    if not math.isnan(float(row.get("roc_auc_mean", float("nan"))))
                    else "nan"
                ),
                pr=(
                    f"{float(row['pr_auc_mean']):.6f}"
                    if not math.isnan(float(row.get("pr_auc_mean", float("nan"))))
                    else "nan"
                ),
                latency=float(row.get("latency_ms_per_sample_mean", float("nan"))),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Evaluation used rolling expanding windows with chronological order preserved.")
    lines.append("- Some folds can be single-class under heavy temporal drift; trainer uses fallback probabilities instead of failing.")
    lines.append("- Artifact is intended for trusted local loading only.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()

    profile = _load_profile_settings(Path(args.profile_config), args.profile)
    max_rows = args.max_rows
    if max_rows is None:
        prof_max_rows = profile.get("max_rows")
        if isinstance(prof_max_rows, int) and prof_max_rows > 0:
            max_rows = prof_max_rows

    processed_paths = sorted(Path(p) for p in glob.glob(args.processed_glob))
    if not processed_paths:
        print("[PHASE2] No files matched processed-glob.")
        return 2

    x, y, kept_rows = _load_dataset(processed_paths, args.target_column, max_rows)
    if x.shape[0] == 0:
        print("[PHASE2] Dataset empty after filtering.")
        return 2

    splits, requested_folds, dropped_folds = _rolling_label_aware_splits(
        y=y,
        n_splits=args.n_splits,
        test_size=args.test_size,
        gap=args.gap,
        min_train_size=args.min_train_size,
    )
    if not splits:
        print("[PHASE2] Unable to create rolling splits with current settings.")
        return 2
    if dropped_folds > 0:
        print(
            f"[PHASE2] Label-aware CV dropped {dropped_folds} fold(s) without both classes in test windows."
        )

    results: list[BaselineResult] = []
    results.append(_evaluate_persistence_baseline(x, y, splits))

    if HAS_XGBOOST:
        results.append(
            _evaluate_learned_baseline(
                name="xgboost_binary_classifier_v1",
                model_kind="xgboost",
                estimator_factory=_fit_xgboost,
                x=x,
                y=y,
                splits=splits,
                seed=args.seed,
            )
        )
    else:
        print("[PHASE2] xgboost unavailable; skipping boosted baseline.")

    if HAS_SKLEARN:
        results.append(
            _evaluate_learned_baseline(
                name="hist_gradient_boosting_v1",
                model_kind="sklearn",
                estimator_factory=_fit_histgb,
                x=x,
                y=y,
                splits=splits,
                seed=args.seed,
            )
        )
        results.append(
            _evaluate_learned_baseline(
                name="spatiotemporal_proxy_v1",
                model_kind="sklearn",
                estimator_factory=_fit_spatiotemporal_proxy,
                x=x,
                y=y,
                splits=splits,
                seed=args.seed,
            )
        )
    else:
        print("[PHASE2] sklearn unavailable; skipping sklearn baselines.")

    best = _select_best_baseline(results)
    fitted_model, model_filename = _train_final_model(best, x, y, args.seed)

    output_root = ensure_dir(Path(args.output_dir))
    run_dir = ensure_dir(output_root / f"phase2_forecast_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")

    model_file_path = None
    if fitted_model is not None and model_filename != "none":
        model_file_path = run_dir / model_filename
        if model_filename.endswith(".json"):
            fitted_model.save_model(str(model_file_path))
        else:
            with model_file_path.open("wb") as handle:
                pickle.dump(fitted_model, handle)

    baselines_payload = []
    for result in results:
        row = {
            "name": result.name,
            "model_kind": result.model_kind,
        }
        row.update(result.summary)
        baselines_payload.append(row)

    run_ids = sorted({str(row.get("run_id", "unknown")) for row in kept_rows})
    artifact_payload: dict[str, Any] = {
        "artifact_version": "phase2_forecast_artifact_v1",
        "generated_utc": now_utc_iso(),
        "target_column": args.target_column,
        "model": {
            "name": best.name,
            "kind": best.model_kind,
            "model_file": model_filename if model_file_path is not None else None,
            "trusted_local_only": True,
        },
        "feature_contract": {
            "feature_names": FEATURE_NAMES,
            "source": "models.forecast.feature_builder",
        },
        "dataset": {
            "file_count": len(processed_paths),
            "rows_used": int(x.shape[0]),
            "positive_rate": float(y.mean()),
            "run_ids": run_ids,
            "processed_files": [str(path) for path in processed_paths],
        },
        "cv_config": {
            "n_splits_requested": int(args.n_splits),
            "n_splits_constructed": int(requested_folds),
            "n_splits_actual": int(len(splits)),
            "n_splits_dropped_single_class_test": int(dropped_folds),
            "test_size": int(args.test_size),
            "gap": int(args.gap),
            "min_train_size": int(args.min_train_size),
        },
        "baseline_summaries": baselines_payload,
        "selected_model": {
            "name": best.name,
            "summary": best.summary,
        },
        "fold_details": {
            result.name: result.fold_metrics for result in results
        },
        "inference_output_contract": {
            "p_congestion": "float[0,1]",
            "confidence": "float[0,1]",
            "uncertainty": "float[0,1]",
        },
    }

    artifact_path = run_dir / "forecast_artifact.json"
    artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    latest_dir = ensure_dir(output_root / "latest")
    for stale_model in latest_dir.glob("model.*"):
        try:
            stale_model.unlink()
        except OSError:
            pass
    shutil.copy2(artifact_path, latest_dir / "forecast_artifact.json")
    if model_file_path is not None:
        shutil.copy2(model_file_path, latest_dir / model_filename)

    report_payload = {
        "generated_utc": artifact_payload["generated_utc"],
        "target_column": args.target_column,
        "dataset": {
            "file_count": len(processed_paths),
            "row_count": int(x.shape[0]),
            "positive_rate": float(y.mean()),
        },
        "cv_config": artifact_payload["cv_config"],
        "selected_model": artifact_payload["selected_model"],
        "baseline_summaries": baselines_payload,
    }
    _write_phase2_report(Path(args.report_path), report_payload)

    print("[PHASE2] Training complete.")
    print(f"[PHASE2] Best baseline: {best.name}")
    print(f"[PHASE2] Artifact: {artifact_path}")
    print(f"[PHASE2] Latest link dir: {latest_dir}")
    print(f"[PHASE2] Report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
