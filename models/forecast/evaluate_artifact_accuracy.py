from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from models.forecast.inference import ForecastInferenceEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate latest forecast artifact on held-out split CSV files and report classification metrics."
        )
    )
    parser.add_argument(
        "--artifact",
        default="models/forecast/artifacts/latest/forecast_artifact.json",
        help="Path to forecast artifact JSON.",
    )
    parser.add_argument(
        "--split-glob",
        default="data/splits/phase2sweep_*/test.csv",
        help="Glob for split CSV files.",
    )
    parser.add_argument(
        "--run-id-list",
        default=None,
        help=(
            "Optional text file with one run_id per line (filters splits to data/splits/<run_id>/test.csv)."
        ),
    )
    parser.add_argument(
        "--target-column",
        default="label_congestion_60s",
        help="Binary target column in split CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for hard classification metrics.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional global row cap for quick checks.",
    )
    return parser.parse_args()


def _row_to_payload(row: dict[str, str]) -> dict:
    try:
        vehicle_count = int(float(row.get("connected_vehicle_count", 0)))
    except (TypeError, ValueError):
        vehicle_count = 0

    try:
        timestamp_s = float(row.get("timestamp_s", row.get("frame_idx", 0)))
    except (TypeError, ValueError):
        timestamp_s = 0.0

    try:
        avg_latency_s = float(row.get("avg_latency_s", 0.0))
    except (TypeError, ValueError):
        avg_latency_s = 0.0

    return {
        "rsu_id": str(row.get("rsu_node", "RSU_UNKNOWN")),
        "timestamp": timestamp_s,
        "vehicle_count": vehicle_count,
        "vehicle_ids": [],
        "emergency_vehicle_ids": [],
        "avg_speed_mps": max(0.0, min(15.0, 15.0 - avg_latency_s * 10.0)),
        "features": {
            "registered_telemetry_count": float(
                row.get("registered_telemetry_count", vehicle_count) or vehicle_count
            ),
            "packets_received": float(row.get("packets_received", vehicle_count) or vehicle_count),
            "bytes_received": float(row.get("bytes_received", vehicle_count * 128) or vehicle_count * 128),
            "avg_latency_s": avg_latency_s,
            "congested_local": float(row.get("congested_local", 0) or 0),
            "congested_global": float(row.get("congested_global", 0) or 0),
        },
    }


def _resolve_split_paths(split_glob: str, run_id_list_path: str | None) -> tuple[list[Path], int]:
    if run_id_list_path is None:
        return sorted(Path(p) for p in glob.glob(split_glob)), 0

    run_ids_path = Path(run_id_list_path)
    if not run_ids_path.exists():
        raise FileNotFoundError(f"run-id list not found: {run_ids_path}")

    run_ids = [line.strip() for line in run_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths: list[Path] = []
    missing = 0
    for run_id in run_ids:
        path = Path("data/splits") / run_id / "test.csv"
        if path.exists():
            paths.append(path)
        else:
            missing += 1
    return paths, missing


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return num / den


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # Tie correction via average ranks.
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = float((i + 1 + j) / 2.0)
            ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[y_true == 1].sum())
    u_stat = sum_pos_ranks - (pos * (pos + 1) / 2.0)
    return float(u_stat / (pos * neg))


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = int((y_true == 1).sum())
    if pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / pos

    ap = 0.0
    prev_recall = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            recall_i = float(recall[i])
            ap += (recall_i - prev_recall) * float(precision[i])
            prev_recall = recall_i
    return float(ap)


def main() -> int:
    args = _parse_args()

    artifact = Path(args.artifact)
    if not artifact.exists():
        print(f"[PHASE2][EVAL] Artifact not found: {artifact}")
        return 2

    if args.threshold < 0.0 or args.threshold > 1.0:
        print("[PHASE2][EVAL] threshold must be in [0, 1]")
        return 2

    split_paths, missing_runs = _resolve_split_paths(args.split_glob, args.run_id_list)
    if not split_paths:
        print("[PHASE2][EVAL] No split files matched.")
        return 2

    engine = ForecastInferenceEngine.from_artifact_path(artifact)

    y_true: list[int] = []
    y_prob: list[float] = []
    rows = 0

    for split_path in split_paths:
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if args.max_rows is not None and args.max_rows > 0 and rows >= args.max_rows:
                    break

                raw_target = row.get(args.target_column, "0")
                try:
                    label = int(float(raw_target))
                except (TypeError, ValueError):
                    continue
                label = 1 if label >= 1 else 0

                payload = _row_to_payload(row)
                pred = engine.predict_from_route_payload(payload)

                try:
                    p = float(pred.get("p_congestion", 0.0))
                except (TypeError, ValueError):
                    p = 0.0
                p = max(0.0, min(1.0, p))

                y_true.append(label)
                y_prob.append(p)
                rows += 1

            if args.max_rows is not None and args.max_rows > 0 and rows >= args.max_rows:
                break

    if rows == 0:
        print("[PHASE2][EVAL] No valid labeled rows found.")
        return 2

    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= args.threshold).astype(int)

    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())

    accuracy = float((y_pred_arr == y_true_arr).mean())
    recall_pos = _safe_div(float(tp), float(tp + fn))
    recall_neg = _safe_div(float(tn), float(tn + fp))
    balanced_accuracy = (
        (recall_pos + recall_neg) / 2.0
        if (not math.isnan(recall_pos) and not math.isnan(recall_neg))
        else float("nan")
    )
    precision = _safe_div(float(tp), float(tp + fp))
    recall = recall_pos
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0.0)
        else float("nan")
    )

    roc_auc = _binary_roc_auc(y_true_arr, y_prob_arr)
    pr_auc = _binary_average_precision(y_true_arr, y_prob_arr)

    print("[PHASE2][EVAL] Artifact evaluation complete")
    print(f"[PHASE2][EVAL] model={engine.model_name}")
    print(f"[PHASE2][EVAL] split_files_used={len(split_paths)}")
    if args.run_id_list is not None:
        print(f"[PHASE2][EVAL] missing_runs_from_list={missing_runs}")
    print(f"[PHASE2][EVAL] rows={rows}")
    print(f"[PHASE2][EVAL] positive_rate={float(y_true_arr.mean()):.6f}")
    print(f"[PHASE2][EVAL] threshold={args.threshold:.3f}")
    print(f"[PHASE2][EVAL] accuracy={accuracy:.6f}")
    print(f"[PHASE2][EVAL] balanced_accuracy={balanced_accuracy:.6f}")
    print(f"[PHASE2][EVAL] precision={precision:.6f}")
    print(f"[PHASE2][EVAL] recall={recall:.6f}")
    print(f"[PHASE2][EVAL] f1={f1:.6f}")
    print(f"[PHASE2][EVAL] roc_auc={roc_auc:.6f}" if not math.isnan(roc_auc) else "[PHASE2][EVAL] roc_auc=nan")
    print(f"[PHASE2][EVAL] pr_auc={pr_auc:.6f}" if not math.isnan(pr_auc) else "[PHASE2][EVAL] pr_auc=nan")
    print(f"[PHASE2][EVAL] confusion={{'tp': {tp}, 'tn': {tn}, 'fp': {fp}, 'fn': {fn}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
