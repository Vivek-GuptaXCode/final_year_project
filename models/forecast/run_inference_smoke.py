from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from models.forecast.common import clamp01
from models.forecast.inference import ForecastInferenceEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for Phase 2 inference artifact.")
    parser.add_argument(
        "--artifact",
        default="models/forecast/artifacts/latest/forecast_artifact.json",
        help="Path to forecast artifact JSON.",
    )
    parser.add_argument(
        "--input-csv",
        default="data/splits/smoke_phase1_logger_seed47/test.csv",
        help="Input CSV for synthetic /route payload generation.",
    )
    parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to evaluate.")
    return parser.parse_args()


def _iter_payloads_from_split_csv(path: Path, max_rows: int) -> list[dict]:
    payloads: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
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

            payloads.append(
                {
                    "rsu_id": str(row.get("rsu_node", "RSU_UNKNOWN")),
                    "timestamp": timestamp_s,
                    "vehicle_count": vehicle_count,
                    "vehicle_ids": [],
                    "emergency_vehicle_ids": [],
                    "avg_speed_mps": max(0.0, min(15.0, 15.0 - avg_latency_s * 10.0)),
                    "features": {
                        "registered_telemetry_count": float(row.get("registered_telemetry_count", vehicle_count) or vehicle_count),
                        "packets_received": float(row.get("packets_received", vehicle_count) or vehicle_count),
                        "bytes_received": float(row.get("bytes_received", vehicle_count * 128) or vehicle_count * 128),
                        "avg_latency_s": avg_latency_s,
                        "congested_local": float(row.get("congested_local", 0) or 0),
                        "congested_global": float(row.get("congested_global", 0) or 0),
                    },
                }
            )
    return payloads


def main() -> int:
    args = _parse_args()
    artifact = Path(args.artifact)
    split_csv = Path(args.input_csv)

    if not artifact.exists():
        print(f"[PHASE2][SMOKE] Artifact not found: {artifact}")
        return 2
    if not split_csv.exists():
        print(f"[PHASE2][SMOKE] Input CSV not found: {split_csv}")
        return 2

    engine = ForecastInferenceEngine.from_artifact_path(artifact)
    payloads = _iter_payloads_from_split_csv(split_csv, max(1, args.max_rows))
    if not payloads:
        print("[PHASE2][SMOKE] No payloads to evaluate.")
        return 2

    start = time.perf_counter()
    preds = [engine.predict_from_route_payload(payload) for payload in payloads]
    elapsed = time.perf_counter() - start

    avg_p = sum(float(p["p_congestion"]) for p in preds) / len(preds)
    avg_conf = sum(float(p["confidence"]) for p in preds) / len(preds)
    avg_unc = sum(float(p["uncertainty"]) for p in preds) / len(preds)
    latency_ms = (elapsed * 1000.0) / len(preds)

    print("[PHASE2][SMOKE] Inference complete")
    print(f"[PHASE2][SMOKE] rows={len(preds)} latency_ms_per_row={latency_ms:.6f}")
    print(f"[PHASE2][SMOKE] avg_p={clamp01(avg_p):.6f} avg_conf={clamp01(avg_conf):.6f} avg_unc={clamp01(avg_unc):.6f}")
    print(f"[PHASE2][SMOKE] model={preds[0].get('model')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
