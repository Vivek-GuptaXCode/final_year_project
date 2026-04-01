from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .common import stable_rsu_hash

# Feature contract for training and inference artifact compatibility.
FEATURE_NAMES = [
    "connected_vehicle_count",
    "registered_telemetry_count",
    "packets_received",
    "bytes_received",
    "avg_latency_s",
    "congested_local",
    "congested_global",
    "lag_connected_vehicle_count",
    "lag_avg_latency_s",
    "rolling_connected_vehicle_count_5",
    "rsu_hash",
    "time_phase_300s",
]


class FeatureState:
    """Causal state used to build lag/rolling features per RSU stream."""

    def __init__(self) -> None:
        self.connected_counts: deque[float] = deque(maxlen=5)
        self.latencies: deque[float] = deque(maxlen=5)



def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)



def _to_binary(value: Any) -> float:
    return 1.0 if _to_float(value, 0.0) >= 0.5 else 0.0



def build_feature_vector(
    *,
    rsu_id: str,
    timestamp_s: float,
    connected_vehicle_count: float,
    registered_telemetry_count: float,
    packets_received: float,
    bytes_received: float,
    avg_latency_s: float,
    congested_local: float,
    congested_global: float,
    state: FeatureState,
    update_state: bool = True,
) -> np.ndarray:
    """Build one feature vector with only present/past state."""
    lag_count = state.connected_counts[-1] if state.connected_counts else connected_vehicle_count
    lag_latency = state.latencies[-1] if state.latencies else avg_latency_s
    rolling_count = (
        float(sum(state.connected_counts) / len(state.connected_counts))
        if state.connected_counts
        else connected_vehicle_count
    )

    features = np.array(
        [
            connected_vehicle_count,
            registered_telemetry_count,
            packets_received,
            bytes_received,
            avg_latency_s,
            _to_binary(congested_local),
            _to_binary(congested_global),
            lag_count,
            lag_latency,
            rolling_count,
            stable_rsu_hash(rsu_id),
            float(timestamp_s % 300.0) / 300.0,
        ],
        dtype=float,
    )

    if update_state:
        state.connected_counts.append(connected_vehicle_count)
        state.latencies.append(avg_latency_s)

    return features



def build_training_features_from_row(
    row: dict[str, Any],
    state_by_stream: dict[tuple[str, str], FeatureState],
) -> np.ndarray:
    run_id = str(row.get("run_id", "unknown"))
    rsu_id = str(row.get("rsu_node", "RSU_UNKNOWN"))
    stream_key = (run_id, rsu_id)
    state = state_by_stream.setdefault(stream_key, FeatureState())

    connected_vehicle_count = _to_float(row.get("connected_vehicle_count"), 0.0)
    registered_telemetry_count = _to_float(row.get("registered_telemetry_count"), connected_vehicle_count)
    packets_received = _to_float(row.get("packets_received"), registered_telemetry_count)
    bytes_received = _to_float(row.get("bytes_received"), packets_received * 128.0)
    avg_latency_s = _to_float(row.get("avg_latency_s"), 0.0)
    congested_local = _to_float(row.get("congested_local"), 0.0)
    congested_global = _to_float(row.get("congested_global"), congested_local)

    timestamp_s = _to_float(row.get("timestamp_s"), _to_float(row.get("frame_idx"), 0.0))

    return build_feature_vector(
        rsu_id=rsu_id,
        timestamp_s=timestamp_s,
        connected_vehicle_count=connected_vehicle_count,
        registered_telemetry_count=registered_telemetry_count,
        packets_received=packets_received,
        bytes_received=bytes_received,
        avg_latency_s=avg_latency_s,
        congested_local=congested_local,
        congested_global=congested_global,
        state=state,
        update_state=True,
    )



def build_inference_features_from_route_payload(
    payload: dict[str, Any],
    state_by_rsu: dict[str, FeatureState],
) -> np.ndarray:
    rsu_id = str(payload.get("rsu_id", "global"))
    state = state_by_rsu.setdefault(rsu_id, FeatureState())

    vehicle_ids = payload.get("vehicle_ids")
    if isinstance(vehicle_ids, list):
        inferred_registered = float(len(vehicle_ids))
    else:
        inferred_registered = 0.0

    connected_vehicle_count = _to_float(payload.get("vehicle_count"), inferred_registered)
    timestamp_s = _to_float(payload.get("timestamp"), 0.0)
    avg_speed_mps = _to_float(payload.get("avg_speed_mps"), 0.0)

    raw_features = payload.get("features")
    features_obj: dict[str, Any] = raw_features if isinstance(raw_features, dict) else {}
    registered_telemetry_count = _to_float(
        features_obj.get("registered_telemetry_count"), inferred_registered
    )
    packets_received = _to_float(features_obj.get("packets_received"), registered_telemetry_count)
    bytes_received = _to_float(features_obj.get("bytes_received"), packets_received * 128.0)
    avg_latency_s = _to_float(features_obj.get("avg_latency_s"), 0.0)

    congested_local = features_obj.get("congested_local")
    if congested_local is None:
        congested_local = 1.0 if (connected_vehicle_count >= 5.0 and avg_speed_mps < 2.0) else 0.0

    congested_global = features_obj.get("congested_global", congested_local)

    return build_feature_vector(
        rsu_id=rsu_id,
        timestamp_s=timestamp_s,
        connected_vehicle_count=connected_vehicle_count,
        registered_telemetry_count=registered_telemetry_count,
        packets_received=packets_received,
        bytes_received=bytes_received,
        avg_latency_s=avg_latency_s,
        congested_local=_to_float(congested_local, 0.0),
        congested_global=_to_float(congested_global, 0.0),
        state=state,
        update_state=True,
    )
