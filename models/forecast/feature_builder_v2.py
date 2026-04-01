from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .common import stable_rsu_hash

# Extended 31-feature contract: deeper lags, rolling stats, congestion dynamics.
FEATURE_NAMES_V2 = [
    # --- Raw current-step features ---
    "connected_vehicle_count",
    "registered_telemetry_count",
    "packets_received",
    "bytes_received",
    "avg_latency_s",
    "congested_local",
    "congested_global",
    # --- Derived ratio features ---
    "bytes_per_vehicle",
    "packets_per_vehicle",
    # --- Lag features: vehicle count ---
    "lag1_vehicle_count",
    "lag2_vehicle_count",
    "lag3_vehicle_count",
    "lag5_vehicle_count",
    # --- Lag features: latency ---
    "lag1_latency",
    "lag2_latency",
    # --- Lag features: local congestion flag ---
    "lag1_congested_local",
    "lag2_congested_local",
    "lag3_congested_local",
    # --- Rolling stats: vehicle count ---
    "roll5_mean_vehicle",
    "roll5_std_vehicle",
    "roll5_max_vehicle",
    "roll10_mean_vehicle",
    "roll10_std_vehicle",
    # --- Rolling stats: latency ---
    "roll5_mean_latency",
    "roll5_std_latency",
    # --- Trend / rate-of-change ---
    "diff1_vehicle_count",   # current - lag1
    "diff2_vehicle_count",   # lag1 - lag2  (second derivative / acceleration)
    # --- Congestion dynamics ---
    "congestion_duration",   # consecutive steps (including current) in congested state
    "congestion_onset",      # 1 if transitioned into congestion this step
    # --- Identity & temporal ---
    "rsu_hash",
    "time_phase_300s",
]

assert len(FEATURE_NAMES_V2) == 31, f"Expected 31 features, got {len(FEATURE_NAMES_V2)}"


class FeatureStateV2:
    """Per-RSU causal state for building lag and rolling features."""

    def __init__(self) -> None:
        self.vehicle_counts: deque[float] = deque(maxlen=10)
        self.latencies: deque[float] = deque(maxlen=10)
        self.congestion_local_history: deque[float] = deque(maxlen=5)
        self.congestion_duration: int = 0


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_binary(value: Any) -> float:
    return 1.0 if _to_float(value, 0.0) >= 0.5 else 0.0


def _lag(d: deque, n: int, default: float) -> float:
    """n=1 → most recent element, n=2 → second most recent, etc."""
    if len(d) >= n:
        return d[-n]
    return default


def _roll_mean(d: deque, n: int, default: float) -> float:
    vals = list(d)[-n:]
    return float(np.mean(vals)) if vals else default


def _roll_std(d: deque, n: int) -> float:
    vals = list(d)[-n:]
    return float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0


def _roll_max(d: deque, n: int, default: float) -> float:
    vals = list(d)[-n:]
    return float(max(vals)) if vals else default


# --------------------------------------------------------------------------- #
# Core builder                                                                 #
# --------------------------------------------------------------------------- #

def build_feature_vector_v2(
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
    state: FeatureStateV2,
    update_state: bool = True,
) -> np.ndarray:
    """Build one 31-feature vector using only present/past state (no leakage)."""

    cvc = float(connected_vehicle_count)
    lat = float(avg_latency_s)
    cong = _to_binary(congested_local)
    cong_g = _to_binary(congested_global)

    # Derived ratios
    bpv = float(bytes_received) / max(1.0, cvc)
    ppv = float(packets_received) / max(1.0, cvc)

    # Lag: vehicle count (from history BEFORE this step)
    lag1_vc = _lag(state.vehicle_counts, 1, cvc)
    lag2_vc = _lag(state.vehicle_counts, 2, cvc)
    lag3_vc = _lag(state.vehicle_counts, 3, cvc)
    lag5_vc = _lag(state.vehicle_counts, 5, cvc)

    # Lag: latency
    lag1_lat = _lag(state.latencies, 1, lat)
    lag2_lat = _lag(state.latencies, 2, lat)

    # Lag: congestion
    lag1_cong = _lag(state.congestion_local_history, 1, cong)
    lag2_cong = _lag(state.congestion_local_history, 2, cong)
    lag3_cong = _lag(state.congestion_local_history, 3, cong)

    # Rolling stats: vehicle count
    r5_mean_vc = _roll_mean(state.vehicle_counts, 5, cvc)
    r5_std_vc = _roll_std(state.vehicle_counts, 5)
    r5_max_vc = _roll_max(state.vehicle_counts, 5, cvc)
    r10_mean_vc = _roll_mean(state.vehicle_counts, 10, cvc)
    r10_std_vc = _roll_std(state.vehicle_counts, 10)

    # Rolling stats: latency
    r5_mean_lat = _roll_mean(state.latencies, 5, lat)
    r5_std_lat = _roll_std(state.latencies, 5)

    # Trend: first difference (velocity) and second difference (acceleration)
    diff1 = cvc - lag1_vc
    diff2 = lag1_vc - lag2_vc

    # Congestion dynamics
    cong_dur = float(state.congestion_duration)
    cong_onset = 1.0 if (cong >= 0.5 and lag1_cong < 0.5) else 0.0

    # Identity & temporal
    rsu_h = stable_rsu_hash(rsu_id)
    time_ph = float(timestamp_s % 300.0) / 300.0

    features = np.array([
        cvc,
        float(registered_telemetry_count),
        float(packets_received),
        float(bytes_received),
        lat,
        cong,
        cong_g,
        bpv,
        ppv,
        lag1_vc,
        lag2_vc,
        lag3_vc,
        lag5_vc,
        lag1_lat,
        lag2_lat,
        lag1_cong,
        lag2_cong,
        lag3_cong,
        r5_mean_vc,
        r5_std_vc,
        r5_max_vc,
        r10_mean_vc,
        r10_std_vc,
        r5_mean_lat,
        r5_std_lat,
        diff1,
        diff2,
        cong_dur,
        cong_onset,
        rsu_h,
        time_ph,
    ], dtype=float)

    if update_state:
        # Update congestion duration counter
        if cong >= 0.5:
            state.congestion_duration += 1
        else:
            state.congestion_duration = 0
        state.vehicle_counts.append(cvc)
        state.latencies.append(lat)
        state.congestion_local_history.append(cong)

    return features


# --------------------------------------------------------------------------- #
# Training adapter                                                             #
# --------------------------------------------------------------------------- #

def build_training_features_from_row_v2(
    row: dict[str, Any],
    state_by_stream: dict[tuple[str, str], FeatureStateV2],
) -> np.ndarray:
    run_id = str(row.get("run_id", "unknown"))
    rsu_id = str(row.get("rsu_node", "RSU_UNKNOWN"))
    state = state_by_stream.setdefault((run_id, rsu_id), FeatureStateV2())

    cvc = _to_float(row.get("connected_vehicle_count"), 0.0)
    rtc = _to_float(row.get("registered_telemetry_count"), cvc)
    pkts = _to_float(row.get("packets_received"), rtc)
    bts = _to_float(row.get("bytes_received"), pkts * 128.0)
    lat = _to_float(row.get("avg_latency_s"), 0.0)
    cl = _to_float(row.get("congested_local"), 0.0)
    cg = _to_float(row.get("congested_global"), cl)
    ts = _to_float(row.get("timestamp_s"), _to_float(row.get("frame_idx"), 0.0))

    return build_feature_vector_v2(
        rsu_id=rsu_id,
        timestamp_s=ts,
        connected_vehicle_count=cvc,
        registered_telemetry_count=rtc,
        packets_received=pkts,
        bytes_received=bts,
        avg_latency_s=lat,
        congested_local=cl,
        congested_global=cg,
        state=state,
        update_state=True,
    )


# --------------------------------------------------------------------------- #
# Inference adapter                                                            #
# --------------------------------------------------------------------------- #

def build_inference_features_from_route_payload_v2(
    payload: dict[str, Any],
    state_by_rsu: dict[str, FeatureStateV2],
) -> np.ndarray:
    rsu_id = str(payload.get("rsu_id", "global"))
    state = state_by_rsu.setdefault(rsu_id, FeatureStateV2())

    vehicle_ids = payload.get("vehicle_ids")
    inferred_reg = float(len(vehicle_ids)) if isinstance(vehicle_ids, list) else 0.0

    cvc = _to_float(payload.get("vehicle_count"), inferred_reg)
    ts = _to_float(payload.get("timestamp"), 0.0)
    avg_speed = _to_float(payload.get("avg_speed_mps"), 0.0)

    raw_feats = payload.get("features")
    fobj: dict[str, Any] = raw_feats if isinstance(raw_feats, dict) else {}

    rtc = _to_float(fobj.get("registered_telemetry_count"), inferred_reg)
    pkts = _to_float(fobj.get("packets_received"), rtc)
    bts = _to_float(fobj.get("bytes_received"), pkts * 128.0)
    lat = _to_float(fobj.get("avg_latency_s"), 0.0)

    cl = fobj.get("congested_local")
    if cl is None:
        cl = 1.0 if (cvc >= 5.0 and avg_speed < 2.0) else 0.0
    cg = _to_float(fobj.get("congested_global", cl), 0.0)

    return build_feature_vector_v2(
        rsu_id=rsu_id,
        timestamp_s=ts,
        connected_vehicle_count=cvc,
        registered_telemetry_count=rtc,
        packets_received=pkts,
        bytes_received=bts,
        avg_latency_s=lat,
        congested_local=_to_float(cl, 0.0),
        congested_global=cg,
        state=state,
        update_state=True,
    )
