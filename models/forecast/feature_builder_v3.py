"""Feature builder V3: Extended 45-feature set for improved congestion forecasting.

Key improvements over V2:
1. Cross-RSU spatial features (neighbor congestion propagation)
2. Enhanced time encoding (hour-of-day, peak hour detection)
3. Advanced rolling windows (median, range, EMA)
4. Velocity/acceleration ratios

Target: Improve accuracy from 87.3% → 95%
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from .common import stable_rsu_hash

# Extended 45-feature contract for V3
FEATURE_NAMES_V3 = [
    # --- Raw current-step features (7) ---
    "connected_vehicle_count",
    "registered_telemetry_count",
    "packets_received",
    "bytes_received",
    "avg_latency_s",
    "congested_local",
    "congested_global",
    # --- Derived ratio features (2) ---
    "bytes_per_vehicle",
    "packets_per_vehicle",
    # --- Lag features: vehicle count (4) ---
    "lag1_vehicle_count",
    "lag2_vehicle_count",
    "lag3_vehicle_count",
    "lag5_vehicle_count",
    # --- Lag features: latency (2) ---
    "lag1_latency",
    "lag2_latency",
    # --- Lag features: local congestion flag (3) ---
    "lag1_congested_local",
    "lag2_congested_local",
    "lag3_congested_local",
    # --- Rolling stats: vehicle count - basic (5) ---
    "roll5_mean_vehicle",
    "roll5_std_vehicle",
    "roll5_max_vehicle",
    "roll10_mean_vehicle",
    "roll10_std_vehicle",
    # --- Rolling stats: vehicle count - advanced (4) [NEW] ---
    "roll3_mean_vehicle",
    "roll5_median_vehicle",
    "roll5_range_vehicle",
    "ema_vehicle_count",
    # --- Rolling stats: latency (2) ---
    "roll5_mean_latency",
    "roll5_std_latency",
    # --- Trend / rate-of-change (2) ---
    "diff1_vehicle_count",
    "diff2_vehicle_count",
    # --- Velocity/acceleration ratio [NEW] (1) ---
    "velocity_accel_ratio",
    # --- Congestion dynamics (2) ---
    "congestion_duration",
    "congestion_onset",
    # --- Identity & temporal - enhanced [NEW] (6) ---
    "rsu_hash",
    "time_phase_300s",
    "hour_sin",
    "hour_cos",
    "is_peak_hour",
    "minute_norm",
    # --- Cross-RSU spatial features [NEW] (5) ---
    "neighbor_avg_vehicle",
    "neighbor_congested_count",
    "neighbor_max_vehicle",
    "spatial_congestion_gradient",
    "isolation_score",
]

assert len(FEATURE_NAMES_V3) == 45, f"Expected 45 features, got {len(FEATURE_NAMES_V3)}"


class FeatureStateV3:
    """Per-RSU causal state for building lag and rolling features."""

    def __init__(self) -> None:
        self.vehicle_counts: deque[float] = deque(maxlen=15)
        self.latencies: deque[float] = deque(maxlen=10)
        self.congestion_local_history: deque[float] = deque(maxlen=5)
        self.congestion_duration: int = 0
        self.ema_vehicle: float = 0.0
        self.ema_alpha: float = 0.3


class RSUNeighborhood:
    """Manages cross-RSU neighbor relationships for spatial features."""

    def __init__(self) -> None:
        self.neighbors: dict[str, set[str]] = {}
        self.current_states: dict[str, dict[str, float]] = {}

    def register_neighbor(self, rsu_id: str, neighbor_id: str) -> None:
        """Register a neighbor relationship (bidirectional)."""
        if rsu_id not in self.neighbors:
            self.neighbors[rsu_id] = set()
        if neighbor_id not in self.neighbors:
            self.neighbors[neighbor_id] = set()
        self.neighbors[rsu_id].add(neighbor_id)
        self.neighbors[neighbor_id].add(rsu_id)

    def update_state(
        self, rsu_id: str, vehicle_count: float, congested: float
    ) -> None:
        """Update the current state of an RSU."""
        self.current_states[rsu_id] = {
            "vehicle_count": vehicle_count,
            "congested": congested,
        }

    def get_neighbor_features(self, rsu_id: str) -> dict[str, float]:
        """Get aggregated neighbor features for spatial context."""
        neighbor_ids = self.neighbors.get(rsu_id, set())
        if not neighbor_ids:
            return {
                "neighbor_avg_vehicle": 0.0,
                "neighbor_congested_count": 0.0,
                "neighbor_max_vehicle": 0.0,
                "spatial_congestion_gradient": 0.0,
                "isolation_score": 1.0,
            }

        neighbor_vehicles = []
        neighbor_congested = []
        for nid in neighbor_ids:
            if nid in self.current_states:
                neighbor_vehicles.append(self.current_states[nid]["vehicle_count"])
                neighbor_congested.append(self.current_states[nid]["congested"])

        if not neighbor_vehicles:
            return {
                "neighbor_avg_vehicle": 0.0,
                "neighbor_congested_count": 0.0,
                "neighbor_max_vehicle": 0.0,
                "spatial_congestion_gradient": 0.0,
                "isolation_score": 1.0,
            }

        self_state = self.current_states.get(rsu_id, {"vehicle_count": 0.0, "congested": 0.0})
        self_vehicle = self_state["vehicle_count"]
        self_congested = self_state["congested"]

        avg_neighbor_vehicle = float(np.mean(neighbor_vehicles))
        congested_count = sum(1 for c in neighbor_congested if c >= 0.5)
        max_neighbor_vehicle = float(max(neighbor_vehicles))

        # Spatial gradient: positive if self is more congested than neighbors
        avg_neighbor_congested = float(np.mean(neighbor_congested))
        gradient = self_congested - avg_neighbor_congested

        # Isolation score: 0 if well-connected with data, 1 if isolated
        coverage = len(neighbor_vehicles) / max(1, len(neighbor_ids))
        isolation = 1.0 - coverage

        return {
            "neighbor_avg_vehicle": avg_neighbor_vehicle,
            "neighbor_congested_count": float(congested_count),
            "neighbor_max_vehicle": max_neighbor_vehicle,
            "spatial_congestion_gradient": gradient,
            "isolation_score": isolation,
        }


# Global neighborhood manager (singleton pattern for inference)
_global_neighborhood: RSUNeighborhood | None = None


def get_global_neighborhood() -> RSUNeighborhood:
    """Get or create the global RSU neighborhood manager."""
    global _global_neighborhood
    if _global_neighborhood is None:
        _global_neighborhood = RSUNeighborhood()
    return _global_neighborhood


def reset_global_neighborhood() -> None:
    """Reset the global neighborhood (for testing/training)."""
    global _global_neighborhood
    _global_neighborhood = None


# --------------------------------------------------------------------------- #
# Helpers
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


def _roll_median(d: deque, n: int, default: float) -> float:
    vals = list(d)[-n:]
    return float(np.median(vals)) if vals else default


def _roll_range(d: deque, n: int) -> float:
    vals = list(d)[-n:]
    if len(vals) < 2:
        return 0.0
    return float(max(vals) - min(vals))


def _compute_ema(prev_ema: float, current: float, alpha: float) -> float:
    """Exponential moving average."""
    return alpha * current + (1 - alpha) * prev_ema


def _time_features(timestamp_s: float) -> dict[str, float]:
    """Compute enhanced time features."""
    # Assume timestamp_s is seconds since simulation start or epoch
    # For simulation, we'll treat it as time-of-day in a 24-hour cycle
    # scaled to 86400 seconds (24 hours)

    # Normalize to hour of day (0-24 cycle repeated)
    hour_of_day = (timestamp_s % 86400) / 3600.0

    # Sin/cos encoding for cyclical nature
    hour_rad = 2 * math.pi * hour_of_day / 24.0
    hour_sin = math.sin(hour_rad)
    hour_cos = math.cos(hour_rad)

    # Peak hour detection (7-9 AM and 5-7 PM)
    is_peak = 1.0 if (7 <= hour_of_day < 9 or 17 <= hour_of_day < 19) else 0.0

    # Minute of hour normalized
    minute_norm = (timestamp_s % 3600) / 3600.0

    # 300s phase (existing)
    time_phase_300 = float(timestamp_s % 300.0) / 300.0

    return {
        "time_phase_300s": time_phase_300,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_peak_hour": is_peak,
        "minute_norm": minute_norm,
    }


# --------------------------------------------------------------------------- #
# Core builder
# --------------------------------------------------------------------------- #

def build_feature_vector_v3(
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
    state: FeatureStateV3,
    neighborhood: RSUNeighborhood | None = None,
    update_state: bool = True,
) -> np.ndarray:
    """Build one 45-feature vector using only present/past state (no leakage)."""

    cvc = float(connected_vehicle_count)
    lat = float(avg_latency_s)
    cong = _to_binary(congested_local)
    cong_g = _to_binary(congested_global)

    # Derived ratios
    bpv = float(bytes_received) / max(1.0, cvc)
    ppv = float(packets_received) / max(1.0, cvc)

    # Lag: vehicle count
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

    # Rolling stats: vehicle count - basic
    r5_mean_vc = _roll_mean(state.vehicle_counts, 5, cvc)
    r5_std_vc = _roll_std(state.vehicle_counts, 5)
    r5_max_vc = _roll_max(state.vehicle_counts, 5, cvc)
    r10_mean_vc = _roll_mean(state.vehicle_counts, 10, cvc)
    r10_std_vc = _roll_std(state.vehicle_counts, 10)

    # Rolling stats: vehicle count - advanced (NEW)
    r3_mean_vc = _roll_mean(state.vehicle_counts, 3, cvc)
    r5_median_vc = _roll_median(state.vehicle_counts, 5, cvc)
    r5_range_vc = _roll_range(state.vehicle_counts, 5)
    ema_vc = state.ema_vehicle if len(state.vehicle_counts) > 0 else cvc

    # Rolling stats: latency
    r5_mean_lat = _roll_mean(state.latencies, 5, lat)
    r5_std_lat = _roll_std(state.latencies, 5)

    # Trend: first difference (velocity) and second difference (acceleration)
    diff1 = cvc - lag1_vc
    diff2 = lag1_vc - lag2_vc

    # Velocity/acceleration ratio (NEW)
    if abs(diff2) > 0.01:
        vel_accel_ratio = diff1 / diff2
        vel_accel_ratio = max(-10.0, min(10.0, vel_accel_ratio))  # Clamp extreme values
    else:
        vel_accel_ratio = 0.0

    # Congestion dynamics
    cong_dur = float(state.congestion_duration)
    cong_onset = 1.0 if (cong >= 0.5 and lag1_cong < 0.5) else 0.0

    # Identity & temporal - enhanced
    rsu_h = stable_rsu_hash(rsu_id)
    time_feats = _time_features(timestamp_s)

    # Cross-RSU spatial features (NEW)
    if neighborhood is not None:
        neighbor_feats = neighborhood.get_neighbor_features(rsu_id)
    else:
        neighbor_feats = {
            "neighbor_avg_vehicle": 0.0,
            "neighbor_congested_count": 0.0,
            "neighbor_max_vehicle": 0.0,
            "spatial_congestion_gradient": 0.0,
            "isolation_score": 1.0,
        }

    features = np.array([
        # Raw current-step (7)
        cvc,
        float(registered_telemetry_count),
        float(packets_received),
        float(bytes_received),
        lat,
        cong,
        cong_g,
        # Derived ratios (2)
        bpv,
        ppv,
        # Lag: vehicle count (4)
        lag1_vc,
        lag2_vc,
        lag3_vc,
        lag5_vc,
        # Lag: latency (2)
        lag1_lat,
        lag2_lat,
        # Lag: congestion (3)
        lag1_cong,
        lag2_cong,
        lag3_cong,
        # Rolling stats: vehicle - basic (5)
        r5_mean_vc,
        r5_std_vc,
        r5_max_vc,
        r10_mean_vc,
        r10_std_vc,
        # Rolling stats: vehicle - advanced (4) [NEW]
        r3_mean_vc,
        r5_median_vc,
        r5_range_vc,
        ema_vc,
        # Rolling stats: latency (2)
        r5_mean_lat,
        r5_std_lat,
        # Trend (2)
        diff1,
        diff2,
        # Velocity/acceleration ratio (1) [NEW]
        vel_accel_ratio,
        # Congestion dynamics (2)
        cong_dur,
        cong_onset,
        # Identity & temporal (6) [NEW expanded]
        rsu_h,
        time_feats["time_phase_300s"],
        time_feats["hour_sin"],
        time_feats["hour_cos"],
        time_feats["is_peak_hour"],
        time_feats["minute_norm"],
        # Cross-RSU spatial (5) [NEW]
        neighbor_feats["neighbor_avg_vehicle"],
        neighbor_feats["neighbor_congested_count"],
        neighbor_feats["neighbor_max_vehicle"],
        neighbor_feats["spatial_congestion_gradient"],
        neighbor_feats["isolation_score"],
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
        # Update EMA
        state.ema_vehicle = _compute_ema(state.ema_vehicle, cvc, state.ema_alpha)
        # Update neighborhood state
        if neighborhood is not None:
            neighborhood.update_state(rsu_id, cvc, cong)

    return features


# --------------------------------------------------------------------------- #
# Training adapter
# --------------------------------------------------------------------------- #

def build_training_features_from_row_v3(
    row: dict[str, Any],
    state_by_stream: dict[tuple[str, str], FeatureStateV3],
    neighborhood: RSUNeighborhood | None = None,
) -> np.ndarray:
    run_id = str(row.get("run_id", "unknown"))
    rsu_id = str(row.get("rsu_node", "RSU_UNKNOWN"))
    state = state_by_stream.setdefault((run_id, rsu_id), FeatureStateV3())

    cvc = _to_float(row.get("connected_vehicle_count"), 0.0)
    rtc = _to_float(row.get("registered_telemetry_count"), cvc)
    pkts = _to_float(row.get("packets_received"), rtc)
    bts = _to_float(row.get("bytes_received"), pkts * 128.0)
    lat = _to_float(row.get("avg_latency_s"), 0.0)
    cl = _to_float(row.get("congested_local"), 0.0)
    cg = _to_float(row.get("congested_global"), cl)
    ts = _to_float(row.get("timestamp_s"), _to_float(row.get("frame_idx"), 0.0))

    return build_feature_vector_v3(
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
        neighborhood=neighborhood,
        update_state=True,
    )


# --------------------------------------------------------------------------- #
# Inference adapter
# --------------------------------------------------------------------------- #

def build_inference_features_from_route_payload_v3(
    payload: dict[str, Any],
    state_by_rsu: dict[str, FeatureStateV3],
    neighborhood: RSUNeighborhood | None = None,
) -> np.ndarray:
    rsu_id = str(payload.get("rsu_id", "global"))
    state = state_by_rsu.setdefault(rsu_id, FeatureStateV3())

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

    # Use global neighborhood if none provided
    if neighborhood is None:
        neighborhood = get_global_neighborhood()

    return build_feature_vector_v3(
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
        neighborhood=neighborhood,
        update_state=True,
    )
