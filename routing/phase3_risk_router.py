from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class Phase3RoutingConfig:
    """Tunable config for confidence-aware risk routing."""

    low_confidence_threshold: float = 0.55
    uncertainty_penalty_weight: float = 0.35
    delay_scale_seconds: float = 90.0
    high_risk_score_threshold: float = 0.85
    medium_risk_score_threshold: float = 0.45
    max_reroute_fraction: float = 0.40

    @classmethod
    def from_env(cls) -> "Phase3RoutingConfig":
        def _read(name: str, default: float) -> float:
            raw = os.getenv(name, "")
            if raw == "":
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        return cls(
            low_confidence_threshold=_clamp(_read("HYBRID_P3_LOW_CONFIDENCE_THRESHOLD", 0.55)),
            uncertainty_penalty_weight=max(0.0, _read("HYBRID_P3_UNCERTAINTY_WEIGHT", 0.35)),
            delay_scale_seconds=max(1.0, _read("HYBRID_P3_DELAY_SCALE_SECONDS", 90.0)),
            high_risk_score_threshold=_clamp(_read("HYBRID_P3_HIGH_RISK_SCORE", 0.85)),
            medium_risk_score_threshold=_clamp(_read("HYBRID_P3_MEDIUM_RISK_SCORE", 0.45)),
            max_reroute_fraction=_clamp(_read("HYBRID_P3_MAX_REROUTE_FRACTION", 0.40)),
        )


def _derive_risk_level(score: float, config: Phase3RoutingConfig) -> str:
    if score >= config.high_risk_score_threshold:
        return "high"
    if score >= config.medium_risk_score_threshold:
        return "medium"
    return "low"


def build_phase3_decision(
    *,
    rsu_id: str,
    sim_timestamp: float,
    vehicle_ids: list[str],
    emergency_vehicle_ids: list[str],
    vehicle_count: int,
    avg_speed_mps: float,
    p_congestion: float,
    confidence: float,
    uncertainty: float,
    config: Phase3RoutingConfig,
) -> dict[str, Any]:
    """Build confidence-aware rerouting policy with deterministic fallback.

    The output preserves the existing recommended_action contract while adding
    explainability fields required for Phase 3 auditability.
    """
    p_congestion = _clamp(float(p_congestion))
    confidence = _clamp(float(confidence))
    uncertainty = _clamp(float(uncertainty))

    vehicle_count = max(0, int(vehicle_count))
    speed_floor = max(1.0, float(avg_speed_mps))

    # Expected delay proxy grows with load and predicted congestion, but is speed-normalized.
    estimated_delay_s = (vehicle_count / speed_floor) * (1.0 + 2.0 * p_congestion)
    delay_term = _clamp(estimated_delay_s / config.delay_scale_seconds)
    uncertainty_term = _clamp(uncertainty * config.uncertainty_penalty_weight)
    risk_score = _clamp(delay_term + uncertainty_term)

    emergency_active = len(emergency_vehicle_ids) > 0
    low_confidence = confidence < config.low_confidence_threshold

    risk_level = _derive_risk_level(risk_score, config)

    if emergency_active:
        reroute_fraction = 1.0
        reroute_mode = "dijkstra"
        min_confidence = 0.0
        fallback_triggered = False
        strategy = "emergency_override"
    else:
        # BALANCED: Target high-delay vehicles with moderate rerouting
        # Key insight: Reroute stuck vehicles, not random traffic
        if risk_level == "high" and p_congestion >= 0.5:
            # High risk + congestion: moderate rerouting (12%)
            reroute_fraction = min(config.max_reroute_fraction, 0.12)
        elif risk_level == "medium" and p_congestion >= 0.4:
            # Medium risk: light rerouting (6%)
            reroute_fraction = min(config.max_reroute_fraction, 0.06)
        else:
            # Low risk: no rerouting
            reroute_fraction = 0.0

        if low_confidence:
            # Conservative fallback policy under uncertain forecast.
            reroute_mode = "travel_time"
            reroute_fraction = min(reroute_fraction, 0.08)
            min_confidence = 0.0
            fallback_triggered = True
            strategy = "confidence_fallback"
        else:
            reroute_mode = "travel_time"
            min_confidence = 0.55
            fallback_triggered = False
            strategy = "risk_aware_primary"

    reroute_enabled = emergency_active or (reroute_fraction > 0.0)

    directives: list[dict[str, Any]] = []
    if reroute_enabled and vehicle_ids:
        emergency_id_set = {str(vid) for vid in emergency_vehicle_ids}
        if emergency_active:
            target_ids = [str(vid) for vid in vehicle_ids if str(vid) in emergency_id_set]
        else:
            target = max(1, int(len(vehicle_ids) * reroute_fraction))
            target_ids = [str(vid) for vid in vehicle_ids[:target]]

        for vid in target_ids:
            directives.append(
                {
                    "vehicle_id": vid,
                    "mode": reroute_mode,
                    "priority": "emergency" if vid in emergency_id_set else "normal",
                }
            )

    alternatives = [
        {
            "name": "risk_aware_primary",
            "mode": "gnn_effort",
            "score": round(float(risk_score), 6),
        },
        {
            "name": "confidence_fallback",
            "mode": "travel_time",
            "score": round(float(1.0 - confidence), 6),
        },
        {
            "name": "emergency_override",
            "mode": "dijkstra",
            "score": 1.0 if emergency_active else 0.0,
        },
    ]

    return {
        "routing_engine": {
            "primary": "phase3_risk_router_v1",
            "fallback": "dijkstra",
        },
        "risk_level": risk_level,
        "recommended_action": {
            "reroute_bias": "avoid_hotspots" if risk_level != "low" else "normal",
            "signal_priority": "inbound_relief" if risk_level == "high" else "balanced",
            "reroute_enabled": reroute_enabled,
            "reroute_mode": reroute_mode,
            "reroute_fraction": _clamp(reroute_fraction),
            "min_confidence": _clamp(min_confidence),
            "fallback_algorithm": "dijkstra",
        },
        "route_directives": directives,
        "phase3": {
            "enabled": True,
            "strategy": strategy,
            "fallback_triggered": fallback_triggered,
            "risk_score": round(float(risk_score), 6),
            "risk_components": {
                "delay_term": round(float(delay_term), 6),
                "uncertainty_term": round(float(uncertainty_term), 6),
                "estimated_delay_s": round(float(estimated_delay_s), 3),
            },
            "alternatives": alternatives,
            "decision_context": {
                "rsu_id": rsu_id,
                "sim_timestamp": float(sim_timestamp),
                "vehicle_count": vehicle_count,
                "avg_speed_mps": float(avg_speed_mps),
                "p_congestion": float(p_congestion),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "emergency_active": emergency_active,
            },
        },
    }
