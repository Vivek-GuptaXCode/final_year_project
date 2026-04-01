from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Any

import networkx as nx


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class GNNRerouteConfig:
    """Config for lightweight graph message-passing rerouting inference."""

    message_passing_steps: int = 2
    self_weight: float = 0.65
    neighbor_weight: float = 0.35
    degree_boost: float = 0.12
    medium_risk_threshold: float = 0.45
    high_risk_threshold: float = 0.70
    low_confidence_threshold: float = 0.55
    max_reroute_fraction: float = 0.40

    @classmethod
    def from_env(cls) -> "GNNRerouteConfig":
        def _read_float(name: str, default: float) -> float:
            raw = os.getenv(name, "")
            if raw == "":
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def _read_int(name: str, default: int) -> int:
            raw = os.getenv(name, "")
            if raw == "":
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        cfg = cls(
            message_passing_steps=max(1, _read_int("HYBRID_GNN_STEPS", 2)),
            self_weight=max(0.0, _read_float("HYBRID_GNN_SELF_WEIGHT", 0.65)),
            neighbor_weight=max(0.0, _read_float("HYBRID_GNN_NEIGHBOR_WEIGHT", 0.35)),
            degree_boost=max(0.0, _read_float("HYBRID_GNN_DEGREE_BOOST", 0.12)),
            medium_risk_threshold=_clamp(_read_float("HYBRID_GNN_MEDIUM_RISK_THRESHOLD", 0.45)),
            high_risk_threshold=_clamp(_read_float("HYBRID_GNN_HIGH_RISK_THRESHOLD", 0.70)),
            low_confidence_threshold=_clamp(_read_float("HYBRID_GNN_LOW_CONFIDENCE_THRESHOLD", 0.55)),
            max_reroute_fraction=_clamp(_read_float("HYBRID_GNN_MAX_REROUTE_FRACTION", 0.40)),
        )

        # Normalize message-passing weights for stable inference.
        total_weight = cfg.self_weight + cfg.neighbor_weight
        if total_weight <= 0.0:
            cfg.self_weight = 0.65
            cfg.neighbor_weight = 0.35
        else:
            cfg.self_weight = cfg.self_weight / total_weight
            cfg.neighbor_weight = cfg.neighbor_weight / total_weight

        if cfg.high_risk_threshold < cfg.medium_risk_threshold:
            cfg.high_risk_threshold = cfg.medium_risk_threshold

        return cfg


class GNNRerouteEngine:
    """GNN-style, dependency-free rerouting engine for live server inference.

    This engine approximates graph neural message passing over the RSU graph and
    returns a route policy payload compatible with existing server contracts.
    """

    def __init__(self, config: GNNRerouteConfig | None = None) -> None:
        self._config = config or GNNRerouteConfig()

    @staticmethod
    def _base_congestion_signal(
        *,
        vehicle_count: int,
        avg_speed_mps: float,
        emergency_vehicle_count: int,
    ) -> float:
        count_signal = _clamp(float(max(0, vehicle_count)) / 50.0)
        speed_signal = 1.0 - _clamp(max(0.0, float(avg_speed_mps)) / 15.0)
        emergency_signal = 1.0 if emergency_vehicle_count > 0 else 0.0
        return _clamp(0.55 * count_signal + 0.35 * speed_signal + 0.10 * emergency_signal)

    @staticmethod
    def _risk_level(score: float, config: GNNRerouteConfig) -> str:
        if score >= config.high_risk_threshold:
            return "high"
        if score >= config.medium_risk_threshold:
            return "medium"
        return "low"

    @staticmethod
    def _stable_vehicle_jitter(rsu_id: str, vehicle_id: str) -> float:
        digest = hashlib.sha256(f"{rsu_id}:{vehicle_id}".encode("utf-8")).hexdigest()
        # 8 hex chars gives stable deterministic jitter in [0, 1].
        return int(digest[:8], 16) / float(16**8 - 1)

    def _message_pass(
        self,
        *,
        rsu_graph: nx.Graph,
        rsu_id: str,
        base_signal: float,
    ) -> tuple[float, dict[str, float]]:
        node_count = rsu_graph.number_of_nodes()
        if node_count <= 0:
            return base_signal, {
                "target_node_score": base_signal,
                "network_mean_score": base_signal,
                "graph_support": 0.0,
            }

        # Build initial node states with stronger signal at the reporting RSU.
        states: dict[str, float] = {}
        for node in rsu_graph.nodes:
            node_id = str(node)
            if node_id == rsu_id:
                states[node_id] = base_signal
            else:
                states[node_id] = _clamp(base_signal * 0.35)

        if rsu_id not in states:
            states[rsu_id] = base_signal

        for _ in range(self._config.message_passing_steps):
            next_states: dict[str, float] = {}
            for node_id, current_value in states.items():
                if node_id in rsu_graph:
                    neighbors = [str(n) for n in rsu_graph.neighbors(node_id)]
                    degree = rsu_graph.degree(node_id)
                else:
                    neighbors = []
                    degree = 0

                if neighbors:
                    neighbor_mean = sum(states.get(n, current_value) for n in neighbors) / float(len(neighbors))
                else:
                    neighbor_mean = current_value

                degree_term = 0.0
                if degree > 0:
                    degree_term = _clamp(math.log1p(float(degree)) / math.log(8.0))

                updated = (
                    self._config.self_weight * current_value
                    + self._config.neighbor_weight * neighbor_mean
                    + self._config.degree_boost * degree_term
                )
                next_states[node_id] = _clamp(updated)
            states = next_states

        target_score = states.get(rsu_id, base_signal)
        network_mean = _clamp(sum(states.values()) / float(len(states)))

        max_possible_edges = max(1.0, (node_count * (node_count - 1)) / 2.0)
        connectivity = _clamp(float(rsu_graph.number_of_edges()) / max_possible_edges)
        if rsu_id in rsu_graph and node_count > 1:
            local_density = _clamp(float(rsu_graph.degree(rsu_id)) / float(node_count - 1))
        else:
            local_density = 0.0
        graph_support = _clamp(0.6 * local_density + 0.4 * connectivity)

        gnn_score = _clamp(0.7 * target_score + 0.3 * network_mean)
        return gnn_score, {
            "target_node_score": round(target_score, 6),
            "network_mean_score": round(network_mean, 6),
            "graph_support": round(graph_support, 6),
        }

    def _derive_confidence(
        self,
        *,
        p_congestion: float,
        graph_support: float,
        vehicle_count: int,
        avg_speed_mps: float,
    ) -> tuple[float, float]:
        telemetry_support = _clamp(float(max(0, vehicle_count)) / 20.0)
        # High absolute speed variance around operating midpoint increases uncertainty.
        speed_center = 7.5
        speed_stability = 1.0 - _clamp(abs(float(avg_speed_mps) - speed_center) / speed_center)

        # Scores farther away from decision boundary are typically more stable.
        boundary_margin = _clamp(abs(float(p_congestion) - self._config.medium_risk_threshold) / 0.40)

        confidence = _clamp(
            0.50 * graph_support + 0.25 * telemetry_support + 0.15 * speed_stability + 0.10 * boundary_margin
        )
        uncertainty = _clamp(1.0 - confidence)
        return confidence, uncertainty

    def _build_recommended_action(
        self,
        *,
        risk_level: str,
        confidence: float,
        emergency_active: bool,
    ) -> dict[str, Any]:
        if emergency_active:
            reroute_fraction = 1.0
            reroute_mode = "dijkstra"
            min_confidence = 0.0
        else:
            if risk_level == "high":
                reroute_fraction = min(self._config.max_reroute_fraction, 0.35)
            elif risk_level == "medium":
                reroute_fraction = min(self._config.max_reroute_fraction, 0.20)
            else:
                reroute_fraction = 0.0

            if confidence < self._config.low_confidence_threshold:
                reroute_mode = "travel_time"
                reroute_fraction = min(reroute_fraction, 0.15)
                min_confidence = 0.0
            else:
                reroute_mode = "gnn_effort"
                min_confidence = 0.50

        reroute_enabled = emergency_active or (reroute_fraction > 0.0)

        return {
            "reroute_bias": "avoid_hotspots" if risk_level != "low" else "normal",
            "signal_priority": "inbound_relief" if risk_level == "high" else "balanced",
            "reroute_enabled": reroute_enabled,
            "reroute_mode": reroute_mode,
            "reroute_fraction": _clamp(reroute_fraction),
            "min_confidence": _clamp(min_confidence),
            "fallback_algorithm": "dijkstra",
        }

    def _build_route_directives(
        self,
        *,
        rsu_id: str,
        vehicle_ids: list[str],
        emergency_vehicle_ids: list[str],
        recommended_action: dict[str, Any],
        p_congestion: float,
        confidence: float,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        unique_vehicle_ids: list[str] = []
        seen: set[str] = set()
        for raw_id in vehicle_ids:
            vid = str(raw_id)
            if vid and vid not in seen:
                seen.add(vid)
                unique_vehicle_ids.append(vid)

        emergency_set = {str(vid) for vid in emergency_vehicle_ids if str(vid) in seen}

        scored: list[tuple[float, str]] = []
        for vid in unique_vehicle_ids:
            if vid in emergency_set:
                score = 2.0
            else:
                jitter = self._stable_vehicle_jitter(rsu_id, vid)
                score = 0.70 * p_congestion + 0.20 * (1.0 - confidence) + 0.10 * jitter
            scored.append((float(score), vid))
        scored.sort(key=lambda row: row[0], reverse=True)
        priority_order = [vid for _score, vid in scored]

        if not bool(recommended_action.get("reroute_enabled", False)):
            return [], priority_order

        if emergency_set:
            target_ids = [vid for vid in priority_order if vid in emergency_set]
        else:
            try:
                frac = float(recommended_action.get("reroute_fraction", 0.0))
            except Exception:
                frac = 0.0
            frac = _clamp(frac)
            if frac <= 0.0:
                return [], priority_order
            target_count = max(1, int(len(priority_order) * frac))
            target_ids = priority_order[:target_count]

        mode = str(recommended_action.get("reroute_mode", "travel_time"))
        directives: list[dict[str, Any]] = []
        for vid in target_ids:
            directives.append(
                {
                    "vehicle_id": vid,
                    "mode": "dijkstra" if vid in emergency_set else mode,
                    "priority": "emergency" if vid in emergency_set else "normal",
                }
            )

        return directives, priority_order

    def predict(
        self,
        *,
        rsu_graph: nx.Graph,
        rsu_id: str,
        sim_timestamp: float,
        vehicle_ids: list[str],
        emergency_vehicle_ids: list[str],
        vehicle_count: int,
        avg_speed_mps: float,
    ) -> dict[str, Any]:
        emergency_count = len(emergency_vehicle_ids)
        base_signal = self._base_congestion_signal(
            vehicle_count=vehicle_count,
            avg_speed_mps=avg_speed_mps,
            emergency_vehicle_count=emergency_count,
        )

        gnn_score, graph_metrics = self._message_pass(
            rsu_graph=rsu_graph,
            rsu_id=rsu_id,
            base_signal=base_signal,
        )

        confidence, uncertainty = self._derive_confidence(
            p_congestion=gnn_score,
            graph_support=float(graph_metrics.get("graph_support", 0.0)),
            vehicle_count=vehicle_count,
            avg_speed_mps=avg_speed_mps,
        )
        risk_level = self._risk_level(gnn_score, self._config)
        emergency_active = emergency_count > 0

        recommended_action = self._build_recommended_action(
            risk_level=risk_level,
            confidence=confidence,
            emergency_active=emergency_active,
        )
        route_directives, priority_order = self._build_route_directives(
            rsu_id=rsu_id,
            vehicle_ids=vehicle_ids,
            emergency_vehicle_ids=emergency_vehicle_ids,
            recommended_action=recommended_action,
            p_congestion=gnn_score,
            confidence=confidence,
        )

        if emergency_active:
            strategy = "gnn_emergency_override"
        elif confidence < self._config.low_confidence_threshold:
            strategy = "gnn_confidence_fallback"
        else:
            strategy = "gnn_primary"

        return {
            "model": "gnn_reroute_v1",
            "source": "graph_message_passing",
            "p_congestion": _clamp(gnn_score),
            "confidence": _clamp(confidence),
            "uncertainty": _clamp(uncertainty),
            "risk_level": risk_level,
            "strategy": strategy,
            "recommended_action": recommended_action,
            "route_directives": route_directives,
            "vehicle_priority_order": priority_order,
            "diagnostics": {
                "sim_timestamp": float(sim_timestamp),
                "graph_nodes": int(rsu_graph.number_of_nodes()),
                "graph_edges": int(rsu_graph.number_of_edges()),
                "base_signal": round(base_signal, 6),
                **graph_metrics,
            },
        }