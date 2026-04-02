"""Hybrid Fusion Orchestrator: Coordinates forecasting, routing, and signal control.

The fusion controller implements event-triggered coordination:
1. Forecast (Phase 2) provides congestion probability and confidence
2. Risk-aware routing (Phase 3) uses forecast to compute risk scores and reroute decisions
3. Signal control (Phase 4) receives coordination hints from routing decisions
4. Fusion orchestrator coordinates timing and resolves conflicts

Key design decisions based on literature review:
- Soft coordination: Routing provides "signal priority hints" but signals make final decisions
- Pre-emptive action: High-confidence forecasts trigger proactive measures before congestion
- Graceful degradation: If any subsystem fails, others continue operating
- Ablation support: Any combination of subsystems can be enabled/disabled for evaluation

References:
- PressLight (KDD'19): Pressure-based coordination for signal-routing interaction
- CoLight (CIKM'19): Attention-based coordination between neighboring signals
- SUMO-RL (Alegre, 2019): Standard observation/action spaces for traffic signal RL
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class FusionMode(Enum):
    """Operating modes for the fusion controller."""
    
    FULL_HYBRID = "full_hybrid"           # All systems active + coordination
    FORECAST_ONLY = "forecast_only"       # Phase 2 only (no action)
    ROUTING_ONLY = "routing_only"         # Phase 3 only (no signals)
    SIGNAL_ONLY = "signal_only"           # Phase 4 only (no routing)
    REACTIVE_BASELINE = "reactive_baseline"  # No forecasting (reactive control)
    NO_AI = "no_ai"                       # Fixed-time signals, no rerouting


@dataclass
class FusionConfig:
    """Configuration for the hybrid fusion controller.
    
    Parameters
    ----------
    mode : FusionMode
        Operating mode (full_hybrid, ablation variants, etc.)
    forecast_enabled : bool
        Enable Phase 2 congestion forecasting
    routing_enabled : bool
        Enable Phase 3 risk-aware routing
    signal_enabled : bool
        Enable Phase 4 adaptive signal control
    coordination_enabled : bool
        Enable cross-system coordination (signal hints to routing)
    pre_emptive_threshold : float
        p_congestion threshold to trigger pre-emptive action (default 0.7)
    pre_emptive_lookahead_s : float
        Seconds ahead to start pre-emptive action (default 30.0)
    confidence_threshold : float
        Minimum confidence to trust forecast (default 0.6)
    signal_priority_weight : float
        Weight for signal priority in routing cost (default 0.15)
    emergency_override : bool
        Enable emergency vehicle override (always True in production)
    log_decisions : bool
        Log all fusion decisions for analysis
    """
    
    mode: FusionMode = FusionMode.FULL_HYBRID
    forecast_enabled: bool = True
    routing_enabled: bool = True
    signal_enabled: bool = True
    coordination_enabled: bool = True
    pre_emptive_threshold: float = 0.70
    pre_emptive_lookahead_s: float = 30.0
    confidence_threshold: float = 0.60
    signal_priority_weight: float = 0.15
    emergency_override: bool = True
    log_decisions: bool = True
    
    @classmethod
    def from_mode(cls, mode: FusionMode) -> "FusionConfig":
        """Create config from a predefined mode."""
        if mode == FusionMode.FULL_HYBRID:
            return cls(
                mode=mode,
                forecast_enabled=True,
                routing_enabled=True,
                signal_enabled=True,
                coordination_enabled=True,
            )
        elif mode == FusionMode.FORECAST_ONLY:
            return cls(
                mode=mode,
                forecast_enabled=True,
                routing_enabled=False,
                signal_enabled=False,
                coordination_enabled=False,
            )
        elif mode == FusionMode.ROUTING_ONLY:
            return cls(
                mode=mode,
                forecast_enabled=True,  # Routing needs forecast input
                routing_enabled=True,
                signal_enabled=False,
                coordination_enabled=False,
            )
        elif mode == FusionMode.SIGNAL_ONLY:
            return cls(
                mode=mode,
                forecast_enabled=False,
                routing_enabled=False,
                signal_enabled=True,
                coordination_enabled=False,
            )
        elif mode == FusionMode.REACTIVE_BASELINE:
            return cls(
                mode=mode,
                forecast_enabled=False,
                routing_enabled=True,
                signal_enabled=True,
                coordination_enabled=False,  # No forecast → no pre-emptive coordination
            )
        elif mode == FusionMode.NO_AI:
            return cls(
                mode=mode,
                forecast_enabled=False,
                routing_enabled=False,
                signal_enabled=False,
                coordination_enabled=False,
            )
        else:
            return cls(mode=mode)
    
    @classmethod
    def from_env(cls) -> "FusionConfig":
        """Load configuration from environment variables."""
        mode_str = os.environ.get("HYBRID_FUSION_MODE", "full_hybrid")
        try:
            mode = FusionMode(mode_str)
        except ValueError:
            mode = FusionMode.FULL_HYBRID
        
        config = cls.from_mode(mode)
        
        # Override individual settings from env
        if os.environ.get("HYBRID_FUSION_FORECAST_ENABLED"):
            config.forecast_enabled = os.environ.get("HYBRID_FUSION_FORECAST_ENABLED", "true").lower() == "true"
        if os.environ.get("HYBRID_FUSION_ROUTING_ENABLED"):
            config.routing_enabled = os.environ.get("HYBRID_FUSION_ROUTING_ENABLED", "true").lower() == "true"
        if os.environ.get("HYBRID_FUSION_SIGNAL_ENABLED"):
            config.signal_enabled = os.environ.get("HYBRID_FUSION_SIGNAL_ENABLED", "true").lower() == "true"
        if os.environ.get("HYBRID_FUSION_COORDINATION_ENABLED"):
            config.coordination_enabled = os.environ.get("HYBRID_FUSION_COORDINATION_ENABLED", "true").lower() == "true"
        if os.environ.get("HYBRID_FUSION_PREEMPTIVE_THRESHOLD"):
            config.pre_emptive_threshold = float(os.environ.get("HYBRID_FUSION_PREEMPTIVE_THRESHOLD", "0.7"))
        
        return config


@dataclass
class FusionDecision:
    """Output of a single fusion decision cycle.
    
    Captures the coordinated decision across all subsystems for a single
    time step, including inputs, intermediate values, and final actions.
    """
    
    timestamp: float
    rsu_id: str
    
    # Phase 2: Forecast outputs
    p_congestion: float | None = None
    forecast_confidence: float | None = None
    forecast_model: str | None = None
    
    # Phase 3: Routing outputs
    risk_level: str | None = None
    risk_score: float | None = None
    reroute_fraction: float | None = None
    reroute_mode: str | None = None
    route_directives: list[dict] = field(default_factory=list)
    
    # Phase 4: Signal outputs  
    signal_actions: dict[str, int] = field(default_factory=dict)
    signal_rewards: dict[str, float] = field(default_factory=dict)
    
    # Fusion coordination
    pre_emptive_triggered: bool = False
    coordination_hints: dict[str, Any] = field(default_factory=dict)
    emergency_override_active: bool = False
    
    # Metadata
    mode: str = "full_hybrid"
    decision_time_ms: float = 0.0
    subsystems_active: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize decision to dictionary."""
        return {
            "timestamp": self.timestamp,
            "rsu_id": self.rsu_id,
            "forecast": {
                "p_congestion": self.p_congestion,
                "confidence": self.forecast_confidence,
                "model": self.forecast_model,
            },
            "routing": {
                "risk_level": self.risk_level,
                "risk_score": self.risk_score,
                "reroute_fraction": self.reroute_fraction,
                "reroute_mode": self.reroute_mode,
                "directives_count": len(self.route_directives),
            },
            "signals": {
                "actions": self.signal_actions,
                "rewards": self.signal_rewards,
            },
            "fusion": {
                "pre_emptive_triggered": self.pre_emptive_triggered,
                "coordination_hints": self.coordination_hints,
                "emergency_override": self.emergency_override_active,
            },
            "meta": {
                "mode": self.mode,
                "decision_time_ms": self.decision_time_ms,
                "subsystems_active": self.subsystems_active,
            },
        }


class FusionOrchestrator:
    """Orchestrates the hybrid fusion of forecasting, routing, and signal control.
    
    The orchestrator implements a soft coordination protocol:
    1. Forecast provides congestion probability and confidence
    2. If pre-emptive threshold exceeded, trigger proactive measures
    3. Routing computes risk-aware decisions using forecast
    4. Signal control receives "priority hints" from routing (not hard constraints)
    5. Each subsystem makes its final decision independently
    
    This design ensures graceful degradation if any subsystem fails.
    """
    
    def __init__(
        self,
        config: FusionConfig | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.config = config or FusionConfig()
        self._log_dir = Path(log_dir) if log_dir else None
        self._decision_log: list[FusionDecision] = []
        self._step_count = 0
        
        # State for pre-emptive coordination
        self._pre_emptive_active = False
        self._pre_emptive_start_time: float | None = None
        self._last_forecast: dict[str, Any] | None = None
        
        # Signal priority hints (from routing to signals)
        self._signal_priority_hints: dict[str, str] = {}  # tls_id → "extend_green" | "reduce_green"
        
        # Statistics
        self._stats = {
            "total_decisions": 0,
            "pre_emptive_triggers": 0,
            "emergency_overrides": 0,
            "coordination_hints_sent": 0,
            "forecast_calls": 0,
            "routing_calls": 0,
            "signal_calls": 0,
        }
    
    def step(
        self,
        sim_time: float,
        rsu_id: str,
        *,
        # Phase 2 inputs (forecast)
        forecast_result: dict[str, Any] | None = None,
        # Phase 3 inputs (routing)
        routing_result: dict[str, Any] | None = None,
        vehicle_ids: list[str] | None = None,
        emergency_vehicle_ids: list[str] | None = None,
        # Phase 4 inputs (signals)
        signal_result: dict[str, Any] | None = None,
        # Optional context
        context: dict[str, Any] | None = None,
    ) -> FusionDecision:
        """Execute one fusion decision cycle.
        
        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds
        rsu_id : str
            RSU/junction identifier
        forecast_result : dict, optional
            Phase 2 forecast output (p_congestion, confidence, model)
        routing_result : dict, optional
            Phase 3 routing decision (risk_level, reroute_fraction, directives)
        vehicle_ids : list[str], optional
            Vehicle IDs in the RSU coverage area
        emergency_vehicle_ids : list[str], optional
            Emergency vehicle IDs (triggers override)
        signal_result : dict, optional
            Phase 4 signal control output (actions, rewards)
        context : dict, optional
            Additional context (scenario, seed, etc.)
        
        Returns
        -------
        FusionDecision
            Coordinated decision with actions and metadata
        """
        t0 = time.perf_counter()
        self._step_count += 1
        
        decision = FusionDecision(
            timestamp=sim_time,
            rsu_id=rsu_id,
            mode=self.config.mode.value,
        )
        
        # Track active subsystems
        subsystems = []
        
        # ── Phase 2: Process forecast ─────────────────────────────────────
        if self.config.forecast_enabled and forecast_result is not None:
            self._stats["forecast_calls"] += 1
            subsystems.append("forecast")
            
            decision.p_congestion = forecast_result.get("p_congestion")
            decision.forecast_confidence = forecast_result.get("confidence")
            decision.forecast_model = forecast_result.get("model")
            
            self._last_forecast = forecast_result
            
            # Check pre-emptive trigger
            if (
                decision.p_congestion is not None
                and decision.forecast_confidence is not None
                and decision.p_congestion >= self.config.pre_emptive_threshold
                and decision.forecast_confidence >= self.config.confidence_threshold
            ):
                if not self._pre_emptive_active:
                    self._pre_emptive_active = True
                    self._pre_emptive_start_time = sim_time
                    self._stats["pre_emptive_triggers"] += 1
                decision.pre_emptive_triggered = True
        
        # Deactivate pre-emptive if congestion subsides
        if (
            self._pre_emptive_active
            and decision.p_congestion is not None
            and decision.p_congestion < self.config.pre_emptive_threshold * 0.7
        ):
            self._pre_emptive_active = False
            self._pre_emptive_start_time = None
        
        # ── Emergency override check ──────────────────────────────────────
        if (
            self.config.emergency_override
            and emergency_vehicle_ids
            and len(emergency_vehicle_ids) > 0
        ):
            decision.emergency_override_active = True
            self._stats["emergency_overrides"] += 1
        
        # ── Phase 3: Process routing ──────────────────────────────────────
        if self.config.routing_enabled and routing_result is not None:
            self._stats["routing_calls"] += 1
            subsystems.append("routing")
            
            decision.risk_level = routing_result.get("risk_level")
            decision.risk_score = routing_result.get("phase3", {}).get("risk_score")
            
            recommended = routing_result.get("recommended_action", {})
            decision.reroute_fraction = recommended.get("reroute_fraction")
            decision.reroute_mode = recommended.get("reroute_mode")
            decision.route_directives = routing_result.get("route_directives", [])
            
            # Generate coordination hints for signal control
            if self.config.coordination_enabled:
                self._generate_signal_hints(decision, routing_result)
        
        # ── Phase 4: Process signal control ───────────────────────────────
        if self.config.signal_enabled and signal_result is not None:
            self._stats["signal_calls"] += 1
            subsystems.append("signals")
            
            decision.signal_actions = signal_result.get("actions", {})
            decision.signal_rewards = signal_result.get("rewards", {})
        
        # ── Finalize decision ─────────────────────────────────────────────
        decision.subsystems_active = subsystems
        decision.coordination_hints = dict(self._signal_priority_hints)
        decision.decision_time_ms = (time.perf_counter() - t0) * 1000
        
        self._stats["total_decisions"] += 1
        
        # Log decision
        if self.config.log_decisions:
            self._decision_log.append(decision)
        
        return decision
    
    def _generate_signal_hints(
        self,
        decision: FusionDecision,
        routing_result: dict[str, Any],
    ) -> None:
        """Generate signal priority hints based on routing decisions.
        
        The hints are soft suggestions, not hard constraints. Signal control
        can choose to follow or ignore based on local traffic conditions.
        
        Hint types:
        - "extend_green": Extend green phase for incoming vehicles
        - "reduce_green": Reduce green to clear congestion faster
        - "preempt": Emergency vehicle approaching (hard constraint)
        """
        self._signal_priority_hints.clear()
        
        # Emergency preemption (hard constraint)
        if decision.emergency_override_active:
            # Signal preemption for emergency corridor
            # In a real system, this would identify specific junctions
            self._signal_priority_hints["emergency"] = "preempt"
            self._stats["coordination_hints_sent"] += 1
            return
        
        # Pre-emptive coordination when high congestion forecast
        if decision.pre_emptive_triggered:
            risk_level = decision.risk_level or "low"
            reroute_fraction = decision.reroute_fraction or 0.0
            
            if risk_level == "high" and reroute_fraction > 0.2:
                # High reroute → extend green on alternate routes
                self._signal_priority_hints["strategy"] = "extend_alternates"
                self._signal_priority_hints["reason"] = "high_reroute"
            elif risk_level == "medium":
                # Medium risk → balance green times
                self._signal_priority_hints["strategy"] = "balanced"
                self._signal_priority_hints["reason"] = "medium_risk"
            
            if self._signal_priority_hints:
                self._stats["coordination_hints_sent"] += 1
    
    def get_signal_hint(self, tls_id: str) -> str | None:
        """Get the current signal hint for a specific junction.
        
        Parameters
        ----------
        tls_id : str
            Traffic light system ID
        
        Returns
        -------
        str | None
            Hint string ("extend_green", "reduce_green", "preempt") or None
        """
        if "emergency" in self._signal_priority_hints:
            return "preempt"
        return self._signal_priority_hints.get("strategy")
    
    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "config": {
                "mode": self.config.mode.value,
                "forecast_enabled": self.config.forecast_enabled,
                "routing_enabled": self.config.routing_enabled,
                "signal_enabled": self.config.signal_enabled,
                "coordination_enabled": self.config.coordination_enabled,
                "pre_emptive_threshold": self.config.pre_emptive_threshold,
            },
            "stats": dict(self._stats),
            "decision_count": len(self._decision_log),
            "pre_emptive_active": self._pre_emptive_active,
        }
    
    def get_decision_log(self) -> list[dict[str, Any]]:
        """Return all logged decisions as dictionaries."""
        return [d.to_dict() for d in self._decision_log]
    
    def save_decision_log(self, path: str | Path) -> None:
        """Save decision log to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            "summary": self.summary(),
            "decisions": self.get_decision_log(),
        }
        path.write_text(json.dumps(log_data, indent=2, default=str))
    
    def reset(self) -> None:
        """Reset orchestrator state for a new episode."""
        self._decision_log.clear()
        self._step_count = 0
        self._pre_emptive_active = False
        self._pre_emptive_start_time = None
        self._last_forecast = None
        self._signal_priority_hints.clear()
        
        # Keep cumulative stats across episodes
        # Reset per-episode stats if needed
