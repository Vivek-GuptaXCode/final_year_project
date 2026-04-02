"""Ablation experiment configurations for Phase 5 evaluation.

Provides standardized configurations for systematic ablation studies
to isolate the contribution of each subsystem (forecasting, routing, signals).

Based on best practices from:
- PressLight (KDD'19): Component ablation methodology
- CoLight (CIKM'19): Multi-agent ablation analysis
- RESCO benchmarks: Standardized evaluation protocols
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from controllers.fusion.fusion_orchestrator import FusionConfig, FusionMode


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment.
    
    Parameters
    ----------
    name : str
        Human-readable name for the ablation
    description : str
        Description of what this ablation tests
    fusion_config : FusionConfig
        The fusion controller configuration
    expected_behavior : str
        What behavior we expect to observe
    hypothesis : str
        The hypothesis being tested
    """
    
    name: str
    description: str
    fusion_config: FusionConfig
    expected_behavior: str
    hypothesis: str
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.fusion_config.mode.value,
            "forecast_enabled": self.fusion_config.forecast_enabled,
            "routing_enabled": self.fusion_config.routing_enabled,
            "signal_enabled": self.fusion_config.signal_enabled,
            "coordination_enabled": self.fusion_config.coordination_enabled,
            "expected_behavior": self.expected_behavior,
            "hypothesis": self.hypothesis,
        }


# ── Predefined ablation configurations ────────────────────────────────────────

ABLATION_PRESETS: dict[str, AblationConfig] = {
    # Full system (control)
    "full_hybrid": AblationConfig(
        name="Full Hybrid System",
        description="All subsystems enabled with coordination",
        fusion_config=FusionConfig.from_mode(FusionMode.FULL_HYBRID),
        expected_behavior="Best overall performance through coordinated pre-emptive action",
        hypothesis="Coordination between subsystems provides synergistic benefit beyond individual contributions",
    ),
    
    # Individual subsystem ablations
    "no_forecast": AblationConfig(
        name="No Forecasting (Reactive)",
        description="Routing and signals active, but no congestion forecasting",
        fusion_config=FusionConfig(
            mode=FusionMode.REACTIVE_BASELINE,
            forecast_enabled=False,
            routing_enabled=True,
            signal_enabled=True,
            coordination_enabled=False,  # No forecast → no pre-emptive hints
        ),
        expected_behavior="Reactive-only control, no pre-emptive action possible",
        hypothesis="Forecasting enables pre-emptive action that improves over reactive control",
    ),
    
    "no_routing": AblationConfig(
        name="No Routing",
        description="Forecasting and signals active, but no risk-aware routing",
        fusion_config=FusionConfig(
            mode=FusionMode.SIGNAL_ONLY,
            forecast_enabled=True,
            routing_enabled=False,
            signal_enabled=True,
            coordination_enabled=False,
        ),
        expected_behavior="Signal optimization without traffic redistribution",
        hypothesis="Routing contributes by redistributing traffic across network",
    ),
    
    "no_signals": AblationConfig(
        name="No Adaptive Signals",
        description="Forecasting and routing active, but fixed-time signals",
        fusion_config=FusionConfig(
            mode=FusionMode.ROUTING_ONLY,
            forecast_enabled=True,
            routing_enabled=True,
            signal_enabled=False,
            coordination_enabled=False,
        ),
        expected_behavior="Routing adapts but signals are static",
        hypothesis="Adaptive signals contribute by optimizing intersection throughput",
    ),
    
    "no_coordination": AblationConfig(
        name="No Coordination",
        description="All subsystems active but operating independently (no hints)",
        fusion_config=FusionConfig(
            mode=FusionMode.FULL_HYBRID,
            forecast_enabled=True,
            routing_enabled=True,
            signal_enabled=True,
            coordination_enabled=False,  # Key difference
        ),
        expected_behavior="Each subsystem optimizes locally without shared information",
        hypothesis="Coordination provides additional benefit through information sharing",
    ),
    
    # Baseline comparisons
    "baseline_no_ai": AblationConfig(
        name="No AI (Fixed-Time Baseline)",
        description="No AI systems - fixed-time signals, no rerouting",
        fusion_config=FusionConfig.from_mode(FusionMode.NO_AI),
        expected_behavior="Traditional traffic control without AI enhancement",
        hypothesis="AI systems provide measurable improvement over traditional control",
    ),
    
    "forecast_only": AblationConfig(
        name="Forecast Only",
        description="Only forecasting active (for monitoring, no action)",
        fusion_config=FusionConfig.from_mode(FusionMode.FORECAST_ONLY),
        expected_behavior="Congestion prediction without intervention",
        hypothesis="Baseline to measure forecast accuracy in isolation",
    ),
    
    # Sensitivity analyses
    "low_confidence_threshold": AblationConfig(
        name="Low Confidence Threshold",
        description="Full hybrid with lower confidence threshold (0.4 instead of 0.6)",
        fusion_config=FusionConfig(
            mode=FusionMode.FULL_HYBRID,
            forecast_enabled=True,
            routing_enabled=True,
            signal_enabled=True,
            coordination_enabled=True,
            confidence_threshold=0.40,  # Lower threshold
        ),
        expected_behavior="More aggressive pre-emptive action, possibly false positives",
        hypothesis="Lower threshold trades precision for recall in pre-emptive action",
    ),
    
    "high_preemptive_threshold": AblationConfig(
        name="High Pre-emptive Threshold",
        description="Full hybrid with higher pre-emptive threshold (0.85 instead of 0.7)",
        fusion_config=FusionConfig(
            mode=FusionMode.FULL_HYBRID,
            forecast_enabled=True,
            routing_enabled=True,
            signal_enabled=True,
            coordination_enabled=True,
            pre_emptive_threshold=0.85,  # Higher threshold
        ),
        expected_behavior="More conservative pre-emptive action, potentially late intervention",
        hypothesis="Higher threshold trades early action for certainty",
    ),
}


def get_ablation_suite() -> list[AblationConfig]:
    """Return the standard ablation suite for Phase 5 evaluation.
    
    The suite is ordered for systematic evaluation:
    1. Full system (control)
    2. Individual subsystem ablations
    3. Coordination ablation
    4. Baseline comparisons
    """
    return [
        ABLATION_PRESETS["full_hybrid"],
        ABLATION_PRESETS["no_forecast"],
        ABLATION_PRESETS["no_routing"],
        ABLATION_PRESETS["no_signals"],
        ABLATION_PRESETS["no_coordination"],
        ABLATION_PRESETS["baseline_no_ai"],
    ]


def get_sensitivity_suite() -> list[AblationConfig]:
    """Return the sensitivity analysis suite."""
    return [
        ABLATION_PRESETS["low_confidence_threshold"],
        ABLATION_PRESETS["high_preemptive_threshold"],
    ]
