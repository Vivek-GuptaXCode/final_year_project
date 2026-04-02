"""Phase 5: Hybrid Fusion Controller.

Combines forecasting (Phase 2), risk-aware routing (Phase 3), and adaptive
signal control (Phase 4) into a coordinated pre-emptive traffic management system.

Architecture:
- Event-triggered fusion: Forecast triggers pre-emptive action
- Soft coordination: Signal timing hints to routing, not hard constraints
- Graceful degradation: Each subsystem can operate independently
"""
from controllers.fusion.fusion_orchestrator import (
    FusionOrchestrator,
    FusionConfig,
    FusionDecision,
    FusionMode,
)
from controllers.fusion.ablation_configs import (
    AblationConfig,
    ABLATION_PRESETS,
)

__all__ = [
    "FusionOrchestrator",
    "FusionConfig", 
    "FusionDecision",
    "FusionMode",
    "AblationConfig",
    "ABLATION_PRESETS",
]
