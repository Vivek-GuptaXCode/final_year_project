"""Baseline traffic signal policies for Phase 4 comparison.

Two baselines:
  FixedTimePolicy   — rotates through phases on a fixed cycle regardless of
                      queue state.  Represents pre-installed timer control.
  SimpleActuatedPolicy — extends the current phase if queue exceeds a threshold;
                         switches early if lanes are empty.  Represents standard
                         SCATS-style vehicle-actuated control.

Both share the same interface as the DQN policy:
    action = policy.select_action(obs, tls_id, sim_time)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from controllers.rl.traffic_signal_env import (
    CURRENT_PHASE_QUEUE_IDX,
    MAX_LANES,
    MAX_PHASES,
    NEXT_PHASE_QUEUE_IDX,
    PHASE_ELAPSED_IDX,
    QUEUE_END_IDX,
    QUEUE_START_IDX,
)


@dataclass
class FixedTimePolicy:
    """Rotates phases on a fixed wall-clock cycle.

    Args:
        cycle_seconds: Total cycle length (shared equally among phases).
        n_phases: Number of phases in the TLS program.
    """

    cycle_seconds: float = 90.0
    n_phases: int = 4
    _phase_duration: float = field(init=False)
    _last_switch: dict[str, float] = field(default_factory=dict, init=False)
    _current_phase: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._phase_duration = max(1.0, self.cycle_seconds / max(1, self.n_phases))

    def select_action(
        self,
        obs: np.ndarray,
        tls_id: str,
        sim_time: float,
        *,
        n_phases: int | None = None,
    ) -> int:
        """Return 0 (KEEP) or 1 (SWITCH).

        obs is accepted for interface compatibility but not used.
        """
        effective_n = n_phases if n_phases is not None else self.n_phases
        duration = max(1.0, self.cycle_seconds / max(1, effective_n))

        last = self._last_switch.get(tls_id, -1e9)
        if sim_time - last >= duration:
            self._last_switch[tls_id] = sim_time
            return 1
        return 0

    def reset(self, tls_id: str, sim_time: float) -> None:
        self._last_switch[tls_id] = sim_time
        self._current_phase[tls_id] = 0


@dataclass
class SimpleActuatedPolicy:
    """Vehicle-actuated policy: extend green if queue is high, switch if clear.

    Decision logic per call:
      1. If halting density on current-phase lanes exceeds `extend_threshold`
         AND phase has not exceeded `max_green_seconds` → KEEP (extend green).
      2. If halting density on cross-phase lanes exceeds `switch_threshold`
         AND current phase has run at least `min_green_seconds` → SWITCH.
      3. Otherwise → KEEP.

    The observation vector must follow the TrafficSignalEnv format:
        obs[0:MAX_PHASES]                  — phase one-hot
        obs[PHASE_ELAPSED_IDX]             — phase elapsed (normalised 0-1, scale=120 s)
        obs[QUEUE_START_IDX:QUEUE_END_IDX] — lane queue densities
    """

    min_green_seconds: float = 15.0
    max_green_seconds: float = 90.0
    extend_threshold: float = 0.30
    switch_threshold: float = 0.20
    # Constants matching TrafficSignalEnv defaults
    max_phases: int = MAX_PHASES
    max_lanes: int = MAX_LANES
    elapsed_scale_seconds: float = 120.0

    def select_action(
        self,
        obs: np.ndarray,
        tls_id: str,
        sim_time: float,
        *,
        n_phases: int | None = None,
    ) -> int:
        # Decode observation
        phase_onehot = obs[: self.max_phases]
        current_phase = int(np.argmax(phase_onehot))
        elapsed_s = float(obs[PHASE_ELAPSED_IDX]) * self.elapsed_scale_seconds
        halting = obs[QUEUE_START_IDX:QUEUE_END_IDX]

        effective_n = n_phases if n_phases is not None else self.max_phases
        lanes_per_phase = max(1, self.max_lanes // max(1, effective_n))

        # Lanes assigned to current phase (rough even split)
        start_idx = current_phase * lanes_per_phase
        end_idx = start_idx + lanes_per_phase
        current_phase_density = float(np.mean(halting[start_idx:end_idx]))

        # Cross-phase density (all other lanes)
        mask = np.ones(self.max_lanes, dtype=bool)
        mask[start_idx:end_idx] = False
        cross_density = float(np.mean(halting[mask]))

        # Decision
        if elapsed_s < self.min_green_seconds:
            return 0  # min green not satisfied

        if elapsed_s < self.max_green_seconds and current_phase_density >= self.extend_threshold:
            return 0  # extend green — current approach still congested

        if cross_density >= self.switch_threshold:
            return 1  # cross-phase demand warrants switch

        return 0

    def reset(self, tls_id: str, sim_time: float) -> None:  # noqa: ARG002
        pass  # stateless per call


@dataclass
class MaxPressurePolicy:
    """Binary max-pressure-style baseline for current-vs-next phase control.

    This adapts max-pressure intuition to the repo's binary action space:
    keep the current phase unless the next phase queue clearly dominates.
    """

    min_green_seconds: float = 15.0
    max_green_seconds: float = 90.0
    switch_margin: float = 0.03
    max_phases: int = MAX_PHASES
    max_lanes: int = MAX_LANES
    elapsed_scale_seconds: float = 120.0

    def _phase_queue_pair(
        self,
        obs: np.ndarray,
        current_phase: int,
        effective_n: int,
    ) -> tuple[float, float]:
        if obs.shape[0] > NEXT_PHASE_QUEUE_IDX:
            return float(obs[CURRENT_PHASE_QUEUE_IDX]), float(obs[NEXT_PHASE_QUEUE_IDX])

        halting = obs[QUEUE_START_IDX:QUEUE_END_IDX]
        lanes_per_phase = max(1, self.max_lanes // max(1, effective_n))
        next_phase = (current_phase + 1) % effective_n

        current_start = current_phase * lanes_per_phase
        current_end = current_start + lanes_per_phase
        next_start = next_phase * lanes_per_phase
        next_end = next_start + lanes_per_phase

        current_queue = float(np.mean(halting[current_start:current_end]))
        next_queue = float(np.mean(halting[next_start:next_end]))
        return current_queue, next_queue

    def select_action(
        self,
        obs: np.ndarray,
        tls_id: str,
        sim_time: float,
        *,
        n_phases: int | None = None,
    ) -> int:
        phase_onehot = obs[: self.max_phases]
        current_phase = int(np.argmax(phase_onehot))
        elapsed_s = float(obs[PHASE_ELAPSED_IDX]) * self.elapsed_scale_seconds
        effective_n = max(1, min(self.max_phases, n_phases if n_phases is not None else self.max_phases))
        current_queue, next_queue = self._phase_queue_pair(obs, current_phase, effective_n)

        if elapsed_s < self.min_green_seconds:
            return 0
        if next_queue > current_queue + self.switch_margin:
            return 1
        if elapsed_s >= self.max_green_seconds and next_queue > 0.01:
            return 1
        return 0

    def reset(self, tls_id: str, sim_time: float) -> None:  # noqa: ARG002
        pass


def make_baseline(name: str, **kwargs: Any) -> FixedTimePolicy | SimpleActuatedPolicy | MaxPressurePolicy:
    """Factory for named baselines used in train_phase4.py evaluations."""
    if name == "fixed_time":
        return FixedTimePolicy(**kwargs)
    if name == "simple_actuated":
        return SimpleActuatedPolicy(**kwargs)
    if name == "max_pressure":
        return MaxPressurePolicy(**kwargs)
    raise ValueError(
        f"Unknown baseline: {name!r}. Choose 'fixed_time', 'simple_actuated', or 'max_pressure'."
    )
