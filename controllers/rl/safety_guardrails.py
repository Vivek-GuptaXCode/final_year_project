"""Safety guardrails for adaptive traffic signal control.

Enforces hard constraints that must never be violated:
  - Minimum green time before a phase switch is permitted.
  - Mandatory yellow-transition window between conflicting green phases.
  - Anti-oscillation: limits switch rate within a rolling time window.

All constraint parameters are configurable.  The guardrail sits between the RL
policy and the TraCI setPhase call so that no policy — trained or rule-based —
can violate physical signal safety.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailConfig:
    min_green_seconds: float = 15.0
    """Minimum duration a phase must remain active before a switch is allowed."""

    yellow_duration_seconds: float = 3.0
    """Mandatory yellow window inserted between green→green phase transitions.
    During this window the guardrail returns KEEP so SUMO sees no change; the
    yellow state is tracked internally and applied by the inference hook via a
    direct state-string override if enabled."""

    anti_oscillation_window_seconds: float = 60.0
    """Rolling window used to count recent switches."""

    max_switches_per_window: int = 4
    """Maximum allowed switches within the rolling window above."""


@dataclass
class _JunctionState:
    phase: int = 0
    phase_start_time: float = 0.0
    switch_times: deque = field(default_factory=deque)
    in_yellow_until: float = -1.0
    yellow_target_phase: int = 0
    violations_blocked: int = 0


class TLSSafetyGuardrail:
    """Per-junction safety enforcement layer.

    Usage pattern (inference mode — called once per SUMO step per junction):

        safe_action = guardrail.filter_action(tls_id, raw_action, sim_time)
        if safe_action == 1:
            next_phase = (current_phase + 1) % n_phases
            guardrail.record_switch(tls_id, sim_time, next_phase)
            traci.trafficlight.setPhase(tls_id, next_phase)

    Yellow transition:
        If yellow_duration_seconds > 0 the guardrail internally tracks a yellow
        window.  During that window filter_action() returns 0 (KEEP) regardless
        of the policy output so the junction holds until yellow expires.
        The calling code is responsible for setting the yellow state string while
        is_in_yellow(tls_id, sim_time) is True.
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self._cfg = config or GuardrailConfig()
        self._states: dict[str, _JunctionState] = {}

    def _get(self, tls_id: str) -> _JunctionState:
        if tls_id not in self._states:
            self._states[tls_id] = _JunctionState()
        return self._states[tls_id]

    # ── Public API ─────────────────────────────────────────────────────────

    def init_junction(self, tls_id: str, initial_phase: int, sim_time: float) -> None:
        """Register a junction at simulation start (optional but recommended)."""
        st = self._get(tls_id)
        st.phase = initial_phase
        st.phase_start_time = sim_time

    def filter_action(self, tls_id: str, requested_action: int, sim_time: float) -> int:
        """Return safe action (0 = KEEP, 1 = SWITCH).

        The guardrail may downgrade a SWITCH request to KEEP when any constraint
        would be violated.  A KEEP request is always passed through.
        """
        if requested_action == 0:
            return 0

        st = self._get(tls_id)

        # 1. Yellow window still active → must keep.
        if sim_time < st.in_yellow_until:
            return 0

        # 2. If yellow just expired, transition to target phase is implicit;
        #    the caller should have already set that phase.  Nothing to block.

        # 3. Minimum green enforcement.
        elapsed = sim_time - st.phase_start_time
        if elapsed < self._cfg.min_green_seconds:
            st.violations_blocked += 1
            return 0

        # 4. Anti-oscillation: purge stale entries, then check count.
        window_start = sim_time - self._cfg.anti_oscillation_window_seconds
        while st.switch_times and st.switch_times[0] < window_start:
            st.switch_times.popleft()
        if len(st.switch_times) >= self._cfg.max_switches_per_window:
            st.violations_blocked += 1
            return 0

        return 1

    def record_switch(
        self, tls_id: str, sim_time: float, new_phase: int, *, insert_yellow: bool = True
    ) -> None:
        """Call this *after* deciding to switch but *before* calling TraCI.

        If insert_yellow is True, the guardrail starts a yellow window; the
        caller should apply the yellow state string to TraCI during this window
        (see is_in_yellow / yellow_target_phase).
        """
        st = self._get(tls_id)
        st.switch_times.append(sim_time)
        if insert_yellow and self._cfg.yellow_duration_seconds > 0:
            st.in_yellow_until = sim_time + self._cfg.yellow_duration_seconds
            st.yellow_target_phase = new_phase
        else:
            st.phase = new_phase
            st.phase_start_time = sim_time

    def complete_yellow(self, tls_id: str, sim_time: float) -> None:
        """Call after the yellow window expires to officially start new phase."""
        st = self._get(tls_id)
        st.phase = st.yellow_target_phase
        st.phase_start_time = sim_time
        st.in_yellow_until = -1.0

    def is_in_yellow(self, tls_id: str, sim_time: float) -> bool:
        """True while a yellow transition is active for this junction."""
        return sim_time < self._get(tls_id).in_yellow_until

    def has_pending_yellow(self, tls_id: str) -> bool:
        """True when a yellow transition has been scheduled and not completed."""
        return self._get(tls_id).in_yellow_until >= 0.0

    def yellow_expires_at(self, tls_id: str) -> float:
        """Return the simulation timestamp when the active yellow window expires."""
        return float(self._get(tls_id).in_yellow_until)

    def yellow_target_phase(self, tls_id: str) -> int:
        """The green phase that follows the current yellow window."""
        return self._get(tls_id).yellow_target_phase

    def record_phase_start(self, tls_id: str, phase: int, sim_time: float) -> None:
        """Sync guardrail state when SUMO changes phase externally."""
        st = self._get(tls_id)
        st.phase = phase
        st.phase_start_time = sim_time

    def violations_blocked(self, tls_id: str) -> int:
        """Cumulative count of SWITCH requests blocked by this guardrail."""
        return self._get(tls_id).violations_blocked

    def diagnostics(self, tls_id: str, sim_time: float) -> dict[str, Any]:
        st = self._get(tls_id)
        window_start = sim_time - self._cfg.anti_oscillation_window_seconds
        recent = sum(1 for t in st.switch_times if t >= window_start)
        return {
            "tls_id": tls_id,
            "current_phase": st.phase,
            "phase_elapsed_s": round(sim_time - st.phase_start_time, 2),
            "in_yellow": self.is_in_yellow(tls_id, sim_time),
            "recent_switches": recent,
            "violations_blocked": st.violations_blocked,
            "min_green_s": self._cfg.min_green_seconds,
            "max_switches_per_window": self._cfg.max_switches_per_window,
        }
