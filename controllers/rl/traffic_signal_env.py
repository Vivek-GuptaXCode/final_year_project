"""Gymnasium-compatible traffic signal environment backed by SUMO TraCI.

This module defines TrafficSignalEnv — a per-junction RL environment that
wraps SUMO's TraCI interface with a standard (obs, reward, done, info) API,
no gymnasium package required.

Observation vector (fixed length = OBS_DIM = 42)
-------------------------------------------------
  [0 : MAX_PHASES]              — current phase one-hot (8 dims)
  [MAX_PHASES]                  — phase elapsed normalised (÷ 120 s)
  [MAX_PHASES+1 : +MAX_LANES]   — per-lane halting count normalised (÷ 20 veh)
  [+MAX_LANES : +2*MAX_LANES]   — per-lane occupancy (%)  normalised (÷ 100)
  [-1]                          — emergency vehicle on any controlled lane (0/1)

Total: 8 + 1 + 16 + 16 + 1 = 42

Action space
------------
  0 — KEEP: maintain current phase.
  1 — SWITCH: advance to (current_phase + 1) % n_phases.

Reward
------
  r = −(total_halting / max_expected_halting) + throughput_bonus
  where max_expected_halting = MAX_LANES * 20.

  throughput_bonus = vehicles_cleared_step / 10.0  (counts passed detectors)

Usage modes
-----------
  Training mode (owns simulationStep):
      env.reset()         — starts/resets SUMO episode
      obs, r, done, info = env.step(action)  — applies action + calls simulationStep
      env.close()         — closes SUMO

  Inference mode (step loop owned by run_sumo_pipeline.py):
      obs = env.observe(sim_time)        — read current state
      env.apply_action(action, sim_time) — apply TLS change if safe
      r   = env.compute_reward()         — evaluate (for logging only)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from controllers.rl.safety_guardrails import GuardrailConfig, TLSSafetyGuardrail


MAX_PHASES = 8
MAX_LANES = 16
ELAPSED_SCALE_S = 120.0
HALTING_SCALE = 20.0   # vehicles per lane normalisation
OBS_DIM = MAX_PHASES + 1 + MAX_LANES + MAX_LANES + 1  # = 42


@dataclass
class EnvConfig:
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    reward_halting_weight: float = 1.0
    reward_throughput_weight: float = 0.5
    reward_waiting_time_weight: float = 0.0  # optional secondary signal
    max_episode_steps: int = 1200
    sumo_step_length_s: float = 1.0


class TrafficSignalEnv:
    """Single-junction RL environment.

    Parameters
    ----------
    traci_module : The TraCI/libsumo module returned by SumoAdapter._traci.
    tls_id       : SUMO traffic light ID (string).
    config       : EnvConfig instance; defaults constructed if None.
    """

    # Class-level constants (also importable by other modules)
    OBS_DIM: int = OBS_DIM
    N_ACTIONS: int = 2

    def __init__(
        self,
        traci_module: Any,
        tls_id: str,
        config: EnvConfig | None = None,
    ) -> None:
        self._traci = traci_module
        self.tls_id = tls_id
        self._cfg = config or EnvConfig()
        self._guardrail = TLSSafetyGuardrail(self._cfg.guardrail)

        # Discovered at first observe() call (deferred so env can be created
        # before SUMO starts — useful for type-checking / unit tests).
        self._incoming_lanes: list[str] = []
        self._n_phases: int = 4
        self._initialized: bool = False

        # Episode bookkeeping
        self._step_count: int = 0
        self._prev_halting: float = 0.0
        self._vehicles_seen: set[str] = set()
        self._program_before_yellow: str | None = None

    # ── Lazy initialisation ───────────────────────────────────────────────

    def _ensure_initialized(self, sim_time: float = 0.0) -> None:
        if self._initialized:
            return

        try:
            self._n_phases = int(self._traci.trafficlight.getPhaseNumber(self.tls_id))
        except Exception:
            try:
                logics = self._traci.trafficlight.getAllProgramLogics(self.tls_id)
                if logics:
                    self._n_phases = len(logics[0].phases)
                else:
                    self._n_phases = 4
            except Exception:
                self._n_phases = 4

        try:
            controlled_links = self._traci.trafficlight.getControlledLinks(self.tls_id)
            seen: set[str] = set()
            lanes: list[str] = []
            for link_group in controlled_links:
                for link in link_group:
                    incoming = str(link[0])
                    if incoming and incoming not in seen:
                        seen.add(incoming)
                        lanes.append(incoming)
            self._incoming_lanes = lanes[:MAX_LANES]
        except Exception:
            self._incoming_lanes = []

        current_phase = 0
        try:
            current_phase = int(self._traci.trafficlight.getPhase(self.tls_id))
        except Exception:
            pass

        self._guardrail.init_junction(self.tls_id, current_phase, sim_time)
        self._initialized = True

    def _phase_elapsed_seconds(self, sim_time: float) -> float:
        """Best-effort phase elapsed time in seconds.

        Preferred source is TraCI getSpentDuration(). Fallback computes elapsed
        from total phase duration and next-switch timestamp when available.
        """
        try:
            spent = float(self._traci.trafficlight.getSpentDuration(self.tls_id))
            if spent >= 0.0:
                return spent
        except Exception:
            pass

        try:
            total = float(self._traci.trafficlight.getPhaseDuration(self.tls_id))
            next_switch = float(self._traci.trafficlight.getNextSwitch(self.tls_id))
            remaining = max(0.0, next_switch - sim_time)
            return max(0.0, total - remaining)
        except Exception:
            pass

        try:
            return max(0.0, float(self._traci.trafficlight.getPhaseDuration(self.tls_id)))
        except Exception:
            return 0.0

    def _apply_yellow_state(self) -> None:
        """Apply an all-yellow transition state for the currently controlled links."""
        try:
            if self._program_before_yellow is None:
                current_program = str(self._traci.trafficlight.getProgram(self.tls_id))
                if current_program and current_program != "online":
                    self._program_before_yellow = current_program
            state_str = self._traci.trafficlight.getRedYellowGreenState(self.tls_id)
            yellow_state = state_str.replace("G", "y").replace("g", "y")
            self._traci.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
        except Exception:
            pass

    def _restore_program_after_yellow(self) -> None:
        """Restore the pre-yellow TLS program after online-state control."""
        if not self._program_before_yellow:
            return
        try:
            current_program = str(self._traci.trafficlight.getProgram(self.tls_id))
            if current_program == "online":
                self._traci.trafficlight.setProgram(self.tls_id, self._program_before_yellow)
        except Exception:
            pass

    def _current_phase_count(self) -> int:
        """Return current active-program phase count with conservative fallback."""
        try:
            return max(1, int(self._traci.trafficlight.getPhaseNumber(self.tls_id)))
        except Exception:
            return max(1, int(self._n_phases))

    # ── Observation ───────────────────────────────────────────────────────

    def observe(self, sim_time: float) -> np.ndarray:
        """Build the fixed-length observation vector at the current sim step."""
        self._ensure_initialized(sim_time)

        # Phase one-hot
        try:
            phase = int(self._traci.trafficlight.getPhase(self.tls_id))
        except Exception:
            phase = 0
        phase_oh = [0.0] * MAX_PHASES
        if 0 <= phase < MAX_PHASES:
            phase_oh[phase] = 1.0

        # Phase elapsed (normalised)
        elapsed_s = self._phase_elapsed_seconds(sim_time)
        elapsed_norm = min(1.0, elapsed_s / ELAPSED_SCALE_S)

        # Per-lane halting + occupancy
        halting = []
        occupancy = []
        for lane in self._incoming_lanes:
            try:
                h = float(self._traci.lane.getLastStepHaltingNumber(lane))
            except Exception:
                h = 0.0
            try:
                occ = float(self._traci.lane.getLastStepOccupancy(lane))
            except Exception:
                occ = 0.0
            halting.append(min(1.0, h / HALTING_SCALE))
            occupancy.append(min(1.0, occ / 100.0))

        # Pad to MAX_LANES
        while len(halting) < MAX_LANES:
            halting.append(0.0)
            occupancy.append(0.0)

        # Emergency flag
        emergency = 0.0
        try:
            for lane in self._incoming_lanes:
                vids = self._traci.lane.getLastStepVehicleIDs(lane)
                for vid in vids:
                    vtype = str(self._traci.vehicle.getTypeID(vid))
                    vclass = str(self._traci.vehicle.getVehicleClass(vid))
                    if "emergency" in vtype or "emergency" in vclass:
                        emergency = 1.0
                        break
                if emergency:
                    break
        except Exception:
            pass

        obs = np.array(
            phase_oh + [elapsed_norm] + halting[:MAX_LANES] + occupancy[:MAX_LANES] + [emergency],
            dtype=np.float32,
        )
        assert obs.shape == (OBS_DIM,), f"obs shape mismatch: {obs.shape}"
        return obs

    # ── Reward ────────────────────────────────────────────────────────────

    def compute_reward(self) -> float:
        """Compute per-step reward from TraCI state."""
        total_halting = 0.0
        try:
            for lane in self._incoming_lanes:
                total_halting += float(self._traci.lane.getLastStepHaltingNumber(lane))
        except Exception:
            pass

        max_halt = max(1.0, len(self._incoming_lanes) * HALTING_SCALE)
        halt_penalty = -(total_halting / max_halt) * self._cfg.reward_halting_weight

        # Throughput bonus: vehicles that left incoming lanes this step
        throughput = 0.0
        if self._cfg.reward_throughput_weight > 0:
            try:
                current_on_lanes: set[str] = set()
                for lane in self._incoming_lanes:
                    for vid in self._traci.lane.getLastStepVehicleIDs(lane):
                        current_on_lanes.add(str(vid))
                throughput = float(len(self._vehicles_seen - current_on_lanes))
                self._vehicles_seen = current_on_lanes
            except Exception:
                pass

        throughput_bonus = (throughput / 10.0) * self._cfg.reward_throughput_weight

        self._prev_halting = total_halting
        return halt_penalty + throughput_bonus

    # ── Action application ────────────────────────────────────────────────

    def apply_action(self, action: int, sim_time: float) -> dict[str, Any]:
        """Apply a KEEP/SWITCH action with safety guardrail enforcement.

        Does NOT call simulationStep() — that is the caller's responsibility.
        Returns info dict with guardrail diagnostics.
        """
        self._ensure_initialized(sim_time)
        switched = False
        completed_yellow = False

        # If a pending yellow has elapsed, return from online control and apply target phase.
        if self._guardrail.has_pending_yellow(self.tls_id) and not self._guardrail.is_in_yellow(self.tls_id, sim_time):
            self._restore_program_after_yellow()
            phase_count = self._current_phase_count()
            self._n_phases = phase_count
            if phase_count > 1:
                target = self._guardrail.yellow_target_phase(self.tls_id) % phase_count
                try:
                    self._traci.trafficlight.setPhase(self.tls_id, target)
                    switched = True
                except Exception:
                    pass
            self._guardrail.complete_yellow(self.tls_id, sim_time)
            self._program_before_yellow = None
            completed_yellow = True

        # While yellow is active we hold and ignore policy switch requests.
        if self._guardrail.is_in_yellow(self.tls_id, sim_time):
            self._apply_yellow_state()
            safe = 0
            return {
                **self._guardrail.diagnostics(self.tls_id, sim_time),
                "action_requested": action,
                "action_safe": safe,
                "switched": switched,
                "completed_yellow": completed_yellow,
            }

        phase_count = self._current_phase_count()
        self._n_phases = phase_count

        # Some network controllers expose a single immutable phase. Treat these
        # nodes as non-switchable to avoid invalid TraCI setPhase() commands.
        if phase_count <= 1:
            return {
                **self._guardrail.diagnostics(self.tls_id, sim_time),
                "action_requested": action,
                "action_safe": 0,
                "switched": switched,
                "completed_yellow": completed_yellow,
                "non_switchable_tls": True,
            }

        safe = self._guardrail.filter_action(self.tls_id, action, sim_time)
        if safe == 1:
            try:
                current = int(self._traci.trafficlight.getPhase(self.tls_id))
                next_phase = (current + 1) % phase_count
                self._guardrail.record_switch(self.tls_id, sim_time, next_phase)

                if self._cfg.guardrail.yellow_duration_seconds > 0:
                    self._apply_yellow_state()
                    switched = True
                else:
                    self._traci.trafficlight.setPhase(self.tls_id, next_phase)
                    switched = True
            except Exception:
                pass

        return {
            **self._guardrail.diagnostics(self.tls_id, sim_time),
            "action_requested": action,
            "action_safe": safe,
            "switched": switched,
            "completed_yellow": completed_yellow,
        }

    # ── Training mode: step() and reset() ────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply action, advance simulation one step, return (obs, r, done, info).

        Only valid in training mode where this env owns the simulationStep.
        """
        sim_time = float(self._traci.simulation.getTime())
        info = self.apply_action(action, sim_time)

        self._traci.simulationStep()
        self._step_count += 1

        new_sim_time = float(self._traci.simulation.getTime())
        obs = self.observe(new_sim_time)
        reward = self.compute_reward()

        done = (
            self._step_count >= self._cfg.max_episode_steps
            or self._traci.simulation.getMinExpectedNumber() <= 0
        )
        info["step"] = self._step_count
        info["sim_time"] = new_sim_time
        return obs, reward, done, info

    def reset(self, sim_time: float = 0.0) -> np.ndarray:
        """Reset episode bookkeeping (SUMO restart is handled by train_phase4.py)."""
        self._step_count = 0
        self._prev_halting = 0.0
        self._vehicles_seen = set()
        self._program_before_yellow = None
        self._initialized = False
        self._ensure_initialized(sim_time)
        return self.observe(sim_time)

    # ── Utilities ─────────────────────────────────────────────────────────

    @property
    def n_phases(self) -> int:
        return self._n_phases

    @property
    def incoming_lanes(self) -> list[str]:
        return list(self._incoming_lanes)

    def guardrail_diagnostics(self, sim_time: float) -> dict[str, Any]:
        return self._guardrail.diagnostics(self.tls_id, sim_time)


class MultiJunctionEnv:
    """Thin wrapper around multiple TrafficSignalEnv instances for MARL.

    Each junction has its own independent state and action.  Coordination is
    implicit: each agent's observation includes the mean congestion of its K
    nearest neighbours (appended as a scalar at obs[-2]).

    Observation dimension per junction: OBS_DIM + 1 = 43
    """

    OBS_DIM: int = OBS_DIM + 1  # +1 for neighbour congestion signal
    N_ACTIONS: int = 2

    def __init__(
        self,
        traci_module: Any,
        tls_ids: list[str],
        config: EnvConfig | None = None,
        neighbour_k: int = 2,
    ) -> None:
        self._traci = traci_module
        self.tls_ids = list(tls_ids)
        self._envs: dict[str, TrafficSignalEnv] = {
            tid: TrafficSignalEnv(traci_module, tid, config) for tid in tls_ids
        }
        self._neighbour_k = neighbour_k
        self._neighbours = self._build_topology_neighbours(self.tls_ids)

    @staticmethod
    def _lane_to_edge_id(lane_id: str) -> str:
        lane = str(lane_id).strip()
        if not lane or lane.startswith(":"):
            return ""
        if "_" in lane:
            return lane.rsplit("_", 1)[0]
        return lane

    def _build_topology_neighbours(self, tls_ids: list[str]) -> dict[str, list[str]]:
        """Build K-hop neighbours from controlled-link edge connectivity.

        Two TLS nodes are adjacent when outgoing edges of one intersect incoming
        edges of the other (in either direction).
        """
        if not tls_ids:
            return {}

        incoming_by_tls: dict[str, set[str]] = {}
        outgoing_by_tls: dict[str, set[str]] = {}
        adjacency: dict[str, set[str]] = {tid: set() for tid in tls_ids}

        for tid in tls_ids:
            incoming_edges: set[str] = set()
            outgoing_edges: set[str] = set()
            try:
                controlled_links = self._traci.trafficlight.getControlledLinks(tid)
            except Exception:
                controlled_links = []

            for link_group in controlled_links:
                for link in link_group:
                    if not link or len(link) < 2:
                        continue
                    in_edge = self._lane_to_edge_id(str(link[0]))
                    out_edge = self._lane_to_edge_id(str(link[1]))
                    if in_edge:
                        incoming_edges.add(in_edge)
                    if out_edge:
                        outgoing_edges.add(out_edge)

            incoming_by_tls[tid] = incoming_edges
            outgoing_by_tls[tid] = outgoing_edges

        for idx, tid in enumerate(tls_ids):
            for other in tls_ids[idx + 1 :]:
                connected = bool(
                    outgoing_by_tls.get(tid, set()) & incoming_by_tls.get(other, set())
                    or outgoing_by_tls.get(other, set()) & incoming_by_tls.get(tid, set())
                )
                if connected:
                    adjacency[tid].add(other)
                    adjacency[other].add(tid)

        neighbours: dict[str, list[str]] = {}
        for tid in tls_ids:
            if self._neighbour_k <= 0:
                neighbours[tid] = []
                continue

            visited = {tid}
            frontier = [tid]
            ordered: list[str] = []

            for _hop in range(self._neighbour_k):
                next_frontier: list[str] = []
                for node in frontier:
                    for nbr in sorted(adjacency.get(node, set())):
                        if nbr in visited:
                            continue
                        visited.add(nbr)
                        ordered.append(nbr)
                        next_frontier.append(nbr)
                if not next_frontier:
                    break
                frontier = next_frontier

            # Keep a deterministic fallback when topology extraction yields isolates.
            if not ordered and len(tls_ids) > 1:
                idx = tls_ids.index(tid)
                ordered = [tls_ids[(idx + 1) % len(tls_ids)]]

            neighbours[tid] = ordered

        return neighbours

    def observe_all(self, sim_time: float) -> dict[str, np.ndarray]:
        """Return per-junction observations augmented with neighbour congestion."""
        base_obs: dict[str, np.ndarray] = {
            tid: self._envs[tid].observe(sim_time) for tid in self.tls_ids
        }
        # Compute mean halting density for each junction
        halt_start = MAX_PHASES + 1
        halt_end = halt_start + MAX_LANES
        congestion: dict[str, float] = {
            tid: float(np.mean(obs[halt_start:halt_end]))
            for tid, obs in base_obs.items()
        }

        augmented: dict[str, np.ndarray] = {}
        for tid, obs in base_obs.items():
            nbrs = self._neighbours.get(tid, [])
            nbr_congestion = float(np.mean([congestion[n] for n in nbrs])) if nbrs else 0.0
            augmented[tid] = np.append(obs, nbr_congestion).astype(np.float32)
        return augmented

    def apply_actions(
        self, actions: dict[str, int], sim_time: float
    ) -> dict[str, dict[str, Any]]:
        """Apply per-junction actions (no simulationStep)."""
        return {tid: self._envs[tid].apply_action(actions[tid], sim_time) for tid in self.tls_ids}

    def compute_rewards(self) -> dict[str, float]:
        return {tid: self._envs[tid].compute_reward() for tid in self.tls_ids}

    def reset_all(self, sim_time: float = 0.0) -> dict[str, np.ndarray]:
        return {tid: self._envs[tid].reset(sim_time) for tid in self.tls_ids}
