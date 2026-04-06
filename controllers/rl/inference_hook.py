"""RLSignalController — inference-mode RL hook for run_sumo_pipeline.py.

Plugs into the _on_step() callback of run_sumo_pipeline.py to apply
per-junction RL signal control without owning the simulationStep.

Usage in run_sumo_pipeline.py _on_step():

    controller = RLSignalController.from_args(args, traci_module)
    # Inside _on_step():
    if controller is not None:
        controller.step(sim_time, traci_module)

The controller:
  1. Discovers controllable TLS junctions on first call.
  2. Loads pre-trained DQN weights if a model path is provided; otherwise
     falls back to SimpleActuatedPolicy (reasonable default with no training).
  3. Applies SafetyGuardrail-filtered actions to each junction every call.
  4. Emits diagnostic logs at configurable intervals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from controllers.rl.dqn_agent import DQNAgent
from controllers.rl.improved_dqn_agent import ImprovedDQNAgent
from controllers.rl.baselines import SimpleActuatedPolicy
from controllers.rl.safety_guardrails import GuardrailConfig, TLSSafetyGuardrail
from controllers.rl.traffic_signal_env import (
    MAX_PHASES,
    MAX_LANES,
    OBS_DIM,
    EnvConfig,
    MultiJunctionEnv,
    TrafficSignalEnv,
)


RLPolicy = DQNAgent | ImprovedDQNAgent | SimpleActuatedPolicy


class RLSignalController:
    """Manages per-junction RL agents in inference (no-training) mode.

    Parameters
    ----------
    traci_module   : TraCI/libsumo module.
    tls_ids        : Explicit list of junction IDs to control. If empty,
                     auto-discovers all TLS junctions at first step.
    model_dir      : Path to DQN weights directory produced by train_phase4.py.
                     If None or path does not exist, falls back to SimpleActuated.
    guardrail_cfg  : Safety guardrail configuration.
    log_interval   : Print diagnostics every N sim steps.
    neighbour_k    : Neighbours each agent sees for MARL coordination.
    """

    def __init__(
        self,
        traci_module: Any,
        *,
        tls_ids: list[str] | None = None,
        model_dir: str | Path | None = None,
        guardrail_cfg: GuardrailConfig | None = None,
        log_interval: int = 100,
        neighbour_k: int = 2,
        max_controlled_tls: int = 96,
        step_interval_steps: int = 2,
    ) -> None:
        self._traci = traci_module
        self._tls_ids_override = tls_ids or []
        self._model_dir = Path(model_dir) if model_dir else None
        self._guardrail_cfg = guardrail_cfg or GuardrailConfig()
        self._log_interval = log_interval
        self._neighbour_k = neighbour_k
        self._max_controlled_tls = int(max_controlled_tls)
        self._step_interval_steps = max(1, int(step_interval_steps))

        self._env: MultiJunctionEnv | None = None
        self._agents: dict[str, RLPolicy] = {}
        self._initialized = False
        self._step_count = 0
        self._decision_steps = 0
        self._total_switches = 0
        self._cumulative_reward: dict[str, float] = {}

    def _rank_tls_by_incoming_lanes(self, tls_ids: list[str]) -> list[str]:
        scored: list[tuple[int, str]] = []
        for tid in tls_ids:
            score = 0
            try:
                controlled_links = self._traci.trafficlight.getControlledLinks(tid)
                incoming_lanes: set[str] = set()
                for link_group in controlled_links:
                    for link in link_group:
                        incoming = str(link[0])
                        if incoming:
                            incoming_lanes.add(incoming)
                score = len(incoming_lanes)
            except Exception:
                score = 0
            scored.append((score, tid))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [tid for _score, tid in scored]

    def _load_saved_agent(self) -> DQNAgent | ImprovedDQNAgent:
        if self._model_dir is None:
            raise FileNotFoundError("model_dir is not configured")

        run_dir = self._model_dir / "latest"
        meta_path = run_dir / "meta.json"
        agent_version = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                agent_version = str(meta.get("agent_version", "")).lower()
            except Exception:
                agent_version = ""

        if agent_version.startswith("improved_dqn"):
            return ImprovedDQNAgent.load(self._model_dir, run_id="latest")

        try:
            return DQNAgent.load(self._model_dir, run_id="latest")
        except Exception:
            return ImprovedDQNAgent.load(self._model_dir, run_id="latest")

    # ── Lazy initialisation ───────────────────────────────────────────────

    def _initialize(self, sim_time: float) -> None:
        if self._initialized:
            return

        # Discover TLS junctions
        if self._tls_ids_override:
            tls_ids = self._tls_ids_override
        else:
            try:
                tls_ids = list(self._traci.trafficlight.getIDList())
            except Exception:
                tls_ids = []

        if (
            self._max_controlled_tls > 0
            and not self._tls_ids_override
            and len(tls_ids) > self._max_controlled_tls
        ):
            ranked_ids = self._rank_tls_by_incoming_lanes(tls_ids)
            original_count = len(tls_ids)
            tls_ids = ranked_ids[: self._max_controlled_tls]
            print(
                "[RL] Large-map optimization: limiting TLS control to "
                f"{len(tls_ids)}/{original_count} junctions (ranked by incoming lane coverage)."
            )

        if not tls_ids:
            print("[RL] No TLS junctions found — signal control disabled.")
            self._initialized = True
            return

        print(f"[RL] Controlling {len(tls_ids)} TLS junction(s): {tls_ids[:10]}{'...' if len(tls_ids) > 10 else ''}")

        env_cfg = EnvConfig(guardrail=self._guardrail_cfg)
        self._env = MultiJunctionEnv(
            self._traci, tls_ids, config=env_cfg, neighbour_k=self._neighbour_k
        )
        self._env.reset_all(sim_time)

        # Load agents
        use_saved_model = self._model_dir is not None and (self._model_dir / "latest" / "weights.npz").exists()
        loaded_agent_name = "SimpleActuated"

        for tid in tls_ids:
            if use_saved_model:
                try:
                    agent = self._load_saved_agent()
                    agent.epsilon = 0.0  # pure greedy in inference
                    self._agents[tid] = agent
                    loaded_agent_name = type(agent).__name__
                except Exception as exc:
                    print(f"[RL] Failed to load RL weights for {tid}: {exc} — using SimpleActuated fallback")
                    self._agents[tid] = SimpleActuatedPolicy()
            else:
                self._agents[tid] = SimpleActuatedPolicy()

        policy_name = loaded_agent_name if use_saved_model else "SimpleActuated(fallback)"
        print(f"[RL] Policy: {policy_name}")
        self._cumulative_reward = {tid: 0.0 for tid in tls_ids}
        self._initialized = True

    # ── Per-step call ─────────────────────────────────────────────────────

    def step(self, sim_time: float, traci_module: Any | None = None) -> dict[str, Any]:
        """Called once per SUMO step from _on_step().  Applies TLS actions.

        Returns per-junction action/reward info dict (for logging).
        """
        if traci_module is not None:
            self._traci = traci_module

        self._initialize(sim_time)

        if self._env is None:
            return {}

        self._step_count += 1
        if self._step_interval_steps > 1 and self._step_count % self._step_interval_steps != 0:
            return {
                "step": self._step_count,
                "sim_time": sim_time,
                "skipped": True,
            }

        obs_map = self._env.observe_all(sim_time)
        tls_ids = self._env.tls_ids

        actions: dict[str, int] = {}
        for tid in tls_ids:
            agent = self._agents.get(tid)
            if agent is None:
                actions[tid] = 0
            elif isinstance(agent, (DQNAgent, ImprovedDQNAgent)):
                obs = obs_map[tid]
                # DQN was trained on OBS_DIM; MultiJunctionEnv adds 1 dim
                if obs.shape[0] == OBS_DIM + 1:
                    # Trim neighbour signal if model was trained without it
                    obs_input = obs[:OBS_DIM] if agent.obs_dim == OBS_DIM else obs
                else:
                    obs_input = obs
                actions[tid] = agent.select_action(obs_input, greedy=True)
            else:
                # SimpleActuatedPolicy
                actions[tid] = agent.select_action(
                    obs_map[tid][:OBS_DIM],  # strip neighbour dim if present
                    tid,
                    sim_time,
                )

        info_map = self._env.apply_actions(actions, sim_time)
        rewards = self._env.compute_rewards()
        step_switches = sum(1 for info in info_map.values() if info.get("switched"))
        self._total_switches += step_switches

        for tid, r in rewards.items():
            self._cumulative_reward[tid] = self._cumulative_reward.get(tid, 0.0) + r

        self._decision_steps += 1

        if self._decision_steps % self._log_interval == 0:
            total_halt = sum(
                float(np.mean(obs_map[tid][MAX_PHASES + 1 : MAX_PHASES + 1 + MAX_LANES]))
                for tid in tls_ids
            )
            print(
                f"[RL] step={self._step_count} decisions={self._decision_steps} junctions={len(tls_ids)} "
                f"mean_halting_norm={total_halt/max(1,len(tls_ids)):.3f} "
                f"switches={step_switches}"
            )

        return {
            "step": self._step_count,
            "decision_step": self._decision_steps,
            "sim_time": sim_time,
            "actions": actions,
            "rewards": rewards,
            "info": info_map,
        }

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args: Any, traci_module: Any) -> "RLSignalController | None":
        """Construct from CLI args namespace.  Returns None if RL is disabled."""
        if not getattr(args, "enable_rl_signal_control", False):
            return None

        model_dir = getattr(args, "rl_model_dir", None)
        tls_ids_raw = getattr(args, "rl_tls_ids", None) or ""
        tls_ids = [t.strip() for t in tls_ids_raw.split(",") if t.strip()] if tls_ids_raw else []

        min_green = getattr(args, "rl_min_green_seconds", 15.0)
        yellow_dur = getattr(args, "rl_yellow_duration_seconds", 3.0)
        max_switches = getattr(args, "rl_max_switches_per_window", 4)
        max_controlled_tls = getattr(args, "rl_max_controlled_tls", 96)
        step_interval_steps = getattr(args, "rl_step_interval_steps", 2)

        guardrail_cfg = GuardrailConfig(
            min_green_seconds=float(min_green),
            yellow_duration_seconds=float(yellow_dur),
            max_switches_per_window=int(max_switches),
        )

        return cls(
            traci_module,
            tls_ids=tls_ids if tls_ids else None,
            model_dir=model_dir,
            guardrail_cfg=guardrail_cfg,
            max_controlled_tls=int(max_controlled_tls),
            step_interval_steps=int(step_interval_steps),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return cumulative per-junction stats for end-of-run reporting."""
        return {
            "total_steps": self._step_count,
            "decision_steps": self._decision_steps,
            "decision_interval_steps": self._step_interval_steps,
            "junctions_controlled": len(self._agents),
            "signal_switches": int(self._total_switches),
            "cumulative_rewards": {
                tid: round(r, 3) for tid, r in self._cumulative_reward.items()
            },
        }
