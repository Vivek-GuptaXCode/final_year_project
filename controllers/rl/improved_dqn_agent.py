"""Improved DQN agent for traffic signal control.

Implements a research-backed NumPy agent with:
- Double DQN target selection
- Dueling value / advantage heads
- Prioritized experience replay with importance sampling
- Huber loss
- Global gradient clipping
- Soft target-network updates
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(pre_activation: np.ndarray) -> np.ndarray:
    return (pre_activation > 0).astype(np.float32)


class ImprovedDQNAgent:
    """Dueling Double DQN with prioritized replay implemented in pure NumPy."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 2,
        hidden_dims: tuple[int, int] = (128, 64),
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,
        grad_clip: float = 10.0,
        double_dqn: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 100_000,
        per_eps: float = 1e-5,
        huber_delta: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip = grad_clip
        self.double_dqn = double_dqn
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_steps = per_beta_steps
        self.per_eps = per_eps
        self.huber_delta = huber_delta
        self.buffer_size = buffer_size

        rng = np.random.default_rng(seed)
        self._rng = rng

        h1, h2 = hidden_dims
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / h1)
        scale3 = np.sqrt(2.0 / h2)

        # Online network: shared trunk + dueling heads.
        self.W1 = rng.normal(0, scale1, (obs_dim, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.Wv = rng.normal(0, scale3, (h2, 1)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        self.Wa = rng.normal(0, scale3, (h2, n_actions)).astype(np.float32)
        self.ba = np.zeros(n_actions, dtype=np.float32)

        # Target network.
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()
        self.Wv_t = self.Wv.copy()
        self.bv_t = self.bv.copy()
        self.Wa_t = self.Wa.copy()
        self.ba_t = self.ba.copy()

        # Replay buffer stored in contiguous arrays for faster sampling.
        self._states = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._actions = np.zeros(buffer_size, dtype=np.int32)
        self._rewards = np.zeros(buffer_size, dtype=np.float32)
        self._next_states = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._dones = np.zeros(buffer_size, dtype=np.float32)
        self._priorities = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_len = 0
        self._max_priority = 1.0

        self._train_steps = 0
        self._total_episodes = 0

        self.loss_history: list[float] = []
        self.imitation_loss_history: list[float] = []
        self.reward_history: list[float] = []
        self.q_value_history: list[float] = []

    # ── Forward pass ─────────────────────────────────────────────────────

    def _forward(
        self,
        states: np.ndarray,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        Wv: np.ndarray,
        bv: np.ndarray,
        Wa: np.ndarray,
        ba: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return trunk activations plus value, advantage, and Q-values."""
        h1_pre = states @ W1 + b1
        h1 = _relu(h1_pre)
        h2_pre = h1 @ W2 + b2
        h2 = _relu(h2_pre)
        value = h2 @ Wv + bv
        advantage = h2 @ Wa + ba
        q = value + (advantage - advantage.mean(axis=1, keepdims=True))
        return h1_pre, h1, h2_pre, h2, value, advantage, q

    def _forward_fast(
        self,
        states: np.ndarray,
        *,
        use_target: bool = False,
    ) -> np.ndarray:
        """Fast inference-only forward pass returning Q-values."""
        s = np.ascontiguousarray(states, dtype=np.float32)
        if s.ndim == 1:
            s = s.reshape(1, -1)

        if use_target:
            h1 = np.maximum(0.0, s @ self.W1_t + self.b1_t)
            h2 = np.maximum(0.0, h1 @ self.W2_t + self.b2_t)
            value = h2 @ self.Wv_t + self.bv_t
            advantage = h2 @ self.Wa_t + self.ba_t
        else:
            h1 = np.maximum(0.0, s @ self.W1 + self.b1)
            h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
            value = h2 @ self.Wv + self.bv
            advantage = h2 @ self.Wa + self.ba
        return value + (advantage - advantage.mean(axis=1, keepdims=True))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Online Q-values for a single state."""
        return self._forward_fast(state)[0]

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Online Q-values for a batch of states."""
        return self._forward_fast(states)

    def select_actions_batch(self, states: np.ndarray, *, greedy: bool = False) -> np.ndarray:
        """Select actions for a batch of observations."""
        q = self.q_values_batch(states)
        actions = np.argmax(q, axis=1)
        if not greedy:
            explore_mask = self._rng.random(len(actions)) < self.epsilon
            random_actions = self._rng.integers(self.n_actions, size=len(actions))
            actions = np.where(explore_mask, random_actions, actions)
        return actions.astype(np.int32)

    # ── Action selection ─────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    # ── Replay buffer ────────────────────────────────────────────────────

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._buffer_pos
        self._states[idx] = state.astype(np.float32, copy=False)
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = next_state.astype(np.float32, copy=False)
        self._dones[idx] = float(done)
        self._priorities[idx] = self._max_priority

        self._buffer_pos = (idx + 1) % self.buffer_size
        self._buffer_len = min(self._buffer_len + 1, self.buffer_size)

    def _beta(self) -> float:
        progress = min(1.0, self._train_steps / max(1, self.per_beta_steps))
        return self.per_beta_start + progress * (self.per_beta_end - self.per_beta_start)

    def _sample_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        priorities = self._priorities[: self._buffer_len]
        scaled = np.power(priorities + self.per_eps, self.per_alpha).astype(np.float32)
        total = float(np.sum(scaled))
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(self._buffer_len, 1.0 / self._buffer_len, dtype=np.float32)
        else:
            probs = scaled / total

        indices = self._rng.choice(
            self._buffer_len,
            size=self.batch_size,
            replace=False,
            p=probs,
        )

        beta = self._beta()
        sample_probs = probs[indices]
        is_weights = np.power(self._buffer_len * sample_probs, -beta).astype(np.float32)
        is_weights /= max(float(np.max(is_weights)), 1e-6)

        return (
            indices,
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            is_weights.astype(np.float32),
        )

    # ── Target updates / optimization ────────────────────────────────────

    def _soft_update_target(self) -> None:
        self.W1_t = self.tau * self.W1 + (1.0 - self.tau) * self.W1_t
        self.b1_t = self.tau * self.b1 + (1.0 - self.tau) * self.b1_t
        self.W2_t = self.tau * self.W2 + (1.0 - self.tau) * self.W2_t
        self.b2_t = self.tau * self.b2 + (1.0 - self.tau) * self.b2_t
        self.Wv_t = self.tau * self.Wv + (1.0 - self.tau) * self.Wv_t
        self.bv_t = self.tau * self.bv + (1.0 - self.tau) * self.bv_t
        self.Wa_t = self.tau * self.Wa + (1.0 - self.tau) * self.Wa_t
        self.ba_t = self.tau * self.ba + (1.0 - self.tau) * self.ba_t

    def sync_target(self) -> None:
        """Hard-sync target network from the online network."""
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()
        self.Wv_t = self.Wv.copy()
        self.bv_t = self.bv.copy()
        self.Wa_t = self.Wa.copy()
        self.ba_t = self.ba.copy()

    def _clip_grads(self, grads: list[np.ndarray]) -> list[np.ndarray]:
        total_norm_sq = 0.0
        for grad in grads:
            total_norm_sq += float(np.sum(np.square(grad, dtype=np.float32)))
        total_norm = np.sqrt(total_norm_sq)
        if total_norm <= self.grad_clip or total_norm == 0.0:
            return grads

        scale = self.grad_clip / (total_norm + 1e-8)
        return [grad * scale for grad in grads]

    def _apply_q_gradients(
        self,
        states: np.ndarray,
        h1_pre: np.ndarray,
        h1: np.ndarray,
        h2_pre: np.ndarray,
        h2: np.ndarray,
        dQ: np.ndarray,
        *,
        lr: float | None = None,
    ) -> None:
        """Backpropagate a Q-value gradient tensor through the dueling network."""
        dV = dQ.sum(axis=1, keepdims=True)
        dA = dQ - dQ.mean(axis=1, keepdims=True)

        dWv = h2.T @ dV
        dbv = dV.sum(axis=0)
        dWa = h2.T @ dA
        dba = dA.sum(axis=0)

        dh2 = dV @ self.Wv.T + dA @ self.Wa.T
        dh2_pre = dh2 * _relu_grad(h2_pre)
        dW2 = h1.T @ dh2_pre
        db2 = dh2_pre.sum(axis=0)

        dh1 = dh2_pre @ self.W2.T
        dh1_pre = dh1 * _relu_grad(h1_pre)
        dW1 = states.T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        grads = self._clip_grads([dW1, db1, dW2, db2, dWv, dbv, dWa, dba])
        dW1, db1, dW2, db2, dWv, dbv, dWa, dba = grads

        step_lr = self.lr if lr is None else float(lr)
        self.W1 -= step_lr * dW1
        self.b1 -= step_lr * db1
        self.W2 -= step_lr * dW2
        self.b2 -= step_lr * db2
        self.Wv -= step_lr * dWv
        self.bv -= step_lr * dbv
        self.Wa -= step_lr * dWa
        self.ba -= step_lr * dba

    # ── Training ─────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        if self._buffer_len < self.batch_size:
            return None

        (
            indices,
            states,
            actions,
            rewards,
            next_states,
            dones,
            is_weights,
        ) = self._sample_batch()

        batch_size = states.shape[0]

        if self.double_dqn:
            q_next_online = self._forward_fast(next_states, use_target=False)
            best_actions = np.argmax(q_next_online, axis=1)
            q_next_target = self._forward_fast(next_states, use_target=True)
            max_q_next = q_next_target[np.arange(batch_size), best_actions]
        else:
            q_next_target = self._forward_fast(next_states, use_target=True)
            max_q_next = np.max(q_next_target, axis=1)

        targets = rewards + self.gamma * max_q_next * (1.0 - dones)

        h1_pre, h1, h2_pre, h2, _value, _advantage, q_curr = self._forward(
            states,
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.Wv,
            self.bv,
            self.Wa,
            self.ba,
        )

        q_selected = q_curr[np.arange(batch_size), actions]
        td_errors = q_selected - targets
        abs_td = np.abs(td_errors)

        quadratic = np.minimum(abs_td, self.huber_delta)
        linear = abs_td - quadratic
        sample_losses = 0.5 * quadratic**2 + self.huber_delta * linear
        loss = float(np.mean(is_weights * sample_losses))

        grad_selected = np.where(
            abs_td <= self.huber_delta,
            td_errors,
            self.huber_delta * np.sign(td_errors),
        ).astype(np.float32)
        grad_selected = (grad_selected * is_weights.astype(np.float32)) / batch_size

        dQ = np.zeros((batch_size, self.n_actions), dtype=np.float32)
        dQ[np.arange(batch_size), actions] = grad_selected
        self._apply_q_gradients(states, h1_pre, h1, h2_pre, h2, dQ)

        new_priorities = abs_td + self.per_eps
        self._priorities[indices] = new_priorities.astype(np.float32)
        self._max_priority = max(self._max_priority, float(np.max(new_priorities)))

        self._soft_update_target()
        self._train_steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.loss_history.append(loss)
        self.q_value_history.append(float(np.mean(np.max(q_curr, axis=1))))
        return loss

    def _imitation_step(
        self,
        states: np.ndarray,
        expert_actions: np.ndarray,
        *,
        margin: float = 0.8,
    ) -> float:
        """One DQfD-style large-margin imitation step on expert actions."""
        batch_size = states.shape[0]
        h1_pre, h1, h2_pre, h2, _value, _advantage, q_curr = self._forward(
            states,
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.Wv,
            self.bv,
            self.Wa,
            self.ba,
        )

        margin_bonus = np.full_like(q_curr, float(margin), dtype=np.float32)
        margin_bonus[np.arange(batch_size), expert_actions] = 0.0
        competing_actions = np.argmax(q_curr + margin_bonus, axis=1)
        losses = (q_curr + margin_bonus)[np.arange(batch_size), competing_actions] - q_curr[
            np.arange(batch_size), expert_actions
        ]
        loss = float(np.mean(losses))

        dQ = np.zeros_like(q_curr, dtype=np.float32)
        active = competing_actions != expert_actions
        if np.any(active):
            scale = 1.0 / batch_size
            active_idx = np.arange(batch_size)[active]
            dQ[active_idx, competing_actions[active]] += scale
            dQ[active_idx, expert_actions[active]] -= scale
            self._apply_q_gradients(states, h1_pre, h1, h2_pre, h2, dQ)

        self.imitation_loss_history.append(loss)
        return loss

    def pretrain_from_demonstrations(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        n_updates: int = 200,
        batch_size: int = 128,
        margin: float = 0.8,
    ) -> dict[str, float] | None:
        """Warm-start the Q-network from expert demonstrations."""
        if states.size == 0 or actions.size == 0 or n_updates <= 0:
            return None

        demo_states = np.ascontiguousarray(states, dtype=np.float32)
        demo_actions = np.ascontiguousarray(actions, dtype=np.int32)
        sample_count = demo_states.shape[0]
        losses: list[float] = []

        for _ in range(int(n_updates)):
            replace = sample_count < batch_size
            indices = self._rng.choice(sample_count, size=min(batch_size, sample_count), replace=replace)
            loss = self._imitation_step(
                demo_states[indices],
                demo_actions[indices],
                margin=margin,
            )
            losses.append(loss)

        self.sync_target()
        return {
            "n_samples": float(sample_count),
            "n_updates": float(n_updates),
            "mean_imitation_loss": float(np.mean(losses)) if losses else 0.0,
            "final_imitation_loss": float(losses[-1]) if losses else 0.0,
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, directory: str | Path, run_id: str = "latest") -> Path:
        out_dir = Path(directory) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        weights_path = out_dir / "weights.npz"
        np.savez(
            weights_path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            Wv=self.Wv,
            bv=self.bv,
            Wa=self.Wa,
            ba=self.ba,
        )

        meta: dict[str, Any] = {
            "agent_version": "improved_dqn_v3",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dims": list(self.hidden_dims),
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "tau": self.tau,
            "grad_clip": self.grad_clip,
            "double_dqn": self.double_dqn,
            "per_alpha": self.per_alpha,
            "per_beta_start": self.per_beta_start,
            "per_beta_end": self.per_beta_end,
            "per_beta_steps": self.per_beta_steps,
            "per_eps": self.per_eps,
            "huber_delta": self.huber_delta,
            "train_steps": self._train_steps,
            "total_episodes": self._total_episodes,
            "buffer_len": self._buffer_len,
        }
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return weights_path

    @classmethod
    def load(cls, directory: str | Path, run_id: str = "latest") -> "ImprovedDQNAgent":
        out_dir = Path(directory) / run_id
        meta_path = out_dir / "meta.json"
        weights_path = out_dir / "weights.npz"

        meta = json.loads(meta_path.read_text())
        hidden_dims = tuple(meta.get("hidden_dims", [128, 64]))
        agent = cls(
            obs_dim=meta["obs_dim"],
            n_actions=meta["n_actions"],
            hidden_dims=hidden_dims,
            lr=meta["lr"],
            gamma=meta["gamma"],
            epsilon_start=meta.get("epsilon_start", meta.get("epsilon", meta["epsilon_min"])),
            epsilon_min=meta["epsilon_min"],
            epsilon_decay=meta["epsilon_decay"],
            buffer_size=meta.get("buffer_size", 50_000),
            batch_size=meta["batch_size"],
            tau=meta.get("tau", 0.005),
            grad_clip=meta.get("grad_clip", 10.0),
            double_dqn=meta.get("double_dqn", True),
            per_alpha=meta.get("per_alpha", 0.6),
            per_beta_start=meta.get("per_beta_start", 0.4),
            per_beta_end=meta.get("per_beta_end", 1.0),
            per_beta_steps=meta.get("per_beta_steps", 100_000),
            per_eps=meta.get("per_eps", 1e-5),
            huber_delta=meta.get("huber_delta", 1.0),
        )
        agent.epsilon = meta["epsilon"]
        agent._train_steps = meta["train_steps"]
        agent._total_episodes = meta["total_episodes"]

        data = np.load(weights_path)
        agent.W1 = data["W1"]
        agent.b1 = data["b1"]
        agent.W2 = data["W2"]
        agent.b2 = data["b2"]

        if {"Wv", "bv", "Wa", "ba"}.issubset(set(data.files)):
            agent.Wv = data["Wv"]
            agent.bv = data["bv"]
            agent.Wa = data["Wa"]
            agent.ba = data["ba"]
        else:
            # Backward compatibility with the previous single-head v1 agent.
            legacy_W3 = data["W3"]
            legacy_b3 = data["b3"]
            mean_w = np.mean(legacy_W3, axis=1, keepdims=True)
            mean_b = np.array([np.mean(legacy_b3)], dtype=np.float32)
            agent.Wv = mean_w.astype(np.float32)
            agent.bv = mean_b
            agent.Wa = (legacy_W3 - mean_w).astype(np.float32)
            agent.ba = (legacy_b3 - float(np.mean(legacy_b3))).astype(np.float32)

        agent.W1_t = agent.W1.copy()
        agent.b1_t = agent.b1.copy()
        agent.W2_t = agent.W2.copy()
        agent.b2_t = agent.b2.copy()
        agent.Wv_t = agent.Wv.copy()
        agent.bv_t = agent.bv.copy()
        agent.Wa_t = agent.Wa.copy()
        agent.ba_t = agent.ba.copy()
        return agent

    # ── Diagnostics ──────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        recent_loss = float(np.mean(self.loss_history[-100:])) if self.loss_history else None
        recent_q = float(np.mean(self.q_value_history[-100:])) if self.q_value_history else None
        return {
            "agent_version": "improved_dqn_v3",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dims": self.hidden_dims,
            "double_dqn": self.double_dqn,
            "epsilon": round(self.epsilon, 4),
            "train_steps": self._train_steps,
            "buffer_len": self._buffer_len,
            "recent_loss_100": recent_loss,
            "recent_imitation_loss_100": float(np.mean(self.imitation_loss_history[-100:]))
            if self.imitation_loss_history else None,
            "recent_q_100": recent_q,
            "total_params": self._count_params(),
        }

    def _count_params(self) -> int:
        return (
            self.W1.size
            + self.b1.size
            + self.W2.size
            + self.b2.size
            + self.Wv.size
            + self.bv.size
            + self.Wa.size
            + self.ba.size
        )


DoubleDQNAgent = ImprovedDQNAgent
