"""Lightweight numpy DQN agent for adaptive traffic signal control.

Implements Deep Q-Network (Mnih et al., 2015) without any deep-learning
framework dependency — only numpy is required.

Architecture
------------
  Input:   obs_dim-dimensional state vector (float32)
  Hidden:  hidden_dim ReLU units
  Output:  n_actions Q-values (typically 2: KEEP / SWITCH)

Training
--------
  - Experience replay buffer (circular deque, configurable capacity)
  - Separate target network updated every `target_update_freq` gradient steps
  - Epsilon-greedy exploration with exponential decay
  - Mini-batch SGD with manual backpropagation (no autograd)

Persistence
-----------
  Weights stored in NumPy .npz format in `models/rl/artifacts/<run_id>/`.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(pre_activation: np.ndarray) -> np.ndarray:
    return (pre_activation > 0).astype(np.float32)


class DQNAgent:
    """Two-layer MLP DQN implemented in pure NumPy.

    Parameters
    ----------
    obs_dim      : Dimension of the state observation vector.
    n_actions    : Number of discrete actions (default 2: KEEP / SWITCH).
    hidden_dim   : Width of the single hidden layer (default 64).
    lr           : Learning rate for gradient descent (default 1e-3).
    gamma        : Discount factor (default 0.99).
    epsilon_start: Initial exploration rate (default 1.0).
    epsilon_min  : Minimum exploration rate (default 0.05).
    epsilon_decay: Multiplicative decay applied after each train step (0.995).
    buffer_size  : Capacity of the experience replay buffer (default 20 000).
    batch_size   : Mini-batch size for each gradient step (default 64).
    target_update_freq: Steps between hard target-network copies (default 200).
    seed         : Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 2,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 20_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        seed: int = 42,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        rng = np.random.default_rng(seed)
        self._rng = rng

        # He initialisation for ReLU networks
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        # Online network weights
        self.W1: np.ndarray = rng.normal(0, scale1, (obs_dim, hidden_dim)).astype(np.float32)
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W2: np.ndarray = rng.normal(0, scale2, (hidden_dim, n_actions)).astype(np.float32)
        self.b2: np.ndarray = np.zeros(n_actions, dtype=np.float32)

        # Target network (hard copy)
        self.W1_t = self.W1.copy()
        self.b1_t = self.b1.copy()
        self.W2_t = self.W2.copy()
        self.b2_t = self.b2.copy()

        # Replay buffer stores (state, action, reward, next_state, done) tuples
        self._buffer: deque = deque(maxlen=buffer_size)

        self._train_steps: int = 0
        self._total_episodes: int = 0

        # Training history for reporting
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []

    # ── Forward pass ───────────────────────────────────────────────────────

    def _forward(
        self,
        states: np.ndarray,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (h_pre, h, q) for a batch of states."""
        h_pre = states @ W1 + b1          # (B, hidden_dim)
        h = _relu(h_pre)                  # (B, hidden_dim)
        q = h @ W2 + b2                   # (B, n_actions)
        return h_pre, h, q

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Online Q-values for a single state (shape: (n_actions,))."""
        s = state.reshape(1, -1).astype(np.float32)
        _, _, q = self._forward(s, self.W1, self.b1, self.W2, self.b2)
        return q[0]

    # ── Action selection ───────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Epsilon-greedy action selection.

        Args:
            state:  Observation vector.
            greedy: If True, always pick argmax (evaluation mode).
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    # ── Experience replay ─────────────────────────────────────────────────

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self._buffer.append(
            (
                state.astype(np.float32),
                int(action),
                float(reward),
                next_state.astype(np.float32),
                float(done),
            )
        )

    # ── Training ───────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """Sample a mini-batch and perform one gradient-descent step.

        Returns the batch TD-error MSE loss, or None if buffer too small.
        """
        if len(self._buffer) < self.batch_size:
            return None

        # Sample without replacement
        indices = self._rng.choice(len(self._buffer), size=self.batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        states = np.array([b[0] for b in batch], dtype=np.float32)       # (B, D)
        actions = np.array([b[1] for b in batch], dtype=np.int32)        # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)      # (B,)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)  # (B, D)
        dones = np.array([b[4] for b in batch], dtype=np.float32)        # (B,)

        B = self.batch_size

        # Target Q using frozen target network (no gradient)
        _, _, q_next = self._forward(next_states, self.W1_t, self.b1_t, self.W2_t, self.b2_t)
        max_q_next = np.max(q_next, axis=1)  # (B,)
        targets = rewards + self.gamma * max_q_next * (1.0 - dones)  # (B,)

        # Online Q
        h_pre, h, q_curr = self._forward(states, self.W1, self.b1, self.W2, self.b2)

        # TD error: only the taken action's Q-value has a gradient
        td_errors_full = np.zeros((B, self.n_actions), dtype=np.float32)
        td_errors_full[np.arange(B), actions] = (
            q_curr[np.arange(B), actions] - targets
        )  # δ = Q(s,a) - y

        # Loss = mean(δ²) for logging
        loss = float(np.mean(td_errors_full[np.arange(B), actions] ** 2))

        # Backpropagation through 2-layer MLP
        # ∂L/∂Q = td_errors_full (per element)
        # Layer 2 gradients
        dW2 = h.T @ td_errors_full / B              # (hidden_dim, n_actions)
        db2 = td_errors_full.sum(axis=0) / B        # (n_actions,)

        # Gradient through hidden layer
        dh = td_errors_full @ self.W2.T             # (B, hidden_dim)
        dh_pre = dh * _relu_grad(h_pre)             # (B, hidden_dim) — ReLU mask

        # Layer 1 gradients
        dW1 = states.T @ dh_pre / B                 # (obs_dim, hidden_dim)
        db1 = dh_pre.sum(axis=0) / B                # (hidden_dim,)

        # SGD update (in-place to avoid re-allocation)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Hard target-network update
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.W1_t = self.W1.copy()
            self.b1_t = self.b1.copy()
            self.W2_t = self.W2.copy()
            self.b2_t = self.b2.copy()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.loss_history.append(loss)
        return loss

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: str | Path, run_id: str = "latest") -> Path:
        """Save weights and metadata under `directory/<run_id>/`."""
        out_dir = Path(directory) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        weights_path = out_dir / "weights.npz"
        np.savez(
            weights_path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
        )

        meta: dict[str, Any] = {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "train_steps": self._train_steps,
            "total_episodes": self._total_episodes,
            "buffer_len": len(self._buffer),
        }
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return weights_path

    @classmethod
    def load(cls, directory: str | Path, run_id: str = "latest") -> "DQNAgent":
        """Load agent from a saved directory."""
        out_dir = Path(directory) / run_id
        meta_path = out_dir / "meta.json"
        weights_path = out_dir / "weights.npz"

        meta = json.loads(meta_path.read_text())
        agent = cls(
            obs_dim=meta["obs_dim"],
            n_actions=meta["n_actions"],
            hidden_dim=meta["hidden_dim"],
            lr=meta["lr"],
            gamma=meta["gamma"],
            epsilon_start=meta.get("epsilon", meta.get("epsilon_min", 0.05)),
            epsilon_min=meta["epsilon_min"],
            epsilon_decay=meta["epsilon_decay"],
            batch_size=meta["batch_size"],
            target_update_freq=meta["target_update_freq"],
        )
        agent.epsilon = meta["epsilon"]
        agent._train_steps = meta["train_steps"]
        agent._total_episodes = meta["total_episodes"]

        data = np.load(weights_path)
        agent.W1 = data["W1"]
        agent.b1 = data["b1"]
        agent.W2 = data["W2"]
        agent.b2 = data["b2"]
        # Sync target network
        agent.W1_t = agent.W1.copy()
        agent.b1_t = agent.b1.copy()
        agent.W2_t = agent.W2.copy()
        agent.b2_t = agent.b2.copy()
        return agent

    # ── Diagnostics ────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        recent_loss = (
            float(np.mean(self.loss_history[-100:])) if self.loss_history else None
        )
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "epsilon": round(self.epsilon, 4),
            "train_steps": self._train_steps,
            "buffer_len": len(self._buffer),
            "recent_loss_100": recent_loss,
        }
