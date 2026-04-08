"""PyTorch-based Temporal Graph Convolutional Network for Traffic Prediction.

Production-grade implementation following the official T-GCN paper and code:
- T-GCN: Zhao et al. "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction"
         IEEE T-ITS, 2019. https://arxiv.org/abs/1811.05320
- A3T-GCN: Bai et al. "A3T-GCN: Attention Temporal Graph Convolutional Network"
           ISPRS IJGI, 2020. https://arxiv.org/abs/2006.11583

Architecture (faithful to paper):
- TGCNCell: Combines GCN and GRU in a single cell (Eq. 7-9 in paper)
  - Graph convolution: A[x,h]W + b for spatial dependencies
  - GRU gates: reset gate r, update gate u for temporal dynamics
- Temporal Attention: Weights importance of different time steps (A3T-GCN)
- Output head: Predicts congestion probability and reroute decision

Key Metrics Tracked:
- MAE (Mean Absolute Error): Primary metric for traffic prediction
- RMSE (Root Mean Squared Error): Penalizes large errors
- MAPE (Mean Absolute Percentage Error): Scale-invariant accuracy
- Accuracy: Classification accuracy for congestion detection
- F1 Score: Balanced metric for imbalanced congestion events
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TGCNConfig:
    """Configuration for the PyTorch T-GCN model - BALANCED for speed + accuracy."""

    # BALANCED architecture based on research (STGCN, PyTorch Geometric Temporal)
    node_feature_dim: int = 4       # 4 features for richer representation
    hidden_dim: int = 24            # Balanced size (16->24, still compact)
    output_dim: int = 1             # Only predict congestion

    # Optimal temporal modeling
    sequence_length: int = 5        # 5 timesteps captures short-term patterns
    prediction_horizon: int = 1     # Steps ahead to predict
    use_attention: bool = False     # Disabled (overhead not worth it for small seq)

    # Optimized training schedule
    learning_rate: float = 0.01     # High LR for fast learning
    weight_decay: float = 5e-4      # Stronger regularization
    batch_size: int = 12            # Larger batch for better gradients
    buffer_size: int = 500          # Small buffer for recency
    warmup_steps: int = 10          # Minimal warmup

    # Rerouting thresholds (tuned for Kolkata traffic)
    congestion_threshold: float = 0.60   # Higher threshold
    medium_risk_threshold: float = 0.55
    high_risk_threshold: float = 0.80
    max_reroute_fraction: float = 0.30

    # Minimal regularization
    dropout_rate: float = 0.1       # Very low dropout

    # Metrics tracking - VERY SPARSE
    log_interval: int = 200         # Reduced logging
    checkpoint_interval: int = 1000  # Reduced checkpoints

    # AGGRESSIVE performance optimization
    inference_every_n_steps: int = 10    # Predict every 10 steps (good balance)
    train_every_n_steps: int = 25        # Train every 25 steps (slower but stable)
    use_cached_features: bool = True     # Cache everything

    # NEW: Prediction smoothing for stability
    use_ema_smoothing: bool = True       # Exponential moving average
    ema_alpha: float = 0.3               # Smoothing factor (0.2-0.4 optimal)
    cache_duration_steps: int = 15       # Cache predictions for 15 steps


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Track and compute traffic prediction metrics.

    Implements standard metrics from traffic prediction literature:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - Accuracy: Binary classification accuracy
    - Precision/Recall/F1: For congestion detection
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.predictions = deque(maxlen=self.window_size)
        self.targets = deque(maxlen=self.window_size)
        self.losses = deque(maxlen=self.window_size)
        self.binary_preds = deque(maxlen=self.window_size)
        self.binary_targets = deque(maxlen=self.window_size)
        self.total_samples = 0
        self.total_correct = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(
        self,
        pred: float | np.ndarray,
        target: float | np.ndarray,
        loss: float = 0.0,
        threshold: float = 0.5
    ):
        """Update metrics with new prediction-target pair."""
        pred = np.atleast_1d(pred).flatten()
        target = np.atleast_1d(target).flatten()

        for p, t in zip(pred, target):
            self.predictions.append(p)
            self.targets.append(t)
            self.losses.append(loss)

            # Binary classification metrics
            pred_class = 1 if p >= threshold else 0
            true_class = 1 if t >= threshold else 0
            self.binary_preds.append(pred_class)
            self.binary_targets.append(true_class)

            self.total_samples += 1
            if pred_class == true_class:
                self.total_correct += 1

            if pred_class == 1 and true_class == 1:
                self.true_positives += 1
            elif pred_class == 1 and true_class == 0:
                self.false_positives += 1
            elif pred_class == 0 and true_class == 1:
                self.false_negatives += 1

    def compute_mae(self) -> float:
        """Mean Absolute Error."""
        if not self.predictions:
            return 0.0
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return float(np.mean(np.abs(preds - targets)))

    def compute_rmse(self) -> float:
        """Root Mean Squared Error."""
        if not self.predictions:
            return 0.0
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return float(np.sqrt(np.mean((preds - targets) ** 2)))

    def compute_mape(self, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        if not self.predictions:
            return 0.0
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        # Avoid division by zero
        valid_mask = np.abs(targets) > epsilon
        if not np.any(valid_mask):
            return 0.0
        return float(np.mean(np.abs((targets[valid_mask] - preds[valid_mask]) / targets[valid_mask])) * 100)

    def compute_accuracy(self) -> float:
        """Binary classification accuracy."""
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples

    def compute_precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    def compute_recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    def compute_f1(self) -> float:
        """F1 Score: 2 * (Precision * Recall) / (Precision + Recall)."""
        precision = self.compute_precision()
        recall = self.compute_recall()
        denom = precision + recall
        return 2 * precision * recall / denom if denom > 0 else 0.0

    def compute_avg_loss(self) -> float:
        """Average training loss."""
        if not self.losses:
            return 0.0
        return float(np.mean(self.losses))

    def get_all_metrics(self) -> dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            "mae": self.compute_mae(),
            "rmse": self.compute_rmse(),
            "mape": self.compute_mape(),
            "accuracy": self.compute_accuracy(),
            "precision": self.compute_precision(),
            "recall": self.compute_recall(),
            "f1_score": self.compute_f1(),
            "avg_loss": self.compute_avg_loss(),
            "total_samples": self.total_samples,
        }

    def format_metrics(self) -> str:
        """Format metrics as a human-readable string."""
        m = self.get_all_metrics()
        return (
            f"MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | MAPE={m['mape']:.2f}% | "
            f"Acc={m['accuracy']:.2%} | F1={m['f1_score']:.2%} | Loss={m['avg_loss']:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph Utilities (from official T-GCN code)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_laplacian_with_self_loop(adj: Tensor) -> Tensor:
    """Calculate normalized Laplacian: D^-1/2 (A + I) D^-1/2.

    This is the graph convolution normalization from Kipf & Welling.
    Exactly follows the official T-GCN implementation.
    """
    # Add self-loops
    adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree matrix
    row_sum = adj_with_self_loops.sum(dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    # Symmetric normalization: D^-1/2 A D^-1/2
    normalized_laplacian = (
        adj_with_self_loops.matmul(d_mat_inv_sqrt).t().matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


# ─────────────────────────────────────────────────────────────────────────────
# T-GCN Graph Convolution (official implementation)
# ─────────────────────────────────────────────────────────────────────────────

class TGCNGraphConvolution(nn.Module):
    """Graph convolution for T-GCN cell.

    Implements: output = A[x, h]W + b
    where A is normalized Laplacian, [x, h] is concatenation of
    input and hidden state.

    Follows official implementation exactly.
    """

    def __init__(
        self,
        adj: np.ndarray,
        num_gru_units: int,
        output_dim: int,
        bias_init: float = 0.0
    ):
        super().__init__()
        self.num_gru_units = num_gru_units
        self.output_dim = output_dim
        self.bias_init = bias_init

        # Register normalized Laplacian as buffer (not trained)
        laplacian = calculate_laplacian_with_self_loop(
            torch.FloatTensor(adj)
        )
        self.register_buffer("laplacian", laplacian)

        # Learnable parameters
        # Input: [x, h] has dimension (num_gru_units + 1) if x is 1D
        # For multi-feature input, we adapt this
        self.weights = nn.Parameter(
            torch.empty(num_gru_units + 1, output_dim)
        )
        self.biases = nn.Parameter(torch.empty(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self.bias_init)

    def forward(self, inputs: Tensor, hidden_state: Tensor) -> Tensor:
        """
        Args:
            inputs: [batch_size, num_nodes] - single feature per node
            hidden_state: [batch_size, num_nodes * num_gru_units]
        Returns:
            output: [batch_size, num_nodes * output_dim]
        """
        batch_size, num_nodes = inputs.shape

        # Reshape inputs: [batch, nodes] -> [batch, nodes, 1]
        inputs = inputs.unsqueeze(-1)

        # Reshape hidden: [batch, nodes*units] -> [batch, nodes, units]
        hidden_state = hidden_state.view(batch_size, num_nodes, self.num_gru_units)

        # Concatenate: [batch, nodes, units+1]
        concatenation = torch.cat([inputs, hidden_state], dim=2)

        # Reshape for graph conv: [nodes, (units+1)*batch]
        concatenation = concatenation.permute(1, 2, 0)
        concatenation = concatenation.reshape(num_nodes, -1)

        # Graph convolution: A @ [x, h]
        a_times_concat = self.laplacian @ concatenation

        # Reshape back: [nodes, units+1, batch]
        a_times_concat = a_times_concat.view(num_nodes, self.num_gru_units + 1, batch_size)

        # [batch, nodes, units+1]
        a_times_concat = a_times_concat.permute(2, 0, 1)

        # [batch*nodes, units+1]
        a_times_concat = a_times_concat.reshape(-1, self.num_gru_units + 1)

        # Linear transform: A[x,h]W + b
        outputs = a_times_concat @ self.weights + self.biases

        # [batch, nodes, output_dim] -> [batch, nodes*output_dim]
        outputs = outputs.view(batch_size, num_nodes, self.output_dim)
        outputs = outputs.view(batch_size, -1)

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# T-GCN Cell (official implementation)
# ─────────────────────────────────────────────────────────────────────────────

class TGCNCell(nn.Module):
    """T-GCN Cell combining GCN and GRU.

    Implements equations 7-9 from the T-GCN paper:
    - r, u = σ(A[x, h]W + b)           # Reset and update gates
    - c = tanh(A[x, r⊙h]W + b)         # Candidate state
    - h' = u⊙h + (1-u)⊙c              # New hidden state
    """

    def __init__(self, adj: np.ndarray, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        num_nodes = adj.shape[0]

        # Store adjacency
        self.register_buffer("adj", torch.FloatTensor(adj))

        # Graph conv for reset and update gates (outputs 2*hidden_dim)
        self.graph_conv1 = TGCNGraphConvolution(
            adj, hidden_dim, hidden_dim * 2, bias_init=1.0
        )

        # Graph conv for candidate state
        self.graph_conv2 = TGCNGraphConvolution(
            adj, hidden_dim, hidden_dim, bias_init=0.0
        )

    def forward(
        self,
        inputs: Tensor,
        hidden_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            inputs: [batch_size, num_nodes]
            hidden_state: [batch_size, num_nodes * hidden_dim]
        Returns:
            output: [batch_size, num_nodes * hidden_dim]
            new_hidden: [batch_size, num_nodes * hidden_dim]
        """
        # Reset and update gates: [r, u] = σ(A[x, h]W + b)
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))

        # Split into reset and update gates
        r, u = torch.chunk(concatenation, chunks=2, dim=1)

        # Candidate state: c = tanh(A[x, r⊙h]W + b)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))

        # New hidden: h' = u⊙h + (1-u)⊙c
        new_hidden_state = u * hidden_state + (1.0 - u) * c

        return new_hidden_state, new_hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Attention (A3T-GCN)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Temporal attention mechanism from A3T-GCN.

    Learns to weight the importance of different time steps in the sequence.
    """

    def __init__(self, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # Attention parameters
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: list[Tensor]) -> Tensor:
        """
        Args:
            hidden_states: List of [batch, num_nodes * hidden_dim] tensors
        Returns:
            context: [batch, num_nodes, hidden_dim] attention-weighted output
        """
        batch_size = hidden_states[0].size(0)
        seq_len = len(hidden_states)

        # Stack and reshape: [seq_len, batch, nodes, hidden]
        stacked = torch.stack(hidden_states, dim=0)
        stacked = stacked.view(seq_len, batch_size, self.num_nodes, self.hidden_dim)

        # Compute attention scores
        # [seq_len, batch, nodes, hidden]
        transformed = torch.tanh(self.W(stacked))

        # [seq_len, batch, nodes, 1]
        scores = self.V(transformed)

        # Softmax over sequence dimension: [seq_len, batch, nodes, 1]
        weights = F.softmax(scores, dim=0)

        # Weighted sum: [batch, nodes, hidden]
        context = (stacked * weights).sum(dim=0)

        return context


# ─────────────────────────────────────────────────────────────────────────────
# Full T-GCN Model
# ─────────────────────────────────────────────────────────────────────────────

class TGCN(nn.Module):
    """Temporal Graph Convolutional Network.

    Full model that processes a sequence of traffic states through T-GCN cells
    and outputs congestion predictions.
    """

    def __init__(self, adj: np.ndarray, config: TGCNConfig):
        super().__init__()
        self.config = config
        self.num_nodes = adj.shape[0]
        self.hidden_dim = config.hidden_dim

        # Input projection (features -> single value per node)
        self.input_proj = nn.Sequential(
            nn.Linear(config.node_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, 1)
        )

        # T-GCN cell
        self.tgcn_cell = TGCNCell(adj, 1, config.hidden_dim)

        # Temporal attention (A3T-GCN)
        if config.use_attention:
            self.attention = TemporalAttention(config.hidden_dim, self.num_nodes)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, config.output_dim),
            nn.Sigmoid()  # Output probabilities
        )

        # Store adjacency
        self.register_buffer("adj", torch.FloatTensor(adj))

    def forward(
        self,
        x_seq: Tensor,
        h_init: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x_seq: [batch, seq_len, num_nodes, features] or [seq_len, num_nodes, features]
            h_init: Initial hidden state [batch, num_nodes * hidden_dim]
        Returns:
            output: [batch, num_nodes, output_dim] predictions
            h_final: [batch, num_nodes * hidden_dim] final hidden state
        """
        # Handle unbatched input
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(0)

        batch_size, seq_len, num_nodes, features = x_seq.shape
        device = x_seq.device

        # Initialize hidden state
        if h_init is None:
            h_init = torch.zeros(
                batch_size, num_nodes * self.hidden_dim,
                device=device
            )

        # Project input features to single value per node
        # [batch, seq, nodes, feat] -> [batch, seq, nodes, 1] -> [batch, seq, nodes]
        x_proj = self.input_proj(x_seq).squeeze(-1)

        # Process sequence through T-GCN cells
        hidden_states = []
        h = h_init

        for t in range(seq_len):
            x_t = x_proj[:, t, :]  # [batch, nodes]
            output, h = self.tgcn_cell(x_t, h)
            hidden_states.append(h)

        # Apply temporal attention or use last hidden state
        if self.config.use_attention:
            # [batch, nodes, hidden]
            context = self.attention(hidden_states)
        else:
            # [batch, nodes, hidden]
            context = h.view(batch_size, num_nodes, self.hidden_dim)

        # Output predictions: [batch, nodes, output_dim]
        output = self.output_head(context)

        return output, h


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """Prioritized experience replay for online learning.

    Experiences with higher TD-error get sampled more frequently.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state_seq: np.ndarray,
        target: np.ndarray,
        priority: float = 1.0
    ):
        """Store experience with priority."""
        experience = (state_seq.copy(), target.copy())

        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
        """Sample batch with importance sampling weights."""
        if self.size == 0:
            return [], [], []

        # Compute sampling probabilities
        probs = self.priorities[:self.size]
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(
            self.size,
            min(batch_size, self.size),
            p=probs,
            replace=False
        )

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        # Get experiences
        batch = [self.buffer[i] for i in indices]
        states, targets = zip(*batch)

        return list(states), list(targets), weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD-error."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha

    def __len__(self) -> int:
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# Main Rerouting Engine
# ─────────────────────────────────────────────────────────────────────────────

class PyTorchGNNRerouteEngine:
    """PyTorch-based T-GCN for traffic congestion prediction and rerouting.

    This class provides:
    - Real-time congestion prediction using T-GCN
    - Online learning from traffic observations
    - Comprehensive metrics tracking
    - Model checkpointing and loading

    Compatible with existing GNNRerouteEngine interface.
    """

    def __init__(
        self,
        road_graph: nx.Graph | nx.DiGraph,
        rsu_junctions: list[str],
        config: TGCNConfig | None = None,
        model_path: str | None = None,
    ):
        self.config = config or TGCNConfig()
        self.road_graph = road_graph
        self.rsu_junctions = list(rsu_junctions)
        self.rsu_to_idx = {j: i for i, j in enumerate(self.rsu_junctions)}
        self.num_nodes = len(self.rsu_junctions)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[T-GCN] Using device: {self.device}")

        # Build adjacency matrix from road graph
        self.adj_matrix = self._build_rsu_adjacency()

        # Initialize model
        self.model = TGCN(self.adj_matrix, self.config).to(self.device)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )

        # Loss function
        self.criterion = nn.MSELoss(reduction='none')

        # Experience replay with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(self.config.buffer_size)

        # Metrics tracking
        self.train_metrics = MetricsTracker(window_size=200)
        self.eval_metrics = MetricsTracker(window_size=100)

        # State tracking
        self.feature_history: deque[np.ndarray] = deque(
            maxlen=self.config.sequence_length
        )
        self.hidden_state: Tensor | None = None
        self.step_count = 0
        self.training_enabled = True
        self.total_train_steps = 0

        # Metrics history for visualization
        self.metrics_history: list[dict] = []

        # NEW: EMA smoothing for prediction stability
        if self.config.use_ema_smoothing:
            self.ema_predictions: dict[str, float] = {}  # Smoothed congestion per RSU
            print(f"[T-GCN] EMA smoothing enabled (alpha={self.config.ema_alpha})")

        # NEW: Previous features for gradient calculation
        self.previous_features: np.ndarray | None = None

        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.load(model_path)

        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[T-GCN] Initialized with {self.num_nodes} RSU nodes")
        print(f"[T-GCN] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"[T-GCN] Architecture: hidden_dim={self.config.hidden_dim}, "
              f"seq_len={self.config.sequence_length}, features={self.config.node_feature_dim}")

    def _build_rsu_adjacency(self) -> np.ndarray:
        """Build adjacency matrix between RSU nodes based on road connectivity."""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        # For each pair of RSUs, check connectivity in road graph
        for i, rsu_i in enumerate(self.rsu_junctions):
            for j, rsu_j in enumerate(self.rsu_junctions):
                if i != j:
                    try:
                        # Check if nodes exist and are connected
                        if (self.road_graph.has_node(rsu_i) and
                            self.road_graph.has_node(rsu_j)):
                            if nx.has_path(self.road_graph, rsu_i, rsu_j):
                                path_len = nx.shortest_path_length(
                                    self.road_graph, rsu_i, rsu_j
                                )
                                # Connected if within 5 hops, weight by inverse distance
                                if path_len <= 5:
                                    adj[i, j] = 1.0 / max(path_len, 1)
                    except (nx.NetworkXError, nx.NodeNotFound):
                        pass

        # Ensure graph is connected (add minimal edges if needed)
        row_sums = adj.sum(axis=1)
        for i in range(self.num_nodes):
            if row_sums[i] < 0.1:
                # Add connection to nearest neighbors by index
                adj[i, (i + 1) % self.num_nodes] = 0.3
                adj[(i + 1) % self.num_nodes, i] = 0.3

        return adj

    def _build_node_features(self, rsu_states: dict[str, dict]) -> np.ndarray:
        """Build feature matrix from RSU states.

        BALANCED features (4-dim) for improved accuracy:
        1. congestion_score: Combined density + speed + queue metric
        2. speed_gradient: Rate of change in speed (temporal derivative)
        3. incident_flag: Binary congestion indicator
        4. spatial_gradient: Difference from average neighbor congestion

        This provides both instantaneous state and trend information.
        """
        features = np.zeros(
            (self.num_nodes, self.config.node_feature_dim),
            dtype=np.float32
        )

        # First pass: compute base features
        congestion_scores = np.zeros(self.num_nodes)
        for rsu_id, state in rsu_states.items():
            if rsu_id not in self.rsu_to_idx:
                continue
            idx = self.rsu_to_idx[rsu_id]

            # Extract raw features
            vehicle_count = state.get("vehicle_count", 0)
            avg_speed = state.get("avg_speed", 13.89)
            queue_length = state.get("queue_length", 0)
            incident = 1.0 if state.get("incident", False) else 0.0

            # Compute congestion score (composite metric)
            vehicle_density = min(vehicle_count / 30.0, 1.5)
            speed_ratio = max(0, 1.0 - avg_speed / 15.0)
            queue_ratio = min(queue_length / max(1, vehicle_count), 1.0)
            congestion_score = min(1.0, (
                0.50 * vehicle_density +
                0.35 * speed_ratio +
                0.15 * queue_ratio
            ))
            congestion_scores[idx] = congestion_score

            # Compute speed gradient (temporal derivative)
            speed_gradient = 0.0
            if self.previous_features is not None:
                prev_speed_ratio = 1.0 - self.previous_features[idx, 0] / 0.85
                # Speed gradient: positive = getting slower, negative = getting faster
                speed_gradient = speed_ratio - prev_speed_ratio
                speed_gradient = np.clip(speed_gradient, -1.0, 1.0)

            # Store base features (will add spatial gradient in second pass)
            features[idx, 0] = congestion_score
            features[idx, 1] = speed_gradient
            features[idx, 2] = incident
            # features[idx, 3] will be spatial_gradient

        # Second pass: compute spatial gradients
        for idx in range(self.num_nodes):
            # Get neighbor congestion scores (from adjacency matrix)
            neighbors_mask = self.adj_matrix[idx] > 0
            if neighbors_mask.any():
                neighbor_congestion = congestion_scores[neighbors_mask].mean()
                spatial_gradient = congestion_scores[idx] - neighbor_congestion
                spatial_gradient = np.clip(spatial_gradient, -1.0, 1.0)
            else:
                spatial_gradient = 0.0

            features[idx, 3] = spatial_gradient

        # Save for next gradient calculation
        self.previous_features = features.copy()

        return features

    def predict(
        self,
        rsu_states: dict[str, dict],
        current_step: int = 0,
        force_inference: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Predict congestion and rerouting decisions for all RSU nodes.

        Optimized for performance with cached predictions.

        Args:
            rsu_states: Current traffic state per RSU
            current_step: Simulation step number
            force_inference: Skip step-based caching

        Returns:
            Dict mapping RSU ID to prediction dict with keys:
                - p_congestion: Probability of congestion [0, 1]
                - reroute_fraction: Fraction of vehicles to reroute [0, max]
                - risk_level: "low", "medium", or "high"
                - confidence: Model confidence [0, 1]
                - should_reroute: Boolean - should vehicles in this RSU be rerouted
                - avoid_rsus: List of RSU IDs to avoid (high risk)
        """
        self.step_count = current_step

        # Build features and add to history
        features = self._build_node_features(rsu_states)
        self.feature_history.append(features)

        # Performance optimization: only run inference every N steps
        if not force_inference and current_step % self.config.inference_every_n_steps != 0:
            if hasattr(self, '_cached_predictions') and self._cached_predictions:
                return self._cached_predictions

        # Need full sequence for prediction
        if len(self.feature_history) < self.config.sequence_length:
            # Pad with current features
            pad_count = self.config.sequence_length - len(self.feature_history)
            seq = np.array([features] * pad_count + list(self.feature_history))
        else:
            seq = np.array(list(self.feature_history))

        # Convert to tensor: [seq_len, nodes, features]
        x_seq = torch.tensor(seq, dtype=torch.float32, device=self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output, self.hidden_state = self.model(x_seq, self.hidden_state)

        # Parse predictions: output is [1, nodes, 1] (only congestion)
        output_np = output.squeeze(0).cpu().numpy()

        predictions = {}
        high_risk_rsus = []

        for rsu_id in self.rsu_junctions:
            idx = self.rsu_to_idx[rsu_id]
            p_congestion_raw = float(output_np[idx, 0])

            # Apply EMA smoothing if enabled
            if self.config.use_ema_smoothing:
                if rsu_id in self.ema_predictions:
                    # Smooth prediction: new = alpha * raw + (1-alpha) * prev
                    p_congestion = (
                        self.config.ema_alpha * p_congestion_raw +
                        (1 - self.config.ema_alpha) * self.ema_predictions[rsu_id]
                    )
                else:
                    # First prediction - no smoothing
                    p_congestion = p_congestion_raw
                self.ema_predictions[rsu_id] = p_congestion
            else:
                p_congestion = p_congestion_raw

            # Reroute score based on congestion level (simple heuristic)
            # Higher congestion = higher reroute score
            reroute_score = min(p_congestion * 1.5, 1.0)

            # Determine risk level and reroute decision
            if p_congestion >= self.config.high_risk_threshold:
                risk_level = "high"
                max_reroute = self.config.max_reroute_fraction
                should_reroute = True
                high_risk_rsus.append(rsu_id)
            elif p_congestion >= self.config.medium_risk_threshold:
                risk_level = "medium"
                max_reroute = self.config.max_reroute_fraction * 0.5
                should_reroute = reroute_score > 0.5
            else:
                risk_level = "low"
                max_reroute = 0.0
                should_reroute = False

            reroute_fraction = reroute_score * max_reroute

            # Confidence based on prediction certainty
            confidence = 0.5 + 0.5 * abs(2 * p_congestion - 1)

            predictions[rsu_id] = {
                "p_congestion": p_congestion,
                "p_congestion_raw": p_congestion_raw,  # Store unsmoothed for debugging
                "reroute_fraction": reroute_fraction,
                "risk_level": risk_level,
                "confidence": confidence,
                "should_reroute": should_reroute,
                "avoid_rsus": [],  # Will be filled below
            }

        # Add avoid list to each RSU (other high-risk RSUs to avoid)
        for rsu_id in predictions:
            predictions[rsu_id]["avoid_rsus"] = [r for r in high_risk_rsus if r != rsu_id]

        # Cache predictions for performance
        self._cached_predictions = predictions

        return predictions

    def get_reroute_decision(
        self,
        rsu_states: dict[str, dict],
        current_step: int = 0,
    ) -> dict[str, Any]:
        """Get a global rerouting decision summary for the traffic controller.

        This method provides a simplified interface for the SUMO pipeline to
        integrate T-GCN predictions into actual rerouting logic.

        Returns:
            Dictionary with:
                - should_reroute: Boolean - whether rerouting is recommended
                - high_risk_rsus: List of RSU IDs with high congestion
                - medium_risk_rsus: List of RSU IDs with medium congestion
                - reroute_fraction: Suggested fraction of vehicles to reroute (0-1)
                - edge_penalties: Dict mapping RSU -> penalty factor for routing
                - confidence: Overall prediction confidence
        """
        predictions = self.predict(rsu_states, current_step, force_inference=True)

        high_risk = []
        medium_risk = []
        total_congestion = 0.0
        max_congestion = 0.0
        confidence_sum = 0.0

        for rsu_id, pred in predictions.items():
            total_congestion += pred["p_congestion"]
            max_congestion = max(max_congestion, pred["p_congestion"])
            confidence_sum += pred["confidence"]

            if pred["risk_level"] == "high":
                high_risk.append(rsu_id)
            elif pred["risk_level"] == "medium":
                medium_risk.append(rsu_id)

        avg_congestion = total_congestion / max(1, len(predictions))
        avg_confidence = confidence_sum / max(1, len(predictions))

        # Compute edge penalties for routing (higher = avoid more)
        edge_penalties = {}
        for rsu_id, pred in predictions.items():
            if pred["p_congestion"] >= self.config.medium_risk_threshold:
                # Penalty proportional to congestion probability
                penalty = 1.0 + pred["p_congestion"] * 10.0  # 1x to 11x penalty
                edge_penalties[rsu_id] = penalty

        # Determine if global rerouting should happen (MORE SELECTIVE)
        # Require either:
        # - Multiple high-risk RSUs (>= 2) to avoid false positives
        # - OR sustained high average congestion with high confidence
        should_reroute = (
            len(high_risk) >= 2 or  # Changed from > 0 to >= 2
            (len(high_risk) >= 1 and avg_congestion > 0.5 and avg_confidence > 0.7)
        )

        # Suggested reroute fraction based on overall congestion
        if max_congestion >= self.config.high_risk_threshold:
            reroute_fraction = self.config.max_reroute_fraction
        elif max_congestion >= self.config.medium_risk_threshold:
            reroute_fraction = self.config.max_reroute_fraction * 0.5
        else:
            reroute_fraction = 0.0

        return {
            "should_reroute": should_reroute,
            "high_risk_rsus": high_risk,
            "medium_risk_rsus": medium_risk,
            "reroute_fraction": reroute_fraction,
            "edge_penalties": edge_penalties,
            "avg_congestion": avg_congestion,
            "max_congestion": max_congestion,
            "confidence": avg_confidence,
        }

    def train_step(
        self,
        rsu_states: dict[str, dict],
        actual_congestion: dict[str, float],
        reward: float = 0.0,
    ) -> dict[str, float]:
        """Perform one training step with observed data.

        Args:
            rsu_states: Current RSU states
            actual_congestion: Observed congestion per RSU [0, 1]
            reward: Optional reward signal for RL-style training

        Returns:
            Dictionary of training metrics
        """
        if not self.training_enabled:
            return {"loss": 0.0, "skipped": True}

        # Build current features
        features = self._build_node_features(rsu_states)

        # Build target vector: [nodes, 1] - only congestion now
        targets = np.zeros((self.num_nodes, self.config.output_dim), dtype=np.float32)
        for rsu_id, congestion in actual_congestion.items():
            if rsu_id in self.rsu_to_idx:
                idx = self.rsu_to_idx[rsu_id]
                targets[idx, 0] = np.clip(congestion, 0, 1)

        # Store experience
        if len(self.feature_history) >= self.config.sequence_length:
            seq = np.array(list(self.feature_history))
            # Priority based on congestion level (prioritize learning from congested states)
            priority = np.mean(targets[:, 0]) + 0.5
            self.replay_buffer.push(seq, targets, priority)

        # Check if enough samples and past warmup
        if (len(self.replay_buffer) < self.config.batch_size or
            self.step_count < self.config.warmup_steps):
            return {"loss": 0.0, "buffer_size": len(self.replay_buffer), "warmup": True}

        # Sample batch
        batch_seqs, batch_targets, weights, indices = self.replay_buffer.sample(
            self.config.batch_size,
            beta=min(1.0, 0.4 + self.total_train_steps * 0.001)
        )

        if not batch_seqs:
            return {"loss": 0.0, "empty_batch": True}

        # Prepare tensors
        x_batch = torch.tensor(
            np.array(batch_seqs),
            dtype=torch.float32,
            device=self.device
        )
        y_batch = torch.tensor(
            np.array(batch_targets),
            dtype=torch.float32,
            device=self.device
        )
        w_batch = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device
        )

        # Training mode
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass: x_batch is [batch, seq, nodes, features]
        output, _ = self.model(x_batch, None)  # [batch, nodes, 2]

        # Compute loss with importance sampling
        # output: [batch, nodes, 2], y_batch: [batch, nodes, 2]
        loss_per_sample = self.criterion(output, y_batch).mean(dim=(1, 2))  # [batch]
        weighted_loss = (loss_per_sample * w_batch).mean()

        # Backward pass
        weighted_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()
        self.total_train_steps += 1

        # Update priorities
        td_errors = loss_per_sample.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Update learning rate scheduler
        self.scheduler.step(weighted_loss.item())

        # Update metrics
        preds = output.detach().cpu().numpy()
        targets_np = y_batch.cpu().numpy()
        for i in range(len(batch_seqs)):
            self.train_metrics.update(
                preds[i, :, 0],  # p_congestion predictions
                targets_np[i, :, 0],  # actual congestion
                loss_per_sample[i].item(),
                threshold=self.config.congestion_threshold
            )

        # Log metrics periodically
        metrics = {"loss": weighted_loss.item()}
        if self.step_count % self.config.log_interval == 0:
            all_metrics = self.train_metrics.get_all_metrics()
            metrics.update(all_metrics)
            self.metrics_history.append({
                "step": self.step_count,
                **all_metrics
            })
            print(f"[T-GCN] Step {self.step_count}: {self.train_metrics.format_metrics()}")

        # Checkpoint periodically
        if self.step_count % self.config.checkpoint_interval == 0:
            self.save(f"models/tgcn_checkpoint_{self.step_count}.pt")

        return metrics

    def evaluate(
        self,
        test_sequences: list[np.ndarray],
        test_targets: list[np.ndarray]
    ) -> dict[str, float]:
        """Evaluate model on test data.

        Args:
            test_sequences: List of [seq_len, nodes, features] arrays
            test_targets: List of [nodes, 2] target arrays

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.eval_metrics.reset()

        with torch.no_grad():
            for seq, target in zip(test_sequences, test_targets):
                x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                y = torch.tensor(target, dtype=torch.float32, device=self.device)

                output, _ = self.model(x, None)
                output = output.squeeze(0)  # [nodes, 2]

                # Update metrics
                self.eval_metrics.update(
                    output[:, 0].cpu().numpy(),
                    y[:, 0].cpu().numpy(),
                    threshold=self.config.congestion_threshold
                )

        return self.eval_metrics.get_all_metrics()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "train": self.train_metrics.get_all_metrics(),
            "eval": self.eval_metrics.get_all_metrics(),
            "history": self.metrics_history[-100:],  # Last 100 entries
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "seq_length": self.config.sequence_length,
                "learning_rate": self.config.learning_rate,
                "num_nodes": self.num_nodes,
            },
            "training_stats": {
                "total_steps": self.total_train_steps,
                "buffer_size": len(self.replay_buffer),
                "device": str(self.device),
            }
        }

    def save(self, path: str):
        """Save model weights, optimizer state, and metrics."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "step_count": self.step_count,
            "total_train_steps": self.total_train_steps,
            "metrics_history": self.metrics_history,
            "rsu_junctions": self.rsu_junctions,
        }
        torch.save(state, path)
        print(f"[T-GCN] Model saved to {path}")

    def load(self, path: str):
        """Load model weights and state from file."""
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.step_count = state.get("step_count", 0)
            self.total_train_steps = state.get("total_train_steps", 0)
            self.metrics_history = state.get("metrics_history", [])
            print(f"[T-GCN] Model loaded from {path} (step {self.step_count})")
        except Exception as e:
            print(f"[T-GCN] Warning: Could not load model from {path}: {e}")

    def reset(self):
        """Reset internal state for new episode."""
        self.feature_history.clear()
        self.hidden_state = None


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_pytorch_gnn_engine(
    road_graph: nx.Graph | nx.DiGraph,
    rsu_config_path: str,
    model_path: str | None = None,
) -> PyTorchGNNRerouteEngine:
    """Factory function to create PyTorch GNN engine from RSU config.

    Args:
        road_graph: NetworkX graph of road network
        rsu_config_path: Path to RSU configuration JSON
        model_path: Optional path to pretrained model

    Returns:
        Initialized PyTorchGNNRerouteEngine
    """
    with open(rsu_config_path, "r") as f:
        rsu_config = json.load(f)

    rsu_junctions = [rsu["junction_id"] for rsu in rsu_config["rsus"]]

    return PyTorchGNNRerouteEngine(
        road_graph=road_graph,
        rsu_junctions=rsu_junctions,
        model_path=model_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Test and Validation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Testing PyTorch T-GCN Implementation")
    print("=" * 70)

    # Create test graph
    G = nx.DiGraph()
    nodes = [f"rsu_{i}" for i in range(10)]
    G.add_nodes_from(nodes)
    for i in range(9):
        G.add_edge(nodes[i], nodes[i + 1])
        G.add_edge(nodes[i + 1], nodes[i])
    # Add some cross connections
    G.add_edge(nodes[0], nodes[5])
    G.add_edge(nodes[5], nodes[0])
    G.add_edge(nodes[2], nodes[7])
    G.add_edge(nodes[7], nodes[2])

    # Config
    config = TGCNConfig(
        node_feature_dim=8,
        hidden_dim=32,
        sequence_length=8,
        batch_size=16,
        buffer_size=500,
        warmup_steps=20,
        log_interval=10,
    )

    # Initialize engine
    engine = PyTorchGNNRerouteEngine(
        road_graph=G,
        rsu_junctions=nodes,
        config=config,
    )

    print("\n" + "=" * 70)
    print("Running Training Simulation...")
    print("=" * 70)

    # Simulate training
    for step in range(200):
        # Generate synthetic RSU states
        rsu_states = {}
        for rsu in nodes:
            # Simulate varying traffic
            base_count = 10 + 20 * np.sin(step / 20 + hash(rsu) % 10)
            rsu_states[rsu] = {
                "vehicle_count": int(max(0, base_count + np.random.normal(0, 5))),
                "avg_speed": max(5, 15 - base_count / 5 + np.random.normal(0, 2)),
                "occupancy": np.clip(base_count / 50, 0, 1),
                "queue_length": int(max(0, base_count / 3)),
                "incident": np.random.random() < 0.05,
            }

        # Predict
        predictions = engine.predict(rsu_states, step)

        # Generate ground truth (with some correlation to inputs)
        actual = {}
        for rsu in nodes:
            state = rsu_states[rsu]
            # Ground truth based on vehicle count and speed
            congestion = np.clip(
                state["vehicle_count"] / 40 * (1 - state["avg_speed"] / 20),
                0, 1
            )
            actual[rsu] = congestion + np.random.normal(0, 0.1)

        # Train
        metrics = engine.train_step(rsu_states, actual)

    print("\n" + "=" * 70)
    print("Final Metrics Summary")
    print("=" * 70)

    summary = engine.get_metrics_summary()
    print(f"\nTraining Metrics:")
    for key, value in summary["train"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nModel Config:")
    for key, value in summary["config"].items():
        print(f"  {key}: {value}")

    print(f"\nTraining Stats:")
    for key, value in summary["training_stats"].items():
        print(f"  {key}: {value}")

    # Test save/load
    test_path = "/tmp/tgcn_test.pt"
    engine.save(test_path)
    engine.load(test_path)

    print("\n" + "=" * 70)
    print("✓ PyTorch T-GCN Test Complete!")
    print("=" * 70)
