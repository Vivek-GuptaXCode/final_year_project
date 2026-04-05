# Controllers

This directory contains Phase 4 (Adaptive Signal Control) and Phase 5 (Hybrid Fusion) controllers.

## Directory Structure

```
controllers/
├── rl/                      # Phase 4: Reinforcement Learning Traffic Signal Control
│   ├── dqn_agent.py         # Base Deep Q-Network agent
│   ├── improved_dqn_agent.py # Enhanced DQN with Double DQN
│   ├── traffic_signal_env.py # SUMO traffic signal environment
│   ├── train_phase4.py      # Training pipeline
│   ├── inference_hook.py    # Model deployment for SUMO integration
│   ├── baselines.py         # Fixed-time baseline controller
│   ├── safety_guardrails.py # Safety constraints and bounds
│   └── policies/            # Trained policy checkpoints
│
└── fusion/                  # Phase 5: Hybrid Fusion Controller
    ├── fusion_orchestrator.py # Coordinates routing + signal control
    ├── ablation_configs.py    # 9 ablation experiment configurations
    └── run_ablation.py        # Ablation study runner
```

## Phase 4: Reinforcement Learning Signal Control ✅

### Overview

The RL module implements adaptive traffic signal control using Deep Q-Networks (DQN). The agent learns optimal signal timing policies by interacting with the SUMO traffic simulation.

### Key Components

- **DQNAgent**: Base implementation with experience replay and target network
- **ImprovedDQNAgent**: Enhanced version with Double DQN for reduced overestimation
- **TrafficSignalEnv**: OpenAI Gym-compatible environment wrapping SUMO
- **SafetyGuardrails**: Enforces minimum green times and maximum phase durations

### Training

```bash
# Smoke test (quick training run)
python3 controllers/rl/train_phase4.py --profile smoke

# Medium training run
python3 controllers/rl/train_phase4.py --profile medium

# Full training
python3 controllers/rl/train_phase4.py --profile full
```

### Inference

The `inference_hook.py` module provides real-time signal control decisions:

```python
from controllers.rl.inference_hook import RLInferenceHook

hook = RLInferenceHook(model_path="models/rl/artifacts/latest")
action = hook.get_action(state_vector)
```

## Phase 5: Hybrid Fusion Controller ✅

### Overview

The fusion module combines Phase 3 (risk-aware routing) and Phase 4 (RL signal control) into a unified controller that coordinates both systems for optimal traffic management.

**Key Results**: +2.6% improvement over baseline at 3x traffic scale with conservative routing parameters.

### Key Optimizations

1. **Conservative Routing** - Max 12% reroute fraction prevents route oscillation
2. **Cooldown Period** - 45-second minimum between rerouting decisions
3. **Selective Rerouting** - Only reroutes vehicles in congested RSU zones
4. **SUMO Actuated Baseline** - Fair comparison against adaptive signals

### Ablation Studies

Nine configurations for systematic evaluation:

| Config | Routing | Signal Control | Description |
|--------|---------|----------------|-------------|
| `full_hybrid` | Risk-aware | RL | Full system (+2.6% @ 3x) |
| `routing_only` | Risk-aware | Fixed | Routing without RL |
| `signal_only` | None | RL | RL signals only |
| `baseline` | None | Actuated | SUMO actuated (baseline) |
| `no_uncertainty` | Probability only | RL | No uncertainty estimation |
| ... | ... | ... | ... |

### Running Ablation Studies

```bash
# Single configuration
python3 controllers/fusion/run_ablation.py --config full_hybrid --steps 1800

# All configurations
python3 controllers/fusion/run_ablation.py --all --steps 1800

# High traffic stress test (3x scale)
python3 controllers/fusion/run_ablation.py --config full_hybrid --steps 1800 --traffic-scale 3.0
```

## Integration with SUMO

The controllers integrate with SUMO via the `sumo_adapter.py`:

```python
# In sumo/run_sumo_pipeline.py
from controllers.rl.inference_hook import RLInferenceHook

# Initialize RL hook
rl_hook = RLInferenceHook(model_path="models/rl/artifacts/latest")

# In simulation loop
for step in range(max_steps):
    state = adapter.get_traffic_state()
    action = rl_hook.get_action(state)
    adapter.apply_signal_action(action)
```

## Model Artifacts

Trained models are stored in `models/rl/artifacts/`:

```
models/rl/artifacts/
├── latest/                    # Current production model
│   ├── weights.npz            # Model weights
│   └── meta.json              # Training metadata
└── phase4_YYYYMMDDTHHMMSS/    # Timestamped checkpoints
```

## KPI Metrics

Phase 4 evaluation metrics (see `evaluation/`):

| Metric | Target | Description |
|--------|--------|-------------|
| Average Wait Time | ↓ 15% vs baseline | Mean vehicle wait at signals |
| Throughput | ↑ 10% vs baseline | Vehicles processed per hour |
| Queue Length | ↓ 20% vs baseline | Average queue at intersections |
