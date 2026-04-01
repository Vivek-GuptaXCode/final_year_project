# Phase 4 Report — Adaptive Signal Control (RL/MARL)

- **Generated:** 2026-03-29
- **Environment:** `controllers/rl/traffic_signal_env.py` — `TrafficSignalEnv` / `MultiJunctionEnv`
- **Agent:** `controllers/rl/dqn_agent.py` — `DQNAgent` (numpy DQN, 2-layer MLP)
- **Safety layer:** `controllers/rl/safety_guardrails.py` — `TLSSafetyGuardrail`
- **Baselines:** `controllers/rl/baselines.py` — `FixedTimePolicy`, `SimpleActuatedPolicy`
- **Training driver:** `controllers/rl/train_phase4.py`
- **Inference hook:** `controllers/rl/inference_hook.py` — `RLSignalController`
- **KPI results:** `evaluation/phase4_kpi_results.json` (written by `train_phase4.py`)

---

## 1. What Changed: Fixed-Time Control → RL Adaptive Signal Control

### Pre-Phase-4 baseline (SUMO default)
SUMO's built-in traffic signal programs are either:
- **Fixed-time**: phase rotates every N seconds regardless of demand.
- **Static actuated**: hard-coded vehicle-actuated extension rules, no learning.

No per-step optimisation, no uncertainty propagation, no coordination between junctions.

### Phase 4 policy stack

| Component | Description |
|-----------|-------------|
| `TrafficSignalEnv` | Gymnasium-compatible per-junction environment (no gym dependency). Obs dim = 42. |
| `MultiJunctionEnv` | MARL wrapper: K-hop neighbour congestion appended to each obs (dim = 43). |
| `DQNAgent` | 2-layer MLP (hidden=64), numpy only — He init, ReLU, experience replay (20k), target network (hard copy every 200 steps), ε-greedy decay. |
| `TLSSafetyGuardrail` | Hard constraints: min_green=15 s, yellow_transition=3 s, anti-oscillation (≤4 switches/60 s). Never skipped. |
| `FixedTimePolicy` | Phase rotates every `cycle/n_phases` seconds — pure timer. |
| `SimpleActuatedPolicy` | Extends green if current density ≥ threshold; switches if cross-phase demand rises. |

---

## 2. Observation Space (OBS_DIM = 42)

| Range | Content | Normalisation |
|-------|---------|---------------|
| `[0:8]` | Current phase one-hot | Binary |
| `[8]` | Phase elapsed | ÷ 120 s → [0, 1] |
| `[9:25]` | Per-lane halting count (16 lanes, padded) | ÷ 20 vehicles → [0, 1] |
| `[25:41]` | Per-lane occupancy (16 lanes, padded) | ÷ 100 % → [0, 1] |
| `[41]` | Emergency vehicle on controlled approach | Binary |

Multi-agent augmentation appends mean neighbour congestion (index 42 → `OBS_DIM+1 = 43`).

---

## 3. Action Space

| Action | Meaning |
|--------|---------|
| `0` | KEEP — maintain current green phase |
| `1` | SWITCH — advance to `(current_phase + 1) % n_phases` |

Safety guardrail may override SWITCH → KEEP if:
- Phase elapsed < `min_green_seconds` (default 15 s)
- Currently in yellow transition window (default 3 s)
- Switch rate in last 60 s ≥ `max_switches_per_window` (default 4)

---

## 4. Reward Function

```
r = −(total_halting / (MAX_LANES × 20)) × halt_weight
  + (vehicles_cleared / 10) × throughput_weight
```

Default weights: `halt_weight=1.0`, `throughput_weight=0.5`.

---

## 5. DQN Architecture

| Parameter | Value |
|-----------|-------|
| Input dim | 42 |
| Hidden dim | 64 (configurable) |
| Output dim | 2 |
| Activation | ReLU |
| Initialisation | He (scale = √(2/fan_in)) |
| Optimiser | SGD, lr=1e-3 |
| Discount γ | 0.99 |
| ε start / min / decay | 1.0 / 0.05 / 0.99 (per episode) |
| Replay buffer | 20 000 transitions (circular deque) |
| Batch size | 64 |
| Target network | Hard copy every 200 gradient steps |

Backpropagation is implemented manually with numpy — no autograd framework required.

---

## 6. Gate P4.1 — Environment API Reproducible

`TrafficSignalEnv` and `MultiJunctionEnv` satisfy:
- Fixed `OBS_DIM = 42` regardless of junction phase count or lane count (padded).
- Deterministic observation given TraCI state (no random noise).
- `EnvConfig` fully parameterises reward weights and guardrail thresholds.
- Importable and testable without SUMO running (deferred TraCI calls).

**Smoke test (offline):** DQN forward + backprop, guardrail constraint enforcement,
FixedTime and SimpleActuated policy selection — all verified with `python3` unit test.

Gate **PASS** — environment API is reproducible and testable without SUMO.

---

## 7. Gate P4.2 — Single-Agent Policy vs Fixed-Time Baseline

Run `train_phase4.py` to generate `evaluation/phase4_kpi_results.json`.

Evaluation metric: mean normalised halting across `eval_episodes` episodes.

Pass condition: `DQN_halting ≤ FixedTime_halting × 1.05` (5% tolerance accounts for
variance in short evaluation runs; a full `--profile full` training run is expected
to show DQN ≥5% reduction in mean halting).

Gate **PENDING EXECUTION** — run training script on paired SUMO sessions.

```bash
python3 controllers/rl/train_phase4.py \
    --scenario city \
    --profile medium \
    --output-dir models/rl/artifacts
```

---

## 8. Gate P4.3 — Multi-Agent Run Stable Under Mixed Demand

`MultiJunctionEnv` wraps all discovered TLS junctions.  Each agent acts
independently; coordination is through shared neighbour congestion signal
(K=2 ring topology by default).

`train_phase4.py` runs a MARL evaluation pass after single-agent training
and checks that per-episode mean rewards do not diverge (no NaN, no collapse).

Gate **PENDING EXECUTION** (same run as P4.2 above).

---

## 9. Gate P4.4 — Safety Constraints Never Violated

`TLSSafetyGuardrail` is architecturally interposed between every policy output
and every TraCI `setPhase()` call.  There is no code path that bypasses it.

Constraint enforcement:

| Constraint | Mechanism |
|-----------|-----------|
| Minimum green | `phase_elapsed < min_green_s → return KEEP` |
| Yellow transition | `in_yellow_until > sim_time → apply yellow state → return KEEP` |
| Anti-oscillation | `recent_switches ≥ max_switches → return KEEP` |
| Emergency override | Emergency vehicles routed by Phase 3 pipeline (not TLS phase), not by RL |

`violations_blocked(tls_id)` counter accumulates any blocked SWITCH requests for
post-run audit.

Gate **PASS** — structural enforcement verified; no bypass path exists.

---

## 10. Integration with run_sumo_pipeline.py

Activate with:

```bash
python3 sumo/run_sumo_pipeline.py \
    --scenario city \
    --gui \
    --enable-rl-signal-control \
    --rl-model-dir models/rl/artifacts \
    --rl-min-green-seconds 15 \
    --rl-yellow-duration-seconds 3
```

Without `--rl-model-dir` (or if weights not found), falls back to `SimpleActuatedPolicy`.

Combined with full hybrid stack:

```bash
HYBRID_ENABLE_FORECAST_MODEL=1 \
HYBRID_ENABLE_PHASE3_ROUTING=1 \
python3 server.py
```

```bash
python3 sumo/run_sumo_pipeline.py \
    --scenario demo \
    --gui \
    --max-steps 1800 \
    --traffic-scale 1.8 \
    --enable-hybrid-uplink-stub \
    --server-url http://127.0.0.1:5000 \
    --enable-emergency-priority \
    --enable-rl-signal-control \
    --rl-model-dir models/rl/artifacts
```

---

## 11. Configuration Reference

| Flag | Default | Effect |
|------|---------|--------|
| `--enable-rl-signal-control` | off | Activate Phase 4 adaptive signal control |
| `--rl-model-dir` | None | DQN weights directory; falls back to SimpleActuated if not set |
| `--rl-tls-ids` | None (auto) | Comma-separated junction IDs to control |
| `--rl-min-green-seconds` | 15 | Safety guardrail minimum green time |
| `--rl-yellow-duration-seconds` | 3 | Safety guardrail yellow window |
| `--rl-max-switches-per-window` | 4 | Anti-oscillation cap per 60-s window |

---

## 12. Phase 4 Gate Summary

| Gate | Requirement | Status |
|------|-------------|--------|
| P4.1 | Environment API reproducible and testable | **PASS** — OBS_DIM=42 fixed; smoke tests pass without SUMO |
| P4.2 | Single-agent policy ≤ fixed-time halting (5% tolerance) | **PENDING** — run `train_phase4.py --profile medium` |
| P4.3 | Multi-agent MARL stable under mixed demand | **PENDING** — same run |
| P4.4 | Safety constraints never violated | **PASS** — structural enforcement; no bypass path |

**Phase 4 status: P4.1 and P4.4 PASS; P4.2 and P4.3 pending training run.**
