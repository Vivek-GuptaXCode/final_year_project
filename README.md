# Hybrid AI Traffic Management System (V2X)

[![CI](https://github.com/Programmerlogic/final_year_project/actions/workflows/ci.yml/badge.svg)](https://github.com/Programmerlogic/final_year_project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SUMO](https://img.shields.io/badge/SUMO-1.18+-green.svg)](https://sumo.dlr.de/)

A hybrid AI traffic management stack built on SUMO simulation that combines **congestion forecasting**, **uncertainty-aware routing**, and **adaptive signal control** for V2X (Vehicle-to-Everything) networks.

## Architecture

```
Vehicle (OBU)  -->  RSU (junction)  -->  Central Server  -->  Route Policy
     ^                                                           |
     '------------- SUMO simulation loop (1 Hz) ----------------'
```

1. Vehicles in SUMO send telemetry to the nearest RSU
2. RSUs batch metrics (vehicle count, speed, latency, congestion flags) and uplink to the server via SocketIO
3. Server runs the forecasting model + Phase 3 risk-aware router, returns routing policy
4. SUMO pipeline applies rerouting decisions via TraCI/libsumo

## Project Structure

```
server.py                    # V2X central server (Flask + SocketIO)
sumo/
  run_sumo_pipeline.py       # SUMO simulation orchestrator
  sumo_adapter.py            # TraCI/libsumo wrapper
  networks/                  # .net.xml road networks
  routes/                    # .rou.xml demand definitions
  scenarios/                 # .sumocfg scenario configs
models/
  forecast/
    feature_builder_v2.py    # 31-feature contract (lags, rolling stats, dynamics)
    train_phase2_improved.py # LightGBM/XGBoost/MLP training
    inference.py             # ForecastInferenceEngine (serves p_congestion + uncertainty)
    artifacts/latest/        # Active model artifact
routing/
  phase3_risk_router.py      # Confidence-aware risk routing
  route_audit_logger.py      # JSONL decision audit trail
pipelines/
  processing/                # Data pipeline: labeling, splitting, leakage validation
  logging/                   # 1 Hz runtime logger for SUMO
evaluation/
  phase3_comparison.py       # Baseline vs risk-aware routing comparison
config/experiments/          # Experiment contract configs
data/                        # Raw logs, processed datasets, train/val/test splits
docs/
  HYBRID_AI_STRICT_CHECKLIST.md  # Phase gate checklist (authoritative)
  reports/                       # Phase completion reports
```

## Quick Start

### 1. Start the server

```bash
# Basic (deterministic routing only)
python3 server.py

# With forecast model + Phase 3 risk-aware routing
HYBRID_ENABLE_FORECAST_MODEL=1 \
HYBRID_ENABLE_PHASE3_ROUTING=1 \
python3 server.py
```

### 2. Run SUMO simulation

```bash
# Smoke test (GUI, 2 minutes)
python3 sumo/run_sumo_pipeline.py --scenario demo --gui --max-steps 120

# Kolkata city map (large network)
python3 sumo/run_sumo_pipeline.py --scenario kolkata --gui --max-steps 300 --traffic-scale 0.5

# Full pipeline with server uplink + emergency priority
python3 sumo/run_sumo_pipeline.py \
  --scenario demo --gui --max-steps 1800 \
  --traffic-scale 1.8 \
  --enable-hybrid-uplink-stub \
  --server-url http://127.0.0.1:5000 \
  --enable-emergency-priority \
  --controlled-count 25 \
  --controlled-source RSU_A \
  --controlled-destination RSU_K
```

See `sumo/README.md` for full flag reference.

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Pipeline (logging, labeling, splitting) | ✅ Complete |
| 2 | Congestion Forecasting (LightGBM, 87.3% acc, 91.3% F1) | ✅ Complete |
| 3 | Uncertainty-Aware Routing (risk score + confidence fallback) | ✅ Complete |
| 4 | Adaptive Signal Control (RL/MARL) | ✅ Complete |
| 5 | Hybrid Fusion Controller | 🔄 In Progress |

## Requirements

- Python 3.10+
- SUMO 1.18+ (with TraCI or libsumo)
- Dependencies: `pip install -r requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/Programmerlogic/final_year_project.git
cd final_year_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify SUMO installation
sumo --version
```

## Documentation

| Module | README |
|--------|--------|
| SUMO Simulation | [sumo/README.md](sumo/README.md) |
| Forecasting Models | [models/forecast/README.md](models/forecast/README.md) |
| Data Pipeline | [pipelines/processing/README.md](pipelines/processing/README.md) |
| Controllers (RL/Fusion) | [controllers/README.md](controllers/README.md) |
| Routing | [routing/README.md](routing/README.md) |
| Evaluation | [evaluation/README.md](evaluation/README.md) |
| Experiments | [experiments/README.md](experiments/README.md) |
| Data | [data/README.md](data/README.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.
