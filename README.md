# Hybrid AI Traffic Management System (V2X)

[![CI](https://github.com/Programmerlogic/final_year_project/actions/workflows/ci.yml/badge.svg)](https://github.com/Programmerlogic/final_year_project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SUMO](https://img.shields.io/badge/SUMO-1.18+-green.svg)](https://sumo.dlr.de/)

A hybrid AI traffic management stack built on SUMO simulation that combines **congestion forecasting**, **uncertainty-aware routing**, **adaptive signal control**, and **hybrid fusion** for V2X (Vehicle-to-Everything) networks.

## Key Features

- 🚦 **RL-Based Signal Control** - DQN-trained traffic light optimization with safety guardrails
- 🛣️ **Risk-Aware Routing** - Uncertainty-based path selection with congestion forecasting
- 🚑 **Emergency Vehicle Priority** - Corridor preemption and optimal routing for emergency vehicles
- 📡 **V2X Communication** - RSU-based telemetry and vehicle-to-infrastructure messaging
- 🗺️ **Real City Networks** - Kolkata central area with 19 named RSU locations
- 🔄 **Hybrid Fusion** - Integrated controller combining all components (+2.6% improvement over baseline)

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

# Kolkata city map with custom RSU names
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --rsu-config data/rsu_config_kolkata.json \
  --traffic-scale 1.5

# Full hybrid demo (all features enabled)
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --rsu-config data/rsu_config_kolkata.json \
  --traffic-scale 2.0 \
  --enable-rl-signal-control \
  --enable-emergency-priority \
  --enable-hybrid-uplink-stub \
  --enable-runtime-logging \
  --controlled-count 10 \
  --controlled-source ESPLANADE \
  --controlled-destination SEALDAH \
  --emergency-count 3 \
  --emergency-source PARK_STREET \
  --emergency-destination COLLEGE_STREET \
  --max-steps 600
```

See [docs/SUMO_FLAGS_REFERENCE.md](docs/SUMO_FLAGS_REFERENCE.md) for full flag reference.

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Pipeline (logging, labeling, splitting) | ✅ Complete |
| 2 | Congestion Forecasting (LightGBM, 87.3% acc, 91.3% F1) | ✅ Complete |
| 3 | Uncertainty-Aware Routing (risk score + confidence fallback) | ✅ Complete |
| 4 | Adaptive Signal Control (RL/DQN with safety guardrails) | ✅ Complete |
| 5 | Hybrid Fusion Controller (+2.6% improvement @ 3x traffic) | ✅ Complete |

## Kolkata RSU Locations

The system includes 19 strategically placed RSUs in central Kolkata:

| RSU ID | Location | Description |
|--------|----------|-------------|
| ESPLANADE | Esplanade | Central bus terminus, Metro hub |
| PARK_STREET | Park Street | Entertainment and restaurant district |
| SEALDAH | Sealdah Station | Major railway terminus |
| COLLEGE_STREET | College Street | Book market, Presidency University |
| DALHOUSIE | Dalhousie Square | Near Writers Building, historic area |
| BOWBAZAR | Bowbazar | Metro corridor, jewelry market |
| CHANDNI_CHOWK | Chandni Chowk | Near Metro station, major intersection |
| ... | ... | [See full list](data/README.md#rsu-configuration) |

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
| CLI Flags Reference | [docs/SUMO_FLAGS_REFERENCE.md](docs/SUMO_FLAGS_REFERENCE.md) |
| Forecasting Models | [models/forecast/README.md](models/forecast/README.md) |
| Data Pipeline | [pipelines/processing/README.md](pipelines/processing/README.md) |
| Controllers (RL/Fusion) | [controllers/README.md](controllers/README.md) |
| Routing | [routing/README.md](routing/README.md) |
| Evaluation | [evaluation/README.md](evaluation/README.md) |
| Experiments | [experiments/README.md](experiments/README.md) |
| Data & RSU Config | [data/README.md](data/README.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.
