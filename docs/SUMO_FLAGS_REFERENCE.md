# SUMO Pipeline - Complete Flag Reference

Comprehensive guide to all command-line flags for `sumo/run_sumo_pipeline.py`.

## Table of Contents
1. [Basic Configuration](#basic-configuration)
2. [RSU Configuration](#rsu-configuration)
3. [Traffic Control](#traffic-control)
4. [Controlled Test Vehicles](#controlled-test-vehicles)
5. [Emergency Vehicles](#emergency-vehicles)
6. [Utility Functions](#utility-functions)
7. [V2X Server Integration](#v2x-server-integration)
8. [Data Logging](#data-logging)
9. [Output Files](#output-files)
10. [RL Signal Control](#rl-signal-control)
11. [Example Commands](#example-commands)

---

## Basic Configuration

### `--contract CONTRACT`
- **Default**: `sumo/scenarios/sumo_contract.json`
- **Description**: Path to SUMO scenario contract JSON file
- **Example**: `--contract my_custom_contract.json`

### `--scenario {low,medium,high,demo,city,kolkata}`
- **Default**: `demo`
- **Description**: Scenario name from contract
- **Options**:
  - `low`: Low-demand scaffold
  - `medium`: Medium-demand scaffold
  - `high`: High-demand scaffold
  - `demo`: Hackathon demo (real-city 3D)
  - `city`: Real-city OpenStreetMap network
  - `kolkata`: Kolkata city network (36K edges, 6K junctions)
- **Example**: `--scenario kolkata`

### `--seed SEED`
- **Default**: `11`
- **Description**: SUMO random seed for reproducibility
- **Example**: `--seed 42`

### `--max-steps MAX_STEPS`
- **Default**: From contract (typically 3600)
- **Description**: Override contract max simulation steps
- **Example**: `--max-steps 1800` (30 minutes at 1 Hz)

### `--gui`
- **Default**: False (headless)
- **Description**: Use `sumo-gui` instead of `sumo` for visual simulation
- **Example**: `--gui`

### `--three-d`
- **Default**: False
- **Description**: Enable OpenSceneGraph 3D renderer (requires SUMO with OSG support)
- **Example**: `--gui --three-d`

### `--dry-run`
- **Default**: False
- **Description**: Print resolved command/config only; don't run simulation
- **Example**: `--dry-run`

---

## RSU Configuration

Road Side Units (RSUs) are placed at strategic junctions for V2X communication.

### `--rsu-range-m RSU_RANGE_M`
- **Default**: `300.0`
- **Description**: RSU coverage radius in meters (for GUI overlays)
- **Example**: `--rsu-range-m 500`

### `--rsu-min-inc-lanes RSU_MIN_INC_LANES`
- **Default**: `4`
- **Description**: Place RSU only on junctions with ≥ this many incoming lanes
- **Example**: `--rsu-min-inc-lanes 3` (include smaller junctions)

### `--rsu-max-count RSU_MAX_COUNT`
- **Default**: `40`
- **Description**: Maximum number of RSU circles to draw
- **Example**: `--rsu-max-count 50`

### `--rsu-min-spacing-m RSU_MIN_SPACING_M`
- **Default**: `1.8 * rsu-range-m` (540m if range=300m)
- **Description**: Minimum center-to-center spacing between RSUs
- **Example**: `--rsu-min-spacing-m 400`

### `--rsu-whitelist RSU_WHITELIST`
- **Default**: None (all RSUs active)
- **Description**: Comma-separated list of RSU aliases to keep active. Only these RSUs will be displayed and used for V2X communication. Useful for focusing on specific intersections.
- **Format**: Accepts `A`, `RSU_A`, or `RSU-A` format
- **Example**: `--rsu-whitelist "A,K,M,R,P,T,V,Y"` (keep only 8 specific RSUs)
- **Example**: `--rsu-whitelist "RSU_A,RSU_K,RSU_M"` (same with prefix)

### `--rsu-config RSU_CONFIG`
- **Default**: None (auto-detect RSUs)
- **Description**: Path to a JSON configuration file with custom RSU placements and real place names. When specified, this overrides auto-detection.
- **Example**: `--rsu-config data/rsu_config_kolkata.json`
- **JSON Format**:
```json
{
    "rsus": [
        {
            "id": "ESPLANADE",
            "display_name": "Esplanade",
            "junction_id": "663940665",
            "x": 2413.83,
            "y": 9027.56,
            "lat": 22.577782,
            "lon": 88.369245,
            "description": "Central bus terminus, Metro hub"
        }
    ]
}
```
- **Features**:
  - Custom RSU IDs and display names (shown in GUI)
  - Junction IDs validated against network
  - Coordinates used to find nearest junction if junction_id not exact match
  - Lat/lon stored for documentation purposes

---

## Traffic Control

### `--traffic-scale TRAFFIC_SCALE`
- **Default**: `1.0`
- **Description**: Global demand multiplier via SUMO `--scale`
  - `< 1.0`: Reduced traffic
  - `1.0`: Normal traffic
  - `> 1.0`: Heavy traffic (jam scenarios)
- **Example**: `--traffic-scale 1.5` (50% more vehicles)

### `--traffic-reduction-pct TRAFFIC_REDUCTION_PCT`
- **Default**: `0.0`
- **Description**: Optional traffic reduction percentage applied to traffic-scale
- **Example**: `--traffic-reduction-pct 20` (20% reduction)
- **Formula**: `final_scale = traffic_scale * (1 - reduction_pct/100)`

---

## Controlled Test Vehicles

Create a dedicated flow of AI-controlled vehicles for testing routing algorithms.

### `--controlled-count CONTROLLED_COUNT`
- **Default**: `0`
- **Description**: Number of controlled test vehicles
- **Example**: `--controlled-count 30`
- **⚠️ Requires**: `--controlled-source` and `--controlled-destination`

### `--controlled-source CONTROLLED_SOURCE`
- **Required if**: `--controlled-count > 0`
- **Description**: Source junction/edge ID for controlled vehicles
- **Example**: `--controlled-source 123456789`

### `--controlled-destination CONTROLLED_DESTINATION`
- **Required if**: `--controlled-count > 0`
- **Description**: Destination junction/edge ID for controlled vehicles
- **Example**: `--controlled-destination 987654321`

### `--controlled-via-rsus CONTROLLED_VIA_RSUS`
- **Default**: None
- **Description**: Comma-separated intermediate waypoints (junction IDs)
- **Example**: `--controlled-via-rsus "1234,5678,9012"`

### `--controlled-begin CONTROLLED_BEGIN`
- **Default**: `0`
- **Description**: Begin time for controlled vehicle flow (simulation seconds)
- **Example**: `--controlled-begin 100`

### `--controlled-end CONTROLLED_END`
- **Default**: `max_steps`
- **Description**: End time for controlled vehicle flow
- **Example**: `--controlled-end 1200`

---

## Emergency Vehicles

Simulate emergency vehicles with priority routing and traffic preemption.

### `--emergency-count EMERGENCY_COUNT`
- **Default**: `0`
- **Description**: Base emergency vehicle count (actual count = 3x this value)
- **Example**: `--emergency-count 5` (generates 15 emergency vehicles)

### `--emergency-source EMERGENCY_SOURCE`
- **Default**: Auto (random fringe)
- **Description**: Source junction/edge ID for emergency vehicles
- **Example**: `--emergency-source 123456789`

### `--emergency-destination EMERGENCY_DESTINATION`
- **Default**: Auto (random fringe)
- **Description**: Destination junction/edge ID for emergency vehicles
- **Example**: `--emergency-destination 987654321`

### `--emergency-via-rsus EMERGENCY_VIA_RSUS`
- **Default**: None
- **Description**: Comma-separated intermediate waypoints
- **Example**: `--emergency-via-rsus "1234,5678"`

### `--emergency-begin EMERGENCY_BEGIN`
- **Default**: `0`
- **Description**: Begin time for emergency vehicle flow
- **Example**: `--emergency-begin 200`

### `--emergency-end EMERGENCY_END`
- **Default**: `max_steps`
- **Description**: End time for emergency vehicle flow
- **Example**: `--emergency-end 1500`

### `--enable-emergency-priority`
- **Default**: False
- **Description**: Enable emergency-vehicle priority routing + corridor preemption
- **Example**: `--enable-emergency-priority`

### `--emergency-corridor-lookahead-edges EMERGENCY_CORRIDOR_LOOKAHEAD_EDGES`
- **Default**: `5`
- **Description**: Number of upcoming edges treated as emergency corridor
- **Example**: `--emergency-corridor-lookahead-edges 8`

### `--emergency-hold-seconds EMERGENCY_HOLD_SECONDS`
- **Default**: `10.0`
- **Description**: Duration to hold non-emergency traffic stopped on corridor
- **Example**: `--emergency-hold-seconds 15`

### `--emergency-priority-interval-steps EMERGENCY_PRIORITY_INTERVAL_STEPS`
- **Default**: `2`
- **Description**: Run emergency routing/corridor policy every N simulation steps to reduce per-step overhead on large maps
- **Example**: `--emergency-priority-interval-steps 2`

### `--marker-refresh-steps MARKER_REFRESH_STEPS`
- **Default**: `4`
- **Description**: Refresh controlled/emergency GUI vehicle markers every N steps (lower frequency improves GUI smoothness)
- **Example**: `--marker-refresh-steps 4`

---

## Utility Functions

### `--suggest-near-junction SUGGEST_NEAR_JUNCTION`
- **Description**: Print nearby valid junction IDs around given junction and exit
- **Example**: `--suggest-near-junction 123456789`
- **Use case**: Find valid source/destination junctions for controlled vehicles

### `--suggest-purpose {source,destination,checkpoint,any}`
- **Default**: `any`
- **Description**: Filter suggested junctions by suitability
- **Example**: `--suggest-purpose source`

### `--suggest-count SUGGEST_COUNT`
- **Default**: `10`
- **Description**: Number of nearest suggestions to print
- **Example**: `--suggest-count 20`

### `--list-rsus`
- **Description**: Print RSU aliases (A, B, ... AA) mapped to junction IDs and exit
- **Example**: `--list-rsus`

### `--auto-fallback-junctions`
- **Default**: False
- **Description**: Auto-replace invalid junctions with nearest valid ones
- **Example**: `--auto-fallback-junctions`

---

## V2X Server Integration

Connect SUMO simulation to the V2X central server for hybrid AI routing.

### `--enable-hybrid-uplink-stub`
- **Default**: False
- **Description**: Send periodic RSU batch payloads to server `/route` endpoint
- **Example**: `--enable-hybrid-uplink-stub`

### `--server-url SERVER_URL`
- **Default**: `http://localhost:5000`
- **Description**: Base URL for V2X central server
- **Example**: `--server-url http://127.0.0.1:5000`

### `--hybrid-batch-seconds HYBRID_BATCH_SECONDS`
- **Default**: `30.0`
- **Description**: Batch period for uplink payloads (simulation seconds)
- **Example**: `--hybrid-batch-seconds 60`

### `--route-timeout-seconds ROUTE_TIMEOUT_SECONDS`
- **Default**: `2.0`
- **Description**: HTTP timeout for server `/route` call
- **Example**: `--route-timeout-seconds 5`

### `--reroute-highlight-seconds REROUTE_HIGHLIGHT_SECONDS`
- **Default**: `5.0`
- **Description**: Duration to keep GUI highlight on rerouted vehicles
- **Example**: `--reroute-highlight-seconds 10`

---

## Data Logging

Phase 1 data pipeline: log telemetry at 1 Hz for ML training.

### `--enable-runtime-logging`
- **Default**: False
- **Description**: Enable Phase-1 1 Hz logging to `data/raw/<run_id>/`
- **Output Files**:
  - `rsu_features_1hz.csv`: Per-RSU telemetry
  - `edge_flow_1hz.csv`: Per-edge traffic flow
  - `logger_manifest.json`: Run metadata
- **Example**: `--enable-runtime-logging`

### `--runtime-log-root RUNTIME_LOG_ROOT`
- **Default**: `data/raw`
- **Description**: Output root directory for runtime logs
- **Example**: `--runtime-log-root /tmp/sumo_logs`

### `--runtime-log-run-id RUNTIME_LOG_RUN_ID`
- **Default**: Auto (`timestamp_scenario_seed`)
- **Description**: Explicit run ID for runtime logs
- **Example**: `--runtime-log-run-id kolkata_test_001`

---

## Output Files

Generate SUMO XML outputs for post-simulation analysis.

### `--statistics-output STATISTICS_OUTPUT`
- **Default**: None
- **Description**: SUMO statistics XML output path (`--statistic-output`)
- **Example**: `--statistics-output results/stats.xml`

### `--summary-output SUMMARY_OUTPUT`
- **Default**: None
- **Description**: SUMO summary XML output path (`--summary-output`)
- **Example**: `--summary-output results/summary.xml`

### `--tripinfo-output TRIPINFO_OUTPUT`
- **Default**: None
- **Description**: SUMO tripinfo XML output path (`--tripinfo-output`)
- **Example**: `--tripinfo-output results/tripinfo.xml`

### `--tripinfo-write-unfinished`
- **Default**: False
- **Description**: Include unfinished vehicles in tripinfo output
- **Example**: `--tripinfo-write-unfinished`

### `--kpi-output-dir KPI_OUTPUT_DIR`
- **Default**: None
- **Description**: Auto-generate KPI XML files (statistics/summary/tripinfo)
- **Example**: `--kpi-output-dir evaluation/hackathon_kpi`
- **Generates**:
  - `<prefix>_statistics.xml`
  - `<prefix>_summary.xml`
  - `<prefix>_tripinfo.xml`

### `--kpi-output-prefix KPI_OUTPUT_PREFIX`
- **Default**: Auto (`timestamp_scenario_seed`)
- **Description**: Filename prefix for auto-named KPI files
- **Example**: `--kpi-output-prefix kolkata_demo`

---

## RL Signal Control

Phase 4: Adaptive traffic signal control with Deep Q-Networks.

### `--enable-rl-signal-control`
- **Default**: False
- **Description**: Enable Phase-4 RL adaptive traffic signal control
- **Fallback**: Uses SimpleActuated if no model weights found
- **Example**: `--enable-rl-signal-control`

### `--rl-model-dir RL_MODEL_DIR`
- **Default**: `models/rl/artifacts/latest`
- **Description**: Path to DQN weights directory
- **Example**: `--rl-model-dir models/rl/artifacts/phase4_v2`

### `--rl-tls-ids RL_TLS_IDS`
- **Default**: Auto-discover traffic lights (bounded by `--rl-max-controlled-tls`)
- **Description**: Comma-separated TLS junction IDs to control
- **Example**: `--rl-tls-ids "123456,789012,345678"`

### `--rl-min-green-seconds RL_MIN_GREEN_SECONDS`
- **Default**: `15.0`
- **Description**: Minimum green duration (safety guardrail)
- **Example**: `--rl-min-green-seconds 10`

### `--rl-yellow-duration-seconds RL_YELLOW_DURATION_SECONDS`
- **Default**: `3.0`
- **Description**: Yellow transition window between phases
- **Example**: `--rl-yellow-duration-seconds 4`

### `--rl-max-switches-per-window RL_MAX_SWITCHES_PER_WINDOW`
- **Default**: `4`
- **Description**: Max phase switches per 60-s rolling window (anti-oscillation)
- **Example**: `--rl-max-switches-per-window 6`

### `--rl-max-controlled-tls RL_MAX_CONTROLLED_TLS`
- **Default**: `96`
- **Description**: Upper bound on auto-controlled traffic lights for large maps (`0` disables limit)
- **Example**: `--rl-max-controlled-tls 120`

### `--rl-step-interval-steps RL_STEP_INTERVAL_STEPS`
- **Default**: `2`
- **Description**: Apply RL control every N simulation steps to reduce control-loop overhead
- **Example**: `--rl-step-interval-steps 2`

---

## Example Commands

### 1. Quick Demo (GUI, 2 minutes)
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario demo \
  --gui \
  --max-steps 120
```

### 2. Kolkata Map (Reduced Traffic)
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 600 \
  --traffic-scale 0.5 \
  --seed 42
```

### 3. Full V2X System (Server + SUMO)

**Terminal 1** (Server):
```bash
python3 server.py
```

**Terminal 2** (SUMO):
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --traffic-scale 1.0 \
  --enable-hybrid-uplink-stub \
  --server-url http://127.0.0.1:5000 \
  --hybrid-batch-seconds 30
```

### 4. Emergency Vehicle Priority
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --traffic-scale 0.8 \
  --emergency-count 5 \
  --enable-emergency-priority \
  --emergency-corridor-lookahead-edges 8
```

### 5. Data Collection Run (Phase 1)
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --max-steps 3600 \
  --traffic-scale 1.0 \
  --seed 42 \
  --enable-runtime-logging \
  --runtime-log-run-id kolkata_run_001
```

### 6. RL Signal Control (Phase 4)
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --traffic-scale 1.2 \
  --enable-rl-signal-control \
  --rl-model-dir models/rl/artifacts/latest \
  --rl-min-green-seconds 12
```

### 7. Full-Featured Demo
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --traffic-scale 0.8 \
  --seed 42 \
  --rsu-range-m 500 \
  --rsu-max-count 40 \
  --emergency-count 5 \
  --enable-emergency-priority \
  --enable-hybrid-uplink-stub \
  --server-url http://127.0.0.1:5000 \
  --enable-rl-signal-control \
  --enable-runtime-logging \
  --runtime-log-run-id kolkata_full_demo
```

### 8. Controlled Test Vehicles

First, find valid junctions:
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --list-rsus
```

Then run with controlled vehicles:
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --controlled-count 30 \
  --controlled-source 123456789 \
  --controlled-destination 987654321 \
  --enable-hybrid-uplink-stub
```

### 9. KPI Evaluation Run
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --max-steps 3600 \
  --traffic-scale 1.0 \
  --seed 42 \
  --kpi-output-dir evaluation/hackathon_kpi \
  --kpi-output-prefix kolkata_baseline \
  --tripinfo-write-unfinished
```

### 10. Dry Run (Test Configuration)
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --traffic-scale 1.5 \
  --max-steps 1800 \
  --dry-run
```

---

## Flag Combinations

### Minimal (Quick Test)
```bash
--scenario kolkata --gui --max-steps 120
```

### Typical Development
```bash
--scenario kolkata --gui --max-steps 600 --traffic-scale 0.5 --seed 42
```

### Full Demo (No Server)
```bash
--scenario kolkata --gui --max-steps 1800 --traffic-scale 0.8 \
--emergency-count 5 --enable-emergency-priority \
--enable-runtime-logging
```

### Production (Headless + Logging)
```bash
--scenario kolkata --max-steps 3600 --traffic-scale 1.0 --seed 42 \
--enable-runtime-logging --runtime-log-run-id prod_run_001 \
--kpi-output-dir results --tripinfo-write-unfinished
```

---

## Tips

1. **Start Small**: Use `--max-steps 120` and `--traffic-scale 0.5` for quick tests
2. **Use Seeds**: Always specify `--seed` for reproducible results
3. **Check Junctions**: Use `--list-rsus` to find valid junction IDs before using controlled vehicles
4. **Monitor Performance**: Kolkata network is large - expect 150-300ms per step
5. **Server First**: Always start `server.py` before using `--enable-hybrid-uplink-stub`
6. **Dry Run**: Test configuration with `--dry-run` before long simulations
7. **Emergency Priority**: Combine `--emergency-count` with `--enable-emergency-priority` for realistic behavior

---

## See Also

- **SUMO Documentation**: `sumo/README.md`
- **Kolkata Map Guide**: `docs/KOLKATA_MAP_INTEGRATION.md`
- **Server Integration**: Root `README.md` - V2X Architecture section
- **RL Control**: `controllers/README.md`
- **Data Pipeline**: `pipelines/processing/README.md`
