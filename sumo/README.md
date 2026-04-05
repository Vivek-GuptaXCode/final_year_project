# SUMO Integration Scaffold

This folder holds SUMO assets and adapters for the Hybrid AI Traffic Management System.

## Quick Reference

For full CLI flag documentation, see **[docs/SUMO_FLAGS_REFERENCE.md](../docs/SUMO_FLAGS_REFERENCE.md)**.

## Structure

- `sumo/networks/`: road network files (`.net.xml`)
- `sumo/routes/`: demand/route files (`.rou.xml`)
- `sumo/scenarios/`: SUMO config files (`.sumocfg`) and scenario presets

## Available Scenarios

| Scenario | Description | Network Size |
|----------|-------------|--------------|
| `demo` | Real city hackathon demo (3D enabled) | Medium |
| `city` | City scenario with traffic lights | Medium |
| `kolkata` | Kolkata central area (19 RSU locations) | Large (36K edges) |
| `low` / `medium` / `high` | Traffic density variants | Small |

## Kolkata Scenario with Real RSU Names

The Kolkata scenario supports custom RSU configuration with real place names:

```bash
# List RSU locations with real names
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --rsu-config data/rsu_config_kolkata.json \
  --list-rsus

# Run GUI with named RSUs
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --rsu-config data/rsu_config_kolkata.json \
  --controlled-source ESPLANADE \
  --controlled-destination SEALDAH \
  --controlled-count 10
```

Available RSU locations include: `ESPLANADE`, `PARK_STREET`, `SEALDAH`, `COLLEGE_STREET`, `DALHOUSIE`, `BOWBAZAR`, `CHANDNI_CHOWK`, `MOULALI`, and more. See [data/README.md](../data/README.md) for the full list.

## Phase 1 objective
Connect SUMO step loop to data logging at 1 Hz for:
- RSU-level features,
- edge-level flow statistics.

## Runtime Logging (Phase 1 Day 1)
- Runtime logging is opt-in via `--enable-runtime-logging` to avoid changing default run behavior.
- Output layout:
	- `data/raw/<run_id>/rsu_features_1hz.csv`
	- `data/raw/<run_id>/edge_flow_1hz.csv`
	- `data/raw/<run_id>/logger_manifest.json`
- Optional control flags:
	- `--runtime-log-root` (default `data/raw`)
	- `--runtime-log-run-id` (default auto-generated timestamp + scenario + seed)

Example:

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario demo \
	--max-steps 300 \
	--enable-runtime-logging
```

## Optional KPI XML Outputs
- SUMO KPI XML outputs are opt-in and do not change default run behavior.
- You can provide explicit file paths:
	- `--statistics-output`
	- `--summary-output`
	- `--tripinfo-output`
- Or generate all three via one directory:
	- `--kpi-output-dir`
	- `--kpi-output-prefix` (optional; defaults to auto timestamp + scenario + seed)

Example (auto-generate all three KPI files under one directory):

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario demo \
	--dry-run \
	--kpi-output-dir data/raw/kpi \
	--kpi-output-prefix baseline_seed11
```

## Notes
- Run smoke, medium, and full profiles from this local repository runtime.
- Use staged profiles to control resource usage and keep iteration stable.
- Runtime-generated helper files (for example RSU overlay add-files) are not source assets.

## Default Hackathon Demo
- `demo` now points to the real-city scenario (`sumo/scenarios/city.sumocfg`).
- GUI runs use `sumo/scenarios/city_3d.settings.xml`.
- Default GUI mode is 2D so RSU POIs and RSU range polygons stay clearly visible.
- Use `--three-d` when you explicitly want OSG 3D mode.
- GUI runs auto-generate RSU overlays from network junctions into `sumo/scenarios/<scenario>_rsu_pois.add.xml` and load them via `--additional-files`:
	- RSU circular coverage polygon per junction (transparent fill, red circumference).
	- RSU alias label anchors and text (`RSU_A`, `RSU_B`, ...) are auto-placed to the road side (off lane centerlines).
- RSU coverage radius can be tuned with `--rsu-range-m` (default: `120`).
- RSU placement is filtered to major intersections by default:
	- only junctions with at least `4` incoming lanes (`--rsu-min-inc-lanes`),
	- spacing-based downsampling (`--rsu-min-spacing-m`),
	- hard cap on number of RSUs (`--rsu-max-count`, default `40`).
- In OSG 3D mode, overlay visibility can be limited; press `F` to switch camera mode or run without `--three-d` for debugging overlays.
- GUI settings now force POI text-param rendering for RSU labels (`poiTextParam=name`), so RSU alias text is visible by default in 2D GUI runs.
- If a generated overlay file looks stale, rerun the same command; it is recreated automatically.
- Recommended local smoke test command:

```bash
python3 sumo/run_sumo_pipeline.py --scenario demo --gui --max-steps 120 --rsu-range-m 120 --rsu-min-inc-lanes 4 --rsu-max-count 40
```

Optional 3D run:

```bash
python3 sumo/run_sumo_pipeline.py --scenario demo --gui --three-d --max-steps 120 --rsu-range-m 120
```

## Heavy Traffic (Jam) Mode
- Use `--traffic-scale` to globally multiply route demand at runtime.
- Keep this as your first knob for creating congestion without editing route assets.
- `--traffic-reduction-pct` is opt-in (default `0`), so demand is not reduced unless explicitly requested.
- Suggested starting values:
	- `1.6` to `2.0`: visibly dense
	- `2.2` to `3.0`: jam-like pressure around key intersections

Example (local smoke test, short horizon):

```bash
python3 sumo/run_sumo_pipeline.py --scenario city --gui --max-steps 300 --traffic-scale 2.4
```

## Controlled AI Vehicle Cohort (20-30 Cars)
- Use a dedicated generated flow to inject a deterministic cohort with fixed source, destination, and intermediate RSU waypoints.
- Controlled cohort vehicles are rendered with deep blue markers (deep blue circle highlight + deep blue vehicle color) so they are easy to separate from regular traffic.
- You can select RSUs by alias labels (`A`, `B`, `C`, ..., `AA`) instead of raw junction IDs.
- Alias input forms accepted in route flags: `A`, `RSU_A`, `RSU-AA`.
- Main flags:
	- `--controlled-count`
	- `--controlled-source`
	- `--controlled-destination`
	- `--controlled-via-rsus` (comma-separated)
	- `--controlled-begin`, `--controlled-end`
- ID mode must be consistent:
	- either all junction IDs, or
	- all edge IDs.
- Junction mode auto-enables `--junction-taz` internally.

Example (25 controlled vehicles with junction constraints):

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario city \
	--gui \
	--traffic-scale 2.2 \
	--controlled-count 25 \
	--controlled-source RSU_A \
	--controlled-destination RSU_K \
	--controlled-via-rsus RSU_C,RSU_F \
	--controlled-begin 120 \
	--controlled-end 1500
```

List aliases before choosing source/destination/via:

```bash
python3 sumo/run_sumo_pipeline.py --scenario city --list-rsus
```

Dry-run only (validate command + generated controlled flow file, no SUMO session):

```bash
python3 sumo/run_sumo_pipeline.py --scenario city --dry-run --traffic-scale 2.2 --controlled-count 25 --controlled-source 101343850 --controlled-destination 1293775049 --controlled-via-rsus 1292264851,1292264938
```

## Emergency Cohort (Less Frequent)
- Emergency vehicles can be injected as a separate low-frequency cohort with their own source, destination, and intermediate RSU waypoints.
- Emergency cohort vehicles are rendered with bright yellow markers (bright yellow circle highlight + bright yellow vehicle color).
- Emergency count is automatically scaled by 3x at generation time.
- Main flags:
	- `--emergency-count`
	- `--emergency-source`
	- `--emergency-destination`
	- `--emergency-via-rsus` (comma-separated)
	- `--emergency-begin`, `--emergency-end`

Example (`--emergency-count 3` generates 9 emergency vehicles over a long horizon):

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario city \
	--gui \
	--emergency-count 3 \
	--emergency-source RSU_B \
	--emergency-destination RSU_M \
	--emergency-via-rsus RSU_D,RSU_G \
	--emergency-begin 150 \
	--emergency-end 2400
```

## ID Selection Tips
- Junction IDs and edge IDs come from the scenario network file (`sumo/networks/city.net.xml`).
- Quick extraction examples:

```bash
rg '<junction ' sumo/networks/city.net.xml | head
```

```bash
rg '<edge id="' sumo/networks/city.net.xml | head
```

Generated controlled flow file location:
- `sumo/scenarios/<scenario>_controlled_group.rou.xml`

Generated emergency flow file location:
- `sumo/scenarios/<scenario>_emergency_group.rou.xml`

## Emergency Vehicle Priority Mode
- Normal traffic remains under hybrid fusion control.
- Emergency vehicles use priority override:
	- optimal route recomputation,
	- corridor preemption (non-emergency vehicles on corridor edges are temporarily stopped),
	- automatic release after timeout/passage.
- Emergency vehicles are detected by vehicle class `emergency` or type-id tokens such as `emergency`, `ambulance`, `fire`, `police`.

Flags:
- `--enable-emergency-priority`
- `--emergency-corridor-lookahead-edges` (default `6`)
- `--emergency-hold-seconds` (default `8`)

Example:

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario city \
	--gui \
	--enable-hybrid-uplink-stub \
	--enable-emergency-priority \
	--server-url http://localhost:5000
```

## Validated Long GUI Full-Pipeline Run
- This command set was validated locally with the latest updates (RSU aliases, controlled + emergency cohorts, hybrid uplink, emergency priority).
- Start server first (terminal 1):

```bash
python3 server.py
```

- Run SUMO GUI long test (terminal 2):

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario demo \
	--gui \
	--max-steps 1800 \
	--traffic-scale 1.8 \
	--enable-hybrid-uplink-stub \
	--server-url http://127.0.0.1:5000 \
	--hybrid-batch-seconds 5 \
	--enable-emergency-priority \
	--controlled-count 25 \
	--controlled-source RSU_A \
	--controlled-destination RSU_K \
	--controlled-via-rsus RSU_C,RSU_F \
	--controlled-begin 120 \
	--controlled-end 1500 \
	--emergency-count 3 \
	--emergency-source RSU_B \
	--emergency-destination RSU_M \
	--emergency-via-rsus RSU_D,RSU_G \
	--emergency-begin 150 \
	--emergency-end 2400 \
	--auto-fallback-junctions
```

- Same command as a single line:

```bash
python3 sumo/run_sumo_pipeline.py --scenario demo --gui --max-steps 1800 --traffic-scale 1.8 --enable-hybrid-uplink-stub --server-url http://127.0.0.1:5000 --hybrid-batch-seconds 5 --enable-emergency-priority --controlled-count 25 --controlled-source RSU_A --controlled-destination RSU_K --controlled-via-rsus RSU_C,RSU_F --controlled-begin 120 --controlled-end 1500 --emergency-count 3 --emergency-source RSU_B --emergency-destination RSU_M --emergency-via-rsus RSU_D,RSU_G --emergency-begin 150 --emergency-end 2400 --auto-fallback-junctions
```

Note: With the current runner behavior, `--emergency-count 3` produces an effective emergency cohort of 9 vehicles (3x).

## Full Hybrid Demo Command

Run all features at once:

```bash
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

## Forecast Artifact + Visible Hybrid Demo
- Start server with forecast artifact inference enabled (terminal 1):

```bash
HYBRID_ENABLE_FORECAST_MODEL=1 \
HYBRID_ENABLE_PHASE3_ROUTING=1 \
HYBRID_ROUTE_AUDIT_PATH=data/raw/route_audit/route_decisions.jsonl \
HYBRID_FORECAST_ARTIFACT=models/forecast/artifacts/latest/forecast_artifact.json \
python3 server.py
```

- Phase 3 routing is additive and rollback-safe.
	- Keep `HYBRID_ENABLE_PHASE3_ROUTING=1` to enable confidence-aware risk routing + audit logs.
	- Remove that variable (or set it to `0`) to return to legacy server policy behavior.

- Run SUMO GUI with hybrid uplink and visible reroute highlighting (terminal 2):

```bash
python3 sumo/run_sumo_pipeline.py \
	--scenario demo \
	--gui \
	--max-steps 1800 \
	--traffic-scale 1.8 \
	--enable-hybrid-uplink-stub \
	--server-url http://127.0.0.1:5000 \
	--hybrid-batch-seconds 5 \
	--enable-emergency-priority \
	--reroute-highlight-seconds 10 \
	--controlled-count 25 \
	--controlled-source RSU_A \
	--controlled-destination RSU_K \
	--controlled-via-rsus RSU_C,RSU_F \
	--controlled-begin 120 \
	--controlled-end 1500 \
	--emergency-count 3 \
	--emergency-source RSU_B \
	--emergency-destination RSU_M \
	--emergency-via-rsus RSU_D,RSU_G \
	--emergency-begin 150 \
	--emergency-end 2400 \
	--auto-fallback-junctions
```

- Same command as a single line:

```bash
python3 sumo/run_sumo_pipeline.py --scenario demo --gui --max-steps 1800 --traffic-scale 1.8 --enable-hybrid-uplink-stub --server-url http://127.0.0.1:5000 --hybrid-batch-seconds 5 --enable-emergency-priority --reroute-highlight-seconds 10 --controlled-count 25 --controlled-source RSU_A --controlled-destination RSU_K --controlled-via-rsus RSU_C,RSU_F --controlled-begin 120 --controlled-end 1500 --emergency-count 3 --emergency-source RSU_B --emergency-destination RSU_M --emergency-via-rsus RSU_D,RSU_G --emergency-begin 150 --emergency-end 2400 --auto-fallback-junctions
```

- Visual cues in GUI:
	- Deep blue markers: controlled vehicles.
	- Bright yellow markers: emergency vehicles.
	- Magenta ring highlight: vehicles rerouted by server policy (visible for `--reroute-highlight-seconds`).
