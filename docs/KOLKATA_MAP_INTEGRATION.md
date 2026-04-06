# Kolkata Map Integration

## Summary

Successfully integrated Kolkata city map (12.6 MB OSM data) into the traffic management system.

## Traffic Light Recovery (April 2026)

Kolkata traffic signal coverage was rebuilt using SUMO `netconvert` TLS guessing.

- **Before rebuild**: 21 traffic lights / 2,612 non-internal junctions
- **After rebuild**: 459 traffic lights / 2,612 non-internal junctions

Rebuild command used:

```bash
netconvert \
  --sumo-net-file sumo/networks/kolkata.net.xml \
  -o sumo/networks/kolkata.net.tlsfix.tmp.xml \
  --tls.guess true \
  --tls.guess-signals true \
  --tls.discard-simple true \
  --tls.ignore-internal-junction-jam true
mv sumo/networks/kolkata.net.tlsfix.tmp.xml sumo/networks/kolkata.net.xml
```

## Generated Files

| File | Size | Description |
|------|------|-------------|
| `sumo/networks/kolkata.net.xml` | 22 MB | Road network (36,255 edges, 6,246 junctions) |
| `sumo/networks/kolkata.poly.xml` | 2.1 MB | Building polygons for visualization |
| `sumo/routes/kolkata_passenger.rou.xml` | 1.6 MB | 7,653 passenger vehicles |
| `sumo/routes/kolkata_freight.rou.xml` | 278 KB | 1,385 freight vehicles |
| `sumo/scenarios/kolkata.sumocfg` | 358 B | Scenario configuration |

## Network Statistics

- **Area**: Kolkata, India (22.5667-22.6029°N, 88.3601-88.3766°E)
- **Size**: ~4km × 4km
- **Edges**: 36,255 (largest network in system, 4x bigger than "city")
- **Junctions**: 6,246
- **Traffic lights**: 459 (`<tlLogic>` entries after rebuild)
- **Expected RSUs**: ~20-25 (based on high-degree junctions)

## Usage

### Basic Simulation

```bash
# Headless (no GUI)
python3 sumo/run_sumo_pipeline.py --scenario kolkata --max-steps 1800

# With GUI
python3 sumo/run_sumo_pipeline.py --scenario kolkata --gui --max-steps 1800

# With GUI (recommended low-lag settings for large maps)
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --marker-refresh-steps 4 \
  --emergency-priority-interval-steps 2

# Reduced traffic for testing
python3 sumo/run_sumo_pipeline.py --scenario kolkata --traffic-scale 0.5 --max-steps 300
```

### With Full V2X System

Terminal 1 (Server):
```bash
python3 server.py
```

Terminal 2 (SUMO):
```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --traffic-scale 1.0 \
  --enable-hybrid-uplink-stub \
  --server-url http://127.0.0.1:5000
```

### With RL Signal Control

```bash
python3 sumo/run_sumo_pipeline.py \
  --scenario kolkata \
  --gui \
  --max-steps 1800 \
  --enable-rl-signal-control \
  --rl-model-dir models/rl/artifacts/latest \
  --rl-max-controlled-tls 96 \
  --rl-step-interval-steps 2
```

## Code Changes

### Modified Files

1. **`sumo/run_sumo_pipeline.py`** (line 48-53)
   - Added "kolkata" to scenario choices

2. **`sumo/scenarios/sumo_contract.json`** (lines 39-44)
   - Added kolkata scenario entry

### No Changes Needed

All Phase 1-5 components work automatically:
- ✅ Forecasting model (Phase 2)
- ✅ Risk-aware routing (Phase 3)  
- ✅ RL signal control (Phase 4)
- ✅ Hybrid fusion (Phase 5)
- ✅ Data pipeline (Phase 1)

## Performance Considerations

### Expected Performance

| Metric | Demo/City | Kolkata |
|--------|-----------|---------|
| Loading time | 2-3 sec | 8-10 sec |
| Step time | 50-100 ms | 150-300 ms |
| Memory | 150 MB | 400-500 MB |
| RSU count | 6-8 | 20-25 |

### Optimization Tips

1. **Reduce traffic scale**: Use `--traffic-scale 0.5` for testing
2. **Shorter simulations**: Start with `--max-steps 300` 
3. **Disable GUI**: Run headless for data collection
4. **Throttle expensive controls**: Use `--marker-refresh-steps 4` and `--emergency-priority-interval-steps 2`
5. **Limit RL scope on large maps**: Use `--rl-max-controlled-tls 96 --rl-step-interval-steps 2`

## Validation

Tested:
- ✅ Network loads successfully
- ✅ Vehicles spawn and navigate
- ✅ Simulation runs to completion
- ✅ No SUMO errors or warnings (only minor OSM import warnings)

Not yet tested:
- ⏳ Full V2X integration with server
- ⏳ RSU graph generation
- ⏳ GNN routing on Kolkata topology
- ⏳ RL signal control performance

## Next Steps

1. Run full V2X pipeline to generate RSU locations
2. Analyze RSU graph topology for Kolkata
3. Benchmark routing and signal control performance
4. Tune traffic parameters for realistic Kolkata patterns
5. Compare KPIs vs demo/city scenarios

## Troubleshooting

### Issue: Slow simulation

**Solution**: Reduce traffic scale
```bash
python3 sumo/run_sumo_pipeline.py --scenario kolkata --traffic-scale 0.3
```

### Issue: Memory issues

**Solution**: Close other applications, or reduce max-steps
```bash
python3 sumo/run_sumo_pipeline.py --scenario kolkata --max-steps 600
```

### Issue: Want to visualize specific area

**Solution**: Use SUMO GUI and zoom to area of interest
```bash
python3 sumo/run_sumo_pipeline.py --scenario kolkata --gui
# Then zoom in GUI to desired location
```
