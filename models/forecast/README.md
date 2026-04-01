# Phase 2 Forecasting Module

Binary congestion forecasting with uncertainty output for the V2X hybrid traffic stack.

## Architecture

- **Feature contract v2**: 31-dimensional vector with lag features, rolling statistics, congestion dynamics, and rate-of-change signals (`feature_builder_v2.py`)
- **Selected model**: LightGBM (600 trees, 127 leaves, balanced class weight)
- **Hold-out performance**: 87.3% accuracy, 91.3% F1, 90.6% ROC-AUC on 10,000 rows

## Files

| File | Purpose |
|------|---------|
| `feature_builder.py` | v1 feature contract (12 features) — kept for backward compatibility |
| `feature_builder_v2.py` | v2 feature contract (31 features) — used by current artifact |
| `common.py` | Shared utilities: clamp, ECE, rolling CV splits, RSU hashing |
| `train_phase2_baselines.py` | v1 baseline training (persistence, HistGB, XGBoost, spatiotemporal proxy) |
| `train_phase2_improved.py` | v2 improved training (LightGBM, XGBoost-GPU, DART, MLP, Ensemble) |
| `inference.py` | `ForecastInferenceEngine` — loads artifact and serves predictions |
| `run_inference_smoke.py` | Quick artifact load + latency check |
| `evaluate_artifact_accuracy.py` | Hold-out evaluation with threshold metrics and ranking metrics |

## Artifact structure

```
artifacts/latest/
  forecast_artifact.json   # metadata, feature contract, model config
  model.pkl                # serialized LightGBM model
```

The artifact JSON includes `feature_contract.version` (`"v1"` or `"v2"`) so inference auto-selects the correct feature builder.

## Usage

```bash
# Train v2 models (requires processed data in data/splits/)
python3 models/forecast/train_phase2_improved.py

# Smoke test inference latency
python3 models/forecast/run_inference_smoke.py \
  --artifact models/forecast/artifacts/latest/forecast_artifact.json \
  --input-csv data/splits/<run_id>/test.csv \
  --max-rows 100

# Evaluate on hold-out test splits
python3 models/forecast/evaluate_artifact_accuracy.py \
  --artifact models/forecast/artifacts/latest/forecast_artifact.json \
  --test-glob 'data/splits/phase2sweep_*/test.csv' \
  --max-rows 10000
```

## Server integration

Enable with environment variables:
```bash
HYBRID_ENABLE_FORECAST_MODEL=1 \
HYBRID_FORECAST_ARTIFACT=models/forecast/artifacts/latest/forecast_artifact.json \
python3 server.py
```
