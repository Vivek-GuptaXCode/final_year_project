# Phase 1 Processing Pipeline

This folder contains lightweight, no-install scripts for transforming raw 1 Hz RSU logs
into leakage-safe training data artifacts.

## Scripts

- `horizon_labeler.py`
  - Input: `rsu_features_1hz.csv`
  - Output: labeled CSV with `label_congestion_<horizon>s` columns
  - Default horizons: `60,120`

- `temporal_split.py`
  - Input: labeled CSV
  - Output: `train.csv`, `val.csv`, `test.csv`, `split_manifest.json`
  - Supports a configurable temporal `--gap-seconds` to reduce leakage risk

- `leakage_validator.py`
  - Input: split directory
  - Output: `leakage_report.json`
  - Checks chronology, overlap, and expected gap

- `export_dataset_bundle.py`
  - Input: run ids plus `data/raw`, `data/processed`, `data/splits` artifacts
  - Output: unified manifest (`phase1_dataset_manifest.json`), archive bundle (`.tar.gz`), and report (`docs/reports/phase1_data_report.md`)
  - Computes file SHA-256 hashes, verifies required horizons, and summarizes leakage/split status

- `run_phase1_closure.sh`
  - Input: optional run ids as positional args (defaults to 3 smoke run ids)
  - Output: refreshed labels/splits/leakage reports + unified export bundle/report
  - Runs `horizon_labeler.py`, `temporal_split.py`, `leakage_validator.py`, and `export_dataset_bundle.py` in one command

## Example

```bash
python3 pipelines/processing/horizon_labeler.py \
  --input-rsu data/raw/<run_id>/rsu_features_1hz.csv \
  --output data/processed/<run_id>/rsu_horizon_labels.csv \
  --horizons 60,120

python3 pipelines/processing/temporal_split.py \
  --input data/processed/<run_id>/rsu_horizon_labels.csv \
  --output-dir data/splits/<run_id> \
  --gap-seconds 120

python3 pipelines/processing/leakage_validator.py \
  --split-dir data/splits/<run_id> \
  --expected-gap-seconds 120

python3 pipelines/processing/export_dataset_bundle.py \
  --run-ids <run_id_1>,<run_id_2>,<run_id_3> \
  --require-horizons 60,120

bash pipelines/processing/run_phase1_closure.sh

# Optional: custom runs
bash pipelines/processing/run_phase1_closure.sh <run_id_1> <run_id_2> <run_id_3>
```

## Notes

- If splits are empty, reduce `--gap-seconds` or run on longer logs.
- Keep split generation strictly chronological.
- Apply normalization/statistics only after split creation (in downstream training code).
