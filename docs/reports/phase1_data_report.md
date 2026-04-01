# Phase 1 Data Report

- Generated UTC: 2026-03-27T10:38:30Z
- Manifest schema version: 1.1
- Required horizons: 60, 120
- Bundle: data/exports/phase1_export_20260327T103827Z/phase1_dataset_bundle.tar.gz
- Bundle SHA256: 17dcba734b92096ba5c9e9143aa42ef80d06c3ded7f2a005ba4c0666329fe33b

## Summary

- Runs selected: 3
- Runs gate-ready: 3
- Runs leakage PASS: 3
- Missing artifact entries: 0
- Total labeled rows: 9360

## Horizon Balance

- Horizon 60s: positive_rows=8762, positive_rate=0.936111
- Horizon 120s: positive_rows=8762, positive_rate=0.936111

## Per-Run Status

| run_id | scenario | seed | required_files | horizons | leakage | gate_ready |
|---|---:|---:|---:|---:|---:|---:|
| smoke_phase1_logger | demo | 42 | PASS | PASS | PASS | PASS |
| smoke_phase1_logger_seed43 | demo | 43 | PASS | PASS | PASS | PASS |
| smoke_phase1_logger_seed47 | demo | 47 | PASS | PASS | PASS | PASS |

## Notes

- This report is generated from local CSV and JSON artifacts only.
- File hashes are recorded in the unified manifest for reproducibility.
- Any missing artifacts are listed under each run in the manifest.

