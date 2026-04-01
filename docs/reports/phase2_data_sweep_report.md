# Phase-2 Data Sweep Report

- Sweep id: phase2sweep_20260327T193955Z
- Created (UTC): 2026-03-27T19:52:46Z
- Execution policy: heavy training/evaluation remains local-only in this repository runtime.

## Matrix

- Scenarios: demo, low, medium, high
- Seeds: 31, 47, 59
- Traffic scales: 0.8, 1.0, 1.2
- Traffic reduction pcts: 20.0, 40.0
- Horizons: 60, 120
- Gap seconds: 5

## Summary

- Runs planned: 72
- SUMO stage success: 72/72
- Processing stage success: 72/72
- Quality gates passed: 62/72 (evaluated runs)

## Per-Run Status

| Run ID | SUMO | Processing | Quality Gate | Notes |
|---|---:|---:|---:|---|
| phase2sweep_demo_seed31_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed31_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed31_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed31_ts1_tr40 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed31_ts1p2_tr20 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed31_ts1p2_tr40 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed47_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed47_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed47_ts1_tr20 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed47_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed47_ts1p2_tr20 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed47_ts1p2_tr40 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed59_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed59_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_demo_seed59_ts1_tr20 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed59_ts1_tr40 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed59_ts1p2_tr20 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_demo_seed59_ts1p2_tr40 | PASS | PASS | FAIL | failed checks: horizon_positive_rate_range |
| phase2sweep_low_seed31_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed31_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed31_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed31_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed31_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed31_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed47_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed47_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed47_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed47_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed47_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed47_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed59_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed59_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed59_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed59_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_low_seed59_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_low_seed59_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed31_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed31_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed31_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed31_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed31_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed31_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed47_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed47_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed47_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed47_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed47_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed47_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed59_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed59_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed59_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed59_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed59_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_medium_seed59_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed31_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed31_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed31_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed31_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed31_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed31_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed47_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed47_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed47_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed47_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed47_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed47_ts1p2_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed59_ts0p8_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed59_ts0p8_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed59_ts1_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed59_ts1_tr40 | PASS | PASS | PASS |  |
| phase2sweep_high_seed59_ts1p2_tr20 | PASS | PASS | PASS |  |
| phase2sweep_high_seed59_ts1p2_tr40 | PASS | PASS | PASS |  |

## Next Step

Use run IDs with passing quality gates as training input for Phase-2 baselines in local runtime.
