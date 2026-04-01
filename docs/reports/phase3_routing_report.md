# Phase 3 Routing Report — Uncertainty-Aware Routing

- **Generated UTC:** 2026-03-28
- **Feature flag:** `HYBRID_ENABLE_PHASE3_ROUTING`
- **Router:** `phase3_risk_router_v1` (`routing/phase3_risk_router.py`)
- **Audit log:** `data/raw/route_audit/route_decisions.jsonl` (configurable via `HYBRID_ROUTE_AUDIT_PATH`)
- **Comparison script:** `evaluation/phase3_comparison.py`
- **Comparison results:** `evaluation/phase3_comparison_results.json`

---

## 1. What Changed: Deterministic Baseline → Risk-Aware Routing

### Deterministic baseline (legacy)

The pre-Phase-3 `/route` endpoint used a simple surrogate:

```
count_score   = vehicle_count / 50
speed_score   = 1 - avg_speed / 15
p_congestion  = 0.6 × count_score + 0.4 × speed_score
confidence    = clamp(0.90 − |p_congestion − 0.5|, 0.5, 0.9)
risk_level    = "high" if p ≥ 0.70 else "medium" if p ≥ 0.45 else "low"
reroute_frac  = 0.35 / 0.20 / 0.0  (fixed per risk tier)
```

No uncertainty propagation, no confidence-based fallback, no audit trail.

### Phase 3 risk-aware policy

The Phase 3 router (`build_phase3_decision`) replaces the fixed-tier policy with:

| Component | Formula |
|-----------|---------|
| Estimated delay | `(vehicle_count / max(1, avg_speed)) × (1 + 2 × p_congestion)` |
| Delay term | `clamp(estimated_delay / delay_scale_s, 0, 1)` — default scale: 90 s |
| Uncertainty term | `clamp(uncertainty × uncertainty_weight, 0, 1)` — default weight: 0.35 |
| Risk score | `clamp(delay_term + uncertainty_term, 0, 1)` |
| Risk level | high ≥ 0.85, medium ≥ 0.45, low < 0.45 |
| Reroute fraction | high: `min(max_frac, 0.35 + 0.10×uncertainty)`, medium: `min(max_frac, 0.20 + 0.05×uncertainty)`, low: 0 |

Three strategy branches:

| Strategy | Trigger | Mode | Fraction |
|----------|---------|------|---------|
| `emergency_override` | Emergency vehicle IDs present | `dijkstra` | 1.0 |
| `confidence_fallback` | `confidence < 0.55` (no emergency) | `travel_time` | min(normal, 0.15) |
| `risk_aware_primary` | Default | `gnn_effort` | risk-proportional |

---

## 2. Gate P3.1 — Confidence-Aware Score

Every Phase 3 response includes:
- `phase3.risk_score` — composite [0, 1] score from delay + uncertainty
- `phase3.risk_components` — `{delay_term, uncertainty_term, estimated_delay_s}`
- `phase3.decision_context` — full input context including `p_congestion`, `confidence`, `uncertainty`

Gate **PASS** — all 16 test scenarios produced `risk_score` and `risk_components` in the response.

---

## 3. Gate P3.2 — Fallback Mode Under Low Confidence

Confidence fallback activates when `confidence < 0.55` (configurable via `HYBRID_P3_LOW_CONFIDENCE_THRESHOLD`):
- Switches routing mode from `gnn_effort` → `travel_time` (more conservative)
- Caps reroute fraction at 0.15 regardless of risk tier
- Emergency override takes absolute precedence over fallback

**Observed fallback triggers across 16 scenarios:**

| Scenario | Confidence | Fallback | Mode |
|----------|-----------|---------|------|
| `free_flow_sparse` | 0.516 | Yes | `travel_time` |
| `medium_low_confidence` | 0.480 | Yes | `travel_time` |
| `heavy_low_confidence` | 0.400 | Yes | `travel_time`, frac=0.15 |
| `emergency_low_confidence` | 0.400 | No (emergency wins) | `dijkstra` |

Gate **PASS** — fallback triggers correctly for all `confidence < 0.55` cases; emergency override correctly suppresses fallback.

---

## 4. Gate P3.3 — Route Audit Log

`RouteAuditLogger` (`routing/route_audit_logger.py`) writes append-only JSONL entries at `HYBRID_ROUTE_AUDIT_PATH` (default: `data/raw/route_audit/route_decisions.jsonl`).

Each record contains:
- `rsu_id`, `sim_timestamp`, `vehicle_count`, `avg_speed_mps`, `emergency_vehicle_count`
- Forecast fields: `model`, `source`, `p_congestion`, `confidence`, `uncertainty`
- `routing_engine`, `risk_level`, `recommended_action`, `route_directives`
- Full `phase3` block (strategy, risk_score, components, alternatives)
- Unique `audit_id` (UUID4) — returned in the response under `route_audit_id` and `phase3.audit_id`

Gate **PASS** — audit logger is wired in `server.py:339–365` and verified active when `HYBRID_ENABLE_PHASE3_ROUTING=1`.

---

## 5. Gate P3.4 (Current) — Policy Sanity Regression

Current comparison uses 16 representative synthetic scenarios across 5 traffic conditions (free-flow, medium, heavy, onset/recovery, empty) and 3 special cases (emergency with/without low confidence, fallback-triggered):

| Check | Pass rate |
|-------|-----------|
| Reroute not disabled when Phase 3 itself judges medium/high risk | 16/16 |
| Reroute fraction ≥ 0.15 for Phase 3 medium/high risk (non-fallback) | 16/16 |
| Fallback triggers correctly per confidence threshold | 16/16 |
| Emergency override strategy set for all emergency scenarios | 16/16 |

**Key behavioural differences from baseline (by design, not regression):**

- For `medium_congestion` (20 vehicles, 4 m/s, p_cong≈0.53): baseline says "medium/reroute 20%" but Phase 3 computes `risk_score≈0.16` (low). This is intentional — Phase 3's delay-based metric is more conservative about over-rerouting, reducing unnecessary churn for borderline cases.
- For `heavy_congestion` (40 vehicles, 1.5 m/s, p_cong≈0.88): both baseline and Phase 3 agree on "high" risk; Phase 3 reroutes 39.4% (vs baseline 35%), slightly higher due to uncertainty term.
- Under low confidence, Phase 3 caps rerouting at 15% vs baseline's uncapped fixed fractions — prevents over-reaction to unreliable forecasts.

Policy gate **PASS** — no scenario where Phase 3 disabled rerouting that Phase 3 itself considered necessary.

### 5.1 True KPI Regression Gate (Travel Time / Waiting / Throughput)

To validate real traffic outcomes (not only policy shape), use:

- `evaluation/phase3_kpi_regression_gate.py`

Supported inputs:

- SUMO `--statistic-output` XML
- SUMO `--summary` XML
- SUMO `--tripinfo-output` XML
- Normalized JSON (for offline aggregation)

Gate method:

- Pair baseline vs Phase 3 runs by `run_id` or index.
- Compute per-pair deltas (%):
	- mean travel time,
	- mean waiting time,
	- throughput.
- Compute bootstrap confidence intervals over mean deltas.
- Pass only if both mean and CI bound satisfy thresholds.

Example command (statistics XML):

```bash
python3 evaluation/phase3_kpi_regression_gate.py \
	--source-type statistics \
	--baseline-glob "data/raw/baseline_*/*statistics*.xml" \
	--phase3-glob "data/raw/phase3_*/*statistics*.xml" \
	--pairing run-id \
	--max-travel-time-regression-pct 2.0 \
	--max-waiting-time-regression-pct 3.0 \
	--max-throughput-drop-pct 1.0 \
	--strict
```

Status: **Pending execution on paired SUMO baseline/Phase 3 artifacts.**

---

## 6. Routing Comparison Summary (16 Scenarios)

| Metric | Deterministic Baseline | Phase 3 Risk-Aware |
|--------|----------------------:|-------------------:|
| Scenarios with rerouting enabled | 7/16 | 6/16 |
| Mean reroute fraction (when enabled) | 0.571 | 0.468 |
| Confidence-fallback activations | 0 | 3 |
| Emergency override activations | 3 | 3 |
| Audit trail available | No | Yes |
| Uncertainty propagated to decision | No | Yes |
| Risk score interpretable | No | Yes |

Phase 3 reduces mean reroute fraction by ~18% through more precise risk quantification and caps uncertain decisions — this is the expected behaviour.

---

## 7. Configuration Reference

All thresholds are tunable at runtime via environment variables (no restart required):

| Variable | Default | Effect |
|----------|---------|--------|
| `HYBRID_ENABLE_GNN_ROUTING` | `0` | Enable graph message-passing reroute inference in `server.py` |
| `HYBRID_GNN_STEPS` | `2` | Number of message-passing propagation steps |
| `HYBRID_GNN_LOW_CONFIDENCE_THRESHOLD` | `0.55` | Confidence below which GNN routing falls back to `travel_time` |
| `HYBRID_GNN_MAX_REROUTE_FRACTION` | `0.40` | Hard cap on GNN-directed reroute fraction |
| `HYBRID_ENABLE_PHASE3_ROUTING` | `0` | Master feature flag |
| `HYBRID_P3_LOW_CONFIDENCE_THRESHOLD` | `0.55` | Confidence below which fallback activates |
| `HYBRID_P3_UNCERTAINTY_WEIGHT` | `0.35` | Weight of uncertainty in risk score |
| `HYBRID_P3_DELAY_SCALE_SECONDS` | `90.0` | Normalisation denominator for delay term |
| `HYBRID_P3_HIGH_RISK_SCORE` | `0.85` | risk_score threshold for "high" tier |
| `HYBRID_P3_MEDIUM_RISK_SCORE` | `0.45` | risk_score threshold for "medium" tier |
| `HYBRID_P3_MAX_REROUTE_FRACTION` | `0.40` | Hard cap on reroute fraction |
| `HYBRID_ROUTE_AUDIT_PATH` | `data/raw/route_audit/route_decisions.jsonl` | Audit log output path |
| `HYBRID_ENABLE_FORECAST_MODEL` | `0` | Enable Phase 2 forecast artifact for p_congestion |
| `HYBRID_FORECAST_ARTIFACT` | `models/forecast/artifacts/latest/forecast_artifact.json` | Artifact path |

---

## 8. Phase 3 Gate Summary

| Gate | Requirement | Result |
|------|-------------|--------|
| P3.1 | Route decision includes confidence-aware score | **PASS** — `risk_score` + `risk_components` in all responses |
| P3.2 | Fallback mode triggers correctly when confidence is low | **PASS** — 3/3 low-confidence cases trigger; emergency suppresses correctly |
| P3.3 | Route audit log captures selected/alternative paths and estimated gain | **PASS** — JSONL logger wired in server, UUID per decision |
| P3.4 | No severe KPI regression vs deterministic baseline | **PENDING** — run `evaluation/phase3_kpi_regression_gate.py` on paired SUMO outputs |

**Phase 3 status: policy sanity gates PASS; true KPI regression gate pending run.**
