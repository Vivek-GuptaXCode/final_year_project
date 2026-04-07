#!/usr/bin/env bash
# ============================================================
# Sealdah → Park Circus  |  Dalhousie Square BLOCKED demo
# ============================================================
# Scenario
#   • 30 AI-controlled vehicles depart from Sealdah RSU and
#     head to Park Circus RSU via the normal direct path.
#   • At step 60 (≈60 s), Dalhousie Square RSU is declared
#     FULLY CONGESTED: every connected edge receives a 9999 s
#     travel-time penalty.
#   • All vehicles are instantly re-routed around Dalhousie
#     and switch body colour to RED in the GUI.
#   • 3 emergency vehicles (×3 = 9 total) also depart from
#     Park Street toward College Street with corridor priority.
#   • RL adaptive signal control, runtime logging, and RSU
#     overlays are all enabled.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Junction IDs (from data/rsu_config_kolkata.json) ─────────
SEALDAH_JID="9491482575"
PARK_CIRCUS_JID="cluster_10281986033_10302557856_10302557859_638354058"
DALHOUSIE_JID="cluster_10281869257_12438122826_12438122827_663940666"
PARK_STREET_JID="664472799"
COLLEGE_STREET_JID="cluster_10282080280_10846969131_11365834325_2281800978_#2more"

python3 -u sumo/run_sumo_pipeline.py \
    --scenario            kolkata \
    --gui \
    --seed                11 \
    --max-steps           3600 \
    --traffic-scale       2.0 \
    \
    --rsu-config          data/rsu_config_kolkata.json \
    --rsu-range-m         120 \
    --rsu-min-inc-lanes   4 \
    \
    --controlled-count    30 \
    --controlled-source   "$SEALDAH_JID" \
    --controlled-destination "$PARK_CIRCUS_JID" \
    --controlled-begin    10 \
    --controlled-end      600 \
    \
    --force-congestion-at-junction "$DALHOUSIE_JID" \
    --force-congestion-at-step     60 \
    --reroute-highlight-seconds    3600 \
    \
    --enable-emergency-priority \
    --emergency-count     3 \
    --emergency-source    "$PARK_STREET_JID" \
    --emergency-destination "$COLLEGE_STREET_JID" \
    --emergency-begin     120 \
    --emergency-end       1800 \
    \
    --enable-hybrid-uplink-stub \
    --server-url          http://localhost:5000 \
    --hybrid-batch-seconds 5 \
    --route-timeout-seconds 1.5 \
    \
    --enable-rl-signal-control \
    --rl-model-dir        models/rl/artifacts \
    --rl-min-green-seconds 15 \
    --rl-yellow-duration-seconds 3 \
    --rl-max-controlled-tls 96 \
    \
    --enable-runtime-logging \
    --runtime-log-root    data/raw \
    \
    --marker-refresh-steps 4 \
    --emergency-corridor-lookahead-edges 6 \
    --emergency-hold-seconds 8 \
    "$@"
