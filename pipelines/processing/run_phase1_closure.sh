#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HORIZONS="${HORIZONS:-60,120}"
GAP_SECONDS="${GAP_SECONDS:-5}"

if [[ "$#" -gt 0 ]]; then
  RUN_IDS=("$@")
else
  RUN_IDS=(
    "smoke_phase1_logger"
    "smoke_phase1_logger_seed43"
    "smoke_phase1_logger_seed47"
  )
fi

if [[ "${#RUN_IDS[@]}" -eq 0 ]]; then
  echo "[phase1] No run ids provided"
  exit 1
fi

join_by_comma() {
  local IFS=","
  echo "$*"
}

cd "${REPO_ROOT}"

echo "[phase1] repo_root=${REPO_ROOT}"
echo "[phase1] run_ids=$(join_by_comma "${RUN_IDS[@]}")"
echo "[phase1] horizons=${HORIZONS} gap_seconds=${GAP_SECONDS}"

for run_id in "${RUN_IDS[@]}"; do
  raw_csv="data/raw/${run_id}/rsu_features_1hz.csv"
  processed_csv="data/processed/${run_id}/rsu_horizon_labels.csv"
  split_dir="data/splits/${run_id}"

  if [[ ! -f "${raw_csv}" ]]; then
    echo "[phase1] missing raw file: ${raw_csv}"
    exit 1
  fi

  echo "[phase1] labeling run_id=${run_id}"
  python3 pipelines/processing/horizon_labeler.py \
    --input-rsu "${raw_csv}" \
    --output "${processed_csv}" \
    --horizons "${HORIZONS}"

  echo "[phase1] splitting run_id=${run_id}"
  python3 pipelines/processing/temporal_split.py \
    --input "${processed_csv}" \
    --output-dir "${split_dir}" \
    --gap-seconds "${GAP_SECONDS}"

  echo "[phase1] leakage-check run_id=${run_id}"
  python3 pipelines/processing/leakage_validator.py \
    --split-dir "${split_dir}" \
    --expected-gap-seconds "${GAP_SECONDS}"
done

RUN_IDS_CSV="$(join_by_comma "${RUN_IDS[@]}")"

echo "[phase1] exporting unified bundle+manifest"
python3 pipelines/processing/export_dataset_bundle.py \
  --run-ids "${RUN_IDS_CSV}" \
  --require-horizons "${HORIZONS}" \
  --project-root "${REPO_ROOT}"

echo "[phase1] completed successfully"
