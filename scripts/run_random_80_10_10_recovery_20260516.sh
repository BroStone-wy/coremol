#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/legion-w4/LabData/Stone/BORF/CORMOL"
PYTHON_BIN="/home/legion-w4/anaconda3/envs/GNRF/bin/python"

cd "$ROOT"

"$PYTHON_BIN" scripts/run_curvflow_classification_sweep.py \
  --config configs/curvflow_classification_tox21_delta_channelgate_random_80_10_10_balanced_seeds3_30.json \
  --results_name curvflow_classification_sweep/tox21_delta_channelgate_random_80_10_10_balanced_seeds3_30_fixed20260516

"$PYTHON_BIN" scripts/run_curvflow_classification_sweep.py \
  --config configs/curvflow_classification_hiv_delta_channelgate_random_80_10_10_seeds3_30.json \
  --results_name curvflow_classification_sweep/hiv_delta_channelgate_random_80_10_10_seeds3_30_20260516

"$PYTHON_BIN" scripts/run_curvflow_classification_sweep.py \
  --config configs/curvflow_classification_toxcast_delta_channelgate_random_80_10_10_seeds3_30.json \
  --results_name curvflow_classification_sweep/toxcast_delta_channelgate_random_80_10_10_seeds3_30_20260516
