#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
B="--world worlds/adversarial_v7_infer --train-world worlds/adversarial_v7_train \
   --config configs/adversarial_v7_scenario_frozen.yaml --k 6 --horizon 6 --loans 4000"
$PY -m benchmark.closed_loop_v7 board $B --arms bayes_filter,bayes_filter:frozen --out outputs/e0_board
$PY -m benchmark.exact_filter gates --world worlds/adversarial_v7_infer \
  --config configs/adversarial_v7_scenario_frozen.yaml --k 6 --horizon 6 --loans 4000 --out outputs/e0_gates
$PY -m benchmark.closed_loop_v7 board $B --posterior-feat-cache outputs/posterior_feats_train.parquet \
  --arms history_sequential,lags_exactF_sequential,posterior_gbm_static_sequential,posterior_gbm_sequential,longlags_sequential \
  --out outputs/decomp_board
$PY -m benchmark.closed_loop_v7 board $B --arms structured_sequential,lags_exactF_sequential --out outputs/c7a_board
$PY -m benchmark.closed_loop_v7 board $B \
  --arms hybrid_none_exact_sequential,hybrid_contact_exact_sequential,hybrid_none_exact_static_sequential,hybrid_contact_exact_static_sequential \
  --out outputs/e1_board
$PY -m benchmark.e1_bias_map $B --e1-dir outputs/e1_board --lags-dir outputs/decomp_board --out outputs/e1_bias
$PY -m benchmark.drivers.c7a_diag
$PY -m benchmark.drivers.e0_boot
$PY -m benchmark.drivers.decomp_stats
$PY -m benchmark.drivers.e1_stats
