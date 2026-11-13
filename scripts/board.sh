#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
B="--world worlds/adversarial_v7_infer --train-world worlds/adversarial_v7_train \
   --config configs/adversarial_v7_scenario_frozen.yaml --k 6 --horizon 6 --loans 4000"
$PY -m benchmark.closed_loop_v7 board $B \
  --arms oracle_W6,oracle_W1,random,flat_myopic,history_myopic,flat_sequential,history_sequential \
  --out outputs/cl_v7_board
$PY -m benchmark.closed_loop_v7 board $B --p7-cache models/gbm_cash/gbm_cash.pkl \
  --arms p7_myopic,p7_myopic:frozen --out outputs/cl_v7_p7
$PY -m benchmark.closed_loop_v7 board $B --analyst-policy policy/policy.json \
  --arms analyst,analyst:frozen --out outputs/cl_v7_analyst
$PY -m benchmark.closed_loop_v7 board $B \
  --arms flat_myopic:frozen,history_myopic:frozen,flat_sequential:frozen,history_sequential:frozen \
  --out outputs/cl_v7_frozen
$PY -m benchmark.closed_loop_v7 board $B --belief-model models/ce_seq/belief_v7.pt \
  --arms belief_sequential,belief_sequential:frozen --out outputs/cl_v7_ce_seq
$PY -m benchmark.closed_loop_report --runs outputs/cl_v7_board outputs/cl_v7_p7 \
  outputs/cl_v7_analyst outputs/cl_v7_frozen \
  outputs/cl_v7_ce_seq --out outputs/cl_v7_final
