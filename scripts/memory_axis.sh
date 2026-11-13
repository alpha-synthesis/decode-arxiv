#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
ST=${STATE_TIMING:-entering}
$PY -m benchmark.drivers.rot_sweep --state-timing $ST --out outputs/rot_sweep_results_$ST.json
$PY -m benchmark.drivers.geometry_sweep --state-timing $ST --out outputs/geometry_sweep_results_$ST.json
$PY -m benchmark.b_demo_probe --world worlds/adversarial_v7_train --config configs/adversarial_v7_scenario_frozen.yaml --out outputs/bdemo_v7train_control.json --state-timing $ST
$PY -m benchmark.b_demo_probe --world worlds/adversarial_v72_d05_train --config configs/adversarial_v72_d05_scenario_frozen.yaml --out outputs/bdemo_d05train.json --state-timing $ST
$PY -m benchmark.b_demo_probe --world worlds/adversarial_v72_d40_rev2_train --config configs/adversarial_v72_d40_rev2_scenario_frozen.yaml --out outputs/bdemo_d40train.json --state-timing $ST
for m in v7:adversarial_v7 d05:adversarial_v72_d05 d40:adversarial_v72_d40_rev2; do
  tag=${m%%:*}; w=${m#*:}
  $PY -m benchmark.b_demo_probe --world worlds/${w}_train --config configs/${w}_scenario_frozen.yaml \
    --rungs lags --archs gru128x2,lstm128x2,tfm64x2,gru32long --out outputs/archsweep_${tag}.json \
    --state-timing $ST
done
$PY -m benchmark.b_demo_probe --world worlds/adversarial_v72_d40_rev2_train \
  --config configs/adversarial_v72_d40_rev2_scenario_frozen.yaml --rungs lags,gru \
  --archs gru128x2,lstm128x2,tfm64x2,gru32long --seeds 2,3,4 --out outputs/archsweep_d40_seeds234.json \
  --state-timing $ST
