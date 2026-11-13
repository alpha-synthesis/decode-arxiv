#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
for m in q01 q02; do
  $PY -m benchmark.v8_reobservation --world worlds/adversarial_v8_${m}_infer \
    --config configs/adversarial_v8_${m}_scenario_frozen.yaml --loans 2000 --reps 24 \
    --out outputs/v8_curve_${m}.json
done
