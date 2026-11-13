#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
$PY -m benchmark.belief_v7 train --train-world worlds/adversarial_v7_train \
  --months 24 --mask-mode iid --hid 64 --out outputs/ce_seq_retrained
