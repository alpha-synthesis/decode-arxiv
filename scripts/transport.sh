#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
ST=${STATE_TIMING:-entering}
if [ "$ST" = post_outcome_archived ]; then TI="--interval archived_pooled --n-boot 1000 --partitions 0"; else TI=""; fi
for m in v7:adversarial_v7 d05:adversarial_v72_d05 d40:adversarial_v72_d40_rev2; do
  tag=${m%%:*}; w=${m#*:}
  $PY -m benchmark.fannie_transport --side synthetic --world worlds/${w}_infer --state-timing $ST $TI --out outputs/transport_syn_${tag}.json
done
if [ -f "${FANNIE_DIR:-data/fannie}/artifacts_collections/cure_episodes.csv" ]; then
  $PY -m benchmark.fannie_transport --side real $TI --out outputs/fannie_real.json
else
  echo "real side skipped: ${FANNIE_DIR:-data/fannie}/artifacts_collections/cure_episodes.csv not found"
fi
