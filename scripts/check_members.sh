#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
PY=${PY:-python}
ST=${STATE_TIMING:-entering}
rc=0
$PY -m gates.adversarial_v7_check_gate --world worlds/adversarial_v7_check \
  --config configs/adversarial_v7_scenario_frozen.yaml \
  --bands configs/adversarial_v7_realism_bands.yaml --state-timing $ST || { echo "check did not complete: BASE"; rc=1; }
for m in v8_q01 v8_q02 v72_d05 v72_d40_rev2; do
  $PY -m gates.adversarial_v72_check_gate --member ${m#v72_} \
    --train worlds/adversarial_${m}_train --infer worlds/adversarial_${m}_infer \
    --config configs/adversarial_${m}_scenario_frozen.yaml \
    --bands configs/adversarial_v72_realism_bands.yaml --code-note "decode-release" \
    --state-timing $ST \
    || { echo "check did not complete: $m (see the note above for the memory members)"; rc=1; }
done
exit $rc
