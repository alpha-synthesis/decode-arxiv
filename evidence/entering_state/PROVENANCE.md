# Outputs with decision-time account fields

Produced on 2026-11-04 by this package with `--state-timing entering` (the
default), on the released worlds. The twelve geometry-sweep pilot worlds were
regenerated with the released generator and configuration; with the
post-outcome timing they reproduce `evidence/geometry_sweep/geometry_sweep_results.json`
exactly (`geometry_sweep/geometry_sweep_results_archived_timing_regenerated.json`).

Environment: the pinned `requirements.txt` (CPython 3.12.3, numpy 2.5.2,
scikit-learn 1.9.0, torch 2.12.0+cu130, duckdb 1.5.5), Linux, one NVIDIA A100
for the recurrent and attention configurations. The real (Fannie Mae) side was
computed on the licensed data's host with scikit-learn 1.7.2; its post-outcome
point and interval are reproduced exactly there.

Before any value here was computed, this package with
`--state-timing post_outcome_archived` reproduced every corresponding output in
the parent folders (probe, check, sweep, decomposition, training-world probes,
recurrent and attention configurations, and the transport point, interval and
P(<=0) with `--interval archived_pooled --n-boot 1000 --partitions 0`). Row sets
are identical under both timings on every world; only the three account fields
change.

| file | post-outcome counterpart | command |
|---|---|---|
| `base_experiments/rot_sweep_results.json`, `gain_sweep_results.json` | `../base_experiments/` | `python -m benchmark.drivers.{rot,gain}_sweep --state-timing entering` |
| `geometry_sweep/geometry_sweep_results.json` | `../geometry_sweep/` | `python -m benchmark.drivers.geometry_sweep --state-timing entering --keep-worlds` |
| `memory_probes/bdemo_*.json`, `archsweep_*.json` | `../memory_probes/` | `scripts/memory_axis.sh` commands with `--state-timing entering` |
| `transport/transport_syn_*.json` | `../transport/` | `python -m benchmark.fannie_transport --side synthetic --state-timing entering` (10,000 resamples, seed 0; 20 partitions, seed 7) |
| `transport/fannie_real.json` | `../transport/fannie_real.json` | real side, same estimator; also records the reproduction of the post-outcome interval |
| `gate_reports/v72_*_gate_report.json` | `../gate_reports/` | `python -m gates.adversarial_v72_check_gate --state-timing entering` (thresholds and pilot references unchanged) |
| `gate_reports/v7_check_probe_clauses.json` | bands 6-7 of `../gate_reports/v7_check_gate_report.json` | `python gates/v7_pilot_gap.py --world worlds/adversarial_v7_check --state-timing entering` |
| `mem05_decomposition/*.txt` | the post-outcome decomposition logs | `python -m benchmark.drivers.d05_{flat,channel}_decomp --state-timing entering` |

`bdemo_identity_d05pilot.json` was run with the flat, history and true-state
rungs only. The `v72_d40` report belongs to a retired member configuration (it
fails a realism band under either timing) and is kept for completeness.
