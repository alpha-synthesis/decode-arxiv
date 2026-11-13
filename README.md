# DECODE: code, configurations and evidence

This repository accompanies the paper *DECODE: Decomposable Environments for
COntrolled Sequential-Decision Evaluation* (Silva Prado, Caldas de Souza,
Amaral Penteado de Jesus, Guimarães Herrmann, Breuel and Bevitori Maffia De
Oliveira, arXiv preprint, 2026). An archived copy of this release is at
https://doi.org/10.5281/zenodo.24200270.

It contains:
- the world generator and the configurations and seeds of the five released
  members (Base, Mem-0.5, Mem-4, Noise-0.1, Noise-0.2);
- the construction checks (recovery-rate bands and information clauses);
- the references (exact optimum, one-month optimum, reference filter,
  information reference, Monte Carlo references) and the probes;
- the code of every graded solver: CE-Seq with its trained weights, GBM-cash
  with its fitted model, GBM-lags and GBM-flat, which are refit from the
  regenerated training world, and the frozen rules of LLM-agent;
- the experiment drivers;
- the per-loan evidence behind every closed-loop interval on Base;
- the code of the data figures (Figs. 3-5).

## Layout

| path | contents |
|---|---|
| `decode_sim/` | world generator, answer key (`trajectory_key.py`), decision-table builder, member generator |
| `configs/` | frozen configurations of the five members, realism bands, transport features |
| `manifests/` | per member, the released configuration hash, seeds, batch size and check timestamps |
| `gates/` | construction checks (recovery-rate bands, information clauses) |
| `benchmark/` | closed-loop harness, solvers, references, probes, experiment drivers, the CE-Seq belief engine |
| `models/` | trained weights of CE-Seq (`ce_seq/`) and the fitted GBM-cash model (`gbm_cash/`) |
| `scripts/` | one shell script per experiment family, run from the repository root |
| `policy/policy.json` | the frozen rule policy of LLM-agent |
| `evidence/` | frozen outputs behind the reported results: statistics, per-loan values, executed plans, check reports |
| `figures/make_figures.py` | regenerates the paper's data figures (Figs. 3-5) from `evidence/` into `outputs/figures/`; Figs. 1-2 are diagrams |
| `tests/` | tests of the account-field timing |

## Installation

**Platform.** Linux x86_64 or aarch64 with glibc 2.28 or newer (Ubuntu
20.04+, Debian 10+, RHEL/Rocky 8+) and bash. Alpine/musl is not supported.
Every pinned package has manylinux_2_28 wheels, so nothing builds from source.
LightGBM needs the OpenMP runtime (`sudo apt-get install libgomp1` on minimal
images).

**Python 3.12.** The results were produced with CPython 3.12.3. Python 3.11
and older do not work, because numpy 2.5.2 and scipy 1.18.0 require 3.12. If
`python3.12` is not on your system, create the environment with
`uv venv -p 3.12 .venv` or `conda create -p ./.venv python=3.12` instead of the
`venv` line below. On Debian/Ubuntu, `python3.12 -m venv` needs the
`python3.12-venv` package.

```
git clone https://github.com/alpha-synthesis/decode-arxiv && cd decode-arxiv   # or unpack the archived copy
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt     # CPU only: CPU build of torch, about 0.2 GB
# or, with an NVIDIA GPU (driver 580 or newer, CUDA 13; about 2.7 GB of wheels):
#   pip install -r requirements.txt
# older drivers: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
export PYTHONPATH="$(pwd)"
mkdir -p outputs worlds
python -c "import numpy, pandas, scipy, sklearn, duckdb, torch, lightgbm, yaml, matplotlib, pyarrow; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

`requirements-cpu.txt` has the same pins as `requirements.txt`, with the CPU
build of torch. The scripts use `python` from the active environment (set
`PY=/path/to/python` to override), change to the repository root themselves,
and create `outputs/` and `worlds/`. Worlds are written to `worlds/`, results
to `outputs/`.

**GPU is optional.** CE-Seq grading (`board.sh`), CE-Seq retraining
(`train_ce_seq.sh`) and the recurrent and attention probes (`memory_axis.sh`)
use a CUDA GPU when one is visible and the CPU otherwise. The checkpoint
`models/ce_seq/belief_v7.pt` is loaded with `torch.load(..., weights_only=False)`.
It pickles numpy arrays and a scikit-learn 1.9.0 `IsotonicRegression`, so it
needs the pinned versions. Both `models/ce_seq/belief_v7.pt` and
`models/gbm_cash/gbm_cash.pkl` are pickles, and unpickling can execute
arbitrary code. Load them only after checking them against `SHA256SUMS` of
this release.

**Hardware and run times.** Measured on 2026-09-25 by a clean run of an
earlier build of this package, with the same generator, solver and board code, on a shared 48-core Linux host with 500 GB of RAM and one A100. Other
jobs held about 40 cores during the first half of the run, so these are upper
bounds. The transport row was measured in November 2026 with this release's
code. The checks on the shipped evidence run on a laptop in minutes.

| step | wall clock | peak RAM |
|---|---|---|
| `pip install -r requirements.txt` (GPU wheels) / `requirements-cpu.txt` | 2 min / 40 s | < 1 GB |
| checks on the shipped evidence (report, contrasts, figures) | < 30 s each | < 1 GB |
| one training world (1M borrowers x 24 months, incl. CSV export and decision table) | 1 h 18 min | 26 GB |
| one evaluation world (100k x 12) / Base check world (100k x 24) | 7-8 min / 5 min | 2 GB / 3 GB |
| board, reference arms (exact and one-month optimum, random), 4,000 loans | 24 min | 0.5 GB |
| board, LLM-agent / reference filter | 16 min / 22 min | 0.5 GB / 4 GB |
| board, GBM-lags and GBM-flat (4 arms) / their frozen-information arms | 16 min / 14 min | 13 GB |
| board, GBM-cash / CE-Seq | 11 min / 12 min | 17 GB / 3 GB |
| regrading four executed plans | 22 min | 0.5 GB |
| Base construction check | 7 min | 2.3 GB |
| synthetic transport statistic, one member (10,000 resamples and 20 refitted partitions) | 9 min | 1.5 GB |
| re-observation Monte Carlo, one noisy member | 2 h 17 min | 2 GB |

Disk: about 90 GB for the ten released worlds (16 GB per training world) plus
about 4 GB of pilot worlds for the memory axis. Steps are independent across
members, so they parallelize well.

## Checks on the shipped evidence (no worlds, CPU, minutes)

```
python -m benchmark.closed_loop_report --runs evidence/runs/cl_v7_board \
    evidence/runs/cl_v7_p7 evidence/runs/cl_v7_analyst evidence/runs/cl_v7_frozen \
    --out outputs/evidence_check
cmp outputs/evidence_check/board_final.json evidence/closed_loop/board_final.json && echo "board: identical"
python -m benchmark.drivers.board_contrasts --out outputs/board_contrasts.json
python -m benchmark.drivers.ladder_boot --out outputs/ladder_boot.json
cmp outputs/ladder_boot.json evidence/base_experiments/ladder_boot.json && echo "ladder: identical"
python figures/make_figures.py               # writes outputs/figures/
python -m pytest -q tests/test_entering_state.py   # after pip install pytest; about a minute
```

`board_contrasts` prints the paired contrasts between solvers on Base and the
intervals of the declared intervention sequence, in points of the exact
optimum. Board contrasts divide by the cohort's mean optimum, as
`closed_loop_report` does. Sequence contrasts divide each resample's mean
difference by that resample's mean optimum, as `ladder_boot` does.

The statistics drivers `benchmark/drivers/e0_boot.py`, `decomp_stats.py` and
`e1_stats.py` read `outputs/`. To run them on the shipped per-loan values, copy
the evidence into a fresh `outputs/` before running the pipeline (the copy
overwrites run folders of the same name):

```
cp -r evidence/runs/. outputs/
python -m benchmark.drivers.e0_boot && python -m benchmark.drivers.decomp_stats && python -m benchmark.drivers.e1_stats
```

`e0_boot` and `decomp_stats` print the archived run statistics, which divide
by the fixed cohort optimum. The paper's intervals for the declared
intervention sequence and the two fitted refinements are the ones printed by
`board_contrasts` and `ladder_boot`, which divide each resample's mean
difference by that resample's mean optimum. `e1_stats` prints the
response-component contrasts of the paper's Appendix D, with the fixed
normalizer as the paper states. The line `E0 - bayes_filter:frozen` of
`e0_boot` is the Base point of Fig. 5(b).

Every closed-loop interval is a paired bootstrap over loans (10,000 resamples
on shared indices).

## Account-field timing in the probes and the transport statistic

The generator writes each `monthly_panel` row after the month's outcome, so
`days_past_due`, `days_in_credito_em_liquidacao` and `outstanding_balance` in
row t hold the state after month t's repayment or charge-off draw (a recovery
resets the first two to 0). The probes (`gates/v7_pilot_gap.py`,
`benchmark/b_demo_probe.py`), the member checks and sweeps that call them, the
two Mem-0.5 decomposition drivers and the synthetic side of
`benchmark/fannie_transport.py` read these three fields as the value entering
month t (row t-1; the loan's opening snapshot at month 0). The solvers' training
loader uses the same convention (`decode_sim/bridge/entering_state.py`). Every
other field these scripts read is recorded before the decision.

- The default is `entering`, which produces the paper's values. The outputs are
  in `evidence/entering_state/`, with their provenance in
  `evidence/entering_state/PROVENANCE.md`.
- `--state-timing post_outcome_archived` reads the three fields from row t. It
  reproduces the outputs in the parent folders (`evidence/memory_probes/`,
  `geometry_sweep/`, `transport/`, `gate_reports/` and the sweep files in
  `base_experiments/`). The shipped scripts take the timing from the
  environment variable `STATE_TIMING` (default `entering`). With
  `STATE_TIMING=post_outcome_archived`, `scripts/transport.sh` also selects the
  pooled-AUC interval of those outputs (`--interval archived_pooled --n-boot 1000
  --partitions 0`).
- The thresholds and pilot reference values in
  `configs/adversarial_v7_realism_bands.yaml` and
  `configs/adversarial_v72_realism_bands.yaml` were set with the post-outcome
  probe. Under the entering timing, the hard clauses keep their verdicts on
  Base, Mem-0.5 and Mem-4. The report-only pilot drift lines compare entering
  values with post-outcome pilot references.
- The transport statistic's interval resamples loans within the five folds and
  recomputes the same mean-over-folds statistic as the point estimate (10,000
  resamples, recorded seed). The output also reports the point over 20 random
  loan-grouped partitions with refitted models (recorded seed).
- The timing does not affect the solvers, the closed-loop board, the
  references, the re-observation runs or the real (Fannie Mae) features.
- `tests/test_entering_state.py` tests the timing. It intervenes on one
  month's outcome inside the generator, checks that the entering fields equal
  the generator's decision inputs, guards the panel alignment, and checks the
  consumers statically.

## Full pipeline, in order

The worlds are not shipped. They are regenerated bit-identically from the
frozen configurations and recorded seeds (verified on the test host, see
"Verification"). The generation batch size must stay
10000, because the first loans of a world are identical across population
sizes only at a fixed batch size. Existing worlds are skipped, so never run a
smoke test into `worlds/`. Use a separate folder:
`python -m decode_sim.generate_members --out smoke_worlds --members BASE --scale 0.02`.

1. `bash scripts/generate_worlds.sh` (five members, training and evaluation worlds, Base check world; `--members BASE,MEM-4` selects members)
2. `bash scripts/check_members.sh` (realism bands of every member; see the note below)
3. `bash scripts/board.sh` (writes `outputs/cl_v7_final`, which step 4 reads)
4. `bash scripts/disclosures.sh`, then `python -m benchmark.drivers.self_report` (self-report inflation and its interval, written to `outputs/self_report.json`)
5. `bash scripts/memory_axis.sh`
6. `bash scripts/reobservation.sh`
7. `bash scripts/transport.sh` (synthetic side; the real side runs only when the licensed Fannie Mae data are present, see "Real data")
8. `python figures/make_figures.py`
9. Optional: `bash scripts/train_ce_seq.sh`, then grade the retrained model with
   `python -m benchmark.closed_loop_v7 board --world worlds/adversarial_v7_infer --train-world worlds/adversarial_v7_train --config configs/adversarial_v7_scenario_frozen.yaml --k 6 --horizon 6 --loans 4000 --belief-model outputs/ce_seq_retrained/belief_v7.pt --arms belief_sequential,belief_sequential:frozen --out outputs/cl_v7_ce_seq_retrained`
10. Regrade any executed plan with the answer key:
    `python -m benchmark.regrade_plans --world worlds/adversarial_v7_infer --config configs/adversarial_v7_scenario_frozen.yaml evidence/runs/cl_v7_board/plan_belief_sequential.csv`

**Note on the member checks.** `check_members.sh` re-runs the realism bands of
every member and continues past a member whose check cannot run. The
information clauses of Mem-0.5 and Mem-4 also need the delta = 0 control world
`worlds/control_d0_seed17002` and each evaluation world's `answer_key.csv`,
which this release does not generate. For those clauses, the shipped reports in
`evidence/gate_reports/` and `evidence/entering_state/gate_reports/` are the
record of the release.

| paper result | command | evidence |
|---|---|---|
| released-members table and its checks | `scripts/check_members.sh` | `evidence/gate_reports/`, `manifests/`; probe clauses under the entering timing: `evidence/entering_state/gate_reports/` |
| Fig. 3(a), closed-loop values and paired contrasts on Base | `scripts/board.sh`, `python -m benchmark.drivers.board_contrasts` | `evidence/closed_loop/board_final.json`, `per_loan_merged.csv`, `evidence/runs/cl_v7_*` |
| Fig. 3(b), declared intervention sequence, refinements | `scripts/disclosures.sh`, `python -m benchmark.drivers.ladder_boot`, `python -m benchmark.drivers.board_contrasts` | `evidence/base_experiments/` (incl. `ladder_boot.json`), `evidence/runs/{e0,e1,decomp,c7a}_board` |
| oracle-side linear diagnostic of the hidden score (Section 3.1) | `python -m benchmark.drivers.c7a_diag` (run by `scripts/disclosures.sh`; prints to the terminal) | printed lines stored in `evidence/base_experiments/linear_diagnostic.txt` |
| self-report inflation of certainty-equivalent chaining and its interval | `python -m benchmark.drivers.self_report` | `evidence/base_experiments/self_report.json`; cohort means also in `evidence/base_experiments/e0_jensen.json` |
| Fig. 4, memory axis and geometry draws | `scripts/memory_axis.sh` | `evidence/entering_state/geometry_sweep/`, `evidence/entering_state/base_experiments/` (probe levels per delta in `delta_sweep_levels.csv`, collected from the probe printouts of the memory-axis run) |
| Fig. 5(a), memory values and attention tables, stopping logs (`*_stops` keys) | `scripts/memory_axis.sh` | `evidence/entering_state/memory_probes/` |
| Fig. 5(b), re-observation under process noise | `scripts/reobservation.sh`; Base point from `e0_boot` | `evidence/reobservation/`; Base point in `evidence/base_experiments/e0_stats.json` |
| predictive comparison with real data (Section 3.4 and its appendix table) | `scripts/transport.sh` | `evidence/entering_state/transport/` |
| re-anchoring and amended-clause records | none (release records) | every member's final configuration (`configs/`), manifest (`manifests/`) and check report; for Mem-4 also the failed definitive world, `configs/adversarial_v72_d40_scenario_frozen.yaml`, `manifests/adversarial_v72_d40.json` and `evidence/gate_reports/v72_d40_gate_report.json`; the Mem-0.5 amendment in `configs/adversarial_v72_realism_bands.yaml` and `evidence/gate_reports/v72_d05_gate_report.json` |
| data figures (Figs. 3-5) | `python figures/make_figures.py` | `evidence/` and values recorded in the script; `f2_board.pdf` is Fig. 3, `f3_dial.pdf` Fig. 4, `f5_arch.pdf` Fig. 5(a), `f6_reobs.pdf` Fig. 5(b) |

On regenerated worlds, the board's merged table and its paired intervals come
from

```
python -m benchmark.closed_loop_report --runs outputs/cl_v7_board outputs/cl_v7_p7 \
    outputs/cl_v7_analyst outputs/cl_v7_frozen outputs/cl_v7_ce_seq \
    --out outputs/cl_v7_final
```

which `scripts/board.sh` runs. The noisy members' Monte Carlo intervals are
recomputed by `benchmark/v8_reobservation.py` from the recorded seeds (seed 0,
24 innovation paths, 2,000 loans, as `scripts/reobservation.sh` passes them).

## Names in the code

| paper | in the code |
|---|---|
| Base, Mem-0.5, Mem-4, Noise-0.1, Noise-0.2 | `BASE`, `MEM-0.5`, `MEM-4`, `NOISE-0.1`, `NOISE-0.2`; configurations `adversarial_v7`, `adversarial_v72_d05`, `adversarial_v72_d40_rev2`, `adversarial_v8_q01`, `adversarial_v8_q02` |
| GBM-lags | arms `history_myopic`, `history_sequential` |
| GBM-flat | arms `flat_myopic`, `flat_sequential` |
| GBM-cash | arm `p7_myopic` |
| LLM-agent | arm `analyst` (`policy/policy.json`, `benchmark/rule_policy.py`) |
| CE-Seq | arm `belief_sequential` (`benchmark/belief_v7.py`, `models/ce_seq/`) |
| exact optimum, one-month optimum | arms `oracle_W6`, `oracle_W1` |
| reference filter and its planner | arm `bayes_filter` (`benchmark/exact_filter.py`) |
| declared intervention sequence | arms `lags_exactF_sequential`, `posterior_gbm_static_sequential`, `posterior_gbm_sequential`, `bayes_filter` |
| structured estimator, 11-lag window | arms `structured_sequential`, `longlags_sequential` |
| response-component substitutions (Appendix D) | arms `hybrid_*` |
| oracle-side linear diagnostic | `benchmark/drivers/c7a_diag.py` |
| construction checks | `gates/`, `*_gate_report.json` |
| recovery-rate bands, information clauses | `recovered_1m`, `recovered_9m`; `prob_*`, `myopia_gap_pct`, `flat_board_pct`, `top1_path_share`, `krule_gap_pp`, `coverage_min_cell` and the other entries of `configs/*_realism_bands.yaml` |
| snapshot, history and true-state probes; information reference; fitted reference filter | `flat`, `lags`, `oracle`; `bayes_analytic`; `filter` |
| exact optimum and one-month optimum in the per-loan files | `ceiling`, `myopic_bound` |
| predictive comparison with real data | "transport": `benchmark/fannie_transport.py`, `scripts/transport.sh`, `evidence/*/transport/` |

CE-Seq is graded from its trained weights, which reproduces its published
values, and can be retrained with `scripts/train_ce_seq.sh`. GPU training is
not bit-deterministic, so a retrained model is close to, not identical with,
the shipped one. GBM-cash is graded from its fitted model
(`models/gbm_cash/gbm_cash.pkl`, passed by `board.sh` as `--p7-cache`), which
reproduces its published values exactly on the test host. Passing a new `--p7-cache` path
retrains it. LightGBM's row bagging then runs over a training query whose row
order is not fixed, so a retrained model is close to, not identical with, the
shipped one (66.56 instead of 66.32 on the test host). GBM-lags and GBM-flat
are refit at run time and reproduce their published values exactly on the
test host. Every
executed plan is also in `evidence/runs/`, and `benchmark/regrade_plans.py`
regrades any plan with the answer key.

CE-Seq reads the account-state sequence before the decision point directly
from the evaluation world's database, in addition to the readings, channels,
offers and fatigue that the harness supplies to every solver.

## Real data

The real side of the transport statistic uses the Fannie Mae Single-Family
Loan Performance Data (free registration), available at
<https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data>.
The data licence does not allow redistribution, so this release contains no
loan-level Fannie Mae content. The drawn cohort, 57,641 loans across
the 104 quarterly files 2000Q1-2025Q4, is recorded by its fingerprint in
`evidence/fannie/cohort_fingerprint.json`: the number of loans, the loans per
quarterly file, and the SHA-256 of the sorted identifier list. The analysed
sample is the 20,677 delinquency spells of 8,050 of these loans
(`evidence/entering_state/transport/fannie_real.json`). The identifier list is
withheld under the data licence, which does not allow it to be shared. A
licence holder who reconstructs the cohort checks it against the fingerprint
and rebuilds the real side with:

```
python -m benchmark.fannie.cohort_fingerprint --ids <cohort_loan_ids.csv>
python -m benchmark.fannie.build_panel --raw <directory with 2000Q1.csv ... 2025Q4.csv> \
    --ids <cohort_loan_ids.csv> \
    --out data/fannie/fannie_performance_loan_month_large.csv
python -m benchmark.fannie.build_episodes       # delinquency spells; reads $FANNIE_DIR (default data/fannie)
bash scripts/transport.sh                       # now runs the real side too
```

`build_panel.py` checks the list against the fingerprint before reading any
file, uses the column layout of `evidence/fannie/panel_columns.json` (the
public file layout), and asserts the field count of every quarterly file. The
real-side values were computed with scikit-learn 1.7.2. Their reproduction
under the pinned 1.9.0 environment has not been verified.

## Labels and edits in the released files

- Region codes and city names in the generator's location table are
  placeholder labels assigned in alphabetical order, so every sorted-category
  encoding is unchanged. The coordinates and weights that drive the draws are
  untouched.
- The frozen configurations carry no comments. The parsed configurations equal
  the released ones, which `manifests/*.json` records as equal semantic hashes
  next to the byte hashes of the commented configurations recorded at
  generation, which are not shipped.
- Source files carry no comments or docstrings. Removing them leaves every
  Python module's syntax tree unchanged apart from the removed docstrings.
  Some help, error and print messages were reworded, unused data-dictionary
  entries were removed, and `e0_jensen_gap` gained an optional `per_loan`
  output. No computed value changes.
- In the rule policy's free-text rationale, a currency symbol and two program
  names were replaced, and one note about actions outside the released action
  set was rewritten in general terms; its rules are unchanged.
- Some evidence files keep the labels of the runs that produced them. In
  `evidence/base_experiments/ladder_boot.json`, "exact inference" names the
  reference-filter features of the declared intervention sequence. In the
  Mem-0.5 check reports, "paired control" names the delta = 0 control world
  `control_d0_seed17002`, which has its own seed and is not matched to the
  member's population. The same holds for "paired rise" in
  `evidence/entering_state/mem05_decomposition/d05_flat_decomp.txt`. In
  `evidence/closed_loop/board_final.json`, `published_open_loop_pct` and
  `frozen_reproduces_published` refer to an earlier open-loop grading that
  the paper does not report. "Bayes value" in
  `evidence/base_experiments/e0_jensen.json` means joint integration under the
  filter belief.
- `SHA256SUMS` lists the checksum of every file in the release.

## Verification

- **End-to-end run (2026-09-25).** On a clean host, with an earlier build of
  this package that has the same generator, solver and board code, following
  these instructions: installation from `requirements.txt`, Base regenerated in full
  (training, evaluation and check worlds) and the evaluation worlds of
  Noise-0.1 and Noise-0.2, then the board and the other Base steps. Against the
  shipped evidence, per-loan values on all 4,000 loans agree to within 4e-12 for the exact and one-month optima, the random policy, the
  LLM-agent, the reference filter, GBM-lags and GBM-flat (refit, both horizons,
  closed-loop and frozen-information), and CE-Seq and GBM-cash (from their
  shipped models). Regrading the shipped executed plans gives 82.28, 84.40,
  64.74 and 93.75. The Base construction report matches band by band, both
  re-observation curves match to the last digit, and the check world is a
  prefix of the training world.
- **Worlds.** A 20,000-borrower prefix of every member's evaluation world and
  of Base's training world, generated with this code, is identical, table by
  table and in the decision table, to the released worlds.
- **Account-field timing (2026-11-04).** With `--state-timing
  post_outcome_archived`, the code reproduces every post-outcome probe, check,
  sweep, decomposition and transport output exactly. See
  `evidence/entering_state/PROVENANCE.md`.
- **Board.** `closed_loop_report` over `evidence/runs/` reproduces
  `evidence/closed_loop/board_final.json` exactly.
- **Self-report and linear diagnostic (2026-11-12).** On the Base worlds and
  with this release's code, `benchmark.drivers.self_report` reproduces the
  cohort means of `evidence/base_experiments/e0_jensen.json` exactly and wrote
  `evidence/base_experiments/self_report.json`. `benchmark.drivers.c7a_diag`
  wrote `evidence/base_experiments/linear_diagnostic.txt`.
- **Figures.** `make_figures.py` regenerates the paper's data figures
  (Figs. 3-5) from `evidence/`.

## Licence

Code: MIT (`LICENSE`). Configurations, manifests, evidence, model weights and
the rule policy: Creative Commons Attribution 4.0 International
(`LICENSE-DATA`).

## Citation

See `CITATION.cff`.
