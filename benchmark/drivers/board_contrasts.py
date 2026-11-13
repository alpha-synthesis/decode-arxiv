import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark.closed_loop_report import paired_bootstrap
from benchmark.drivers.ladder_boot import contrasts

BOARD = [
    ("GBM-lags, six months minus one month", "history_sequential", "history_myopic"),
    ("GBM-flat, six months minus one month", "flat_sequential", "flat_myopic"),
    ("GBM-lags minus GBM-flat, six months", "history_sequential", "flat_sequential"),
    ("GBM-lags minus CE-Seq, six months", "history_sequential", "belief_sequential"),
    ("CE-Seq minus GBM-flat, six months", "belief_sequential", "flat_sequential"),
    ("GBM-lags minus GBM-flat, one month", "history_myopic", "flat_myopic"),
    ("GBM-lags minus GBM-cash, one month", "history_myopic", "p7_myopic"),
    ("GBM-flat minus LLM-agent, one month", "flat_myopic", "analyst"),
    ("exact optimum minus GBM-lags, six months", "oracle_W6", "history_sequential"),
    ("exact optimum minus one-month optimum", "oracle_W6", "oracle_W1"),
]

SEQUENCE = [
    ("step: exact fatigue", "lags_exactF_sequential", "history_sequential"),
    ("step: filter features", "posterior_gbm_static_sequential", "lags_exactF_sequential"),
    ("step: rolled belief", "posterior_gbm_sequential", "posterior_gbm_static_sequential"),
    ("step: exact response", "bayes_filter", "posterior_gbm_sequential"),
    ("full disclosure minus GBM-lags", "bayes_filter", "history_sequential"),
    ("exact fatigue minus GBM-lags", "lags_exactF_sequential", "history_sequential"),
    ("filter features minus GBM-lags", "posterior_gbm_static_sequential", "history_sequential"),
    ("rolled belief minus GBM-lags", "posterior_gbm_sequential", "history_sequential"),
    ("11-lag window minus GBM-lags", "longlags_sequential", "history_sequential"),
]

STRUCTURED = [
    ("structured estimator minus GBM-lags", "structured_sequential", "history_sequential"),
    ("structured estimator minus exact fatigue", "structured_sequential", "lags_exactF_sequential"),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired contrasts of the Base closed-loop board and intervals of the "
                    "declared intervention sequence, in points of the exact optimum.")
    ap.add_argument("--evidence", type=Path, default=Path("evidence"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ev = args.evidence

    board = pd.read_csv(ev / "closed_loop" / "per_loan_merged.csv")
    ceiling = float(board["ceiling"].mean())
    res_board = {name: paired_bootstrap(board[a], board[b], ceiling, args.n_boot, args.seed)
                 for name, a, b in BOARD}

    seq = pd.read_csv(ev / "base_experiments" / "decomp_per_loan.csv")
    e0 = pd.read_csv(ev / "runs" / "e0_board" / "per_loan_values.csv")
    seq = seq.merge(e0[["loan_id", "bayes_filter"]], on="loan_id", how="inner", validate="one_to_one")
    st = pd.read_csv(ev / "base_experiments" / "c7a_per_loan.csv")
    st = st.merge(seq[["loan_id", "history_sequential"]], on="loan_id", how="inner",
                  validate="one_to_one")
    if not (len(seq) == len(st) == len(board)):
        raise SystemExit("per-loan files do not cover the same cohort")
    res_seq = contrasts(seq, SEQUENCE, args.n_boot, args.seed)
    res_seq.update(contrasts(st, STRUCTURED, args.n_boot, args.seed))

    out = {
        "units": "points of the exact optimum",
        "n_loans": int(len(board)),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "board_protocol": "paired bootstrap over loans, fixed cohort mean optimum in the denominator "
                          "(benchmark.closed_loop_report.paired_bootstrap)",
        "sequence_protocol": "paired bootstrap over loans, each resample's mean optimum in the "
                             "denominator (benchmark.drivers.ladder_boot.contrasts)",
        "board": res_board,
        "sequence": res_seq,
    }
    for title, block in (("board", res_board), ("sequence", res_seq)):
        print(title)
        for name, v in block.items():
            lo, hi = v["ci95"]
            print(f"  {name:44s} {v['delta_pp']:+7.2f}  [{lo:+.2f}, {hi:+.2f}]")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
