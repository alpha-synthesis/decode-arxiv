from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from decode_sim.config import load_config
from decode_sim.trajectory_key import export_answer_key, grade_plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("plans", nargs="+", type=Path)
    args = ap.parse_args()
    cfg = load_config(str(args.config))
    db = args.world / "sim.duckdb"
    for f in args.plans:
        plan = pd.read_csv(f)
        ids = list(dict.fromkeys(plan["loan_id"]))
        key = export_answer_key(db, cfg, H=args.horizon, loan_ids=ids,
                                start_month=args.k).set_index("loan_id").loc[ids]
        g = grade_plan(plan, db, cfg, H=args.horizon,
                       start_month=args.k).set_index("loan_id").loc[ids]
        pct = 100.0 * g["value"].mean() / key["ceiling"].mean()
        print(f"{f.name}: {pct:.2f}% of the exact optimum over {len(ids)} loans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
