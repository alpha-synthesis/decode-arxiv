from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from decode_sim.config import load_config
from decode_sim.trajectory_key import (
    _book, export_answer_key, grade_plan)

BAND_LOG: list[dict] = []


def band(name, value, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value} {detail}", flush=True)
    BAND_LOG.append({"band": name, "value": value, "pass": bool(ok),
                     "detail": detail})
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--bands", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--state-timing", choices=("entering", "post_outcome_archived"),
                    default="entering",
                    help="account-field timing of the probe bands 6-7; the "
                         "archived report used post_outcome_archived")
    args = ap.parse_args()
    cfg = load_config(str(args.config))
    bands = yaml.safe_load(args.bands.read_text())
    db = args.world / "sim.duckdb"
    ok = True

    from decode_sim.trajectory_key import _book_at_month
    cohort = sorted(_book_at_month(db, args.k)["loan_id"])
    rng = np.random.default_rng(20260828)
    ids = sorted(rng.choice(cohort, size=min(args.sample, len(cohort)),
                            replace=False))
    key, plans = export_answer_key(db, cfg, H=args.horizon, loan_ids=ids,
                                   include_plans=True, start_month=args.k)
    gap = 100 * (1 - key.myopic_bound.mean() / key.ceiling.mean())
    b = bands["myopia_gap_pct"]
    ok &= band("myopia_gap_pct", round(gap, 2), b["min"] <= gap <= b["max"],
               f"(band [{b['min']}, {b['max']}])")
    plans = plans.sort_values(["loan_id", "month"])
    paths = plans.groupby("loan_id")["channel"].agg(tuple)
    top1 = 100 * paths.value_counts(normalize=True).iloc[0]
    ok &= band("top1_path_share", round(top1, 2),
               top1 <= bands["top1_path_share_max"],
               f"(max {bands['top1_path_share_max']})")

    top_paths = paths.value_counts().head(12).index
    best_rule = -np.inf
    best_path = None
    n = len(ids)
    for path in top_paths:
        seqs = plans[plans.loan_id.isin(
            paths[paths == path].index[:1])].sort_values("month")["channel"].tolist()
        plan = pd.DataFrame({
            "loan_id": np.repeat(ids, args.horizon),
            "month": np.tile(np.arange(args.horizon), n),
            "channel": np.tile(seqs, n),
            "discount_offer": np.tile(
                [0.2 if c != "none" else 0.0 for c in seqs], n),
        })
        v = grade_plan(plan, db, cfg, H=args.horizon, start_month=args.k)["value"].mean()
        if v > best_rule:
            best_rule, best_path = v, seqs
    kgap = 100 * (1 - best_rule / key.ceiling.mean())
    ok &= band("krule_gap_pp", round(kgap, 2),
               kgap >= bands["krule_gap_min_pp"],
               f"(min {bands['krule_gap_min_pp']}; best path {best_path})")

    import duckdb as ddb
    con = ddb.connect(str(db), read_only=True)
    cov = con.execute("""
        WITH j AS (
          SELECT b.monthly_income, p.contact_fatigue, a.channel
          FROM monthly_panel p
          JOIN actions a ON a.action_id = p.action_id
          JOIN borrowers b ON b.borrower_id = p.borrower_id)
        SELECT ntile_income, fat_bucket, ch_class, count(*) AS c FROM (
          SELECT ntile(3) OVER (ORDER BY monthly_income) AS ntile_income,
                 CASE WHEN contact_fatigue = 0 THEN '0'
                      WHEN contact_fatigue <= 1 THEN '(0,1]' ELSE '>1' END AS fat_bucket,
                 CASE WHEN channel = 'none' THEN 'none'
                      WHEN channel = 'phone' THEN 'human' ELSE 'digital' END AS ch_class
          FROM j) GROUP BY 1, 2, 3
    """).df()
    con.close()
    min_cell = int(cov["c"].min())
    ok &= band("coverage_min_cell", min_cell,
               min_cell >= bands["coverage_min_per_cell"],
               f"(min {bands['coverage_min_per_cell']}, {len(cov)} cells)")

    dt = pd.read_csv(args.world / "decision_table.csv",
                     usecols=["recovered_1m", "recovered_9m"])
    for col in ("recovered_1m", "recovered_9m"):
        b = bands[col]
        v = float(dt[col].mean())
        ok &= band(col, round(v, 4), b["min"] <= v <= b["max"],
                   f"(band [{b['min']}, {b['max']}])")

    r = subprocess.run(
        [sys.executable, "gates/v7_pilot_gap.py", "--world", str(args.world),
         "--k", str(args.k), "--state-timing", args.state_timing],
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        return 1
    sep = float(r.stdout.split("SEPARATION GAP (history - flat): ")[1].split()[0])
    head = float(r.stdout.split("HEADROOM LEFT (oracle - history): ")[1].split()[0])
    ok &= band("prob_separation", round(sep, 4),
               sep >= bands["prob_separation_min"],
               f"(min {bands['prob_separation_min']})")
    ok &= band("prob_headroom", round(head, 4),
               head >= bands["prob_headroom_min"],
               f"(min {bands['prob_headroom_min']})")

    r = subprocess.run(
        [sys.executable, "gates/v7_board_pilot.py", "--world", str(args.world),
         "--config", str(args.config), "--k", str(args.k), "--horizon", str(args.horizon),
         "--eval-loans", "2000"], capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        return 1
    board = json.loads((args.world / "v7_board_pilot.json").read_text())
    flat = board["rows"]["flat_sequential"]
    ok &= band("flat_board_pct", flat, flat <= bands["flat_board_max_pct"],
               f"(max {bands['flat_board_max_pct']})")

    report = args.world / "v7_check_gate_report.json"
    report.write_text(json.dumps({"pass": bool(ok), "bands": BAND_LOG,
                                  "probe_state_timing": args.state_timing},
                                 indent=2) + "\n")
    print(f"GATE: {'PASS' if ok else 'FAIL'} -> {report}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
