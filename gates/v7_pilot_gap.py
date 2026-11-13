from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from decode_sim.bridge.entering_state import (
    DEFAULT_STATE_TIMING, STATE_TIMINGS, assert_panel_aligned, state_select_sql)

FLAT = ["days_past_due", "days_in_credito_em_liquidacao", "outstanding_balance",
        "contacts_last_3m", "contact_fatigue", "loan_age_months",
        "remaining_maturity_months", "age", "monthly_income", "dependents",
        "original_principal", "interest_rate_monthly"]
CH_CODE = {"none": 0.0, "sms": 1.0, "whatsapp": 2.0, "phone": 3.0}


def load_frame(world: Path, state_timing: str = DEFAULT_STATE_TIMING):
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    obs_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()]
    zman_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()]
    assert obs_cols and zman_cols, "not a manifold world"
    state_sel, state_join = state_select_sql(state_timing)
    if state_timing == "entering":
        assert_panel_aligned(con)
    df = con.execute(f"""
        SELECT p.loan_id, p.month, {', '.join('p.' + c for c in obs_cols)},
               {state_sel},
               p.contacts_last_3m, p.contact_fatigue,
               p.loan_age_months, p.remaining_maturity_months,
               l.original_principal, l.interest_rate_monthly,
               b.age, b.monthly_income, b.dependents,
               a.channel, a.discount_offer, a.is_exploration,
               t.p_recovery, {', '.join('t.' + c for c in zman_cols)}
        FROM monthly_panel p
        JOIN actions a USING (action_id)
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        JOIN oracle_truth t ON t.loan_id = p.loan_id AND t.month = p.month
        {state_join}
        ORDER BY p.loan_id, p.month
    """).df()
    con.close()
    return df, obs_cols, zman_cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6, help="history window (months)")
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--state-timing", choices=STATE_TIMINGS,
                    default=DEFAULT_STATE_TIMING,
                    help="entering (decision-time values, default) or "
                         "post_outcome_archived (reproduces the archived runs)")
    args = ap.parse_args()

    df, obs_cols, zman_cols = load_frame(args.world, args.state_timing)
    print(f"state timing: {args.state_timing}", flush=True)
    print(f"panel rows: {len(df)}", flush=True)

    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)
    lag_cols = []
    df["ch_code"] = df["channel"].map(CH_CODE).astype(float)
    for j in range(1, args.k + 1):
        for c in obs_cols + ["ch_code"]:
            name = f"{c}_lag{j}"
            df[name] = g[c].shift(j)
            lag_cols.append(name)

    m = df[(df.is_exploration) & (df.channel == "phone")
           & (df.month >= args.k)].dropna(subset=lag_cols)
    if len(m) > args.rows:
        m = m.sample(args.rows, random_state=0)
    if len(m) < 2000:
        raise ValueError(
            f"v7_pilot_gap: only {len(m)} measurement rows — the R2 gap would "
            "be degenerate noise; use a bigger world or lower --k")
    y = np.log(np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)
               / (1 - np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)))
    print(f"measurement rows (phone explore, month>={args.k}): {len(m)}", flush=True)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    def score(cols, label):
        X = m[cols].to_numpy(float)
        r2 = cross_val_score(
            HistGradientBoostingRegressor(max_iter=300, random_state=0),
            X, y, cv=3, scoring="r2")
        print(f"{label:28s} R2 = {r2.mean():.4f} (+/- {r2.std():.4f})", flush=True)
        return r2.mean()

    d_cols = ["discount_offer"]
    r_flat = score(FLAT + obs_cols + d_cols, "flat (row + current keyhole)")
    r_hist = score(FLAT + obs_cols + d_cols + lag_cols, f"history (+{args.k}m lags)")
    r_orac = score(FLAT + obs_cols + d_cols + zman_cols, "oracle (+ true hidden z)")
    print(f"\nSEPARATION GAP (history - flat): {r_hist - r_flat:+.4f}")
    print(f"HEADROOM LEFT (oracle - history): {r_orac - r_hist:+.4f}")
    return 0


if __name__ == "__main__":
    main()
