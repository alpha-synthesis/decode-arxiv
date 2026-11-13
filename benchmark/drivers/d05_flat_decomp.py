import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FLAT = ["days_past_due", "days_in_credito_em_liquidacao", "outstanding_balance",
        "contacts_last_3m", "contact_fatigue", "loan_age_months",
        "remaining_maturity_months", "age", "monthly_income", "dependents",
        "original_principal", "interest_rate_monthly"]
CH_CODE = {"none": 0.0, "sms": 1.0, "whatsapp": 2.0, "phone": 3.0}
K = 6

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from decode_sim.bridge.entering_state import (
    DEFAULT_STATE_TIMING, STATE_TIMINGS, assert_panel_aligned, state_select_sql)

_ap = argparse.ArgumentParser()
_ap.add_argument("--state-timing", choices=STATE_TIMINGS, default=DEFAULT_STATE_TIMING)
_ap.add_argument("--worlds", default="worlds", help="directory holding both worlds")
ARGS = _ap.parse_args()
print(f"state timing: {ARGS.state_timing}", flush=True)

def load(world):
    con = duckdb.connect(f"{world}/sim.duckdb", read_only=True)
    obs_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE "
        "table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()]
    state_sel, state_join = state_select_sql(ARGS.state_timing)
    if ARGS.state_timing == "entering":
        assert_panel_aligned(con)
    df = con.execute(f"""
        SELECT p.loan_id, p.month, {', '.join('p.' + c for c in obs_cols)},
               {state_sel},
               p.contacts_last_3m, p.contact_fatigue,
               p.loan_age_months, p.remaining_maturity_months,
               l.original_principal, l.interest_rate_monthly,
               b.age, b.monthly_income, b.dependents,
               a.channel, a.discount_offer, a.is_exploration, t.p_recovery
        FROM monthly_panel p
        JOIN actions a USING (action_id)
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        JOIN oracle_truth t ON t.loan_id = p.loan_id AND t.month = p.month
        {state_join}
        ORDER BY p.loan_id, p.month""").df()
    con.close()
    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)
    df["ch_code"] = df["channel"].map(CH_CODE).astype(float)
    lag_cols = []
    for j in range(1, K + 1):
        for c in obs_cols + ["ch_code"]:
            name = f"{c}_lag{j}"
            df[name] = g[c].shift(j)
            lag_cols.append(name)
    m = df[(df.is_exploration) & (df.channel == "phone")
           & (df.month >= K)].dropna(subset=lag_cols)
    return m, obs_cols, lag_cols

M2 = ARGS.worlds
m_d, obs_d, lag_d = load(f"{M2}/adversarial_v72_d05_infer")
m_c, obs_c, lag_c = load(f"{M2}/control_d0_seed17002")
print(f"rows d05 {len(m_d)}  control {len(m_c)}")
kd = set(map(tuple, m_d[["loan_id", "month"]].to_numpy()))
kc = set(map(tuple, m_c[["loan_id", "month"]].to_numpy()))
inter = kd & kc
print(f"intersection rows {len(inter)}  (jaccard "
      f"{len(inter)/len(kd | kc):.3f})")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def flat_r2(m, obs_cols, restrict=None, label=""):
    if restrict is not None:
        mask = [tuple(x) in restrict
                for x in m[["loan_id", "month"]].to_numpy()]
        m = m[np.asarray(mask)]
    y = np.log(np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)
               / (1 - np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)))
    X = m[FLAT + obs_cols + ["discount_offer"]].to_numpy(float)
    r2 = cross_val_score(HistGradientBoostingRegressor(
        max_iter=300, random_state=0), X, y, cv=3, scoring="r2")
    print(f"flat {label:24s} n={len(m):6d}  R2 = {r2.mean():.4f} "
          f"(+/- {r2.std():.4f})", flush=True)
    return r2.mean()

f_d_all = flat_r2(m_d, obs_d, None, "d05 (all rows)")
f_c_all = flat_r2(m_c, obs_c, None, "control (all rows)")
f_d_int = flat_r2(m_d, obs_d, inter, "d05 (intersection)")
f_c_int = flat_r2(m_c, obs_c, inter, "control (intersection)")
print(f"\nrise over the control world, all rows:    {f_d_all - f_c_all:+.4f}")
print(f"rise over the control world, shared rows: {f_d_int - f_c_int:+.4f}")
print(f"composition share:           {(f_d_all - f_c_all) - (f_d_int - f_c_int):+.4f}")
for tag, m_, oc in (("d05", m_d, obs_d), ("ctrl", m_c, obs_c)):
    mm = m_[[tuple(x) in inter for x in m_[["loan_id", "month"]].to_numpy()]]
    print(f"{tag} dpd mean all {m_['days_past_due'].mean():.1f} / "
          f"inter {mm['days_past_due'].mean():.1f}")
print("== D05_DECOMP_DONE ==", flush=True)
