from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from decode_sim.config import load_config
from decode_sim.generator import _chargeoff_probability
from decode_sim.trajectory_key import (
    _book_at_month, export_answer_key, grade_plan, paid_share_mean)

CHANNELS = ["none", "sms", "whatsapp", "phone"]
D_STEPS = 21
STATE = ["days_past_due", "days_in_credito_em_liquidacao", "outstanding_balance",
         "contact_fatigue", "loan_age_months", "age", "monthly_income",
         "original_principal", "interest_rate_monthly"]
CH_CODE = {"none": 0.0, "sms": 1.0, "whatsapp": 2.0, "phone": 3.0}
F_KNOTS = np.array([0.0, 0.35, 0.7, 1.1, 1.5, 1.9, 2.3, 2.8, 3.4])


def load_rows(db: Path, k: int):
    con = duckdb.connect(str(db), read_only=True)
    obs_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()]
    df = con.execute(f"""
        SELECT p.loan_id, p.month,
               COALESCE(pe.days_past_due, l.days_past_due) AS days_past_due,
               COALESCE(pe.days_in_credito_em_liquidacao,
                        l.days_in_credito_em_liquidacao) AS days_in_credito_em_liquidacao,
               COALESCE(pe.outstanding_balance, l.outstanding_balance) AS outstanding_balance,
               p.contact_fatigue, {', '.join('p.' + c for c in obs_cols)},
               l.loan_age_months + p.month AS loan_age_months,
               l.interest_rate_monthly, l.original_principal,
               b.age, b.monthly_income,
               a.channel, a.discount_offer, a.is_exploration,
               p.recovered
        FROM monthly_panel p
        LEFT JOIN monthly_panel pe ON pe.loan_id = p.loan_id AND pe.month = p.month - 1
        JOIN actions a ON a.action_id = p.action_id
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        ORDER BY p.loan_id, p.month
    """).df()
    zman_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()]
    if zman_cols:
        zt = con.execute(f"""SELECT loan_id, month,
            {', '.join(zman_cols)} FROM oracle_truth""").df()
        df = df.merge(zt, on=["loan_id", "month"], how="left")
    con.close()
    df["ch_code"] = df["channel"].map(CH_CODE).astype(float)
    g = df.groupby("loan_id", sort=False)
    lag_cols = []
    for j in range(1, k + 1):
        for c in obs_cols + ["ch_code"]:
            name = f"{c}_lag{j}"
            df[name] = g[c].shift(j)
            lag_cols.append(name)
    return df, obs_cols, lag_cols


def fit(df, obs_cols, lag_cols, k, history: bool, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    feats = STATE + obs_cols + (lag_cols if history else [])
    xcols = feats + ["discount_offer"] + [f"ch_{c}" for c in CHANNELS[1:]]

    def frame(rows):
        d = rows.copy()
        for c in CHANNELS[1:]:
            d[f"ch_{c}"] = (d["channel"] == c).astype(float)
        return d[xcols].to_numpy(float)

    tr = df[df.is_exploration & (df.month >= k)].dropna(subset=lag_cols)
    tr = tr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    X, y = frame(tr), tr["recovered"].to_numpy(int)
    if len(tr) < 2000:
        raise ValueError(f'too few explore rows with full history: {len(tr)}')
    n_val = max(500, int(0.15 * len(tr)))
    gbm = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    gbm.fit(X[n_val:], y[n_val:])
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(gbm.predict_proba(X[:n_val])[:, 1], y[:n_val])
    trn = df[(df.channel == "none") & (~df.is_exploration)
             & (df.month >= k)].dropna(subset=lag_cols)
    Xn = trn[feats].to_numpy(float)
    yn = trn["recovered"].to_numpy(int)
    n_val_n = max(500, int(0.15 * len(trn)))
    gbm_n = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    gbm_n.fit(Xn[n_val_n:], yn[n_val_n:])
    iso_n = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso_n.fit(gbm_n.predict_proba(Xn[:n_val_n])[:, 1], yn[:n_val_n])
    print(f"{'history' if history else 'flat':7s} estimator: "
          f"contact rows {len(tr)}, none rows {len(trn)}", flush=True)

    def p_contact(base_feats, d, ci):
        x = np.concatenate([base_feats,
                            np.full((len(base_feats), 1), d),
                            np.zeros((len(base_feats), len(CHANNELS) - 1))], axis=1)
        x[:, len(feats) + 1 + ci - 1] = 1.0
        return iso.predict(gbm.predict_proba(x)[:, 1])

    def p_none(base_feats):
        return iso_n.predict(gbm_n.predict_proba(base_feats)[:, 1])

    return feats, p_contact, p_none


def make_plans(eval_df, feats, p_contact, p_none, sim_cfg, k, H, pshare):
    n = len(eval_df)
    d_grid = np.linspace(0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
    oc = sim_cfg["oracle"]["contact_fatigue"]
    human = set(sim_cfg["actions"]["human_channels"])
    costs = np.array([float((sim_cfg["actions"].get("channel_costs") or {})
                            .get(c, 0.0)) for c in CHANNELS])
    fat_add = np.array([0.0 if c == "none" else float(oc["increment"])
                        * (float(oc["human_multiplier"]) if c in human else 1.0)
                        for c in CHANNELS])
    decay = float(oc["decay"])
    C = len(CHANNELS)
    fi = feats.index("contact_fatigue")

    dpd = eval_df["days_past_due"].to_numpy(float).copy()
    liq = eval_df["days_in_credito_em_liquidacao"].to_numpy(float).copy()
    bal = eval_df["outstanding_balance"].to_numpy(float).copy()
    rate = eval_df["interest_rate_monthly"].to_numpy(float)
    la = eval_df["loan_age_months"].to_numpy(float).copy()
    F0 = eval_df["contact_fatigue"].to_numpy(float)
    base0 = eval_df[feats].to_numpy(float)

    P = np.empty((H, n, C, D_STEPS, len(F_KNOTS)), dtype=np.float32)
    bal_t = np.empty((H, n))
    q_t = np.empty((H, n))
    di_, li_, ba_, la_ = dpd.copy(), liq.copy(), bal.copy(), la.copy()
    for t in range(H):
        q_t[t] = [_chargeoff_probability(int(v), sim_cfg["oracle"]) for v in li_]
        bal_t[t] = ba_
        base = base0.copy()
        base[:, feats.index("days_past_due")] = di_
        base[:, feats.index("days_in_credito_em_liquidacao")] = li_
        base[:, feats.index("outstanding_balance")] = ba_
        base[:, feats.index("loan_age_months")] = la_
        for kk, fk in enumerate(F_KNOTS):
            b = base.copy(); b[:, fi] = fk
            P[t, :, 0, :, kk] = p_none(b)[:, None]
            for ci in range(1, C):
                for di, d in enumerate(d_grid):
                    P[t, :, ci, di, kk] = p_contact(b, d, ci)
        ba_ = ba_ * (1.0 + rate * 0.35)
        di_ = di_ + 30.0
        li_ = np.where(di_ >= 90.0, li_ + 30.0, li_)
        la_ = la_ + 1
        print(f"  p-grid month {t} done", flush=True)

    def interp(P_t_c, fvals):
        idx = np.clip(np.searchsorted(F_KNOTS, fvals, side="right") - 1,
                      0, len(F_KNOTS) - 2)
        lo, hi = F_KNOTS[idx], F_KNOTS[idx + 1]
        wgt = np.clip((fvals - lo) / (hi - lo), 0.0, 1.0)
        a = P_t_c[np.arange(len(fvals)), :, idx]
        b = P_t_c[np.arange(len(fvals)), :, idx + 1]
        return a * (1.0 - wgt[:, None]) + b * wgt[:, None]

    out = {}
    for myopic in (True, False):
        exec_ci = np.zeros((H, n), dtype=np.int64)
        exec_di = np.zeros((H, n), dtype=np.int64)
        if myopic:
            F = F0.copy()
            for t in range(H):
                best_v = np.full(n, -np.inf)
                for ci in range(C):
                    p = interp(P[t, :, ci], F)
                    imm = p * (bal_t[t][:, None] * (1.0 - d_grid)[None, :]
                               * pshare) - costs[ci]
                    kk = np.argmax(imm, axis=1)
                    v = imm[np.arange(n), kk]
                    upd = v > best_v
                    best_v[upd] = v[upd]
                    exec_ci[t][upd] = ci
                    exec_di[t][upd] = kk[upd]
                F = decay * F + fat_add[exec_ci[t]]
        else:
            best_V = np.full(n, -np.inf)
            bp = np.zeros((n, H), dtype=np.int64)
            bd = np.zeros((n, H), dtype=np.int64)
            for path in itertools.product(range(C), repeat=H):
                g = np.empty(H); f = 0.0
                for t, ci in enumerate(path):
                    g[t] = f
                    f = decay * f + fat_add[ci]
                V = np.zeros(n)
                d_pick = np.zeros((n, H), dtype=np.int64)
                for t in range(H - 1, -1, -1):
                    ci = path[t]
                    p = interp(P[t, :, ci], decay ** t * F0 + g[t])
                    cash = bal_t[t][:, None] * (1.0 - d_grid)[None, :] * pshare
                    tot = -costs[ci] + p * cash + ((1.0 - p)
                          * ((1.0 - q_t[t]) * V)[:, None])
                    kk = np.argmax(tot, axis=1)
                    V = tot[np.arange(n), kk]
                    d_pick[:, t] = kk
                upd = V > best_V
                if upd.any():
                    best_V[upd] = V[upd]
                    bp[upd] = np.asarray(path)
                    bd[upd] = d_pick[upd]
            exec_ci, exec_di = bp.T, bd.T
        out["myopic" if myopic else "sequential"] = pd.DataFrame({
            "loan_id": np.tile(eval_df["loan_id"].to_numpy(), H),
            "month": np.repeat(np.arange(H), n),
            "channel": [CHANNELS[c] for c in exec_ci.ravel()],
            "discount_offer": d_grid[exec_di.ravel()],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True,
                    help="EVAL world (decision at k, grading)")
    ap.add_argument("--train-world", type=Path, default=None,
                    help="world supplying the training rows (default: --world, "
                         "with an even/odd loan split)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--eval-loans", type=int, default=4000)
    ap.add_argument("--cheat-z", action="store_true",
                    help="UPPER REFERENCE: include the true hidden zman_* as "
                         "features (perfect frozen belief) — never a "
                         "band-bearing row")
    args = ap.parse_args()
    k, H = args.k, args.horizon
    sim_cfg = load_config(str(args.config))
    db = args.world / "sim.duckdb"

    df, obs_cols, lag_cols = load_rows(db, k)
    if args.train_world is not None:
        train_df, _, _ = load_rows(args.train_world / "sim.duckdb", k)
        ev_pool = sorted(set(df["loan_id"]))
    else:
        lnum = df["loan_id"].str.slice(1).astype(int)
        train_df = df[lnum % 2 == 0]
        ev_pool = sorted(set(df.loc[lnum % 2 == 1, "loan_id"]))
    book_k = _book_at_month(db, k)
    alive_set = set(book_k["loan_id"])
    alive = [l for l in ev_pool if l in alive_set][:args.eval_loans]
    ev = df[(df.loan_id.isin(alive)) & (df.month == k)].dropna(subset=lag_cols)
    ev = ev.sort_values("loan_id").reset_index(drop=True)
    print(f"train rows {len(train_df)} | eval loans {len(ev)}", flush=True)
    pshare = paid_share_mean(sim_cfg["oracle"])

    key = export_answer_key(db, sim_cfg, H=H, loan_ids=list(ev.loan_id),
                            start_month=k)
    ceiling = key["ceiling"].mean()
    board = {"k": k, "H": H, "n_loans": int(len(key)),
             "ceiling_per_loan": round(float(ceiling), 2),
             "myopic_bound_pct": round(100 * float(key["myopic_bound"].mean() / ceiling), 2),
             "rows": {}}
    if args.cheat_z:
        zc = sorted((c for c in df.columns if c.startswith("zman_")),
                    key=lambda c: int(c.split("_")[1]))
        global STATE
        STATE = STATE + zc
        print(f"CHEAT-Z upper reference: + {len(zc)} true hidden features",
              flush=True)
        arms = (False,)
    else:
        arms = (False, True)
    for hist in arms:
        feats, p_c, p_n = fit(train_df, obs_cols, lag_cols, k, history=hist)
        plans = make_plans(ev, feats, p_c, p_n, sim_cfg, k, H, pshare)
        for mode, plan in plans.items():
            arm = ("cheat_frozen_z" if args.cheat_z
                   else ("history" if hist else "flat"))
            name = f"{arm}_{mode}"
            print(f"grading {name} ...", flush=True)
            g = grade_plan(plan, db, sim_cfg, H=H, start_month=k)
            m = key.merge(g, on="loan_id")
            assert len(m) == len(key)
            board["rows"][name] = round(100 * float(m["value"].mean() / ceiling), 2)
            m[["loan_id", "ceiling", "myopic_bound", "value"]].to_csv(
                args.world / f"per_loan_{name}.csv", index=False)
    if "history_sequential" in board["rows"]:
        board["separation_pp"] = {
            "sequential": round(board["rows"]["history_sequential"]
                                - board["rows"]["flat_sequential"], 2),
            "myopic": round(board["rows"]["history_myopic"]
                            - board["rows"]["flat_myopic"], 2)}
    board["cheat_z"] = bool(args.cheat_z)
    print(json.dumps(board, indent=2))
    fname = "v7_board_cheatz.json" if args.cheat_z else "v7_board_pilot.json"
    (args.world / fname).write_text(json.dumps(board, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    main()
