from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from benchmark.closed_loop_arms import (
    STATE, fit_gbm, load_train_rows)
from benchmark.closed_loop_v7 import (
    CHANNELS, D_STEPS, build_world, load_cohort, world_seed_of)
from benchmark.exact_filter import (
    GaussianBelief, manifold_params, oracle_terms, posterior_pay_prob)

DPD_BANDS = [(-1, 30, "early"), (30, 90, "mid"), (90, 10 ** 9, "late")]
DISC_BANDS = [(0.0, 0.15, "low"), (0.15, 0.35, "mid"), (0.35, 0.51, "high")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--train-world", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--e1-dir", type=Path, required=True)
    ap.add_argument("--lags-dir", type=Path, required=True,
                    help="dir holding plan_lags_exactF_sequential.csv (the "
                         "allfitted-equivalent plans at 4,000)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--loans", type=int, default=4000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from decode_sim.config import load_config

    sim_cfg = load_config(str(args.config))
    k, H = args.k, args.horizon
    book, pre, obs_cols, obs_k, *_ = load_cohort(args.world, sim_cfg, k,
                                                 args.loans)
    ids = list(book["loan_id"])
    n = len(ids)
    man = manifold_params(sim_cfg)
    w = build_world(book, sim_cfg, H)
    terms = oracle_terms(w)
    d_grid = terms["d_grid"]

    tdf, lag_cols = load_train_rows(args.train_world / "sim.duckdb", k,
                                    obs_cols)
    feats, p_contact, p_none = fit_gbm(tdf, obs_cols, lag_cols, k,
                                       history=True, seed=0)

    idx = {l: i for i, l in enumerate(ids)}
    pre_obs = np.zeros((n, k, len(obs_cols)))
    pre_ch = np.zeros((n, k))
    pre_ci = np.zeros((n, k), dtype=np.int64)
    from benchmark.closed_loop_arms import CH_CODE
    for _, r in pre.iterrows():
        i, mo = idx[r["loan_id"]], int(r["month"])
        pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
        pre_ch[i, mo] = CH_CODE[r["channel"]]
        pre_ci[i, mo] = CHANNELS.index(r["channel"])
    lag_block = []
    for j in range(1, k + 1):
        mo = k - j
        for dcol in range(len(obs_cols)):
            lag_block.append(pre_obs[:, mo, dcol])
        lag_block.append(pre_ch[:, mo])
    lag_block = np.stack(lag_block, axis=1)
    s0 = np.stack([book["days_past_due"].to_numpy(float),
                   book["days_in_credito_em_liquidacao"].to_numpy(float),
                   book["outstanding_balance"].to_numpy(float),
                   book["entering_fatigue"].to_numpy(float),
                   book["loan_age_months"].to_numpy(float),
                   book["age"].to_numpy(float),
                   book["monthly_income"].to_numpy(float),
                   book["original_principal"].to_numpy(float),
                   book["interest_rate_monthly"].to_numpy(float)], axis=1)
    base_feats = np.concatenate([s0, obs_k, lag_block], axis=1)

    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], n)
    for s in range(k):
        bel.observe(pre_obs[:, s, :])
        bel.act(pre_ci[:, s])
    bel.observe(obs_k)
    post = bel.posterior_z0()
    Wm = np.stack([man["W"][c] for c in CHANNELS])

    dpd = book["days_past_due"].to_numpy(float)
    F0 = book["entering_fatigue"].to_numpy(float)
    dpd_band = np.select([dpd <= 30, dpd <= 90], ["early", "mid"], "late")

    rows = []
    flagged_cells = set()
    for ci, ch in enumerate(CHANNELS):
        mean, sd = bel.score_moments(Wm[ci], man["coef"], 0, np.zeros(bel.d),
                                     post=post)
        pen = terms["beta_i"] * F0 if ch != "none" else np.zeros(n)
        exact = posterior_pay_prob(
            terms["logits0"][0, :, ci, :] - pen[:, None], mean, sd)
        if ch == "none":
            fitted = np.repeat(p_none(base_feats)[:, None], D_STEPS, 1)
        else:
            fitted = np.stack([p_contact(base_feats, d, ci)
                               for d in d_grid], 1)
        p_lo, p_hi = fitted.min(), fitted.max()
        for lo, hi, dband in DISC_BANDS:
            dmask = (d_grid >= lo) & (d_grid < hi)
            if ch == "none" and dband != "low":
                continue
            for b in ("early", "mid", "late"):
                m = dpd_band == b
                if not m.any():
                    continue
                fsub = fitted[np.ix_(m, dmask)]
                esub = exact[np.ix_(m, dmask)]
                at_clip = float(((fsub <= p_lo + 1e-9)
                                 | (fsub >= p_hi - 1e-9)).mean())
                cell = (ch, dband, b)
                if at_clip > 0.5:
                    flagged_cells.add(cell)
                rows.append({
                    "channel": ch, "disc_band": dband, "dpd_band": b,
                    "n_cells": int(fsub.size),
                    "bias_mean": float((fsub - esub).mean()),
                    "bias_abs_mean": float(np.abs(fsub - esub).mean()),
                    "exact_mean": float(esub.mean()),
                    "share_at_calibrator_extreme": at_clip,
                })
    bias = pd.DataFrame(rows)

    def load_plan(path):
        p = pd.read_csv(path)
        return p.pivot(index="loan_id", columns="month",
                       values=["channel", "discount_offer"]).loc[ids]

    ref = load_plan(args.lags_dir / "plan_lags_exactF_sequential.csv")
    flips = {}
    for arm in ("hybrid_none_exact_sequential",
                "hybrid_contact_exact_sequential",
                "hybrid_none_exact_static_sequential",
                "hybrid_contact_exact_static_sequential"):
        f = args.e1_dir / f"plan_{arm}.csv"
        if not f.exists():
            continue
        hyb = load_plan(f)
        diff = ((hyb["channel"] != ref["channel"])
                | (hyb["discount_offer"] != ref["discount_offer"]))
        month0 = diff[0]
        ch0 = hyb["channel"][0]
        d0 = hyb["discount_offer"][0].to_numpy(float)
        dband0 = np.select([d0 < 0.15, d0 < 0.35], ["low", "mid"], "high")
        in_flagged = [
            (c, db, b) in flagged_cells
            for c, db, b in zip(ch0, dband0, dpd_band)]
        flips[arm] = {
            "plan_cells_flipped": int(diff.to_numpy().sum()),
            "of_total": int(diff.size),
            "month0_flips": int(month0.sum()),
            "month0_flips_into_flagged_cells":
                int((month0.to_numpy() & np.asarray(in_flagged)).sum()),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    bias.to_csv(args.out / "bias_map.csv", index=False)
    (args.out / "flips.json").write_text(json.dumps(
        {"flagged_cells": sorted(map(list, flagged_cells)),
         "flips": flips}, indent=2) + "\n")
    print(bias.sort_values("bias_abs_mean", ascending=False)
          .head(12).to_string(index=False))
    print(json.dumps(flips, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
