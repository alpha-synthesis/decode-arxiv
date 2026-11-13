from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


FNMA = Path(os.environ.get("FANNIE_DIR", "data/fannie"))

NUMERIC = ["delinquency_depth_at_entry", "original_interest_rate",
           "current_interest_rate", "original_upb", "current_actual_upb",
           "original_loan_term", "loan_age", "remaining_months_to_maturity",
           "remaining_months_to_legal_maturity", "original_ltv",
           "original_cltv", "number_of_borrowers"]
CATEG = ["channel", "first_time_home_buyer_indicator", "loan_purpose",
         "property_type", "occupancy_status"]


def period_ordinal(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    s = re.sub(r"\.0$", "", s)
    if not s.isdigit() or len(s) < 5:
        return np.nan
    month = int(s[:-4])
    year = int(s[-4:])
    if not 1 <= month <= 12:
        return np.nan
    return year * 12 + (month - 1)


def parse_status(value) -> float:
    if pd.isna(value):
        return np.nan
    m = re.match(r"^(\d+)", str(value).strip())
    return float(m.group(1)) if m else np.nan


def fold_predictions(X_flat, X_hist, y, groups, fold_of_loan=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    if fold_of_loan is None:
        splits = list(GroupKFold(n_splits=5).split(X_flat, y, groups))
    else:
        rf = fold_of_loan[groups]
        splits = [(np.flatnonzero(rf != f), np.flatnonzero(rf == f))
                  for f in np.unique(rf)]
    fold = np.full(len(y), -1)
    pf = np.full(len(y), np.nan)
    ph = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(splits):
        pf[te] = HistGradientBoostingClassifier(max_iter=300, random_state=0) \
            .fit(X_flat[tr], y[tr]).predict_proba(X_flat[te])[:, 1]
        ph[te] = HistGradientBoostingClassifier(max_iter=300, random_state=0) \
            .fit(X_hist[tr], y[tr]).predict_proba(X_hist[te])[:, 1]
        fold[te] = f
    return fold, pf, ph


def fold_aucs(y, pf, ph, fold, w=None):
    from sklearn.metrics import roc_auc_score

    af, ah = [], []
    for f in np.unique(fold):
        s = fold == f
        ww = None if w is None else w[s]
        af.append(roc_auc_score(y[s], pf[s], sample_weight=ww))
        ah.append(roc_auc_score(y[s], ph[s], sample_weight=ww))
    return np.array(af), np.array(ah)


def delta_auc(X_flat, X_hist, y, groups, label, preds=None):
    fold, pf, ph = preds if preds is not None else \
        fold_predictions(X_flat, X_hist, y, groups)
    af, ah = fold_aucs(y, pf, ph, fold)
    aucs = {"flat": list(map(float, af)), "hist": list(map(float, ah))}
    d = float(np.mean(ah) - np.mean(af))
    print(f"{label:28s} AUC flat {np.mean(af):.4f} "
          f"(+/- {np.std(af):.4f})  "
          f"hist {np.mean(ah):.4f} "
          f"(+/- {np.std(ah):.4f})  dAUC {d:+.4f}", flush=True)
    return d, aucs


def grouped_bootstrap_ci(X_flat, X_hist, y, groups, n_boot=10_000, seed=0,
                         preds=None):
    fold, pf, ph = preds if preds is not None else \
        fold_predictions(X_flat, X_hist, y, groups)
    rng = np.random.default_rng(seed)
    G = int(groups.max()) + 1
    folds = np.unique(fold)
    loans_by_fold = [np.unique(groups[fold == f]) for f in folds]
    deltas = np.empty(n_boot)
    redraws = 0
    b = 0
    while b < n_boot:
        w = np.zeros(G)
        for L in loans_by_fold:
            w += np.bincount(rng.choice(L, size=len(L), replace=True),
                             minlength=G)
        wr = w[groups]
        if any(len(np.unique(y[(fold == f) & (wr > 0)])) < 2 for f in folds):
            redraws += 1
            continue
        af, ah = fold_aucs(y, pf, ph, fold, wr)
        deltas[b] = float(np.mean(ah - af))
        b += 1
    return ([float(np.percentile(deltas, 2.5)),
             float(np.percentile(deltas, 97.5))],
            float((deltas <= 0).mean()), float(np.std(deltas, ddof=1)), redraws)


def pooled_bootstrap_ci_archived(y, pf, ph, fold, groups, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score

    order = pd.unique(groups[np.argsort(fold, kind="stable")])
    rng = np.random.default_rng(seed)
    G = int(groups.max()) + 1
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(order, size=len(order), replace=True)
        w = np.bincount(pick, minlength=G)[groups].astype(float)
        if len(np.unique(y[w > 0])) < 2:
            continue
        deltas.append(roc_auc_score(y, ph, sample_weight=w)
                      - roc_auc_score(y, pf, sample_weight=w))
    deltas = np.array(deltas)
    return ([float(np.percentile(deltas, 2.5)),
             float(np.percentile(deltas, 97.5))],
            float((deltas <= 0).mean()))


def partition_spread(X_flat, X_hist, y, groups, n_partitions=20, seed=7):
    rng = np.random.default_rng(seed)
    G = int(groups.max()) + 1
    pts = []
    for _ in range(n_partitions):
        fold_of_loan = np.empty(G, dtype=int)
        fold_of_loan[rng.permutation(G)] = np.arange(G) % 5
        fo, pf, ph = fold_predictions(X_flat, X_hist, y, groups, fold_of_loan)
        af, ah = fold_aucs(y, pf, ph, fo)
        pts.append(float(np.mean(ah - af)))
    return pts


def transport_statistic(X_flat, X_hist, y, groups, label, n_boot, boot_seed,
                        n_partitions, partition_seed, interval="aligned"):
    import hashlib

    import sklearn

    preds = fold_predictions(X_flat, X_hist, y, groups)
    d, aucs = delta_auc(X_flat, X_hist, y, groups, label, preds=preds)
    if interval == "archived_pooled":
        ci, p_le0 = pooled_bootstrap_ci_archived(y, preds[1], preds[2], preds[0],
                                                 groups, n_boot, boot_seed)
        extra = {"interval": "archived: loans resampled over all folds, pooled "
                             "AUC difference of the concatenated predictions"}
    elif interval == "aligned":
        ci, p_le0, se, redraws = grouped_bootstrap_ci(
            X_flat, X_hist, y, groups, n_boot, boot_seed, preds=preds)
        extra = {"boot_se": se, "boot_redraws_one_class": int(redraws),
                 "interval": "loans resampled within GroupKFold(5) folds; mean "
                             "over folds of the AUC difference; conditional on "
                             "the fitted fold models and the partition"}
    else:
        raise ValueError(interval)
    out = {"dauc": d, "ci": ci, "p_le_0": p_le0, **extra,
           "fold_aucs": aucs, "n_boot": int(n_boot), "boot_seed": int(boot_seed),
           "fold_assignment_sha256": hashlib.sha256(
               np.ascontiguousarray(preds[0]).tobytes()).hexdigest(),
           "n_loans": int(len(np.unique(groups))),
           "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__}}
    if n_partitions:
        pts = partition_spread(X_flat, X_hist, y, groups, n_partitions,
                               partition_seed)
        out["partitions"] = {"n": int(n_partitions), "seed": int(partition_seed),
                             "points": pts, "mean": float(np.mean(pts)),
                             "sd": float(np.std(pts, ddof=1)),
                             "min": float(np.min(pts)), "max": float(np.max(pts))}
    print(f"{label:28s} dAUC {d:+.4f}  CI95 {ci}  P(<=0) {p_le0:.4f}"
          + (f"  partitions mean {out['partitions']['mean']:+.4f} "
             f"sd {out['partitions']['sd']:.4f}" if n_partitions else ""),
          flush=True)
    return out


def real_side(k_hist: int = 12, n_boot: int = 10_000, boot_seed: int = 0,
              n_partitions: int = 20, partition_seed: int = 7,
              interval: str = "aligned"):
    import yaml
    cfg = yaml.safe_load(
        Path("configs/fannie.yaml")
        .read_text())
    declared = set(cfg["features"]["numeric"])
    assert set(NUMERIC) <= declared, \
        f"entry-state drift vs fannie.yaml: {set(NUMERIC) - declared}"
    declared_cat = set(cfg["features"].get("categorical") or [])
    assert set(CATEG) <= declared_cat, \
        f"categorical drift vs fannie.yaml: {set(CATEG) - declared_cat}"

    epi = pd.read_csv(FNMA / "artifacts_collections" / "cure_episodes.csv")
    epi = epi.dropna(subset=["cure_h6"]).reset_index(drop=True)
    print(f"episodes with cure_h6 label: {len(epi)}", flush=True)

    con = duckdb.connect()
    monthly = con.execute(f"""
        SELECT loan_id, monthly_reporting_period,
               current_loan_delinquency_status AS dlq
        FROM read_csv_auto('{FNMA / "fannie_performance_loan_month_large.csv"}',
                           ALL_VARCHAR=TRUE)
    """).df()
    con.close()
    monthly["ord"] = monthly["monthly_reporting_period"].map(period_ordinal)
    monthly["dlq_n"] = monthly["dlq"].map(parse_status)
    monthly = monthly.dropna(subset=["ord"])
    monthly["loan_id"] = monthly["loan_id"].astype(np.int64)
    epi["loan_id"] = epi["loan_id"].astype(np.int64)
    idx = {(l, o): v for l, o, v in zip(monthly["loan_id"],
                                        monthly["ord"].astype(int),
                                        monthly["dlq_n"])}
    matched = np.fromiter(
        ((l, int(t)) in idx or (l, int(t) - 1) in idx
         for l, t in zip(epi["loan_id"],
                         epi["spell_start_period_ord"])), bool)
    assert matched.mean() > 0.99, \
        f"loan_id join failure: only {matched.mean():.3f} episodes matched"
    H = np.full((len(epi), k_hist), np.nan)
    t0 = epi["spell_start_period_ord"].astype(int).to_numpy()
    lids = epi["loan_id"].to_numpy()
    for r in range(len(epi)):
        for j in range(1, k_hist + 1):
            H[r, j - 1] = idx.get((lids[r], t0[r] - j), np.nan)
    msc = np.full(len(epi), np.nan)
    epi_sorted = epi.sort_values(["loan_id", "spell_no"])
    prev = {}
    for i, row in epi_sorted.iterrows():
        key = row["loan_id"]
        if key in prev:
            p_t0, p_cured, p_ttc = prev[key]
            if p_cured and np.isfinite(p_ttc):
                msc[i] = row["spell_start_period_ord"] - (p_t0 + p_ttc)
        prev[key] = (row["spell_start_period_ord"],
                     bool(row.get("cure_event", 0)), row.get("time_to_cure"))
    hist_cols = [f"dlq_lag{j}" for j in range(1, k_hist + 1)]
    hist_df = pd.DataFrame(H, columns=hist_cols)
    hist_df["spell_no"] = epi["spell_no"].to_numpy(float)
    hist_df["months_since_prev_cure"] = np.clip(msc, 0, 120)

    flat = epi[NUMERIC].astype(float).copy()
    for c in CATEG:
        flat[c] = pd.factorize(epi[c].astype(str))[0]
    X_flat = flat.to_numpy(float)
    X_hist = np.column_stack([X_flat, hist_df.to_numpy(float)])
    y = epi["cure_h6"].astype(int).to_numpy()
    groups = pd.factorize(epi["loan_id"])[0]
    print(f"y rate {y.mean():.4f}  loans {len(np.unique(groups))}  "
          f"hist NaN share {np.isnan(H).mean():.3f}", flush=True)
    stat = transport_statistic(X_flat, X_hist, y, groups, "REAL (fannie cure_h6)",
                               n_boot, boot_seed, n_partitions, partition_seed,
                               interval)
    return {"side": "real", **stat,
            "n_rows": int(len(epi)), "y_rate": float(y.mean())}


def synthetic_side(world: Path, k: int = 6, state_timing: str = "entering",
                   n_boot: int = 10_000, boot_seed: int = 0,
                   n_partitions: int = 20, partition_seed: int = 7,
                   interval: str = "aligned"):
    from benchmark.b_demo_probe import (
        FLAT, load_panel, measurement_frame)
    df, obs_cols, _ = load_panel(world, state_timing)
    dfl, m_all, _, lag_cols = measurement_frame(
        df, obs_cols, k, rows=10 ** 9, min_rows=1)
    m = dfl[dfl["month"] == k].dropna(subset=lag_cols)
    dt = pd.read_csv(world / "decision_table.csv",
                     usecols=["loan_id", "month", "recovered_9m"])
    n_before = len(m)
    m = m.merge(dt[dt["month"] == k][["loan_id", "recovered_9m"]],
                on="loan_id", how="inner")
    assert len(m) <= n_before, "decision_table merge duplicated rows"
    y = m["recovered_9m"].astype(int).to_numpy()
    d_cols = ["discount_offer"]
    X_flat = m[FLAT + obs_cols + d_cols].to_numpy(float)
    X_hist = m[FLAT + obs_cols + d_cols + lag_cols].to_numpy(float)
    groups = pd.factorize(m["loan_id"])[0]
    print(f"{world.name}: rows {len(m)}  y rate {y.mean():.4f}  "
          f"state timing {state_timing}", flush=True)
    stat = transport_statistic(X_flat, X_hist, y, groups, f"SYN ({world.name})",
                               n_boot, boot_seed, n_partitions, partition_seed,
                               interval)
    return {"side": "synthetic", "world": world.name, **stat,
            "state_timing": state_timing, "n_rows": int(len(m)),
            "y_rate": float(y.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("real", "synthetic"), required=True)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--state-timing", choices=("entering", "post_outcome_archived"),
                    default="entering",
                    help="synthetic side: entering (decision-time values, "
                         "default) or post_outcome_archived (archived runs)")
    ap.add_argument("--interval", choices=("aligned", "archived_pooled"),
                    default="aligned",
                    help="aligned (default) or archived_pooled; the archived "
                         "outputs need --interval archived_pooled --n-boot 1000 "
                         "--partitions 0 (and, synthetic side, --state-timing "
                         "post_outcome_archived)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--partitions", type=int, default=20,
                    help="random loan-grouped partitions with refitted models "
                         "(0 skips)")
    ap.add_argument("--partition-seed", type=int, default=7)
    args = ap.parse_args()
    kw = dict(n_boot=args.n_boot, boot_seed=args.boot_seed,
              n_partitions=args.partitions, partition_seed=args.partition_seed,
              interval=args.interval)
    res = real_side(**kw) if args.side == "real" \
        else synthetic_side(args.world, args.k, args.state_timing, **kw)
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
