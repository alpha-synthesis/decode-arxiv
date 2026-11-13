from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIGITAL = {"sms", "whatsapp", "app_push", "email"}
HUMAN = {"phone", "field_visit", "personal"}
ACTIVE_STATUS = {"delinquent", "credito_em_liquidacao"}
HORIZON = 9


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)
    y = df["recovered"].astype(int)
    is_contact = (df["channel"] != "none").astype(int)
    is_dig = df["channel"].isin(DIGITAL).astype(int)
    is_hum = df["channel"].isin(HUMAN).astype(int)
    is_non = (df["channel"] == "none").astype(int)
    m = df["month"].clip(lower=1)

    def cum_before(s):
        return (s.groupby(df["loan_id"], sort=False).cumsum()
                 .groupby(df["loan_id"], sort=False).shift(1).fillna(0.0))

    df["prev_disc"] = g["discount_offer"].shift(1).fillna(0.0)
    df["prev_y"] = g["recovered"].shift(1).astype("float").fillna(0.0)
    df["prev2_y"] = g["recovered"].shift(2).astype("float").fillna(0.0)
    df["cum_contacts"] = cum_before(is_contact)
    df["cum_y"] = cum_before(y)
    df["run_rate"] = df["cum_y"] / m
    df["dig_n"] = cum_before(is_dig)
    df["hum_n"] = cum_before(is_hum)
    df["non_n"] = cum_before(is_non)
    df["dig_rate"] = df["dig_n"] / m
    df["hum_rate"] = df["hum_n"] / m
    df["non_rate"] = df["non_n"] / m
    contact_month = df["month"].where(is_contact == 1)
    last_cm = (contact_month.groupby(df["loan_id"], sort=False).ffill()
               .groupby(df["loan_id"], sort=False).shift(1))
    df["mo_since"] = (df["month"] - last_cm).fillna(df["month"]).astype(float)
    df["pchan"] = g["channel"].shift(1).fillna("none")
    df["p2chan"] = g["channel"].shift(2).fillna("none")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    df["recovered_1m"] = df["recovered"].astype(int)
    df["amount_1m"] = df["recovered_amount"].astype(float)
    rec9 = np.zeros(len(df), dtype=int)
    evt = np.zeros(len(df), dtype=int)
    evo = np.zeros(len(df), dtype=int)
    amt9 = np.zeros(len(df), dtype=float)
    rec = df["recovered"].to_numpy().astype(int)
    amt = df["recovered_amount"].to_numpy().astype(float)
    for _, idx in df.groupby("loan_id", sort=False).indices.items():
        idx = np.sort(idx)
        n = len(idx)
        for k in range(n):
            win = idx[k:k + HORIZON]
            r = rec[win]
            paid = r.max() == 1 if len(r) else False
            first = int(np.argmax(r == 1)) if paid else max(len(r) - 1, 0)
            row = idx[k]
            rec9[row] = int(paid)
            evt[row] = first
            evo[row] = int(paid)
            amt9[row] = float(amt[win].sum())
    df["recovered_9m"] = rec9
    df["event_time"] = evt
    df["event_observed"] = evo
    df["amount_9m"] = amt9
    return df


def build(ds_dir: str | Path) -> Path:

    ds_dir = Path(ds_dir)
    panel = pd.read_csv(ds_dir / "flat.csv", low_memory=False)
    loans = pd.read_csv(ds_dir / "loans.csv", low_memory=False)
    panel = add_history(panel)
    panel = add_targets(panel)
    panel["post_outstanding_balance"] = panel["outstanding_balance"]
    panel["post_days_past_due"] = panel["days_past_due"]
    panel["post_days_in_credito_em_liquidacao"] = panel["days_in_credito_em_liquidacao"]
    panel["post_status"] = panel["next_status"]
    dt = build_decision_table(panel, loans)
    dt["dpd_band"] = pd.cut(dt["days_past_due"], [-1, 30, 90, 10**9],
                            labels=["early", "mid", "late"]).astype(str)
    env_statuses = os.environ.get("BENCH_STATUSES", "").strip()
    keep = {s.strip() for s in env_statuses.split(",") if s.strip()} or ACTIVE_STATUS
    before = len(dt)
    dt = dt[dt["status"].isin(keep)].reset_index(drop=True)
    out = ds_dir / "decision_table.csv"
    dt.to_csv(out, index=False)
    rr1 = dt["recovered_1m"].mean()
    rr9 = dt["recovered_9m"].mean()
    print(f"  {out}: {len(dt):,}/{before:,} active rows | recovered_1m={rr1:.4f} "
          f"recovered_9m={rr9:.4f} | dpd_band={dt['dpd_band'].value_counts().to_dict()}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m decode_sim.bridge.decision_table <world-dir>")
        return 2
    for d in argv[1:]:
        build(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


LEAKY_END_OF_MONTH = [
    "outstanding_balance",
    "days_past_due",
    "days_in_credito_em_liquidacao",
    "contacts_last_3m",
]
OPENING_FROM_LOANS = {
    "outstanding_balance": "outstanding_balance",
    "days_past_due": "days_past_due",
    "days_in_credito_em_liquidacao": "days_in_credito_em_liquidacao",
    "contacts_last_3m": None,
}
ID_COLS = ("loan_id", "month")


def build_decision_table(panel, loans, *, leaky_end_of_month=None, opening_from_loans=None):
    if leaky_end_of_month is None:
        leaky_end_of_month = LEAKY_END_OF_MONTH
    if opening_from_loans is None:
        opening_from_loans = OPENING_FROM_LOANS
    missing = [c for c in leaky_end_of_month if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing dynamic columns: {missing}")
    df = panel.sort_values(list(ID_COLS)).reset_index(drop=True).copy()
    opening = loans.drop_duplicates("loan_id").set_index("loan_id")
    is_m0 = (df["month"] == 0).to_numpy()
    for col in leaky_end_of_month:
        entering = df.groupby("loan_id")[col].shift(1)
        src = opening_from_loans[col]
        if src is not None and src in opening.columns:
            m0_vals = df["loan_id"].map(opening[src])
        else:
            m0_vals = pd.Series(0.0, index=df.index)
        df[col] = entering.where(~is_m0, m0_vals)
    n_bad = int(df[leaky_end_of_month].isna().any(axis=1).sum())
    if n_bad:
        df = df.dropna(subset=leaky_end_of_month).reset_index(drop=True)
    if "origination_month" in opening.columns:
        df["origination_month"] = df["loan_id"].map(opening["origination_month"])
    return df
