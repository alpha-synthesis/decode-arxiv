import json
import os
import re
import sys

import numpy as np
import pandas as pd

FANNIE_DIR = os.environ.get("FANNIE_DIR", "data/fannie")
CSV_PATH = os.path.join(FANNIE_DIR, "fannie_performance_loan_month_large.csv")
OUT_DIR = os.path.join(FANNIE_DIR, "artifacts_collections")
CHUNKSIZE = 200_000

HORIZONS = [6, 12]
TREATMENT_WINDOW = 3
ENTRY_STATUS_THRESHOLD = 1

STATUS_COL = "current_loan_delinquency_status"
PERIOD_COL = "monthly_reporting_period"

STATIC_COLS = [
    "channel",
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "original_dti",
    "borrower_credit_score_at_origination",
    "co_borrower_credit_score_at_origination",
    "first_time_home_buyer_indicator",
    "loan_purpose",
    "property_type",
    "number_of_units",
    "occupancy_status",
    "property_state",
    "mortgage_insurance_percentage",
    "amortization_type",
    "interest_only_loan_indicator",
    "high_balance_loan_indicator",
    "relocation_mortgage_indicator",
    "special_eligibility_program",
]

DYNAMIC_COLS = [
    "current_interest_rate",
    "current_actual_upb",
    "loan_age",
    "remaining_months_to_maturity",
    "remaining_months_to_legal_maturity",
]

TREATMENT_COLS = [
    "borrower_assistance_plan",
    "modification_flag",
    "payment_deferral_modification_event_indicator",
    "total_deferral_amount",
    "principal_forgiveness_amount",
    "alternative_delinquency_resolution",
    "alternative_delinquency_resolution_count",
    "servicing_activity_indicator",
]

OUTCOME_COLS = ["zero_balance_code", "foreclosure_date"]

USECOLS = (
    ["loan_id", PERIOD_COL, STATUS_COL]
    + STATIC_COLS
    + DYNAMIC_COLS
    + TREATMENT_COLS
    + OUTCOME_COLS
)

GOOD_ZB = {"1", "01"}
BAD_ZB = {"2", "02", "3", "03", "6", "06", "9", "09", "15", "96", "97", "98"}


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
    s = str(value).strip()
    m = re.match(r"^(\d+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def norm_zb(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\.0$", "", s)
    return s


def load_panel() -> pd.DataFrame:
    print("[1/4] Streaming CSV in chunks (usecols only)...", flush=True)
    parts = []
    rows = 0
    for chunk in pd.read_csv(
        CSV_PATH, usecols=USECOLS, chunksize=CHUNKSIZE, low_memory=False
    ):
        rows += len(chunk)
        chunk["status_num"] = chunk[STATUS_COL].map(parse_status)
        chunk["period_ord"] = chunk[PERIOD_COL].map(period_ordinal)
        chunk["zb"] = chunk["zero_balance_code"].map(norm_zb)
        chunk = chunk.dropna(subset=["period_ord"])
        parts.append(chunk)
        print(f"    rows read: {rows:,}", flush=True)
    panel = pd.concat(parts, ignore_index=True)
    del parts
    panel = panel.sort_values(["loan_id", "period_ord"], kind="mergesort")
    print(f"    panel: {len(panel):,} rows, {panel['loan_id'].nunique():,} loans", flush=True)
    return panel


def build_episodes(panel: pd.DataFrame):
    print("[2/4] Detecting delinquency spells per loan...", flush=True)
    episodes = []
    stats = {
        "loans_total": int(panel["loan_id"].nunique()),
        "entry_status_threshold": ENTRY_STATUS_THRESHOLD,
        "loans_with_default": 0,
        "loans_with_episode": 0,
        "episodes_detected": 0,
        "episodes_censored": {f"h{h}": 0 for h in HORIZONS},
        "episodes_labeled": {f"h{h}": 0 for h in HORIZONS},
    }

    max_h = max(HORIZONS)
    n_loans_done = 0

    for loan_id, g in panel.groupby("loan_id", sort=False):
        n_loans_done += 1
        if n_loans_done % 5000 == 0:
            print(f"    loans processed: {n_loans_done:,}", flush=True)

        status = g["status_num"].to_numpy()
        period = g["period_ord"].to_numpy()
        zb = g["zb"].to_numpy()
        n = len(g)

        in_default = status >= ENTRY_STATUS_THRESHOLD
        prev_def = np.concatenate(([False], in_default[:-1]))
        starts = np.where(in_default & ~prev_def)[0]
        if len(starts) == 0:
            continue
        stats["loans_with_default"] += 1
        stats["loans_with_episode"] += 1

        last_period = period[-1]
        terminal_zb = next((z for z in zb if z), "")

        for spell_no, i0 in enumerate(starts, start=1):
            stats["episodes_detected"] += 1
            t0 = period[i0]
            row = g.iloc[i0]

            ep = {
                "loan_id": loan_id,
                "spell_no": spell_no,
                "spell_start_period_ord": t0,
                "spell_start_period": row[PERIOD_COL],
                "delinquency_depth_at_entry": status[i0],
            }
            for c in STATIC_COLS + DYNAMIC_COLS:
                ep[c] = row[c]

            after = (period > t0) & (period <= t0 + max_h)
            within = {
                h: (period > t0) & (period <= t0 + h) for h in HORIZONS
            }

            cure_month = np.nan
            cured_idx = np.where((period > t0) & (status < ENTRY_STATUS_THRESHOLD))[0]
            if len(cured_idx):
                cure_month = period[cured_idx[0]]

            for h in HORIZONS:
                key = f"h{h}"
                label = np.nan
                if not np.isnan(cure_month) and cure_month <= t0 + h:
                    label = 1
                else:
                    obs_horizon = min(last_period, t0 + h)
                    if terminal_zb and last_period <= t0 + h:
                        label = 1 if terminal_zb in GOOD_ZB else 0
                    elif last_period >= t0 + h:
                        label = 0
                    else:
                        label = np.nan
                ep[f"cure_{key}"] = label
                if np.isnan(label):
                    stats["episodes_censored"][key] += 1
                else:
                    stats["episodes_labeled"][key] += 1

            observed_end = min(last_period, t0 + max_h)
            ep["followup_months"] = float(observed_end - t0)
            if not np.isnan(cure_month) and cure_month <= observed_end:
                ep["cure_event"] = 1
                ep["time_to_cure"] = float(cure_month - t0)
            else:
                ep["cure_event"] = 0
                ep["time_to_cure"] = float(observed_end - t0)
            ep["terminal_bad"] = int(
                bool(terminal_zb) and terminal_zb in BAD_ZB and last_period <= t0 + max_h
            )

            tw_end = t0 + TREATMENT_WINDOW
            if not np.isnan(cure_month):
                tw_end = min(tw_end, cure_month - 1)
            tw_mask = (period >= t0) & (period <= tw_end)
            tg = g.loc[tw_mask]

            bap = tg["borrower_assistance_plan"].astype(str).str.strip().str.upper()
            ep["treat_forbearance"] = int((bap == "F").any())
            ep["treat_trial_period"] = int((bap == "T").any())
            ep["treat_modification"] = int(
                (tg["modification_flag"].astype(str).str.strip().str.upper() == "Y").any()
            )
            ep["treat_payment_deferral"] = int(
                (
                    tg["payment_deferral_modification_event_indicator"]
                    .astype(str).str.strip().str.upper() == "Y"
                ).any()
            )
            ep["treat_deferral_amount"] = float(
                pd.to_numeric(tg["total_deferral_amount"], errors="coerce").fillna(0).max()
            )
            ep["treat_forgiveness_amount"] = float(
                pd.to_numeric(tg["principal_forgiveness_amount"], errors="coerce").fillna(0).max()
            )
            adr = tg["alternative_delinquency_resolution"].astype(str).str.strip().str.upper()
            ep["treat_alt_resolution"] = int(adr.isin(["Y", "P", "C"]).any())
            ep["treat_alt_resolution_count"] = float(
                pd.to_numeric(
                    tg["alternative_delinquency_resolution_count"], errors="coerce"
                ).fillna(0).max()
            )
            ep["treat_servicing_activity"] = int(
                (
                    tg["servicing_activity_indicator"].astype(str).str.strip().str.upper() == "Y"
                ).any()
            )
            ep["treat_any"] = int(
                ep["treat_forbearance"]
                or ep["treat_trial_period"]
                or ep["treat_modification"]
                or ep["treat_payment_deferral"]
                or ep["treat_alt_resolution"]
            )

            episodes.append(ep)

    df = pd.DataFrame(episodes)
    return df, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel()
    episodes, stats = build_episodes(panel)
    del panel

    print("[3/4] Summarizing...", flush=True)
    for h in HORIZONS:
        col = f"cure_h{h}"
        labeled = episodes[col].dropna()
        stats[f"cure_rate_h{h}"] = float(labeled.mean()) if len(labeled) else None
    stats["treatment_prevalence"] = {
        c: float(episodes[c].mean())
        for c in episodes.columns
        if c.startswith("treat_") and episodes[c].dtype != object
    }
    stats["cure_rate_by_treat_any_h6"] = {
        str(k): float(v)
        for k, v in episodes.dropna(subset=["cure_h6"]).groupby("treat_any")["cure_h6"].mean().items()
    }

    print("[4/4] Saving...", flush=True)
    out_csv = os.path.join(OUT_DIR, "cure_episodes.csv")
    episodes.to_csv(out_csv, index=False)
    with open(os.path.join(OUT_DIR, "episode_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2), flush=True)
    print(f"Saved {len(episodes):,} episodes -> {out_csv}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
