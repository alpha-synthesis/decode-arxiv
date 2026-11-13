from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

BAND_LOG: list[dict] = []
EXPECTED_SEEDS = {"d05": {"train": 17001, "infer": 17002},
                  "d40": {"train": 18001, "infer": 18002},
                  "d40_rev2": {"train": 18011, "infer": 18012},
                  "v8_q01": {"train": 19001, "infer": 19002},
                  "v8_q02": {"train": 20001, "infer": 20002}}


def band(name, value, ok, detail="", warn_only=False):
    tag = ("WARN" if not ok else "ok") if warn_only \
        else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {name}: {value} {detail}", flush=True)
    BAND_LOG.append({"band": name, "value": value, "pass": bool(ok),
                     "warn_only": warn_only, "detail": detail})
    return bool(ok) or warn_only


def probe(world: Path, k: int, state_timing: str = "entering"):
    r = subprocess.run(
        [sys.executable, "gates/v7_pilot_gap.py", "--world", str(world),
         "--k", str(k), "--state-timing", state_timing],
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        raise SystemExit(f"probe failed on {world}")
    flat = float(r.stdout.split("flat (row + current keyhole)")[1]
                 .split("R2 = ")[1].split()[0])
    sep = float(r.stdout.split("SEPARATION GAP (history - flat): ")[1].split()[0])
    head = float(r.stdout.split("HEADROOM LEFT (oracle - history): ")[1].split()[0])
    return flat, sep, head


def obs_only_r2(world: Path, k: int) -> float:
    import duckdb
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    ch_code = {"none": 0.0, "sms": 1.0, "whatsapp": 2.0, "phone": 3.0}
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    obs_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE "
        "table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()]
    df = con.execute(f"""
        SELECT p.loan_id, p.month, {', '.join('p.' + c for c in obs_cols)},
               a.channel, a.discount_offer, a.is_exploration, t.p_recovery
        FROM monthly_panel p
        JOIN actions a USING (action_id)
        JOIN oracle_truth t ON t.loan_id = p.loan_id AND t.month = p.month
        ORDER BY p.loan_id, p.month""").df()
    con.close()
    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)
    df["ch_code"] = df["channel"].map(ch_code).astype(float)
    lag_cols = []
    for j in range(1, k + 1):
        for c in obs_cols + ["ch_code"]:
            name = f"{c}_lag{j}"
            df[name] = g[c].shift(j)
            lag_cols.append(name)
    m = df[(df.is_exploration) & (df.channel == "phone")
           & (df.month >= k)].dropna(subset=lag_cols)
    y = np.log(np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)
               / (1 - np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)))
    X = m[obs_cols + ["discount_offer"]].to_numpy(float)
    r = cross_val_score(HistGradientBoostingRegressor(
        max_iter=300, random_state=0), X, np.asarray(y, float),
        cv=3, scoring="r2")
    return float(r.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member",
                    choices=("d05", "d40", "d40_rev2", "v8_q01", "v8_q02"),
                    required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--infer", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--bands", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--code-note", default="")
    ap.add_argument("--state-timing", choices=("entering", "post_outcome_archived"),
                    default="entering",
                    help="account-field timing of the probe clauses; the archived "
                         "reports used post_outcome_archived")
    args = ap.parse_args()
    bands = yaml.safe_load(args.bands.read_text())
    com, mem = bands["common"], bands[args.member]
    if "clauses_from" in mem:
        pilot = mem["pilot"]
        mem = dict(bands[mem["clauses_from"]])
        mem["pilot"] = pilot
    ok = True

    for kind, world in (("train", args.train), ("infer", args.infer)):
        dt = pd.read_csv(world / "decision_table.csv",
                         usecols=["recovered_1m", "recovered_9m"])
        for col in ("recovered_1m", "recovered_9m"):
            b = com[col]
            v = float(dt[col].mean())
            ok &= band(f"{kind}.{col}", round(v, 4),
                       b["min"] <= v <= b["max"],
                       f"(band [{b['min']}, {b['max']}])")

    if mem.get("realism_only"):
        for kind, world in (("train", args.train), ("infer", args.infer)):
            dtp = pd.read_csv(world / "decision_table.csv",
                              usecols=["recovered_1m", "recovered_9m"])
            band(f"pilot_drift.{kind}.rec9m",
                 round(abs(float(dtp["recovered_9m"].mean())
                           - mem["pilot"]["rec9m"]), 4),
                 abs(float(dtp["recovered_9m"].mean())
                     - mem["pilot"]["rec9m"]) <= 0.03,
                 f"(pilot {mem['pilot']['rec9m']}; warn above 0.03)",
                 warn_only=True)
        cfg_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
        for kind, world in (("train", args.train), ("infer", args.infer)):
            (world / "manifest.json").write_text(json.dumps({
                "member": args.member, "kind": kind,
                "expected_seed": EXPECTED_SEEDS[args.member][kind],
                "config": str(args.config), "config_sha256": cfg_sha,
                "code": args.code_note,
                "gated_utc": datetime.now(timezone.utc).isoformat()},
                indent=2) + "\n")
        report = args.infer / f"v72_{args.member}_gate_report.json"
        report.write_text(json.dumps({"pass": bool(ok), "bands": BAND_LOG},
                                     indent=2) + "\n")
        print(f"GATE {args.member}: {'PASS' if ok else 'FAIL'} -> {report}",
              flush=True)
        return 0 if ok else 1

    flat, sep, head = probe(args.infer, args.k, args.state_timing)
    if "prob_separation_min" in mem:
        ok &= band("prob_separation", round(sep, 4),
                   sep >= mem["prob_separation_min"],
                   f"(min {mem['prob_separation_min']})")
    else:
        band("prob_separation", round(sep, 4), True,
             "(report-only for this member)", warn_only=True)
    if "leak_reading_rise_max" in mem:
        ctrl = args.infer.parent / mem["control_world"]
        rise = obs_only_r2(args.infer, args.k) - obs_only_r2(ctrl, args.k)
        ok &= band("leak_reading_rise", round(rise, 4),
                   rise <= mem["leak_reading_rise_max"],
                   f"(max {mem['leak_reading_rise_max']}, control world "
                   f"{mem['control_world']})")
        band("prob_flat", round(flat, 4),
             flat <= mem["prob_flat_report_only"],
             f"(report-only, retired proxy; was max "
             f"{mem['prob_flat_report_only']})", warn_only=True)
    else:
        ok &= band("prob_flat", round(flat, 4), flat <= mem["prob_flat_max"],
                   f"(max {mem['prob_flat_max']})")
    ok &= band("prob_headroom", round(head, 4),
               head >= mem["prob_headroom_min"],
               f"(min {mem['prob_headroom_min']})")

    band("separation_upper_sanity", round(sep, 4),
         sep <= com["separation_warn_max"],
         f"(warn above {com['separation_warn_max']})", warn_only=True)
    band("headroom_upper_sanity", round(head, 4),
         head <= com["headroom_warn_max"],
         f"(warn above {com['headroom_warn_max']})", warn_only=True)
    for name, v in (("separation", sep), ("flat", flat), ("headroom", head)):
        drift = abs(v - mem["pilot"][name])
        band(f"pilot_drift.{name}", round(drift, 4),
             drift <= com["pilot_drift_warn"],
             f"(pilot {mem['pilot'][name]}; warn above "
             f"{com['pilot_drift_warn']})", warn_only=True)

    key_path = args.infer / "answer_key.csv"
    if not key_path.exists():
        ok &= band("answer_key_present", False, False, f"(missing {key_path})")
    else:
        key = pd.read_csv(key_path)
        min_gap = float((key["ceiling"] - key["myopic_bound"]).min())
        ok &= band("key_min_gap", round(min_gap, 9),
                   min_gap >= com["key_min_gap_min"],
                   f"(rows {len(key)}, min {com['key_min_gap_min']})")

    cfg_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    for kind, world in (("train", args.train), ("infer", args.infer)):
        (world / "manifest.json").write_text(json.dumps({
            "member": args.member, "kind": kind,
            "expected_seed": EXPECTED_SEEDS[args.member][kind],
            "config": str(args.config), "config_sha256": cfg_sha,
            "code": args.code_note,
            "gated_utc": datetime.now(timezone.utc).isoformat()},
            indent=2) + "\n")

    report = args.infer / f"v72_{args.member}_gate_report.json"
    report.write_text(json.dumps({"pass": bool(ok), "bands": BAND_LOG,
                                  "probe_state_timing": args.state_timing},
                                 indent=2) + "\n")
    print(f"GATE {args.member}: {'PASS' if ok else 'FAIL'} -> {report}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
