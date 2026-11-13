from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_LOOP = {
    "flat_myopic": 66.03,
    "flat_sequential": 80.62,
    "history_myopic": 67.73,
    "history_sequential": 82.91,
    "belief_sequential": 83.77,
}

REFERENCE_ARMS = ("ceiling", "myopic_bound", "loan_id")


def paired_bootstrap(a, b, ceiling: float, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    d = (np.asarray(a, float) - np.asarray(b, float)) / ceiling * 100.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta_pp": round(float(d.mean()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p_le_0": float((boot <= 0).mean()),
            "tie_at_95": bool(lo <= 0.0 <= hi)}


PUBLISHED_PROTOCOL = {"n_loans": 4000, "k": 6, "H": 6}


def _check_protocol(path: Path) -> dict | None:
    f = path / "board.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text())
    got = {k: meta.get(k) for k in PUBLISHED_PROTOCOL}
    if any(v is not None for v in got.values()) and got != PUBLISHED_PROTOCOL:
        raise AssertionError(
            f"{path}: protocol {got} != the published protocol "
            f"{PUBLISHED_PROTOCOL} that the OPEN_LOOP reference scalars were "
            f"measured under. Either re-run at the published protocol or drop "
            f"the AC-2 comparison.")
    return meta


def merge_runs(paths):
    base = None
    for p in paths:
        _check_protocol(Path(p))
        df = pd.read_csv(Path(p) / "per_loan_values.csv")
        if base is None:
            base = df
            continue
        if list(df["loan_id"]) != list(base["loan_id"]):
            raise AssertionError(
                f"{p}: cohort differs from the first run — the board would be "
                f"pairing different loans against each other")
        for c in df.columns:
            if c == "loan_id":
                continue
            if c in ("ceiling", "myopic_bound"):
                if not np.allclose(df[c], base[c]):
                    raise AssertionError(f"{p}: {c} differs across runs")
                continue
            if c in base.columns:
                if not np.allclose(df[c], base[c]):
                    raise AssertionError(
                        f"arm {c} appears in two runs with different values — "
                        f"one of them is not reproducible")
                continue
            base[c] = df[c]
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="board output dirs to merge")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = merge_runs(args.runs)
    ceiling = float(df["ceiling"].mean())
    arms = [c for c in df.columns
            if c not in REFERENCE_ARMS and not c.endswith(":frozen")]
    pct = {a: 100.0 * float(df[a].mean()) / ceiling for a in arms}
    pct["ceiling"] = 100.0
    pct["myopic_bound"] = 100.0 * float(df["myopic_bound"].mean()) / ceiling

    ranked = sorted((a for a in arms), key=lambda a: -pct[a])
    board = {"n_loans": int(len(df)), "ceiling_per_loan": round(ceiling, 2),
             "n_boot": args.n_boot,
             "rows_pct_of_ceiling": {a: round(pct[a], 2)
                                     for a in ["ceiling"] + ranked
                                     + ["myopic_bound"]}}

    ac2 = {}
    for a in ranked:
        frozen_col = f"{a}:frozen"
        if frozen_col in df.columns:
            e = {"frozen_pct": round(100.0 * float(df[frozen_col].mean()) / ceiling, 2),
                 "closed_loop_pct": round(pct[a], 2),
                 "source": "paired, both rows from this harness",
                 **paired_bootstrap(df[a], df[frozen_col], ceiling,
                                    args.n_boot, args.seed)}
            if a in OPEN_LOOP:
                e["published_open_loop_pct"] = OPEN_LOOP[a]
                e["frozen_reproduces_published"] = bool(
                    abs(e["frozen_pct"] - OPEN_LOOP[a]) < 0.01)
            ac2[a] = e
        elif a in OPEN_LOOP:
            ac2[a] = {"frozen_pct": OPEN_LOOP[a],
                      "closed_loop_pct": round(pct[a], 2),
                      "delta_pp": round(pct[a] - OPEN_LOOP[a], 2),
                      "source": "published scalar — POINT ESTIMATE, no interval"}
    board["AC2_closed_vs_own_open_loop"] = ac2
    board["AC2_pass"] = bool(any(
        v["delta_pp"] >= 1.0 and not v.get("tie_at_95", False)
        for v in ac2.values()))

    adj = {}
    for hi, lo in zip(ranked, ranked[1:]):
        adj[f"{hi} - {lo}"] = paired_bootstrap(df[hi], df[lo], ceiling,
                                               args.n_boot, args.seed)
    board["adjacent_pairs"] = adj

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "board_final.json").write_text(json.dumps(board, indent=2) + "\n")
    df.to_csv(args.out / "per_loan_merged.csv", index=False)

    w = max(len(a) for a in board["rows_pct_of_ceiling"])
    print(f"\n{'arm'.ljust(w)}   % of ceiling   closed vs frozen (paired)")
    for a, v in board["rows_pct_of_ceiling"].items():
        e = ac2.get(a)
        note = ""
        if e is not None:
            note = "%+.2f pp" % e["delta_pp"]
            if "ci95" in e:
                note += ("  CI95 [%+.2f, %+.2f]%s"
                         % (e["ci95"][0], e["ci95"][1],
                            "  TIE" if e["tie_at_95"] else ""))
            else:
                note += "  (point estimate only)"
        print(f"{a.ljust(w)}   {v:12.2f}   {note}")
    print("\npaired bootstrap, adjacent arms "
          f"({args.n_boot} resamples, same {len(df)} loans):")
    for k, v in adj.items():
        tie = "  TIE at 95%" if v["tie_at_95"] else ""
        print(f"  {k}: {v['delta_pp']:+.2f}pp  CI95 "
              f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  "
              f"P(<=0)={v['p_le_0']:.4f}{tie}")
    print(f"\nAC-2 (some arm gains >= 1.0pp from closing the loop): "
          f"{'PASS' if board['AC2_pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def latent_drift(world: Path, config: Path, plans: dict, k: int, H: int,
                 ids: list) -> dict:
    import sys

    from decode_sim.config import load_config

    from benchmark.closed_loop_v7 import (
        CHANNELS, ObservationContract, load_cohort, world_seed_of)

    sim_cfg = load_config(str(config))
    contract = ObservationContract(sim_cfg, world_seed=world_seed_of(world))
    book, _, obs_cols, _, logged, ztrue, obs_logged = load_cohort(
        world, sim_cfg, k, len(ids))
    if list(book["loan_id"]) != list(ids):
        raise AssertionError("drift diagnostic: cohort mismatch")
    zc = [c for c in book.columns if c.startswith("zman_")]
    z0 = book[zc].to_numpy(float)

    zt = ztrue.set_index(["loan_id", "month"])
    have = [i for i in ids if all((i, k + t) in zt.index for t in range(H))]
    ref = np.concatenate([
        contract.sensor(zt.loc[[(i, k + t) for i in have]].to_numpy(float))
        for t in range(H)])
    mu, sd = ref.mean(0), ref.std(0)

    out = {"logged_sensor_mean": [round(float(x), 4) for x in mu],
           "logged_sensor_std": [round(float(x), 4) for x in sd],
           "n_reference_loans": len(have), "n_cohort": len(ids)}
    for arm, plan_csv in plans.items():
        p = pd.read_csv(plan_csv)
        wide = p.pivot(index="loan_id", columns="month",
                       values="channel").loc[ids]
        z = z0.copy()
        devs, contacts = [], 0.0
        for t in range(H):
            ci = np.array([CHANNELS.index(c) for c in wide[t]])
            contacts += float((ci != 0).mean())
            z = contract.roll_z(z, ci)
            if t < H - 1:
                devs.append(np.abs((contract.sensor(z) - mu) / sd).mean())
        out[arm] = {"months_covered": f"k+1..k+{H - 1}",
                    "mean_abs_z_dev": round(float(np.mean(devs)), 4),
                    "last_observed_month_dev": round(float(devs[-1]), 4),
                    "contacts_per_loan": round(contacts, 3)}
    return out
