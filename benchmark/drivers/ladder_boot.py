from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EV = Path("evidence/base_experiments")


def contrasts(df: pd.DataFrame, pairs, n_boot: int = 10_000, seed: int = 0) -> dict:
    ceil = df["ceiling"].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(df), size=(n_boot, len(df)))
    c_bs = ceil[idx].mean(1)
    out = {}
    for name, a, b in pairs:
        d = df[a].to_numpy(float) - df[b].to_numpy(float)
        bs = 100 * d[idx].mean(1) / c_bs
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = {"delta_pp": 100 * d.mean() / ceil.mean(), "ci95": [float(lo), float(hi)]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    d = pd.read_csv(EV / "decomp_per_loan.csv")
    s = pd.read_csv(EV / "c7a_per_loan.csv")
    res = contrasts(d, [
        ("exact fatigue - GBM-lags", "lags_exactF_sequential", "history_sequential"),
        ("exact inference - exact fatigue", "posterior_gbm_static_sequential", "lags_exactF_sequential"),
        ("rolled belief - exact inference", "posterior_gbm_sequential", "posterior_gbm_static_sequential"),
        ("exact inference - GBM-lags", "posterior_gbm_static_sequential", "history_sequential"),
        ("rolled belief - GBM-lags", "posterior_gbm_sequential", "history_sequential"),
        ("11-lag window - GBM-lags", "longlags_sequential", "history_sequential"),
    ])
    res.update(contrasts(s, [("structured - exact fatigue", "structured_sequential", "lags_exactF_sequential")]))
    for k, v in res.items():
        print(f"{k:34s} {v['delta_pp']:+6.2f}  [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
