from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.exact_filter import e0_jensen_gap
from decode_sim.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Self-report inflation of certainty-equivalent chaining on Base: the filter "
                    "planner's chained valuation of its executed plan against joint integration "
                    "of the same plan under the same belief, with a loan-bootstrap interval.")
    ap.add_argument("--world", type=Path, default=Path("worlds/adversarial_v7_infer"))
    ap.add_argument("--config", default="configs/adversarial_v7_scenario_frozen.yaml")
    ap.add_argument("--plan", type=Path, default=Path("evidence/runs/e0_board/plan_bayes_filter.csv"))
    ap.add_argument("--reference", type=Path, default=Path("evidence/base_experiments/e0_jensen.json"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--loans", type=int, default=4000)
    ap.add_argument("--n-draws", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=20261104)
    ap.add_argument("--out", type=Path, default=Path("outputs/self_report.json"))
    a = ap.parse_args()

    r = e0_jensen_gap(a.world, load_config(a.config), a.plan, a.k, a.horizon, a.loans,
                      n_draws=a.n_draws, seed=a.seed, per_loan=True)
    joint, ce, batch = r.pop("joint"), r.pop("ce"), r.pop("batch")
    ref = json.loads(a.reference.read_text())
    reproduces = bool(abs(r["joint_value_mean"] - ref["joint_value_mean"]) < 1e-6
                      and abs(r["ce_chain_value_mean"] - ref["ce_chain_value_mean"]) < 1e-6)

    rng = np.random.default_rng(a.boot_seed)
    n = len(joint)
    idx = rng.integers(0, n, size=(a.n_boot, n))
    ratio = ce[idx].mean(1) / joint[idx].mean(1) - 1.0
    batch_means = batch.mean(1)
    out = {
        "n": r["n"],
        "n_draws": r["n_draws"],
        "joint_value_mean": r["joint_value_mean"],
        "ce_chain_value_mean": r["ce_chain_value_mean"],
        "reproduces_reference_means": reproduces,
        "inflation_pct": float(100 * (ce.mean() / joint.mean() - 1.0)),
        "inflation_ci95_loan_bootstrap_pct": [float(100 * np.percentile(ratio, 2.5)),
                                              float(100 * np.percentile(ratio, 97.5))],
        "n_boot": a.n_boot,
        "boot_seed": a.boot_seed,
        "mc_se_joint_mean_batch_means": float(batch_means.std(ddof=1) / np.sqrt(len(batch_means))),
        "share_loans_ce_above_joint": float((ce > joint).mean()),
        "plan": a.plan.name,
        "conditioning": "one world and one geometry; the filter planner's executed plan; Gaussian "
                        "filter posterior with true-trait base logits; loans resampled with the "
                        "posterior draws held fixed",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0 if reproduces else 1


if __name__ == "__main__":
    raise SystemExit(main())
