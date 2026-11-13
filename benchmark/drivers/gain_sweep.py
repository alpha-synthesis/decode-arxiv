import argparse
import json
import subprocess
import sys
from pathlib import Path

from decode_sim.config import load_config
from decode_sim.export import export_csv
from decode_sim.generator import generate

GAINS = [0.0, 0.1, 0.2, 0.4, 0.7]
BASE = "configs/adversarial_v7_scenario_frozen.yaml"
OUT_ROOT = Path("worlds")
_ap = argparse.ArgumentParser()
_ap.add_argument("--state-timing", choices=("entering", "post_outcome_archived"),
                 default="entering")
_ap.add_argument("--out", type=Path, default=Path("outputs/gain_sweep_results.json"))
ARGS = _ap.parse_args()
RESULTS = ARGS.out

results = {"_state_timing": ARGS.state_timing}
for gain in GAINS:
    tag = f"dial_g{str(gain).replace('.', 'p')}"
    out = OUT_ROOT / f"pilot_{tag}"
    cfg = load_config(BASE)
    cfg["oracle"]["manifold"]["obs_nonlin_gain"] = float(gain)
    cfg["panel"]["months"] = 12
    if not (out / "sim.duckdb").exists():
        print(f"== generating {tag} ==", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        generate(cfg, out, batch_size=10000)
    else:
        print(f"== {tag} present, skipping generation ==", flush=True)
    print(f"== probe {tag} ==", flush=True)
    r = subprocess.run(
        [sys.executable, "gates/v7_pilot_gap.py", "--world", str(out),
         "--k", "6", "--state-timing", ARGS.state_timing],
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        raise SystemExit(f"probe failed at gain {gain}")
    sep = float(r.stdout.split("SEPARATION GAP (history - flat): ")[1].split()[0])
    head = float(r.stdout.split("HEADROOM LEFT (oracle - history): ")[1].split()[0])
    results[str(gain)] = {"separation": sep, "headroom": head}
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(f"== gain {gain}: separation {sep:+.4f}  headroom {head:+.4f} ==",
          flush=True)
print("== GAIN_SWEEP_DONE ==", flush=True)
