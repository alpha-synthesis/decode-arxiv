import argparse
import json
import subprocess
import sys
from pathlib import Path
from decode_sim.config import load_config
from decode_sim.generator import generate

OUT = Path("worlds")
_ap = argparse.ArgumentParser()
_ap.add_argument("--state-timing", choices=("entering", "post_outcome_archived"),
                 default="entering")
_ap.add_argument("--out", type=Path, default=Path("outputs/geometry_sweep_results.json"))
_ap.add_argument("--keep-worlds", action="store_true")
ARGS = _ap.parse_args()
RESULTS = ARGS.out
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
prev = results.get("_state_timing", "post_outcome_archived" if results else ARGS.state_timing)
if prev != ARGS.state_timing:
    raise SystemExit(f"{RESULTS} holds {prev} results; refusing to resume with "
                     f"{ARGS.state_timing} (use another --out)")
results["_state_timing"] = ARGS.state_timing
for gseed in (25001, 25002, 25003, 25004):
    for delta in (0.0, 0.5, 4.0):
        tag = f"g{gseed}_d{str(delta).replace('.', 'p')}"
        if tag in results:
            print(tag, "already probed - skipping", flush=True)
            continue
        out = OUT / f"geom_{tag}"
        cfg = load_config("configs/adversarial_v7_scenario_frozen.yaml")
        cfg["seed"] = 4500
        cfg["population"]["borrowers"] = 100_000
        cfg["panel"]["months"] = 12
        cfg["oracle"]["manifold"]["seed"] = gseed
        if delta:
            cfg["oracle"]["manifold"]["bilinear_strength"] = delta
        if not (out / "sim.duckdb").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"== generating {tag} ==", flush=True)
            generate(cfg, out, batch_size=10000)
        r = subprocess.run([sys.executable, "gates/v7_pilot_gap.py",
                            "--world", str(out), "--k", "6",
                            "--state-timing", ARGS.state_timing],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-500:], flush=True)
            raise SystemExit(f"probe failed at {tag}")
        flat = float(r.stdout.split("flat (row + current keyhole)")[1]
                     .split("R2 = ")[1].split()[0])
        sep = float(r.stdout.split("SEPARATION GAP (history - flat): ")[1]
                    .split()[0])
        head = float(r.stdout.split("HEADROOM LEFT (oracle - history): ")[1]
                     .split()[0])
        results[tag] = {"flat": flat, "separation": sep, "headroom": head}
        RESULTS.write_text(json.dumps(results, indent=2) + "\n")
        print(f"== {tag}: flat {flat:.4f} sep {sep:+.4f} head {head:+.4f} ==",
              flush=True)
        if not ARGS.keep_worlds:
            import shutil
            shutil.rmtree(out)
print("== GEOMETRY_SWEEP_DONE ==", flush=True)
