from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decode_sim.bridge import decision_table
from decode_sim.config import load_config
from decode_sim.export import export_csv
from decode_sim.generator import generate

BATCH = 10000
MEMBERS = {
    "BASE": ("adversarial_v7_scenario_frozen.yaml", [
        ("adversarial_v7_check", 16001, 100_000, 24),
        ("adversarial_v7_train", 16001, 1_000_000, 24),
        ("adversarial_v7_infer", 16002, 100_000, 12)]),
    "MEM-0.5": ("adversarial_v72_d05_scenario_frozen.yaml", [
        ("adversarial_v72_d05_train", 17001, 1_000_000, 24),
        ("adversarial_v72_d05_infer", 17002, 100_000, 12)]),
    "MEM-4": ("adversarial_v72_d40_rev2_scenario_frozen.yaml", [
        ("adversarial_v72_d40_rev2_train", 18011, 1_000_000, 24),
        ("adversarial_v72_d40_rev2_infer", 18012, 100_000, 12)]),
    "NOISE-0.1": ("adversarial_v8_q01_scenario_frozen.yaml", [
        ("adversarial_v8_q01_train", 19001, 1_000_000, 24),
        ("adversarial_v8_q01_infer", 19002, 100_000, 12)]),
    "NOISE-0.2": ("adversarial_v8_q02_scenario_frozen.yaml", [
        ("adversarial_v8_q02_train", 20001, 1_000_000, 24),
        ("adversarial_v8_q02_infer", 20002, 100_000, 12)]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("worlds"))
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument("--members", default=",".join(MEMBERS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="population multiplier for smoke tests (1.0 = as released)")
    ap.add_argument("--skip-decision-table", action="store_true")
    args = ap.parse_args()
    for member in args.members.split(","):
        cfg_file, worlds = MEMBERS[member.strip()]
        cfg_path = args.configs / cfg_file
        for name, seed, borrowers, months in worlds:
            out = args.out / name
            if (out / "sim.duckdb").exists():
                print(name, "present, skipping", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            cfg = load_config(str(cfg_path))
            cfg["seed"] = seed
            cfg["population"]["borrowers"] = max(BATCH, int(round(borrowers * args.scale)))
            cfg["panel"]["months"] = months
            print("generating", name, "seed", seed, "borrowers",
                  cfg["population"]["borrowers"], flush=True)
            generate(cfg, out, batch_size=BATCH)
            export_csv(out / "sim.duckdb", out, mode="both")
            if not args.skip_decision_table:
                decision_table.build(out)
            (out / "manifest.json").write_text(json.dumps({
                "member": member, "world": name, "seed": seed,
                "borrowers": cfg["population"]["borrowers"], "months": months,
                "config": str(cfg_file),
                "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                "generation_batch_size": BATCH}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
