from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FINGERPRINT = Path("evidence/fannie/cohort_fingerprint.json")


def fingerprint(ids: pd.DataFrame) -> dict:
    ids = ids[["loan_id", "source_file"]].astype(str).sort_values(["source_file", "loan_id"])
    canon = "loan_id,source_file\n" + "".join(
        f"{a},{b}\n" for a, b in zip(ids.loan_id, ids.source_file))
    per = ids.groupby("source_file").size()
    return {"n_loans": int(len(ids)), "n_source_files": int(per.size),
            "sha256": hashlib.sha256(canon.encode()).hexdigest(),
            "loans_per_source_file": {k: int(v) for k, v in per.items()}}


def check(ids: pd.DataFrame, ref_path: Path = FINGERPRINT) -> bool:
    ref = json.loads(ref_path.read_text())
    got = fingerprint(ids)
    print(f"loans {got['n_loans']} (expected {ref['n_loans']}), "
          f"quarterly files {got['n_source_files']} (expected {ref['n_source_files']})")
    diff = {k: (got["loans_per_source_file"].get(k, 0), v)
            for k, v in ref["loans_per_source_file"].items()
            if got["loans_per_source_file"].get(k, 0) != v}
    for k, (g, e) in sorted(diff.items()):
        print(f"  {k}: {g} loans, expected {e}")
    ok = got["sha256"] == ref["sha256"]
    print("sha256", got["sha256"], "MATCH" if ok else "MISMATCH")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--fingerprint", type=Path, default=FINGERPRINT)
    args = ap.parse_args()
    return 0 if check(pd.read_csv(args.ids, dtype=str), args.fingerprint) else 1


if __name__ == "__main__":
    raise SystemExit(main())
