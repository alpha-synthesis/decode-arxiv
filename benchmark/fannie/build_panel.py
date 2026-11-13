from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark.fannie.cohort_fingerprint import check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--columns", type=Path, default=Path("evidence/fannie/panel_columns.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cols = json.loads(args.columns.read_text())
    assert cols[0] == "source_file"
    names = cols[1:]
    ids = pd.read_csv(args.ids, dtype=str)
    if not check(ids):
        raise SystemExit("the --ids list is not the analysed cohort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for src, grp in ids.groupby("source_file"):
        want = set(grp["loan_id"])
        path = args.raw / src
        for chunk in pd.read_csv(path, sep="|", header=None, dtype=str,
                                 chunksize=500_000, low_memory=False):
            if chunk.shape[1] != len(names):
                raise ValueError(f"{src}: {chunk.shape[1]} fields, layout has {len(names)}")
            chunk.columns = names
            keep = chunk[chunk["loan_id"].isin(want)]
            if len(keep):
                keep.insert(0, "source_file", src)
                keep.to_csv(args.out, mode="w" if first else "a", header=first, index=False)
                first = False
        print(src, "done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
