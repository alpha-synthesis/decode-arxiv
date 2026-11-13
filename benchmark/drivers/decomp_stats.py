import json
import numpy as np
import pandas as pd

d = pd.read_csv("outputs/decomp_board/per_loan_values.csv")
e0 = pd.read_csv("outputs/e0_board/per_loan_values.csv")
assert list(d["loan_id"]) == list(e0["loan_id"])
assert np.allclose(d["ceiling"], e0["ceiling"])
ceil = float(d["ceiling"].mean())
cols = {
    "lags(comparator rerun)": d["history_sequential"],
    "lags_exactF": d["lags_exactF_sequential"],
    "D1_static": d["posterior_gbm_static_sequential"],
    "D1": d["posterior_gbm_sequential"],
    "D2_longlags": d["longlags_sequential"],
    "E0": e0["bayes_filter"],
}
pct = {k: 100 * float(v.mean()) / ceil for k, v in cols.items()}
for k, v in pct.items():
    print(f"{k:24s} {v:6.2f}% of ceiling")
print(f"{'published GBM-lags':24s}  84.40  <- the rerun above must match")

rng = np.random.default_rng(0)
n = len(d)
idx = rng.integers(0, n, size=(10000, n))
def bd(a, b):
    dd = (np.asarray(a, float) - np.asarray(b, float)) / ceil * 100
    bs = dd[idx].mean(1)
    return dd.mean(), *np.percentile(bs, [2.5, 97.5]), bs
pairs = [("knots->exactF", cols["lags_exactF"], cols["lags(comparator rerun)"]),
         ("inference (D1s - exactF)", cols["D1_static"], cols["lags_exactF"]),
         ("rolling (D1 - D1s)", cols["D1"], cols["D1_static"]),
         ("response (E0 - D1)", cols["E0"], cols["D1"]),
         ("D2 - lags", cols["D2_longlags"], cols["lags(comparator rerun)"])]
draws = {}
for name, a, b in pairs:
    m, lo, hi, bs = bd(a, b)
    draws[name] = bs
    tie = "  TIE" if lo <= 0 <= hi else ""
    print(f"{name:26s} {m:+6.2f}pp  CI95 [{lo:+.2f}, {hi:+.2f}]{tie}")
diff = draws["inference (D1s - exactF)"] - draws["response (E0 - D1)"]
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"{'inference - response':26s} {diff.mean():+6.2f}pp  CI95 [{lo:+.2f}, {hi:+.2f}]"
      f"{'  TIE' if lo <= 0 <= hi else ''}")
json.dump({"pct": pct}, open("outputs/decomp_board/decomp_stats.json","w"), indent=2)
