import json
import numpy as np
import pandas as pd

base = "outputs"
e1 = pd.read_csv(f"{base}/e1_board/per_loan_values.csv")
d = pd.read_csv(f"{base}/decomp_board/per_loan_values.csv")
e0 = pd.read_csv(f"{base}/e0_board/per_loan_values.csv")
for f in (d, e0):
    assert list(f["loan_id"]) == list(e1["loan_id"])
    assert np.allclose(f["ceiling"], e1["ceiling"])
ceil = float(e1["ceiling"].mean())
cols = {
    "allfitted (lags-exactF)": d["lags_exactF_sequential"],
    "none_exact_static": e1["hybrid_none_exact_static_sequential"],
    "none_exact": e1["hybrid_none_exact_sequential"],
    "contact_exact_static": e1["hybrid_contact_exact_static_sequential"],
    "contact_exact": e1["hybrid_contact_exact_sequential"],
    "E0 (allexact)": e0["bayes_filter"],
}
for k, v in cols.items():
    print(f"{k:26s} {100*float(v.mean())/ceil:6.2f}% of ceiling")

rng = np.random.default_rng(0)
n = len(e1)
idx = rng.integers(0, n, size=(10000, n))
def bd(a, b):
    dd = (np.asarray(a, float) - np.asarray(b, float)) / ceil * 100
    bs = dd[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dd.mean(), lo, hi, bs
pairs = [
    ("organic level (none_static - allfitted)", cols["none_exact_static"], cols["allfitted (lags-exactF)"]),
    ("organic + propagation (none - allfitted)", cols["none_exact"], cols["allfitted (lags-exactF)"]),
    ("contact LEVEL (contact_static - allfitted)", cols["contact_exact_static"], cols["allfitted (lags-exactF)"]),
    ("contact PROPAGATION (contact - contact_static)", cols["contact_exact"], cols["contact_exact_static"]),
    ("residual to E0 (E0 - contact_exact)", cols["E0 (allexact)"], cols["contact_exact"]),
]
draws = {}
for name, a, b in pairs:
    m, lo, hi, bs = bd(a, b)
    draws[name] = bs
    tie = "  TIE" if lo <= 0 <= hi else ""
    print(f"{name:48s} {m:+6.2f}pp  CI95 [{lo:+.2f}, {hi:+.2f}]{tie}")
diff = draws["contact LEVEL (contact_static - allfitted)"] - draws["contact PROPAGATION (contact - contact_static)"]
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"{'level - propagation (same draws)':48s} {diff.mean():+6.2f}pp  CI95 [{lo:+.2f}, {hi:+.2f}]"
      f"{'  TIE' if lo <= 0 <= hi else ''}")
json.dump({k: 100*float(v.mean())/ceil for k, v in cols.items()},
          open(f"{base}/e1_board/e1_stats.json", "w"), indent=2)
