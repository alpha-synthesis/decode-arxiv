import json
import numpy as np
import pandas as pd

m = pd.read_csv("outputs/cl_v7_final/per_loan_merged.csv")
e = pd.read_csv("outputs/e0_board/per_loan_values.csv")
assert list(m["loan_id"]) == list(e["loan_id"]), "cohort mismatch"
assert np.allclose(m["ceiling"], e["ceiling"]), "answer key mismatch"
ceil = float(m["ceiling"].mean())
rng = np.random.default_rng(0)

def boot(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) / ceil * 100
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, float((bs <= 0).mean())

rows = {}
bf = e["bayes_filter"].to_numpy()
print(f"bayes_filter          {100*bf.mean()/ceil:6.2f}% of ceiling")
print(f"bayes_filter:frozen   {100*e['bayes_filter:frozen'].mean()/ceil:6.2f}%")
for name, col in [("history_sequential", m["history_sequential"]),
                  ("belief_sequential (CE-Seq)", m["belief_sequential"]),
                  ("bayes_filter:frozen", e["bayes_filter:frozen"])]:
    d, lo, hi, p = boot(bf, col)
    tie = "  TIE" if lo <= 0 <= hi else ""
    print(f"E0 - {name:24s} {d:+6.2f}pp  CI95 [{lo:+.2f}, {hi:+.2f}]  P(<=0)={p:.5f}{tie}")
    rows[name] = {"delta_pp": round(float(d),3), "ci95":[round(float(lo),3),round(float(hi),3)], "p_le_0": p}
print("reported comparison: E0:frozen vs static frozen-z reference 85.97 (scalar):",
      f"{100*e['bayes_filter:frozen'].mean()/ceil - 85.97:+.2f}pp")
json.dump({"pct": {"bayes_filter": 100*float(bf.mean())/ceil,
                   "bayes_filter_frozen": 100*float(e['bayes_filter:frozen'].mean())/ceil},
           "boot": rows},
          open("outputs/e0_board/e0_stats.json","w"), indent=2)
