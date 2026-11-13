import sys
import numpy as np

import duckdb
from decode_sim.config import load_config
from benchmark.closed_loop_arms import load_train_rows
from benchmark.exact_filter import manifold_params
from benchmark.closed_loop_v7 import CHANNELS

cfg = load_config("configs/adversarial_v7_scenario_frozen.yaml")
man = manifold_params(cfg)
db = "worlds/adversarial_v7_train/sim.duckdb"
obs_cols = ["obs_0", "obs_1"]
k = 6
tdf, lag_cols = load_train_rows(db, k, obs_cols)
tr = tdf[tdf.is_exploration & (tdf.month >= k)].dropna(subset=lag_cols)
tr = tr.sample(frac=1.0, random_state=0).reset_index(drop=True)

con = duckdb.connect(db, read_only=True)
zc = sorted([r[0] for r in con.execute(
    "SELECT column_name FROM information_schema.columns WHERE table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()],
    key=lambda c: int(c.split('_')[1]))
zt = con.execute(f"SELECT loan_id, month, {', '.join(zc)} FROM oracle_truth").df()
con.close()
tr = tr.merge(zt, on=["loan_id", "month"], how="left", validate="1:1")
Wm = np.stack([man["W"][c] for c in CHANNELS])
Wrow = Wm[[CHANNELS.index(c) for c in tr["channel"]]]
score = man["coef"] * (tr[zc].to_numpy(float) * Wrow).sum(1)

def lag_design(L):
    cols = []
    per = len(obs_cols) + 1
    for j in range(L.shape[1] // per):
        b = L[:, j*per:(j+1)*per]
        cols.append(b[:, :len(obs_cols)])
        code = b[:, len(obs_cols)]
        for cv in (1.0, 2.0, 3.0):
            cols.append((code == cv).astype(float)[:, None])
    return np.concatenate(cols, axis=1)

X = lag_design(tr[lag_cols].to_numpy(float))
Xc = np.concatenate([X, tr[obs_cols].to_numpy(float)], axis=1)
n = len(tr); n_te = n // 5
def ols_r2(X_, y_):
    Xd = np.concatenate([np.ones((len(X_),1)), X_], 1)
    b = np.linalg.lstsq(Xd[n_te:], y_[n_te:], rcond=None)[0]
    pred = Xd[:n_te] @ b
    ss = 1 - ((y_[:n_te]-pred)**2).sum()/((y_[:n_te]-y_[:n_te].mean())**2).sum()
    return ss, b
r2_pooled, _ = ols_r2(Xc, score)
print(f"rows {n} | pooled linear model across channels: R2(score) = {r2_pooled:.4f}")
ch = tr["channel"].to_numpy()
sds = []
r2w, wtot = 0.0, 0
for c in sorted(set(ch)):
    m_ = ch == c
    Xc_, y_ = Xc[m_], score[m_]
    nte = max(len(y_) // 5, 200)
    Xd = np.concatenate([np.ones((len(Xc_), 1)), Xc_], 1)
    b = np.linalg.lstsq(Xd[nte:], y_[nte:], rcond=None)[0]
    pred = Xd[:nte] @ b
    r2 = 1 - ((y_[:nte] - pred) ** 2).sum() / ((y_[:nte] - y_[:nte].mean()) ** 2).sum()
    sds.append(float(np.std(Xd @ b)))
    print(f"  channel {c:9s}: n {m_.sum():6d}  R2(score) {r2:.4f}  "
          f"linear-term sd {sds[-1]:.3f}")
    r2w += r2 * m_.sum(); wtot += m_.sum()
print(f"separate linear model per channel: row-weighted mean R2(score) = {r2w / wtot:.4f}")
