from __future__ import annotations

import itertools

import numpy as np


def man_drift_from(w, z: np.ndarray, win: int) -> np.ndarray:
    D = np.empty((win, z.shape[0], len(w.channels)))
    zt = z
    for kk in range(win):
        D[kk] = w.man["coef"] * (zt @ w.man["W"].T)
        zt = zt @ w.man["A"].T
    return D


def path_correction(w, path) -> np.ndarray:
    corr = np.empty(len(path))
    M = np.zeros(w.man["n_lat"])
    for kk, ci in enumerate(path):
        corr[kk] = w.man["coef"] * float(w.man["W"][ci] @ M)
        M = w.man["A"] @ M + w.man["B"] * w.man["w_int"][ci]
    return corr


def _bilinear_projections(w, z: np.ndarray, path, win: int) -> np.ndarray:
    d = w.man["n_lat"]
    Mt, mt = np.eye(d), np.zeros(d)
    proj = np.empty((win, z.shape[0]))
    for u, ci in enumerate(path):
        v = w.man["W"][ci]
        proj[u] = w.man["coef"] * (z @ (Mt.T @ v) + float(v @ mt))
        wv = w.man["w_int"][ci]
        T = w.man["A"] @ w.man["rot"][wv]
        mt = T @ mt + w.man["B"] * wv
        Mt = T @ Mt
    return proj


def window_decision(w, t: int, win: int, F: np.ndarray,
                    z: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from decode_sim.trajectory_key import _sigmoid

    n = len(F)
    C = len(w.channels)
    applies = np.array([c != "none" for c in w.channels])
    fat_add = np.array([w.fat_add[c] for c in w.channels])
    costs = np.array([w.costs[c] for c in w.channels])
    decay = w.decay
    bil = w.man is not None and w.man.get("rot") is not None
    drift = man_drift_from(w, z, win) if (w.man is not None and not bil) \
        else None

    best_V = np.full(n, -np.inf)
    best_ci = np.zeros(n, dtype=np.int64)
    best_di = np.zeros(n, dtype=np.int64)
    for path in itertools.product(range(C), repeat=win):
        g = np.empty(win)
        gk = 0.0
        for kk, ci in enumerate(path):
            g[kk] = gk
            gk = decay * gk + fat_add[ci]
        proj = _bilinear_projections(w, z, path, win) if bil else None
        corr = path_correction(w, path) if (w.man is not None and not bil) \
            else None
        V = np.zeros(n)
        first_di = np.zeros(n, dtype=np.int64)
        for kk in range(win - 1, -1, -1):
            ci = path[kk]
            pen = w.beta_i * (decay ** kk * F + g[kk]) if applies[ci] else np.zeros(n)
            if bil:
                pen = pen - proj[kk]
            elif w.man is not None:
                pen = pen - (drift[kk][:, ci] + corr[kk])
            p = _sigmoid(w.logits0[t + kk, :, ci, :] - pen[:, None])
            tot = -costs[ci] + p * w.cash[t + kk] + ((1.0 - p)
                  * ((1.0 - w.q[t + kk]) * V)[:, None])
            j = np.argmax(tot, axis=1)
            V = tot[np.arange(n), j]
            if kk == 0:
                first_di = j
        upd = V > best_V
        if upd.any():
            best_V[upd] = V[upd]
            best_ci[upd] = path[0]
            best_di[upd] = first_di[upd]
    return best_ci, best_di, best_V


def entering_state(w, book):
    F0_i = (book["entering_fatigue"].to_numpy(float)
            if "entering_fatigue" in book.columns else np.zeros(len(book)))
    z0 = None
    if w.man is not None:
        zc = [c for c in book.columns if c.startswith("zman_")]
        z0 = book[zc].to_numpy(float)
    return F0_i, z0
