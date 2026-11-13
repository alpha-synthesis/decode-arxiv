from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from benchmark.closed_loop_v7 import CHANNELS, D_STEPS, Solver

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(32)
_GH_W = _GH_W / np.sqrt(np.pi)


def posterior_pay_prob(base_logit: np.ndarray, mean: np.ndarray,
                       sd: np.ndarray) -> np.ndarray:
    base_logit = np.asarray(base_logit, float)
    mean = np.asarray(mean, float)
    sd = np.asarray(sd, float)
    if mean.ndim != 1 or sd.shape != mean.shape:
        raise ValueError(f"mean/sd must be (n,); got {mean.shape}/{sd.shape}")
    z = mean[:, None] + np.sqrt(2.0) * sd[:, None] * _GH_X
    if base_logit.ndim == 1:
        if base_logit.shape != mean.shape:
            raise ValueError(f"base {base_logit.shape} vs mean {mean.shape}")
        return (_GH_W * _sigmoid(base_logit[:, None] + z)).sum(-1)
    if base_logit.ndim == 2 and base_logit.shape[0] == mean.shape[0]:
        return (_GH_W * _sigmoid(base_logit[:, :, None]
                                 + z[:, None, :])).sum(-1)
    raise ValueError(f"unsupported base shape {base_logit.shape} for n="
                     f"{mean.shape[0]}")


def _pay_prob_shape_check(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n, D in ((5, 21), (21, 21), (40, 1), (7, 3)):
        base = rng.normal(-2, 1.5, (n, D))
        mean = rng.normal(0, 1, n)
        sd = rng.uniform(0.1, 2.0, n)
        got = posterior_pay_prob(base, mean, sd)
        assert got.shape == (n, D), got.shape
        for i in range(n):
            for j in range(D):
                ref = float((_GH_W * _sigmoid(
                    base[i, j] + mean[i] + np.sqrt(2) * sd[i] * _GH_X)).sum())
                worst = max(worst, abs(got[i, j] - ref))
        flat = posterior_pay_prob(base[:, 0], mean, sd)
        assert flat.shape == (n,)
        worst = max(worst, float(np.abs(flat - got[:, 0]).max()))
    return {"worst_abs_diff": worst, "pass": bool(worst < 1e-12)}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class GaussianBelief:

    def __init__(self, A, B, C, w_int, pop_std, obs_noise, n: int):
        self.A, self.B, self.C = A, B, C
        self.w_int = w_int
        d = A.shape[0]
        self.d = d
        self.inv_prior = np.eye(d) / (pop_std ** 2)
        self.obs_var = float(obs_noise) ** 2
        self.Lam = np.zeros((d, d))
        self.eta = np.zeros((n, d))
        self.m = np.zeros((n, d))
        self.s = 0
        self._An = {0: np.eye(d)}

    def Apow(self, s: int) -> np.ndarray:
        if s not in self._An:
            self._An[s] = self.A @ self.Apow(s - 1)
        return self._An[s]

    def observe(self, obs: np.ndarray) -> None:
        H = self.C @ self.Apow(self.s)
        y = obs - self.m @ self.C.T
        self.Lam += H.T @ H / self.obs_var
        self.eta += y @ H / self.obs_var

    def act(self, ci: np.ndarray) -> None:
        self.m = self.m @ self.A.T + np.outer(self.w_int[np.asarray(ci)],
                                              self.B)
        self.s += 1

    def posterior_z0(self):
        Sig0 = np.linalg.inv(self.Lam + self.inv_prior)
        mu0 = self.eta @ Sig0.T
        return mu0, Sig0

    def score_moments(self, Wv: np.ndarray, coef: float, u: int,
                      prefix_drift: np.ndarray, post=None):
        mu0, Sig0 = self.posterior_z0() if post is None else post
        Au = self.Apow(self.s + u)
        drift = self.m @ self.Apow(u).T + prefix_drift
        v = coef * (Au.T @ Wv)
        mean = mu0 @ v + coef * (drift @ Wv)
        sd = float(np.sqrt(max(v @ Sig0 @ v, 0.0)))
        return mean, np.full(len(mean), sd)


def oracle_terms(w) -> dict:
    return {
        "logits0": w.logits0,
        "beta_i": w.beta_i, "decay": w.decay,
        "fat_add": np.array([w.fat_add[c] for c in CHANNELS]),
        "costs": np.array([w.costs[c] for c in CHANNELS]),
        "cash": w.cash, "q": w.q,
        "d_grid": w.d_grid,
    }


class BayesFilterSolver(Solver):

    name = "bayes_filter"

    def __init__(self, terms: dict, man: dict, k: int, H: int):
        self.t_ = terms
        self.man = man
        self.k, self.H = k, H
        self.Wm = np.stack([man["W"][c] for c in CHANNELS])
        self.coef = float(man["coef"])

    def reset(self, book, prehistory) -> None:
        ids = list(book["loan_id"])
        pos = {l: i for i, l in enumerate(ids)}
        n, k = len(ids), self.k
        obs_cols = sorted((c for c in prehistory.columns
                           if c.startswith("obs_")),
                          key=lambda c: int(c.split("_")[1]))
        pre_obs = np.zeros((n, k, len(obs_cols)))
        pre_ci = np.zeros((n, k), dtype=np.int64)
        seen = np.zeros((n, k), dtype=bool)
        for _, r in prehistory.iterrows():
            i, mo = pos[r["loan_id"]], int(r["month"])
            if 0 <= mo < k:
                pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
                pre_ci[i, mo] = CHANNELS.index(r["channel"])
                seen[i, mo] = True
        if not seen.all():
            missing = int((~seen).sum())
            raise AssertionError(
                f"bayes_filter: {missing} loan-months missing from the "
                f"pre-history — Lambda would no longer be cohort-common and "
                f"the cheap covariance path below would be silently wrong")
        self.bel = GaussianBelief(self.man["A"], self.man["B"], self.man["C"],
                                  self.man["w_int"], self.man["pop_std"],
                                  self.man["obs_noise"], n)
        for s in range(k):
            self.bel.observe(pre_obs[:, s, :])
            self.bel.act(pre_ci[:, s])
        self.n = n

    def decide(self, view):
        t = self.t_
        n, win = self.n, view.months_remaining
        C = len(CHANNELS)
        if not (view.frozen and view.t > 0):
            self.bel.observe(view.obs)
        post = self.bel.posterior_z0()

        def prefix_drift(prefix):
            d = np.zeros(self.bel.d)
            for ci in prefix:
                d = self.man["A"] @ d + self.man["B"] * self.man["w_int"][ci]
            return d

        F0 = view.state["contact_fatigue"].to_numpy(float)
        applies = np.array([c != "none" for c in CHANNELS])
        p_cache: dict[tuple, np.ndarray] = {}

        def p_cells(prefix, ci):
            key = (prefix, ci)
            if key in p_cache:
                return p_cache[key]
            u = len(prefix)
            F = F0.copy()
            for cj in prefix:
                F = t["decay"] * F + t["fat_add"][cj]
            pen = t["beta_i"] * F if applies[ci] else np.zeros(n)
            mean, sd = self.bel.score_moments(self.Wm[ci], self.coef, u,
                                              prefix_drift(prefix), post=post)
            base = t["logits0"][view.t + u, :, ci, :] - pen[:, None]
            out = posterior_pay_prob(base, mean, sd)
            p_cache[key] = out
            return out

        best_V = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        for path in itertools.product(range(C), repeat=win):
            V = np.zeros(n)
            first_di = np.zeros(n, dtype=np.int64)
            for u in range(win - 1, -1, -1):
                p = p_cells(tuple(path[:u]), path[u])
                tot = (-t["costs"][path[u]] + p * t["cash"][view.t + u]
                       + ((1.0 - p) * ((1.0 - t["q"][view.t + u]) * V)[:, None]))
                j = np.argmax(tot, axis=1)
                V = tot[np.arange(n), j]
                if u == 0:
                    first_di = j
            upd = V > best_V
            if upd.any():
                best_V[upd] = V[upd]
                ci_out[upd] = path[0]
                di_out[upd] = first_di[upd]

        self.bel.act(ci_out)
        return ci_out, di_out


def manifold_params(sim_cfg: dict) -> dict:
    from decode_sim import oracle_response as orsp

    man = sim_cfg["oracle"]["manifold"]
    if float(man.get("obs_nonlin_gain", 0.0)) != 0.0:
        raise ValueError(
            "the closed-form filter is exact only for the LINEAR sensor; "
            "obs_nonlin_gain != 0 needs a numerical filter, not this one")
    n_lat, m_obs, chans, A, B, C, W, rot = orsp._manifold(man)
    if rot is not None:
        raise ValueError(
            "manifold_params: this world has a BILINEAR transition "
            "(bilinear_strength > 0); GaussianBelief's fixed-A "
            "algebra is invalid there. The path-matrix filter variant is "
            "required; refusing rather than filtering wrongly.")
    human = set(sim_cfg["actions"].get("human_channels") or [])
    return {"A": A, "B": B, "C": C, "W": W,
            "w_int": np.array([0.0 if c == "none"
                               else (1.0 if c in human else 0.4)
                               for c in CHANNELS]),
            "coef": float(man.get("coef", 1.0)),
            "pop_std": float(man.get("pop_std", 1.0)),
            "obs_noise": float(man.get("obs_noise", 0.5))}


def v_e0_1_brute_force(man: dict, seed: int = 0, n_mc: int = 500_000,
                       months: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    d = man["A"].shape[0]
    z0 = rng.normal(0.0, man["pop_std"], d)
    ci = rng.integers(0, len(CHANNELS), months)
    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], 1)
    z, obs_list = z0.copy(), []
    for s in range(months):
        o = man["C"] @ z + rng.normal(0.0, man["obs_noise"],
                                      man["C"].shape[0])
        obs_list.append(o)
        bel.observe(o[None, :])
        bel.act(np.array([ci[s]]))
        z = man["A"] @ z + man["B"] * man["w_int"][ci[s]]
    mu0, Sig0 = bel.posterior_z0()

    Z = rng.normal(0.0, man["pop_std"], (n_mc, d))
    logw = np.zeros(n_mc)
    m = np.zeros(d)
    for s in range(months):
        pred = Z @ (man["C"] @ bel.Apow(s)).T + m @ man["C"].T
        r = obs_list[s][None, :] - pred
        logw -= (r ** 2).sum(1) / (2 * man["obs_noise"] ** 2)
        m = man["A"] @ m + man["B"] * man["w_int"][ci[s]]
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = 1.0 / float((w ** 2).sum())
    mu_mc = w @ Z
    dz = Z - mu_mc
    Sig_mc = (w[:, None] * dz).T @ dz
    mc_se = 1.0 / np.sqrt(ess)
    err_mu = float(np.abs(mu_mc - mu0[0]).max())
    err_sig = float(np.abs(Sig_mc - Sig0).max())
    tol_mu = 6.0 * mc_se * float(np.sqrt(np.diag(Sig0)).max())
    tol_sig = 8.0 * mc_se * float(np.diag(Sig0).max())
    return {"ess": ess, "err_mu": err_mu, "tol_mu": tol_mu,
            "err_sig": err_sig, "tol_sig": tol_sig,
            "pass": bool(err_mu <= tol_mu and err_sig <= tol_sig)}


def v_e0_2_predictive(world: Path, sim_cfg: dict, k: int, H: int,
                      loans: int) -> dict:
    from scipy.stats import norm

    from benchmark.closed_loop_v7 import load_cohort

    man = manifold_params(sim_cfg)
    book, pre, obs_cols, obs_k, logged, ztrue, obs_logged = load_cohort(
        world, sim_cfg, k, loans)
    ids = list(book["loan_id"])
    counts = logged.groupby("loan_id")["month"].count()
    keep_ids = [l for l in ids if counts.get(l, 0) >= H]
    idx = {l: i for i, l in enumerate(ids)}
    rows = [idx[l] for l in keep_ids]
    n = len(keep_ids)

    pre_obs = np.zeros((len(ids), k, len(obs_cols)))
    pre_ci = np.zeros((len(ids), k), dtype=np.int64)
    for _, r in pre.iterrows():
        i, mo = idx[r["loan_id"]], int(r["month"])
        pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
        pre_ci[i, mo] = CHANNELS.index(r["channel"])
    la = logged.pivot(index="loan_id", columns="month",
                      values="channel").loc[keep_ids]
    ol = obs_logged.pivot_table(index="loan_id", columns="month",
                                values=obs_cols).swaplevel(axis=1)

    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], n)
    for s in range(k):
        bel.observe(pre_obs[rows][:, s, :])
        bel.act(pre_ci[rows][:, s])

    cells = []
    for s in range(k, k + H):
        o_s = np.stack([ol[(s, c)].loc[keep_ids].to_numpy(float)
                        for c in obs_cols], axis=1)
        mu0, Sig0 = bel.posterior_z0()
        Hm = man["C"] @ bel.Apow(s)
        pred = mu0 @ Hm.T + bel.m @ man["C"].T
        cov = Hm @ Sig0 @ Hm.T + man["obs_noise"] ** 2 * np.eye(len(obs_cols))
        sd = np.sqrt(np.diag(cov))
        zres = (o_s - pred) / sd
        for j, c in enumerate(obs_cols):
            cells.append({"month": s, "dim": c,
                          "mean_z": float(zres[:, j].mean() * np.sqrt(n)),
                          "sd": float(zres[:, j].std())})
        bel.observe(o_s)
        bel.act(np.array([CHANNELS.index(c) for c in la[s]]))
    alpha = 0.01 / len(cells)
    zc = float(norm.ppf(1 - alpha / 2))
    sd_tol = zc / np.sqrt(2 * n)
    worst_mean = max(abs(c["mean_z"]) for c in cells)
    worst_sd = max(abs(c["sd"] - 1.0) for c in cells)
    return {"n": n, "cells": cells, "z_crit": zc, "sd_tol": sd_tol,
            "worst_mean_z": worst_mean, "worst_sd_err": worst_sd,
            "pass": bool(worst_mean <= zc and worst_sd <= sd_tol)}


def v_e0_2_synthetic(sim_cfg: dict, k: int, H: int, n: int = 4000,
                     seed: int = 7) -> dict:
    from scipy.stats import norm

    man = manifold_params(sim_cfg)
    rng = np.random.default_rng(seed)
    T = k + H
    z0 = rng.normal(0.0, man["pop_std"], (n, man["A"].shape[0]))
    ci = rng.integers(0, len(CHANNELS), (n, T))
    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], n)
    z = z0.copy()
    cells = []
    m_obs = man["C"].shape[0]
    for s in range(T):
        o = z @ man["C"].T + rng.normal(0.0, man["obs_noise"], (n, m_obs))
        if s >= k:
            mu0, Sig0 = bel.posterior_z0()
            Hm = man["C"] @ bel.Apow(s)
            pred = mu0 @ Hm.T + bel.m @ man["C"].T
            cov = Hm @ Sig0 @ Hm.T + man["obs_noise"] ** 2 * np.eye(m_obs)
            sd = np.sqrt(np.diag(cov))
            zres = (o - pred) / sd
            for j in range(m_obs):
                cells.append({"month": s, "dim": f"obs_{j}",
                              "mean_z": float(zres[:, j].mean() * np.sqrt(n)),
                              "sd": float(zres[:, j].std())})
        bel.observe(o)
        bel.act(ci[:, s])
        z = z @ man["A"].T + np.outer(man["w_int"][ci[:, s]], man["B"])
    alpha = 0.01 / len(cells)
    zc = float(norm.ppf(1 - alpha / 2))
    sd_tol = zc / np.sqrt(2 * n)
    worst_mean = max(abs(c["mean_z"]) for c in cells)
    worst_sd = max(abs(c["sd"] - 1.0) for c in cells)
    return {"n": n, "cells": cells, "z_crit": zc, "sd_tol": sd_tol,
            "worst_mean_z": worst_mean, "worst_sd_err": worst_sd,
            "pass": bool(worst_mean <= zc and worst_sd <= sd_tol)}


def v_e0_3_probe(world: Path, sim_cfg: dict, k: int, loans: int) -> dict:
    import duckdb

    from benchmark.closed_loop_v7 import _sql_ids, load_cohort

    man = manifold_params(sim_cfg)
    book, pre, obs_cols, obs_k, logged, ztrue, obs_logged = load_cohort(
        world, sim_cfg, k, loans)
    ids = list(book["loan_id"])
    n = len(ids)
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    tk = con.execute(f"""
        SELECT t.loan_id, t.p_recovery, a.channel
        FROM oracle_truth t
        JOIN monthly_panel p ON p.loan_id = t.loan_id AND p.month = t.month
        JOIN actions a ON a.action_id = p.action_id
        WHERE t.month = {k} AND t.loan_id IN ({_sql_ids(ids)})
        ORDER BY t.loan_id""").df().set_index("loan_id").loc[ids]
    con.close()
    zc_cols = [c for c in book.columns if c.startswith("zman_")]
    z_true = book[zc_cols].to_numpy(float)

    pre_obs = np.zeros((n, k, len(obs_cols)))
    pre_ci = np.zeros((n, k), dtype=np.int64)
    idx = {l: i for i, l in enumerate(ids)}
    for _, r in pre.iterrows():
        i, mo = idx[r["loan_id"]], int(r["month"])
        pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
        pre_ci[i, mo] = CHANNELS.index(r["channel"])
    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], n)
    for s in range(k):
        bel.observe(pre_obs[:, s, :])
        bel.act(pre_ci[:, s])
    bel.observe(obs_k)

    p_true = np.clip(tk["p_recovery"].to_numpy(float), 1e-9, 1 - 1e-9)
    ch = tk["channel"].to_numpy()
    Wm = np.stack([man["W"][c] for c in CHANNELS])
    Wc = Wm[[CHANNELS.index(c) for c in ch]]
    true_score = man["coef"] * (Wc * z_true).sum(1)
    base = np.log(p_true / (1 - p_true)) - true_score

    mu0, Sig0 = bel.posterior_z0()
    Ak = bel.Apow(k)
    zmu = mu0 @ Ak.T + bel.m
    mean = man["coef"] * (Wc * zmu).sum(1)
    M = Ak.T @ Wc.T
    sd = man["coef"] * np.sqrt(np.einsum("dn,de,en->n", M, Sig0, M))
    p_hat = posterior_pay_prob(base, mean, sd)
    ss_res = float(((p_true - p_hat) ** 2).sum())
    ss_tot = float(((p_true - p_true.mean()) ** 2).sum())
    r2_p = 1.0 - ss_res / ss_tot
    ss_res_s = float(((true_score - mean) ** 2).sum())
    ss_tot_s = float(((true_score - true_score.mean()) ** 2).sum())
    return {"n": n, "R2_p": r2_p, "R2_score": 1.0 - ss_res_s / ss_tot_s,
            "bias_p": float((p_hat - p_true).mean())}


def main() -> int:
    import argparse
    import json

    from decode_sim.config import load_config

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("gates", help="V-E0-1/2/3 (V-E0-4 comes from the board)")
    pg.add_argument("--world", type=Path, required=True)
    pg.add_argument("--config", type=Path, required=True)
    pg.add_argument("--k", type=int, default=6)
    pg.add_argument("--horizon", type=int, default=6)
    pg.add_argument("--loans", type=int, default=4000)
    pg.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sim_cfg = load_config(str(args.config))
    man = manifold_params(sim_cfg)
    r0 = _pay_prob_shape_check()
    print(f"V-E0-0 quadrature shapes vs literal loop: worst "
          f"{r0['worst_abs_diff']:.2e}  ({'PASS' if r0['pass'] else 'FAIL'})",
          flush=True)
    r1 = v_e0_1_brute_force(man)
    print(f"V-E0-1 brute force: err_mu {r1['err_mu']:.2e} (tol {r1['tol_mu']:.2e}) "
          f"err_sig {r1['err_sig']:.2e} (tol {r1['tol_sig']:.2e}) "
          f"ESS {r1['ess']:.0f}  ({'PASS' if r1['pass'] else 'FAIL'})", flush=True)
    r2 = v_e0_2_synthetic(sim_cfg, args.k, args.horizon)
    print(f"V-E0-2 synthetic calibration ({len(r2['cells'])} cells, "
          f"n={r2['n']}): worst |mean z| {r2['worst_mean_z']:.2f} "
          f"(crit {r2['z_crit']:.2f}) worst |sd-1| {r2['worst_sd_err']:.4f} "
          f"(tol {r2['sd_tol']:.4f})  ({'PASS' if r2['pass'] else 'FAIL'})",
          flush=True)
    r2l = v_e0_2_predictive(args.world, sim_cfg, args.k, args.horizon,
                            args.loans)
    print(f"V-E0-2L logged-panel (REPORTED, survival-selected population): "
          f"worst |mean z| {r2l['worst_mean_z']:.2f} — the state-gap tilt of "
          f"the surviving cohort, decomposition in the JSON", flush=True)
    r3 = v_e0_3_probe(args.world, sim_cfg, args.k, args.loans)
    n3 = r3["n"]
    bias_tol = 3.0 * 0.25 / np.sqrt(n3)
    r3["bias_tol"] = bias_tol
    r3["pass"] = bool(abs(r3["bias_p"]) <= bias_tol)
    print(f"V-E0-3 probe: R2(p_hat -> p_true) = {r3['R2_p']:.4f}  "
          f"R2(score) = {r3['R2_score']:.4f}  bias {r3['bias_p']:+.5f} "
          f"(tol {bias_tol:.5f})  ({'PASS' if r3['pass'] else 'FAIL'})",
          flush=True)
    r5 = v_e0_5_quadrature(args.world, sim_cfg, args.k, args.horizon,
                           args.loans)
    print(f"V-E0-5 GH vs MC over lookahead offsets: {r5['cells_checked']} "
          f"cells, worst z {r5['worst_z_vs_mc']:.2f}  "
          f"({'PASS' if r5['pass'] else 'FAIL'}, gate 6 se)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ok = (r0["pass"] and r1["pass"] and r2["pass"] and r3["pass"]
          and r5["pass"])
    rep = {"V_E0_0": r0, "V_E0_1": r1,
           "V_E0_2_synthetic": {kk: r2[kk] for kk in r2 if kk != "cells"},
           "V_E0_2_synthetic_cells": r2["cells"],
           "V_E0_2L_logged_reported": {kk: r2l[kk] for kk in r2l
                                       if kk != "cells"},
           "V_E0_2L_cells": r2l["cells"], "V_E0_3": r3, "V_E0_5": r5,
           "verdict": "PASS" if ok else "FAIL"}
    (args.out / "v_e0_gates.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("V-E0 GATES:", rep["verdict"])
    return 0 if rep["verdict"] == "PASS" else 1


def v_e0_5_quadrature(world: Path, sim_cfg: dict, k: int, H: int,
                      loans: int, seed: int = 0, n_mc: int = 200_000) -> dict:
    from benchmark.closed_loop_v7 import load_cohort

    man = manifold_params(sim_cfg)
    book, pre, obs_cols, obs_k, *_ = load_cohort(world, sim_cfg, k,
                                                 min(loans, 200))
    ids = list(book["loan_id"])
    n = len(ids)
    idx = {l: i for i, l in enumerate(ids)}
    pre_obs = np.zeros((n, k, len(obs_cols)))
    pre_ci = np.zeros((n, k), dtype=np.int64)
    for _, r in pre.iterrows():
        i, mo = idx[r["loan_id"]], int(r["month"])
        pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
        pre_ci[i, mo] = CHANNELS.index(r["channel"])
    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], n)
    for s in range(k):
        bel.observe(pre_obs[:, s, :])
        bel.act(pre_ci[:, s])
    bel.observe(obs_k)

    rng = np.random.default_rng(seed)
    Wm = np.stack([man["W"][c] for c in CHANNELS])
    worst = 0.0
    checked = 0
    for u in range(H):
        for ci in range(len(CHANNELS)):
            drift = np.zeros(bel.d)
            for cj in rng.integers(0, len(CHANNELS), u):
                drift = man["A"] @ drift + man["B"] * man["w_int"][cj]
            mean, sd = bel.score_moments(Wm[ci], man["coef"], u, drift)
            for i in rng.integers(0, n, 3):
                for base in (-4.0, -2.0, 0.0):
                    gh = float(posterior_pay_prob(
                        np.array([base]), np.array([mean[i]]),
                        np.array([sd[i]]))[0])
                    g = rng.normal(mean[i], sd[i], n_mc)
                    mc = float(_sigmoid(base + g).mean())
                    se = float(_sigmoid(base + g).std() / np.sqrt(n_mc))
                    worst = max(worst, abs(gh - mc) / max(se, 1e-12))
                    checked += 1
    return {"cells_checked": checked, "worst_z_vs_mc": worst,
            "pass": bool(worst <= 6.0)}


def e0_jensen_gap(world: Path, sim_cfg: dict, plan_csv: Path, k: int, H: int,
                  loans: int, n_draws: int = 512, seed: int = 0,
                  per_loan: bool = False) -> dict:
    import pandas as pd

    from benchmark.closed_loop_v7 import load_cohort
    from decode_sim.trajectory_key import _World

    man = manifold_params(sim_cfg)
    book, pre, obs_cols, obs_k, *_ = load_cohort(world, sim_cfg, k, loans)
    ids = list(book["loan_id"])
    n = len(ids)
    idx = {l: i for i, l in enumerate(ids)}
    plan = pd.read_csv(plan_csv)
    wide_c = plan.pivot(index="loan_id", columns="month",
                        values="channel").loc[ids]
    wide_d = plan.pivot(index="loan_id", columns="month",
                        values="discount_offer").loc[ids]
    ci = np.stack([[CHANNELS.index(c) for c in wide_c[t]] for t in range(H)])
    dv = np.stack([wide_d[t].to_numpy(float) for t in range(H)])

    pre_obs = np.zeros((n, k, len(obs_cols)))
    pre_ci = np.zeros((n, k), dtype=np.int64)
    for _, r in pre.iterrows():
        i, mo = idx[r["loan_id"]], int(r["month"])
        pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
        pre_ci[i, mo] = CHANNELS.index(r["channel"])
    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], n)
    for s in range(k):
        bel.observe(pre_obs[:, s, :])
        bel.act(pre_ci[:, s])
    bel.observe(obs_k)
    mu0, Sig0 = bel.posterior_z0()
    L = np.linalg.cholesky(Sig0)

    w = _World(book, sim_cfg, H, d_steps=D_STEPS)
    terms = oracle_terms(w)
    d_grid = terms["d_grid"]
    di = np.abs(d_grid[None, None, :] - dv[..., None]).argmin(-1)
    Wm = np.stack([man["W"][c] for c in CHANNELS])
    applies = np.array([c != "none" for c in CHANNELS])
    rng = np.random.default_rng(seed)
    rows = np.arange(n)

    def plan_value(z0_draw):
        z = z0_draw @ bel.Apow(bel.s).T + bel.m
        F = book["entering_fatigue"].to_numpy(float).copy()
        S = np.ones(n)
        val = np.zeros(n)
        for t in range(H):
            c_t, d_t = ci[t], di[t]
            pen = np.where(applies[c_t], terms["beta_i"] * F, 0.0)
            score = man["coef"] * (z * Wm[c_t]).sum(1)
            logit = (terms["logits0"][t, rows, c_t, d_t] - pen + score)
            p = _sigmoid(logit)
            val += S * (p * terms["cash"][t, rows, d_t] - terms["costs"][c_t])
            S *= (1.0 - p) * (1.0 - terms["q"][t])
            F = terms["decay"] * F + terms["fat_add"][c_t]
            z = z @ man["A"].T + np.outer(man["w_int"][c_t], man["B"])
        return val

    if per_loan and n_draws % 8:
        raise ValueError("per_loan needs n_draws divisible by 8")
    tot = np.zeros(n)
    batch = np.zeros((8, n))
    for j in range(n_draws):
        z0 = mu0 + rng.standard_normal((n, bel.d)) @ L.T
        v = plan_value(z0)
        tot += v
        batch[j * 8 // n_draws] += v / (n_draws // 8)
    joint = tot / n_draws

    F = book["entering_fatigue"].to_numpy(float).copy()
    S = np.ones(n)
    ce = np.zeros(n)
    for t in range(H):
        c_t, d_t = ci[t], di[t]
        pen = np.where(applies[c_t], terms["beta_i"] * F, 0.0)
        drift_t = np.zeros((n, bel.d))
        zdrift = np.zeros((n, bel.d))
        for tt in range(t):
            zdrift = zdrift @ man["A"].T + np.outer(man["w_int"][ci[tt]],
                                                    man["B"])
        Wrow = Wm[c_t]
        Au = bel.Apow(bel.s + t)
        mean = (man["coef"]
                * ((mu0 @ Au.T + bel.m @ bel.Apow(t).T + zdrift) * Wrow).sum(1))
        Vs = man["coef"] * (Wrow @ Au)
        sd = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", Vs, Sig0, Vs), 0.0))
        base = terms["logits0"][t, rows, c_t, d_t] - pen
        pbar = posterior_pay_prob(base, mean, sd)
        ce += S * (pbar * terms["cash"][t, rows, d_t] - terms["costs"][c_t])
        S *= (1.0 - pbar) * (1.0 - terms["q"][t])
        F = terms["decay"] * F + terms["fat_add"][c_t]

    gap = joint - ce
    extra = {"joint": joint, "ce": ce, "batch": batch} if per_loan else {}
    return {**extra, "n": n, "n_draws": n_draws,
            "joint_value_mean": float(joint.mean()),
            "ce_chain_value_mean": float(ce.mean()),
            "jensen_gap_mean": float(gap.mean()),
            "jensen_gap_abs_p95": float(np.percentile(np.abs(gap), 95)),
            "note": "gap = E_posterior[exact value of the executed plan] - "
                    "the DP's certainty-equivalent valuation of it. Executed "
                    "VALUE on the board still comes from grade_plan with the "
                    "true z; this measures how far the DP's internal yardstick "
                    "is from the Bayes value, i.e. whether a joint-aware "
                    "solver is worth building."}


if __name__ == "__main__":
    import sys
    raise SystemExit(main())


class RotationBelief:

    def __init__(self, man: dict, n: int):
        self.man = man
        d = man["A"].shape[0]
        self.d, self.n = d, n
        self.inv_prior = np.eye(d) / (man["pop_std"] ** 2)
        self.obs_var = float(man["obs_noise"]) ** 2
        self.Lam = np.zeros((n, d, d))
        self.eta = np.zeros((n, d))
        self.M = np.repeat(np.eye(d)[None], n, axis=0)
        self.m = np.zeros((n, d))
        self.rot = man.get("rot")
        self.T = {float(w): (man["A"] @ self.rot[float(w)]
                             if self.rot is not None else man["A"])
                  for w in np.unique(man["w_int"])}
        self.s = 0

    def observe(self, obs: np.ndarray) -> None:
        Hm = self.man["C"][None] @ self.M
        y = obs - (self.man["C"][None] @ self.m[:, :, None])[:, :, 0]
        self.Lam += np.einsum("nod,noe->nde", Hm, Hm) / self.obs_var
        self.eta += np.einsum("nod,no->nd", Hm, y) / self.obs_var

    def act(self, ci: np.ndarray) -> None:
        wv = self.man["w_int"][np.asarray(ci)]
        Tn = np.stack([self.T[w] for w in wv])
        self.m = np.einsum("nde,ne->nd", Tn, self.m) \
            + wv[:, None] * self.man["B"][None]
        self.M = Tn @ self.M
        self.s += 1

    def posterior_z0(self):
        Sig = np.linalg.inv(self.Lam + self.inv_prior[None])
        mu = np.einsum("nde,ne->nd", Sig, self.eta)
        return mu, Sig

    def score_moments(self, Wv: np.ndarray, coef: float,
                      prefix_T: np.ndarray, prefix_m: np.ndarray, post):
        mu, Sig = post
        G = prefix_T @ self.M
        v = coef * np.einsum("nde,d->ne", G, Wv)
        u = prefix_T.T @ Wv
        mean = (np.einsum("nd,nd->n", v, mu)
                + coef * (self.m @ u)
                + coef * float(Wv @ prefix_m))
        sd = np.sqrt(np.maximum(
            np.einsum("nd,nde,ne->n", v, Sig, v), 0.0))
        return mean, sd


class KalmanBelief:

    def __init__(self, A, B, C, w_int, pop_std, obs_noise,
                 process_noise, n: int):
        self.A, self.B, self.C = A, B, C
        self.w_int = w_int
        d = A.shape[0]
        self.d = d
        self.obs_var = float(obs_noise) ** 2
        self.q2 = float(process_noise) ** 2
        self.mu = np.zeros((n, d))
        self.Sig = np.eye(d) * float(pop_std) ** 2
        self.s = 0

    def observe(self, obs: np.ndarray) -> None:
        C = self.C
        S = C @ self.Sig @ C.T + self.obs_var * np.eye(C.shape[0])
        K = self.Sig @ C.T @ np.linalg.inv(S)
        self.mu = self.mu + (obs - self.mu @ C.T) @ K.T
        IKC = np.eye(self.d) - K @ C
        self.Sig = IKC @ self.Sig @ IKC.T + self.obs_var * (K @ K.T)

    def act(self, ci: np.ndarray) -> None:
        self.mu = self.mu @ self.A.T + np.outer(self.w_int[np.asarray(ci)],
                                                self.B)
        self.Sig = self.A @ self.Sig @ self.A.T + self.q2 * np.eye(self.d)
        self.s += 1

    def posterior_zt(self):
        return self.mu, self.Sig
