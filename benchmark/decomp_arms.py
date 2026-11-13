from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.closed_loop_arms import (
    CH_CODE, F_KNOTS, GBMSolver, STATE, fit_gbm, load_train_rows)
from benchmark.closed_loop_v7 import CHANNELS, D_STEPS
from benchmark.exact_filter import GaussianBelief, manifold_params

def score_feature_names():
    return ([f"score_mean_{c}" for c in CHANNELS]
            + [f"score_sd_{c}" for c in CHANNELS])


def train_posterior_features(train_db, sim_cfg: dict, cache=None):
    import duckdb
    import pandas as pd
    import pickle

    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)

    man = manifold_params(sim_cfg)
    con = duckdb.connect(str(train_db), read_only=True)
    obs_cols = sorted(
        (r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE "
            "table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    df = con.execute(f"""
        SELECT p.loan_id, p.month, a.channel,
               {', '.join('p.' + c for c in obs_cols)}
        FROM monthly_panel p JOIN actions a ON a.action_id = p.action_id
        ORDER BY p.month, p.loan_id
    """).df()
    con.close()

    loans = df["loan_id"].unique()
    pos = {l: i for i, l in enumerate(loans)}
    N, d = len(loans), man["A"].shape[0]
    Wm = np.stack([man["W"][c] for c in CHANNELS])
    coef = man["coef"]

    bel = GaussianBelief(man["A"], man["B"], man["C"], man["w_int"],
                         man["pop_std"], man["obs_noise"], N)
    ch_idx = {c: i for i, c in enumerate(CHANNELS)}
    out_frames = []
    months = sorted(df["month"].unique())
    if months[0] != 0:
        raise AssertionError(
            f"panel starts at month {months[0]}, not 0 — the lockstep "
            f"origination assumption is false and the appearance check below "
            f"would be blind to it")
    prev_alive = None
    for s in months:
        g = df[df["month"] == s]
        rows = np.array([pos[l] for l in g["loan_id"]])
        alive = np.zeros(N, dtype=bool)
        alive[rows] = True
        if prev_alive is not None and bool((alive & ~prev_alive).any()):
            raise AssertionError(
                f"month {s}: loans APPEAR mid-panel — the lockstep prefix "
                f"assumption behind the shared Lambda is false on this world")
        prev_alive = alive
        obs = g[obs_cols].to_numpy(float)
        H = man["C"] @ bel.Apow(s)
        y = obs - bel.m[rows] @ man["C"].T
        bel.Lam += H.T @ H / bel.obs_var
        bel.eta[rows] += y @ H / bel.obs_var
        mu0, Sig0 = bel.posterior_z0()
        As = bel.Apow(s)
        zmu = mu0[rows] @ As.T + bel.m[rows]
        means = coef * (zmu @ Wm.T)
        v = coef * (Wm @ As)
        sds = np.sqrt(np.maximum(np.einsum("cd,de,ce->c", v, Sig0, v), 0.0))
        fr = {"loan_id": g["loan_id"].to_numpy(), "month": s}
        for j, c in enumerate(CHANNELS):
            fr[f"score_mean_{c}"] = means[:, j]
            fr[f"score_sd_{c}"] = np.full(len(rows), sds[j])
        out_frames.append(pd.DataFrame(fr))
        ci = np.array([ch_idx[c] for c in g["channel"]])
        bel.m[rows] = bel.m[rows] @ man["A"].T + np.outer(
            man["w_int"][ci], man["B"])
        bel.s += 1
    feats = pd.concat(out_frames, ignore_index=True)
    if cache is not None:
        feats.to_parquet(cache)
    return feats


def fit_posterior_gbm(train_db, sim_cfg, obs_cols, k: int, seed=0,
                      feat_cache=None):
    feats = train_posterior_features(train_db, sim_cfg, cache=feat_cache)
    tdf, _lag_cols = load_train_rows(train_db, k, obs_cols)
    tdf = tdf.merge(feats, on=["loan_id", "month"], how="left", validate="1:1")
    fcols = score_feature_names()
    miss = int(tdf[fcols].isna().any(axis=1).sum())
    if miss:
        raise AssertionError(
            f"{miss} train rows missing posterior features — the panel and "
            f"the feature table disagree about who was alive when")
    return fit_gbm(tdf, obs_cols, fcols, k, history=True, seed=seed)


class PosteriorGBMSolver(GBMSolver):

    def __init__(self, feats, p_contact, p_none, sim_cfg, k, H, obs_cols,
                 myopic: bool, pshare: float, static_features: bool = False):
        super().__init__(feats, p_contact, p_none, sim_cfg, k, H, obs_cols,
                         history=True, myopic=myopic, pshare=pshare)
        self.static_features = static_features
        stem = "posterior_gbm_static" if static_features else "posterior_gbm"
        self.name = stem + ("_myopic" if myopic else "_sequential")
        self.man = manifold_params(sim_cfg)
        self.Wm = np.stack([self.man["W"][c] for c in CHANNELS])
        self.coef = self.man["coef"]

    def reset(self, book, prehistory) -> None:
        n, k = len(book), self.k
        obs_sorted = sorted(self.obs_cols, key=lambda c: int(c.split("_")[1]))
        pre_obs = np.zeros((n, k, len(obs_sorted)))
        pre_ci = np.zeros((n, k), dtype=np.int64)
        seen = np.zeros((n, k), dtype=bool)
        ids = list(book["loan_id"])
        pos = {l: i for i, l in enumerate(ids)}
        for _, r in prehistory.iterrows():
            i, mo = pos[r["loan_id"]], int(r["month"])
            pre_obs[i, mo] = [float(r[c]) for c in obs_sorted]
            pre_ci[i, mo] = CHANNELS.index(r["channel"])
            seen[i, mo] = True
        if not seen.all():
            raise AssertionError("posterior_gbm: incomplete pre-history")
        self.bel = GaussianBelief(self.man["A"], self.man["B"], self.man["C"],
                                  self.man["w_int"], self.man["pop_std"],
                                  self.man["obs_noise"], n)
        for s in range(k):
            self.bel.observe(pre_obs[:, s, :])
            self.bel.act(pre_ci[:, s])
        self._observed_t = -1

    def _posterior_block(self, view, u: int, prefix: tuple) -> np.ndarray:
        if self.static_features and (u > 0 or prefix):
            return self._posterior_block(view, 0, tuple())
        if view.t != self._observed_t:
            if not (view.frozen and view.t > 0):
                self.bel.observe(view.obs)
            self._post = self.bel.posterior_z0()
            self._observed_t = view.t
        mu0, Sig0 = self._post
        dpref = np.zeros(self.bel.d)
        for cj in prefix:
            dpref = self.man["A"] @ dpref + self.man["B"] * self.man["w_int"][cj]
        Au = self.bel.Apow(self.bel.s + u)
        zmu = mu0 @ Au.T + self.bel.m @ self.bel.Apow(u).T + dpref
        means = self.coef * (zmu @ self.Wm.T)
        v = self.coef * (self.Wm @ Au)
        sds = np.sqrt(np.maximum(np.einsum("cd,de,ce->c", v, Sig0, v), 0.0))
        n = len(zmu)
        return np.concatenate([means, np.tile(sds, (n, 1))], axis=1)

    def decide(self, view):
        from decode_sim.generator import _chargeoff_probability
        import itertools

        n = len(view.loan_ids)
        win = view.months_remaining
        C = len(CHANNELS)
        oc = self.sim_cfg["oracle"]
        s0 = view.state[STATE].to_numpy(float)
        si = {f: STATE.index(f) for f in STATE}

        dpd = s0[:, si["days_past_due"]].copy()
        liq = s0[:, si["days_in_credito_em_liquidacao"]].copy()
        bal = s0[:, si["outstanding_balance"]].copy()
        la = s0[:, si["loan_age_months"]].copy()
        rate = s0[:, si["interest_rate_monthly"]]
        state_u, bal_u, q_u = [], [], []
        for u in range(win):
            q_u.append(np.array([_chargeoff_probability(int(x), oc)
                                 for x in liq]))
            bal_u.append(bal.copy())
            s = s0.copy()
            s[:, si["days_past_due"]] = dpd
            s[:, si["days_in_credito_em_liquidacao"]] = liq
            s[:, si["outstanding_balance"]] = bal
            s[:, si["loan_age_months"]] = la
            state_u.append(s)
            bal = bal * (1.0 + rate * 0.35)
            dpd = dpd + 30.0
            liq = np.where(dpd >= 90.0, liq + 30.0, liq)
            la = la + 1

        F0 = s0[:, self.fi]
        cash = [bal_u[u][:, None] * (1.0 - self.d_grid)[None, :] * self.pshare
                for u in range(win)]
        p_cache: dict[tuple, np.ndarray] = {}

        def p_cells(prefix, ci):
            key = (prefix, ci)
            if key in p_cache:
                return p_cache[key]
            u = len(prefix)
            F = F0.copy()
            for cj in prefix:
                F = self.decay * F + self.fat_add[cj]
            base = np.concatenate(
                [state_u[u], view.obs, self._posterior_block(view, u, prefix)],
                axis=1)
            base[:, self.fi] = F
            if ci == 0:
                out = np.repeat(self.p_none(base)[:, None], D_STEPS, 1)
            else:
                cols = [self.p_contact(base, d, ci) for d in self.d_grid]
                out = np.stack(cols, 1)
            p_cache[key] = out
            return out

        best_V = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        if self.myopic:
            for ci in range(C):
                p = p_cells(tuple(), ci)
                imm = p * cash[0] - self.costs[ci]
                j = np.argmax(imm, axis=1)
                v = imm[np.arange(n), j]
                upd = v > best_V
                best_V[upd] = v[upd]
                ci_out[upd] = ci
                di_out[upd] = j[upd]
        else:
            for path in itertools.product(range(C), repeat=win):
                V = np.zeros(n)
                first_di = np.zeros(n, dtype=np.int64)
                for u in range(win - 1, -1, -1):
                    p = p_cells(tuple(path[:u]), path[u])
                    tot = (-self.costs[path[u]] + p * cash[u]
                           + ((1.0 - p) * ((1.0 - q_u[u]) * V)[:, None]))
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


def fit_longlags_gbm(train_db, obs_cols, k: int, H: int, seed=0):
    tdf, _ = load_train_rows(train_db, k, obs_cols)
    import pandas as pd
    g = tdf.groupby("loan_id", sort=False)
    lag_cols = []
    for j in range(1, k + H):
        for c in list(obs_cols) + ["ch_code"]:
            name = f"{c}_lag{j}"
            if name not in tdf.columns:
                tdf[name] = g[c].shift(j)
            lag_cols.append(name)
    m0 = tdf[tdf["month"] == 0]["loan_id"].nunique()
    if m0 != tdf["loan_id"].nunique():
        raise AssertionError(
            f"{tdf['loan_id'].nunique() - m0} loans lack a month-0 row — "
            f"late origination breaks the lag-NaN semantics")
    first_k = [f"{c}_lag{j}" for j in range(1, k + 1)
               for c in list(obs_cols) + ["ch_code"]]
    return fit_gbm(tdf, obs_cols, lag_cols, k, history=True, seed=seed,
                   dropna_cols=first_k)


class LongLagsSolver(GBMSolver):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.name = ("longlags_myopic" if self.myopic
                     else "longlags_sequential")
        self.window = self.k + self.H - 1

    def _lags(self, view):
        if view.frozen and getattr(self, "_frozen_lags", None) is not None:
            return self._frozen_lags
        cols = []
        for j in range(1, self.window + 1):
            src = view.t - j
            if src >= 0:
                o = view.obs_history[:, src, :]
                ch = np.array([CH_CODE[CHANNELS[c]]
                               for c in view.channel_history[:, src]])
            else:
                mo = self.k + view.t - j
                if mo >= 0:
                    o = self.pre_obs[:, mo, :]
                    ch = self.pre_ch[:, mo]
                else:
                    o = np.full((len(view.loan_ids), view.obs.shape[1]), np.nan)
                    ch = np.full(len(view.loan_ids), np.nan)
            for dcol in range(o.shape[1]):
                cols.append(o[:, dcol])
            cols.append(ch)
        out = np.stack(cols, axis=1)
        if view.frozen:
            self._frozen_lags = out
        return out


class LagsExactFSolver(GBMSolver):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.name = ("lags_exactF_myopic" if self.myopic
                     else "lags_exactF_sequential")

    def decide(self, view):
        import itertools

        from decode_sim.generator import _chargeoff_probability

        n = len(view.loan_ids)
        win = view.months_remaining
        C = len(CHANNELS)
        oc = self.sim_cfg["oracle"]
        s0 = view.state[STATE].to_numpy(float)
        si = {f: STATE.index(f) for f in STATE}
        lag_block = self._lags(view)

        dpd = s0[:, si["days_past_due"]].copy()
        liq = s0[:, si["days_in_credito_em_liquidacao"]].copy()
        bal = s0[:, si["outstanding_balance"]].copy()
        la = s0[:, si["loan_age_months"]].copy()
        rate = s0[:, si["interest_rate_monthly"]]
        state_u, bal_u, q_u = [], [], []
        for u in range(win):
            q_u.append(np.array([_chargeoff_probability(int(x), oc)
                                 for x in liq]))
            bal_u.append(bal.copy())
            s = s0.copy()
            s[:, si["days_past_due"]] = dpd
            s[:, si["days_in_credito_em_liquidacao"]] = liq
            s[:, si["outstanding_balance"]] = bal
            s[:, si["loan_age_months"]] = la
            state_u.append(s)
            bal = bal * (1.0 + rate * 0.35)
            dpd = dpd + 30.0
            liq = np.where(dpd >= 90.0, liq + 30.0, liq)
            la = la + 1

        F0 = s0[:, self.fi]
        cash = [bal_u[u][:, None] * (1.0 - self.d_grid)[None, :] * self.pshare
                for u in range(win)]
        p_cache: dict[tuple, np.ndarray] = {}

        def p_cells(prefix, ci):
            key = (prefix, ci)
            if key in p_cache:
                return p_cache[key]
            u = len(prefix)
            F = F0.copy()
            for cj in prefix:
                F = self.decay * F + self.fat_add[cj]
            base = np.concatenate([state_u[u], view.obs, lag_block], axis=1)
            base[:, self.fi] = F
            if ci == 0:
                out = np.repeat(self.p_none(base)[:, None], D_STEPS, 1)
            else:
                out = np.stack([self.p_contact(base, d, ci)
                                for d in self.d_grid], 1)
            p_cache[key] = out
            return out

        best_V = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        if self.myopic:
            for ci in range(C):
                p = p_cells(tuple(), ci)
                imm = p * cash[0] - self.costs[ci]
                j = np.argmax(imm, axis=1)
                v = imm[np.arange(n), j]
                upd = v > best_V
                best_V[upd] = v[upd]
                ci_out[upd] = ci
                di_out[upd] = j[upd]
        else:
            for path in itertools.product(range(C), repeat=win):
                V = np.zeros(n)
                first_di = np.zeros(n, dtype=np.int64)
                for u in range(win - 1, -1, -1):
                    p = p_cells(tuple(path[:u]), path[u])
                    tot = (-self.costs[path[u]] + p * cash[u]
                           + ((1.0 - p) * ((1.0 - q_u[u]) * V)[:, None]))
                    j = np.argmax(tot, axis=1)
                    V = tot[np.arange(n), j]
                    if u == 0:
                        first_di = j
                upd = V > best_V
                if upd.any():
                    best_V[upd] = V[upd]
                    ci_out[upd] = path[0]
                    di_out[upd] = first_di[upd]
        return ci_out, di_out


class HybridResponseSolver(LagsExactFSolver):

    def __init__(self, feats, p_contact, p_none, sim_cfg, k, H, obs_cols,
                 myopic: bool, pshare: float, terms: dict,
                 exact_none: bool = False, exact_contact: bool = False,
                 static_exact: bool = False):
        super().__init__(feats, p_contact, p_none, sim_cfg, k, H, obs_cols,
                         history=True, myopic=myopic, pshare=pshare)
        self.terms = terms
        self.exact_none = exact_none
        self.exact_contact = exact_contact
        self.static_exact = static_exact
        tag = ("allexact" if exact_none and exact_contact else
               "none_exact" if exact_none else
               "contact_exact" if exact_contact else "allfitted")
        if static_exact:
            tag += "_static"
        self.name = f"hybrid_{tag}" + ("_myopic" if myopic else "_sequential")
        self.man = manifold_params(sim_cfg)
        self.Wm = np.stack([self.man["W"][c] for c in CHANNELS])
        self.coef = self.man["coef"]

    def reset(self, book, prehistory) -> None:
        super().reset(book, prehistory)
        n, k = len(book), self.k
        obs_sorted = sorted(self.obs_cols, key=lambda c: int(c.split("_")[1]))
        pre_obs = np.zeros((n, k, len(obs_sorted)))
        pre_ci = np.zeros((n, k), dtype=np.int64)
        ids = list(book["loan_id"])
        pos = {l: i for i, l in enumerate(ids)}
        for _, r in prehistory.iterrows():
            i, mo = pos[r["loan_id"]], int(r["month"])
            pre_obs[i, mo] = [float(r[c]) for c in obs_sorted]
            pre_ci[i, mo] = CHANNELS.index(r["channel"])
        self.bel = GaussianBelief(self.man["A"], self.man["B"], self.man["C"],
                                  self.man["w_int"], self.man["pop_std"],
                                  self.man["obs_noise"], n)
        for s in range(k):
            self.bel.observe(pre_obs[:, s, :])
            self.bel.act(pre_ci[:, s])
        self._observed_t = -1

    def _exact_cells(self, view, u: int, prefix: tuple, ci: int,
                     F: np.ndarray) -> np.ndarray:
        from benchmark.exact_filter import posterior_pay_prob

        if view.t != self._observed_t:
            if not (view.frozen and view.t > 0):
                self.bel.observe(view.obs)
            self._post = self.bel.posterior_z0()
            self._observed_t = view.t
        if self.static_exact:
            u, prefix = 0, tuple()
        dpref = np.zeros(self.bel.d)
        for cj in prefix:
            dpref = self.man["A"] @ dpref + self.man["B"] * self.man["w_int"][cj]
        mean, sd = self.bel.score_moments(self.Wm[ci], self.coef, u, dpref,
                                          post=self._post)
        pen = (self.terms["beta_i"] * F if CHANNELS[ci] != "none"
               else np.zeros(len(F)))
        base = self.terms["logits0"][view.t + u, :, ci, :] - pen[:, None]
        return posterior_pay_prob(base, mean, sd)

    def decide(self, view):
        import itertools

        from decode_sim.generator import _chargeoff_probability

        n = len(view.loan_ids)
        win = view.months_remaining
        C = len(CHANNELS)
        oc = self.sim_cfg["oracle"]
        s0 = view.state[STATE].to_numpy(float)
        si = {f: STATE.index(f) for f in STATE}
        lag_block = self._lags(view)

        dpd = s0[:, si["days_past_due"]].copy()
        liq = s0[:, si["days_in_credito_em_liquidacao"]].copy()
        bal = s0[:, si["outstanding_balance"]].copy()
        la = s0[:, si["loan_age_months"]].copy()
        rate = s0[:, si["interest_rate_monthly"]]
        state_u, bal_u, q_u = [], [], []
        for u in range(win):
            q_u.append(np.array([_chargeoff_probability(int(x), oc)
                                 for x in liq]))
            bal_u.append(bal.copy())
            s = s0.copy()
            s[:, si["days_past_due"]] = dpd
            s[:, si["days_in_credito_em_liquidacao"]] = liq
            s[:, si["outstanding_balance"]] = bal
            s[:, si["loan_age_months"]] = la
            state_u.append(s)
            bal = bal * (1.0 + rate * 0.35)
            dpd = dpd + 30.0
            liq = np.where(dpd >= 90.0, liq + 30.0, liq)
            la = la + 1

        F0 = s0[:, self.fi]
        cash = [bal_u[u][:, None] * (1.0 - self.d_grid)[None, :] * self.pshare
                for u in range(win)]
        p_cache: dict[tuple, np.ndarray] = {}

        def p_cells(prefix, ci):
            key = (prefix, ci)
            if key in p_cache:
                return p_cache[key]
            u = len(prefix)
            F = F0.copy()
            for cj in prefix:
                F = self.decay * F + self.fat_add[cj]
            exact = self.exact_none if ci == 0 else self.exact_contact
            if exact:
                out = self._exact_cells(view, u, prefix, ci, F)
            else:
                base = np.concatenate([state_u[u], view.obs, lag_block],
                                      axis=1)
                base[:, self.fi] = F
                if ci == 0:
                    out = np.repeat(self.p_none(base)[:, None], D_STEPS, 1)
                else:
                    out = np.stack([self.p_contact(base, d, ci)
                                    for d in self.d_grid], 1)
            p_cache[key] = out
            return out

        best_V = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        if self.myopic:
            for ci in range(C):
                p = p_cells(tuple(), ci)
                imm = p * cash[0] - self.costs[ci]
                j = np.argmax(imm, axis=1)
                v = imm[np.arange(n), j]
                upd = v > best_V
                best_V[upd] = v[upd]
                ci_out[upd] = ci
                di_out[upd] = j[upd]
        else:
            for path in itertools.product(range(C), repeat=win):
                V = np.zeros(n)
                first_di = np.zeros(n, dtype=np.int64)
                for u in range(win - 1, -1, -1):
                    p = p_cells(tuple(path[:u]), path[u])
                    tot = (-self.costs[path[u]] + p * cash[u]
                           + ((1.0 - p) * ((1.0 - q_u[u]) * V)[:, None]))
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


def _irls_logistic_offset(X: np.ndarray, y: np.ndarray, offset: np.ndarray,
                          l2: float = 1.0, iters: int = 30) -> np.ndarray:
    if len(X) < 20 * (X.shape[1] + 1):
        raise ValueError(
            f"IRLS stratum too thin: {len(X)} rows for {X.shape[1] + 1} "
            f"coefficients — refuse rather than return an unstable beta")
    Xd = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    beta = np.zeros(Xd.shape[1])
    lam = l2 * np.eye(Xd.shape[1])
    lam[0, 0] = 0.0
    converged = False
    for _ in range(iters):
        eta = offset + Xd @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        wgt = np.maximum(p * (1 - p), 1e-9)
        z = eta - offset + (y - p) / wgt
        Xw = Xd * wgt[:, None]
        beta_new = np.linalg.solve(Xd.T @ Xw + lam, Xw.T @ z)
        if not np.isfinite(beta_new).all():
            raise FloatingPointError("IRLS produced non-finite coefficients")
        if np.abs(beta_new - beta).max() < 1e-9:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    if not converged:
        print(f"  IRLS: NOT converged in {iters} iters "
              f"(last step {np.abs(beta_new - beta).max():.2e})", flush=True)
    return beta


def fit_structured(train_db, obs_cols, k: int, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    tdf, lag_cols = load_train_rows(train_db, k, obs_cols)
    feats = STATE + obs_cols + lag_cols
    base_cols = STATE + obs_cols
    xcols_i = base_cols + ["discount_offer"] + [f"ch_{c}" for c in CHANNELS[1:]]

    def frame_i(rows):
        d = rows.copy()
        for c in CHANNELS[1:]:
            d[f"ch_{c}"] = (d["channel"] == c).astype(float)
        return d[xcols_i].to_numpy(float)

    tr = tdf[tdf.is_exploration & (tdf.month >= k)].dropna(subset=lag_cols)
    tr = tr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(tr) < 2000:
        raise ValueError(f"too few explore rows: {len(tr)}")
    X_i, y = frame_i(tr), tr["recovered"].to_numpy(int)
    n_val = max(500, int(0.15 * len(tr)))
    gbm = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    gbm.fit(X_i[n_val:], y[n_val:])

    def logit_i(Xi):
        p = np.clip(gbm.predict_proba(Xi)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def lag_design(rows_lags: np.ndarray) -> np.ndarray:
        cols = []
        per = len(obs_cols) + 1
        for j in range(len(lag_cols) // per):
            block = rows_lags[:, j * per:(j + 1) * per]
            cols.append(block[:, :len(obs_cols)])
            code = block[:, len(obs_cols)]
            for cval in (1.0, 2.0, 3.0):
                cols.append((code == cval).astype(float)[:, None])
        return np.concatenate(cols, axis=1)

    L_tr = lag_design(tr[lag_cols].to_numpy(float))
    mu_l, sd_l = L_tr[n_val:].mean(0), np.maximum(L_tr[n_val:].std(0), 1e-9)
    Ls = (L_tr - mu_l) / sd_l
    ch_tr = tr["channel"].to_numpy()
    if (ch_tr == "none").any():
        raise AssertionError(
            "exploration rows contain channel='none' — the per-channel beta "
            "skip would silently miscalibrate them")
    off_all = logit_i(X_i)
    betas = {}
    for ci, cname in enumerate(CHANNELS):
        if cname == "none":
            continue
        m_fit = (ch_tr == cname) & (np.arange(len(tr)) >= n_val)
        betas[ci] = _irls_logistic_offset(Ls[m_fit], y[m_fit], off_all[m_fit])
    tops = {CHANNELS[ci]: float(np.abs(b[1:]).max())
            for ci, b in betas.items()}
    print(f"structured: contact rows {len(tr)}, per-channel |beta|max {tops}",
          flush=True)

    eta_val = off_all[:n_val].copy()
    for ci, cname in enumerate(CHANNELS):
        if ci not in betas:
            continue
        m_ = ch_tr[:n_val] == cname
        eta_val[m_] += betas[ci][0] + Ls[:n_val][m_] @ betas[ci][1:]
    p_val = 1.0 / (1.0 + np.exp(-np.clip(eta_val, -35, 35)))
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(p_val, y[:n_val])

    trn = tdf[(tdf.channel == "none") & (~tdf.is_exploration)
              & (tdf.month >= k)].dropna(subset=lag_cols)
    Xn, yn = trn[feats].to_numpy(float), trn["recovered"].to_numpy(int)
    n_val_n = max(500, int(0.15 * len(trn)))
    gbm_n = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    gbm_n.fit(Xn[n_val_n:], yn[n_val_n:])
    iso_n = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6,
                               out_of_bounds="clip")
    iso_n.fit(gbm_n.predict_proba(Xn[:n_val_n])[:, 1], yn[:n_val_n])

    n_base = len(base_cols)

    def p_contact(base, d, ci):
        Xi = np.concatenate([base[:, :n_base],
                             np.full((len(base), 1), d),
                             np.zeros((len(base), len(CHANNELS) - 1))], axis=1)
        Xi[:, n_base + 1 + ci - 1] = 1.0
        Ls_ = (lag_design(base[:, n_base:]) - mu_l) / sd_l
        b = betas[ci]
        eta = logit_i(Xi) + b[0] + Ls_ @ b[1:]
        return iso.predict(1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35))))

    def p_none(base):
        return iso_n.predict(gbm_n.predict_proba(base)[:, 1])

    return feats, p_contact, p_none
