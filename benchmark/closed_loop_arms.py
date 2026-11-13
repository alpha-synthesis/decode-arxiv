from __future__ import annotations

import numpy as np
from pathlib import Path

from benchmark.closed_loop_v7 import (
    CHANNELS, D_STEPS, Solver, _loan_key)


class OracleSolver(Solver):

    def __init__(self, world, z0: np.ndarray, F0: np.ndarray, H: int,
                 window: int | None = None):
        self.w = world
        self.z0 = np.asarray(z0, float)
        self.F0 = np.asarray(F0, float)
        self.H = H
        self.window = int(window or H)
        self.name = f"oracle_W{self.window}"
        self.fat_add = np.array([self.w.fat_add[c] for c in self.w.channels])

    def reset(self, book, prehistory) -> None:
        self.z = self.z0.copy()
        self.F = self.F0.copy()

    def decide(self, view):
        from decode_sim.receding import window_decision

        got = view.state["contact_fatigue"].to_numpy(float)
        if not np.allclose(got, self.F, atol=1e-9):
            raise AssertionError(
                f"{self.name}: fatigue diverged from the harness at t={view.t} "
                f"(max |diff| {np.abs(got - self.F).max():.3e})")
        win = min(self.window, view.months_remaining)
        ci, di, _ = window_decision(self.w, view.t, win, self.F, self.z)
        self.F = self.w.decay * self.F + self.fat_add[ci]
        if self.w.man is not None:
            self.z = (self.z @ self.w.man["A"].T
                      + np.outer(self.w.man["w_int"][ci], self.w.man["B"]))
        return ci, di


class RandomSolver(Solver):

    name = "random"

    def __init__(self, d_steps: int, seed: int = 0):
        self.d_steps = d_steps
        self.seed = seed

    def decide(self, view):
        n = len(view.loan_ids)
        ci = np.empty(n, dtype=np.int64)
        di = np.empty(n, dtype=np.int64)
        for i, lid in enumerate(view.loan_ids):
            rng = np.random.default_rng(np.random.SeedSequence(
                [0x5EED, self.seed, _loan_key(lid), int(view.t)]))
            ci[i] = rng.integers(len(CHANNELS))
            di[i] = 0 if ci[i] == 0 else rng.integers(self.d_steps)
        return ci, di


STATE = ["days_past_due", "days_in_credito_em_liquidacao", "outstanding_balance",
         "contact_fatigue", "loan_age_months", "age", "monthly_income",
         "original_principal", "interest_rate_monthly"]
CH_CODE = {"none": 0.0, "sms": 1.0, "whatsapp": 2.0, "phone": 3.0}
F_KNOTS = np.array([0.0, 0.35, 0.7, 1.1, 1.5, 1.9, 2.3, 2.8, 3.4])


def fit_gbm(train_df, obs_cols, lag_cols, k, history: bool, seed=0,
            dropna_cols=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    feats = STATE + obs_cols + (lag_cols if history else [])
    xcols = feats + ["discount_offer"] + [f"ch_{c}" for c in CHANNELS[1:]]

    def frame(rows):
        d = rows.copy()
        for c in CHANNELS[1:]:
            d[f"ch_{c}"] = (d["channel"] == c).astype(float)
        return d[xcols].to_numpy(float)

    dn = lag_cols if dropna_cols is None else dropna_cols
    tr = train_df[train_df.is_exploration & (train_df.month >= k)].dropna(
        subset=dn)
    tr = tr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(tr) < 2000:
        raise ValueError(f"too few explore rows with full history: {len(tr)}")
    X, y = frame(tr), tr["recovered"].to_numpy(int)
    n_val = max(500, int(0.15 * len(tr)))
    gbm = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    gbm.fit(X[n_val:], y[n_val:])
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(gbm.predict_proba(X[:n_val])[:, 1], y[:n_val])

    trn = train_df[(train_df.channel == "none") & (~train_df.is_exploration)
                   & (train_df.month >= k)].dropna(subset=dn)
    Xn, yn = trn[feats].to_numpy(float), trn["recovered"].to_numpy(int)
    n_val_n = max(500, int(0.15 * len(trn)))
    gbm_n = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    gbm_n.fit(Xn[n_val_n:], yn[n_val_n:])
    iso_n = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso_n.fit(gbm_n.predict_proba(Xn[:n_val_n])[:, 1], yn[:n_val_n])
    print(f"{'history' if history else 'flat':7s} estimator: "
          f"contact rows {len(tr)}, none rows {len(trn)}", flush=True)

    def p_contact(base, d, ci):
        x = np.concatenate([base, np.full((len(base), 1), d),
                            np.zeros((len(base), len(CHANNELS) - 1))], axis=1)
        x[:, len(feats) + 1 + ci - 1] = 1.0
        return iso.predict(gbm.predict_proba(x)[:, 1])

    def p_none(base):
        return iso_n.predict(gbm_n.predict_proba(base)[:, 1])

    return feats, p_contact, p_none


def _interp_knots(P_t_c, fvals):
    idx = np.clip(np.searchsorted(F_KNOTS, fvals, side="right") - 1,
                  0, len(F_KNOTS) - 2)
    lo, hi = F_KNOTS[idx], F_KNOTS[idx + 1]
    wgt = np.clip((fvals - lo) / (hi - lo), 0.0, 1.0)
    a = P_t_c[np.arange(len(fvals)), :, idx]
    b = P_t_c[np.arange(len(fvals)), :, idx + 1]
    return a * (1.0 - wgt[:, None]) + b * wgt[:, None]


class GBMSolver(Solver):

    def __init__(self, feats, p_contact, p_none, sim_cfg, k, H, obs_cols,
                 history: bool, myopic: bool, pshare: float):
        self.feats, self.p_contact, self.p_none = feats, p_contact, p_none
        self.k, self.H, self.obs_cols, self.history = k, H, obs_cols, history
        self.myopic, self.pshare = myopic, pshare
        oc = sim_cfg["oracle"]["contact_fatigue"]
        human = set(sim_cfg["actions"]["human_channels"])
        self.decay = float(oc["decay"])
        self.fat_add = np.array(
            [0.0 if c == "none" else float(oc["increment"])
             * (float(oc["human_multiplier"]) if c in human else 1.0)
             for c in CHANNELS])
        self.costs = np.array([float((sim_cfg["actions"].get("channel_costs")
                                      or {}).get(c, 0.0)) for c in CHANNELS])
        self.d_grid = np.linspace(
            0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
        self.sim_cfg = sim_cfg
        self.fi = feats.index("contact_fatigue")
        self.name = (("history" if history else "flat")
                     + ("_myopic" if myopic else "_sequential"))

    def reset(self, book, prehistory) -> None:
        ids = list(book["loan_id"])
        pos = {l: i for i, l in enumerate(ids)}
        n, k, m = len(ids), self.k, len(self.obs_cols)
        self.pre_obs = np.zeros((n, k, m))
        self.pre_ch = np.zeros((n, k))
        for _, r in prehistory.iterrows():
            i, mo = pos[r["loan_id"]], int(r["month"])
            if 0 <= mo < k:
                self.pre_obs[i, mo] = [float(r[c]) for c in self.obs_cols]
                self.pre_ch[i, mo] = CH_CODE[r["channel"]]
        self.book = book
        self._frozen_lags = None

    def _lags(self, view):
        if view.frozen:
            if getattr(self, "_frozen_lags", None) is not None:
                return self._frozen_lags
        cols = []
        for j in range(1, self.k + 1):
            src = view.t - j
            if src >= 0:
                o = view.obs_history[:, src, :]
                ch = np.array([CH_CODE[CHANNELS[c]]
                               for c in view.channel_history[:, src]])
            else:
                mo = self.k + view.t - j
                o = self.pre_obs[:, mo, :]
                ch = self.pre_ch[:, mo]
            for d in range(o.shape[1]):
                cols.append(o[:, d])
            cols.append(ch)
        out = np.stack(cols, axis=1)
        if view.frozen:
            self._frozen_lags = out
        return out

    def _state_matrix(self, view):
        return view.state[STATE].to_numpy(float)

    def decide(self, view):
        from decode_sim.generator import _chargeoff_probability

        n = len(view.loan_ids)
        win = view.months_remaining
        C, D = len(CHANNELS), D_STEPS
        oc = self.sim_cfg["oracle"]
        s0 = self._state_matrix(view)
        si = {f: STATE.index(f) for f in STATE}
        lag_block = self._lags(view) if self.history else None

        dpd = s0[:, si["days_past_due"]].copy()
        liq = s0[:, si["days_in_credito_em_liquidacao"]].copy()
        bal = s0[:, si["outstanding_balance"]].copy()
        la = s0[:, si["loan_age_months"]].copy()
        rate = s0[:, si["interest_rate_monthly"]]
        P = np.empty((win, n, C, D, len(F_KNOTS)), dtype=np.float32)
        bal_t = np.empty((win, n))
        q_t = np.empty((win, n))
        for u in range(win):
            q_t[u] = [_chargeoff_probability(int(v), oc) for v in liq]
            bal_t[u] = bal
            s = s0.copy()
            s[:, si["days_past_due"]] = dpd
            s[:, si["days_in_credito_em_liquidacao"]] = liq
            s[:, si["outstanding_balance"]] = bal
            s[:, si["loan_age_months"]] = la
            blocks = [s, view.obs] + ([lag_block] if self.history else [])
            base_u = np.concatenate(blocks, axis=1)
            for kk, fk in enumerate(F_KNOTS):
                b = base_u.copy()
                b[:, self.fi] = fk
                P[u, :, 0, :, kk] = self.p_none(b)[:, None]
                for ci in range(1, C):
                    for di, d in enumerate(self.d_grid):
                        P[u, :, ci, di, kk] = self.p_contact(b, d, ci)
            bal = bal * (1.0 + rate * 0.35)
            dpd = dpd + 30.0
            liq = np.where(dpd >= 90.0, liq + 30.0, liq)
            la = la + 1

        F0 = s0[:, self.fi]
        cash = [bal_t[u][:, None] * (1.0 - self.d_grid)[None, :] * self.pshare
                for u in range(win)]
        if self.myopic:
            best_v = np.full(n, -np.inf)
            ci_out = np.zeros(n, dtype=np.int64)
            di_out = np.zeros(n, dtype=np.int64)
            for ci in range(C):
                p = _interp_knots(P[0, :, ci], F0)
                imm = p * cash[0] - self.costs[ci]
                j = np.argmax(imm, axis=1)
                v = imm[np.arange(n), j]
                upd = v > best_v
                best_v[upd] = v[upd]
                ci_out[upd] = ci
                di_out[upd] = j[upd]
            return ci_out, di_out

        import itertools
        best_V = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        for path in itertools.product(range(C), repeat=win):
            g = np.empty(win)
            f = 0.0
            for u, ci in enumerate(path):
                g[u] = f
                f = self.decay * f + self.fat_add[ci]
            V = np.zeros(n)
            first_di = np.zeros(n, dtype=np.int64)
            for u in range(win - 1, -1, -1):
                ci = path[u]
                p = _interp_knots(P[u, :, ci], self.decay ** u * F0 + g[u])
                tot = -self.costs[ci] + p * cash[u] + ((1.0 - p)
                      * ((1.0 - q_t[u]) * V)[:, None])
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


def load_train_rows(db, k: int, obs_cols):
    import duckdb
    import pandas as pd

    con = duckdb.connect(str(db), read_only=True)
    df = con.execute(f"""
        SELECT p.loan_id, p.month,
               COALESCE(pe.days_past_due, l.days_past_due) AS days_past_due,
               COALESCE(pe.days_in_credito_em_liquidacao,
                        l.days_in_credito_em_liquidacao) AS days_in_credito_em_liquidacao,
               COALESCE(pe.outstanding_balance, l.outstanding_balance) AS outstanding_balance,
               p.contact_fatigue, {', '.join('p.' + c for c in obs_cols)},
               l.loan_age_months + p.month AS loan_age_months,
               l.interest_rate_monthly, l.original_principal,
               b.age, b.monthly_income,
               a.channel, a.discount_offer, a.is_exploration,
               p.recovered
        FROM monthly_panel p
        LEFT JOIN monthly_panel pe ON pe.loan_id = p.loan_id AND pe.month = p.month - 1
        JOIN actions a ON a.action_id = p.action_id
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        ORDER BY p.loan_id, p.month
    """).df()
    con.close()
    bad = [c for c in df.columns if c.startswith("zman_")]
    if bad:
        raise AssertionError(f"LEAK: latent columns in the training frame: {bad}")
    df["ch_code"] = df["channel"].map(CH_CODE).astype(float)
    g = df.groupby("loan_id", sort=False)
    lag_cols = []
    for j in range(1, k + 1):
        for c in list(obs_cols) + ["ch_code"]:
            name = f"{c}_lag{j}"
            df[name] = g[c].shift(j)
            lag_cols.append(name)
    return df, lag_cols


class BeliefSolver(Solver):

    name = "belief_sequential"

    def __init__(self, model, ck, sim_cfg, k, H, obs_cols, device,
                 pshare, eval_frame):
        import torch

        self.torch = torch
        self.model, self.ck = model, ck
        self.k, self.H, self.obs_cols = k, H, obs_cols
        self.device, self.pshare = device, pshare
        self.sim_cfg = sim_cfg
        self.eval_frame = eval_frame
        oc = sim_cfg["oracle"]["contact_fatigue"]
        human = set(sim_cfg["actions"]["human_channels"])
        self.decay = float(oc["decay"])
        self.fat_add = np.array(
            [0.0 if c == "none" else float(oc["increment"])
             * (float(oc["human_multiplier"]) if c in human else 1.0)
             for c in CHANNELS])
        self.costs = np.array([float((sim_cfg["actions"].get("channel_costs")
                                      or {}).get(c, 0.0)) for c in CHANNELS])
        self.d_grid = np.linspace(
            0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
        self.sc = ck["norm"]["sc"]
        self.ocn = ck["norm"]["oc"]
        self.iso_c, self.iso_n = ck["iso_c"], ck["iso_n"]
        self.eye = torch.eye(len(CHANNELS), device=device)

    def reset(self, book, prehistory) -> None:
        from benchmark.belief_v7 import encode_prehistory

        torch = self.torch
        ids = list(book["loan_id"])
        loans, b0 = encode_prehistory(self.model, self.ck, self.eval_frame,
                                      self.obs_cols, self.k, self.device)
        order = {l: i for i, l in enumerate(loans)}
        missing = [l for l in ids if l not in order]
        if missing:
            raise ValueError(
                f"belief arm: {len(missing)} cohort loans have no pre-history "
                f"encoding (first: {missing[:3]})")
        self.b = b0[torch.tensor([order[l] for l in ids], device=self.device)]
        self.n = len(ids)

    def _norm_state(self, s_raw):
        torch = self.torch
        return torch.tensor(((s_raw - self.sc[0]) / self.sc[1]).astype(np.float32),
                            device=self.device)

    def _norm_obs(self, o_raw):
        torch = self.torch
        return torch.tensor(((o_raw - self.ocn[0]) / self.ocn[1]).astype(np.float32),
                            device=self.device)

    def decide(self, view):
        import itertools

        import torch
        from decode_sim.generator import _chargeoff_probability

        n, win = self.n, view.months_remaining
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
            q_u.append(np.array([_chargeoff_probability(int(v), oc) for v in liq]))
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

        obs_now = self._norm_obs(view.obs)
        o_zero = torch.zeros(n, self.ck["o_dim"], device=self.device)
        m_one = torch.ones(n, 1, device=self.device)
        m_zero = torch.zeros(n, 1, device=self.device)
        F0 = s0[:, si["contact_fatigue"]]
        dgrid_t = torch.tensor(self.d_grid, dtype=torch.float32,
                               device=self.device)

        def fatigue_along(prefix):
            F = F0.copy()
            for ci in prefix:
                F = self.decay * F + self.fat_add[ci]
            return F

        def norm_state_u(u, F):
            s = state_u[u].copy()
            s[:, si["contact_fatigue"]] = F
            return self._norm_state(s)

        belief_cache = {(): self.b}

        def belief_at(prefix):
            if prefix in belief_cache:
                return belief_cache[prefix]
            b_prev = belief_at(prefix[:-1])
            u = len(prefix) - 1
            s = norm_state_u(u, fatigue_along(prefix[:-1]))
            real = (view.t + u == 0) if view.frozen else (u == 0)
            o_t, m_t = (obs_now, m_one) if real else (o_zero, m_zero)
            with torch.no_grad():
                b_new = self.model.step(b_prev, s, o_t, m_t,
                                        self.eye[prefix[-1]].expand(n, -1))
            belief_cache[prefix] = b_new
            return b_new

        p_cache = {}

        def p_cells(prefix, ci):
            keyc = (prefix, ci)
            if keyc in p_cache:
                return p_cache[keyc]
            u = len(prefix)
            b = belief_at(prefix)
            s = norm_state_u(u, fatigue_along(prefix))
            real = (view.t + u == 0) if view.frozen else (u == 0)
            o_t, m_t = (obs_now, m_one) if real else (o_zero, m_zero)
            with torch.no_grad():
                if ci == 0:
                    p = torch.sigmoid(
                        self.model.p_org_logit(b, s, o_t, m_t)).cpu().numpy()
                    out = np.repeat(np.asarray(self.iso_n.predict(p))[:, None],
                                    D_STEPS, 1)
                else:
                    ch1 = self.eye[ci].expand(n, -1)
                    cols = []
                    for d in dgrid_t:
                        lg = self.model.p_pay_logit(b, s, o_t, m_t, ch1,
                                                    d.expand(n, 1))
                        cols.append(torch.sigmoid(lg).cpu().numpy())
                    p = np.stack(cols, 1)
                    out = self.iso_c.predict(p.ravel()).reshape(p.shape)
            p_cache[keyc] = out
            return out

        best_V = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        for path in itertools.product(range(C), repeat=win):
            V = np.zeros(n)
            first_di = np.zeros(n, dtype=np.int64)
            for u in range(win - 1, -1, -1):
                p = p_cells(tuple(path[:u]), path[u])
                cash = (bal_u[u][:, None] * (1.0 - self.d_grid)[None, :]
                        * self.pshare)
                tot = -self.costs[path[u]] + p * cash + ((1.0 - p)
                      * ((1.0 - q_u[u]) * V)[:, None])
                j = np.argmax(tot, axis=1)
                V = tot[np.arange(n), j]
                if u == 0:
                    first_di = j
            upd = V > best_V
            if upd.any():
                best_V[upd] = V[upd]
                ci_out[upd] = path[0]
                di_out[upd] = first_di[upd]

        real_step = (view.t == 0) if view.frozen else True
        with torch.no_grad():
            self.b = self.model.step(
                self.b, norm_state_u(0, F0),
                obs_now if real_step else o_zero,
                m_one if real_step else m_zero,
                self.eye[torch.tensor(ci_out, device=self.device)])
        return ci_out, di_out


def _load_public_static(world, ids, k: int, H: int):
    import duckdb

    from benchmark.closed_loop_v7 import _sql_ids

    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    obs_cols = sorted(
        (r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE "
            "table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    idl = _sql_ids(ids)
    q = con.execute(f"""
        SELECT p.loan_id, p.remaining_maturity_months, p.status,
               l.original_principal, l.product_type, l.origination_month,
               b.uf, b.city, b.lat, b.lon, b.gender, b.dependents,
               b.income_band
        FROM monthly_panel p
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        WHERE p.loan_id IN ({idl}) AND p.month = {k}
        ORDER BY p.loan_id""").df()
    ent = con.execute(f"""
        SELECT loan_id, contacts_last_3m FROM monthly_panel
        WHERE loan_id IN ({idl}) AND month = {k - 1}
        ORDER BY loan_id""").df().set_index("loan_id")
    span = con.execute(f"""
        SELECT p.loan_id, p.month, p.contacts_last_3m, a.channel
        FROM monthly_panel p JOIN actions a ON a.action_id = p.action_id
        WHERE p.loan_id IN ({idl}) AND p.month BETWEEN {k - 1} AND {k + H - 2}
        ORDER BY p.loan_id, p.month""").df()
    con.close()
    cp = span.pivot(index="loan_id", columns="month",
                    values="contacts_last_3m").reindex(list(ids))
    ch = span.pivot(index="loan_id", columns="month",
                    values="channel").reindex(list(ids))
    months = [k - 1 + t for t in range(H)]
    contacts_panel = cp[months].to_numpy(float)
    logged_contact = (ch[months] != "none").to_numpy().astype(float)
    bad = [c for c in q.columns if c.startswith("zman_")]
    if bad:
        raise AssertionError(f"LEAK: latent in the static frame: {bad}")
    q = q.set_index("loan_id").loc[list(ids)]
    return {"remaining_maturity_months":
                q["remaining_maturity_months"].to_numpy(float),
            "status_k": q["status"].to_numpy(),
            "contacts_entering_k":
                ent.loc[list(ids), "contacts_last_3m"].to_numpy(float),
            "contacts_panel": contacts_panel,
            "logged_contact": logged_contact,
            "uf": q["uf"].to_numpy(), "gender": q["gender"].to_numpy(),
            "original_principal": q["original_principal"].to_numpy(float),
            "dependents": q["dependents"].to_numpy(float),
            "product_type": q["product_type"].to_numpy(),
            "income_band": q["income_band"].to_numpy(),
            "city": q["city"].to_numpy(), "lat": q["lat"].to_numpy(float),
            "lon": q["lon"].to_numpy(float),
            "origination_month": q["origination_month"].to_numpy(),
            "obs_cols": obs_cols}


P7_NUMERIC = ["dpd_entering", "liq_days_entering", "bal_entering",
              "utilization", "contacts_last_3m", "loan_age_months",
              "remaining_maturity_months", "interest_rate_monthly",
              "original_principal", "monthly_income", "income_zero",
              "age", "dependents"]
P7_CATEG = ["product_type", "uf", "gender", "income_band"]


def fit_p7(train_db, obs_cols, seed: int = 2027, cache=None):
    import pickle

    import duckdb
    import pandas as pd

    if cache is not None and Path(cache).exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    import lightgbm as lgb

    con = duckdb.connect(str(train_db), read_only=True)
    tr = con.execute(f"""
        SELECT p.month, a.channel, a.discount_offer, p.recovered_amount,
               COALESCE(pe.days_past_due, l.days_past_due) AS dpd_entering,
               COALESCE(pe.days_in_credito_em_liquidacao,
                        l.days_in_credito_em_liquidacao) AS liq_days_entering,
               COALESCE(pe.outstanding_balance, l.outstanding_balance) AS bal_entering,
               LEAST(COALESCE(pe.outstanding_balance, l.outstanding_balance)
                     / nullif(l.original_principal, 0), 10) AS utilization,
               COALESCE(pe.contacts_last_3m, 0) AS contacts_last_3m,
               l.loan_age_months + p.month AS loan_age_months,
               p.remaining_maturity_months, l.interest_rate_monthly,
               l.original_principal, b.monthly_income,
               (b.monthly_income <= 0)::INT AS income_zero,
               b.age, b.dependents, l.product_type, b.uf, b.gender,
               b.income_band, {', '.join('p.' + c for c in obs_cols)}
        FROM monthly_panel p
        LEFT JOIN monthly_panel pe ON pe.loan_id = p.loan_id AND pe.month = p.month - 1
        JOIN actions a ON a.action_id = p.action_id
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
    """).df()
    con.close()
    bad = [c for c in tr.columns if c.startswith("zman_")]
    if bad:
        raise AssertionError(f"LEAK: latent in the Tweedie training frame: {bad}")
    tr["utilization"] = tr["utilization"].fillna(0)
    for c in P7_CATEG + ["channel"]:
        tr[c] = tr[c].astype("category")
    feats = P7_NUMERIC + P7_CATEG + list(obs_cols)
    rfeats = feats + ["channel", "discount_offer"]
    m_fit = tr["month"] <= 17
    m_val = tr["month"].between(18, 19)
    reward = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5, n_estimators=2000,
        learning_rate=0.05, num_leaves=63, min_child_samples=500,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        random_state=seed, n_jobs=16, verbose=-1)
    reward.fit(tr.loc[m_fit, rfeats], tr.loc[m_fit, "recovered_amount"],
               eval_set=[(tr.loc[m_val, rfeats],
                          tr.loc[m_val, "recovered_amount"])],
               eval_metric="tweedie",
               callbacks=[lgb.early_stopping(100, verbose=False),
                          lgb.log_evaluation(0)])
    print(f"Tweedie reward model: {int(m_fit.sum()):,} fit rows, "
          f"{int(m_val.sum()):,} val rows, best_iter {reward.best_iteration_}",
          flush=True)
    art = {"model": reward, "feats": feats, "rfeats": rfeats,
           "channel_categories": list(tr["channel"].cat.categories),
           "categ_categories": {c: list(tr[c].cat.categories)
                                for c in P7_CATEG}}
    if cache is not None:
        with open(cache, "wb") as f:
            pickle.dump(art, f)
    return art


class P7Solver(Solver):

    name = "p7_myopic"

    def __init__(self, art, sim_cfg, k, H, obs_cols, static):
        import pandas as pd

        self.pd = pd
        self.art = art
        self.k, self.H, self.obs_cols = k, H, obs_cols
        self.static = static
        self.costs = {c: float((sim_cfg["actions"].get("channel_costs") or {})
                               .get(c, 0.0)) for c in CHANNELS}
        self.d_grid = np.linspace(
            0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)

    def reset(self, book, prehistory) -> None:
        self.contacts_k = self.static["contacts_entering_k"].copy()

    def _contacts(self, view):
        c = self.contacts_k.copy()
        for u in range(view.t):
            c = np.minimum(3.0, c + (view.channel_history[:, u] != 0))
        return c

    def decide(self, view):
        pd = self.pd
        st = view.state
        n = len(view.loan_ids)
        bal = st["outstanding_balance"].to_numpy(float)
        X = pd.DataFrame({
            "dpd_entering": st["days_past_due"].to_numpy(float),
            "liq_days_entering":
                st["days_in_credito_em_liquidacao"].to_numpy(float),
            "bal_entering": bal,
            "utilization": np.minimum(
                np.divide(bal, self.static["original_principal"],
                          out=np.zeros_like(bal),
                          where=self.static["original_principal"] != 0), 10.0),
            "contacts_last_3m": self._contacts(view),
            "loan_age_months": st["loan_age_months"].to_numpy(float),
            "remaining_maturity_months": np.maximum(
                self.static["remaining_maturity_months"] - view.t, 0.0),
            "interest_rate_monthly": st["interest_rate_monthly"].to_numpy(float),
            "original_principal": self.static["original_principal"],
            "monthly_income": st["monthly_income"].to_numpy(float),
            "income_zero": (st["monthly_income"].to_numpy(float) <= 0)
                            .astype(int),
            "age": st["age"].to_numpy(float),
            "dependents": self.static["dependents"],
        })
        for c in P7_CATEG:
            X[c] = pd.Categorical(self.static[c],
                                  categories=self.art["categ_categories"][c])
        for j, c in enumerate(self.obs_cols):
            X[c] = view.obs[:, j]

        cats = self.art["channel_categories"]
        best_net = np.full(n, -np.inf)
        ci_out = np.zeros(n, dtype=np.int64)
        di_out = np.zeros(n, dtype=np.int64)
        for ci, ch in enumerate(CHANNELS):
            d_opts = [0] if ch == "none" else range(len(self.d_grid))
            for di in d_opts:
                Xa = X.copy()
                Xa["channel"] = pd.Categorical([ch] * n, categories=cats)
                Xa["discount_offer"] = self.d_grid[di]
                net = (self.art["model"].predict(Xa[self.art["rfeats"]])
                       - self.costs[ch])
                upd = net > best_net
                best_net[upd] = net[upd]
                ci_out[upd] = ci
                di_out[upd] = di
        return ci_out, di_out


_DIGITAL = {"sms", "whatsapp"}
_HUMAN = {"phone"}


class AnalystSolver(Solver):

    def __init__(self, policy_path, sim_cfg, k, H, obs_cols,
                 static, name: str = "analyst"):
        import json as _json

        from benchmark.rule_policy import decide, validate_rules

        self._decide = decide
        self._validate = validate_rules
        art = _json.loads(Path(policy_path).read_text())
        self.policy = art.get("policy", art)
        if not isinstance(self.policy.get("rules"), list)                 or not self.policy.get("default"):
            raise ValueError(f"{policy_path}: no rules/default found")
        self.k, self.H, self.obs_cols = k, H, obs_cols
        self.static = static
        self.d_grid = np.linspace(
            0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
        self.name = name

    def reset(self, book, prehistory) -> None:
        import pandas as pd

        ids = list(book["loan_id"])
        pos = {l: i for i, l in enumerate(ids)}
        n, k = len(ids), self.k
        self.n = n
        self.pre_ch = np.full((n, k), "none", dtype=object)
        self.pre_d = np.zeros((n, k))
        for _, r in prehistory.iterrows():
            i, mo = pos[r["loan_id"]], int(r["month"])
            if 0 <= mo < k:
                self.pre_ch[i, mo] = r["channel"]
                self.pre_d[i, mo] = float(r["discount_offer"])
        self.contacts_k = self.static["contacts_entering_k"].copy()
        frame_cols = set(self._frame_columns())
        kept = self._validate(self.policy, frame_cols)
        if len(kept) != len(self.policy.get("rules", [])):
            names = [r["if"] for r in self.policy["rules"]
                     if r not in kept]
            raise AssertionError(
                f"{self.name}: {len(self.policy['rules']) - len(kept)} policy "
                f"rules reference columns this loop cannot serve and would be "
                f"SILENTLY dropped: {names[:3]} — the graded policy would not "
                f"be the committed one")
        _ = pd.DataFrame

    def _frame_columns(self):
        return (["month", "status", "loan_age_months",
                 "remaining_maturity_months", "days_past_due",
                 "days_in_credito_em_liquidacao", "outstanding_balance",
                 "contacts_last_3m", "contact_fatigue"]
                + list(self.obs_cols)
                + ["product_type", "original_principal",
                   "interest_rate_monthly", "uf", "city", "lat", "lon", "age",
                   "gender", "income_band", "monthly_income", "dependents",
                   "prev_disc", "cum_contacts", "dig_n", "hum_n", "non_n",
                   "dig_rate", "hum_rate", "non_rate", "mo_since", "pchan",
                   "p2chan", "origination_month", "dpd_band"])

    def _history_block(self, view):
        n, k, t = self.n, self.k, view.t
        m_abs = k + t
        own_ch = np.array([[CHANNELS[c] for c in row]
                           for row in view.channel_history], dtype=object) \
            if t else np.empty((n, 0), dtype=object)
        ch_all = np.concatenate([self.pre_ch, own_ch], axis=1)
        d_all = np.concatenate([self.pre_d, view.discount_history], axis=1) \
            if t else self.pre_d
        is_con = ch_all != "none"
        m_clip = max(m_abs, 1)
        out = {
            "prev_disc": d_all[:, m_abs - 1] if m_abs else np.zeros(n),
            "cum_contacts": is_con.sum(1).astype(float),
            "dig_n": np.isin(ch_all, list(_DIGITAL)).sum(1).astype(float),
            "hum_n": np.isin(ch_all, list(_HUMAN)).sum(1).astype(float),
            "non_n": (ch_all == "none").sum(1).astype(float),
            "pchan": ch_all[:, m_abs - 1] if m_abs else np.full(n, "none"),
            "p2chan": ch_all[:, m_abs - 2] if m_abs >= 2 else np.full(n, "none"),
        }
        out["dig_rate"] = out["dig_n"] / m_clip
        out["hum_rate"] = out["hum_n"] / m_clip
        out["non_rate"] = out["non_n"] / m_clip
        mo_since = np.full(n, float(m_abs))
        idx = np.where(is_con.any(1))[0]
        if len(idx):
            last = m_abs - 1 - np.argmax(is_con[idx][:, ::-1], axis=1)
            mo_since[idx] = m_abs - last
        out["mo_since"] = mo_since
        return out

    def decide(self, view):
        import pandas as pd

        st = view.state
        n = self.n
        dpd = st["days_past_due"].to_numpy(float)
        liq = st["days_in_credito_em_liquidacao"].to_numpy(float)
        c = self.contacts_k.copy()
        for u in range(view.t):
            c = np.minimum(3.0, c + (view.channel_history[:, u] != 0))
        df = pd.DataFrame({
            "month": np.full(n, self.k + view.t),
            "status": np.where(dpd >= 90, "credito_em_liquidacao",
                               np.where(dpd >= 30, "delinquent", "performing")),
            "loan_age_months": st["loan_age_months"].to_numpy(float),
            "remaining_maturity_months": np.maximum(
                self.static["remaining_maturity_months"] - view.t, 0.0),
            "days_past_due": dpd,
            "days_in_credito_em_liquidacao": liq,
            "outstanding_balance": st["outstanding_balance"].to_numpy(float),
            "contacts_last_3m": c,
            "contact_fatigue": st["contact_fatigue"].to_numpy(float),
            "product_type": self.static["product_type"],
            "original_principal": self.static["original_principal"],
            "interest_rate_monthly": st["interest_rate_monthly"].to_numpy(float),
            "uf": self.static["uf"], "city": self.static["city"],
            "lat": self.static["lat"], "lon": self.static["lon"],
            "age": st["age"].to_numpy(float),
            "gender": self.static["gender"],
            "income_band": self.static["income_band"],
            "monthly_income": st["monthly_income"].to_numpy(float),
            "dependents": self.static["dependents"],
            "origination_month": self.static["origination_month"],
            "dpd_band": pd.cut(dpd, [-1, 30, 90, 10 ** 9],
                               labels=["early", "mid", "late"]).astype(str),
            **self._history_block(view),
        })
        for j, cn in enumerate(self.obs_cols):
            df[cn] = view.obs[:, j]
        out = self._decide(df, self.policy)
        ci = np.array([CHANNELS.index(ch) for ch in out["channel"]],
                      dtype=np.int64)
        d = out["discount_offer"].to_numpy(float)
        di = np.abs(self.d_grid[None, :] - d[:, None]).argmin(1).astype(np.int64)
        di[ci == 0] = 0
        return ci, di
