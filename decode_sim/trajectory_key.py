from __future__ import annotations

import itertools
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from decode_sim import oracle_response
from decode_sim.generator import (
    RecoveryInputs,
    _chargeoff_probability,
    recovery_probability,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def paid_share_mean(oracle_cfg: dict) -> float:
    if oracle_cfg.get("classical_policy"):
        return 1.0
    rng = np.random.default_rng(0)
    return float(np.minimum(1.0, 0.7 + rng.beta(2.0, 5.0, size=2_000_000)).mean())


def _book(db_path: str | Path, loan_ids=None) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tcols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'oracle_truth'").fetchall()}
        fbeta_sel = ", t.fatigue_beta" if "fatigue_beta" in tcols else ""
        zman_cols = sorted((c for c in tcols if c.startswith("zman_")),
                           key=lambda c: int(c.split("_")[1]))
        zman_sel = "".join(f", t.{c}" for c in zman_cols)
        book = con.execute(f"""
            SELECT l.loan_id, l.outstanding_balance, l.days_past_due,
                   l.days_in_credito_em_liquidacao, l.loan_age_months,
                   l.interest_rate_monthly, l.original_principal, l.initial_status,
                   b.age, b.monthly_income,
                   t.borrower_resilience, t.payment_friction,
                   t.regional_macro_stress, t.contactability{fbeta_sel}{zman_sel}
            FROM loans l
            JOIN borrowers b ON b.borrower_id = l.borrower_id
            JOIN (SELECT DISTINCT ON (loan_id) * FROM oracle_truth
                  ORDER BY loan_id, month) t USING (loan_id)
            ORDER BY l.loan_id
        """).df()
    finally:
        con.close()
    if "fatigue_beta" not in book.columns:
        book["fatigue_beta"] = np.nan
    if loan_ids is not None:
        book = book[book["loan_id"].isin(set(loan_ids))]
    return book.reset_index(drop=True)


def _book_at_month(db_path: str | Path, start_month: int, loan_ids=None) -> pd.DataFrame:
    if start_month == 0:
        return _book(db_path, loan_ids)
    k = int(start_month)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tcols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'oracle_truth'").fetchall()}
        pcols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'monthly_panel'").fetchall()}
        fbeta_sel = ", t.fatigue_beta" if "fatigue_beta" in tcols else ""
        zman_cols = sorted((c for c in tcols if c.startswith("zman_")),
                           key=lambda c: int(c.split("_")[1]))
        zman_sel = "".join(f", t.{c}" for c in zman_cols)
        fat_sel = (", pk.contact_fatigue AS entering_fatigue"
                   if "contact_fatigue" in pcols else "")
        book = con.execute(f"""
            SELECT l.loan_id,
                   pe.outstanding_balance, pe.days_past_due,
                   pe.days_in_credito_em_liquidacao,
                   l.loan_age_months + {k} AS loan_age_months,
                   l.interest_rate_monthly, l.original_principal,
                   l.initial_status,
                   b.age, b.monthly_income,
                   t.borrower_resilience, t.payment_friction,
                   t.regional_macro_stress, t.contactability
                   {fbeta_sel}{zman_sel}{fat_sel}
            FROM monthly_panel pk
            JOIN monthly_panel pe ON pe.loan_id = pk.loan_id
                                 AND pe.month = {k - 1}
            JOIN loans l ON l.loan_id = pk.loan_id
            JOIN borrowers b ON b.borrower_id = l.borrower_id
            JOIN oracle_truth t ON t.loan_id = pk.loan_id AND t.month = {k}
            WHERE pk.month = {k}
            ORDER BY l.loan_id
        """).df()
    finally:
        con.close()
    if "fatigue_beta" not in book.columns:
        book["fatigue_beta"] = np.nan
    if "entering_fatigue" not in book.columns:
        book["entering_fatigue"] = 0.0
    if loan_ids is not None:
        book = book[book["loan_id"].isin(set(loan_ids))]
    return book.reset_index(drop=True)


class _World:

    def __init__(self, book: pd.DataFrame, config: dict, H: int, d_steps: int = 21):
        oc = config["oracle"]
        fat = oc.get("contact_fatigue") or {}
        self.oc, self.H = oc, H
        self.book = book
        self.beta_cfg = float(fat.get("beta", 0.0))
        self.decay = float(fat.get("decay", 0.5))
        self.split_share = float((fat.get("channel_split") or {}).get("share", 0.0))
        self.gamma = float((oc.get("discount_fatigue") or {}).get("gamma", 0.0))
        human = set(config["actions"].get("human_channels") or [])
        chans_all = list(config["actions"]["channels"])
        self.channels = chans_all
        if "sms" in chans_all and "whatsapp" in chans_all:
            eff = oc.get("channel_effects") or {}
            ch_costs0 = config["actions"].get("channel_costs") or {}
            dominated = (
                float(eff.get("sms", 0.0)) <= float(eff.get("whatsapp", 0.0))
                and float(ch_costs0.get("sms", 0.0)) >= float(ch_costs0.get("whatsapp", 0.0))
                and ("sms" in human) == ("whatsapp" in human))
            if dominated and not self.split_share and not oc.get("manifold"):
                self.channels = [c for c in chans_all if c != "sms"]
        self.all_channels = list(config["actions"]["channels"])
        ch_costs = config["actions"].get("channel_costs") or {}
        self.costs = {c: float(ch_costs.get(c, 0.0)) for c in self.all_channels}
        inc = float(fat.get("increment", 1.0))
        mult = float(fat.get("human_multiplier", 1.0))
        self.fat_add = {c: 0.0 if c == "none" else inc * (mult if c in human else 1.0)
                        for c in self.all_channels}
        man = oc.get("manifold")
        self.man = None
        if man:
            from decode_sim import oracle_response as _orsp
            n_lat, m_obs, mchans, A, B, Cm, Wd, rot = _orsp._manifold(man)
            coef = float(man.get("coef", 1.0))
            human = set(config["actions"].get("human_channels") or [])
            w_int = np.array([0.0 if c == "none" else (1.0 if c in human else 0.4)
                              for c in self.channels])
            zc = [c for c in book.columns if c.startswith("zman_")]
            if len(zc) != n_lat:
                raise ValueError(
                    f"manifold world: oracle_truth has {len(zc)} zman columns, "
                    f"config says n_latent={n_lat}")
            z0 = book[zc].to_numpy(float)
            Wm = np.stack([Wd[c] for c in self.channels])
            if rot is None:
                self.man_drift = np.empty((H, len(book), len(self.channels)))
                zt = z0.copy()
                for t in range(H):
                    self.man_drift[t] = coef * (zt @ Wm.T)
                    zt = zt @ A.T
            else:
                self.man_drift = None
            self.man = {"A": A, "B": B, "W": Wm, "coef": coef,
                        "w_int": w_int, "n_lat": n_lat, "rot": rot,
                        "z0": z0}
        self.d_grid = np.linspace(0.0, float(config["actions"]["max_discount"]), d_steps)
        self.pshare = paid_share_mean(oc)
        n = len(book)
        self.beta_i = book["fatigue_beta"].to_numpy(float)
        self.beta_i = np.where(np.isnan(self.beta_i), self.beta_cfg, self.beta_i)

        dpd = book["days_past_due"].to_numpy(float)
        liq = book["days_in_credito_em_liquidacao"].to_numpy(float)
        bal = book["outstanding_balance"].to_numpy(float)
        rate = book["interest_rate_monthly"].to_numpy(float)
        self.q = np.empty((H, n))
        self.cash = np.empty((H, n, len(self.d_grid)))
        self.logits0 = np.empty((H, n, len(self.channels), len(self.d_grid)))
        for t in range(H):
            self.q[t] = [_chargeoff_probability(int(x), oc) for x in liq]
            self.cash[t] = bal[:, None] * (1.0 - self.d_grid)[None, :] * self.pshare
            for ci, c in enumerate(self.channels):
                for gi, g in enumerate(self.d_grid):
                    p = np.array([recovery_probability(RecoveryInputs(
                        channel=c, discount_offer=float(g),
                        age=int(book["age"][i]),
                        loan_age_months=int(book["loan_age_months"][i]) + t,
                        days_past_due=int(dpd[i]),
                        days_in_credito_em_liquidacao=int(liq[i]),
                        borrower_resilience=book["borrower_resilience"][i],
                        payment_friction=book["payment_friction"][i],
                        regional_macro_stress=book["regional_macro_stress"][i],
                        contactability=book["contactability"][i],
                        contact_fatigue=0.0, owed_amount=float(bal[i]),
                        manifold_z=((0.0,) * self.man["n_lat"]) if self.man else (),
                        monthly_income=book["monthly_income"][i],
                        original_principal=book["original_principal"][i],
                        noise=0.0), oc) for i in range(n)])
                    self.logits0[t, :, ci, gi] = _logit(p)
            bal = bal * (1.0 + rate * 0.35)
            dpd = dpd + 30.0
            liq = np.where(dpd >= 90.0, liq + 30.0, liq)


def _refuse_unmodeled(config: dict) -> None:
    man = (config.get("oracle") or {}).get("manifold") or {}
    if man and float(man.get("process_noise", 0.1)) > 0.0:
        raise ValueError(
            "trajectory_key can price manifold worlds only with "
            "process_noise: 0 (deterministic hidden state given actions) — "
            "stochastic latent drift has no closed-form ceiling; use Monte "
            "Carlo grading")
    js = (config.get("oracle") or {}).get("judicial_shock") or {}
    if float(js.get("monthly_rate", 0.0)) > 0.0:
        raise ValueError(
            "trajectory_key cannot price judicial_shock worlds — the analytic "
            "DP/grader roll a deterministic dpd/liq clock. Grade shocked "
            "worlds by Monte Carlo over the shock process or extend the roll.")


def _memory_tree_dp(w: "_World", F0_i: np.ndarray, applies: np.ndarray,
                    fat_add: np.ndarray, costs: np.ndarray, H: int):
    n = len(F0_i)
    C = len(w.channels)
    D = len(w.d_grid)
    grid = w.d_grid
    rows = np.arange(n)
    s = w.split_share
    ch_choice: dict = {}
    d_choice: dict = {}

    def rec(prefix: tuple, F: float, fc: np.ndarray) -> np.ndarray:
        t = len(prefix)
        if t == H:
            return np.zeros((n, D))
        best = np.full((n, D), -np.inf)
        bc = np.zeros((n, D), dtype=np.int8)
        bd = np.zeros((n, D), dtype=np.int16)
        for ci in range(C):
            fc2 = w.decay * fc
            fc2[ci] += fat_add[ci]
            Vn = rec(prefix + (ci,), w.decay * F + fat_add[ci], fc2)
            cont = (1.0 - w.q[t])[:, None] * Vn
            if applies[ci]:
                feff = (1.0 - s) * F + s * fc[ci] if s else F
                pen = w.beta_i * (feff + w.decay ** t * F0_i)
            else:
                pen = np.zeros(n)
            base = w.logits0[t, :, ci, :] - pen[:, None]
            cash = w.cash[t]
            if applies[ci]:
                p0 = _sigmoid(base)
                tot0 = -costs[ci] + p0 * cash + (1.0 - p0) * cont
                cand_v = tot0.copy()
                cand_k = np.tile(np.arange(D, dtype=np.int64), (n, 1))
                for d in range(D - 2, -1, -1):
                    better = cand_v[:, d + 1] > cand_v[:, d]
                    cand_v[better, d] = cand_v[better, d + 1]
                    cand_k[better, d] = cand_k[better, d + 1]
                for m in range(1, D):
                    pen_d = w.gamma * (grid[m] - grid[:m])
                    pp = _sigmoid(base[:, :m] - pen_d[None, :])
                    tt = -costs[ci] + pp * cash[:, :m] + (1.0 - pp) * cont[:, :m]
                    kk = np.argmax(tt, axis=1)
                    vv = tt[rows, kk]
                    better = vv > cand_v[:, m]
                    cand_v[better, m] = vv[better]
                    cand_k[better, m] = kk[better]
            else:
                p = _sigmoid(base)
                imm = -costs[ci] + p * cash
                tt = imm[:, None, :] + (1.0 - p)[:, None, :] * cont[:, :, None]
                cand_k = np.argmax(tt, axis=2)
                cand_v = np.take_along_axis(tt, cand_k[:, :, None], axis=2)[:, :, 0]
            upd = cand_v > best
            best[upd] = cand_v[upd]
            bc[upd] = ci
            bd[upd] = cand_k[upd]
        ch_choice[prefix] = bc
        d_choice[prefix] = bd
        return best

    V_root = rec((), 0.0, np.zeros(C))
    ceiling = V_root[:, 0].copy()
    best_path = np.zeros((n, H), dtype=np.int64)
    best_d = np.zeros((n, H))
    m_cur = np.zeros(n, dtype=np.int64)
    groups: dict = {(): rows}
    for t in range(H):
        new_groups: dict = {}
        for prefix, idx in groups.items():
            cs = ch_choice[prefix][idx, m_cur[idx]]
            ks = d_choice[prefix][idx, m_cur[idx]].astype(np.int64)
            for ci in np.unique(cs):
                ci = int(ci)
                sel = idx[cs == ci]
                ksel = ks[cs == ci]
                best_path[sel, t] = ci
                best_d[sel, t] = grid[ksel]
                if applies[ci]:
                    m_cur[sel] = ksel
                nk = prefix + (ci,)
                new_groups[nk] = (np.concatenate([new_groups[nk], sel])
                                  if nk in new_groups else sel)
        groups = new_groups
    return ceiling, best_path, best_d


def export_answer_key(db_path, config: dict, H: int = 6, *, d_steps: int = 21,
                      loan_ids=None, include_plans: bool = False,
                      start_month: int = 0):
    _refuse_unmodeled(config)
    if start_month and (config["oracle"].get("contact_fatigue") or {}).get("channel_split"):
        raise ValueError("start_month>0 with channel_split is not implemented")
    if start_month and config["oracle"].get("discount_fatigue"):
        raise ValueError("start_month>0 with discount_fatigue is not implemented "
                         "(the entering last_discount_seen state is not loaded)")
    book = _book_at_month(db_path, start_month, loan_ids)
    F0_i = (book["entering_fatigue"].to_numpy(float)
            if "entering_fatigue" in book.columns else np.zeros(len(book)))
    w = _World(book, config, H, d_steps)
    n = len(book)
    C = len(w.channels)
    applies = np.array([c != "none" for c in w.channels])
    fat_add = np.array([w.fat_add[c] for c in w.channels])
    costs = np.array([w.costs[c] for c in w.channels])

    best_V = np.full(n, -np.inf)
    best_path = np.zeros((n, H), dtype=np.int64)
    best_d = np.zeros((n, H))
    s_split = w.split_share
    if w.gamma:
        if w.man is not None:
            raise ValueError("discount_fatigue with a manifold world is not "
                             "implemented in the exact key")
        best_V, best_path, best_d = _memory_tree_dp(w, F0_i, applies, fat_add, costs, H)
    for path in ([] if w.gamma else itertools.product(range(C), repeat=H)):
        F = np.empty(H); Fown = np.empty(H); f = 0.0
        fc = np.zeros(C)
        for t, ci in enumerate(path):
            F[t] = f
            Fown[t] = fc[ci]
            f = w.decay * f + fat_add[ci]
            fc = w.decay * fc
            fc[ci] += fat_add[ci]
        Feff = (1.0 - s_split) * F + s_split * Fown if s_split else F
        bil = w.man is not None and w.man["rot"] is not None
        if bil:
            d_ = w.man["n_lat"]
            Mt = np.eye(d_)
            mt = np.zeros(d_)
            proj = np.empty((H, len(w.book)))
            for t, ci in enumerate(path):
                v = w.man["W"][ci]
                proj[t] = w.man["coef"] * (
                    w.man["z0"] @ (Mt.T @ v) + float(v @ mt))
                wv = w.man["w_int"][ci]
                T = w.man["A"] @ w.man["rot"][wv]
                mt = T @ mt + w.man["B"] * wv
                Mt = T @ Mt
        elif w.man is not None:
            corr = np.empty(H)
            M = np.zeros(w.man["n_lat"])
            for t, ci in enumerate(path):
                corr[t] = w.man["coef"] * float(w.man["W"][ci] @ M)
                M = w.man["A"] @ M + w.man["B"] * w.man["w_int"][ci]
        V = np.zeros(n)
        d_pick = np.zeros((n, H))
        for t in range(H - 1, -1, -1):
            ci = path[t]
            pen = (w.beta_i * (Feff[t] + w.decay ** t * F0_i)
                   if applies[ci] else np.zeros(n))
            if bil:
                pen = pen - proj[t]
            elif w.man is not None:
                pen = pen - (w.man_drift[t][:, ci] + corr[t])
            p = _sigmoid(w.logits0[t, :, ci, :] - pen[:, None])
            tot = -costs[ci] + p * w.cash[t] + ((1.0 - p)
                  * ((1.0 - w.q[t]) * V)[:, None])
            k = np.argmax(tot, axis=1)
            V = tot[np.arange(n), k]
            d_pick[:, t] = w.d_grid[k]
        upd = V > best_V
        if upd.any():
            best_V[upd] = V[upd]
            best_path[upd] = np.asarray(path)
            best_d[upd] = d_pick[upd]

    f_vec, S, myo = F0_i.copy(), np.ones(n), np.zeros(n)
    fch = np.zeros((C, n))
    dseen = np.zeros(n)
    if w.man is not None:
        zc = [c for c in book.columns if c.startswith("zman_")]
        z_vec = book[zc].to_numpy(float)
    for t in range(H):
        if w.man is not None:
            man_adj = w.man["coef"] * (z_vec @ w.man["W"].T)
        best_imm = np.full(n, -np.inf); bp = np.zeros(n); bci = np.zeros(n, int)
        bd = np.zeros(n)
        for ci in range(C):
            f_eff = ((1.0 - s_split) * f_vec + s_split * fch[ci]) if s_split else f_vec
            pen = (w.beta_i * f_eff) if applies[ci] else np.zeros(n)
            if w.man is not None:
                pen = pen - man_adj[:, ci]
            lg = w.logits0[t, :, ci, :] - pen[:, None]
            if w.gamma and applies[ci]:
                lg = lg - w.gamma * np.maximum(0.0, dseen[:, None] - w.d_grid[None, :])
            p = _sigmoid(lg)
            imm = p * w.cash[t] - costs[ci]
            k = np.argmax(imm, axis=1)
            v = imm[np.arange(n), k]
            upd = v > best_imm
            best_imm[upd] = v[upd]; bp[upd] = p[np.arange(n), k][upd]; bci[upd] = ci
            bd[upd] = w.d_grid[k][upd]
        myo += S * best_imm
        S *= (1.0 - bp) * (1.0 - w.q[t])
        f_vec = w.decay * f_vec + fat_add[bci]
        if w.gamma:
            dseen = np.where(applies[bci], bd, dseen)
        if s_split:
            fch = w.decay * fch
            fch[bci, np.arange(n)] += fat_add[bci]
        if w.man is not None:
            wv = w.man["w_int"][bci]
            if w.man["rot"] is not None:
                z_rot = np.empty_like(z_vec)
                for w_val, R in w.man["rot"].items():
                    m_ = wv == w_val
                    if m_.any():
                        z_rot[m_] = z_vec[m_] @ R.T
                z_vec = z_rot @ w.man["A"].T + np.outer(wv, w.man["B"])
            else:
                z_vec = z_vec @ w.man["A"].T + np.outer(wv, w.man["B"])

    key = pd.DataFrame({"loan_id": book["loan_id"], "ceiling": best_V,
                        "myopic_bound": myo})
    if not include_plans:
        return key
    plans = pd.DataFrame({
        "loan_id": np.repeat(book["loan_id"].to_numpy(), H),
        "month": np.tile(np.arange(H), n),
        "channel": [w.channels[c] for c in best_path.ravel()],
        "discount_offer": best_d.ravel(),
    })
    return key, plans


def grade_plan(plan_df: pd.DataFrame, db_path, config: dict, H: int = 6,
               start_month: int = 0) -> pd.DataFrame:
    _refuse_unmodeled(config)
    if start_month and (config["oracle"].get("contact_fatigue") or {}).get("channel_split"):
        raise ValueError("start_month>0 with channel_split is not implemented")
    if start_month and config["oracle"].get("discount_fatigue"):
        raise ValueError("start_month>0 with discount_fatigue is not implemented "
                         "(the entering last_discount_seen state is not loaded)")
    oc = config["oracle"]
    fat = oc.get("contact_fatigue") or {}
    split = bool(fat.get("channel_split"))
    man_cfg = oc.get("manifold")
    if man_cfg:
        (n_lat, _, _, man_A, man_B, _, _,
         man_rot) = oracle_response._manifold(man_cfg)
        man_human = set(config["actions"].get("human_channels") or [])
    human = set(config["actions"].get("human_channels") or [])
    ch_costs = config["actions"].get("channel_costs") or {}
    pshare = paid_share_mean(oc)
    book = _book_at_month(db_path, start_month,
                          plan_df["loan_id"].unique()).set_index("loan_id")
    plans = plan_df.sort_values(["loan_id", "month"]).groupby("loan_id")

    out = []
    for lid, g in plans:
        if lid not in book.index:
            continue
        b = book.loc[lid]
        beta_i = b["fatigue_beta"]
        beta_i = None if pd.isna(beta_i) else float(beta_i)
        f = float(b["entering_fatigue"]) if "entering_fatigue" in b.index else 0.0
        S, val = 1.0, 0.0
        f_ch: dict[str, float] = {}
        dseen = 0.0
        if man_cfg:
            zc = [c for c in book.columns if c.startswith("zman_")]
            z_man = book.loc[lid, zc].to_numpy(dtype=float)
        dpd = float(b["days_past_due"]); liq = float(b["days_in_credito_em_liquidacao"])
        bal = float(b["outstanding_balance"]); la = int(b["loan_age_months"])
        for _, row in g.head(H).iterrows():
            c, d = row["channel"], float(row["discount_offer"])
            p = recovery_probability(RecoveryInputs(
                channel=c, discount_offer=d, age=int(b["age"]),
                loan_age_months=la, days_past_due=int(dpd),
                days_in_credito_em_liquidacao=int(liq),
                borrower_resilience=b["borrower_resilience"],
                payment_friction=b["payment_friction"],
                regional_macro_stress=b["regional_macro_stress"],
                contactability=b["contactability"], contact_fatigue=f,
                manifold_z=(tuple(z_man) if man_cfg else ()),
                contact_fatigue_own=(f_ch.get(c, 0.0) if split else None),
                last_discount_seen=dseen,
                fatigue_beta=beta_i, owed_amount=bal,
                monthly_income=b["monthly_income"],
                original_principal=b["original_principal"], noise=0.0), oc)
            q = _chargeoff_probability(int(liq), oc)
            val += S * (p * bal * (1.0 - d) * pshare - float(ch_costs.get(c, 0.0)))
            S *= (1.0 - p) * (1.0 - q)
            bal *= 1.0 + float(b["interest_rate_monthly"]) * 0.35
            dpd += 30.0
            if dpd >= 90.0:
                liq += 30.0
            la += 1
            add = 0.0 if c == "none" else float(fat.get("increment", 1.0)) * (
                float(fat.get("human_multiplier", 1.0)) if c in human else 1.0)
            if man_cfg:
                w_i = 0.0 if c == "none" else (1.0 if c in man_human else 0.4)
                if man_rot is not None:
                    z_man = man_A @ (man_rot[w_i] @ z_man) + man_B * w_i
                else:
                    z_man = man_A @ z_man + man_B * w_i
            dec = float(fat.get("decay", 0.5))
            f = dec * f + add
            if split:
                for cc in list(f_ch):
                    f_ch[cc] = dec * f_ch[cc]
                if c != "none":
                    f_ch[c] = f_ch.get(c, 0.0) + add
            if c != "none":
                dseen = d
        out.append((lid, val))
    return pd.DataFrame(out, columns=["loan_id", "value"])
