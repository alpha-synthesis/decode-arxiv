from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import os
import sys


CHANNELS = ["none", "sms", "whatsapp", "phone"]
STATE = ["days_past_due", "days_in_credito_em_liquidacao", "outstanding_balance",
         "contact_fatigue", "loan_age_months", "age", "monthly_income",
         "original_principal", "interest_rate_monthly"]
D_STEPS = 21
HID = 128


def load_frame(db: Path):
    con = duckdb.connect(str(db), read_only=True)
    obs_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()]
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
    return df, obs_cols


def pack(df, obs_cols, months):
    loans = df["loan_id"].unique()
    lidx = {l: i for i, l in enumerate(loans)}
    n, T, S, O = len(loans), months, len(STATE), len(obs_cols)
    st = np.zeros((n, T, S), np.float32)
    ob = np.zeros((n, T, O), np.float32)
    ch = np.zeros((n, T), np.int64)
    dc = np.zeros((n, T), np.float32)
    valid = np.zeros((n, T), bool)
    expl = np.zeros((n, T), bool)
    y = np.zeros((n, T), np.float32)
    r = df[df.month < months]
    i = r["loan_id"].map(lidx).to_numpy()
    t = r["month"].to_numpy()
    st[i, t] = r[STATE].to_numpy(np.float32)
    ob[i, t] = r[obs_cols].to_numpy(np.float32)
    ch[i, t] = r["channel"].map({c: k for k, c in enumerate(CHANNELS)}).to_numpy()
    dc[i, t] = r["discount_offer"].to_numpy(np.float32)
    valid[i, t] = True
    expl[i, t] = r["is_exploration"].to_numpy(bool)
    y[i, t] = r["recovered"].to_numpy(np.float32)
    return dict(loans=loans, state=st, obs=ob, ch=ch, dc=dc,
                valid=valid, expl=expl, y=y)


class Norm:

    def __init__(self, st, ob, valid):
        v = valid.reshape(-1)
        s = st.reshape(-1, st.shape[-1])[v]
        o = ob.reshape(-1, ob.shape[-1])[v]
        self.sc = (np.median(s, 0),
                   np.clip(np.percentile(s, 75, 0) - np.percentile(s, 25, 0),
                           1e-6, None))
        self.oc = (np.median(o, 0),
                   np.clip(np.percentile(o, 75, 0) - np.percentile(o, 25, 0),
                           1e-6, None))

    def state(self, x):
        return (x - self.sc[0]) / self.sc[1]

    def obs(self, x):
        return (x - self.oc[0]) / self.oc[1]


class BeliefModel(nn.Module):
    def __init__(self, s_dim, o_dim, hid=HID):
        super().__init__()
        in_dim = s_dim + o_dim + 1 + len(CHANNELS)
        self.cell = nn.GRUCell(in_dim, hid)
        head_in = hid + s_dim + o_dim + 1
        self.pay = nn.Sequential(
            nn.Linear(head_in + len(CHANNELS) + 1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.org = nn.Sequential(
            nn.Linear(head_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.hid = hid

    def step(self, b, s, o, om, ci_onehot):
        x = torch.cat([s, o, om, ci_onehot], -1)
        return self.cell(x, b)

    def p_pay_logit(self, b, s, o, om, ci_onehot, d):
        return self.pay(torch.cat([b, s, o, om, ci_onehot, d], -1)).squeeze(-1)

    def p_org_logit(self, b, s, o, om):
        return self.org(torch.cat([b, s, o, om], -1)).squeeze(-1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, obs_cols = load_frame(args.train_world / "sim.duckdb")
    P = pack(df, obs_cols, args.months)
    norm = Norm(P["state"], P["obs"], P["valid"])
    n, T = P["valid"].shape
    st = torch.from_numpy(norm.state(P["state"]).astype(np.float32))
    ob = torch.from_numpy(norm.obs(P["obs"]).astype(np.float32))
    ch = torch.from_numpy(P["ch"])
    dc = torch.from_numpy(P["dc"])
    valid = torch.from_numpy(P["valid"])
    expl = torch.from_numpy(P["expl"])
    yv = torch.from_numpy(P["y"])
    none_i = CHANNELS.index("none")
    is_none = (ch == none_i) & valid
    is_con = expl & (ch != none_i) & valid
    pos = float(yv[is_con].sum()); neg = float(is_con.sum()) - pos
    pw = torch.tensor(min(neg / max(pos, 1.0), 50.0), device=device)
    posn = float(yv[is_none].sum()); negn = float(is_none.sum()) - posn
    pwn = torch.tensor(min(negn / max(posn, 1.0), 50.0), device=device)
    print(f"loans={n} T={T} contact-explore rows={int(is_con.sum())} "
          f"none rows={int(is_none.sum())}", flush=True)

    torch.manual_seed(args.seed)
    model = BeliefModel(len(STATE), len(obs_cols), hid=args.hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    eye = torch.eye(len(CHANNELS), device=device)
    holdout = torch.arange(n) % 10 == 0
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for s0 in range(0, n, args.batch):
            idx = perm[s0:s0 + args.batch]
            idx = idx[~holdout[idx]]
            if not len(idx):
                continue
            b = torch.zeros(len(idx), model.hid, device=device)
            loss = 0.0
            nl = 0
            if args.mask_mode == "suffix":
                dark = torch.rand(len(idx), device=device) < args.obs_dropout
                cut = torch.where(dark,
                                  torch.randint(1, T, (len(idx),), device=device),
                                  torch.full((len(idx),), T, device=device))
            else:
                cut = None
            S_ = st[idx].to(device); O_ = ob[idx].to(device)
            C_ = ch[idx].to(device); D_ = dc[idx].to(device)
            V_ = valid[idx].to(device); E_ = expl[idx].to(device)
            Y_ = yv[idx].to(device)
            for t in range(T):
                sc_t, oc_t = S_[:, t], O_[:, t]
                con_m = E_[:, t] & (C_[:, t] != none_i) & V_[:, t]
                non_m = (C_[:, t] == none_i) & V_[:, t]
                if cut is not None:
                    om = (t < cut).float().unsqueeze(1) * V_[:, t:t + 1].float()
                else:
                    om = (torch.rand(len(idx), 1, device=device)
                          > args.obs_dropout).float() * V_[:, t:t + 1].float()
                oc_m = oc_t * om
                oc_m = oc_t * om
                if con_m.any():
                    lg = model.p_pay_logit(b[con_m], sc_t[con_m],
                                           oc_m[con_m], om[con_m],
                                           eye[C_[con_m, t]],
                                           D_[con_m, t:t + 1])
                    loss = loss + (bce(lg, Y_[con_m, t])
                                   * torch.where(Y_[con_m, t] > 0, pw, 1.0)).sum()
                    nl += int(con_m.sum())
                if non_m.any():
                    lg = model.p_org_logit(b[non_m], sc_t[non_m],
                                           oc_m[non_m], om[non_m])
                    loss = loss + (bce(lg, Y_[non_m, t])
                                   * torch.where(Y_[non_m, t] > 0, pwn, 1.0)).sum()
                    nl += int(non_m.sum())
                b_new = model.step(b, sc_t, oc_m, om, eye[C_[:, t]])
                b = torch.where(V_[:, t:t + 1], b_new, b)
            if nl:
                (loss / nl).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                tot += float(loss); cnt += nl
        model.eval()
        with torch.no_grad():
            hidx = torch.where(holdout)[0][:20000]
            b = torch.zeros(len(hidx), model.hid, device=device)
            vloss, vn = 0.0, 0
            S_ = st[hidx].to(device); O_ = ob[hidx].to(device)
            C_ = ch[hidx].to(device); D_ = dc[hidx].to(device)
            V_ = valid[hidx].to(device); E_ = expl[hidx].to(device)
            Y_ = yv[hidx].to(device)
            for t in range(T):
                con_m = E_[:, t] & (C_[:, t] != none_i) & V_[:, t]
                om = V_[:, t:t + 1].float()
                if con_m.any():
                    lg = model.p_pay_logit(b[con_m], S_[con_m, t],
                                           O_[con_m, t] * om[con_m], om[con_m],
                                           eye[C_[con_m, t]], D_[con_m, t:t + 1])
                    vloss += float(bce(lg, Y_[con_m, t]).sum()); vn += int(con_m.sum())
                b_new = model.step(b, S_[:, t], O_[:, t] * om, om, eye[C_[:, t]])
                b = torch.where(V_[:, t:t + 1], b_new, b)
            v = vloss / max(vn, 1)
        print(f"[belief] epoch {ep + 1}/{args.epochs} train={tot / max(cnt, 1):.4f} "
              f"val={v:.4f}", flush=True)
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.detach().cpu().clone()
                                        for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    from sklearn.isotonic import IsotonicRegression
    with torch.no_grad():
        hidx = torch.where(holdout)[0]
        b = torch.zeros(len(hidx), model.hid, device=device)
        rows_c, rows_n = [], []
        S_ = st[hidx].to(device); O_ = ob[hidx].to(device)
        C_ = ch[hidx].to(device); D_ = dc[hidx].to(device)
        V_ = valid[hidx].to(device); E_ = expl[hidx].to(device)
        Y_ = yv[hidx].to(device)
        for t in range(T):
            con_m = E_[:, t] & (C_[:, t] != none_i) & V_[:, t]
            non_m = (C_[:, t] == none_i) & V_[:, t]
            om = V_[:, t:t + 1].float()
            if con_m.any():
                lg = model.p_pay_logit(b[con_m], S_[con_m, t],
                                       O_[con_m, t] * om[con_m], om[con_m],
                                       eye[C_[con_m, t]], D_[con_m, t:t + 1])
                rows_c.append((torch.sigmoid(lg).cpu().numpy(),
                               Y_[con_m, t].cpu().numpy()))
            if non_m.any():
                lg = model.p_org_logit(b[non_m], S_[non_m, t],
                                       O_[non_m, t] * om[non_m], om[non_m])
                rows_n.append((torch.sigmoid(lg).cpu().numpy(),
                               Y_[non_m, t].cpu().numpy()))
            b = torch.where(V_[:, t:t + 1],
                            model.step(b, S_[:, t], O_[:, t] * om, om, eye[C_[:, t]]), b)
    iso_c = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    pc = np.concatenate([a for a, _ in rows_c]); yc = np.concatenate([b_ for _, b_ in rows_c])
    iso_c.fit(pc, yc)
    iso_n = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    pn = np.concatenate([a for a, _ in rows_n]); yn = np.concatenate([b_ for _, b_ in rows_n])
    iso_n.fit(pn, yn)
    out = args.out / "belief_v7.pt"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hid": model.hid,
                "s_dim": len(STATE), "o_dim": len(obs_cols),
                "norm": {"sc": norm.sc, "oc": norm.oc},
                "iso_c": iso_c, "iso_n": iso_n,
                "obs_cols": obs_cols, "months": args.months}, out)
    print(f"saved {out}", flush=True)


def _load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = BeliefModel(ck["s_dim"], ck["o_dim"], ck["hid"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def encode_prehistory(model, ck, df, obs_cols, k, device):
    P = pack(df, obs_cols, k)
    sc, oc = ck["norm"]["sc"], ck["norm"]["oc"]
    st = torch.from_numpy(((P["state"] - sc[0]) / sc[1]).astype(np.float32)).to(device)
    ob = torch.from_numpy(((P["obs"] - oc[0]) / oc[1]).astype(np.float32)).to(device)
    ch = torch.from_numpy(P["ch"]).to(device)
    valid = torch.from_numpy(P["valid"]).to(device)
    eye = torch.eye(len(CHANNELS), device=device)
    b = torch.zeros(len(P["loans"]), model.hid, device=device)
    with torch.no_grad():
        for t in range(k):
            om = valid[:, t:t + 1].float()
            b_new = model.step(b, st[:, t], ob[:, t] * om, om, eye[ch[:, t]])
            b = torch.where(valid[:, t:t + 1], b_new, b)
    return P["loans"], b


def probe(args):
    from decode_sim.config import load_config
    from decode_sim.trajectory_key import _book_at_month, _World
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = _load_model(args.model, device)
    df, obs_cols = load_frame(args.eval_world / "sim.duckdb")
    k = args.k
    book = _book_at_month(args.eval_world / "sim.duckdb", k)
    keep = sorted(book["loan_id"])[:args.loans]
    dfe = df[df.loan_id.isin(keep)]
    loans, b = encode_prehistory(model, ck, dfe, obs_cols, k, device)
    order = {l: i for i, l in enumerate(loans)}
    book = book[book.loan_id.isin(set(loans))].sort_values("loan_id").reset_index(drop=True)
    bidx = np.array([order[l] for l in book["loan_id"]])
    B = b.cpu().numpy()[bidx]
    cfg = load_config(str(args.config))
    w = _World(book, cfg, 1)
    pi = list(w.channels).index("phone")
    y = 1 / (1 + np.exp(-w.logits0[0, :, pi, 8]))
    row_k = dfe[dfe.month == k].sort_values("loan_id").reset_index(drop=True)
    row_k = row_k[row_k.loan_id.isin(set(book.loan_id))]
    flat = row_k[STATE + obs_cols].to_numpy(float)
    for name, X in (("flat(row@k)", flat), ("belief(b_k)", B),
                    ("flat+belief", np.concatenate([flat, B], 1))):
        r2 = cross_val_score(HistGradientBoostingRegressor(max_iter=300, random_state=0),
                             X, y, cv=3, scoring="r2")
        print(f"{name:14s} R2 = {r2.mean():.4f} (+/- {r2.std():.4f})", flush=True)


def _cheat_rolled_sources(args, ev, dfe, device):
    from decode_sim import oracle_response as orsp
    from decode_sim.config import load_config
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    sim_cfg = load_config(str(args.config))
    man = sim_cfg["oracle"]["manifold"]
    n_lat, _, _, A, B, _, _, _rot = orsp._manifold(man)
    if _rot is not None:
        raise ValueError("cheat-rolled-z assumes the linear transition; "
                         "bilinear worlds need the path-matrix roll")
    human = set(sim_cfg["actions"]["human_channels"])
    w_int = np.array([0.0 if c == "none" else (1.0 if c in human else 0.4)
                      for c in CHANNELS])
    con = duckdb.connect(str(args.train_world / "sim.duckdb"), read_only=True)
    zc = sorted((r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE "
        "table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    tr = con.execute(f"""
        SELECT COALESCE(pe.days_past_due, l.days_past_due) AS days_past_due,
               COALESCE(pe.days_in_credito_em_liquidacao,
                        l.days_in_credito_em_liquidacao) AS days_in_credito_em_liquidacao,
               COALESCE(pe.outstanding_balance, l.outstanding_balance) AS outstanding_balance,
               p.contact_fatigue,
               l.loan_age_months + p.month AS loan_age_months,
               b.age, b.monthly_income, l.original_principal,
               l.interest_rate_monthly,
               {', '.join('t.' + c for c in zc)},
               a.channel, a.discount_offer, a.is_exploration, p.recovered
        FROM monthly_panel p
        LEFT JOIN monthly_panel pe ON pe.loan_id = p.loan_id AND pe.month = p.month - 1
        JOIN actions a ON a.action_id = p.action_id
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        JOIN oracle_truth t ON t.loan_id = p.loan_id AND t.month = p.month
    """).df()
    con.close()
    feats = STATE + zc
    for c in CHANNELS[1:]:
        tr[f"ch_{c}"] = (tr["channel"] == c).astype(float)
    xcols = feats + ["discount_offer"] + [f"ch_{c}" for c in CHANNELS[1:]]
    trc = tr[tr.is_exploration].sample(frac=1.0, random_state=0)
    Xc, yc = trc[xcols].to_numpy(float), trc["recovered"].to_numpy(int)
    nv = max(500, int(0.15 * len(trc)))
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=63, early_stopping=True,
                                         random_state=0)
    gbm.fit(Xc[nv:], yc[nv:])
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(gbm.predict_proba(Xc[:nv])[:, 1], yc[:nv])
    trn = tr[(tr.channel == "none") & (~tr.is_exploration)].sample(frac=1.0, random_state=0)
    Xn, yn = trn[feats].to_numpy(float), trn["recovered"].to_numpy(int)
    nvn = max(500, int(0.15 * len(trn)))
    gbm_n = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=63, early_stopping=True,
                                           random_state=0)
    gbm_n.fit(Xn[nvn:], yn[nvn:])
    iso_n = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso_n.fit(gbm_n.predict_proba(Xn[:nvn])[:, 1], yn[:nvn])
    cone = duckdb.connect(str(args.eval_world / "sim.duckdb"), read_only=True)
    ze = cone.execute(f"""SELECT loan_id, {', '.join(zc)} FROM oracle_truth
                          WHERE month = {args.k}""").df()
    cone.close()
    ze = ze.set_index("loan_id").loc[ev["loan_id"]].to_numpy(float)
    print(f"cheat-rolled-z: explore rows {len(trc)}, none rows {len(trn)}",
          flush=True)
    return dict(A=A, B=B, w_int=w_int, z_k=ze, gbm=gbm, iso=iso,
                gbm_n=gbm_n, iso_n=iso_n)


def board(args):
    from decode_sim.config import load_config
    from decode_sim.generator import _chargeoff_probability
    from decode_sim.trajectory_key import (
        _book_at_month, export_answer_key, grade_plan, paid_share_mean)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = _load_model(args.model, device)
    sim_cfg = load_config(str(args.config))
    db = args.eval_world / "sim.duckdb"
    k, H = args.k, args.horizon
    df, obs_cols = load_frame(db)
    book = _book_at_month(db, k)
    keep = sorted(book["loan_id"])[:args.loans]
    dfe = df[df.loan_id.isin(keep)]
    loans, b0 = encode_prehistory(model, ck, dfe, obs_cols, k, device)
    order = {l: i for i, l in enumerate(loans)}
    ev = dfe[dfe.month == k].sort_values("loan_id").reset_index(drop=True)
    ev = ev[ev.loan_id.isin(set(loans))].reset_index(drop=True)
    bidx = torch.tensor([order[l] for l in ev["loan_id"]], device=device)
    b0 = b0[bidx]
    n = len(ev)
    print(f"eval loans: {n}", flush=True)

    oc = sim_cfg["oracle"]["contact_fatigue"]
    human = set(sim_cfg["actions"]["human_channels"])
    costs = np.array([float((sim_cfg["actions"].get("channel_costs") or {})
                            .get(c, 0.0)) for c in CHANNELS])
    fat_add = np.array([0.0 if c == "none" else float(oc["increment"])
                        * (float(oc["human_multiplier"]) if c in human else 1.0)
                        for c in CHANNELS])
    decay = float(oc["decay"])
    d_grid = np.linspace(0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
    F0 = ev["contact_fatigue"].to_numpy(float)
    dpd = ev["days_past_due"].to_numpy(float).copy()
    liq = ev["days_in_credito_em_liquidacao"].to_numpy(float).copy()
    bal = ev["outstanding_balance"].to_numpy(float).copy()
    rate = ev["interest_rate_monthly"].to_numpy(float)
    la = ev["loan_age_months"].to_numpy(float).copy()
    pshare = paid_share_mean(sim_cfg["oracle"])
    sc = ck["norm"]["sc"]
    base0 = ev[STATE].to_numpy(float)
    si = {f: STATE.index(f) for f in STATE}
    state_t, bal_t, q_t = [], [], []
    d_, l_, b_, a_ = dpd.copy(), liq.copy(), bal.copy(), la.copy()
    for t in range(H):
        q_t.append(np.array([_chargeoff_probability(int(v), sim_cfg["oracle"])
                             for v in l_]))
        bal_t.append(b_.copy())
        s = base0.copy()
        s[:, si["days_past_due"]] = d_
        s[:, si["days_in_credito_em_liquidacao"]] = l_
        s[:, si["outstanding_balance"]] = b_
        s[:, si["loan_age_months"]] = a_
        state_t.append(s)
        b_ = b_ * (1.0 + rate * 0.35)
        d_ = d_ + 30.0
        l_ = np.where(d_ >= 90.0, l_ + 30.0, l_)
        a_ = a_ + 1
    q_t = np.stack(q_t)
    bal_t = np.stack(bal_t)

    eye = torch.eye(len(CHANNELS), device=device)
    o_zero = torch.zeros(n, ck["o_dim"], device=device)
    m_zero = torch.zeros(n, 1, device=device)
    ocn = ck["norm"]["oc"]
    obs_k = torch.tensor(((ev[obs_cols].to_numpy(float) - ocn[0]) / ocn[1])
                         .astype(np.float32), device=device)
    m_one = torch.ones(n, 1, device=device)
    iso_c, iso_n = ck["iso_c"], ck["iso_n"]
    dgrid_t = torch.tensor(d_grid, dtype=torch.float32, device=device)

    def norm_state(t, F):
        s = state_t[t].copy()
        s[:, si["contact_fatigue"]] = F
        return torch.tensor(((s - sc[0]) / sc[1]).astype(np.float32), device=device)

    cheat = None
    if getattr(args, "cheat_rolled_z", False):
        cheat = _cheat_rolled_sources(args, ev, dfe, device)
        z_cache: dict[tuple, np.ndarray] = {(): cheat["z_k"]}

        def z_at(prefix):
            if prefix in z_cache:
                return z_cache[prefix]
            zp = z_at(prefix[:-1])
            z = zp @ cheat["A"].T + np.outer(
                np.full(len(zp), cheat["w_int"][prefix[-1]]), cheat["B"])
            z_cache[prefix] = z
            return z

    belief_cache: dict[tuple, torch.Tensor] = {}

    def belief_at(prefix):
        if prefix in belief_cache:
            return belief_cache[prefix]
        if not prefix:
            belief_cache[()] = b0
            return b0
        b_prev = belief_at(prefix[:-1])
        t = len(prefix) - 1
        F = _fatigue_along(prefix[:-1])
        s = norm_state(t, F)
        o_t, m_t = (obs_k, m_one) if t == 0 else (o_zero, m_zero)
        with torch.no_grad():
            b_new = model.step(b_prev, s, o_t, m_t, eye[prefix[-1]].expand(n, -1))
        belief_cache[prefix] = b_new
        return b_new

    def _fatigue_along(prefix):
        F = F0.copy()
        for ci in prefix:
            F = decay * F + fat_add[ci]
        return F

    p_cache: dict[tuple, np.ndarray] = {}

    def p_cells(prefix, ci):
        keyc = (prefix, ci)
        if keyc in p_cache:
            return p_cache[keyc]
        t = len(prefix)
        b = belief_at(prefix)
        F = _fatigue_along(prefix)
        s = norm_state(t, F)
        if cheat is not None:
            zz = z_at(prefix)
            F = _fatigue_along(prefix)
            sraw = state_t[t].copy()
            sraw[:, si["contact_fatigue"]] = F
            base = np.concatenate([sraw, zz], 1)
            if ci == 0:
                p = np.asarray(cheat["iso_n"].predict(
                    cheat["gbm_n"].predict_proba(base)[:, 1]))
                out = np.repeat(p[:, None], D_STEPS, 1)
            else:
                cols = []
                oneh = np.zeros((n, len(CHANNELS) - 1)); oneh[:, ci - 1] = 1.0
                for d in d_grid:
                    x = np.concatenate([base, np.full((n, 1), d), oneh], 1)
                    cols.append(cheat["gbm"].predict_proba(x)[:, 1])
                p = np.stack(cols, 1)
                out = cheat["iso"].predict(p.ravel()).reshape(p.shape)
            p_cache[keyc] = out
            return out
        o_t, m_t = (obs_k, m_one) if t == 0 else (o_zero, m_zero)
        with torch.no_grad():
            if ci == 0:
                p = torch.sigmoid(model.p_org_logit(b, s, o_t, m_t)).cpu().numpy()
                p = np.asarray(iso_n.predict(p))
                out = np.repeat(p[:, None], D_STEPS, 1)
            else:
                ch1 = eye[ci].expand(n, -1)
                cols = []
                for d in dgrid_t:
                    lg = model.p_pay_logit(b, s, o_t, m_t, ch1, d.expand(n, 1))
                    cols.append(torch.sigmoid(lg).cpu().numpy())
                p = np.stack(cols, 1)
                out = iso_c.predict(p.ravel()).reshape(p.shape)
        p_cache[keyc] = out
        return out

    best_V = np.full(n, -np.inf)
    bp = np.zeros((n, H), dtype=np.int64)
    bd = np.zeros((n, H), dtype=np.int64)
    C = len(CHANNELS)
    for path in itertools.product(range(C), repeat=H):
        V = np.zeros(n)
        d_pick = np.zeros((n, H), dtype=np.int64)
        for t in range(H - 1, -1, -1):
            p = p_cells(tuple(path[:t]), path[t])
            cash = bal_t[t][:, None] * (1.0 - d_grid)[None, :] * pshare
            tot = -costs[path[t]] + p * cash + ((1.0 - p)
                  * ((1.0 - q_t[t]) * V)[:, None])
            kk = np.argmax(tot, axis=1)
            V = tot[np.arange(n), kk]
            d_pick[:, t] = kk
        upd = V > best_V
        if upd.any():
            best_V[upd] = V[upd]
            bp[upd] = np.asarray(path)
            bd[upd] = d_pick[upd]
    plan = pd.DataFrame({
        "loan_id": np.tile(ev["loan_id"].to_numpy(), H),
        "month": np.repeat(np.arange(H), n),
        "channel": [CHANNELS[c] for c in bp.T.ravel()],
        "discount_offer": d_grid[bd.T.ravel()],
    })
    key = export_answer_key(db, sim_cfg, H=H, loan_ids=list(ev["loan_id"]),
                            start_month=k)
    g = grade_plan(plan, db, sim_cfg, H=H, start_month=k)
    m = key.merge(g, on="loan_id")
    assert len(m) == len(key)
    ceiling = key["ceiling"].mean()
    m[["loan_id", "ceiling", "myopic_bound", "value"]].to_csv(
        args.out / "per_loan_values.csv", index=False)
    out = {"n_loans": int(len(m)), "k": k, "H": H,
           "ceiling_per_loan": round(float(ceiling), 2),
           "myopic_bound_pct": round(100 * float(key["myopic_bound"].mean() / ceiling), 2),
           "belief_sequential_pct": round(100 * float(m["value"].mean() / ceiling), 2)}
    print(json.dumps(out, indent=2))
    (args.out / "board_belief.json").write_text(json.dumps(out, indent=2) + "\n")
    plan.to_csv(args.out / "plan_belief_sequential.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train")
    pt.add_argument("--train-world", type=Path, required=True)
    pt.add_argument("--months", type=int, default=24)
    pt.add_argument("--epochs", type=int, default=8)
    pt.add_argument("--batch", type=int, default=4096)
    pt.add_argument("--obs-dropout", type=float, default=0.3)
    pt.add_argument("--mask-mode", choices=["iid", "suffix"], default="iid",
                    help="iid = readings masked independently (the shipped checkpoint); "
                         "suffix = a contiguous masked suffix")
    pt.add_argument("--hid", type=int, default=64)
    pt.add_argument("--patience", type=int, default=3)
    pt.add_argument("--seed", type=int, default=777)
    pt.add_argument("--out", type=Path, required=True)
    pp = sub.add_parser("probe")
    pp.add_argument("--model", type=Path, required=True)
    pp.add_argument("--eval-world", type=Path, required=True)
    pp.add_argument("--config", type=Path, required=True)
    pp.add_argument("--k", type=int, default=6)
    pp.add_argument("--loans", type=int, default=4000)
    pb = sub.add_parser("board")
    pb.add_argument("--model", type=Path, required=True)
    pb.add_argument("--eval-world", type=Path, required=True)
    pb.add_argument("--train-world", type=Path, default=None,
                    help="required with --cheat-rolled-z")
    pb.add_argument("--config", type=Path, required=True)
    pb.add_argument("--k", type=int, default=6)
    pb.add_argument("--horizon", type=int, default=6)
    pb.add_argument("--loans", type=int, default=4000)
    pb.add_argument("--out", type=Path, required=True)
    pb.add_argument("--cheat-rolled-z", action="store_true",
                    help="UPPER REFERENCE for rolling engines: replace the "
                         "learned belief with the TRUE hidden z, rolled "
                         "exactly along each path (z' = Az + B*w(channel)); "
                         "p from a GBM fit on (z, state, action). Never a "
                         "band-bearing row.")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "probe":
        probe(args)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        board(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
