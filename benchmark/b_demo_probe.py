from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


from benchmark.exact_filter import RotationBelief
from decode_sim.bridge.entering_state import (
    DEFAULT_STATE_TIMING, STATE_TIMINGS, assert_panel_aligned, state_select_sql)

FLAT = ["days_past_due", "days_in_credito_em_liquidacao", "outstanding_balance",
        "contacts_last_3m", "contact_fatigue", "loan_age_months",
        "remaining_maturity_months", "age", "monthly_income", "dependents",
        "original_principal", "interest_rate_monthly"]
CH_CODE = {"none": 0.0, "sms": 1.0, "whatsapp": 2.0, "phone": 3.0}


def load_panel(world: Path, state_timing: str = DEFAULT_STATE_TIMING):
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    obs_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()]
    zman_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()]
    assert obs_cols and zman_cols, "not a manifold world"
    state_sel, state_join = state_select_sql(state_timing)
    if state_timing == "entering":
        assert_panel_aligned(con)
    df = con.execute(f"""
        SELECT p.loan_id, p.month, {', '.join('p.' + c for c in obs_cols)},
               {state_sel},
               p.contacts_last_3m, p.contact_fatigue,
               p.loan_age_months, p.remaining_maturity_months,
               l.original_principal, l.interest_rate_monthly,
               b.age, b.monthly_income, b.dependents,
               a.channel, a.discount_offer, a.is_exploration,
               t.p_recovery, {', '.join('t.' + c for c in zman_cols)}
        FROM monthly_panel p
        JOIN actions a USING (action_id)
        JOIN loans l ON l.loan_id = p.loan_id
        JOIN borrowers b ON b.borrower_id = p.borrower_id
        JOIN oracle_truth t ON t.loan_id = p.loan_id AND t.month = p.month
        {state_join}
        ORDER BY p.loan_id, p.month
    """).df()
    con.close()
    return df, obs_cols, zman_cols


def measurement_frame(df: pd.DataFrame, obs_cols, k: int, rows: int,
                      min_rows: int = 2000):
    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)
    df["ch_code"] = df["channel"].map(CH_CODE).astype(float)
    lag_cols = []
    for j in range(1, k + 1):
        for c in obs_cols + ["ch_code"]:
            name = f"{c}_lag{j}"
            df[name] = g[c].shift(j)
            lag_cols.append(name)
    m = df[(df.is_exploration) & (df.channel == "phone")
           & (df.month >= k)].dropna(subset=lag_cols)
    if len(m) > rows:
        m = m.sample(rows, random_state=0)
    if len(m) < min_rows:
        raise ValueError(f"only {len(m)} measurement rows")
    y = np.log(np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)
               / (1 - np.clip(m["p_recovery"], 1e-9, 1 - 1e-9)))
    return df, m, np.asarray(y, float), lag_cols


def manifold_dict(config: Path):
    from decode_sim.config import load_config
    from decode_sim import oracle_response as orsp
    cfg = load_config(str(config))
    mc = cfg["oracle"]["manifold"]
    n_, m_, ch, A, B, C, W, rot = orsp._manifold(mc)
    human = set(cfg["actions"]["human_channels"])
    channels = list(cfg["actions"]["channels"])
    return {"A": A, "B": B, "C": C, "W": W, "rot": rot,
            "coef": float(mc.get("coef", 1.0)),
            "pop_std": float(mc.get("pop_std", 1.0)),
            "obs_noise": float(mc.get("obs_noise", 0.5)),
            "w_int": np.array([0.0 if c == "none"
                               else (1.0 if c in human else 0.4)
                               for c in channels])}, channels


def filter_features(df: pd.DataFrame, m: pd.DataFrame, obs_cols,
                    man: dict, channels) -> np.ndarray:
    ch_idx = {c: i for i, c in enumerate(channels)}
    d = man["A"].shape[0]
    loans = np.sort(m["loan_id"].unique())
    lpos = {lid: i for i, lid in enumerate(loans)}
    sub = df[df["loan_id"].isin(set(loans))]
    months = int(sub["month"].max()) + 1
    obs = np.full((len(loans), months, len(obs_cols)), np.nan)
    act = np.zeros((len(loans), months), dtype=np.int64)
    alive = np.zeros((len(loans), months), dtype=bool)
    li = sub["loan_id"].map(lpos).to_numpy()
    mo = sub["month"].to_numpy(int)
    obs[li, mo] = sub[obs_cols].to_numpy(float)
    act[li, mo] = sub["channel"].map(ch_idx).to_numpy(int)
    alive[li, mo] = True

    need = {}
    for r, (lid, t) in enumerate(zip(m["loan_id"].to_numpy(),
                                     m["month"].to_numpy(int))):
        need[(lpos[lid], t)] = r
    feat = np.full((len(m), d + 1), np.nan)

    rb = RotationBelief(man, len(loans))
    Wphone = man["W"]["phone"] if isinstance(man["W"], dict) \
        else man["W"][ch_idx["phone"]]
    for s in range(months):
        a_s = alive[:, s]
        if a_s.any():
            o = obs[:, s, :].copy()
            Hm = np.einsum("od,nde->noe", man["C"], rb.M[a_s])
            yv = o[a_s] - np.einsum("od,nd->no", man["C"], rb.m[a_s])
            rb.Lam[a_s] += np.einsum("nod,noe->nde", Hm, Hm) / rb.obs_var
            rb.eta[a_s] += np.einsum("nod,no->nd", Hm, yv) / rb.obs_var
        hits = [(lp, t) for (lp, t) in need if t == s]
        if hits:
            post_mu, post_Sig = rb.posterior_z0()
            for lp, t in hits:
                r = need[(lp, t)]
                mu_zt = rb.M[lp] @ post_mu[lp] + rb.m[lp]
                Sig_zt = rb.M[lp] @ post_Sig[lp] @ rb.M[lp].T
                var_ph = float(Wphone @ Sig_zt @ Wphone) * man["coef"] ** 2
                feat[r, :d] = mu_zt
                feat[r, d] = np.sqrt(max(var_ph, 0.0))
        wv = man["w_int"][act[:, s]] * a_s
        ints = np.unique(man["w_int"])
        T_by = np.stack([man["A"] @ man["rot"][w] if man["rot"] is not None
                         else man["A"] for w in ints])
        Tn = T_by[np.searchsorted(ints, wv)]
        rb.m = np.einsum("nde,ne->nd", Tn, rb.m) + wv[:, None] * man["B"][None]
        rb.M = Tn @ rb.M
    assert not np.isnan(feat).any(), "unfilled filter features"

    rows = list(need.items())[:40]
    for (lp, t), r in rows:
        rb1 = RotationBelief(man, 1)
        for s in range(t + 1):
            if not alive[lp, s]:
                break
            rb1.observe(obs[lp, s][None])
            if s < t:
                rb1.act(np.array([act[lp, s]]))
        mu1, _ = rb1.posterior_z0()
        mu_zt1 = rb1.M[0] @ mu1[0] + rb1.m[0]
        worst = np.abs(mu_zt1 - feat[r, :d]).max()
        assert worst <= 1e-9, \
            f"filter_features diverges from canonical RotationBelief ({worst:.2e})"
    return feat


def gru_rung(df, m, obs_cols, y, folds, seed: int, device: str):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_cols = obs_cols + ["ch_code"]
    hist = {lid: g[["month"] + seq_cols].to_numpy(float)
            for lid, g in df[df["loan_id"].isin(set(m["loan_id"]))]
            .groupby("loan_id", sort=False)}
    T = int(m["month"].max())
    X_seq = np.zeros((len(m), T, len(seq_cols)), dtype=np.float32)
    lens = np.zeros(len(m), dtype=np.int64)
    for r, (lid, t) in enumerate(zip(m["loan_id"].to_numpy(),
                                     m["month"].to_numpy(int))):
        h = hist[lid]
        pre = h[h[:, 0] < t][:, 1:]
        X_seq[r, :len(pre)] = pre
        lens[r] = len(pre)
    X_head = m[FLAT + obs_cols + ["discount_offer"]].to_numpy(np.float32)

    class Net(nn.Module):
        def __init__(self, d_seq, d_head, hid=32):
            super().__init__()
            self.gru = nn.GRU(d_seq, hid, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hid + d_head, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, ln, xh):
            packed = nn.utils.rnn.pack_padded_sequence(
                xs, ln.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.head(torch.cat([h[-1], xh], dim=1)).squeeze(1)

    r2s = []
    for tr_idx, te_idx in folds:
        rng = np.random.default_rng(seed)
        val_n = max(1, int(0.1 * len(tr_idx)))
        perm = rng.permutation(len(tr_idx))
        va_idx = tr_idx[perm[:val_n]]
        fit_idx = tr_idx[perm[val_n:]]

        valid = np.arange(X_seq.shape[1])[None, :] < lens[fit_idx][:, None]
        vals = X_seq[fit_idx][valid]
        mu_s = vals.mean(0)
        sd_s = vals.std(0) + 1e-8
        mu_h = X_head[fit_idx].mean(0)
        sd_h = X_head[fit_idx].std(0) + 1e-8
        mu_y, sd_y = y[fit_idx].mean(), y[fit_idx].std() + 1e-8

        def tensors(idx):
            import torch as th
            return (th.tensor((X_seq[idx] - mu_s) / sd_s),
                    th.tensor(np.maximum(lens[idx], 1)),
                    th.tensor((X_head[idx] - mu_h) / sd_h),
                    th.tensor((y[idx] - mu_y) / sd_y, dtype=th.float32))

        net = Net(X_seq.shape[-1], X_head.shape[-1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xs_f, ln_f, xh_f, y_f = tensors(fit_idx)
        xs_v, ln_v, xh_v, y_v = tensors(va_idx)
        best, best_state, patience = np.inf, None, 0
        for epoch in range(30):
            net.train()
            perm_e = torch.randperm(len(fit_idx))
            for b in range(0, len(fit_idx), 1024):
                sl = perm_e[b:b + 1024]
                opt.zero_grad()
                pred = net(xs_f[sl].to(device), ln_f[sl], xh_f[sl].to(device))
                loss = ((pred - y_f[sl].to(device)) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(((net(xs_v.to(device), ln_v, xh_v.to(device))
                             - y_v.to(device)) ** 2).mean())
            if vl < best - 1e-5:
                best, patience = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 5:
                    break
        net.load_state_dict(best_state)
        net.eval()
        xs_t, ln_t, xh_t, _ = tensors(te_idx)
        with torch.no_grad():
            pred = (net(xs_t.to(device), ln_t, xh_t.to(device)).cpu().numpy()
                    * sd_y + mu_y)
        ss_res = float(((y[te_idx] - pred) ** 2).sum())
        ss_tot = float(((y[te_idx] - y[te_idx].mean()) ** 2).sum())
        r2s.append(1.0 - ss_res / ss_tot)
    return r2s


def arch_rung(df, m, obs_cols, y, folds, seed: int, device: str, arch: str):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_cols = obs_cols + ["ch_code"]
    hist = {lid: g[["month"] + seq_cols].to_numpy(float)
            for lid, g in df[df["loan_id"].isin(set(m["loan_id"]))]
            .groupby("loan_id", sort=False)}
    T = int(m["month"].max())
    X_seq = np.zeros((len(m), T, len(seq_cols)), dtype=np.float32)
    lens = np.zeros(len(m), dtype=np.int64)
    for r, (lid, t) in enumerate(zip(m["loan_id"].to_numpy(),
                                     m["month"].to_numpy(int))):
        h = hist[lid]
        pre = h[h[:, 0] < t][:, 1:]
        X_seq[r, :len(pre)] = pre
        lens[r] = len(pre)
    X_head = m[FLAT + obs_cols + ["discount_offer"]].to_numpy(np.float32)

    lr = 3e-4 if arch == "tfm64x2" else 1e-3
    max_epochs = 100 if arch == "gru32long" else 30
    patience_lim = 15 if arch == "gru32long" else 5

    class Enc(nn.Module):
        def __init__(self, d_seq, T_max):
            super().__init__()
            self.arch = arch
            if arch == "gru128x2":
                self.rnn = nn.GRU(d_seq, 128, num_layers=2, batch_first=True)
                self.d_out = 128
            elif arch == "lstm128x2":
                self.rnn = nn.LSTM(d_seq, 128, num_layers=2, batch_first=True)
                self.d_out = 128
            elif arch == "gru32long":
                self.rnn = nn.GRU(d_seq, 32, batch_first=True)
                self.d_out = 32
            elif arch == "tfm64x2":
                self.proj = nn.Linear(d_seq, 64)
                self.pos = nn.Embedding(T_max, 64)
                layer = nn.TransformerEncoderLayer(
                    d_model=64, nhead=4, dim_feedforward=128,
                    batch_first=True, dropout=0.0)
                self.tfm = nn.TransformerEncoder(layer, num_layers=2)
                self.d_out = 64
            else:
                raise ValueError(arch)

        def forward(self, xs, ln):
            if self.arch == "tfm64x2":
                B, T_, _ = xs.shape
                pos = torch.arange(T_, device=xs.device)
                h = self.proj(xs) + self.pos(pos)[None]
                causal = torch.triu(
                    torch.ones(T_, T_, device=xs.device, dtype=torch.bool),
                    diagonal=1)
                pad = pos[None, :] >= ln.to(xs.device)[:, None]
                out = self.tfm(h, mask=causal, src_key_padding_mask=pad)
                return out[torch.arange(B, device=xs.device),
                           (ln.to(xs.device) - 1).clamp(min=0)]
            packed = nn.utils.rnn.pack_padded_sequence(
                xs, ln.cpu(), batch_first=True, enforce_sorted=False)
            if isinstance(self.rnn, nn.LSTM):
                _, (h, _) = self.rnn(packed)
            else:
                _, h = self.rnn(packed)
            return h[-1]

    class Net(nn.Module):
        def __init__(self, d_seq, d_head, T_max):
            super().__init__()
            self.enc = Enc(d_seq, T_max)
            self.head = nn.Sequential(
                nn.Linear(self.enc.d_out + d_head, 64), nn.ReLU(),
                nn.Linear(64, 1))

        def forward(self, xs, ln, xh):
            return self.head(torch.cat([self.enc(xs, ln), xh],
                                       dim=1)).squeeze(1)

    r2s, stop_epochs = [], []
    for tr_idx, te_idx in folds:
        rng = np.random.default_rng(seed)
        val_n = max(1, int(0.1 * len(tr_idx)))
        perm = rng.permutation(len(tr_idx))
        va_idx = tr_idx[perm[:val_n]]
        fit_idx = tr_idx[perm[val_n:]]

        valid = np.arange(X_seq.shape[1])[None, :] < lens[fit_idx][:, None]
        vals = X_seq[fit_idx][valid]
        mu_s = vals.mean(0)
        sd_s = vals.std(0) + 1e-8
        mu_h = X_head[fit_idx].mean(0)
        sd_h = X_head[fit_idx].std(0) + 1e-8
        mu_y, sd_y = y[fit_idx].mean(), y[fit_idx].std() + 1e-8

        def tensors(idx):
            import torch as th
            return (th.tensor((X_seq[idx] - mu_s) / sd_s),
                    th.tensor(np.maximum(lens[idx], 1)),
                    th.tensor((X_head[idx] - mu_h) / sd_h),
                    th.tensor((y[idx] - mu_y) / sd_y, dtype=th.float32))

        net = Net(X_seq.shape[-1], X_head.shape[-1], X_seq.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        xs_f, ln_f, xh_f, y_f = tensors(fit_idx)
        xs_v, ln_v, xh_v, y_v = tensors(va_idx)
        best, best_state, patience = np.inf, None, 0
        best_epoch = -1
        for epoch in range(max_epochs):
            net.train()
            perm_e = torch.randperm(len(fit_idx))
            for b in range(0, len(fit_idx), 1024):
                sl = perm_e[b:b + 1024]
                opt.zero_grad()
                pred = net(xs_f[sl].to(device), ln_f[sl], xh_f[sl].to(device))
                loss = ((pred - y_f[sl].to(device)) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(((net(xs_v.to(device), ln_v, xh_v.to(device))
                             - y_v.to(device)) ** 2).mean())
            if vl < best - 1e-5:
                best, patience = vl, 0
                best_epoch = epoch
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_lim:
                    break
        stop_epochs.append({"best_epoch": best_epoch, "last_epoch": epoch,
                            "hit_max": epoch == max_epochs - 1})
        net.load_state_dict(best_state)
        net.eval()
        xs_t, ln_t, xh_t, _ = tensors(te_idx)
        with torch.no_grad():
            preds = []
            for b in range(0, len(te_idx), 4096):
                preds.append(net(xs_t[b:b + 4096].to(device),
                                 ln_t[b:b + 4096],
                                 xh_t[b:b + 4096].to(device)).cpu().numpy())
            pred = np.concatenate(preds) * sd_y + mu_y
        ss_res = float(((y[te_idx] - pred) ** 2).sum())
        ss_tot = float(((y[te_idx] - y[te_idx].mean()) ** 2).sum())
        r2s.append(1.0 - ss_res / ss_tot)
    return r2s, stop_epochs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--min-rows", type=int, default=2000,
                    help="smoke runs may lower this; definitive runs keep 2000")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--rungs", default="flat,lags,filter,gru,oracle")
    ap.add_argument("--archs", default="",
                    help="architecture sweep: comma list from "
                         "gru128x2,lstm128x2,tfm64x2,gru32long")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--state-timing", choices=STATE_TIMINGS,
                    default=DEFAULT_STATE_TIMING,
                    help="entering (decision-time values, default) or "
                         "post_outcome_archived (reproduces the archived runs)")
    args = ap.parse_args()
    rungs = set(args.rungs.split(","))

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    df, obs_cols, zman_cols = load_panel(args.world, args.state_timing)
    print(f"state timing: {args.state_timing}", flush=True)
    print(f"panel rows: {len(df)}", flush=True)
    df, m, y, lag_cols = measurement_frame(df, obs_cols, args.k, args.rows,
                                           args.min_rows)
    print(f"measurement rows (phone explore, month>={args.k}): {len(m)}",
          flush=True)

    folds = list(KFold(n_splits=3).split(np.arange(len(m))))

    def gbm(cols_or_X, label):
        X = (m[cols_or_X].to_numpy(float) if isinstance(cols_or_X, list)
             else cols_or_X)
        r2 = []
        for tr, te in folds:
            mdl = HistGradientBoostingRegressor(max_iter=300, random_state=0)
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            r2.append(1 - ((y[te] - pred) ** 2).sum()
                      / ((y[te] - y[te].mean()) ** 2).sum())
        r2 = np.array(r2)
        print(f"{label:28s} R2 = {r2.mean():.4f} (+/- {r2.std():.4f})",
              flush=True)
        return r2

    d_cols = ["discount_offer"]
    res = {"world": str(args.world), "rows": len(m),
           "state_timing": args.state_timing}
    if "flat" in rungs:
        res["flat"] = gbm(FLAT + obs_cols + d_cols,
                          "flat (row + current keyhole)").tolist()
    if "lags" in rungs:
        res["lags"] = gbm(FLAT + obs_cols + d_cols + lag_cols,
                          f"history (+{args.k}m lags)").tolist()

    if "filter" in rungs:
        man, channels = manifold_dict(args.config)
        ff = filter_features(df, m, obs_cols, man, channels)
        Xf = np.column_stack([m[FLAT + obs_cols + d_cols].to_numpy(float), ff])
        res["filter"] = gbm(Xf, "filter (posterior z_t + sd)").tolist()
        Wp = man["W"]["phone"] if isinstance(man["W"], dict) \
            else man["W"][channels.index("phone")]
        z_true = m[[c for c in m.columns if c.startswith("zman_")]] \
            .to_numpy(float)
        resid = man["coef"] * ((z_true - ff[:, :z_true.shape[1]]) @ Wp)
        r2_bayes = 1.0 - float(np.var(resid)) / float(np.var(y))
        res["bayes_analytic"] = r2_bayes
        print(f"{'bayes ceiling (analytic)':28s} R2 = {r2_bayes:.4f} "
              f"(no fit)", flush=True)

    if "gru" in rungs:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GRU device: {device}", flush=True)
        for seed in [int(s) for s in args.seeds.split(",")]:
            r2s = gru_rung(df, m, obs_cols, y, folds, seed, device)
            res[f"gru_seed{seed}"] = list(map(float, r2s))
            print(f"{'GRU (full prefix) s' + str(seed):28s} "
                  f"R2 = {np.mean(r2s):.4f} (+/- {np.std(r2s):.4f})",
                  flush=True)

    if args.archs:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"arch sweep device: {device}", flush=True)
        for arch in args.archs.split(","):
            arch_r2 = []
            for seed in [int(s) for s in args.seeds.split(",")]:
                r2s, stops = arch_rung(df, m, obs_cols, y, folds, seed,
                                       device, arch)
                res[f"{arch}_seed{seed}"] = list(map(float, r2s))
                res[f"{arch}_seed{seed}_stops"] = stops
                arch_r2.append(np.mean(r2s))
                print(f"{arch + ' s' + str(seed):28s} "
                      f"R2 = {np.mean(r2s):.4f} (+/- {np.std(r2s):.4f})  "
                      f"stops {[s['best_epoch'] for s in stops]}"
                      f"{' HIT_MAX' if any(s['hit_max'] for s in stops) else ''}",
                      flush=True)
            if "lags" in res:
                print(f"DELTA {arch} - lags: "
                      f"{np.mean(arch_r2) - np.mean(res['lags']):+.4f}",
                      flush=True)

    if "oracle" in rungs:
        res["oracle"] = gbm(FLAT + obs_cols + d_cols + list(zman_cols),
                            "oracle (+ true hidden z)").tolist()

    gru_r2s = [np.mean(v) for k_, v in res.items()
               if str(k_).startswith("gru_seed")]
    if gru_r2s and "lags" in res:
        print(f"\nDELTA GRU - lags: {np.mean(gru_r2s) - np.mean(res['lags']):+.4f}")
    if "filter" in res and "lags" in res:
        print(f"DELTA filter - lags: {np.mean(res['filter']) - np.mean(res['lags']):+.4f}")
    if "oracle" in res and "filter" in res:
        print(f"DELTA oracle - filter: {np.mean(res['oracle']) - np.mean(res['filter']):+.4f}")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
