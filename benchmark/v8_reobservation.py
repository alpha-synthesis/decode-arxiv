from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import sys

from benchmark.closed_loop_v7 import (
    CHANNELS, build_world)
from benchmark.exact_filter import (
    KalmanBelief, manifold_params, oracle_terms, posterior_pay_prob, _sigmoid)


def _load_cohort(world: Path, cfg: dict, k: int, loans: int, seed: int):
    from decode_sim.trajectory_key import _book_at_month
    book = _book_at_month(world / "sim.duckdb", k)
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(np.asarray(sorted(book["loan_id"])),
                            size=min(loans, len(book)), replace=False))
    book = book[book["loan_id"].isin(set(ids))].sort_values("loan_id") \
        .reset_index(drop=True)
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    obs_cols = sorted((r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE "
        "table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    zc = sorted((r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE "
        "table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    idlist = ",".join(f"'{i}'" for i in book["loan_id"])
    pre = con.execute(f"""
        SELECT p.loan_id, p.month, {', '.join('p.' + c for c in obs_cols)},
               a.channel
        FROM monthly_panel p JOIN actions a USING (action_id)
        WHERE p.loan_id IN ({idlist}) AND p.month <= {k}
        ORDER BY p.loan_id, p.month""").df()
    zk = con.execute(f"""
        SELECT loan_id, {', '.join(zc)} FROM oracle_truth
        WHERE loan_id IN ({idlist}) AND month = {k}
        ORDER BY loan_id""").df()
    con.close()
    n = len(book)
    pos = {l: i for i, l in enumerate(book["loan_id"])}
    pre_obs = np.full((n, k, len(obs_cols)), np.nan)
    pre_ci = np.zeros((n, k), dtype=np.int64)
    obs_k = np.full((n, len(obs_cols)), np.nan)
    for _, r in pre.iterrows():
        i, mo = pos[r["loan_id"]], int(r["month"])
        if mo < k:
            pre_obs[i, mo] = [float(r[c]) for c in obs_cols]
            pre_ci[i, mo] = CHANNELS.index(r["channel"])
        elif mo == k:
            obs_k[i] = [float(r[c]) for c in obs_cols]
    assert not np.isnan(pre_obs).any() and not np.isnan(obs_k).any(), \
        "cohort not fully observed through month k"
    z_k = zk.set_index("loan_id").loc[book["loan_id"]][zc].to_numpy(float)
    return book, pre_obs, pre_ci, obs_k, z_k


class KalmanCEPlanner:

    def __init__(self, terms: dict, man: dict, sigma: float, n: int, k: int):
        self.t_ = terms
        self.man = man
        self.sigma = float(sigma)
        self.Wm = np.stack([man["W"][c] for c in CHANNELS])
        self.coef = float(man["coef"])
        self.n, self.k = n, k
        self.bel = KalmanBelief(man["A"], man["B"], man["C"], man["w_int"],
                                man["pop_std"], man["obs_noise"], sigma, n)
        d = man["A"].shape[0]
        self._Apow = {0: np.eye(d)}
        self._Qu = {0: np.zeros((d, d))}

    def Apow(self, u):
        if u not in self._Apow:
            self._Apow[u] = self.man["A"] @ self.Apow(u - 1)
        return self._Apow[u]

    def Qu(self, u):
        if u not in self._Qu:
            A = self.man["A"]
            self._Qu[u] = (A @ self.Qu(u - 1) @ A.T
                           + self.sigma ** 2 * np.eye(A.shape[0]))
        return self._Qu[u]

    def prime(self, pre_obs, pre_ci):
        for s in range(pre_obs.shape[1]):
            self.bel.observe(pre_obs[:, s, :])
            self.bel.act(pre_ci[:, s])

    def _moments(self, prefix, ci, z_true=None):
        u = len(prefix)
        d = np.zeros(self.bel.d)
        for cj in prefix:
            d = self.man["A"] @ d + self.man["B"] * self.man["w_int"][cj]
        Au = self.Apow(u)
        Wv = self.Wm[ci]
        if z_true is not None:
            mean = self.coef * ((z_true @ Au.T + d) @ Wv)
            var = self.coef ** 2 * float(Wv @ self.Qu(u) @ Wv)
            return mean, np.full(self.n, np.sqrt(max(var, 0.0)))
        mu, Sig = self.bel.posterior_zt()
        mean = self.coef * ((mu @ Au.T + d) @ Wv)
        var = self.coef ** 2 * float(
            Wv @ (Au @ Sig @ Au.T + self.Qu(u)) @ Wv)
        return mean, np.full(self.n, np.sqrt(max(var, 0.0)))

    def search(self, t0: int, win: int, F0: np.ndarray, z_true=None,
               full_plan: bool = False):
        t = self.t_
        n, C = self.n, len(CHANNELS)
        applies = np.array([c != "none" for c in CHANNELS])
        cache: dict[tuple, np.ndarray] = {}

        def p_cells(prefix, ci):
            key = (prefix, ci)
            if key not in cache:
                F = F0.copy()
                for cj in prefix:
                    F = t["decay"] * F + t["fat_add"][cj]
                pen = t["beta_i"] * F if applies[ci] else np.zeros(n)
                mean, sd = self._moments(prefix, ci, z_true)
                base = t["logits0"][t0 + len(prefix), :, ci, :] - pen[:, None]
                cache[key] = posterior_pay_prob(base, mean, sd)
            return cache[key]

        best_V = np.full(n, -np.inf)
        best_first = np.zeros((n, 2), dtype=np.int64)
        best_path = np.zeros((n, win), dtype=np.int64)
        best_dis = np.zeros((n, win), dtype=np.int64)
        for path in itertools.product(range(C), repeat=win):
            V = np.zeros(n)
            dis = np.zeros((n, win), dtype=np.int64)
            for u in range(win - 1, -1, -1):
                p = p_cells(tuple(path[:u]), path[u])
                tot = (-t["costs"][path[u]] + p * t["cash"][t0 + u]
                       + ((1.0 - p) * ((1.0 - t["q"][t0 + u]) * V)[:, None]))
                j = np.argmax(tot, axis=1)
                V = tot[np.arange(n), j]
                dis[:, u] = j
            upd = V > best_V
            if upd.any():
                best_V[upd] = V[upd]
                best_first[upd, 0] = path[0]
                best_first[upd, 1] = dis[upd, 0]
                if full_plan:
                    best_path[upd] = np.asarray(path)[None, :]
                    best_dis[upd] = dis[upd]
        if full_plan:
            return best_path, best_dis, best_V
        return best_first[:, 0], best_first[:, 1], best_V


def run(world: Path, cfg: dict, k: int, H: int, loans: int, reps: int,
        seed: int, arms=("OL", "CL_belief", "CL_oracle", "OL_oracle"),
        config_path: Path | None = None):
    man = manifold_params(cfg)
    sigma = float(cfg["oracle"]["manifold"].get("process_noise", 0.0))
    mf = world / "manifest.json"
    if mf.exists() and config_path is not None:
        from decode_sim.provenance import config_matches
        rec = json.loads(mf.read_text())
        ok, got = config_matches(rec.get("config_sha256"), config_path)
        if not ok:
            raise ValueError(
                f"config/world mismatch: {world.name} was gated under config "
                f"sha {rec['config_sha256'][:12]}…, got {got[:12]}…")
    elif not mf.exists():
        print(f"WARNING: {world.name} has no manifest.json — provenance "
              f"unchecked (pilot world?)", flush=True)
    book, pre_obs, pre_ci, obs_k, z_k = _load_cohort(world, cfg, k,
                                                     loans, seed)
    n = len(book)
    w = build_world(book, cfg, H)
    terms = oracle_terms(w)
    F0 = (book["entering_fatigue"].to_numpy(float)
          if "entering_fatigue" in book.columns else np.zeros(n))
    applies = np.array([c != "none" for c in CHANNELS])

    plans: dict[str, tuple] = {}
    if "OL" in arms or "CL_belief" in arms:
        pl = KalmanCEPlanner(terms, man, sigma, n, k)
        pl.prime(pre_obs, pre_ci)
        pl.bel.observe(obs_k)
        if "OL" in arms:
            plans["OL"] = pl.search(0, H, F0, full_plan=True)[:2]
        bel_mu0, bel_Sig0 = pl.bel.mu.copy(), pl.bel.Sig.copy()
    if "OL_oracle" in arms:
        po = KalmanCEPlanner(terms, man, sigma, n, k)
        plans["OL_oracle"] = po.search(0, H, F0, z_true=z_k,
                                       full_plan=True)[:2]

    d = man["A"].shape[0]
    m_obs = man["C"].shape[0]
    dec = terms["decay"]
    values = {a: np.zeros((reps, n)) for a in arms}
    dp_value_clo_t0 = None
    for rep in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        eps = rng.normal(0.0, sigma or 0.0, (H, n, d)) if sigma > 0 \
            else np.zeros((H, n, d))
        nu = rng.normal(0.0, man["obs_noise"], (H, n, m_obs))
        state = {}
        for arm in arms:
            state[arm] = {"z": z_k.copy(), "F": F0.copy(),
                          "S": np.ones(n), "val": np.zeros(n)}
            if arm == "CL_belief":
                p2 = KalmanCEPlanner(terms, man, sigma, n, k)
                p2.bel.mu, p2.bel.Sig = bel_mu0.copy(), bel_Sig0.copy()
                state[arm]["pl"] = p2
            if arm == "CL_oracle":
                state[arm]["pl"] = KalmanCEPlanner(terms, man, sigma, n, k)
        for t in range(H):
            for arm in arms:
                st = state[arm]
                if arm in ("OL", "OL_oracle"):
                    ci = plans[arm][0][:, t]
                    di = plans[arm][1][:, t]
                elif arm == "CL_belief":
                    if t > 0:
                        st["pl"].bel.observe(st["obs"])
                    ci, di, V = st["pl"].search(t, H - t, st["F"])
                else:
                    ci, di, V = st["pl"].search(t, H - t, st["F"],
                                                z_true=st["z"])
                    if t == 0 and rep == 0:
                        dp_value_clo_t0 = V.copy()
                proj = (st["z"] * np.take(
                    np.stack([man["W"][c] for c in CHANNELS]), ci,
                    axis=0)).sum(1) * man["coef"]
                pen = terms["beta_i"] * st["F"] * applies[ci]
                logit = (terms["logits0"][t, np.arange(n), ci, di]
                         - pen + proj)
                p = _sigmoid(logit)
                cash = terms["cash"][t][np.arange(n), di]
                st["val"] += st["S"] * (p * cash - terms["costs"][ci])
                st["S"] *= (1.0 - p) * (1.0 - terms["q"][t])
                wv = man["w_int"][ci]
                st["z"] = (st["z"] @ man["A"].T
                           + np.outer(wv, man["B"]) + eps[t])
                st["F"] = dec * st["F"] + terms["fat_add"][ci]
                if arm == "CL_belief":
                    st["pl"].bel.act(ci)
                    st["obs"] = st["z"] @ man["C"].T + nu[t]
        for arm in arms:
            values[arm][rep] = state[arm]["val"]
    out = {a: values[a].mean(0) for a in arms}
    return book, out, values, dp_value_clo_t0


def main() -> int:
    from decode_sim.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--loans", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-q0", action="store_true",
                    help="sigma=0 identity gate: CL_oracle MC value == its "
                         "own t=0 DP value and == OL_oracle, per loan")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(str(args.config))
    book, out, values, dp_clo = run(
        args.world, cfg, args.k, args.horizon, args.loans,
        1 if args.gate_q0 else args.reps, args.seed,
        config_path=args.config)
    n = len(book)
    if args.gate_q0:
        gap1 = float(np.abs(out["CL_oracle"] - dp_clo).max())
        gap2 = float(np.abs(out["CL_oracle"] - out["OL_oracle"]).max())
        print(f"GATE q0: |MC - DP| max {gap1:.2e}  "
              f"|CL_oracle - OL_oracle| max {gap2:.2e}  "
              f"{'PASS' if max(gap1, gap2) <= 1e-9 else 'FAIL'}", flush=True)
        return 0 if max(gap1, gap2) <= 1e-9 else 1
    boot = np.random.default_rng(12345)
    reobs = values["CL_belief"].mean(0) - values["OL"].mean(0)
    bs = np.array([reobs[boot.integers(0, n, n)].mean()
                   for _ in range(10_000)])
    res = {a: float(out[a].mean()) for a in out}
    res["reobs_mean"] = float(reobs.mean())
    res["reobs_ci"] = [float(np.percentile(bs, 2.5)),
                       float(np.percentile(bs, 97.5))]
    res["reobs_p_le_0"] = float((bs <= 0).mean())
    res["mc_se_reobs"] = float(
        (values["CL_belief"] - values["OL"]).mean(1).std()
        / np.sqrt(values["OL"].shape[0]))
    ceiling = res.get("CL_oracle", np.nan)
    gmc_ok = True
    for a in values:
        se = float(values[a].mean(1).std() / np.sqrt(values[a].shape[0]))
        res[f"mc_se_{a}"] = se
        gmc_ok &= se < 0.005 * ceiling
    res["gmc_pass"] = bool(gmc_ok)
    for k_, v in res.items():
        print(f"{k_:24s} {v}", flush=True)
    print(f"G-MC (all arm SEs < 0.5% of ceiling {ceiling:.1f}): "
          f"{'PASS' if gmc_ok else 'FAIL — raise reps'}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
