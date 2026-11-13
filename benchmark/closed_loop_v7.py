from __future__ import annotations

import argparse
import json
import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["none", "sms", "whatsapp", "phone"]
D_STEPS = 21

OBS_SALT = 0x7C105ED

PUBLIC_BOOK_COLUMNS = (
    "loan_id", "outstanding_balance", "days_past_due",
    "days_in_credito_em_liquidacao", "loan_age_months",
    "interest_rate_monthly", "original_principal", "initial_status",
    "age", "monthly_income", "entering_fatigue",
)

PUBLIC_STATE_COLUMNS = (
    "loan_id", "outstanding_balance", "days_past_due",
    "days_in_credito_em_liquidacao", "loan_age_months", "contact_fatigue",
    "interest_rate_monthly", "original_principal", "age", "monthly_income",
)

PUBLIC_PREHISTORY_COLUMNS = (
    "loan_id", "month", "contact_fatigue", "channel", "discount_offer",
)

FORBIDDEN_PREFIXES = ("zman_", "p_recovery", "p_chargeoff", "recovery_draw",
                      "chargeoff_draw", "borrower_resilience",
                      "payment_friction", "regional_macro_stress",
                      "contactability", "fatigue_beta")


def _loan_key(loan_id) -> int:
    return int.from_bytes(
        hashlib.blake2b(str(loan_id).encode(), digest_size=8).digest(), "big")


def _sql_ids(ids) -> str:
    for i in ids:
        if "'" in str(i):
            raise ValueError(f"unexpected quote in loan_id {i!r}")
    return ",".join("'" + str(i) + "'" for i in ids)


class ObservationContract:

    def __init__(self, sim_cfg: dict, world_seed: int):
        from decode_sim import oracle_response as orsp

        man = (sim_cfg.get("oracle") or {}).get("manifold")
        if not man:
            raise ValueError("closed_loop_v7 requires a manifold world")
        if float(man.get("process_noise", 0.1)) > 0.0:
            raise ValueError(
                "counterfactual observations require process_noise = 0 — with "
                "stochastic latent drift the counterfactual z is not a "
                "function of the actions and this contract is void")
        n_lat, m_obs, chans, A, B, C, W, rot = orsp._manifold(man)
        self.n_lat, self.m_obs = n_lat, m_obs
        self.A, self.B, self.C = A, B, C
        self.rot = rot
        self.obs_noise = float(man.get("obs_noise", 0.5))
        self.gain = float(man.get("obs_nonlin_gain", 0.0))
        pop_std = float(man.get("pop_std", 1.0))
        self.s2 = (pop_std ** 2) * (C * C).sum(1)
        cfg_channels = list(sim_cfg["actions"]["channels"])
        if cfg_channels != CHANNELS:
            raise ValueError(
                f"channel index space mismatch: scenario has {cfg_channels}, "
                f"this harness indexes {CHANNELS}. w_int and fat_add are "
                f"positional, so a reordered menu would silently roll the "
                f"wrong physics for every action.")
        human = set(sim_cfg["actions"].get("human_channels") or [])
        self.w_int = np.array(
            [0.0 if c == "none" else (1.0 if c in human else 0.4) for c in CHANNELS])
        self.world_seed = int(world_seed)

    def roll_z(self, z: np.ndarray, ci: np.ndarray) -> np.ndarray:
        wv = self.w_int[np.asarray(ci)]
        zz = z
        if self.rot is not None:
            zz = np.empty_like(z)
            for w_val, R in self.rot.items():
                m_ = wv == w_val
                if m_.any():
                    zz[m_] = z[m_] @ R.T
        return zz @ self.A.T + np.outer(wv, self.B)

    def sensor(self, z: np.ndarray) -> np.ndarray:
        L = z @ self.C.T
        if self.gain:
            L = L + self.gain * (L ** 2 - self.s2)
        return L

    def noise(self, loan_ids, month: int) -> np.ndarray:
        eps = np.empty((len(loan_ids), self.m_obs))
        for i, lid in enumerate(np.asarray(loan_ids)):
            rng = np.random.default_rng(np.random.SeedSequence(
                [OBS_SALT, self.world_seed, _loan_key(lid), int(month)]))
            eps[i] = rng.normal(0.0, self.obs_noise, self.m_obs)
        return eps

    def observe(self, z: np.ndarray, loan_ids: np.ndarray, month: int) -> np.ndarray:
        return self.sensor(z) + self.noise(loan_ids, month)


class PublicState:

    def __init__(self, book: pd.DataFrame, sim_cfg: dict):
        fat = (sim_cfg["oracle"].get("contact_fatigue") or {})
        if fat.get("channel_split"):
            raise NotImplementedError(
                "channel_split fatigue is out of scope for this harness (the answer "
                "key refuses start_month>0 with channel_split)")
        if sim_cfg["oracle"].get("discount_fatigue"):
            raise NotImplementedError(
                "discount_fatigue (last-discount-seen memory) is not rolled by "
                "this closed-loop mirror (the answer key refuses start_month>0 too)")
        self.decay = float(fat.get("decay", 0.5))
        inc = float(fat.get("increment", 1.0))
        mult = float(fat.get("human_multiplier", 1.0))
        human = set(sim_cfg["actions"].get("human_channels") or [])
        self.fat_add = np.array(
            [0.0 if c == "none" else inc * (mult if c in human else 1.0)
             for c in CHANNELS])
        self.rate = book["interest_rate_monthly"].to_numpy(float)
        self.reset(book)

    def reset(self, book: pd.DataFrame) -> None:
        self.bal = book["outstanding_balance"].to_numpy(float).copy()
        self.dpd = book["days_past_due"].to_numpy(float).copy()
        self.liq = book["days_in_credito_em_liquidacao"].to_numpy(float).copy()
        self.age_m = book["loan_age_months"].to_numpy(float).copy()
        self.fatigue = (book["entering_fatigue"].to_numpy(float).copy()
                        if "entering_fatigue" in book.columns
                        else np.zeros(len(book)))

    def frame(self, book: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "loan_id": book["loan_id"].to_numpy(),
            "outstanding_balance": self.bal,
            "days_past_due": self.dpd,
            "days_in_credito_em_liquidacao": self.liq,
            "loan_age_months": self.age_m,
            "contact_fatigue": self.fatigue,
            "interest_rate_monthly": self.rate,
            "original_principal": book["original_principal"].to_numpy(float),
            "age": book["age"].to_numpy(float),
            "monthly_income": book["monthly_income"].to_numpy(float),
        })
        _assert_public(out, PUBLIC_STATE_COLUMNS, what="StepView.state")
        return out

    def step(self, ci: np.ndarray) -> None:
        self.bal = self.bal * (1.0 + self.rate * 0.35)
        self.dpd = self.dpd + 30.0
        self.liq = np.where(self.dpd >= 90.0, self.liq + 30.0, self.liq)
        self.age_m = self.age_m + 1
        self.fatigue = self.decay * self.fatigue + self.fat_add[np.asarray(ci)]


def _assert_public(df: pd.DataFrame, allow: tuple[str, ...], *,
                   obs_ok: bool = False, what: str = "solver view") -> None:
    ok = set(allow)
    extra = [c for c in df.columns
             if c not in ok and not (obs_ok and c.startswith("obs_"))]
    if extra:
        raise AssertionError(
            f"LEAK: non-public columns in {what}: {sorted(extra)} "
            f"(allowed: {sorted(ok)}{' + obs_*' if obs_ok else ''})")
    bad = [c for c in df.columns
           if any(c.startswith(p) for p in FORBIDDEN_PREFIXES)]
    if bad:
        raise AssertionError(f"LEAK: oracle-truth columns in {what}: {bad}")


def public_book(book: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PUBLIC_BOOK_COLUMNS if c in book.columns]
    out = book[cols].copy()
    _assert_public(out, PUBLIC_BOOK_COLUMNS, what="the book handed to reset()")
    return out


@dataclass(frozen=True)
class StepView:
    t: int
    months_remaining: int
    loan_ids: np.ndarray
    state: pd.DataFrame
    obs: np.ndarray
    obs_history: np.ndarray
    channel_history: np.ndarray
    discount_history: np.ndarray
    prehistory: pd.DataFrame
    frozen: bool = False


class Solver:

    name = "abstract"

    def reset(self, book: pd.DataFrame, prehistory: pd.DataFrame) -> None:
        pass

    def decide(self, view: StepView) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def run_closed_loop(solver: Solver, book: pd.DataFrame, prehistory: pd.DataFrame,
                    contract: ObservationContract, sim_cfg: dict,
                    k: int, H: int, d_grid: np.ndarray,
                    obs_k: np.ndarray, frozen: bool = False) -> pd.DataFrame:
    n = len(book)
    loan_ids = book["loan_id"].to_numpy()
    zc = [c for c in book.columns if c.startswith("zman_")]
    z = book[zc].to_numpy(float).copy()
    state = PublicState(book, sim_cfg)
    solver.reset(public_book(book), prehistory)

    obs_hist = np.empty((n, H, contract.m_obs))
    ch_hist = np.zeros((n, H), dtype=np.int64)
    d_hist = np.zeros((n, H))
    cur_obs = obs_k

    for t in range(H):
        obs_hist[:, t, :] = cur_obs
        view = StepView(
            t=t, months_remaining=H - t, loan_ids=loan_ids,
            state=state.frame(book), obs=cur_obs.copy(),
            obs_history=obs_hist[:, :t + 1, :].copy(),
            channel_history=ch_hist[:, :t].copy(),
            discount_history=d_hist[:, :t].copy(),
            prehistory=prehistory, frozen=frozen)
        ci, di = solver.decide(view)
        ci = np.asarray(ci, dtype=np.int64)
        di = np.asarray(di, dtype=np.int64)
        if ci.shape != (n,) or di.shape != (n,):
            raise ValueError(f"{solver.name}: decide must return two (n,) arrays")
        ch_hist[:, t] = ci
        d_hist[:, t] = d_grid[di]
        z = contract.roll_z(z, ci)
        state.step(ci)
        if t + 1 < H:
            cur_obs = (obs_k if frozen
                       else contract.observe(z, loan_ids, k + t + 1))

    return pd.DataFrame({
        "loan_id": np.tile(loan_ids, H),
        "month": np.repeat(np.arange(H), n),
        "channel": [CHANNELS[c] for c in ch_hist.T.ravel()],
        "discount_offer": d_hist.T.ravel(),
    })


def world_seed_of(world: Path) -> int:
    import duckdb
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    try:
        rows = con.execute("SELECT DISTINCT seed FROM oracle_truth").fetchall()
    finally:
        con.close()
    if len(rows) != 1:
        raise ValueError(f"{world}: expected exactly one seed, got {rows}")
    return int(rows[0][0])


def load_cohort(world: Path, sim_cfg: dict, k: int, loans: int):
    import duckdb
    from decode_sim.trajectory_key import _book_at_month

    db = world / "sim.duckdb"
    book = _book_at_month(db, k)
    ids = sorted(book["loan_id"])[:loans]
    book = book[book["loan_id"].isin(set(ids))].reset_index(drop=True)
    ids = book["loan_id"].to_list()

    con = duckdb.connect(str(db), read_only=True)
    obs_cols = sorted(
        (r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE "
            "table_name='monthly_panel' AND column_name LIKE 'obs_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    idl = _sql_ids(ids)
    pre = con.execute(f"""
        SELECT p.loan_id, p.month, p.contact_fatigue, {', '.join('p.' + c for c in obs_cols)},
               a.channel, a.discount_offer
        FROM monthly_panel p JOIN actions a ON a.action_id = p.action_id
        WHERE p.loan_id IN ({idl}) AND p.month < {k}
        ORDER BY p.loan_id, p.month""").df()
    logged = con.execute(f"""
        SELECT p.loan_id, p.month, a.channel, a.discount_offer
        FROM monthly_panel p JOIN actions a ON a.action_id = p.action_id
        WHERE p.loan_id IN ({idl}) AND p.month >= {k}
        ORDER BY p.loan_id, p.month""").df()
    zcols = sorted(
        (r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE "
            "table_name='oracle_truth' AND column_name LIKE 'zman_%'").fetchall()),
        key=lambda c: int(c.split("_")[1]))
    ztrue = con.execute(f"""
        SELECT loan_id, month, {', '.join(zcols)} FROM oracle_truth
        WHERE loan_id IN ({idl}) AND month >= {k} ORDER BY loan_id, month""").df()
    obs_logged = con.execute(f"""
        SELECT loan_id, month, {', '.join(obs_cols)} FROM monthly_panel
        WHERE loan_id IN ({idl}) AND month >= {k} ORDER BY loan_id, month""").df()
    con.close()

    obs_k = (obs_logged[obs_logged["month"] == k]
             .set_index("loan_id").loc[ids, obs_cols].to_numpy(float))
    _assert_public(pre, PUBLIC_PREHISTORY_COLUMNS, obs_ok=True,
                   what="StepView.prehistory")
    return book, pre, obs_cols, obs_k, logged, ztrue, obs_logged


def build_world(book: pd.DataFrame, sim_cfg: dict, H: int):
    from decode_sim.trajectory_key import _World

    w = _World(book, sim_cfg, H, d_steps=D_STEPS)
    if list(w.channels) != CHANNELS:
        raise ValueError(
            f"_World pruned or reordered the channel menu ({w.channels}); the "
            f"harness indexes {CHANNELS} and the two must agree. Manifold "
            f"worlds are supposed to disable pruning.")
    return w


def board(args) -> int:
    from decode_sim.config import load_config
    from decode_sim.receding import entering_state
    from decode_sim.trajectory_key import (
        export_answer_key, grade_plan)

    from benchmark.closed_loop_arms import (
        AnalystSolver, BeliefSolver, GBMSolver,
        OracleSolver, P7Solver, RandomSolver, _load_public_static,
        fit_gbm, fit_p7, load_train_rows)

    sim_cfg = load_config(str(args.config))
    k, H = args.k, args.horizon
    book, pre, obs_cols, obs_k, logged, ztrue, obs_logged = load_cohort(
        args.world, sim_cfg, k, args.loans)
    seed = args.world_seed if args.world_seed is not None else world_seed_of(args.world)
    contract = ObservationContract(sim_cfg, world_seed=seed)
    d_grid = np.linspace(0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
    db = args.world / "sim.duckdb"
    ids = list(book["loan_id"])
    print(f"cohort {len(ids)} loans, k={k}, H={H}, world seed {seed}", flush=True)

    key = export_answer_key(db, sim_cfg, H=H, loan_ids=ids, start_month=k)
    key = key.set_index("loan_id").loc[ids]
    ceiling = float(key["ceiling"].mean())
    print(f"ceiling/loan {ceiling:.2f}  myopic bound "
          f"{100 * key['myopic_bound'].mean() / ceiling:.2f}%", flush=True)

    w = build_world(book, sim_cfg, H)
    F0, z0 = entering_state(w, book)

    from decode_sim.trajectory_key import paid_share_mean
    pshare = paid_share_mean(sim_cfg["oracle"])

    gbm_cache: dict[bool, tuple] = {}

    def gbm_parts(history: bool):
        if history not in gbm_cache:
            if args.train_world is None:
                raise ValueError("GBM arms need --train-world")
            tdf, lag_cols = load_train_rows(
                args.train_world / "sim.duckdb", k, obs_cols)
            gbm_cache[history] = fit_gbm(tdf, obs_cols, lag_cols, k,
                                         history=history, seed=args.seed)
        return gbm_cache[history]

    def make(name):
        if name.startswith("oracle_W"):
            return OracleSolver(w, z0, F0, H, window=int(name.split("W")[1]))
        if name == "random":
            return RandomSolver(D_STEPS, seed=args.seed)
        if name in ("posterior_gbm_myopic", "posterior_gbm_sequential",
                    "posterior_gbm_static_myopic",
                    "posterior_gbm_static_sequential",
                    "longlags_myopic", "longlags_sequential",
                    "lags_exactF_myopic", "lags_exactF_sequential",
                    "hybrid_none_exact_sequential",
                    "hybrid_contact_exact_sequential",
                    "hybrid_allfitted_sequential",
                    "hybrid_allexact_sequential",
                    "hybrid_contact_exact_static_sequential",
                    "hybrid_none_exact_static_sequential",
                    "structured_sequential", "structured_myopic"):
            from benchmark.decomp_arms import (
                LagsExactFSolver, LongLagsSolver, PosteriorGBMSolver,
                fit_longlags_gbm, fit_posterior_gbm)

            if args.train_world is None:
                raise ValueError(f"{name} needs --train-world")
            myo = name.endswith("_myopic")
            if name.startswith("posterior_gbm"):
                if "pgbm" not in gbm_cache:
                    gbm_cache["pgbm"] = fit_posterior_gbm(
                        args.train_world / "sim.duckdb", sim_cfg, obs_cols, k,
                        seed=args.seed, feat_cache=args.posterior_feat_cache)
                f_, pc_, pn_ = gbm_cache["pgbm"]
                return PosteriorGBMSolver(
                    f_, pc_, pn_, sim_cfg, k, H, obs_cols, myopic=myo,
                    pshare=pshare, static_features="_static_" in name)
            if name.startswith("hybrid_"):
                from benchmark.decomp_arms import HybridResponseSolver
                from benchmark.exact_filter import oracle_terms

                HYBRIDS = {
                    "hybrid_allfitted": (False, False, False),
                    "hybrid_allexact": (True, True, False),
                    "hybrid_none_exact": (True, False, False),
                    "hybrid_contact_exact": (False, True, False),
                    "hybrid_none_exact_static": (True, False, True),
                    "hybrid_contact_exact_static": (False, True, True),
                }
                stem = name.rsplit("_", 1)[0]
                if stem not in HYBRIDS:
                    raise ValueError(f"unknown hybrid arm {name!r}; known: "
                                     f"{sorted(HYBRIDS)}")
                en, ec, st = HYBRIDS[stem]
                feats_, p_c_, p_n_ = gbm_parts(True)
                return HybridResponseSolver(
                    feats_, p_c_, p_n_, sim_cfg, k, H, obs_cols, myopic=myo,
                    pshare=pshare, terms=oracle_terms(w),
                    exact_none=en, exact_contact=ec, static_exact=st)
            if name.startswith("structured"):
                from benchmark.decomp_arms import fit_structured
                if "structured" not in gbm_cache:
                    gbm_cache["structured"] = fit_structured(
                        args.train_world / "sim.duckdb", obs_cols, k,
                        seed=args.seed)
                f_, pc_, pn_ = gbm_cache["structured"]
                sol = LagsExactFSolver(f_, pc_, pn_, sim_cfg, k, H, obs_cols,
                                       history=True, myopic=myo,
                                       pshare=pshare)
                sol.name = ("structured_myopic" if myo
                            else "structured_sequential")
                return sol
            if name.startswith("lags_exactF"):
                feats_, p_c_, p_n_ = gbm_parts(True)
                return LagsExactFSolver(feats_, p_c_, p_n_, sim_cfg, k, H,
                                        obs_cols, history=True, myopic=myo,
                                        pshare=pshare)
            if "longlags" not in gbm_cache:
                gbm_cache["longlags"] = fit_longlags_gbm(
                    args.train_world / "sim.duckdb", obs_cols, k, H,
                    seed=args.seed)
            f_, pc_, pn_ = gbm_cache["longlags"]
            return LongLagsSolver(f_, pc_, pn_, sim_cfg, k, H, obs_cols,
                                  history=True, myopic=myo, pshare=pshare)
        if name == "bayes_filter":
            from benchmark.exact_filter import (
                BayesFilterSolver, manifold_params, oracle_terms)

            return BayesFilterSolver(oracle_terms(w), manifold_params(sim_cfg),
                                     k, H)
        if name == "analyst":
            if args.analyst_policy is None:
                raise ValueError("analyst needs --analyst-policy")
            static = _load_public_static(args.world, ids, k, H)
            return AnalystSolver(args.analyst_policy,
                                 sim_cfg, k, H, obs_cols, static)
        if name == "p7_myopic":
            if args.train_world is None:
                raise ValueError("p7_myopic needs --train-world")
            art = fit_p7(args.train_world / "sim.duckdb", obs_cols,
                         cache=args.p7_cache)
            static = _load_public_static(args.world, ids, k, H)
            return P7Solver(art, sim_cfg, k, H, obs_cols, static)
        if name == "belief_sequential":
            import torch

            from benchmark.belief_v7 import _load_model, load_frame

            if args.belief_model is None:
                raise ValueError("belief_sequential needs --belief-model")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model, ck = _load_model(args.belief_model, dev)
            dfe, bcols = load_frame(db)
            bad = [c for c in dfe.columns if c.startswith("zman_")]
            if bad:
                raise AssertionError(f"LEAK: latent in the belief frame: {bad}")
            dfe = dfe[dfe.loan_id.isin(set(ids))]
            return BeliefSolver(model, ck, sim_cfg, k, H, bcols, dev,
                                pshare, dfe)
        if name in ("flat_myopic", "flat_sequential",
                    "history_myopic", "history_sequential"):
            history = name.startswith("history")
            feats, p_c, p_n = gbm_parts(history)
            return GBMSolver(feats, p_c, p_n, sim_cfg, k, H, obs_cols,
                             history=history,
                             myopic=name.endswith("_myopic"), pshare=pshare)
        raise ValueError("unknown arm %r" % name)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    per_loan = {"loan_id": ids,
                "ceiling": key["ceiling"].to_numpy(float),
                "myopic_bound": key["myopic_bound"].to_numpy(float)}
    for name in args.arms.split(","):
        name = name.strip()
        frozen = name.endswith(":frozen")
        base_name = name[:-len(":frozen")] if frozen else name
        print("-- arm " + name, flush=True)
        plan = run_closed_loop(make(base_name), book, pre, contract, sim_cfg,
                               k, H, d_grid, obs_k, frozen=frozen)
        g = grade_plan(plan, db, sim_cfg, H=H, start_month=k)
        g = g.set_index("loan_id").loc[ids]
        v = g["value"].to_numpy(float)
        per_loan[name] = v
        rows[name] = round(100.0 * float(v.mean()) / ceiling, 2)
        print("   %s: %.2f%% of ceiling" % (name, rows[name]), flush=True)
        plan.to_csv(args.out / ("plan_%s.csv" % name.replace(":", "_")),
                    index=False)

    pd.DataFrame(per_loan).to_csv(args.out / "per_loan_values.csv", index=False)
    out = {"n_loans": len(ids), "k": k, "H": H, "world_seed": int(seed),
           "obs_salt": hex(OBS_SALT), "ceiling_per_loan": round(ceiling, 2),
           "myopic_bound_pct": round(
               100.0 * float(key["myopic_bound"].mean()) / ceiling, 2),
           "rows": rows}

    for nm, ref, label in (("oracle_W%d" % H, "ceiling", "ceiling"),
                           ("oracle_W1", "myopic_bound", "myopic bound")):
        if nm in per_loan:
            d = float(np.abs(per_loan[nm] - key[ref].to_numpy(float)).max())
            ok = d <= 1e-9
            out["V3_harness_%s_vs_%s" % (nm, ref)] = {"max_abs_diff": d,
                                                      "pass": ok}
            print("V3(harness) %s vs %s: max |diff| %.3e (%s)"
                  % (nm, label, d, "PASS" if ok else "FAIL"), flush=True)

    (args.out / "board.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("validate", help="harness validation checks V1/V2")
    pv.add_argument("--world", type=Path, required=True)
    pv.add_argument("--config", type=Path, required=True)
    pv.add_argument("--k", type=int, default=6)
    pv.add_argument("--horizon", type=int, default=6)
    pv.add_argument("--loans", type=int, default=4000)
    pv.add_argument("--world-seed", type=int, default=None,
                    help="override; default reads oracle_truth.seed")
    pv.add_argument("--out", type=Path, required=True)
    pb = sub.add_parser("board", help="run arms closed-loop and grade them")
    pb.add_argument("--world", type=Path, required=True)
    pb.add_argument("--train-world", type=Path, default=None,
                    help="world the estimator arms are fitted on")
    pb.add_argument("--config", type=Path, required=True)
    pb.add_argument("--k", type=int, default=6)
    pb.add_argument("--horizon", type=int, default=6)
    pb.add_argument("--loans", type=int, default=4000)
    pb.add_argument("--arms", required=True,
                    help="comma-separated, e.g. oracle_W6,oracle_W1,random")
    pb.add_argument("--analyst-policy", type=Path, default=None,
                    help="frozen policy.json from the analyst investigation")
    pb.add_argument("--posterior-feat-cache", type=Path, default=None,
                    help="parquet cache for D1's train-world posterior features")
    pb.add_argument("--p7-cache", type=Path, default=None,
                    help="pickle path for the fitted Tweedie reward model")
    pb.add_argument("--belief-model", type=Path, default=None,
                    help="CE-Seq checkpoint for the belief_sequential arm")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--world-seed", type=int, default=None)
    pb.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "validate":
        return validate(args)
    if args.cmd == "board":
        return board(args)
    return 1


MUTATIONS = {
    "transpose_A": lambda c: setattr(c, "A", c.A.T.copy()),
    "permute_C_rows": lambda c: setattr(c, "C", c.C[::-1].copy()),
    "negate_C_row0": lambda c: setattr(
        c, "C", np.vstack([-c.C[0], c.C[1:]]).copy()),
    "scale_B_1pct": lambda c: setattr(c, "B", (c.B * 1.01).copy()),
    "reverse_w_int": lambda c: setattr(c, "w_int", c.w_int[::-1].copy()),
}


def _measure(contract, ids, keep, book, zt, ol, obs_cols, k, H, la, ch_idx):
    from scipy.stats import ks_2samp, norm as _norm
    scipy_norm_ppf = _norm.ppf

    zc = [c for c in book.columns if c.startswith("zman_")]
    z = book.loc[keep, zc].to_numpy(float).copy()
    max_dz = 0.0
    per_cell = []
    for t in range(H):
        rows = [(ids[i], k + t) for i in keep]
        ref = zt.loc[rows].to_numpy(float)
        max_dz = max(max_dz, float(np.abs(z - ref).max()))
        o_ref = ol.loc[rows, obs_cols].to_numpy(float)
        res = o_ref - contract.sensor(ref)
        syn = contract.noise(np.array([ids[i] for i in keep]), k + t)
        for j in range(len(obs_cols)):
            ks = ks_2samp(res[:, j], syn[:, j])
            per_cell.append({"month": k + t, "dim": obs_cols[j],
                             "mean": float(res[:, j].mean()),
                             "std": float(res[:, j].std()),
                             "ks_p": float(ks.pvalue)})
        z = contract.roll_z(z, np.array([ch_idx[c] for c in la[k + t]]))

    means = np.array([c["mean"] for c in per_cell])
    stds = np.array([c["std"] for c in per_cell])
    ps = np.array([c["ks_p"] for c in per_cell])
    nobs = len(keep)
    alpha = 0.01 / len(per_cell)
    zc = float(scipy_norm_ppf(1.0 - alpha / 2.0))
    mean_tol = zc * stds / np.sqrt(nobs)
    std_tol = zc * contract.obs_noise / np.sqrt(2 * nobs)
    for c, mt in zip(per_cell, mean_tol):
        c["mean_z"] = float(c["mean"] / (c["std"] / np.sqrt(nobs)))
        c["mean_tol"] = float(mt)
    return {
        "V1_max_dz": max_dz,
        "V1_pass": bool(max_dz <= 1e-12),
        "V2_worst_mean_z": float(np.abs([c["mean_z"] for c in per_cell]).max()),
        "V2_worst_std_err": float(np.abs(stds - contract.obs_noise).max()),
        "V2_std_tol": float(std_tol),
        "V2_min_ks_p": float(ps.min()),
        "V2_z_crit": zc,
        "V2_alpha_bonferroni": alpha,
        "V2_pass": bool((np.abs(means) <= mean_tol).all()
                        and np.abs(stds - contract.obs_noise).max() <= std_tol
                        and ps.min() > alpha),
        "per_cell": per_cell,
    }


class _LoggedReplaySolver(Solver):

    name = "logged_replay"

    def __init__(self, la, k: int, d_grid: np.ndarray):
        self.la, self.k, self.d_grid = la, k, d_grid
        self.seen = []

    def decide(self, view):
        self.seen.append(view.obs.copy())
        ci = np.array([CHANNELS.index(c) for c in self.la["ch"][self.k + view.t]],
                      dtype=np.int64)
        d = self.la["d"][self.k + view.t].to_numpy(float)
        di = np.abs(self.d_grid[None, :] - d[:, None]).argmin(1)
        return ci, di.astype(np.int64)


def _v6_executor_wiring(contract, book, pre, sim_cfg, k, H, obs_k, ids, keep,
                        zt, logged, d_grid):
    sub = [i for i, l in enumerate(ids) if l in set(book.loc[keep, "loan_id"])]
    ids_k = [ids[i] for i in sub]
    bk = book.iloc[sub].reset_index(drop=True)
    la = {"ch": logged.pivot(index="loan_id", columns="month",
                             values="channel").loc[ids_k],
          "d": logged.pivot(index="loan_id", columns="month",
                            values="discount_offer").loc[ids_k]}
    solver = _LoggedReplaySolver(la, k, d_grid)
    run_closed_loop(solver, bk, pre[pre.loan_id.isin(set(ids_k))], contract,
                    sim_cfg, k, H, d_grid, obs_k[sub])
    worst = 0.0
    for t in range(H):
        if t == 0:
            want = obs_k[sub]
        else:
            ztrue_t = zt.loc[[(i, k + t) for i in ids_k]].to_numpy(float)
            want = contract.sensor(ztrue_t) + contract.noise(
                np.array(ids_k), k + t)
        worst = max(worst, float(np.abs(solver.seen[t] - want).max()))
    return {"max_abs_obs_diff": worst, "pass": bool(worst <= 1e-12),
            "n_loans": len(ids_k)}


def validate(args) -> int:
    import copy

    from decode_sim.config import load_config

    sim_cfg = load_config(str(args.config))
    k, H = args.k, args.horizon
    book, pre, obs_cols, obs_k, logged, ztrue, obs_logged = load_cohort(
        args.world, sim_cfg, k, args.loans)
    seed = args.world_seed if args.world_seed is not None else world_seed_of(args.world)
    if seed != int(sim_cfg.get("seed", -1)):
        print(f"note: world seed {seed} (scenario yaml says "
              f"{sim_cfg.get('seed')}) — keying on the world's own seed",
              flush=True)
    contract = ObservationContract(sim_cfg, world_seed=seed)
    ids = book["loan_id"].to_list()
    n = len(ids)
    print(f"cohort: {n} loans, k={k}, H={H}, obs dims {obs_cols}", flush=True)

    counts = logged.groupby("loan_id")["month"].count()
    full = set(counts[counts >= H].index)
    keep = np.array([i for i, lid in enumerate(ids) if lid in full])
    print(f"V1/V2 cohort (panel alive through k+H-1): {len(keep)}/{n}", flush=True)
    ch_idx = {c: i for i, c in enumerate(CHANNELS)}
    la = logged.pivot(index="loan_id", columns="month", values="channel")
    la = la.loc[[ids[i] for i in keep]]
    zt = ztrue.set_index(["loan_id", "month"])
    ol = obs_logged.set_index(["loan_id", "month"])

    truth = _measure(contract, ids, keep, book, zt, ol, obs_cols, k, H, la, ch_idx)
    print(f"V1 z-replay: max |z_cf - z_logged| = {truth['V1_max_dz']:.3e}  "
          f"({'PASS' if truth['V1_pass'] else 'FAIL'}, gate 1e-12)", flush=True)
    print(f"V2 sensor+noise ({len(truth['per_cell'])} month x dim cells): "
          f"worst mean z={truth['V2_worst_mean_z']:+.2f} "
          f"(crit {truth['V2_z_crit']:.2f}) "
          f"worst |std-{contract.obs_noise}|={truth['V2_worst_std_err']:.5f} "
          f"(tol {truth['V2_std_tol']:.5f}) "
          f"min KS p={truth['V2_min_ks_p']:.4g} "
          f"(alpha={truth['V2_alpha_bonferroni']:.2g})  "
          f"({'PASS' if truth['V2_pass'] else 'FAIL'})", flush=True)

    a = contract.noise(np.array(ids[:64]), k + 1)
    b = contract.noise(np.array(ids[:64]), k + 1)
    c4 = bool((a == b).all())
    print(f"C4 determinism: repeat draw identical  ({'PASS' if c4 else 'FAIL'})",
          flush=True)

    print("V5 mutation test — each row must be CAUGHT (V1 or V2 fails):",
          flush=True)
    v5 = {}
    for name, mutate in MUTATIONS.items():
        mc = copy.deepcopy(contract)
        mutate(mc)
        m = _measure(mc, ids, keep, book, zt, ol, obs_cols, k, H, la, ch_idx)
        caught = (not m["V1_pass"]) or (not m["V2_pass"])
        v5[name] = {"caught": bool(caught), "V1_pass": m["V1_pass"],
                    "V2_pass": m["V2_pass"], "V1_max_dz": m["V1_max_dz"],
                    "V2_min_ks_p": m["V2_min_ks_p"]}
        print(f"  {name:16s} V1={'ok' if m['V1_pass'] else 'FAILS':5s} "
              f"V2={'ok' if m['V2_pass'] else 'FAILS':5s} -> "
              f"{'CAUGHT' if caught else 'MISSED  <-- hole in the gate'}",
              flush=True)
    v5_pass = all(r["caught"] for r in v5.values())

    d_grid = np.linspace(0.0, float(sim_cfg["actions"]["max_discount"]), D_STEPS)
    v6 = _v6_executor_wiring(contract, book, pre, sim_cfg, k, H, obs_k, ids,
                             keep, zt, logged, d_grid)
    print(f"V6 executor wiring: max |obs_executor - obs_expected| = "
          f"{v6['max_abs_obs_diff']:.3e} over {v6['n_loans']} loans  "
          f"({'PASS' if v6['pass'] else 'FAIL'}, gate 1e-12)", flush=True)

    ok = truth["V1_pass"] and truth["V2_pass"] and c4 and v5_pass and v6["pass"]
    args.out.mkdir(parents=True, exist_ok=True)
    rep = {"n_loans": n, "n_v1": int(len(keep)), "k": k, "H": H,
           "obs_salt": hex(OBS_SALT), "world_seed": int(seed),
           "obs_noise_cfg": contract.obs_noise,
           "V1": {kk: truth[kk] for kk in ("V1_max_dz", "V1_pass")},
           "V2": {kk: truth[kk] for kk in
                  ("V2_worst_mean_z", "V2_worst_std_err", "V2_std_tol",
                   "V2_min_ks_p", "V2_z_crit", "V2_alpha_bonferroni",
                   "V2_pass")},
           "V2_per_cell": truth["per_cell"],
           "C4_determinism_pass": c4,
           "V5_mutations": v5, "V5_pass": v5_pass,
           "V6_executor_wiring": v6,
           "verdict": "PASS" if ok else "FAIL"}
    (args.out / "validate_v1_v2_v5.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"\nAPPARATUS GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
