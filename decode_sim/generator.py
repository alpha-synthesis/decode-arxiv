from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decode_sim import oracle_response
from decode_sim.dictionaries import (
    ALLOWED_STATES,
    CHANNELS,
    GENDERS,
    INCOME_BANDS,
    LOCATIONS,
    PRODUCTS,
    TERMINAL_STATES,
)
from decode_sim.storage import SimulationResult


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _weighted_choice(rng: np.random.Generator, records: list[dict[str, Any]], key: str, n: int):
    weights = np.asarray([float(r.get("weight", 1.0)) for r in records], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(records), size=n, p=weights)
    return [records[i][key] for i in idx], idx


def _split_own(f_ch: dict, channel: str):
    if channel == "none":
        return 0.0
    if channel not in f_ch:
        raise ValueError(
            f"contact_fatigue.channel_split is on but channel {channel!r} is "
            "not covered by the per-channel fatigue state (actions.channels + "
            "program channels) — extend the channel list")
    return f_ch[channel]


@dataclass(frozen=True)
class RecoveryInputs:
    channel: str
    discount_offer: float
    age: int
    loan_age_months: int
    days_past_due: int
    days_in_credito_em_liquidacao: int
    borrower_resilience: float
    payment_friction: float
    regional_macro_stress: float
    contactability: float
    contact_fatigue: float = 0.0
    fatigue_beta: float | None = None
    contact_fatigue_own: float | None = None
    last_discount_seen: float = 0.0
    latent_respo: float = 0.0
    borrower_type: int = 0
    profile: int = 0
    manifold_z: tuple = ()
    mixed_pref: float = 0.0
    owed_amount: float = 0.0
    monthly_income: float = 0.0
    original_principal: float = 0.0
    contacts_last_3m: float = 0.0
    noise: float = 0.0
    wallet_type: str = ""
    purchase_profile: str = ""
    payment_plan: str = ""


CHANNEL_EFFECT = {
    "none": -0.85,
    "sms": 0.10,
    "whatsapp": 0.18,
    "phone": 0.35,
    "field_visit": 0.48,
}


def _channel_effect(channel: str, oracle_cfg: dict[str, Any]) -> float:
    table = oracle_cfg.get("channel_effects") or CHANNEL_EFFECT
    return float(table.get(channel, 0.0))


def _dpd_term(days_past_due: int, oracle_cfg: dict[str, Any]) -> float:
    curve = oracle_cfg.get("dpd_curve") or {"shape": "linear"}
    shape = curve.get("shape", "linear")
    if shape == "linear":
        return float(oracle_cfg["dpd_effect"]) * (days_past_due / 30.0)
    if shape == "power_law":
        knee = float(curve.get("knee_days", 90.0))
        plateau = float(curve.get("plateau", 0.0))
        if days_past_due <= knee:
            return plateau
        decay = float(curve.get("decay", 1.0))
        exponent = float(curve.get("exponent", 1.5))
        term = plateau - decay * ((days_past_due - knee) / 30.0) ** exponent
        if "floor" in curve:
            return max(term, float(curve["floor"]))
        return term
    raise ValueError(f"unknown oracle.dpd_curve.shape: {shape!r}")


def recovery_probability(inputs: RecoveryInputs, oracle_cfg: dict[str, Any]) -> float:
    if oracle_cfg.get("classical_policy"):
        return oracle_response.classical_recovery(
            inputs.channel,
            {"days_past_due": inputs.days_past_due, "owed_amount": inputs.owed_amount},
            oracle_cfg,
        )
    target = float(oracle_cfg["target_monthly_recovery_rate"])
    offer = oracle_response.offer_response(inputs.discount_offer, oracle_cfg) \
        * oracle_response.profile_offer_scale(inputs.profile, oracle_cfg) \
        * oracle_response.offer_gate(inputs.channel, oracle_cfg)
    age_curve = -0.0009 * ((inputs.age - 43) / 10.0) ** 2 + 0.07
    dpd = _dpd_term(inputs.days_past_due, oracle_cfg)
    liq = float(oracle_cfg["liquidacao_age_effect"]) * (
        inputs.days_in_credito_em_liquidacao / 30.0
    )
    loan_age = float(oracle_cfg["loan_age_effect"]) * (inputs.loan_age_months / 12.0)
    hidden = (
        0.55 * inputs.borrower_resilience
        - 0.60 * inputs.payment_friction
        - 0.45 * inputs.regional_macro_stress
        + (0.35 * inputs.contactability if inputs.channel != "none" else 0.0)
    )
    feat = float((oracle_cfg.get("wallet_type_effects") or {}).get(inputs.wallet_type, 0.0)) + float(
        (oracle_cfg.get("purchase_profile_effects") or {}).get(inputs.purchase_profile, 0.0)
    )
    feats = {
        "wallet_type": inputs.wallet_type,
        "purchase_profile": inputs.purchase_profile,
        "days_past_due": inputs.days_past_due,
        "loan_age_months": inputs.loan_age_months,
        "borrower_resilience": inputs.borrower_resilience,
        "payment_friction": inputs.payment_friction,
        "monthly_income": inputs.monthly_income,
        "utilization": (inputs.owed_amount / inputs.original_principal)
        if inputs.original_principal else 0.0,
        "contacts_last_3m": inputs.contacts_last_3m,
    }
    interaction = oracle_response.interaction_term(
        inputs.channel, inputs.discount_offer, feats, oracle_cfg)
    obliq = oracle_response.oblique_term(inputs.channel, feats, oracle_cfg)
    curv = oracle_response.offer_curvature_term(inputs.discount_offer, feats, oracle_cfg)
    hetero = oracle_response.channel_heterogeneity(inputs.channel, feats, oracle_cfg)
    payment_plan = oracle_response.payment_plan_outcome(inputs.payment_plan, oracle_cfg)
    fatigue = oracle_response.fatigue_term(
        inputs.channel, inputs.contact_fatigue, oracle_cfg,
        beta_override=inputs.fatigue_beta, fatigue_own=inputs.contact_fatigue_own
    )
    dfat = (oracle_response.discount_fatigue_term(
        inputs.channel, inputs.discount_offer, inputs.last_discount_seen, oracle_cfg)
        if oracle_cfg.get("discount_fatigue") else 0.0)
    latent = oracle_response.latent_term(inputs.channel, inputs.latent_respo, oracle_cfg)
    btype = oracle_response.type_term(inputs.channel, inputs.borrower_type, oracle_cfg)
    prof = oracle_response.profile_term(inputs.channel, inputs.profile, oracle_cfg)
    mani = oracle_response.manifold_term(inputs.channel, np.asarray(inputs.manifold_z, dtype=float), oracle_cfg)
    mixed = oracle_response.mixed_pref_term(inputs.channel, inputs.mixed_pref, oracle_cfg)
    logit = (
        _logit(target)
        + _channel_effect(inputs.channel, oracle_cfg)
        + offer
        + age_curve
        + dpd
        + liq
        + loan_age
        + hidden
        + feat
        + interaction
        + obliq
        + curv
        + hetero
        + payment_plan
        + fatigue
        + dfat
        + latent
        + btype
        + prof
        + mani
        + mixed
        + inputs.noise
    )
    return _sigmoid(logit)


def generate(
    config: dict[str, Any],
    out_dir: str | Path,
    batch_size: int | None = None,
    action_plan: pd.DataFrame | None = None,
    plan_epsilon: float | None = None,
) -> SimulationResult:
    import duckdb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    plan_lookup = None
    if action_plan is not None:
        plan_lookup = {(str(r.loan_id), int(r.month)): (str(r.channel), float(r.discount_offer))
                       for r in action_plan.itertuples()}
    rng = np.random.default_rng(seed)
    total = int(config["population"]["borrowers"])
    _validate_assignment(config)
    streaming = batch_size is not None
    step = total if not streaming else int(batch_size)

    bundle_path = config["population"].get("calibration_bundle")
    bundle = None
    if bundle_path:
        from decode_sim.bundle import load_bundle

        bundle = load_bundle(bundle_path)

    regional = None
    if streaming:
        strength = float(config["oracle"]["hidden_confounding_strength"])
        ufs = (
            sorted(set(bundle["cells"]["uf"].astype(str)))
            if bundle is not None
            else [loc["uf"] for loc in LOCATIONS]
        )
        regional = {uf: rng.normal(0.0, 0.45 * strength) for uf in ufs}

    db_path = out_dir / "sim.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    loan_next, action_next, created = 1, 1, False
    kept: dict[str, pd.DataFrame] = {}
    try:
        start = 0
        while start < total:
            n = min(step, total - start)
            if bundle is not None:
                from decode_sim.bundle import sample_population

                borrowers, hidden, loans = sample_population(
                    bundle, config, rng, n=n, start_idx=start,
                    loan_start=loan_next, regional=regional,
                )
            else:
                borrowers, hidden = _generate_borrowers(
                    config, rng, n=n, start_idx=start, regional=regional
                )
                loans = _generate_loans(config, borrowers, rng, loan_start=loan_next)
            tables = _generate_panel(
                config, borrowers, hidden, loans, rng, seed, action_start=action_next,
                plan_lookup=plan_lookup, plan_epsilon=plan_epsilon,
            )
            batch = {"borrowers": borrowers, "loans": loans, **tables}
            for name, frame in batch.items():
                con.register("_chunk", frame)
                con.execute(
                    f"INSERT INTO {name} SELECT * FROM _chunk"
                    if created
                    else f"CREATE TABLE {name} AS SELECT * FROM _chunk"
                )
                con.unregister("_chunk")
            created = True
            loan_next += len(loans)
            action_next += len(tables["actions"])
            start += n
            if not streaming:
                kept = batch
    finally:
        con.close()
    return SimulationResult(db_path=db_path, tables=kept)


def _generate_borrowers(
    config: dict[str, Any],
    rng: np.random.Generator,
    n: int | None = None,
    start_idx: int = 0,
    regional: dict[str, float] | None = None,
):
    if n is None:
        n = int(config["population"]["borrowers"])
    cfg = config["demographics"]

    _, loc_idx = _weighted_choice(rng, LOCATIONS, "city", n)
    locs = [LOCATIONS[i] for i in loc_idx]
    income_bands, inc_idx = _weighted_choice(rng, INCOME_BANDS, "band", n)
    bands = [INCOME_BANDS[i] for i in inc_idx]

    gender_weights = np.array([cfg["gender_weights"].get(g, 0.0) for g in GENDERS], dtype=float)
    gender_weights = gender_weights / gender_weights.sum()
    genders = rng.choice(GENDERS, size=n, p=gender_weights)
    ages = np.clip(
        rng.normal(float(cfg["age_mean"]), float(cfg["age_std"]), size=n).round(),
        int(cfg["age_min"]),
        int(cfg["age_max"]),
    ).astype(int)
    monthly_income = np.array([rng.uniform(b["min"], b["max"]) for b in bands]).round(2)

    borrowers = pd.DataFrame(
        {
            "borrower_id": [f"B{i:06d}" for i in range(start_idx + 1, start_idx + n + 1)],
            "uf": [loc["uf"] for loc in locs],
            "city": [loc["city"] for loc in locs],
            "lat": [loc["lat"] for loc in locs],
            "lon": [loc["lon"] for loc in locs],
            "age": ages,
            "gender": genders,
            "income_band": income_bands,
            "monthly_income": monthly_income,
            "dependents": rng.poisson(1.1, size=n).clip(0, 5),
        }
    )

    pop = config["population"]
    for col, key in (("wallet_type", "wallet_type_weights"),
                     ("purchase_profile", "purchase_profile_weights"),
                     ("payment_plan", "payment_plan_weights")):
        if pop.get(key):
            cats = list(pop[key])
            p = np.array([float(pop[key][c]) for c in cats], dtype=float)
            borrowers[col] = rng.choice(cats, size=n, p=p / p.sum())

    strength = float(config["oracle"]["hidden_confounding_strength"])
    if regional is None:
        regional = {loc["uf"]: rng.normal(0.0, 0.45 * strength) for loc in LOCATIONS}
    hidden = pd.DataFrame(
        {
            "borrower_id": borrowers["borrower_id"],
            "borrower_resilience": rng.normal(0.0, strength, size=n),
            "payment_friction": rng.normal(0.0, strength, size=n),
            "contactability": rng.normal(0.0, strength, size=n),
            "regional_macro_stress": [regional[uf] for uf in borrowers["uf"]],
        }
    )
    return borrowers, hidden


def _generate_loans(
    config: dict[str, Any],
    borrowers: pd.DataFrame,
    rng: np.random.Generator,
    loan_start: int = 1,
):
    lo_min, lo_max = config["population"]["loans_per_borrower"]
    counts = rng.integers(int(lo_min), int(lo_max) + 1, size=len(borrowers))
    product_names, product_idx = _weighted_choice(rng, PRODUCTS, "product_type", int(counts.sum()))
    products = [PRODUCTS[i] for i in product_idx]

    cp = config["oracle"].get("classical_policy")
    cp_order = list(cp["channels"]) if cp else []
    if cp:
        cp_left, cp_right = oracle_response.classical_owed_domain(config["oracle"])
        cp_th = cp["thresholds"]
        cp_x2, cp_x1, cp_x = int(cp_th["x2"]), int(cp_th["x1"]), int(cp_th["x"])
        cp_dpd_max = int(cp.get("dpd_max", 180))
        cp_bounds = [(0, cp_x2), (cp_x2 + 1, cp_x1), (cp_x1 + 1, cp_x), (cp_x + 1, cp_dpd_max)]
        cp_shares = np.asarray([float(cp["band_shares"][c]) for c in cp_order], dtype=float)
        cp_shares = cp_shares / cp_shares.sum()

    rows = []
    loan_num = loan_start
    for borrower_id, n_loans in zip(borrowers["borrower_id"], counts, strict=True):
        for _ in range(int(n_loans)):
            k = loan_num - loan_start
            product = products[k]
            term = int(product["term_months"])
            loan_age = int(rng.integers(1, max(term, 2)))
            if cp:
                owed = float(rng.uniform(cp_left, cp_right))
                principal = owed
                balance = round(owed, 2)
                band = int(rng.choice(len(cp_order), p=cp_shares))
                blo, bhi = cp_bounds[band]
                dpd = int(rng.integers(blo, bhi + 1))
            else:
                principal = float(
                    np.exp(
                        rng.uniform(
                            np.log(float(config["loan"]["principal_min"])),
                            np.log(float(config["loan"]["principal_max"])),
                        )
                    )
                )
                dpd = max(
                    0,
                    int(
                        rng.normal(
                            float(config["loan"]["initial_dpd_mean"]),
                            float(config["loan"]["initial_dpd_std"]),
                        )
                    ),
                )
                balance = round(principal * rng.uniform(0.45, 1.05), 2)
            status = "performing"
            if dpd >= 90:
                status = "credito_em_liquidacao"
            elif dpd >= 30:
                status = "delinquent"
            rows.append(
                {
                    "loan_id": f"L{loan_num:06d}",
                    "borrower_id": borrower_id,
                    "product_type": product_names[k],
                    "origination_month": -loan_age,
                    "original_principal": round(principal, 2),
                    "outstanding_balance": balance,
                    "interest_rate_monthly": float(product["rate"]),
                    "term_months": term,
                    "loan_age_months": loan_age,
                    "remaining_maturity_months": max(term - loan_age, 0),
                    "days_past_due": dpd,
                    "days_in_credito_em_liquidacao": max(0, dpd - 90),
                    "initial_status": status,
                }
            )
            loan_num += 1
    return pd.DataFrame(rows)


def program_allowed_channels(
    config: dict[str, Any],
    month: int,
    days_past_due: int,
    balance: float,
    tier: int,
) -> list[str]:
    des = config["actions"].get("program") or {}
    if not des:
        return []
    lo_dpd, hi_dpd = des.get("dpd_window", [91, 720])
    if not (lo_dpd <= days_past_due <= hi_dpd):
        return []
    allowed = []
    for channel in des.get("channels", []):
        tiers_ok = des.get("channel_tiers", {}).get(channel, [])
        if tier not in tiers_ok:
            continue
        window = (des.get("program_months") or {}).get(channel)
        if window is not None and not (window[0] <= month <= window[1]):
            continue
        cap = (des.get("debt_cap_by_channel") or {}).get(channel)
        if cap is not None and balance > float(cap):
            continue
        allowed.append(channel)
    return allowed


def _discount_range(config: dict[str, Any], channel: str) -> tuple[float, float]:
    ranges = config["actions"].get("channel_discount_ranges") or {}
    if channel in ranges:
        lo, hi = ranges[channel]
        return float(lo), float(hi)
    if "default" in ranges:
        lo, hi = ranges["default"]
        return float(lo), float(hi)
    return 0.0, float(config["actions"]["max_discount"])


def settlement_amount(
    config: dict[str, Any],
    channel: str,
    balance: float,
    discount: float,
    fund_avail: float,
) -> float:
    fund_cfg = config["actions"].get("fund") or {}
    if channel == fund_cfg.get("channel"):
        entry_cap = float(fund_cfg.get("entry_cap", 0.3))
        down_payment = min(float(fund_avail), entry_cap * balance)
        return round(down_payment + (balance - down_payment) * (1.0 - discount), 2)
    return round(balance * (1.0 - discount), 2)


def _is_explorable(state: str, liq_days: int, exp_cfg: dict[str, Any]) -> bool:
    states = exp_cfg.get("explore_states", ["delinquent", "credito_em_liquidacao"])
    if state not in states:
        return False
    if state == "credito_em_liquidacao":
        if liq_days > int(exp_cfg.get("explore_max_liquidacao_days", 180)):
            return False
    return True


VALID_ASSIGNMENT_MODES = ("confounded", "deterministic_dpd")


def _validate_assignment(config: dict[str, Any]) -> None:
    actions = config["actions"]
    mode = str(actions.get("assignment_mode", "confounded"))
    if mode not in VALID_ASSIGNMENT_MODES:
        raise ValueError(
            f"actions.assignment_mode must be one of {VALID_ASSIGNMENT_MODES}, "
            f"got {mode!r}"
        )
    if mode == "deterministic_dpd":
        ch = str(actions.get("deterministic_channel", "phone"))
        if ch not in CHANNELS or ch == "none":
            raise ValueError(
                f"actions.deterministic_channel must be a contact channel, got {ch!r}"
            )


def _guided_target_cell(
    exp_cfg: dict[str, Any], days_past_due: int, rng: np.random.Generator
) -> str | None:
    targets = exp_cfg.get("oversample_targets") or []
    for entry in sorted(targets, key=lambda e: float(e.get("dpd_max", 1e9))):
        if days_past_due <= float(entry.get("dpd_max", 1e9)):
            cells = list(entry.get("cells") or [])
            if not cells:
                return None
            return str(rng.choice(cells))
    return None


def _materialize_cell(
    cell: str,
    config: dict[str, Any],
    channels: list[str],
    max_discount: float,
    rng: np.random.Generator,
):
    if cell == "none":
        return "none", 0.0
    human = set(config["actions"].get("human_channels") or [])
    group, _, level = cell.partition("-")
    if group == "human":
        pool = [c for c in channels if c in human]
    else:
        pool = [c for c in channels if c != "none" and c not in human]
    if not pool:
        pool = [c for c in channels if c != "none"]
    channel = str(rng.choice(pool))
    split = 0.30
    if level == "low":
        discount = float(rng.uniform(0.0, min(split, max_discount)))
    else:
        lo = min(split, max_discount)
        discount = float(rng.uniform(lo, max_discount))
    return channel, round(discount, 4)


def _plan_action(
    config: dict[str, Any],
    rng: np.random.Generator,
    planned: tuple[str, float],
    state: str,
    liq_days: int,
    month: int,
    days_past_due: int,
    balance: float,
    tier: int,
    plan_epsilon: float | None,
):
    exp_cfg = config["actions"].get("exploration", {}) or {}
    eps = float(exp_cfg.get("epsilon", 0.0)) if plan_epsilon is None else float(plan_epsilon)
    explore_prob = eps if (eps > 0.0 and _is_explorable(state, liq_days, exp_cfg)) else 0.0
    channels = list(config["actions"].get("channels") or CHANNELS)
    des_cfg = config["actions"].get("program") or {}
    des_channels = set(des_cfg.get("channels", []))
    ordinary = [c for c in channels if c not in des_channels]
    allowed_des = program_allowed_channels(config, month, days_past_due, balance, tier)
    contact_allowed = [c for c in ordinary if c != "none"] + allowed_des
    n_allowed = len(contact_allowed)
    if explore_prob > 0.0 and rng.random() < explore_prob:
        channel = str(rng.choice(contact_allowed))
        lo, hi = _discount_range(config, channel)
        return channel, round(float(rng.uniform(lo, hi)), 4), True, explore_prob, n_allowed
    channel, discount = planned
    if channel not in channels:
        raise ValueError(f"action_plan channel {channel!r} is not in actions.channels {channels}")
    return channel, round(float(discount), 4), False, explore_prob, n_allowed


def _choose_action(
    config: dict[str, Any],
    rng: np.random.Generator,
    days_past_due: int,
    payment_friction: float,
    contactability: float,
    state: str,
    liq_days: int,
    payment_plan: str = "",
    borrower_type: int = 0,
    mixed_hidden: float = 0.0,
    month: int = 0,
    balance: float = 0.0,
    tier: int = 0,
):
    max_discount = float(config["actions"]["max_discount"])
    if str(config["actions"].get("assignment_mode", "confounded")) == "deterministic_dpd":
        thr = int(config["actions"].get("deterministic_dpd_threshold", 90))
        if days_past_due >= thr:
            ch = str(config["actions"].get("deterministic_channel", "phone"))
            discount = round(min(max_discount, 0.10 + (days_past_due - thr) / 360.0), 4)
            return ch, discount, False, 0.0, 1
        return "none", 0.0, False, 0.0, 1
    exp_cfg = config["actions"].get("exploration", {}) or {}
    epsilon = float(exp_cfg.get("epsilon", 0.0))
    explore_prob = epsilon if (epsilon > 0.0 and _is_explorable(state, liq_days, exp_cfg)) else 0.0
    channels = list(config["actions"].get("channels") or CHANNELS)
    des_cfg = config["actions"].get("program") or {}
    des_channels = set(des_cfg.get("channels", []))
    ordinary = [c for c in channels if c not in des_channels]
    allowed_des = program_allowed_channels(config, month, days_past_due, balance, tier)
    contact_allowed = [c for c in ordinary if c != "none"] + allowed_des
    n_allowed = len(contact_allowed)

    if explore_prob > 0.0 and rng.random() < explore_prob:
        mode = str(exp_cfg.get("mode", "uniform")).lower()
        if mode == "guided":
            cell = _guided_target_cell(exp_cfg, days_past_due, rng)
            if cell is not None:
                channel, discount = _materialize_cell(cell, config, channels, max_discount, rng)
                return channel, discount, True, explore_prob, n_allowed
        channel = str(rng.choice(contact_allowed))
        lo, hi = _discount_range(config, channel)
        discount = round(float(rng.uniform(lo, hi)), 4)
        return channel, discount, True, explore_prob, n_allowed

    cp = config["oracle"].get("classical_policy")
    if cp:
        th = cp["thresholds"]
        band = (int(days_past_due > int(th["x2"])) + int(days_past_due > int(th["x1"]))
                + int(days_past_due > int(th["x"])))
        if cp.get("policy_family") == "scorecard":
            tier = cp["tier_by_band"][band]
            tier_actions = cp["tier_actions"][tier]
            names = list(tier_actions.keys())
            weights = np.asarray([float(tier_actions[nm]) for nm in names], dtype=float)
            weights = weights / weights.sum()
            channel = str(rng.choice(names, p=weights))
            return channel, 0.0, False, explore_prob, n_allowed
        return str(list(cp["channels"])[band]), 0.0, False, explore_prob, n_allowed

    if allowed_des:
        offer_rate = float(
            (des_cfg.get("offer_rate_by_month") or {}).get(month, des_cfg.get("offer_rate", 0.0))
        )
        tilt_cfg = des_cfg.get("offer_balance_tilt") or {}
        if offer_rate > 0.0 and tilt_cfg:
            gamma = float(tilt_cfg.get("gamma", 0.0))
            ref = float(tilt_cfg.get("ref_brl", 1000.0))
            max_mult = float(tilt_cfg.get("max_mult", 8.0))
            offer_rate = min(0.95, offer_rate * min((max(balance, 1.0) / ref) ** gamma, max_mult))
        if offer_rate > 0.0 and rng.random() < offer_rate:
            mix = des_cfg.get("offer_mix") or {}
            w = np.asarray([float(mix.get(c, 0.0)) for c in allowed_des], dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(allowed_des))
            channel = str(rng.choice(allowed_des, p=w / w.sum()))
            lo, hi = _discount_range(config, channel)
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            discount = round(float(np.clip(rng.normal(mid, half / 3.0), lo, hi)), 4)
            return channel, discount, False, explore_prob, n_allowed

    weights = np.asarray(
        [float(config["actions"]["channel_weights"].get(ch, 0.0)) for ch in ordinary],
        dtype=float,
    )
    if config["actions"].get("intensity_confounds_with_friction", True):
        coeffs = config["actions"].get("intensity_coeffs") or {
            "none": -0.55, "phone": 0.35, "field_visit": 0.55,
        }
        iscale = float(config["actions"].get("intensity_scale", 1.0))
        plan_bump = oracle_response.payment_plan_intensity(payment_plan, config)
        intensity = min(
            max(iscale * (days_past_due / 180.0 + payment_friction - 0.4 * contactability)
                + plan_bump, -2), 2)
        weights = weights.copy()
        for i, ch in enumerate(ordinary):
            if ch in coeffs:
                weights[i] *= exp(float(coeffs[ch]) * intensity)
    ta = config["actions"].get("type_assignment")
    if ta:
        human_ch = set(config["actions"].get("human_channels") or [])
        tname = oracle_response._TYPE_NAMES[int(borrower_type)]
        bumps = ta.get(tname) or {}
        weights = weights.copy()
        for i, ch in enumerate(channels):
            cls = "none" if ch == "none" else ("human" if ch in human_ch else "digital")
            weights[i] *= exp(float(bumps.get(cls, 0.0)))
    ma = config["actions"].get("mixed_assignment")
    if ma:
        kappa = float(ma.get("kappa", 0.0))
        human_ch = set(config["actions"].get("human_channels") or [])
        weights = weights.copy()
        for i, ch in enumerate(channels):
            sign = 0.0 if ch == "none" else (1.0 if ch in human_ch else -1.0)
            weights[i] *= exp(kappa * sign * mixed_hidden)
    weights = weights / weights.sum()
    channel = str(rng.choice(ordinary, p=weights))
    if channel == "none":
        return channel, 0.0, False, explore_prob, n_allowed
    base_offer = rng.beta(2.0, 5.0) * max_discount
    stress_offer = min(max_discount, base_offer + max(0.0, days_past_due - 60) / 360.0)
    lo, hi = _discount_range(config, channel)
    stress_offer = min(max(stress_offer, lo), hi)
    return channel, round(float(stress_offer), 4), False, explore_prob, n_allowed


def _generate_panel(
    config: dict[str, Any],
    borrowers: pd.DataFrame,
    borrower_hidden: pd.DataFrame,
    loans: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
    action_start: int = 1,
    plan_lookup: dict | None = None,
    plan_epsilon: float | None = None,
) -> dict[str, pd.DataFrame]:
    months = int(config["panel"]["months"])
    oracle_cfg = config["oracle"]
    fat_cfg = oracle_cfg.get("contact_fatigue")
    fat_human = set(config["actions"].get("human_channels") or [])
    split_cfg = (fat_cfg or {}).get("channel_split")
    if split_cfg:
        _chans = [c for c in (config["actions"].get("channels") or []) if c != "none"]
        for c in ((config["actions"].get("program") or {}).get("channels") or []):
            if c != "none" and c not in _chans:
                _chans.append(c)
        split_channels = _chans
    else:
        split_channels = []
    dfat_cfg = oracle_cfg.get("discount_fatigue")
    lat_cfg = oracle_cfg.get("latent_response")
    bt_cfg = oracle_cfg.get("borrower_type")
    bt_human = set((bt_cfg or {}).get("human") or [])
    bt_w = np.array((bt_cfg or {}).get("type_weights") or [1/3, 1/3, 1/3], dtype=float)
    js_cfg = oracle_cfg.get("judicial_shock")
    if js_cfg:
        js_rate = float(js_cfg.get("monthly_rate", 0.0))
        js_dpd = int(js_cfg.get("dpd_jump_days", 360))
        js_liq = int(js_cfg.get("liq_jump_days", 360))
    man_cfg = oracle_cfg.get("manifold")
    if man_cfg:
        (man_n, man_m, man_chans, man_A, man_B, man_C, man_W,
         man_rot) = oracle_response._manifold(man_cfg)
        man_rng = np.random.default_rng(
            int(config["seed"]) * 7919 + 13 + int(action_start) - 1)
        man_pop = float(man_cfg.get("pop_std", 1.0)); man_proc = float(man_cfg.get("process_noise", 0.1)); man_obsn = float(man_cfg.get("obs_noise", 0.5))
        man_nlgain = float(man_cfg.get("obs_nonlin_gain", 0.0))
        man_s2 = (man_pop ** 2) * (man_C * man_C).sum(1)
        man_human = set(config["actions"].get("human_channels") or [])
    mixed_cfg = oracle_cfg.get("mixed_channel_pref")
    mixed_w_obs = float((mixed_cfg or {}).get("w_obs", 1.0))
    mixed_w_hid = float((mixed_cfg or {}).get("w_hid", 1.0))
    mixed_sigma = float(((mixed_cfg or {}).get("hidden") or {}).get("sigma", 1.0))
    borrower_lookup = borrowers.set_index("borrower_id")
    hidden_lookup = borrower_hidden.set_index("borrower_id")

    action_rows = []
    audit_rows = []
    panel_rows = []
    truth_rows = []
    action_num = action_start

    for loan in loans.to_dict("records"):
        state = str(loan["initial_status"])
        balance = float(loan["outstanding_balance"])
        dpd = int(loan["days_past_due"])
        liq_days = int(loan["days_in_credito_em_liquidacao"])
        loan_age = int(loan["loan_age_months"])
        remaining = int(loan["remaining_maturity_months"])
        contacts_last_3m = 0
        fatigue = 0.0
        f_ch = {c: 0.0 for c in split_channels}
        last_disc_seen = 0.0
        js_done = False

        borrower = borrower_lookup.loc[loan["borrower_id"]]
        hidden = hidden_lookup.loc[loan["borrower_id"]]

        loan_fatigue_beta = None
        if fat_cfg and fat_cfg.get("beta_heterogeneity"):
            het_feature = str(fat_cfg["beta_heterogeneity"].get("feature", "monthly_income"))
            loan_fatigue_beta = float(
                oracle_response.fatigue_beta(float(borrower[het_feature]), oracle_cfg)
            )

        rc = fg = inc_n = fin_n = 0.0
        if lat_cfg:
            inc_n = float(np.tanh((float(borrower["monthly_income"]) - 3000.0) / 3000.0))
            fin_n = float(np.tanh((float(loan["original_principal"]) - 8000.0) / 8000.0))
            rc = max(0.0, 0.30 + 0.20 * inc_n + float(rng.normal(0.0, 0.15)))
            fg = max(0.0, 0.20 + float(rng.normal(0.0, 0.15)))
            theta = float(rng.normal(0.0, float(lat_cfg.get("respo_bias_std", 0.0))))
        lat_cap = float(lat_cfg.get("cap", 2.0)) if lat_cfg else 0.0
        if bt_cfg:
            ic = bt_cfg.get("income_coupling")
            if ic:
                center = float(ic.get("income_center", 3000.0))
                scale = float(ic.get("income_scale", 3000.0))
                amp = float(ic.get("amp", 0.08))
                p_b_base = float(bt_w[1] / (bt_w[0] + bt_w[1])) if (bt_w[0] + bt_w[1]) > 0 else 0.5
                tilt = float(np.tanh((float(borrower["monthly_income"]) - center) / scale))
                p_b = float(np.clip(p_b_base + amp * tilt, 0.02, 0.98))
                btype = 1 if rng.random() < p_b else 0
            else:
                btype = int(rng.choice(3, p=bt_w))
        else:
            btype = 0
        prof_cfg = oracle_cfg.get("profiles")
        profile = oracle_response.draw_profile(oracle_cfg, int(borrower["age"]), rng) if prof_cfg else 0
        mixed_sobs = mixed_tau = mixed_pref = mixed_hidden = 0.0
        if mixed_cfg:
            mixed_sobs = float(oracle_response.mixed_observed_score(
                {
                    "monthly_income": float(borrower["monthly_income"]),
                    "wallet_type": str(borrower["wallet_type"]) if "wallet_type" in borrower else "",
                    "purchase_profile": (
                        str(borrower["purchase_profile"]) if "purchase_profile" in borrower else ""
                    ),
                },
                mixed_cfg,
            ))
            mixed_tau = float(rng.normal(0.0, mixed_sigma))
            mixed_hidden = mixed_w_hid * mixed_tau
            mixed_pref = mixed_w_obs * mixed_sobs + mixed_hidden
        annoy = 0.0
        eng = 0.0
        z_man = (man_rng.normal(0.0, 1.0, man_n) * man_pop) if man_cfg else None

        for month in range(months):
            if state in TERMINAL_STATES:
                break
            respo = float(np.clip(rc - fg + theta, -lat_cap, lat_cap)) if lat_cfg else 0.0
            planned = (plan_lookup.get((str(loan["loan_id"]), month))
                       if plan_lookup is not None else None)
            if planned is not None:
                channel, discount, is_exploration, explore_prob, n_allowed = _plan_action(
                    config, rng, planned, state, liq_days, month, dpd, balance,
                    int(loan.get("tier", 0) or 0), plan_epsilon,
                )
            else:
                channel, discount, is_exploration, explore_prob, n_allowed = _choose_action(
                    config,
                    rng,
                    dpd,
                    float(hidden["payment_friction"]),
                    float(hidden["contactability"]),
                    state,
                    liq_days,
                    str(borrower["payment_plan"]) if "payment_plan" in borrower else "",
                    borrower_type=btype,
                    mixed_hidden=mixed_hidden,
                    month=month,
                    balance=balance,
                    tier=int(loan.get("tier", 0) or 0),
                )
            action_id = f"A{action_num:08d}"
            settlement_offer = settlement_amount(
                config, channel, balance, discount, float(loan.get("fund_avail", 0.0) or 0.0)
            )
            public_labels = bool(
                (config["actions"].get("exploration", {}) or {}).get("public_labels", True)
            )
            act_row = {
                "action_id": action_id,
                "loan_id": loan["loan_id"],
                "month": month,
                "channel": channel,
                "discount_offer": discount,
                "settlement_offer_amount": settlement_offer,
                "contact_cost": round(
                    float((config["actions"].get("channel_costs") or {}).get(channel, 0.0)), 4
                ),
            }
            label_cols = {
                "is_exploration": bool(is_exploration),
                "exploration_epsilon": round(float(explore_prob), 4),
                "n_allowed_channels": int(n_allowed),
            }
            if public_labels:
                act_row.update(label_cols)
            else:
                audit_rows.append({"loan_id": loan["loan_id"], "month": month, **label_cols})
            action_rows.append(act_row)

            noise = rng.normal(0.0, float(oracle_cfg["noise_std"]))
            inputs = RecoveryInputs(
                channel=channel,
                discount_offer=discount,
                age=int(borrower["age"]),
                loan_age_months=loan_age,
                days_past_due=dpd,
                days_in_credito_em_liquidacao=liq_days,
                borrower_resilience=float(hidden["borrower_resilience"]),
                payment_friction=float(hidden["payment_friction"]),
                regional_macro_stress=float(hidden["regional_macro_stress"]),
                contactability=float(hidden["contactability"]),
                contact_fatigue=fatigue,
                contact_fatigue_own=(
                    _split_own(f_ch, channel) if split_cfg else None),
                last_discount_seen=last_disc_seen,
                fatigue_beta=loan_fatigue_beta,
                latent_respo=respo,
                borrower_type=btype,
                profile=profile,
                manifold_z=(tuple(z_man) if man_cfg else ()),
                mixed_pref=mixed_pref,
                owed_amount=balance,
                monthly_income=float(borrower["monthly_income"]),
                original_principal=float(loan["original_principal"]),
                contacts_last_3m=float(contacts_last_3m),
                noise=noise,
                wallet_type=str(borrower["wallet_type"]) if "wallet_type" in borrower else "",
                purchase_profile=(
                    str(borrower["purchase_profile"]) if "purchase_profile" in borrower else ""
                ),
                payment_plan=str(borrower["payment_plan"]) if "payment_plan" in borrower else "",
            )
            p_recovery = recovery_probability(inputs, oracle_cfg)
            p_chargeoff = _chargeoff_probability(liq_days, oracle_cfg)
            u_recovery = rng.random()
            u_chargeoff = rng.random()
            recovered = u_recovery < p_recovery
            charged_off = (not recovered) and u_chargeoff < p_chargeoff

            recovered_amount = 0.0
            next_state = state
            js_hit = False
            if recovered:
                if oracle_cfg.get("classical_policy"):
                    paid_share = 1.0
                elif "paid_fraction" in oracle_cfg:
                    mean_frac = float(
                        oracle_response.paid_fraction(channel, discount, oracle_cfg)
                    )
                    mean_frac = min(1.0, max(1e-3, mean_frac))
                    if mean_frac >= 0.999:
                        paid_share = 1.0
                    else:
                        kappa = float(oracle_cfg["paid_fraction"].get("noise_kappa", 12.0))
                        paid_share = float(rng.beta(mean_frac * kappa, (1.0 - mean_frac) * kappa))
                else:
                    paid_share = min(1.0, 0.7 + rng.beta(2.0, 5.0))
                recovered_amount = round(settlement_offer * paid_share, 2)
                balance = max(0.0, balance - recovered_amount)
                dpd = 0
                liq_days = 0
                cure_threshold = max(50.0, 0.05 * loan["original_principal"])
                next_state = "cured" if balance < cure_threshold else "performing"
            elif charged_off:
                next_state = "charged_off"
            else:
                balance = balance * (1.0 + float(loan["interest_rate_monthly"]) * 0.35)
                dpd += 30
                if dpd >= 90:
                    liq_days += 30
                    next_state = "credito_em_liquidacao"
                elif dpd >= 30:
                    next_state = "delinquent"
                else:
                    next_state = "performing"
                if (js_cfg and js_rate > 0.0 and not js_done
                        and rng.random() < js_rate):
                    js_hit = True
                    js_done = True
                    dpd += js_dpd
                    liq_days += js_liq
                    next_state = "credito_em_liquidacao"

            cp_cost = oracle_cfg.get("classical_policy")
            if cp_cost is not None:
                if channel == "phone":
                    cc = cp_cost.get("call_cost") or {"fixed": 30.0, "commission": 0.20}
                    act_row["contact_cost"] = round(
                        float(cc.get("fixed", 30.0))
                        + float(cc.get("commission", 0.20)) * recovered_amount, 4)
                else:
                    act_row["contact_cost"] = round(
                        float((cp_cost.get("channel_costs") or {}).get(channel, 0.0)), 4)
            elif "commission_cost" in oracle_cfg:
                comm = oracle_response.commission_rate(
                    channel, float(inputs.days_past_due), oracle_cfg
                )
                if comm:
                    act_row["contact_cost"] = round(
                        act_row["contact_cost"] + float(comm) * recovered_amount, 4)

            prow = {
                "loan_id": loan["loan_id"],
                "borrower_id": loan["borrower_id"],
                "month": month,
                "status": state,
                "next_status": next_state,
                "loan_age_months": loan_age,
                "remaining_maturity_months": remaining,
                "days_past_due": dpd if recovered else int(dpd),
                "days_in_credito_em_liquidacao": liq_days,
                "outstanding_balance": round(balance, 2),
                "contacts_last_3m": contacts_last_3m,
                "action_id": action_id,
                "recovered": bool(recovered),
                "charged_off": bool(charged_off),
                "recovered_amount": recovered_amount,
            }
            if fat_cfg:
                prow["contact_fatigue"] = round(fatigue, 6)
            if split_cfg:
                for c in split_channels:
                    prow[f"contact_fatigue_{c}"] = round(f_ch[c], 6)
            if dfat_cfg:
                prow["last_discount_seen"] = round(last_disc_seen, 6)
            if js_cfg:
                prow["judicial_shock"] = js_hit
            if man_cfg:
                _lin = man_C @ z_man
                if man_nlgain:
                    _lin = _lin + man_nlgain * (_lin ** 2 - man_s2)
                obs_vec = _lin + man_rng.normal(0.0, man_obsn, man_m)
                for _i in range(man_m):
                    prow[f"obs_{_i}"] = float(obs_vec[_i])
            panel_rows.append(prow)
            trow = {
                "seed": seed,
                "loan_id": loan["loan_id"],
                "borrower_id": loan["borrower_id"],
                "month": month,
                "borrower_resilience": float(hidden["borrower_resilience"]),
                "payment_friction": float(hidden["payment_friction"]),
                "regional_macro_stress": float(hidden["regional_macro_stress"]),
                "contactability": float(hidden["contactability"]),
                "p_recovery": p_recovery,
                "p_chargeoff": p_chargeoff,
                "recovery_draw": u_recovery,
                "chargeoff_draw": u_chargeoff,
            }
            if loan_fatigue_beta is not None:
                trow["fatigue_beta"] = loan_fatigue_beta
            if lat_cfg:
                trow["latent_respo"] = respo
            if bt_cfg:
                trow["borrower_type"] = btype
            if prof_cfg:
                trow["profile"] = profile
            if man_cfg:
                for _i in range(man_n):
                    trow[f"zman_{_i}"] = float(z_man[_i])
            if mixed_cfg:
                trow["mixed_pref"] = mixed_pref
                trow["mixed_sobs"] = mixed_sobs
                trow["mixed_tau"] = mixed_tau
            truth_rows.append(trow)

            if bt_cfg:
                oco = bt_cfg.get("optout")
                if oco:
                    if channel != "none":
                        add = float(oco.get("increment", 1.0))
                        if channel in bt_human:
                            add *= float(oco.get("human_multiplier", 1.0))
                        annoy = float(oco.get("decay", 0.5)) * annoy + add
                        if not recovered and next_state not in TERMINAL_STATES:
                            tmult = float(oco.get("type_c_mult", 1.0)) if btype == 2 else 1.0
                            optp = float(oco.get("base_coef", 0.1)) * annoy * tmult
                            if channel in bt_human:
                                optp *= float(oco.get("human_risk", 1.0))
                            if rng.random() < min(float(oco.get("cap", 0.9)), max(0.0, optp)):
                                next_state = "charged_off"
                    else:
                        annoy = float(oco.get("decay", 0.5)) * annoy
                if float(bt_cfg.get("drift", 0.0)) > 0.0 and rng.random() < float(bt_cfg["drift"]):
                    btype = int(rng.choice(3, p=bt_w))
                pr = bt_cfg.get("promotion")
                if pr:
                    gentle = set(pr.get("gentle") or [])
                    if channel in bt_human:
                        eng = 0.0
                    elif channel in gentle:
                        eng = eng + 1.0
                    else:
                        eng = float(pr.get("decay", 0.7)) * eng
                    if eng >= float(pr.get("threshold", 2.0)):
                        btype = int(pr.get("engaged_type", 0))
                    elif channel in bt_human:
                        btype = int(pr.get("burned_type", 2))
                    else:
                        btype = int(pr.get("mid_type", 1))
            if man_cfg:
                w = 0.0 if channel == "none" else (1.0 if channel in man_human else 0.4)
                if man_rot is not None:
                    z_man = (man_A @ (man_rot[w] @ z_man)
                             + man_B * w + man_rng.normal(0.0, man_proc, man_n))
                else:
                    z_man = man_A @ z_man + man_B * w + man_rng.normal(0.0, man_proc, man_n)

            state = next_state
            loan_age += 1
            remaining = max(remaining - 1, 0)
            contacts_last_3m = min(3, contacts_last_3m + (0 if channel == "none" else 1))
            if fat_cfg:
                if channel == "none":
                    add = 0.0
                else:
                    add = float(fat_cfg.get("increment", 1.0))
                    if channel in fat_human:
                        add *= float(fat_cfg.get("human_multiplier", 1.0))
                fatigue = float(fat_cfg.get("decay", 0.5)) * fatigue + add
                if split_cfg:
                    dec = float(fat_cfg.get("decay", 0.5))
                    for c in f_ch:
                        f_ch[c] = dec * f_ch[c] + (add if c == channel else 0.0)
            if dfat_cfg and channel != "none":
                last_disc_seen = float(discount)
            if lat_cfg:
                if channel == "none":
                    w, gentle = 0.0, 0.0
                elif channel in fat_human:
                    w, gentle = 1.0, 0.0
                else:
                    w, gentle = 0.4, 0.4
                fg = (float(lat_cfg.get("decay_fatigue", 0.6)) * fg
                      + float(lat_cfg.get("gain_fatigue", 0.5)) * w
                        * (1.0 + float(lat_cfg.get("income_impatience", 0.6)) * max(inc_n, 0.0))
                      + float(rng.normal(0.0, float(lat_cfg.get("process_noise", 0.05)))))
                rc = (float(lat_cfg.get("decay_receptivity", 0.7)) * rc
                      + float(lat_cfg.get("gain_receptivity", 0.5)) * gentle
                        * (1.0 + float(lat_cfg.get("financing_curvature", 0.5)) * fin_n * fin_n)
                      + float(rng.normal(0.0, float(lat_cfg.get("process_noise", 0.05)))))
                fg = min(lat_cap, max(0.0, fg))
                rc = min(lat_cap, max(0.0, rc))
            action_num += 1

    panel = pd.DataFrame(panel_rows)
    if not panel.empty:
        bad_states = set(panel["status"]).union(panel["next_status"]) - ALLOWED_STATES
        if bad_states:
            raise ValueError(f"generated invalid states: {sorted(bad_states)}")
    tables = {
        "actions": pd.DataFrame(action_rows),
        "monthly_panel": panel,
        "oracle_truth": pd.DataFrame(truth_rows),
    }
    if audit_rows:
        tables["logging_audit"] = pd.DataFrame(audit_rows)
    return tables


def _chargeoff_probability(liq_days: int, oracle_cfg: dict[str, Any]) -> float:
    threshold = float(oracle_cfg["chargeoff_after_liquidacao_days"])
    if liq_days <= 0:
        return 0.002
    pressure = (liq_days - threshold) / 90.0
    return min(0.60, max(0.005, _sigmoid(-3.5 + pressure)))
