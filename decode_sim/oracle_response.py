from __future__ import annotations

from typing import Any

import numpy as np

Num = "float | np.ndarray"


def _f(x):
    return np.asarray(x, dtype=np.float64) if isinstance(x, np.ndarray) else float(x)


def _chan_map(channel, table: dict[str, Any], default: float = 0.0):
    if isinstance(channel, np.ndarray):
        return np.array([float(table.get(str(c), default)) for c in channel], dtype=np.float64)
    return float(table.get(str(channel), default))


PAYMENT_PLAN_HARDSHIP = {"installment": -0.6, "bullet": 1.0, "mixed": 0.4, "revolving": -0.2}
_PP_OUTCOME_SCALE = 0.9
_PP_INTENSITY_SCALE = 1.2


def payment_plan_outcome(plan, oc: dict[str, Any]):
    strength = float(oc.get("payment_plan_confounding", 0.0))
    if strength == 0.0:
        return np.zeros(len(plan)) if isinstance(plan, np.ndarray) else 0.0
    h = _chan_map(plan, PAYMENT_PLAN_HARDSHIP, 0.0)
    return -strength * _PP_OUTCOME_SCALE * h


def payment_plan_intensity(plan: str, config: dict[str, Any]) -> float:
    strength = float(config["oracle"].get("payment_plan_confounding", 0.0))
    if strength == 0.0:
        return 0.0
    return strength * _PP_INTENSITY_SCALE * float(PAYMENT_PLAN_HARDSHIP.get(str(plan), 0.0))


def paid_fraction(channel, discount, oc: dict[str, Any]):
    spec = oc.get("paid_fraction")
    if not spec:
        return np.ones_like(_f(discount)) if isinstance(discount, np.ndarray) else 1.0
    base = float(spec.get("base", 1.0))
    ce = _chan_map(channel, spec.get("channel_effects") or {}, 1.0)
    dp = float(spec.get("discount_penalty", 0.0))
    lo = float(spec.get("min", 0.0))
    frac = base * ce - dp * _f(discount)
    return np.clip(frac, lo, 1.0)


def commission_rate(channel, dpd, oc: dict[str, Any]):
    spec = oc.get("commission_cost")
    d_arr = np.atleast_1d(np.asarray(dpd, dtype=np.float64))
    scalar_d = not isinstance(dpd, np.ndarray)
    if not spec:
        return 0.0 if scalar_d and not isinstance(channel, np.ndarray) else np.zeros(
            len(channel) if isinstance(channel, np.ndarray) else d_arr.shape[0]
        )
    chans = {str(c) for c in (spec.get("channels") or [])}
    tiers = sorted(spec.get("tiers") or [], key=lambda t: float(t["max_dpd"]))
    if not tiers or not chans:
        return 0.0 if scalar_d and not isinstance(channel, np.ndarray) else np.zeros(
            len(channel) if isinstance(channel, np.ndarray) else d_arr.shape[0]
        )
    rate = np.full(d_arr.shape, float(tiers[-1]["rate"]), dtype=np.float64)
    for t in reversed(tiers[:-1]):
        rate = np.where(d_arr <= float(t["max_dpd"]), float(t["rate"]), rate)
    if isinstance(channel, np.ndarray):
        member = np.isin(channel.astype(str), list(chans))
        out = np.where(member, rate if rate.shape == member.shape else rate[0], 0.0)
        return out
    if str(channel) not in chans:
        return 0.0 if scalar_d else np.zeros_like(d_arr)
    return float(rate[0]) if scalar_d else rate


def offer_response(discount, oc: dict[str, Any]):
    oe = float(oc["offer_effect"])
    sens = float(oc["offer_sensitivity"])
    curve = oc.get("offer_curve") or {"shape": "saturating"}
    shape = curve.get("shape", "saturating")
    base = oe * (1.0 - np.exp(-sens * discount))
    if shape == "saturating":
        return base
    if shape == "hump":
        peak = float(curve.get("peak", 0.3))
        penalty = float(curve.get("penalty", 0.0))
        over = np.maximum(_f(discount) - peak, 0.0)
        return base - penalty * over * over
    if shape == "logistic_decay":
        gain = float(curve.get("gain", oe))
        peak = float(curve.get("peak", 0.2))
        width = float(curve.get("width", 0.15))
        d = _f(discount)
        bump = np.exp(-(((d - peak) / width) ** 2))
        bump0 = np.exp(-(((0.0 - peak) / width) ** 2))
        return gain * (bump - bump0)
    raise ValueError(f"unknown oracle.offer_curve.shape: {shape!r}")


def interaction_term(channel: str, discount, feats: dict[str, Any], oc: dict[str, Any]):
    terms = oc.get("interactions") or []
    total = np.zeros_like(_f(discount)) if isinstance(discount, np.ndarray) else 0.0
    for t in terms:
        kind = t["kind"]
        coef = float(t["coef"])
        if kind == "discount_x_feature":
            col, val = t["feature"]["col"], t["feature"]["value"]
            fv = feats.get(col)
            if fv is None:
                continue
            match = (np.asarray(fv).astype(str) == str(val)).astype(np.float64) if isinstance(
                fv, np.ndarray) else float(str(fv) == str(val))
            total = total + coef * _f(discount) * match
        elif kind == "channel_x_feature":
            chans = set(t["channels"])
            col, val = t["feature"]["col"], t["feature"]["value"]
            fv = feats.get(col)
            if fv is None:
                continue
            fmatch = (np.asarray(fv).astype(str) == str(val)).astype(np.float64) if isinstance(
                fv, np.ndarray) else float(str(fv) == str(val))
            if isinstance(channel, np.ndarray):
                cmatch = np.array([1.0 if str(c) in chans else 0.0 for c in channel])
            else:
                cmatch = 1.0 if str(channel) in chans else 0.0
            total = total + coef * cmatch * fmatch
        elif kind == "channel_x_numeric":
            scale = float(t.get("scale", 1.0))
            chan_scale = _chan_map(channel, t.get("channel_scale") or {}, 0.0)
            num = feats.get(t["numeric"])
            if num is None:
                continue
            total = total + coef * chan_scale * (_f(num) / scale)
        else:
            raise ValueError(f"unknown interaction kind: {kind!r}")
    return total


def channel_heterogeneity(channel, feats: dict[str, Any], oc: dict[str, Any]):
    spec = oc.get("channel_heterogeneity")
    if not spec:
        return 0.0
    driver = feats.get(spec["driver"])
    if driver is None:
        return 0.0
    center = float(spec.get("center", 0.0))
    coef = _chan_map(channel, spec.get("coef") or {}, 0.0)
    return coef * (_f(driver) - center)


def offer_gate(channel, oc: dict[str, Any]):
    if not oc.get("offer_requires_contact"):
        if isinstance(channel, np.ndarray):
            return np.ones(len(channel))
        return 1.0
    return _chan_map(channel, {"none": 0.0}, 1.0)


def fatigue_beta(income, oc: dict[str, Any]):
    cfg = oc.get("contact_fatigue") or {}
    beta = float(cfg.get("beta", 0.0))
    het = cfg.get("beta_heterogeneity")
    if not het:
        if isinstance(income, np.ndarray):
            return np.full_like(_f(income), beta, dtype=float)
        return beta
    amp = float(het.get("amp", 0.0))
    center = float(het.get("center", 3000.0))
    scale = float(het.get("scale", 3000.0))
    return np.maximum(0.0, beta * (1.0 + amp * np.tanh((_f(income) - center) / scale)))


def fatigue_term(channel, contact_fatigue, oc: dict[str, Any], beta_override=None,
                 fatigue_own=None):
    cfg = oc.get("contact_fatigue")
    if not cfg:
        return np.zeros_like(_f(contact_fatigue)) if isinstance(contact_fatigue, np.ndarray) else 0.0
    beta = float(cfg.get("beta", 0.0)) if beta_override is None else _f(beta_override)
    is_contact = _chan_map(channel, {"none": 0.0}, 1.0)
    split = cfg.get("channel_split")
    if split and fatigue_own is not None:
        s = float(split.get("share", 0.0))
        f_eff = (1.0 - s) * _f(contact_fatigue) + s * _f(fatigue_own)
    else:
        f_eff = _f(contact_fatigue)
    return -beta * f_eff * is_contact


def discount_fatigue_term(channel, discount, last_discount_seen, oc: dict[str, Any]):
    cfg = oc.get("discount_fatigue")
    d = _f(discount)
    if not cfg:
        return np.zeros_like(d) if isinstance(d, np.ndarray) else 0.0
    gamma = float(cfg.get("gamma", 0.0))
    is_contact = _chan_map(channel, {"none": 0.0}, 1.0)
    drop = np.maximum(0.0, _f(last_discount_seen) - d)
    return -gamma * drop * is_contact


def latent_term(channel, latent_respo, oc: dict[str, Any]):
    cfg = oc.get("latent_response")
    if not cfg:
        return np.zeros_like(_f(latent_respo)) if isinstance(latent_respo, np.ndarray) else 0.0
    coef = float(cfg.get("coef", 0.0))
    is_contact = _chan_map(channel, {"none": 0.0}, 1.0)
    return coef * _f(latent_respo) * is_contact


_TYPE_NAMES = ("A", "B", "C")


def _feature_index(feats: dict[str, Any], weights_cfg: dict[str, Any]):
    total = None
    for name, spec in (weights_cfg or {}).items():
        f = np.asarray(feats[name], dtype=float)
        v = (f - float(spec.get("center", 0.0))) / float(spec.get("scale", 1.0)) \
            * float(spec["w"])
        total = v if total is None else total + v
    return 0.0 if total is None else total


def oblique_term(channel, feats: dict[str, Any], oc: dict[str, Any]):
    cfg = oc.get("oblique_gain")
    if not cfg:
        return 0.0
    g = np.tanh(_feature_index(feats, cfg["weights"]) - float(cfg.get("cutoff", 0.0)))
    amp = cfg.get("amplitude") or {}
    if isinstance(channel, np.ndarray):
        a = np.array([float(amp.get(str(c), amp.get("default", 0.0))) for c in channel])
    else:
        a = float(amp.get(str(channel), amp.get("default", 0.0)))
    return a * g


def offer_curvature_term(discount, feats: dict[str, Any], oc: dict[str, Any]):
    cfg = oc.get("offer_curvature")
    if not cfg:
        return 0.0
    th, ka = cfg.get("theta") or {}, cfg.get("kappa") or {}
    d = np.asarray(discount, dtype=float)
    theta = float(th.get("base", 0.0)) + _feature_index(feats, th.get("weights"))
    kappa = float(ka.get("base", 0.0)) + _feature_index(feats, ka.get("weights"))
    kappa = np.maximum(kappa, float(ka.get("min", 0.1)))
    out = theta * d - kappa * d * d
    return out if isinstance(out, np.ndarray) and out.ndim > 0 else float(out)


def profile_term(channel, profile, oc: dict[str, Any]):
    cfg = oc.get("profiles")
    pr = np.asarray(profile)
    if not cfg:
        return np.zeros(pr.shape) if pr.ndim > 0 else 0.0
    names = list(cfg["names"])
    gains = cfg["gains"]

    def g(pname, ch):
        tbl = gains[pname]
        return float(tbl.get(str(ch), tbl.get("default", 0.0)))

    if isinstance(channel, np.ndarray):
        return np.array([g(names[int(i)], c) for i, c in zip(pr, channel)])
    if pr.ndim > 0:
        col = np.array([g(n, channel) for n in names])
        return col[pr.astype(int)]
    return g(names[int(pr)], channel)


def profile_offer_scale(profile, oc: dict[str, Any]):
    cfg = oc.get("profiles") or {}
    scales = cfg.get("offer_scale")
    pr = np.asarray(profile)
    if not scales:
        return np.ones(pr.shape) if pr.ndim > 0 else 1.0
    names = list(cfg["names"])
    default = float(scales.get("default", 1.0))
    col = np.array([float(scales.get(n, default)) for n in names])
    if pr.ndim > 0:
        return col[pr.astype(int)]
    return float(col[int(pr)])


def draw_profile(oc: dict[str, Any], age: int, rng) -> int:
    cfg = oc.get("profiles") or {}
    n = len(cfg["names"])
    bands = cfg.get("weights_by_age")
    if bands:
        for band in bands:
            if age <= int(band["max_age"]):
                w = np.asarray(band["weights"], dtype=float)
                return int(rng.choice(n, p=w / w.sum()))
    w = np.asarray(cfg.get("weights", [1.0] * n), dtype=float)
    return int(rng.choice(n, p=w / w.sum()))
def type_term(channel, borrower_type, oc: dict[str, Any]):
    cfg = oc.get("borrower_type")
    if not cfg:
        return np.zeros_like(_f(borrower_type)) if isinstance(borrower_type, np.ndarray) else 0.0
    gains = cfg["channel_gain"]; human = set(cfg.get("human") or [])
    G = np.array([[float(gains[t].get("none", 0.0)), float(gains[t].get("digital", 0.0)),
                   float(gains[t].get("human", 0.0))] for t in _TYPE_NAMES])
    cls = lambda c: 0 if str(c) == "none" else (2 if str(c) in human else 1)
    bt = np.asarray(borrower_type)
    if isinstance(channel, np.ndarray):
        return G[bt.astype(int), np.array([cls(c) for c in channel])]
    cl = cls(channel)
    if bt.ndim > 0:
        return G[bt.astype(int), cl]
    return float(G[int(bt), cl])


def mixed_observed_score(feats: dict[str, Any], cfg: dict[str, Any]):
    obs = cfg.get("observed") or {}
    total: Any = 0.0
    inc = obs.get("income")
    if inc and feats.get("monthly_income") is not None:
        center = float(inc.get("center", 3000.0))
        scale = float(inc.get("scale", 3000.0))
        coef = float(inc.get("coef", 0.0))
        z = (_f(feats["monthly_income"]) - center) / scale
        if inc.get("nonlin"):
            z = np.tanh(z)
        total = total + coef * z
    for col in ("wallet_type", "purchase_profile"):
        m = obs.get(col)
        if m and feats.get(col) is not None:
            total = total + _chan_map(feats[col], m, 0.0)
    return total


def mixed_pref_term(channel, pref, oc: dict[str, Any]):
    cfg = oc.get("mixed_channel_pref")
    if not cfg:
        return np.zeros_like(_f(pref)) if isinstance(pref, np.ndarray) else 0.0
    human = set(cfg.get("human") or [])
    cc = cfg.get("class_coef") or {"none": 0.0, "digital": -1.0, "human": 1.0}

    def _coef(c):
        cl = "none" if str(c) == "none" else ("human" if str(c) in human else "digital")
        return float(cc.get(cl, 0.0))

    if isinstance(channel, np.ndarray):
        cvec = np.array([_coef(c) for c in channel], dtype=np.float64)
        return cvec * _f(pref)
    return _coef(channel) * _f(pref)


_MANIFOLD_CACHE: dict[Any, Any] = {}

W_INTENSITIES = (0.0, 0.4, 1.0)
def _manifold(cfg: dict[str, Any]):
    n = int(cfg["n_latent"]); m = int(cfg["obs_dim"]); chans = tuple(cfg["channels"]); seed = int(cfg.get("seed", 0))
    contrast = float(cfg.get("w_contrast", 0.0))
    bilinear = float(cfg.get("bilinear_strength", 0.0))
    key = (n, m, chans, seed, float(cfg.get("decay", 0.85)), float(cfg.get("obs_gain", 0.8)), float(cfg.get("action_gain", 0.3)), contrast, bilinear)
    if key not in _MANIFOLD_CACHE:
        rng = np.random.default_rng(seed)
        A = float(cfg.get("decay", 0.85)) * np.eye(n) + (0.05 / np.sqrt(n)) * rng.standard_normal((n, n))
        B = float(cfg.get("action_gain", 0.3)) * rng.standard_normal(n)
        C = float(cfg.get("obs_gain", 0.8)) * rng.standard_normal((m, n))
        W = {c: rng.standard_normal(n) for c in chans}
        if contrast:
            contact = [c for c in chans if c != "none"]
            mean_w = np.mean([W[c] for c in contact], axis=0)
            for c in contact:
                W[c] = W[c] - contrast * mean_w
        rot = None
        if bilinear:
            from scipy.linalg import expm
            nA = float(np.linalg.norm(A, 2))
            if nA >= 1.0:
                raise ValueError(
                    f"bilinear_strength > 0 requires ||A||_2 < 1 for "
                    f"stability; got {nA:.4f} — adjust decay/perturbation")
            G = rng.standard_normal((n, n))
            S = bilinear * (G - G.T) / (2.0 * np.sqrt(n))
            rot = {w_: expm(w_ * S) for w_ in W_INTENSITIES}
        _MANIFOLD_CACHE[key] = (n, m, chans, A, B, C, W, rot)
    return _MANIFOLD_CACHE[key]

def manifold_term(channel, z, oc: dict[str, Any]):
    cfg = oc.get("manifold")
    if not cfg:
        return np.zeros(len(z)) if (isinstance(z, np.ndarray) and z.ndim == 2) else 0.0
    n, m, chans, A, B, C, W, _Delta = _manifold(cfg); coef = float(cfg.get("coef", 1.0)); z = np.asarray(z, dtype=np.float64)
    if isinstance(channel, np.ndarray):
        Wm = np.array([W[str(c)] for c in channel])
        return coef * (z * Wm).sum(1)
    w = W[str(channel)]
    if z.ndim == 2:
        return coef * (z @ w)
    return coef * float(z @ w)


_CLASSICAL_BANDS = ("none", "whatsapp", "sms", "phone")


def _classical_bmean(cp: dict[str, Any]) -> float:
    bend = cp["bend"]
    a, pk, pkpp = float(bend["curvature"]), float(bend["peak_owed"]), float(bend["peak_pp"])
    lo, hi = (float(v) for v in cp["owed_domain"])
    mean_sq = ((hi - pk) ** 3 - (lo - pk) ** 3) / (3.0 * (hi - lo))
    return pkpp - a * mean_sq


def _classical_bshape(owed, cp: dict[str, Any]):
    bend = cp["bend"]
    a, pk, pkpp = float(bend["curvature"]), float(bend["peak_owed"]), float(bend["peak_pp"])
    return (pkpp - a * (_f(owed) - pk) ** 2) - _classical_bmean(cp)


def classical_owed_domain(oc: dict[str, Any]) -> tuple[float, float]:
    lo, hi = (float(v) for v in oc["classical_policy"]["owed_domain"])
    return lo, hi


def _scorecard_recovery(act: np.ndarray, band: np.ndarray, owed, cp: dict[str, Any], scalar: bool):
    table = cp["band_channel_pp"]
    lo, hi = (float(v) for v in cp["owed_domain"])
    n_bins = int(cp.get("owed_bins", 5))
    obin = np.clip(((owed - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    p = np.zeros(band.shape[0], dtype=np.float64)
    for name, arr in table.items():
        arr = np.asarray(arr, dtype=np.float64)
        mask = act == name
        if mask.any():
            p[mask] = arr[band[mask], obin[mask]]
    lo_clip, hi_clip = (float(v) for v in cp.get("prob_clip", (1e-4, 0.40)))
    p = np.clip(p, lo_clip, hi_clip)
    return float(p[0]) if scalar else p


def classical_recovery(channel, feats: dict[str, Any], oc: dict[str, Any]):
    cp = oc.get("classical_policy")
    if cp is None:
        return None
    dpd_in, owed_in = feats["days_past_due"], feats["owed_amount"]
    scalar = not any(isinstance(v, np.ndarray) for v in (dpd_in, owed_in, channel))
    dpd = np.atleast_1d(np.asarray(_f(dpd_in), dtype=np.float64))
    owed = np.atleast_1d(np.asarray(_f(owed_in), dtype=np.float64))
    n = max(dpd.size, owed.size, (channel.size if isinstance(channel, np.ndarray) else 1))
    if dpd.size == 1 and n > 1:
        dpd = np.full(n, dpd[0])
    if owed.size == 1 and n > 1:
        owed = np.full(n, owed[0])
    act = channel.astype(str) if isinstance(channel, np.ndarray) else np.full(n, str(channel))

    th = cp["thresholds"]
    edges = [float(th["x2"]), float(th["x1"]), float(th["x"])]
    band = np.zeros(n, dtype=int)
    for e in edges:
        band += (dpd > e).astype(int)

    if cp.get("policy_family") == "scorecard":
        return _scorecard_recovery(act, band, owed, cp, scalar)

    obs = cp["obs_rates"]
    obs_by_band = np.array([float(obs[c]) for c in _CLASSICAL_BANDS], dtype=np.float64)
    b = _classical_bshape(owed, cp)
    diag_self = obs_by_band[band] + b
    p = diag_self.copy()

    below = float(cp.get("below", 0.006))
    lo_clip, hi_clip = (float(v) for v in cp.get("prob_clip", (1e-4, 0.40)))
    cl = cp["call_levels"]
    call_by_band = np.array([float(cl["none"]), float(cl["whatsapp"]), float(cl["sms"]), np.nan])

    m = (act == "phone") & (band < 3)
    p = np.where(m, call_by_band[np.clip(band, 0, 2)] + b, p)

    xover = cp["crossovers"]
    ne = xover["none_wa"]
    wa_lift = float(ne["slope"]) * (owed - float(ne["pivot"])) / 1000.0
    p = np.where((band == 0) & (act == "whatsapp"), diag_self + wa_lift, p)
    p = np.where((band == 0) & (act == "sms"), diag_self + wa_lift - below, p)
    ws = xover["wa_sms"]
    hump = float(ws["hump"]) * (owed - float(ws["pivot"])) * (float(ws["top"]) - owed) / 1.0e6
    p = np.where((band == 1) & (act == "sms"), diag_self + hump, p)
    p = np.where((band == 1) & (act == "none"), diag_self - below, p)
    p = np.where((band == 2) & ((act == "none") | (act == "whatsapp")), diag_self - below, p)
    p = np.where((band == 3) & (act != "phone"), diag_self - below, p)

    p = np.clip(p, lo_clip, hi_clip)
    return float(p[0]) if scalar else p


def classical_net_value(channel, p_recovery, owed, oc: dict[str, Any]):
    cp = oc["classical_policy"]
    fees = cp.get("channel_costs") or {}
    p, v = _f(p_recovery), _f(owed)
    cc = cp.get("call_cost") or {"fixed": 30.0, "commission": 0.20}
    fixed, comm = float(cc.get("fixed", 30.0)), float(cc.get("commission", 0.20))
    ch = channel.astype(str) if isinstance(channel, np.ndarray) else channel
    is_call = ch == "phone"
    call_net = (1.0 - comm) * p * v - fixed
    fee = _chan_map(channel, fees, 0.0)
    cheap_net = p * v - fee
    return np.where(is_call, call_net, cheap_net)

