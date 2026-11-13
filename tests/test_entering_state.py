from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import decode_sim.generator as gen
from decode_sim.bridge.entering_state import (
    POST_OUTCOME_FIELDS, assert_panel_aligned, state_select_sql)
from decode_sim.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "adversarial_v7_scenario_frozen.yaml"
OUTCOME_COLUMNS = {"recovered", "recovered_amount", "charged_off", "next_status",
                   "judicial_shock"}
STATE_FEATURES = ["days_past_due", "days_in_credito_em_liquidacao",
                  "outstanding_balance", "contacts_last_3m", "contact_fatigue",
                  "loan_age_months", "remaining_maturity_months"]


class _Recorder:

    def __init__(self, orig, force=None):
        self.orig, self.force, self.calls = orig, force, []

    def __call__(self, inputs, oracle_cfg):
        p = self.orig(inputs, oracle_cfg)
        if self.force is not None and len(self.calls) == self.force[0]:
            p = float(self.force[1])
        self.calls.append({"p": p, "dpd": float(inputs.days_past_due),
                           "liq": float(inputs.days_in_credito_em_liquidacao),
                           "bal": float(inputs.owed_amount)})
        return p


def _generate(out: Path, force=None):
    cfg = load_config(str(CFG))
    cfg["population"]["borrowers"] = 400
    cfg["panel"]["months"] = 10
    orig = gen.recovery_probability
    rec = _Recorder(orig, force)
    gen.recovery_probability = rec
    try:
        gen.generate(cfg, out, batch_size=None)
    finally:
        gen.recovery_probability = orig
    return rec.calls


def _panel(world: Path) -> pd.DataFrame:
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    df = con.execute("SELECT * FROM monthly_panel ORDER BY loan_id, month").df()
    con.close()
    return df


def _call_index(world: Path, calls) -> pd.DataFrame:
    con = duckdb.connect(str(world / "sim.duckdb"), read_only=True)
    tr = con.execute("SELECT loan_id, month, p_recovery FROM oracle_truth").df()
    con.close()
    pos = {p: i for i, p in enumerate(c["p"] for c in calls)}
    assert len(pos) == len(calls) == len(tr), "probabilities are not unique"
    tr["call"] = tr["p_recovery"].map(pos)
    assert tr["call"].notna().all()
    return tr.astype({"call": int})


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    base = tmp_path_factory.mktemp("entering_state")
    a = base / "A"
    calls = _generate(a)
    idx = _call_index(a, calls)
    pan = _panel(a).merge(idx[["loan_id", "month", "call"]], on=["loan_id", "month"])
    nxt = set(zip(pan["loan_id"], pan["month"] - 1))
    alive_next = np.array([(l, m) in nxt for l, m in zip(pan["loan_id"], pan["month"])])
    nr = pan[(~pan["recovered"]) & (~pan["charged_off"]) & (pan["month"] >= 2)
             & alive_next].iloc[0]
    rc = pan[pan["recovered"] & (pan["month"] >= 2)].iloc[0]
    b, c = base / "B", base / "C"
    _generate(b, force=(int(nr["call"]), 1.0))
    _generate(c, force=(int(rc["call"]), 0.0))
    return {"A": a, "B": b, "C": c, "calls_A": calls, "idx_A": idx,
            "flip_to_recovery": (nr["loan_id"], int(nr["month"])),
            "flip_to_no_recovery": (rc["loan_id"], int(rc["month"]))}


def _row(df, key):
    lid, m = key
    r = df[(df["loan_id"] == lid) & (df["month"] == m)]
    assert len(r) == 1
    return r.iloc[0]


@pytest.mark.parametrize("variant,key", [("B", "flip_to_recovery"),
                                         ("C", "flip_to_no_recovery")])
def test_intervention_changes_only_outcome_columns(worlds, variant, key):
    pa, pv = _panel(worlds["A"]), _panel(worlds[variant])
    lid, m = worlds[key]
    ra, rv = _row(pa, (lid, m)), _row(pv, (lid, m))
    assert bool(ra["recovered"]) != bool(rv["recovered"]), "outcome did not flip"
    changed = {c for c in pa.columns
               if not (pd.isna(ra[c]) and pd.isna(rv[c])) and ra[c] != rv[c]}
    assert changed <= set(POST_OUTCOME_FIELDS) | OUTCOME_COLUMNS, changed
    assert "days_past_due" in changed and "outstanding_balance" in changed
    ea = pa[(pa["loan_id"] == lid) & (pa["month"] < m)].reset_index(drop=True)
    ev = pv[(pv["loan_id"] == lid) & (pv["month"] < m)].reset_index(drop=True)
    pd.testing.assert_frame_equal(ea, ev)


def _loaders():
    from benchmark.b_demo_probe import load_panel
    from gates.v7_pilot_gap import load_frame
    return {"v7_pilot_gap.load_frame": load_frame,
            "b_demo_probe.load_panel": load_panel}


@pytest.mark.parametrize("variant,key", [("B", "flip_to_recovery"),
                                         ("C", "flip_to_no_recovery")])
@pytest.mark.parametrize("loader", ["v7_pilot_gap.load_frame",
                                    "b_demo_probe.load_panel"])
def test_loader_features_do_not_see_the_month_outcome(worlds, variant, key, loader):
    fn = _loaders()[loader]
    feats = STATE_FEATURES + ["discount_offer", "channel"]
    fa, obs, _ = fn(worlds["A"], "entering")
    fv, _, _ = fn(worlds[variant], "entering")
    cols = feats + obs
    pd.testing.assert_series_equal(_row(fa, worlds[key])[cols],
                                   _row(fv, worlds[key])[cols], check_names=False)
    ga, _, _ = fn(worlds["A"], "post_outcome_archived")
    gv, _, _ = fn(worlds[variant], "post_outcome_archived")
    ra, rv = _row(ga, worlds[key]), _row(gv, worlds[key])
    assert ra["days_past_due"] != rv["days_past_due"], \
        "the archived timing should expose the month's outcome"


def test_entering_values_equal_generator_decision_inputs(worlds):
    from benchmark.b_demo_probe import load_panel
    calls = pd.DataFrame(worlds["calls_A"])
    idx = worlds["idx_A"]
    for timing in ("entering", "post_outcome_archived"):
        df, _, _ = load_panel(worlds["A"], timing)
        mrg = df.merge(idx[["loan_id", "month", "call"]], on=["loan_id", "month"])
        assert len(mrg) == len(df) == len(calls)
        c = calls.iloc[mrg["call"].to_numpy()].reset_index(drop=True)
        same = ((mrg["days_past_due"].to_numpy(float) == c["dpd"].to_numpy())
                & (mrg["days_in_credito_em_liquidacao"].to_numpy(float)
                   == c["liq"].to_numpy())
                & (np.abs(mrg["outstanding_balance"].to_numpy(float)
                          - c["bal"].to_numpy()) <= 0.005 + 1e-9))
        if timing == "entering":
            assert same.all(), f"{(~same).sum()} rows differ from the decision inputs"
        else:
            assert same.mean() < 0.1, "archived timing unexpectedly equals the inputs"


def test_panel_alignment_guard():
    con = duckdb.connect()
    con.execute("CREATE TABLE monthly_panel AS SELECT * FROM (VALUES "
                "('L1', 0), ('L1', 1), ('L1', 2), ('L2', 0), ('L2', 1)) t(loan_id, month)")
    assert_panel_aligned(con)
    con.execute("INSERT INTO monthly_panel VALUES ('L2', 3)")
    with pytest.raises(ValueError):
        assert_panel_aligned(con)
    con.execute("DELETE FROM monthly_panel WHERE loan_id = 'L2'")
    con.execute("INSERT INTO monthly_panel VALUES ('L3', 1), ('L3', 2)")
    with pytest.raises(ValueError):
        assert_panel_aligned(con)


def test_state_sql_fragments():
    sel, join = state_select_sql("entering")
    for c in POST_OUTCOME_FIELDS:
        assert f"COALESCE(pe.{c}, l.{c}) AS {c}" in sel
    assert "pe.month = p.month - 1" in join
    sel, join = state_select_sql("post_outcome_archived")
    assert join == "" and all(f"p.{c}" in sel for c in POST_OUTCOME_FIELDS)
    with pytest.raises(ValueError):
        state_select_sql("post")


CONSUMERS = ["gates/v7_pilot_gap.py", "benchmark/b_demo_probe.py",
             "benchmark/drivers/d05_flat_decomp.py",
             "benchmark/drivers/d05_channel_decomp.py"]


@pytest.mark.parametrize("path", CONSUMERS)
def test_no_consumer_selects_current_row_state(path):
    src = (ROOT / path).read_text()
    pat = re.compile(r"\bp\.(" + "|".join(POST_OUTCOME_FIELDS) + r")\b")
    assert not pat.search(src), f"{path} selects a post-outcome field from row t"
    assert "state_select_sql(" in src


def test_transport_uses_the_timed_loader():
    src = (ROOT / "benchmark/fannie_transport.py").read_text()
    assert "load_panel(world, state_timing)" in src


def test_transport_interval_targets_the_point_statistic():
    from sklearn.metrics import roc_auc_score

    from benchmark.fannie_transport import (fold_aucs, fold_predictions,
                                            transport_statistic)
    rng = np.random.default_rng(0)
    n_loans, per = 600, 3
    groups = np.repeat(np.arange(n_loans), per)
    u = rng.normal(size=n_loans)[groups]
    x = rng.normal(size=(len(groups), 2))
    y = (rng.random(len(groups)) < 1 / (1 + np.exp(-(x[:, 0] + u)))).astype(int)
    X_flat, X_hist = x, np.column_stack([x, u + rng.normal(scale=0.5, size=len(u))])
    fold, pf, ph = fold_predictions(X_flat, X_hist, y, groups)
    assert all(len(np.unique(fold[groups == g])) == 1 for g in range(n_loans))
    af, ah = fold_aucs(y, pf, ph, fold)
    point = float(np.mean(ah - af))
    af1, ah1 = fold_aucs(y, pf, ph, fold, np.ones(len(y)))
    assert abs(float(np.mean(ah1 - af1)) - point) < 1e-12
    w = rng.integers(0, 3, size=len(y)).astype(float)
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    assert abs(roc_auc_score(y, ph, sample_weight=w) - roc_auc_score(y[rep], ph[rep])) < 1e-12
    out = transport_statistic(X_flat, X_hist, y, groups, "test", n_boot=200,
                              boot_seed=1, n_partitions=2, partition_seed=3)
    assert abs(out["dauc"] - point) < 1e-12
    assert out["ci"][0] <= out["dauc"] <= out["ci"][1]
    assert out["n_boot"] == 200 and out["boot_seed"] == 1
    assert out["partitions"]["n"] == 2 and len(out["partitions"]["points"]) == 2
    arch = transport_statistic(X_flat, X_hist, y, groups, "test", n_boot=200,
                               boot_seed=0, n_partitions=0, partition_seed=0,
                               interval="archived_pooled")
    assert abs(arch["dauc"] - point) < 1e-12 and arch["ci"][0] < arch["ci"][1]
