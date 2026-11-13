from __future__ import annotations

import re

import duckdb
import pandas as pd

IDENT_COLUMNS = ("loan_id", "borrower_id", "action_id")
LEAKY_COLUMNS = (
    "post_outstanding_balance", "post_days_past_due",
    "post_days_in_credito_em_liquidacao", "post_status", "next_status",
)
_NOT_STATE = {
    "loan_id", "borrower_id", "action_id",
    "channel", "discount_offer", "settlement_offer_amount", "contact_cost",
    "is_exploration", "exploration_epsilon", "n_allowed_channels",
    "recovered", "charged_off", "recovered_amount",
    "recovered_1m", "amount_1m", "recovered_9m", "amount_9m",
    "event_time", "event_observed",
    "next_status", "post_status", "post_outstanding_balance",
    "post_days_past_due", "post_days_in_credito_em_liquidacao",
}
_SQL_WORDS = {
    "and", "or", "not", "in", "between", "is", "null", "like", "ilike", "case",
    "when", "then", "else", "end", "true", "false", "cast", "as", "abs", "coalesce",
    "least", "greatest", "round", "floor", "ceil", "nullif", "double", "integer",
    "bigint", "varchar", "boolean", "float", "decimal", "mod", "log", "exp", "sqrt",
}


class IllegalRule(ValueError):
    pass


def _rule_problem(expr, allowed, banned):
    cleaned = re.sub(r"'[^']*'", " ", expr or "")
    names = {m.group(0).lower() for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)}
    used = names - _SQL_WORDS
    bad = sorted(used & banned)
    if bad:
        return (f"conditions on {', '.join(bad)}, which is an identifier, an outcome "
                f"or post-decision state")
    unknown = sorted(used - allowed - banned)
    if unknown:
        return f"references unknown column(s): {', '.join(unknown)}"
    return None


def validate_rules(policy, available):
    allowed = {c for c in available if c not in _NOT_STATE} - set(LEAKY_COLUMNS) - set(IDENT_COLUMNS)
    banned = set(IDENT_COLUMNS) | set(LEAKY_COLUMNS) | (set(_NOT_STATE) - allowed)
    problem = _rule_problem(policy["default"].get("if", ""), allowed, banned)
    if problem:
        raise IllegalRule(f"default {problem}")
    kept = []
    for i, r in enumerate(policy["rules"]):
        problem = _rule_problem(r.get("if", ""), allowed, banned)
        if problem:
            print(f"[censored] rule {i} dropped -- {problem}", flush=True)
            continue
        kept.append(r)
    return kept


def decide(df: pd.DataFrame, policy) -> pd.DataFrame:
    kept_rules = validate_rules(policy, set(df.columns))
    con = duckdb.connect()
    con.register("cohort", df)
    cases, params = [], []
    for i, r in enumerate(kept_rules):
        cases.append(f"WHEN ({r['if']}) THEN {i}")
        params.append(r)
    case_sql = "\n".join(cases) if cases else ""
    out = con.execute(f"SELECT *, CASE {case_sql} ELSE -1 END AS _rule FROM cohort").df()
    dflt = policy["default"]
    chans, discs = [], []
    for rule_idx in out["_rule"]:
        r = params[int(rule_idx)] if int(rule_idx) >= 0 else dflt
        chans.append(r["channel"])
        discs.append(float(r.get("discount", 0.0) or 0.0))
    out["channel"] = chans
    out["discount_offer"] = discs
    return out
