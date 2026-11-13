from __future__ import annotations

POST_OUTCOME_FIELDS = ("days_past_due", "days_in_credito_em_liquidacao",
                       "outstanding_balance")
STATE_TIMINGS = ("entering", "post_outcome_archived")
DEFAULT_STATE_TIMING = "entering"


def check_timing(timing: str) -> str:
    if timing not in STATE_TIMINGS:
        raise ValueError(f"state timing must be one of {STATE_TIMINGS}, got {timing!r}")
    return timing


def state_select_sql(timing: str, panel: str = "p", prev: str = "pe",
                     loans: str = "l") -> tuple[str, str]:
    check_timing(timing)
    if timing == "post_outcome_archived":
        return ", ".join(f"{panel}.{c}" for c in POST_OUTCOME_FIELDS), ""
    sel = ", ".join(f"COALESCE({prev}.{c}, {loans}.{c}) AS {c}"
                    for c in POST_OUTCOME_FIELDS)
    join = (f"LEFT JOIN monthly_panel {prev} ON {prev}.loan_id = {panel}.loan_id "
            f"AND {prev}.month = {panel}.month - 1")
    return sel, join


def assert_panel_aligned(con) -> None:
    bad = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT loan_id, MIN(month) AS mn, MAX(month) AS mx, COUNT(*) AS n,
                   COUNT(DISTINCT month) AS nd
            FROM monthly_panel GROUP BY loan_id)
        WHERE mn <> 0 OR n <> mx - mn + 1 OR nd <> n""").fetchone()[0]
    if bad:
        raise ValueError(f"{bad} loans have a panel that does not start at month 0, "
                         "has gaps or repeats a month; entering values cannot be "
                         "taken from row t-1")
