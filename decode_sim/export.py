from __future__ import annotations

from pathlib import Path

import duckdb

from decode_sim.dictionaries import FIELD_DOCS, PUBLIC_TABLES, TABLE_DOCS

_DATA_DICTIONARY_FILENAME = "DATA_DICTIONARY.md"


def _write_data_dictionary(out_dir: Path, con: duckdb.DuckDBPyConnection) -> Path:
    L = [
        "# Data dictionary",
        "",
        "Auto-generated from this dataset's schema. One section per public table; every "
        "column the scenario emitted is listed with its type and meaning.",
        "",
    ]
    for table in PUBLIC_TABLES:
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        if not cols:
            continue
        L += [f"## `{table}`", "", f"_{TABLE_DOCS.get(table, '')}_", "",
              "| column | type | description |", "|---|---|---|"]
        for name, dtype in cols:
            L.append(f"| `{name}` | {dtype} | {FIELD_DOCS.get(name, '—')} |")
        L.append("")
    path = out_dir / _DATA_DICTIONARY_FILENAME
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def _flat_sql(con: duckdb.DuckDBPyConnection) -> str:
    def cols(table: str) -> set[str]:
        return {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]
        ).fetchall()}

    a_cols, b_cols = cols("actions"), cols("borrowers")
    l_cols = cols("loans")
    a_extra = "".join(
        f",\n        a.{c}"
        for c in ("is_exploration", "exploration_epsilon", "contact_cost", "n_allowed_channels")
        if c in a_cols
    )
    b_extra = "".join(
        f",\n        b.{c}"
        for c in ("wallet_type", "purchase_profile", "payment_plan") if c in b_cols
    )
    l_extra = "".join(
        f",\n        l.{c}" for c in ("tier", "fund_avail") if c in l_cols
    )
    return f"""
    SELECT
        p.*,
        a.channel,
        a.discount_offer,
        a.settlement_offer_amount{a_extra},
        l.product_type,
        l.original_principal,
        l.interest_rate_monthly{l_extra},
        b.uf,
        b.city,
        b.lat,
        b.lon,
        b.age,
        b.gender,
        b.income_band,
        b.monthly_income,
        b.dependents{b_extra}
    FROM monthly_panel p
    LEFT JOIN actions a ON a.action_id = p.action_id
    LEFT JOIN loans l ON l.loan_id = p.loan_id
    LEFT JOIN borrowers b ON b.borrower_id = p.borrower_id
    ORDER BY p.loan_id, p.month
"""


def export_csv(
    db_path: str | Path,
    out_dir: str | Path | None = None,
    mode: str = "split",
) -> list[Path]:
    db_path = Path(db_path)
    out_dir = Path(out_dir) if out_dir is not None else db_path.parent / "csv"
    modes = ("split", "flat") if mode == "both" else (mode,)
    if any(m not in ("split", "flat") for m in modes):
        raise ValueError("mode must be 'split', 'flat', or 'both'")
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        paths: list[Path] = [_write_data_dictionary(out_dir, con)]
        if "split" in modes:
            for table in PUBLIC_TABLES:
                path = out_dir / f"{table}.csv"
                con.execute(f"COPY {table} TO ? (HEADER, DELIMITER ',')", [str(path)])
                paths.append(path)
        if "flat" in modes:
            path = out_dir / "flat.csv"
            con.execute(f"COPY ({_flat_sql(con)}) TO ? (HEADER, DELIMITER ',')", [str(path)])
            paths.append(path)
    finally:
        con.close()
    return paths


def read_seed(db_path: str | Path) -> int | None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("SELECT DISTINCT seed FROM oracle_truth LIMIT 1").fetchone()
        return int(row[0]) if row else None
    finally:
        con.close()


def sample_subpath(scenario: str, seed: int | None, when) -> str:
    return (
        f"samples/{scenario}/"
        f"{when:%Y-%m-%d}/seed{seed}-{when:%H%M%S}"
    )
