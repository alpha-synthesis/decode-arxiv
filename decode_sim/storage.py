from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    db_path: Path
    tables: dict[str, pd.DataFrame]


def write_duckdb(db_path: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        for name, frame in tables.items():
            con.register(f"_{name}", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
            con.unregister(f"_{name}")
    finally:
        con.close()


def table_counts(db_path: str | Path) -> dict[str, int]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        return {name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0] for name in tables}
    finally:
        con.close()
