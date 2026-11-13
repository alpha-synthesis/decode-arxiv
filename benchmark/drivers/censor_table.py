import duckdb

SRC = "worlds/adversarial_v7_train/decision_table.csv"
OUT = "worlds/adversarial_v7_train/decision_table_analyst.parquet"
DROP = {"prev_y", "prev2_y", "cum_y", "run_rate",
        "next_status", "post_status", "post_outstanding_balance",
        "post_days_past_due", "post_days_in_credito_em_liquidacao"}

con = duckdb.connect()
cols = [r[0] for r in con.execute(
    f"DESCRIBE SELECT * FROM read_csv_auto('{SRC}', sample_size=20000)").fetchall()]
missing = DROP - set(cols)
assert not missing, f"expected outcome-history columns absent: {missing}"
keep = ", ".join(f'"{c}"' for c in cols if c not in DROP)
con.execute(f"COPY (SELECT {keep} FROM read_csv_auto('{SRC}')) "
            f"TO '{OUT}' (FORMAT PARQUET)")
n = con.execute(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
print(f"censored table: {n:,} rows, {len(cols) - len(DROP)} cols "
      f"(dropped {sorted(DROP)})")
