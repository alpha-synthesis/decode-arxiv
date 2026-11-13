import sys
from pathlib import Path
import duckdb
import numpy as np
import shutil
from decode_sim.config import load_config
from decode_sim.generator import generate

out = Path("worlds/repro_check_d05")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
cfg = load_config("configs/adversarial_v72_d05_scenario_frozen.yaml")
cfg["seed"] = 17002
cfg["population"]["borrowers"] = 10_000
cfg["panel"]["months"] = 12
generate(cfg, out, batch_size=10000)
a = duckdb.connect(str(out / "sim.duckdb"), read_only=True)
b = duckdb.connect("worlds/"
                   "adversarial_v72_d05_infer/sim.duckdb", read_only=True)
q = ("SELECT loan_id, month, zman_0, zman_3, zman_5, obs_0, obs_1 "
     "FROM oracle_truth JOIN monthly_panel USING (loan_id, month) "
     "ORDER BY loan_id, month LIMIT 40000")
qa = a.execute(q).df()
ids = qa["loan_id"].unique().tolist()
qb = b.execute(
    "SELECT loan_id, month, zman_0, zman_3, zman_5, obs_0, obs_1 "
    "FROM oracle_truth JOIN monthly_panel USING (loan_id, month) "
    "WHERE loan_id IN (SELECT UNNEST(?)) ORDER BY loan_id, month LIMIT 40000",
    [ids]).df()
a.close(); b.close()
same = qa.shape == qb.shape and (qa["loan_id"].values == qb["loan_id"].values).all()
diff = float(np.abs(qa.drop(columns=["loan_id"]).to_numpy(float)
                    - qb.drop(columns=["loan_id"]).to_numpy(float)).max()) if same else float("nan")
print(f"rows {qa.shape} vs {qb.shape} | max abs diff = {diff}")
print("BYTE-REPRODUCIBLE" if same and diff == 0.0 else "DRIFT DETECTED")
shutil.rmtree(out)
