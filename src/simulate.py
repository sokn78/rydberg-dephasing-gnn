"""
Run all the simulations once and save them to data/simulations.parquet (~5 min).

Re-run only if graphs, thetas, wait times or shots change in rydberg/config.py.
Then run analysis.py, which only reads the parquet file.

    uv run python src/simulate.py
"""

import polars as pl

from rydberg.config import DATA_DIR, SIMULATION_FILE
from rydberg.physics import build_simulation_table

simulation_table = build_simulation_table()
DATA_DIR.mkdir(exist_ok=True)
simulation_table.write_parquet(SIMULATION_FILE)

print(f"\nSaved {simulation_table.shape[0]} rows to {SIMULATION_FILE}")
print(simulation_table.group_by("split").agg(
    pl.col("graph").n_unique().alias("graphs"),
    pl.col("theta").n_unique().alias("distinct thetas"),
    pl.len().alias("rows")))
