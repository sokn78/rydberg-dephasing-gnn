# Estimating Rydberg dephasing on unseen graphs

Simulating a Rydberg-atom array is only possible for small graphs. Can a dephasing rate θ calibrated on 4–5 atom graphs still be read from measurements on **larger, unseen** graphs (6–7 atoms)?

The experiment simulates a Ramsey sequence with [Pulser](https://github.com/pasqal-io/Pulser) on subgraphs of a triangular lattice, and compares three estimators of θ, trained on small graphs and tested on graphs and θ values never seen:

| Estimator | Uses | Mean relative error |
|---|---|---|
| A. Single-atom calibration curve | the measurements only | 213 % |
| B. One calibration curve per degree | the measurements and the atom's degree | 39 % |
| C. Message-passing neural network | the measurements and the neighbourhood | 15.6 % |

The full write-up, with figures and statistics, is in **[reports/results.md](reports/results.md)**.

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv run python src/simulate.py   # ~5 min, writes data/simulations.parquet
uv run python src/analysis.py   # ~5 min, writes reports/figures/*.png and prints the tables
```

`data/simulations.parquet` is included, so `analysis.py` can be run without re-simulating. Both scripts can also be run cell by cell (`# %%`) in VS Code.

## Layout

| Path | Content |
|---|---|
| `src/rydberg/config.py` | all parameters: wait times, θ values, graph definitions |
| `src/rydberg/physics.py` | lattice graphs, couplings, Ramsey simulation |
| `src/rydberg/stats.py` | slope with confidence interval, shot noise, bootstrap |
| `src/simulate.py` | runs all simulations once, saves the parquet table |
| `src/analysis.py` | the estimators (A, B, C), the evaluation and the statistics |
| `src/plots/` | all plotting code, one module per report section |
