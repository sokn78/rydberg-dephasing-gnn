"""
Analysis: can we read the dephasing rate θ of Rydberg atoms from their measurements,
on graphs never seen during training?

Reads data/simulations.parquet (produced by simulate.py), fits three estimators,
compares them on the test graphs, and saves every figure to reports/figures/
(used by reports/results.md).

  A. "isolated"   : invert the calibration curve of a single atom, for every atom.
  B. "per degree" : invert the calibration curve of atoms with the same degree.
  C. "GNN"        : message-passing network that also sees the neighbours.

Run as a script (uv run python src/analysis.py) or cell by cell (# %%).
All plotting code lives in src/plots/; this file only holds the logic.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.interpolate import make_smoothing_spline

from plots import save_figure
from plots.calibration import plot_calibration_by_degree, plot_inversion_example, plot_single_atom
from plots.experiment import plot_graph_family, plot_ramsey_sequence
from plots.gnn import plot_gnn_architecture, plot_training
from plots.results import plot_errors, plot_estimates_vs_truth
from rydberg.config import (N_SHOTS, PULSE_DURATION_NS, SIMULATION_FILE, TEST_GRAPHS, THETAS_TRAIN, TRAIN_GRAPHS, V_REF, WAIT_COLUMNS,
                            WAITS)
from rydberg.physics import COUPLING_MATRICES
from rydberg.stats import binomial_noise, cluster_bootstrap, percentile_interval, slope_with_interval

N_GNN_SEEDS = 5  # number of GNN trainings used to measure the seed-to-seed spread

# %%
# =====================================================================
# 0. Data: one row per (graph, theta, atom)
# =====================================================================
simulation_table = pl.read_parquet(SIMULATION_FILE)
train_rows = simulation_table.filter(pl.col("split") == "train")
test_rows = simulation_table.filter(pl.col("split") == "test")
print(simulation_table.group_by("split").agg(pl.col("graph").n_unique().alias("graphs"), pl.len().alias("rows")))

# %%
# =====================================================================
# 1. The experiment: Ramsey sequence and atom graphs
# =====================================================================
save_figure(plot_ramsey_sequence(), "1_ramsey_sequence.png")
save_figure(plot_graph_family(TRAIN_GRAPHS, TEST_GRAPHS), "1_graphs.png")
plt.show()


# %%
# =====================================================================
# 2. Curve estimators (A and B)
#
# A calibration curve gives, for each wait time, the mean population of the
# selected atoms as a function of θ, smoothed by a spline. To estimate θ for
# one atom, we look for the θ whose curve values (one per wait time) are
# closest, in least squares, to the atom's 3 measurements.
# =====================================================================
class CalibrationCurve:
  """Mean population versus θ, one smoothing spline per wait time."""

  def __init__(self, selected_rows):
    grouped = (selected_rows.group_by("theta")
               .agg(*[pl.col(column).mean() for column in WAIT_COLUMNS], pl.len().alias("n_atoms"))
               .sort("theta"))
    self.thetas = grouped["theta"].to_numpy()
    self.mean_populations = grouped.select(WAIT_COLUMNS).to_numpy()  # [T, len(WAITS)]
    weights = np.sqrt(grouped["n_atoms"].to_numpy())  # a mean over n atoms is less noisy
    self.splines = [make_smoothing_spline(self.thetas, self.mean_populations[:, wait_index], w=weights)
                    for wait_index in range(len(WAITS))]

  def populations_at(self, thetas):
    """Curve evaluated at the given θ values -> [len(thetas), len(WAITS)]."""
    return np.stack([spline(thetas) for spline in self.splines], axis=1)

  def invert(self, populations, resolution=500):
    """Measurements [n, len(WAITS)] -> estimated θ [n] (least squares over the wait times)."""
    theta_candidates = np.linspace(self.thetas[0], self.thetas[-1], resolution)
    candidate_populations = self.populations_at(theta_candidates)
    squared_gaps = ((populations[:, None, :] - candidate_populations[None, :, :]) ** 2).sum(axis=2)
    return theta_candidates[squared_gaps.argmin(axis=1)]


# A: one curve, fitted on the isolated atom (degree 0)
isolated_calibration = CalibrationCurve(train_rows.filter(pl.col("degree") == 0))
# B: one curve per degree, fitted on all training atoms of that degree
calibration_by_degree = {degree: CalibrationCurve(train_rows.filter(pl.col("degree") == degree))
                         for degree in sorted(train_rows["degree"].unique())}


def estimate_isolated(rows):
  return isolated_calibration.invert(rows.select(WAIT_COLUMNS).to_numpy())


def estimate_by_degree(rows):
  """Uses the curve of the atom's degree, or of the closest known degree if never seen (5, 6)."""
  known_degrees = np.array(list(calibration_by_degree))
  atom_degree = rows["degree"].to_numpy()
  populations = rows.select(WAIT_COLUMNS).to_numpy()
  estimates = np.empty(len(rows))
  for degree in np.unique(atom_degree):
    nearest_known_degree = int(known_degrees[np.argmin(abs(known_degrees - degree))])
    selected = atom_degree == degree
    estimates[selected] = calibration_by_degree[nearest_known_degree].invert(populations[selected])
  return estimates


# Example: a degree-3 atom of the diamond, at the training θ closest to 0.65
example_row = (train_rows.filter((pl.col("graph") == "diamond N=4") & (pl.col("degree") == 3))
               .with_columns((pl.col("theta") - 0.65).abs().alias("gap_to_target"))
               .sort("gap_to_target").head(1))

save_figure(plot_single_atom(simulation_table.filter(pl.col("degree") == 0).sort("theta")), "2_single_atom.png")
save_figure(plot_calibration_by_degree(calibration_by_degree), "2_calibration_by_degree.png")
save_figure(plot_inversion_example(example_row, calibration_by_degree[3], estimate_by_degree(example_row)[0]),
            "2_inversion_example.png")
plt.show()

# --- Statistics on the training data: how much do neighbours and θ move P(|1>)?
neighbour_effect = []  # mean population per degree, and linear change per extra neighbour
for wait_index, wait_column in enumerate(WAIT_COLUMNS):
  slope, half_width = slope_with_interval(train_rows["degree"].to_numpy(), train_rows[wait_column].to_numpy())
  neighbour_effect.append({"wait (ns)": WAITS[wait_index],
                           **{f"degree {degree}": float(train_rows.filter(pl.col("degree") == degree)[wait_column].mean())
                              for degree in sorted(train_rows["degree"].unique())},
                           "change per neighbour": slope, "± 95% CI": half_width})
# --- Change of the isolated-atom curve over the whole θ range (to compare with the effect of one neighbour)
full_range_change = (isolated_calibration.populations_at(np.array([THETAS_TRAIN[0]]))
                     - isolated_calibration.populations_at(np.array([THETAS_TRAIN[-1]])))[0]
print("\nIsolated atom: drop of P(|1>) from θ = 0.05 to θ = 1.50, per wait:", np.round(full_range_change, 3))

# --- Single atom: agreement with the analytic formula, and θ resolution along the θ range
isolated_rows = train_rows.filter(pl.col("degree") == 0)
formula_residuals = np.concatenate([
    isolated_rows[wait_column].to_numpy()
    - 0.5 * (1 + np.exp(-isolated_rows["theta"].to_numpy() * (wait_ns + PULSE_DURATION_NS) / 1000))
    for wait_ns, wait_column in zip(WAITS, WAIT_COLUMNS)])
print(f"\nSingle atom vs formula: RMS residual {np.sqrt(np.mean(formula_residuals ** 2)):.4f} "
      f"(shot noise alone: about {binomial_noise(0.7, N_SHOTS):.4f}); largest |residual| {np.abs(formula_residuals).max():.4f}")
single_atom_resolution = []  # from the analytic formula, which the simulation matches to shot-noise level
for theta_value in [0.2, 0.6, 1.0, 1.4]:
  row = {"θ": theta_value}
  for wait_ns in WAITS:
    effective_time = (wait_ns + PULSE_DURATION_NS) / 1000
    probability = 0.5 * (1 + np.exp(-theta_value * effective_time))
    slope = 0.5 * effective_time * np.exp(-theta_value * effective_time)
    row[f"{wait_ns} ns"] = binomial_noise(probability, N_SHOTS) / slope
  single_atom_resolution.append(row)
print("\nSingle atom: θ resolution (shot noise / |dp/dθ|, analytic formula) at several θ:")
print(pl.DataFrame(single_atom_resolution))

with pl.Config(float_precision=3, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, tbl_rows=40):
  print("\nMean P(|1>) per degree (training rows) and linear change per extra neighbour:")
  print(pl.DataFrame(neighbour_effect))


# %%
# =====================================================================
# 3. Message-passing estimator (C)
#
# Features of atom i (6 numbers): its 3 measurements and the W-weighted sum of
# its neighbours' 3 measurements, with W_ij = V_ij / V_REF.
# The network aggregates the neighbours' features once more (message passing),
# builds a latent vector and decodes log θ_i. One training example = one
# (graph, θ) = all table rows sharing those two values, ordered by atom.
# =====================================================================
def neighbor_weights(couplings):
  """How much atom i listens to atom j: W_ij = V_ij / V_REF (∝ 1/d^6, all pairs, not normalised)."""
  return torch.tensor(couplings / V_REF, dtype=torch.float32)


def node_features(populations, couplings):
  """populations [N, len(WAITS)] -> (features [N, 2*len(WAITS)], neighbour weights [N, N])."""
  weights = neighbor_weights(couplings)
  populations = torch.tensor(populations, dtype=torch.float32)
  return torch.cat([populations, weights @ populations], dim=1), weights


class RydbergGNN(nn.Module):
  def __init__(self, input_dim=2 * len(WAITS), latent_dim=16):
    super().__init__()
    self.self_projection = nn.Linear(input_dim, latent_dim)      # "what I measure"
    self.neighbor_projection = nn.Linear(input_dim, latent_dim)  # "what my neighbours tell me"
    self.update = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim), nn.SiLU())
    self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, 1))  # -> log θ

  def forward(self, features, weights):
    messages = weights @ features  # neighbour aggregation
    latent = self.update(self.self_projection(features) + self.neighbor_projection(messages))
    return self.decoder(latent)


def graph_theta_groups(rows):
  """One chunk of rows per (graph, θ), atoms in order."""
  return {key: group.sort("atom") for key, group in rows.partition_by("graph", "theta", as_dict=True).items()}


def estimate_gnn(rows, model=None):
  """Estimated θ for each row of `rows` (same order). Uses the main GNN unless a model is given."""
  model = gnn if model is None else model
  rows = rows.with_row_index("row")
  estimates = np.empty(len(rows))
  for (graph_name, _), group in graph_theta_groups(rows).items():
    features, weights = node_features(group.select(WAIT_COLUMNS).to_numpy(), COUPLING_MATRICES[graph_name])
    with torch.no_grad():
      estimates[group["row"].to_numpy()] = torch.exp(model(features, weights)).squeeze(1).numpy()
  return estimates


gnn_training_rows = train_rows.filter(pl.col("graph").is_in(TRAIN_GRAPHS))  # without the isolated atom
training_samples = []
for (graph_name, theta), group in graph_theta_groups(gnn_training_rows).items():
  features, weights = node_features(group.select(WAIT_COLUMNS).to_numpy(), COUPLING_MATRICES[graph_name])
  training_samples.append((features, weights, torch.full((len(features), 1), float(np.log(theta)))))


def train_gnn(seed, epochs=400):
  """Trains a GNN from scratch. Returns (model, loss history)."""
  torch.manual_seed(seed)
  np.random.seed(seed)
  model = RydbergGNN()
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
  loss_history = []
  for epoch in range(epochs):
    epoch_loss = 0.0
    for sample_index in np.random.permutation(len(training_samples)):
      features, weights, target = training_samples[sample_index]
      optimizer.zero_grad()
      loss = nn.functional.mse_loss(model(features, weights), target)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    loss_history.append(epoch_loss / len(training_samples))
  return model.eval(), loss_history


gnn, loss_history = train_gnn(seed=0)  # the GNN shown in the figures
print(f"final training loss (MSE on log θ): {loss_history[-1]:.4f}")

save_figure(plot_gnn_architecture(), "3_gnn_architecture.png")
save_figure(plot_training(loss_history, np.var(np.log(THETAS_TRAIN)),
                          gnn_training_rows.with_columns(pl.Series("estimate", estimate_gnn(gnn_training_rows)))),
            "3_gnn_training.png")
plt.show()


# %%
# =====================================================================
# 4. Results on the test graphs (never seen) at test θ values (never seen)
# =====================================================================
METHODS = {"A. isolated": estimate_isolated, "B. per degree": estimate_by_degree, "C. GNN": estimate_gnn}

test_results = test_rows.with_columns(
    [pl.Series(method_name, estimate(test_rows)) for method_name, estimate in METHODS.items()]
).with_columns(
    [((pl.col(method_name) - pl.col("theta")).abs() / pl.col("theta")).alias(f"relative_error {method_name}")
     for method_name in METHODS]
)


def mean_error_percent(grouped):
  return grouped.agg([(100 * pl.col(f"relative_error {method_name}").mean()).alias(method_name)
                      for method_name in METHODS])


error_by_graph = mean_error_percent(test_results.group_by("graph", maintain_order=True))
error_by_degree = mean_error_percent(test_results.group_by("degree")).sort("degree")
error_overall = test_results.select([(100 * pl.col(f"relative_error {method_name}").mean()).alias(method_name)
                                     for method_name in METHODS])

save_figure(plot_estimates_vs_truth(test_results, list(METHODS)), "4_estimates_vs_truth.png")
save_figure(plot_errors(error_by_graph, error_by_degree, list(METHODS)), "4_errors.png")
plt.show()

with pl.Config(float_precision=1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
  print("\nMean relative error per atom (%), per test graph:")
  print(error_by_graph)
  print("\nPer atom degree:")
  print(error_by_degree)

# --- Uncertainty 1: which test simulations we happened to draw (bootstrap over (graph, θ) simulations)
method_names = list(METHODS)
error_columns = [f"relative_error {method_name}" for method_name in method_names]
clusters = (test_results.group_by("graph", "theta")
            .agg(*[pl.col(column).sum() for column in error_columns], pl.len().alias("n_atoms"))
            .sort("graph", "theta"))  # fixed order: the bootstrap is then reproducible
resampled = 100 * cluster_bootstrap(clusters.select(error_columns).to_numpy(), clusters["n_atoms"].to_numpy())
observed = error_overall.row(0)  # mean relative error actually observed, per method
print("\nOverall mean relative error (%) and its 95% bootstrap interval over the 48 test simulations:")
for column_index, method_name in enumerate(method_names):
  low, high = percentile_interval(resampled[:, column_index])
  print(f"  {method_name:14s} {observed[column_index]:6.1f}   [{low:.1f}, {high:.1f}]")
for first, second in [(1, 2), (0, 1)]:
  difference = resampled[:, first] - resampled[:, second]
  low, high = percentile_interval(difference)
  print(f"  {method_names[first]} minus {method_names[second]}: {observed[first] - observed[second]:.1f}  [{low:.1f}, {high:.1f}]"
        f"   (share of resamples where {method_names[second]} is better: {(difference > 0).mean():.3f})")

# --- Uncertainty 2: which random initialisation the GNN got (retrain with other seeds)
seed_errors = [float(100 * np.mean(np.abs(estimate_gnn(test_rows, train_gnn(seed)[0]) - test_rows["theta"].to_numpy())
                                   / test_rows["theta"].to_numpy())) for seed in range(N_GNN_SEEDS)]
print(f"\nGNN overall error over {N_GNN_SEEDS} training seeds (%): "
      f"{', '.join(f'{error:.1f}' for error in seed_errors)}  -> mean {np.mean(seed_errors):.1f}, sd {np.std(seed_errors, ddof=1):.1f}")
