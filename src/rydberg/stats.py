"""Small statistics helpers used by analysis.py: slopes with confidence intervals, cluster bootstrap."""

import numpy as np
from scipy import stats


def slope_with_interval(x, y, confidence=0.95):
  """Least-squares slope of y versus x. Returns (slope, half-width of the confidence interval)."""
  fit = stats.linregress(x, y)
  t_value = stats.t.ppf(0.5 + confidence / 2, len(x) - 2)
  return fit.slope, t_value * fit.stderr


def binomial_noise(probability, n_shots):
  """Standard deviation of a measured frequency: each shot is one independent 0/1 outcome."""
  return np.sqrt(probability * (1 - probability) / n_shots)


def cluster_bootstrap(cluster_error_sums, cluster_sizes, n_resamples=4000, seed=0):
  """Bootstrap of the mean error, resampling whole simulations (clusters) with replacement.
  Atoms of the same simulation are not independent, so the unit that is resampled is the
  (graph, θ) simulation, not the atom.

  cluster_error_sums : [C, M] sum over the atoms of each cluster, for each of M methods
  cluster_sizes      : [C]    number of atoms in each cluster
  Returns [n_resamples, M]: the mean error per atom of each method, for each resample.
  """
  rng = np.random.default_rng(seed)
  n_clusters = len(cluster_sizes)
  picks = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
  return cluster_error_sums[picks].sum(axis=1) / cluster_sizes[picks].sum(axis=1)[:, None]


def percentile_interval(samples, confidence=0.95):
  """(low, high) percentile interval of a 1-D array of bootstrap samples."""
  tail = 100 * (1 - confidence) / 2
  return np.percentile(samples, tail), np.percentile(samples, 100 - tail)
