"""Section 4 figures: comparison of the three estimators on the test graphs."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plots import METHOD_COLORS


def plot_estimates_vs_truth(test_results, methods):
  """One panel per method: estimated θ (graph mean) vs true θ, one line per test graph.
  The estimate of each method is in the column named after the method."""
  figure, axes = plt.subplots(1, len(methods), figsize=(16, 4.8), sharey=True)
  for axis, method_name in zip(axes, methods):
    for (graph_name,), graph_rows in test_results.group_by("graph", maintain_order=True):
      per_theta = graph_rows.group_by("theta").agg(pl.col(method_name).mean()).sort("theta")
      axis.plot(per_theta["theta"], per_theta[method_name], "o-", ms=4, label=graph_name)
    axis.plot([0, 1.5], [0, 1.5], "k--", lw=1)
    axis.set(title=method_name, xlabel="true θ (1/µs)", xlim=(0, 1.55), ylim=(0, 1.55))
  axes[0].set_ylabel("estimated θ (mean over the graph's atoms)")
  axes[-1].legend(fontsize=8)
  figure.suptitle("Test graphs (never seen) at test θ values (never seen): perfect estimates lie on the dashed line")
  figure.tight_layout()
  return figure


def plot_errors(error_by_graph, error_by_degree, methods):
  """Mean relative error per test graph and per atom degree."""
  figure, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
  bar_positions = np.arange(len(error_by_graph))
  bar_width = 0.8 / len(methods)
  for method_index, method_name in enumerate(methods):
    axes[0].bar(bar_positions + (method_index - (len(methods) - 1) / 2) * bar_width, error_by_graph[method_name],
                width=bar_width, color=METHOD_COLORS[method_name], label=method_name)
  axes[0].set_xticks(bar_positions, error_by_graph["graph"], rotation=15)
  axes[0].set(ylabel="mean relative error per atom (%)", yscale="log", title="Error per test graph (log scale)")
  axes[0].legend()

  for method_name in methods:
    axes[1].plot(error_by_degree["degree"], error_by_degree[method_name], "o-",
                 color=METHOD_COLORS[method_name], label=method_name)
  axes[1].axvspan(4.5, 6.5, color="gray", alpha=0.15, label="degrees never seen in training")
  axes[1].set(xlabel="atom degree", ylabel="mean relative error (%)", yscale="log", title="Error by atom degree (log scale)")
  axes[1].legend(fontsize=8)
  figure.tight_layout()
  return figure
