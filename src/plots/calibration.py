"""Section 2 figures: single-atom physics, calibration curves per degree, an inversion example."""

import matplotlib.pyplot as plt
import numpy as np

from plots import WAIT_COLORS
from rydberg.config import PULSE_DURATION_NS, THETAS_TRAIN, WAIT_COLUMNS, WAITS


def plot_single_atom(isolated_rows):
  """Isolated atom: simulation vs analytic formula, and sensitivity |dp/dθ| for each wait time."""
  theta_fine = np.linspace(0.0, 1.6, 300)
  figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  for wait_ns, wait_column, color in zip(WAITS, WAIT_COLUMNS, WAIT_COLORS):
    effective_time = (wait_ns + PULSE_DURATION_NS) / 1000  # µs (theta is in 1/µs)
    axes[0].plot(theta_fine, 0.5 * (1 + np.exp(-theta_fine * effective_time)), color=color, alpha=0.5)
    axes[0].plot(isolated_rows["theta"], isolated_rows[wait_column], "o", color=color, ms=4,
                 label=f"wait {wait_ns} ns")
    axes[1].plot(theta_fine, 0.5 * effective_time * np.exp(-theta_fine * effective_time), color=color,
                 label=f"wait {wait_ns} ns")
  axes[0].plot([], [], "k-", alpha=0.5, label="formula ½(1+e^(-θ t_eff))")
  axes[0].set(xlabel="θ (1/µs)", ylabel="P(|1>)", title="Single atom: simulation (dots) vs formula (lines)")
  axes[0].legend()
  axes[1].set(xlabel="θ (1/µs)", ylabel="|dp/dθ|", title="Sensitivity: each wait time covers a θ range")
  axes[1].legend()
  figure.tight_layout()
  return figure


def plot_calibration_by_degree(calibration_by_degree):
  """One panel per wait time: mean population per degree (dots) and fitted spline (lines)."""
  figure, axes = plt.subplots(1, len(WAITS), figsize=(16, 4.5), sharey=True)
  for wait_index, (wait_ns, axis) in enumerate(zip(WAITS, axes)):
    for degree, calibration in calibration_by_degree.items():
      color = "k" if degree == 0 else plt.cm.viridis(degree / 6)
      label = "single atom (degree 0)" if degree == 0 else f"degree {degree}"
      theta_fine = np.linspace(calibration.thetas[0], calibration.thetas[-1], 300)
      axis.plot(calibration.thetas, calibration.mean_populations[:, wait_index], "o", ms=3, color=color, alpha=0.6)
      axis.plot(theta_fine, calibration.populations_at(theta_fine)[:, wait_index],
                "--" if degree == 0 else "-", lw=2, color=color, label=label)
    axis.set(title=f"wait {wait_ns} ns", xlabel="θ (1/µs)")
  axes[0].set_ylabel("P(|1>)")
  axes[-1].legend(fontsize=8)
  figure.suptitle("Calibration curves: mean population per degree (dots) and smoothing spline (lines)")
  figure.tight_layout()
  return figure


def plot_inversion_example(example_row, calibration, estimated_theta):
  """How θ is selected from one atom's 3 measurements.
  Left: the 3 calibration curves (solid), the atom's 3 measurements (dotted), and at the selected θ the
  gap between each curve and its measurement (double arrow, same colour).
  Right: the squared gaps, stacked and in the same colours, as a function of the candidate θ.
  Their total has a minimum: that θ is the estimate."""
  theta_range = np.linspace(THETAS_TRAIN[0], THETAS_TRAIN[-1], 400)
  true_theta = example_row["theta"][0]
  measurements = example_row.select(WAIT_COLUMNS).to_numpy()[0]  # [len(WAITS)]
  curves = calibration.populations_at(theta_range)                # [400, len(WAITS)]
  curves_at_estimate = calibration.populations_at(np.array([estimated_theta]))[0]
  squared_gaps = (curves - measurements) ** 2

  figure, (curve_axis, gap_axis) = plt.subplots(1, 2, figsize=(14, 5))

  for wait_index, (wait_ns, color) in enumerate(zip(WAITS, WAIT_COLORS)):
    curve_axis.plot(theta_range, curves[:, wait_index], "-", color=color, lw=2, label=f"curve, wait {wait_ns} ns")
    curve_axis.axhline(measurements[wait_index], color=color, ls=":", lw=1.5)
    curve_axis.annotate("", xy=(estimated_theta, curves_at_estimate[wait_index]),
                        xytext=(estimated_theta, measurements[wait_index]),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=2.5, shrinkA=0, shrinkB=0), zorder=6)
    curve_axis.plot(estimated_theta, curves_at_estimate[wait_index], "o", color=color, ms=6, zorder=7)
  curve_axis.plot([], [], "k:", label="this atom's measurements")
  curve_axis.plot([], [], color="gray", marker=r"$\leftrightarrow$", ls="none", ms=14, label="gap at the selected θ")
  curve_axis.axvline(estimated_theta, color="tab:green", lw=1.5, alpha=0.8, label=f"selected θ = {estimated_theta:.2f}")
  curve_axis.axvline(true_theta, color="k", lw=1.2, ls="--", label=f"true θ = {true_theta:.2f}")
  curve_axis.set(xlabel="θ (1/µs)", ylabel="P(|1>)", title="Curves, measurements and gaps")
  curve_axis.legend(fontsize=8)

  gap_axis.stackplot(theta_range, squared_gaps.T, colors=WAIT_COLORS, alpha=0.85,
                     labels=[f"(gap at wait {wait_ns} ns)²" for wait_ns in WAITS])
  total = squared_gaps.sum(axis=1)
  gap_axis.plot(theta_range, total, "k-", lw=1.5, label="total")
  gap_axis.plot(estimated_theta, total[np.argmin(abs(theta_range - estimated_theta))], "o", color="tab:green",
                ms=9, markeredgecolor="k", zorder=7, label=f"minimum: θ = {estimated_theta:.2f}")
  gap_axis.axvline(true_theta, color="k", lw=1.2, ls="--", label="true θ")
  gap_axis.set(xlabel="candidate θ (1/µs)", ylabel="sum of squared gaps", title="Accumulated squared gaps")
  gap_axis.legend(fontsize=8)
  figure.tight_layout()
  return figure
