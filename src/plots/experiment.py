"""Section 1 figures: the Ramsey sequence and the atom graphs."""

import matplotlib.pyplot as plt
import numpy as np

from plots import WAIT_COLORS
from rydberg.config import V_REF, WAITS
from rydberg.physics import ATOM_REGISTERS, COUPLING_MATRICES, atom_degrees, atom_positions, ramsey_waveforms


def plot_ramsey_sequence():
  """Rabi amplitude Ω(t) of the protocol, one row per wait time."""
  figure, axes = plt.subplots(len(WAITS), 1, figsize=(10, 5), sharex=True)
  for wait_ns, color, axis in zip(WAITS, WAIT_COLORS, axes):
    first_pulse, free_evolution, second_pulse = ramsey_waveforms(wait_ns)
    amplitude = np.concatenate([first_pulse.samples, free_evolution.samples, second_pulse.samples])
    time_us = np.arange(len(amplitude)) / 1000
    axis.fill_between(time_us, amplitude, color=color, alpha=0.6)
    pulse_end = first_pulse.duration / 1000
    axis.axvspan(pulse_end, pulse_end + wait_ns / 1000, color="gray", alpha=0.12)
    axis.text(pulse_end + wait_ns / 2000, amplitude.max() * 0.5, f"free evolution {wait_ns} ns\n(dephasing θ acts)",
              ha="center", va="center", fontsize=8)
    axis.set_ylabel("Ω (rad/µs)")
  axes[0].set_title("Ramsey protocol: π/2 pulse — free evolution — π/2 pulse, then measurement")
  axes[-1].set_xlabel("time (µs)")
  figure.tight_layout()
  return figure


def draw_atom_graph(axis, graph_name, node_values=None, value_label=None):
  """Atoms at their real positions. Edge width ∝ V_ij / V_REF (all pairs above 1 %).
  Number in each atom = its degree. Colour = degree, or node_values if given."""
  positions = atom_positions(ATOM_REGISTERS[graph_name])
  couplings = COUPLING_MATRICES[graph_name]
  degrees = atom_degrees(couplings)
  relative_coupling = couplings / V_REF
  for first_atom in range(len(positions)):
    for second_atom in range(first_atom + 1, len(positions)):
      strength = relative_coupling[first_atom, second_atom]
      if strength > 0.01:
        axis.plot(*zip(positions[first_atom], positions[second_atom]), color="tab:purple",
                  lw=0.4 + 3.5 * min(strength, 1), alpha=0.25 + 0.5 * min(strength, 1), zorder=1)
  # square window around the atoms (a chain has zero height: without this it collapses to a line)
  middle = (positions.max(axis=0) + positions.min(axis=0)) / 2
  half_width = (positions.max(axis=0) - positions.min(axis=0)).max() / 2 + 6
  node_size = 420 * min(1.0, (18 / half_width) ** 2)  # smaller atoms when the window is wider
  colors = degrees if node_values is None else node_values
  scatter = axis.scatter(*positions.T, s=node_size, c=colors, cmap="viridis", edgecolor="k", zorder=2,
                         vmin=0 if node_values is not None else 0, vmax=6 if node_values is None else None)
  for (x, y), degree in zip(positions, degrees):
    axis.text(x, y, str(degree), ha="center", va="center", weight="bold", color="w", fontsize=9 if node_size > 300 else 7, zorder=3)
  axis.set_title(graph_name, fontsize=10)
  axis.set_xlim(middle[0] - half_width, middle[0] + half_width)
  axis.set_ylim(middle[1] - half_width, middle[1] + half_width)
  axis.set_aspect("equal")
  axis.set_xticks([])
  axis.set_yticks([])
  return scatter


def plot_graph_family(train_graphs, test_graphs):
  """Training graphs (top) and test graphs (bottom)."""
  n_columns = 6
  n_train_rows = int(np.ceil(len(train_graphs) / n_columns))
  n_test_rows = int(np.ceil(len(test_graphs) / n_columns))
  figure, axes = plt.subplots(n_train_rows + n_test_rows, n_columns, figsize=(16, 2.7 * (n_train_rows + n_test_rows)))
  for axis in axes.ravel():
    axis.axis("off")
  for index, graph_name in enumerate(train_graphs):
    axis = axes[index // n_columns, index % n_columns]
    axis.axis("on")
    draw_atom_graph(axis, graph_name)
  for index, graph_name in enumerate(test_graphs):
    axis = axes[n_train_rows + index // n_columns, index % n_columns]
    axis.axis("on")
    draw_atom_graph(axis, graph_name)
    for spine in axis.spines.values():
      spine.set_edgecolor("tab:red")
      spine.set_linewidth(2)
  figure.suptitle("Training graphs (top) and test graphs (bottom)", fontsize=12)
  figure.tight_layout()
  return figure
