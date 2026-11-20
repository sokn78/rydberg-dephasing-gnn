"""Section 3 figures: the GNN architecture (with neighbour aggregation) and its training."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from rydberg.config import V_REF
from rydberg.physics import ATOM_REGISTERS, COUPLING_MATRICES, atom_positions

# one colour per ROLE, shared by the graph (left) and the network (right)
ATOM_COLOR = "tab:orange"     # atom i (the one whose θ we estimate)
NEIGHBOR_COLOR = "tab:blue"   # its neighbours j
MESSAGE_COLOR = "tab:purple"  # aggregated message
LATENT_COLOR = "tab:green"    # latent vector
OUTPUT_COLOR = "tab:red"      # output
LAYER_COLOR = "#e6e6e6"       # layers (trapezoids)


def _vector(axis, x, y_center, size, color, cell=0.22, label=None, alpha=0.85):
  """Column vector of `size` cells centred at (x, y_center). Returns (top, bottom) y."""
  height = size * cell
  bottom = y_center - height / 2
  for index in range(size):
    axis.add_patch(Rectangle((x - cell / 2, bottom + index * cell), cell, cell,
                             facecolor=color, edgecolor="white", lw=0.8, alpha=alpha))
  if label:
    axis.text(x, bottom - 0.18, label, ha="center", va="top", fontsize=8.5)
  return bottom + height, bottom


def _layer(axis, x_left, x_right, y_center, height_left, height_right, text):
  """Trapezoid between two vectors of different sizes (a dense layer)."""
  axis.add_patch(Polygon([(x_left, y_center - height_left / 2), (x_right, y_center - height_right / 2),
                          (x_right, y_center + height_right / 2), (x_left, y_center + height_left / 2)],
                         closed=True, facecolor=LAYER_COLOR, edgecolor="gray", lw=1))
  axis.text((x_left + x_right) / 2, y_center, text, ha="center", va="center", fontsize=8)


def _arrow(axis, start, end, color="k", lw=1.3):
  axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def _draw_neighbourhood(axis, graph_name, focus_atom):
  """Real atom positions; edges towards atom i labelled with W_ij = V_ij / V_REF."""
  positions = atom_positions(ATOM_REGISTERS[graph_name])
  couplings = COUPLING_MATRICES[graph_name]
  weights = couplings[focus_atom] / V_REF
  # other edges, faint, for context
  for first in range(len(positions)):
    for second in range(first + 1, len(positions)):
      if focus_atom not in (first, second) and couplings[first, second] > 0.01 * V_REF:
        axis.plot(*zip(positions[first], positions[second]), color="lightgray", lw=1, zorder=1)
  # edges j -> i, width ∝ W_ij, labelled
  neighbour_names = {}
  for atom in range(len(positions)):
    if atom == focus_atom:
      continue
    neighbour_names[atom] = f"j{len(neighbour_names) + 1}"
    start, end = positions[atom], positions[focus_atom]
    direction = (end - start) / np.linalg.norm(end - start)
    axis.add_patch(FancyArrowPatch(start + 2.2 * direction, end - 2.6 * direction, arrowstyle="-|>",
                                   mutation_scale=14, color=MESSAGE_COLOR, lw=0.6 + 4 * weights[atom],
                                   alpha=0.85, zorder=2))
    middle = (start + end) / 2
    axis.text(*middle, f"{weights[atom]:.2f}", ha="center", va="center", fontsize=9, zorder=4,
              bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=MESSAGE_COLOR, lw=0.8))
  for atom, (x, y) in enumerate(positions):
    is_focus = atom == focus_atom
    axis.add_patch(Circle((x, y), 2.0, facecolor=ATOM_COLOR if is_focus else NEIGHBOR_COLOR,
                          edgecolor="k", lw=1.2, zorder=3))
    axis.text(x, y, "i" if is_focus else neighbour_names[atom], ha="center", va="center",
              color="w", weight="bold", fontsize=11, zorder=4)
  middle = (positions.max(axis=0) + positions.min(axis=0)) / 2
  half_width = (positions.max(axis=0) - positions.min(axis=0)).max() / 2 + 5
  axis.set_xlim(middle[0] - half_width, middle[0] + half_width)
  axis.set_ylim(middle[1] - half_width - 4, middle[1] + half_width)
  axis.set_aspect("equal")
  axis.axis("off")
  axis.set_title(f"Graph ({graph_name}): atom i and its other atoms j\nnumber on each edge = weight W_ij = V_ij / V_ref ∝ 1/d⁶ (1 for a nearest neighbour)",
                 fontsize=10)
  far_atoms = [neighbour_names[atom] for atom in neighbour_names if weights[atom] < 0.1]
  if far_atoms:
    axis.text(middle[0], middle[1] - half_width - 2.5,
              f"{', '.join(far_atoms)}: not a nearest neighbour, but still a small weight (no cutoff)",
              ha="center", va="center", fontsize=8.5, color="dimgray")


def plot_gnn_architecture(graph_name="tail N=5", focus_atom=1, latent_dim=16):
  """Left: a real graph with its edge weights. Right: the network, drawn with vectors."""
  figure, (graph_axis, axis) = plt.subplots(1, 2, figsize=(18, 6.6), gridspec_kw={"width_ratios": [1, 2.7]})
  _draw_neighbourhood(graph_axis, graph_name, focus_atom)

  axis.set_xlim(0, 17)
  axis.set_ylim(-1.6, 9.0)
  axis.axis("off")
  axis.set_title("Neural network: from the measurements to θ_i, the estimate for atom i", fontsize=10)
  cell_input, cell_latent = 0.32, 0.2
  y_self, y_neighbours, y_merge = 6.4, 2.2, 4.3

  # --- inputs: one vector per atom
  top, bottom = _vector(axis, 1.6, y_self, 6, ATOM_COLOR, cell=cell_input)
  for group_bottom, group_size, text in [(3, 3, "3 measurements\np(250), p(800), p(2500)"),
                                         (0, 3, "3 neighbour sums\n(W p) at each wait")]:
    group_top = bottom + (group_bottom + group_size) * cell_input
    group_low = bottom + group_bottom * cell_input
    axis.plot([1.3, 1.25, 1.25, 1.3], [group_low + 0.03, group_low + 0.03, group_top - 0.03, group_top - 0.03],
              color="k", lw=0.8)
    axis.text(1.15, (group_low + group_top) / 2, text, ha="right", va="center", fontsize=7.5)
  axis.text(1.6, top + 0.2, "input x_i  (6)", ha="center", va="bottom", fontsize=9, weight="bold")
  for offset in (0.3, 0.15, 0.0):  # one input vector per neighbour, stacked
    _vector(axis, 1.6 + offset, y_neighbours + offset, 6, NEIGHBOR_COLOR, cell=cell_input)
  axis.text(1.75, y_neighbours - 1.55, "inputs x_j  (6)\none per neighbour", ha="center", va="top", fontsize=9,
            weight="bold")

  # --- first layer: encode each atom (linear 6 -> 16)
  _layer(axis, 2.15, 3.55, y_self, 6 * cell_input, latent_dim * cell_latent, f"Linear\n6→{latent_dim}\n(self)")
  _layer(axis, 2.3, 3.55, y_neighbours, 6 * cell_input, latent_dim * cell_latent, f"Linear\n6→{latent_dim}\n(neighb.)")
  _vector(axis, 4.0, y_self, latent_dim, ATOM_COLOR, cell=cell_latent, label=f"encoded i ({latent_dim})")
  for offset in (0.3, 0.15, 0.0):
    _vector(axis, 4.0 + offset, y_neighbours + offset, latent_dim, NEIGHBOR_COLOR, cell=cell_latent)
  axis.text(4.15, y_neighbours - 1.9, f"encoded j ({latent_dim} each)", ha="center", va="top", fontsize=8.5)

  # --- aggregation: weighted sum over neighbours
  axis.add_patch(Circle((5.6, y_neighbours), 0.42, facecolor="white", edgecolor=MESSAGE_COLOR, lw=1.8))
  axis.text(5.6, y_neighbours, "Σ W_ij", ha="center", va="center", fontsize=8.5, color=MESSAGE_COLOR, weight="bold")
  _arrow(axis, (4.6, y_neighbours), (5.15, y_neighbours), color=MESSAGE_COLOR)
  axis.text(5.6, y_neighbours + 0.65, "aggregation\n(weights from the graph)", ha="center", va="bottom", fontsize=8,
            color=MESSAGE_COLOR)
  _vector(axis, 6.7, y_neighbours, latent_dim, MESSAGE_COLOR, cell=cell_latent, label=f"message ({latent_dim})")
  _arrow(axis, (6.05, y_neighbours), (6.55, y_neighbours), color=MESSAGE_COLOR)

  # --- combine own encoding + message
  axis.add_patch(Circle((8.0, y_merge), 0.3, facecolor="white", edgecolor="k", lw=1.2))
  axis.text(8.0, y_merge, "+", ha="center", va="center", fontsize=15)
  _arrow(axis, (4.2, y_self), (7.8, y_merge + 0.2), color=ATOM_COLOR)
  _arrow(axis, (6.85, y_neighbours + 0.4), (7.82, y_merge - 0.22), color=MESSAGE_COLOR)

  # --- update -> latent vector
  _layer(axis, 8.5, 10.1, y_merge, latent_dim * cell_latent, latent_dim * cell_latent,
         f"SiLU\nLinear\n{latent_dim}→{latent_dim}\nSiLU")
  _arrow(axis, (8.3, y_merge), (8.5, y_merge))
  _vector(axis, 10.6, y_merge, latent_dim, LATENT_COLOR, cell=cell_latent, label=f"latent h_i ({latent_dim})")

  # --- decoder -> output
  _layer(axis, 11.1, 12.7, y_merge, latent_dim * cell_latent, 16 * cell_latent, f"Linear\n{latent_dim}→16\nSiLU")
  _vector(axis, 13.2, y_merge, 16, "lightgray", cell=cell_latent, alpha=1)
  _layer(axis, 13.5, 14.6, y_merge, 16 * cell_latent, 0.35, "Linear\n16→1")
  _vector(axis, 15.0, y_merge, 1, OUTPUT_COLOR, cell=0.35)
  axis.text(15.0, y_merge - 0.4, "log θ_i", ha="center", va="top", fontsize=9, weight="bold")
  _arrow(axis, (15.25, y_merge), (15.75, y_merge))
  axis.text(16.3, y_merge, "θ_i\n= exp(·)", ha="center", va="center", fontsize=9, weight="bold")
  axis.text(11.9, y_merge + 2.2, "decoder", ha="center", fontsize=9, style="italic")
  axis.text(9.3, y_merge + 2.2, "update", ha="center", fontsize=9, style="italic")

  # --- colour legend and notes
  legend_items = [(ATOM_COLOR, "atom i"), (NEIGHBOR_COLOR, "neighbours j"), (MESSAGE_COLOR, "aggregated message"),
                  (LATENT_COLOR, "latent vector"), (OUTPUT_COLOR, "output"), (LAYER_COLOR, "layer")]
  x = 7.6
  for color, text in legend_items:
    axis.add_patch(Rectangle((x, -1.2), 0.3, 0.3, facecolor=color, edgecolor="gray", lw=0.5))
    axis.text(x + 0.4, -1.05, text, va="center", fontsize=8)
    x += 0.75 + 0.1 * len(text)  # next item after this label
  axis.text(7.6, -0.45, "Graph estimate = mean of θ_i over the atoms.  x_j already holds neighbour sums, "
            "so atom i also sees 2-hop neighbours.", fontsize=8, color="dimgray")
  figure.tight_layout()
  return figure


def plot_training(loss_history, baseline_loss, training_estimates):
  """Learning curve and predictions on the training graphs."""
  figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  axes[0].plot(loss_history)
  axes[0].axhline(baseline_loss, color="r", ls="--", label="predict the mean")
  axes[0].set(xlabel="epoch", ylabel="MSE on log θ", yscale="log", title="Learning curve")
  axes[0].legend()
  for (graph_name,), graph_rows in training_estimates.group_by("graph", maintain_order=True):
    per_theta = graph_rows.group_by("theta").agg(pl.col("estimate").mean()).sort("theta")
    axes[1].plot(per_theta["theta"], per_theta["estimate"], "o", ms=4, label=graph_name)
  axes[1].plot([0, 1.5], [0, 1.5], "k--")
  axes[1].set(xlabel="true θ", ylabel="predicted θ (graph mean)", title="GNN on the training graphs")
  axes[1].legend(fontsize=7, ncol=2)
  figure.tight_layout()
  return figure
