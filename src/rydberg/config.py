"""All experiment parameters in one place: physics constants, wait times, thetas, graphs, paths."""

from pathlib import Path

import numpy as np
from pulser.devices import MockDevice

# --- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SIMULATION_FILE = DATA_DIR / "simulations.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- physics
SPACING = 9.0                                      # triangular lattice spacing (µm)
V_REF = MockDevice.interaction_coeff / SPACING**6  # coupling between nearest neighbours (rad/µs)
WAITS = [250, 800, 2500]                           # Ramsey wait times (ns)
PULSE_DURATION_NS = 400                            # duration of each π/2 pulse
N_SHOTS = 5000                                     # measurements per simulation

# --- dephasing rates (1/µs)
THETAS_TRAIN = np.linspace(0.05, 1.50, 20)                                # training grid
THETAS_TEST = np.array([0.12, 0.33, 0.50, 0.65, 0.81, 1.02, 1.27, 1.43])  # off the grid

# --- one column per wait time in the simulation table
WAIT_COLUMNS = [f"population_wait_{wait_ns}" for wait_ns in WAITS]

# --- graphs: subgraphs of a triangular lattice
# Sites of the 3x3 triangular lattice (numbered by row):
#      0   1   2
#    3   4   5          (4 = centre, 6 neighbours: 0 1 3 5 6 7)
#      6   7   8
# Lattice neighbours: 0-1 1-2 3-4 4-5 6-7 7-8 | 0-3 0-4 1-4 1-5 2-5 | 3-6 4-6 4-7 5-7 5-8
GRAPH_SPECS = {  # name: (rows, atoms per row, kept sites ; None = all)
    "isolated N=1":   (1, 1, None),
    # --- train (N = 4-5, degrees 1 to 4)
    "chain N=4":      (1, 4, None),
    "Y N=4":          (3, 3, [4, 3, 1, 7]),
    "diamond N=4":    (3, 3, [0, 1, 3, 4]),
    "S-chain N=4":    (3, 3, [0, 4, 5, 8]),
    "paw A N=4":      (3, 3, [0, 1, 4, 7]),
    "paw B N=4":      (3, 3, [3, 4, 5, 7]),
    "cross N=5":      (3, 3, [4, 3, 5, 1, 7]),
    "tail N=5":       (3, 3, [0, 1, 2, 4, 7]),
    "zigzag N=5":     (3, 3, [0, 3, 4, 7, 8]),
    "chain N=5":      (1, 5, None),
    "arc N=5":        (3, 3, [6, 7, 5, 1, 0]),
    "dense N=5":      (3, 3, [0, 1, 4, 5, 7]),
    # --- test (N = 6-7, never seen; degrees 5 and 6 never seen either)
    "N=6 holes":      (3, 3, [0, 1, 3, 4, 5, 7]),
    "N=6 triangle":   (3, 3, [0, 1, 2, 4, 5, 7]),
    "N=6 ring":       (3, 3, [0, 1, 5, 7, 6, 3]),
    "N=7 two holes":  (3, 3, [1, 2, 3, 4, 5, 6, 7]),
    "N=7 hexagon":    (3, 3, [0, 1, 3, 4, 5, 6, 7]),
    "N=7 chain":      (1, 7, None),
}
ISOLATED_GRAPH = "isolated N=1"
TRAIN_GRAPHS = ["chain N=4", "Y N=4", "diamond N=4", "S-chain N=4", "paw A N=4", "paw B N=4",
                "cross N=5", "tail N=5", "zigzag N=5", "chain N=5", "arc N=5", "dense N=5"]
TEST_GRAPHS = ["N=6 holes", "N=6 triangle", "N=6 ring", "N=7 two holes", "N=7 hexagon", "N=7 chain"]
