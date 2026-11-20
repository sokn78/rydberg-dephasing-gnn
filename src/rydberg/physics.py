"""Physics: atom graphs (positions, couplings, degrees) and the Ramsey simulation."""

import time

import numpy as np
import polars as pl
from pulser import Pulse, Register, Sequence
from pulser.backend import BitStrings
from pulser.devices import MockDevice
from pulser.noise_model import NoiseModel
from pulser.waveforms import BlackmanWaveform, ConstantWaveform
from pulser_simulation import QutipBackendV2, QutipConfig

from rydberg.config import (GRAPH_SPECS, N_SHOTS, PULSE_DURATION_NS, SPACING, TEST_GRAPHS,
                            THETAS_TEST, THETAS_TRAIN, V_REF, WAIT_COLUMNS, WAITS)


# =====================================================================
# Graphs
# =====================================================================
def make_atom_graph(rows, columns, kept_sites=None):
  """Subgraph of a triangular lattice. Returns (Pulser register, coupling matrix V_ij).
  V_ij = C6 / d_ij^6 for EVERY pair (no cutoff), 0 on the diagonal."""
  lattice = Register.triangular_lattice(rows, columns, spacing=SPACING, prefix="q")
  coordinates = [lattice.qubits[qubit_id] for qubit_id in lattice.qubit_ids]
  if kept_sites is not None:
    coordinates = [coordinates[site] for site in kept_sites]
  register = Register({f"q{atom}": coordinate for atom, coordinate in enumerate(coordinates)})

  positions = np.array(coordinates)
  distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
  couplings = np.zeros_like(distances, dtype=np.float32)
  off_diagonal = ~np.eye(len(positions), dtype=bool)
  couplings[off_diagonal] = MockDevice.interaction_coeff / distances[off_diagonal] ** 6
  return register, couplings


ATOM_REGISTERS = {}
COUPLING_MATRICES = {}  # graph name -> coupling matrix V_ij [N, N]
for _graph_name, _graph_spec in GRAPH_SPECS.items():
  ATOM_REGISTERS[_graph_name], COUPLING_MATRICES[_graph_name] = make_atom_graph(*_graph_spec)


def atom_positions(register):
  """[N, 2] positions in µm, in atom order."""
  return np.array([register.qubits[qubit_id] for qubit_id in register.qubit_ids])


def atom_degrees(couplings):
  """Number of nearest neighbours of each atom (coupling >= V_REF / 2)."""
  return (couplings > 0.5 * V_REF).sum(axis=1)


# =====================================================================
# Ramsey simulation
# =====================================================================
def ramsey_waveforms(wait_ns):
  """The three steps of the protocol: π/2 pulse, free evolution, π/2 pulse (Rabi amplitude Ω(t))."""
  return (BlackmanWaveform(PULSE_DURATION_NS, np.pi / 2),
          ConstantWaveform(wait_ns, 0.0),
          BlackmanWaveform(PULSE_DURATION_NS, np.pi / 2))


def ramsey_populations(register, theta, wait_ns):
  """π/2 - wait - π/2 with dephasing theta. Returns P(atom i in |1>) for every atom, shape [N]."""
  sequence = Sequence(register, MockDevice)
  sequence.declare_channel("rydberg_global", "rydberg_global")
  for waveform in ramsey_waveforms(wait_ns):
    sequence.add(Pulse.ConstantDetuning(waveform, detuning=0.0, phase=0.0), "rydberg_global")
  config = QutipConfig(
      observables=[BitStrings(num_shots=N_SHOTS)],
      sampling_rate=0.05,
      noise_model=NoiseModel(dephasing_rate=float(theta)),
  )
  counts = QutipBackendV2(sequence, config=config).run().bitstrings[-1]  # e.g. {"0101": 12, ...}
  outcome_bits = np.array([[int(bit) for bit in bitstring] for bitstring in counts], dtype=np.float64)
  frequencies = np.array(list(counts.values()), dtype=np.float64) / N_SHOTS
  return (frequencies @ outcome_bits).astype(np.float32)


def build_simulation_table():
  """Simulates every graph at every theta and wait time. One row per (graph, theta, atom):
     graph | split | theta | atom | degree | population_wait_250 | population_wait_800 | ..."""
  start_time = time.time()
  rows = []
  for graph_name in GRAPH_SPECS:
    split = "test" if graph_name in TEST_GRAPHS else "train"
    thetas = THETAS_TEST if split == "test" else THETAS_TRAIN
    degrees = atom_degrees(COUPLING_MATRICES[graph_name])
    for theta in thetas:
      populations = np.stack(  # [N, len(WAITS)]
          [ramsey_populations(ATOM_REGISTERS[graph_name], theta, wait_ns) for wait_ns in WAITS], axis=1)
      for atom, atom_populations in enumerate(populations):
        rows.append({"graph": graph_name, "split": split, "theta": float(theta), "atom": atom,
                     "degree": int(degrees[atom]),
                     **dict(zip(WAIT_COLUMNS, atom_populations.tolist()))})
    print(f"  {graph_name:14s} ({time.time() - start_time:6.1f} s elapsed)")
  return pl.DataFrame(rows)
