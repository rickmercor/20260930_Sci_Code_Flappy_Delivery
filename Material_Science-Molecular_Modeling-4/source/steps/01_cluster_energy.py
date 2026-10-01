"""
Compute the total potential energy of a cluster of flexible A-B-A monomers: harmonic B-A bonds and the harmonic A1-B-A2 angle inside every monomer, plus intermonomer Lennard-Jones interactions, B with B and every A with every A, using the parameter values fixed in the problem statement. Validate the coordinate array and raise ValueError on malformed input.

The all-atom potential is the ground truth for the whole pipeline. Bonds and angle are exactly harmonic, so the stiff intramolecular subspace is quadratic by construction.

Returns
-------
float: the total potential energy of the cluster
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cluster_energy(coords):
    """coords: float array (I, 3, 3), atom order [A1, B, A2] per monomer,
    laid out (monomer, atom, xyz).

    Returns float: the total potential energy of the cluster."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: total potential energy of a flexible A-B-A cluster."""

import numpy as np

_PAR = {
    "kb": 800.0, "r0": 1.0, "kth": 120.0, "th0": 1.9106332362490186,
    "epsB": 1.2, "sigB": 2.6, "epsA": 0.05, "sigA": 1.4,
    "mA": 1.0, "mB": 16.0,
}
_PAIRS = ((1, 1, "epsB", "sigB"), (0, 0, "epsA", "sigA"), (0, 2, "epsA", "sigA"),
          (2, 0, "epsA", "sigA"), (2, 2, "epsA", "sigA"))




def _validate_coords(coords):
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 3 or c.shape[1:] != (3, 3) or c.shape[0] < 1 or not np.all(np.isfinite(c)):
        raise ValueError("invalid coordinates")
    return c


def _oracle_cluster_energy(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or r.shape[0] < 1 or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    kb, r0 = _PAR["kb"], _PAR["r0"]
    kth, th0 = _PAR["kth"], _PAR["th0"]
    V = 0.0
    for i in range(I):
        A1, B, A2 = r[i]
        for a in (A1, A2):
            V += 0.5 * kb * (np.linalg.norm(a - B) - r0) ** 2
        u, w = A1 - B, A2 - B
        c = float(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)))
        c = max(-1.0, min(1.0, c))
        V += 0.5 * kth * (np.arccos(c) - th0) ** 2
    for i in range(I):
        for j in range(i + 1, I):
            for (ai, aj, ek, sk) in _PAIRS:
                eps, sig = _PAR[ek], _PAR[sk]
                d = r[i, ai] - r[j, aj]
                s6 = (sig * sig / float(np.dot(d, d))) ** 3
                V += 4.0 * eps * (s6 * s6 - s6)
    return float(V)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'coords = np.array([[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]]])', "call": "cluster_energy(coords)", "gold_call": "_oracle_cluster_energy(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]])', "call": "cluster_energy(coords)", "gold_call": "_oracle_cluster_energy(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[1.15, 0.05, 0.1], [0.1, 0.05, 0.0], [-0.35, 1.05, 0.0]], [[-3.9, 0.0, 0.0], [-2.95, -0.1, 0.0], [-2.6, 0.75, -0.55]]])', "call": "cluster_energy(coords)", "gold_call": "_oracle_cluster_energy(coords)", "tol": 1e-09},
    ]
