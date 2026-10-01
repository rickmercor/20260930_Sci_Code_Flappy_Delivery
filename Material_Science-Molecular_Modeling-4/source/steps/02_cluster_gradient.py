"""
Compute the exact analytic gradient of the cluster potential with respect to every coordinate, distributing bond and angle forces correctly among the three atoms of each monomer and applying action-reaction between Lennard-Jones partners. Same validation as the energy.

Every later stage consumes this gradient: the Hessian blocks differentiate it and the relaxation update steps along it, so a sign or transpose error here poisons everything downstream.

Returns
-------
float64 array (I, 3, 3): dV/dcoords
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cluster_gradient(coords):
    """coords: float array (I, 3, 3).

    Returns float64 array (I, 3, 3): the exact analytic gradient of the
    total potential with respect to every coordinate."""
    return np.zeros_like(np.asarray(coords, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: analytic gradient of the cluster potential."""

import numpy as np

_PAR = {
    "kb": 800.0, "r0": 1.0, "kth": 120.0, "th0": 1.9106332362490186,
    "epsB": 1.2, "sigB": 2.6, "epsA": 0.05, "sigA": 1.4,
    "mA": 1.0, "mB": 16.0,
}
_PAIRS = ((1, 1, "epsB", "sigB"), (0, 0, "epsA", "sigA"), (0, 2, "epsA", "sigA"),
          (2, 0, "epsA", "sigA"), (2, 2, "epsA", "sigA"))




def _oracle_cluster_gradient(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or r.shape[0] < 1 or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    kb, r0 = _PAR["kb"], _PAR["r0"]
    kth, th0 = _PAR["kth"], _PAR["th0"]
    g = np.zeros_like(r)
    for i in range(I):
        A1, B, A2 = r[i]
        for (ai, a) in ((0, A1), (2, A2)):
            d = a - B
            L = np.linalg.norm(d)
            gd = kb * (L - r0) * d / L
            g[i, ai] += gd
            g[i, 1] -= gd
        u, w = A1 - B, A2 - B
        lu, lw = np.linalg.norm(u), np.linalg.norm(w)
        c = float(np.dot(u, w) / (lu * lw))
        c = max(-1.0, min(1.0, c))
        th = np.arccos(c)
        s = np.sqrt(max(1e-15, 1.0 - c * c))
        pref = kth * (th - th0) * (-1.0 / s)
        dc_du = w / (lu * lw) - c * u / (lu * lu)
        dc_dw = u / (lu * lw) - c * w / (lw * lw)
        g[i, 0] += pref * dc_du
        g[i, 2] += pref * dc_dw
        g[i, 1] -= pref * (dc_du + dc_dw)
    for i in range(I):
        for j in range(i + 1, I):
            for (ai, aj, ek, sk) in _PAIRS:
                eps, sig = _PAR[ek], _PAR[sk]
                d = r[i, ai] - r[j, aj]
                L2 = float(np.dot(d, d))
                s6 = (sig * sig / L2) ** 3
                f = 24.0 * eps * (2.0 * s6 * s6 - s6) / L2
                g[i, ai] -= f * d
                g[j, aj] += f * d
    return g.astype(np.float64, copy=False)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'coords = np.array([[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]]])', "call": "cluster_gradient(coords)", "gold_call": "_oracle_cluster_gradient(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]])', "call": "cluster_gradient(coords)", "gold_call": "_oracle_cluster_gradient(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[1.15, 0.05, 0.1], [0.1, 0.05, 0.0], [-0.35, 1.05, 0.0]], [[-3.9, 0.0, 0.0], [-2.95, -0.1, 0.0], [-2.6, 0.75, -0.55]]])', "call": "cluster_gradient(coords)", "gold_call": "_oracle_cluster_gradient(coords)", "tol": 1e-09},
    ]
