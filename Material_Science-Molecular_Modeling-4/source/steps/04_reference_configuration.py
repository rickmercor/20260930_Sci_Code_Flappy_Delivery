"""
Rebuild every monomer at the isolated-monomer internal minimum inside its own frame using the exact frame contract from the problem statement: origin at B, x along the normalized bond bisector, z along the normalized cross product of the two bond vectors, y completing the right-handed frame, A1 at plus half the equilibrium angle from x toward y and A2 at minus half, both at the equilibrium bond length. Handle the near-collinear degeneracy exactly as the statement prescribes: when the sine of the bond angle falls below the stated threshold the plane normal comes from the deterministic basis-vector fallback, and zero-length bonds or a degenerate bisector raise ValueError.

This is the zeroth-order point of the source construction: coarse-grained placement paired with the isolated-monomer internal minimum.

Returns
-------
float64 array (I, 3, 3): the reference-manifold configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reference_configuration(coords):
    """coords: float array (I, 3, 3).

    Returns float64 array (I, 3, 3): every monomer rebuilt at the
    isolated-monomer internal minimum in its own frame per the frame
    contract in the problem statement."""
    return np.zeros_like(np.asarray(coords, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: map a cluster configuration onto the reference manifold."""

import numpy as np

_R0 = 1.0
_TH0 = 1.9106332362490186
_SIN_TOL = 1e-8



def _oracle_reference_configuration(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    out = np.empty_like(r)
    for i in range(I):
        A1, B, A2 = r[i]
        u, w = A1 - B, A2 - B
        nu, nw = np.linalg.norm(u), np.linalg.norm(w)
        if nu <= 0.0 or nw <= 0.0 or not (np.isfinite(nu) and np.isfinite(nw)):
            raise ValueError("degenerate monomer geometry")
        bis = u / nu + w / nw
        nb = np.linalg.norm(bis)
        if nb <= _SIN_TOL or not np.isfinite(nb):
            raise ValueError("degenerate monomer geometry")
        x = bis / nb
        zc = np.cross(u, w)
        nz = np.linalg.norm(zc)
        if nz / (nu * nw) < _SIN_TOL:
            k = int(np.argmin(np.abs(x)))
            e = np.zeros(3)
            e[k] = 1.0
            z = e - np.dot(e, x) * x
            z = z / np.linalg.norm(z)
        else:
            z = zc / nz
        y = np.cross(z, x)
        ca, sa = np.cos(_TH0 / 2.0), np.sin(_TH0 / 2.0)
        out[i, 1] = B
        out[i, 0] = B + _R0 * (ca * x + sa * y)
        out[i, 2] = B + _R0 * (ca * x - sa * y)
    return out.astype(np.float64, copy=False)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'coords = np.array([[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]]])', "call": "reference_configuration(coords)", "gold_call": "_oracle_reference_configuration(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[1.15, 0.05, 0.1], [0.1, 0.05, 0.0], [-0.35, 1.05, 0.0]], [[-3.9, 0.0, 0.0], [-2.95, -0.1, 0.0], [-2.6, 0.75, -0.55]]])', "call": "reference_configuration(coords)", "gold_call": "_oracle_reference_configuration(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]])', "call": "reference_configuration(coords)", "gold_call": "_oracle_reference_configuration(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [0.60000000005, 0.79999999995, 0.0]], [[-3.9, 0.0, 0.0], [-2.95, -0.1, 0.0], [-2.6, 0.75, -0.55]]])', "call": "reference_configuration(coords)", "gold_call": "_oracle_reference_configuration(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[0.55, 0.83, -0.12], [-0.02, 0.03, 0.05], [0.61, -0.79, 0.18]], [[-3.2, 0.4, 0.7], [-2.5, -0.2, 0.1], [-1.7, 0.55, -0.6]]])', "call": "reference_configuration(coords)", "gold_call": "_oracle_reference_configuration(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[1.1, 0.2, 0.9], [0.15, -0.1, 0.2], [-0.6, 0.85, -0.35]], [[-3.9, 0.6, -0.8], [-3.0, 0.05, 0.02], [-2.2, -0.75, 0.55]], [[0.4, -3.6, 0.75], [-0.08, -2.9, 0.03], [0.85, -2.3, -0.7]], [[0.3, 0.8, -3.7], [-0.05, 0.02, -3.0], [0.75, -0.7, -2.4]]])', "call": "reference_configuration(coords)", "gold_call": "_oracle_reference_configuration(coords)", "tol": 1e-09},
    ]
