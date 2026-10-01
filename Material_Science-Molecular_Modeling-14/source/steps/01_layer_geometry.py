"""
Build the rotated geometry of the three graphene layers for a given angle triple: per layer, the Bravais matrix, the reciprocal matrix, the two Dirac points and the B-sublattice offset, packed row-wise as [A (4, row-major), B (4, row-major), K (2), K' (2), tau_B (2)], using the source's base lattice conventions with lattice constant a = 1.42 sqrt(3).

Every truncation criterion and every phase in the momentum space Hamiltonian is expressed through these per-layer matrices and K points; their base forms are the source's.

Returns
-------
return (3, 14) float64: per-layer packed geometry
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def layer_geometry(thetas):
    """thetas: (3,) rotation angles in radians. Returns (3, 14) float64: for
    each layer j the counterclockwise rotation by thetas[j] applied to the
    source's base Bravais matrix, reciprocal matrix, both Dirac points and
    the B-sublattice offset, packed as [A(4), B(4), K(2), K'(2), tau_B(2)]
    with matrices flattened row-major. Raises ValueError on a wrong shape or
    nonfinite input."""
    return np.zeros((3, 14))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: rotated layer geometry with the source's base conventions."""

import numpy as np

_A_CONST = 1.42 * np.sqrt(3.0)


def _base_geometry():
    a = _A_CONST
    A = (a / 2.0) * np.array([[2.0, 1.0], [0.0, np.sqrt(3.0)]])
    B = (2.0 * np.pi / (3.0 * a)) * np.array([[3.0, 0.0], [-np.sqrt(3.0), 2.0 * np.sqrt(3.0)]])
    K = (4.0 * np.pi / (3.0 * a)) * np.array([1.0, 0.0])
    Kp = (2.0 * np.pi / (3.0 * a)) * np.array([1.0, np.sqrt(3.0)])
    tauB = (a / 2.0) * np.array([1.0, np.sqrt(3.0) / 3.0])
    return A, B, K, Kp, tauB


def _oracle_layer_geometry(thetas):
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.shape != (3,) or not np.all(np.isfinite(thetas)):
        raise ValueError("invalid angle triple")
    A0, B0, K0, Kp0, tau0 = _base_geometry()
    out = np.empty((3, 14))
    for j in range(3):
        c, s = np.cos(float(thetas[j])), np.sin(float(thetas[j]))
        R = np.array([[c, -s], [s, c]])
        out[j, 0:4] = (R @ A0).reshape(-1)
        out[j, 4:8] = (R @ B0).reshape(-1)
        out[j, 8:10] = R @ K0
        out[j, 10:12] = R @ Kp0
        out[j, 12:14] = R @ tau0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n', "call": "layer_geometry(_n.array([-0.06,0.0,0.11]))", "gold_call": "_oracle_layer_geometry(_n.array([-0.06,0.0,0.11]))", "tol": 1e-09},
        {"setup": 'import numpy as _n', "call": "layer_geometry(_n.array([-0.09,0.0,0.07]))", "gold_call": "_oracle_layer_geometry(_n.array([-0.09,0.0,0.07]))", "tol": 1e-09},
        {"setup": 'import numpy as _n', "call": "layer_geometry(_n.array([-0.05,0.0,0.09]))", "gold_call": "_oracle_layer_geometry(_n.array([-0.05,0.0,0.09]))", "tol": 1e-09},
    ]
