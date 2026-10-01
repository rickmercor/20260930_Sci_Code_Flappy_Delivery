"""
Implement reconstruct_interfaces to compute slope-limited left and right
values of a cell-averaged quantity at all periodic cell interfaces.

Reconstruct left- and right-biased values of a cell-averaged quantity at every interface of the uniform periodic mesh using van-Leer-limited linear reconstruction. Interface j separates cells (j-1) % N_x and j. Return the value approached from the left and the value approached from the right in that order, with one entry per periodic interface.

Returns
-------
tuple[np.ndarray, np.ndarray], (psi_L, psi_R), left- and right-biased interface values, each of shape (N_x,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconstruct_interfaces(
    psi: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    psi : numpy.ndarray
        Cell-averaged values, shape (N_x,).
    dx : float
        Uniform cell width.

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray]
        (psi_L, psi_R) each of shape (N_x,).
        psi_L[j] is the value reconstructed from the left cell at interface j.
        psi_R[j] is the value reconstructed from the right cell at interface j.

    Raises
    ------
    ValueError
        If dx is not positive.
    """
    return psi_L, psi_R

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _van_leer_slope_periodic(psi: np.ndarray, dx: float) -> np.ndarray:
    """Compute van-Leer limited slopes for each cell with periodic neighbors.

    Parameters
    ----------
    psi : numpy.ndarray
        Cell-averaged values, shape (N_x,).
    dx : float
        Uniform cell width.

    Returns
    -------
    slopes : numpy.ndarray
        Limited slope in each cell, shape (N_x,).
    """
    N = len(psi)
    slopes = np.zeros(N)
    for i in range(N):
        left = psi[(i - 1) % N]
        right = psi[(i + 1) % N]
        dL = (psi[i] - left) / dx
        dR = (right - psi[i]) / dx
        if dL * dR > 0.0:
            slopes[i] = 2.0 * dL * dR / (dL + dR)
    return slopes



def _oracle_reconstruct_interfaces(
    psi: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute van-Leer limited periodic reconstruction at cell interfaces."""
    if dx <= 0:
        raise ValueError("dx must be positive")
    N = len(psi)
    slopes = _van_leer_slope_periodic(psi, dx)

    psi_L = np.zeros(N)
    psi_R = np.zeros(N)

    for j in range(N):
        left_cell = (j - 1) % N
        psi_L[j] = psi[left_cell] + 0.5 * dx * slopes[left_cell]
        psi_R[j] = psi[j] - 0.5 * dx * slopes[j]

    return psi_L, psi_R

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # Normal: linear profile on periodic domain
        {
            "setup": """import numpy as np
dx = 0.1
psi = np.array([(i + 0.5) * dx for i in range(10)])
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in reconstruct_interfaces(psi.copy(), dx))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in _oracle_reconstruct_interfaces(psi.copy(), dx))",
        },
        # Boundary: step function, limiter should suppress oscillation
        {
            "setup": """import numpy as np
dx = 0.125
psi = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in reconstruct_interfaces(psi.copy(), dx))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in _oracle_reconstruct_interfaces(psi.copy(), dx))",
        },
        # Edge: constant profile, all slopes zero, L and R equal psi everywhere
        {
            "setup": """import numpy as np
dx = 0.2
psi = np.full(5, 3.0)
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in reconstruct_interfaces(psi.copy(), dx))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in _oracle_reconstruct_interfaces(psi.copy(), dx))",
        },
        # Smooth non-linear profile: dL != dR with dL*dR > 0, so the harmonic-mean
        # limiter is distinguishable from an unlimited central slope
        {
            "setup": """import numpy as np
N = 12
dx = 1.0 / N
xc = np.array([(i + 0.5) * dx for i in range(N)])
psi = 1.0 + 0.5 * np.sin(2.0 * np.pi * xc) + 0.25 * np.sin(4.0 * np.pi * xc)
""",
            "call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in reconstruct_interfaces(psi.copy(), dx))",
            "gold_call": "tuple(np.round(np.asarray(a, dtype=float), 10) for a in _oracle_reconstruct_interfaces(psi.copy(), dx))",
        },
    ]
