"""
Construct the shifted explicit Runge--Kutta coefficient matrix used in the source's unified stage notation.

The source shifts the nontrivial explicit-stage rows and final RK weights into one square matrix so intermediate and final HERK problems use the same indexing. The benchmark uses Heun, SSP3, and classical RK4.

Returns
-------
np.ndarray, the square shifted RK coefficient matrix of shape (order, order)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shifted_rk_matrix(order: int) -> "np.ndarray":
    """Return the source's shifted RK coefficient matrix for a supported order.

    Parameters
    ----------
    order : int
        Runge--Kutta order and stage count; supported values are 2, 3, and 4.

    Returns
    -------
    alpha : np.ndarray
        Square float array of shape (order, order) containing the shifted
        stage coefficients and final weights.

    Raises
    ------
    ValueError
        If order is not 2, 3, or 4.
    """
    return alpha

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_shifted_rk_matrix(order: int) -> "np.ndarray":
    if order == 2:
        return np.array([[1.0, 0.0],
                         [0.5, 0.5]], dtype=float)
    if order == 3:
        return np.array([[1.0, 0.0, 0.0],
                         [0.25, 0.25, 0.0],
                         [1.0/6.0, 1.0/6.0, 2.0/3.0]], dtype=float)
    if order == 4:
        return np.array([[0.5, 0.0, 0.0, 0.0],
                         [0.0, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]], dtype=float)
    raise ValueError("order must be one of 2, 3, or 4")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the three supported RK orders."""
    return [
        {"setup": "order=2", "call": "shifted_rk_matrix(order)", "gold_call": "_oracle_shifted_rk_matrix(order)", "tol": 1e-14},
        {"setup": "order=3", "call": "shifted_rk_matrix(order)", "gold_call": "_oracle_shifted_rk_matrix(order)", "tol": 1e-14},
        {"setup": "order=4", "call": "shifted_rk_matrix(order)", "gold_call": "_oracle_shifted_rk_matrix(order)", "tol": 1e-14},
    ]
