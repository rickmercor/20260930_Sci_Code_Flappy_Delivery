"""
Evaluate the landscape theorem's conditional Frobenius-error bound expression for an exact second-order critical point, given the measurement design's isometry constants and the noise level.

The theorem combines a restricted lower isometry constant, a tangent-space upper constant,

the adjoint-noise norm, the ranks, and the regularization weight.

Returns
-------
float, the conditional Frobenius-error envelope at the supplied regularization.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certification_bound(op_norm: float, r_star: int, r: int, alpha: float, beta: float, lam: float) -> float:
    """Evaluate the conditional theorem bound on the recovery error of an exact
    second-order critical point.

    Parameters
    ----------
    op_norm : float
        The operator norm of the adjoint measurement map applied to the noise, ||A^*(xi)||_op.
    r_star : int
        The true rank of the target state.
    r : int
        The ansatz rank used by the recovery method, r >= r_star.
    alpha : float
        The restricted lower isometry constant of the measurement design.
    beta : float
        The tangent-space upper isometry constant of the measurement design, beta >= alpha > 0.
    lam : float
        The regularization weight used by the recovery method, lam >= 0.

    Returns
    -------
    result : float
        The conditional theorem-bound value on the Frobenius-norm recovery error, as a
        native Python float. This function evaluates the bound expression; it does not
        establish that a separately computed finite iterate is an exact critical point.

    Raises
    ------
    ValueError
        If alpha <= 0, beta < alpha, or the ratio beta/alpha does not satisfy the bound's
        applicability threshold.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_certification_bound(op_norm: float, r_star: int, r: int, alpha: float, beta: float, lam: float) -> float:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if beta < alpha:
        raise ValueError("beta must be at least alpha")
    if beta / alpha >= 6.0 / (np.sqrt(5.0) + 2.0):
        raise ValueError("beta/alpha exceeds the bound's applicability threshold")

    numerator = (36.0 * lam + 12.0 * op_norm) * np.sqrt(r_star) + 10.0 * np.sqrt(r + r_star) * max(op_norm - lam, 0.0)
    denominator = 6.0 * alpha - (np.sqrt(5.0) + 2.0) * beta
    return float(numerator / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: generic values well within the applicability threshold ---
        {
            "setup": """
op_norm = 0.05
r_star = 1
r = 2
alpha = 3.0
beta = 3.2
lam = 0.01
""",
            "call": "certification_bound(op_norm, r_star, r, alpha, beta, lam)",
            "gold_call": "_oracle_certification_bound(op_norm, r_star, r, alpha, beta, lam)",
        },
        # --- Boundary: lambda equals op_norm exactly, so the positive-part term vanishes ---
        {
            "setup": """
op_norm = 0.04
r_star = 2
r = 3
alpha = 2.2
beta = 2.4
lam = 0.04
""",
            "call": "certification_bound(op_norm, r_star, r, alpha, beta, lam)",
            "gold_call": "_oracle_certification_bound(op_norm, r_star, r, alpha, beta, lam)",
        },
        # --- Edge: anisotropic sensing with nonzero regularization ---
        {
            "setup": """
op_norm = 0.06
r_star = 2
r = 2
alpha = 1.0
beta = 1.2
lam = 0.012
""",
            "call": "certification_bound(op_norm, r_star, r, alpha, beta, lam)",
            "gold_call": "_oracle_certification_bound(op_norm, r_star, r, alpha, beta, lam)",
            "tol": 1e-9,
        },
    ]
