"""
Detective switching test; return 1.0 iff the one-sample branch is selected.

The matvec budget can be spent two ways: almost entirely on the preconditioner with a single residual probe, or split so several probes average down the stochastic error. The right choice depends on whether enriching the sketch is still reducing the residual. Comparing the diagnostic at two sketch widths, weighted by how the budget would be reallocated, decides the branch.

Returns
-------
float, 1.0 if the one-sample branch is selected and 0.0 otherwise, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def detective_one_sample_flag(
    errF2_fine: float,
    errF2_coarse: float,
    ell: int,
    m: int,
    beta: float,
) -> float:
    """Return 1.0 if the one-sample branch is chosen, else 0.0.

    

    Parameters
    ----------
    errF2_fine : float
        Leave-one-out squared residual at rank floor(beta*ell).
    errF2_coarse : float
        Leave-one-out squared residual at rank floor(beta^2*ell).
    ell, m : int
        Budget parameters.
    beta : float
        Detective fraction in (0, 1).

    Returns
    -------
    float
        1.0 for one-sample, 0.0 for alpha-rank.

    Raises
    ------
    ValueError
        If ``beta`` is not strictly inside ``(0, 1)``, or if ``ell < 1`` or
        ``m < 1``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_detective_one_sample_flag(
    errF2_fine: float,
    errF2_coarse: float,
    ell: int,
    m: int,
    beta: float,
) -> float:
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")
    if ell < 1 or m < 1:
        raise ValueError("ell and m must be positive")
    denom = (1.0 - beta) * beta * ell + m
    lhs = (m / denom) * errF2_coarse
    return 1.0 if lhs >= errF2_fine else 0.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "detective_one_sample_flag(25.57396705484538, 58.2027174421603, 16, 5, 0.75)",
            "gold_call": "_oracle_detective_one_sample_flag(25.57396705484538, 58.2027174421603, 16, 5, 0.75)",
        },
        {
            "setup": "",
            "call": "detective_one_sample_flag(100.0, 10.0, 16, 5, 0.75)",
            "gold_call": "_oracle_detective_one_sample_flag(100.0, 10.0, 16, 5, 0.75)",
        },
        {
            "setup": "",
            "call": "detective_one_sample_flag(1.0, 1.0, 8, 2, 0.5)",
            "gold_call": "_oracle_detective_one_sample_flag(1.0, 1.0, 8, 2, 0.5)",
        },
    ]
