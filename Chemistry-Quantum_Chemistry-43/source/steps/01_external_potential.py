"""
Return the optical-lattice potential of the source's periodic Gross-Pitaevskii example sampled on the uniform periodic grid x_j = j a / M for j = 0, ..., M-1 of the cell [0, a): v_ext(x) = -sin^2(pi x / a). Raise ValueError if a is not positive or M is below 8.

The source tests its inversion scheme on a one-dimensional periodic Gross-Pitaevskii model in an optical-lattice potential, which has a single well per cell and a smooth, nodeless ground state, so that all spectral quantities converge rapidly.

Returns
-------
ndarray of float64 with shape (M,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def external_potential(a: float, M: int) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _oracle_external_potential(a: float, M: int) -> "np.ndarray":
    """The optical-lattice potential of the source's Sec IV B on the uniform periodic grid
    x_j = j a / M, j = 0..M-1:  v_ext(x) = -sin^2(pi x / a)."""
    a, M = float(a), int(M)
    if a <= 0.0 or M < 8:
        raise ValueError("need a > 0 and M >= 8")
    x = np.arange(M) * a / M
    return -np.sin(np.pi * x / a) ** 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "a, M = 10.0, 256",
            "call": "external_potential(a, M)",
            "gold_call": "_oracle_external_potential(a, M)",
        },
        {
            "setup": "a, M = 8.0, 64",
            "call": "external_potential(a, M)",
            "gold_call": "_oracle_external_potential(a, M)",
        },
        {
            "setup": "a, M = 12.5, 96",
            "call": "external_potential(a, M)",
            "gold_call": "_oracle_external_potential(a, M)",
        },
    ]
