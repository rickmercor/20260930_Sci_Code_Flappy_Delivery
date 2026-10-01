"""
Rapoport residual reduction factor from the spectral width

Two distinct short-recurrence methods are classically associated with this preconditioned dissipative setting: one is optimal for the error in the energy norm, the other minimizes a residual in the dual norm. They have different guaranteed per-step reduction factors, both expressible through the imaginary half-width of the preconditioned transport spectrum, and the two factors are easy to confuse in the literature. Report the factor belonging to the residual-minimizing method

Returns
-------
float, Rapoport residual reduction factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rapoport_residual_reduction_factor(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    """Return the Rapoport residual reduction factor for the given split.

    
    Returns
    -------
    float
        Rapoport reduction factor.

    Raises
    ------
    ValueError
        If the spectral width is negative or the two reduction factors collapse.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rapoport_residual_reduction_factor(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    lam = _oracle_spectral_width_lambda(n, nu, b_adv, c)
    if lam < 0:
        raise ValueError("spectral width must be nonnegative")
    root = np.sqrt(1.0 + lam**2)
    rho_r = float(lam / (root + 1.0))
    rho_w = float((root - 1.0) / (root + 1.0))
    if lam > 1e-14 and abs(rho_r - rho_w) < 1e-14:
        raise ValueError("Rapoport and Widlund factors collapsed unexpectedly")
    if lam > 1e-14 and rho_r <= rho_w + 1e-15:
        raise ValueError("Rapoport factor must exceed Widlund factor for lambda>0")
    return rho_r

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "rapoport_residual_reduction_factor(8, 0.37, 1.85, 0.18)",
            "gold_call": "_oracle_rapoport_residual_reduction_factor(8, 0.37, 1.85, 0.18)",
        },
        {
            "setup": "import numpy as np",
            "call": "rapoport_residual_reduction_factor(4, 1.0, 0.0, 0.5)",
            "gold_call": "_oracle_rapoport_residual_reduction_factor(4, 1.0, 0.0, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "rapoport_residual_reduction_factor(5, 0.1, 1.5, 0.0)",
            "gold_call": "_oracle_rapoport_residual_reduction_factor(5, 0.1, 1.5, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "rapoport_residual_reduction_factor(9, 0.01, 4.0, 0.001)",
            "gold_call": "_oracle_rapoport_residual_reduction_factor(9, 0.01, 4.0, 0.001)",
        },
        {
            "setup": "import numpy as np",
            "call": "rapoport_residual_reduction_factor(2, 3.0, -5.0, 0.5)",
            "gold_call": "_oracle_rapoport_residual_reduction_factor(2, 3.0, -5.0, 0.5)",
        },
        {
            "setup": "import numpy as np",
            "call": "rapoport_residual_reduction_factor(8, 0.015, 3.5, 0.0)",
            "gold_call": "_oracle_rapoport_residual_reduction_factor(8, 0.015, 3.5, 0.0)",
        },
    ]
