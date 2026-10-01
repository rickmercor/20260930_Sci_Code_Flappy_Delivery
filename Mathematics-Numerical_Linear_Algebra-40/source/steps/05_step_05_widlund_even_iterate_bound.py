"""
Widlund even-iterate theoretical H-error bound from the spectral width

The energy-optimal short-recurrence method for this preconditioned dissipative operator admits a sharper a-priori estimate on its even-numbered iterates than on general iterates, because consecutive steps pair up in the underlying three-term structure. Report that even-iterate estimate for the requested iteration count, using the imaginary half-width from $spectral_width_lambda$ and cross-checking against $rapoport_residual_reduction_factor$ so that the residual-minimizing factor cannot be substituted.

Returns
-------
float, Widlund even-iterate H-error bound for the given even k
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def widlund_even_iterate_bound(
    n: int, nu: float, b_adv: float, c: float, k: int
) -> float:
    """Return the Widlund even-iterate theoretical bound for iteration count k.

    
    Returns
    -------
    float
        Even-iterate Widlund bound.

    Raises
    ------
    ValueError
        If k is not a positive even integer, or the reduction factors are inconsistent.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_widlund_even_iterate_bound(
    n: int, nu: float, b_adv: float, c: float, k: int
) -> float:
    if k < 2 or k % 2 != 0:
        raise ValueError("k must be a positive even integer for Eq. (2.3)")
    lam = _oracle_spectral_width_lambda(n, nu, b_adv, c)
    rho_r = _oracle_rapoport_residual_reduction_factor(n, nu, b_adv, c)
    root = np.sqrt(1.0 + lam**2)
    rho_w = (root - 1.0) / (root + 1.0)
    if lam > 1e-14 and abs(rho_w - rho_r) < 1e-14:
        raise ValueError("Widlund and Rapoport factors collapsed unexpectedly")
    if lam > 1e-14 and rho_w >= rho_r - 1e-15:
        raise ValueError("Widlund factor must be strictly below Rapoport factor for lambda>0")
    m = k // 2
    return float(2.0 * (rho_w**m))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "widlund_even_iterate_bound(8, 0.37, 1.85, 0.18, 6)",
            "gold_call": "_oracle_widlund_even_iterate_bound(8, 0.37, 1.85, 0.18, 6)",
        },
        {
            "setup": "import numpy as np",
            "call": "widlund_even_iterate_bound(4, 1.0, 0.0, 0.5, 2)",
            "gold_call": "_oracle_widlund_even_iterate_bound(4, 1.0, 0.0, 0.5, 2)",
        },
        {
            "setup": "import numpy as np",
            "call": "widlund_even_iterate_bound(5, 0.2, 1.8, 0.05, 6)",
            "gold_call": "_oracle_widlund_even_iterate_bound(5, 0.2, 1.8, 0.05, 6)",
        },
        {
            "setup": "import numpy as np",
            "call": "widlund_even_iterate_bound(8, 0.015, 3.5, 0.0, 8)",
            "gold_call": "_oracle_widlund_even_iterate_bound(8, 0.015, 3.5, 0.0, 8)",
        },
        {
            "setup": "import numpy as np",
            "call": "widlund_even_iterate_bound(3, 2.0, -1.0, 1.0, 2)",
            "gold_call": "_oracle_widlund_even_iterate_bound(3, 2.0, -1.0, 1.0, 2)",
        },
        {
            "setup": "import numpy as np",
            "call": "widlund_even_iterate_bound(9, 0.05, 2.75, 0.1, 6)",
            "gold_call": "_oracle_widlund_even_iterate_bound(9, 0.05, 2.75, 0.1, 6)",
        },
    ]
