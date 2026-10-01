"""
Advance the source's space-first Fourier symbol with the selected explicit RK scheme.

For s=p, the source shows that the VMS-RK amplification is the p-th explicit-RK polynomial in the semi-discrete symbol gamma. The shifted matrix gives the same stage representation.

Returns
-------
complex, the one-step space-first Fourier amplification factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vertical_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float) -> complex:
    """Return the one-step complex amplification factor of the space-first scheme.

    Parameters
    ----------
    order : int
        Supported RK order/stage count 2, 3, or 4.
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.

    Returns
    -------
    zeta : complex
        One-step Fourier amplification factor.
    """
    return zeta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_vertical_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float) -> complex:
    alpha = _oracle_shifted_rk_matrix(order)
    symbol = _oracle_vertical_vms_fourier_symbol(K_star, a_star, kappa_star)
    gamma = complex(symbol[1], symbol[2])
    s = int(order)
    stage = np.ones(s, dtype=complex)
    for i in range(2, s + 1):
        stage[i-1] = 1.0 + gamma * np.dot(alpha[i-2, :i-1], stage[:i-1])
    return complex(1.0 + gamma * np.dot(alpha[-1, :], stage))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative second-, third-, and fourth-order cases."""
    return [
        {"setup": "order=2; K_star=0.75; a_star=0.12; kappa_star=0.060", "call": "vertical_vms_amplification(order,K_star,a_star,kappa_star)", "gold_call": "_oracle_vertical_vms_amplification(order,K_star,a_star,kappa_star)", "tol": 1e-13},
        {"setup": "order=3; K_star=2.15; a_star=0.48; kappa_star=0.025", "call": "vertical_vms_amplification(order,K_star,a_star,kappa_star)", "gold_call": "_oracle_vertical_vms_amplification(order,K_star,a_star,kappa_star)", "tol": 1e-13},
        {"setup": "order=4; K_star=2.50; a_star=0.56; kappa_star=0.020", "call": "vertical_vms_amplification(order,K_star,a_star,kappa_star)", "gold_call": "_oracle_vertical_vms_amplification(order,K_star,a_star,kappa_star)", "tol": 1e-13},
    ]
