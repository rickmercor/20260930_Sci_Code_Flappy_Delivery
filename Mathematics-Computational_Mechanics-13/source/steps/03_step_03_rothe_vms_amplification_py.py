"""
Compute the source's time-first RK/VMS one-step amplification factor.

The time-first construction applies explicit RK before VMS, producing a shifted-stage recursion in the three Fourier coefficients and the earlier stage amplification factors.

Returns
-------
complex, the one-step time-first Fourier amplification factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rothe_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> complex:
    """Return the one-step complex amplification factor of the time-first scheme.

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
    tau_star : float
        Dimensionless time-first fine-scale parameter.

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


def _oracle_rothe_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> complex:
    alpha = _oracle_shifted_rk_matrix(order)
    lambda1, lambda2, lambda3 = _oracle_rothe_vms_fourier_coefficients(K_star, a_star, kappa_star, tau_star)
    s = int(order)
    stage = np.ones(s, dtype=complex)
    stage[1] = 1.0 + (lambda1 + lambda2) * alpha[0, 0]

    for i in range(3, s + 1):
        row = i - 2
        sum_alpha = np.sum(alpha[row, :i-1])
        sum_alpha_zeta = np.dot(alpha[row, :i-1], stage[:i-1])
        nested = 0.0j
        for j in range(2, i):
            nested += alpha[row, j-1] * np.dot(alpha[j-2, :j-1], stage[:j-1])
        stage[i-1] = 1.0 + lambda1 * sum_alpha + lambda2 * sum_alpha_zeta + lambda3 * nested

    nested_final = 0.0j
    for i in range(2, s + 1):
        nested_final += alpha[-1, i-1] * np.dot(alpha[i-2, :i-1], stage[:i-1])

    zeta = (1.0
            + lambda1 * np.sum(alpha[-1, :])
            + lambda2 * np.dot(alpha[-1, :], stage)
            + lambda3 * nested_final)
    return complex(zeta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative second-, third-, and fourth-order cases."""
    return [
        {"setup": "order=2; K_star=0.75; a_star=0.12; kappa_star=0.060; tau_star=0.37", "call": "rothe_vms_amplification(order,K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_rothe_vms_amplification(order,K_star,a_star,kappa_star,tau_star)", "tol": 1e-13},
        {"setup": "order=3; K_star=2.15; a_star=0.48; kappa_star=0.025; tau_star=0.37", "call": "rothe_vms_amplification(order,K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_rothe_vms_amplification(order,K_star,a_star,kappa_star,tau_star)", "tol": 1e-13},
        {"setup": "order=4; K_star=2.50; a_star=0.56; kappa_star=0.020; tau_star=0.43", "call": "rothe_vms_amplification(order,K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_rothe_vms_amplification(order,K_star,a_star,kappa_star,tau_star)", "tol": 1e-13},
    ]
