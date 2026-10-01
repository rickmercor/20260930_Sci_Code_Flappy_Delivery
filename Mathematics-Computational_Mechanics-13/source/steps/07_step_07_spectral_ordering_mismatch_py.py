"""
Measure the three-component spectral discrepancy caused by reversing VMS and RK ordering.

The two source constructions differ in dissipation, dispersion, and one-step magnitude. Their time-first-minus-space-first differences form one source-grounded mismatch vector.

Returns
-------
np.ndarray, [Delta damping ratio, Delta frequency ratio, Delta amplitude]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_ordering_mismatch(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    """Return the spectral mismatch vector between time-first and space-first schemes.

    Parameters
    ----------
    order : int
        Supported RK order/stage count 2, 3, or 4.
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Positive dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.
    tau_star : float
        Time-first dimensionless fine-scale parameter.

    Returns
    -------
    mismatch : np.ndarray
        Float array ``[Delta damping ratio, Delta frequency ratio,
        Delta amplification magnitude]``, with time-first minus space-first.
    """
    return mismatch

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_ordering_mismatch(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    zeta_rothe = _oracle_rothe_vms_amplification(order, K_star, a_star, kappa_star, tau_star)
    zeta_vertical = _oracle_vertical_vms_amplification(order, K_star, a_star, kappa_star)
    diag_rothe = _oracle_spectral_diagnostics(zeta_rothe, K_star, a_star, kappa_star)
    diag_vertical = _oracle_spectral_diagnostics(zeta_vertical, K_star, a_star, kappa_star)
    return np.array([
        diag_rothe[1] - diag_vertical[1],
        diag_rothe[2] - diag_vertical[2],
        diag_rothe[0] - diag_vertical[0],
    ], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return low-, middle-, and high-wavenumber ordering comparisons."""
    return [
        {"setup": "order=2; K_star=0.75; a_star=0.12; kappa_star=0.060; tau_star=0.37", "call": "spectral_ordering_mismatch(order,K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_spectral_ordering_mismatch(order,K_star,a_star,kappa_star,tau_star)", "tol": 2e-12},
        {"setup": "order=2; K_star=1.80; a_star=0.40; kappa_star=0.030; tau_star=0.37", "call": "spectral_ordering_mismatch(order,K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_spectral_ordering_mismatch(order,K_star,a_star,kappa_star,tau_star)", "tol": 2e-12},
        {"setup": "order=4; K_star=2.50; a_star=0.56; kappa_star=0.020; tau_star=0.43", "call": "spectral_ordering_mismatch(order,K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_spectral_ordering_mismatch(order,K_star,a_star,kappa_star,tau_star)", "tol": 2e-12},
    ]
