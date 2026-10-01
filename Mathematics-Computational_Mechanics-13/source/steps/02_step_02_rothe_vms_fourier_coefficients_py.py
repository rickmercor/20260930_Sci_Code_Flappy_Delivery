"""
Evaluate the three complex Fourier coefficients of the source's time-first RK/VMS construction.

The horizontal-method-of-lines Fourier stencil for C1 quadratic B-splines yields three source-specific coefficients that depend on K*, a*, kappa*, and tau* and drive the nested stage recursion.

Returns
-------
np.ndarray, complex vector (lambda1, lambda2, lambda3) of shape (3,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rothe_vms_fourier_coefficients(K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    """Return the three complex Fourier coefficients for the time-first scheme.

    Parameters
    ----------
    K_star : float
        Dimensionless wavenumber, in radians.
    a_star : float
        Dimensionless advective speed (Courant number).
    kappa_star : float
        Positive nondimensional diffusivity.
    tau_star : float
        Dimensionless fine-scale parameter in the open interval (0, 1).

    Returns
    -------
    lambdas : np.ndarray
        Complex array of shape (3,) ordered as (lambda1, lambda2, lambda3).
    """
    return lambdas

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rothe_vms_fourier_coefficients(K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    K = float(K_star)
    a = float(a_star)
    k = float(kappa_star)
    tau = float(tau_star)
    c1 = np.cos(K)
    c2 = np.cos(2.0 * K)
    s1 = np.sin(K)
    s2 = np.sin(2.0 * K)
    C = (1.0 - tau) * (c2 + 26.0 * c1 + 33.0)

    lambda1 = (
        20.0 * tau * k * c2 + 40.0 * tau * k * c1 - 60.0 * tau * k
        + 1j * (-5.0 * tau * a * s2 - 50.0 * tau * a * s1)
    ) / C

    lambda2 = (
        (-40.0 * tau * k + 20.0 * k) * c2
        + (-80.0 * tau * k + 40.0 * k) * c1
        + 120.0 * tau * k - 60.0 * k
        + 1j * ((10.0 * tau * a - 5.0 * a) * s2
                + (100.0 * tau * a - 50.0 * a) * s1)
    ) / C

    lambda3 = (
        (20.0 * tau * a * a + 120.0 * tau * k * k) * c2
        + (40.0 * tau * a * a - 480.0 * tau * k * k) * c1
        + 360.0 * tau * k * k - 60.0 * tau * a * a
        + 1j * (-120.0 * tau * k * a * s2 + 240.0 * tau * k * a * s1)
    ) / C

    return np.array([lambda1, lambda2, lambda3], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return low-, middle-, and high-wavenumber source configurations."""
    return [
        {"setup": "K_star=0.75; a_star=0.12; kappa_star=0.060; tau_star=0.37", "call": "rothe_vms_fourier_coefficients(K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_rothe_vms_fourier_coefficients(K_star,a_star,kappa_star,tau_star)", "tol": 1e-13},
        {"setup": "K_star=1.80; a_star=0.40; kappa_star=0.030; tau_star=0.37", "call": "rothe_vms_fourier_coefficients(K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_rothe_vms_fourier_coefficients(K_star,a_star,kappa_star,tau_star)", "tol": 1e-13},
        {"setup": "K_star=2.50; a_star=0.56; kappa_star=0.020; tau_star=0.43", "call": "rothe_vms_fourier_coefficients(K_star,a_star,kappa_star,tau_star)", "gold_call": "_oracle_rothe_vms_fourier_coefficients(K_star,a_star,kappa_star,tau_star)", "tol": 1e-13},
    ]
