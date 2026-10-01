"""
Evaluate the stabilization scale and complex Fourier symbol of the source's space-first VMS/RK construction.

The conventional vertical method first forms the VMS semi-discrete operator. Its dimensionless stabilization scale and complex Fourier symbol differ from the time-first construction.

Returns
-------
np.ndarray, [tau_diamond_star, Re(gamma), Im(gamma)]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vertical_vms_fourier_symbol(K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    """Return the dimensionless stabilization scale and complex Fourier symbol.

    Parameters
    ----------
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.

    Returns
    -------
    symbol : np.ndarray
        Float array ``[tau_diamond_star, Re(gamma), Im(gamma)]``.
    """
    return symbol

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_vertical_vms_fourier_symbol(K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    K = float(K_star)
    a = float(a_star)
    k = float(kappa_star)
    tau_diamond_star = (4.0 + 4.0 * a * a + 144.0 * k * k) ** (-0.5)
    c1 = np.cos(K)
    c2 = np.cos(2.0 * K)
    s1 = np.sin(K)
    s2 = np.sin(2.0 * K)
    t = tau_diamond_star

    numerator = (
        (20.0 * t * a * a + 20.0 * k + 120.0 * t * k * k) * c2
        + (40.0 * t * a * a + 40.0 * k - 480.0 * t * k * k) * c1
        - 60.0 * t * a * a - 60.0 * k + 360.0 * t * k * k
        - 1j * ((5.0 * a + 120.0 * t * k * a) * s2
                + (50.0 * a - 240.0 * t * k * a) * s1)
    )
    denominator = (
        (1.0 + 20.0 * t * k) * c2
        + (26.0 + 40.0 * t * k) * c1
        + 33.0 - 60.0 * t * k
        - 1j * (5.0 * t * a * s2 + 50.0 * t * a * s1)
    )
    gamma = numerator / denominator
    return np.array([tau_diamond_star, gamma.real, gamma.imag], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three convection/diffusion and wavenumber regimes."""
    return [
        {"setup": "K_star=0.75; a_star=0.12; kappa_star=0.060", "call": "vertical_vms_fourier_symbol(K_star,a_star,kappa_star)", "gold_call": "_oracle_vertical_vms_fourier_symbol(K_star,a_star,kappa_star)", "tol": 1e-13},
        {"setup": "K_star=1.80; a_star=0.40; kappa_star=0.030", "call": "vertical_vms_fourier_symbol(K_star,a_star,kappa_star)", "gold_call": "_oracle_vertical_vms_fourier_symbol(K_star,a_star,kappa_star)", "tol": 1e-13},
        {"setup": "K_star=2.50; a_star=0.56; kappa_star=0.020", "call": "vertical_vms_fourier_symbol(K_star,a_star,kappa_star)", "gold_call": "_oracle_vertical_vms_fourier_symbol(K_star,a_star,kappa_star)", "tol": 1e-13},
    ]
