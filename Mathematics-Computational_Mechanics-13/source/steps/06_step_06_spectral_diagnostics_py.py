"""
Convert a complex amplification factor into dimensionless amplitude, damping, and frequency diagnostics.

The discrete exponent follows from the principal complex logarithm of the amplification factor. Normalizing damping and frequency by the exact advection--diffusion mode yields dimensionless spectral ratios.

Returns
-------
np.ndarray, [abs(zeta), damping_ratio, frequency_ratio]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_diagnostics(zeta: complex, K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    """Return amplitude and normalized damping/frequency diagnostics.

    Parameters
    ----------
    zeta : complex
        Nonzero one-step Fourier amplification factor.
    K_star : float
        Positive dimensionless wavenumber.
    a_star : float
        Positive dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.

    Returns
    -------
    diagnostics : np.ndarray
        Float array ``[abs(zeta), damping_ratio, frequency_ratio]`` using the
        principal complex argument in (-pi, pi].
    """
    return diagnostics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_diagnostics(zeta: complex, K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    zeta = complex(zeta)
    K = float(K_star)
    a = float(a_star)
    k = float(kappa_star)
    amplitude = abs(zeta)
    damping_ratio = -np.log(amplitude) / (k * K * K)
    frequency_ratio = np.angle(zeta) / (-a * K)
    return np.array([amplitude, damping_ratio, frequency_ratio], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return diagnostics for paper-derived and simple analytic amplification factors."""
    return [
        {"setup": "zeta=complex(0.9627681949383489,-0.0869607825162780); K_star=0.75; a_star=0.12; kappa_star=0.060", "call": "spectral_diagnostics(zeta,K_star,a_star,kappa_star)", "gold_call": "_oracle_spectral_diagnostics(zeta,K_star,a_star,kappa_star)", "tol": 1e-13},
        {"setup": "zeta=complex(0.6460145595782506,-0.6419499973683029); K_star=1.80; a_star=0.40; kappa_star=0.030", "call": "spectral_diagnostics(zeta,K_star,a_star,kappa_star)", "gold_call": "_oracle_spectral_diagnostics(zeta,K_star,a_star,kappa_star)", "tol": 1e-13},
        {"setup": "import numpy as np\nK_star=1.25; a_star=0.30; kappa_star=0.08; zeta=np.exp(-kappa_star*K_star*K_star-1j*a_star*K_star)", "call": "spectral_diagnostics(zeta,K_star,a_star,kappa_star)", "gold_call": "_oracle_spectral_diagnostics(zeta,K_star,a_star,kappa_star)", "tol": 1e-13},
    ]
