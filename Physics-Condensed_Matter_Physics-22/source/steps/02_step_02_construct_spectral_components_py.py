"""
Construct the pairwise coefficients, resonance centers, and transport broadenings that define the singular and regular harmonic heat-current spectral contributions.

The harmonic heat-current spectrum separates into a singular low-frequency contribution and a regular finite-frequency contribution. Each ordered pair of vibrational modes contributes a Lorentzian.

For the mode-independent transport damping convention,

$$

\Gamma_{nm}=2\Gamma^{tr}.

$$

The singular contribution is

$$

\Lambda_x^s(\omega)=\sum_{n,m}A_{nm}^s\frac{\Gamma_{nm}}{(\omega-\omega_m+\omega_n)^2+\Gamma_{nm}^2},

$$

where

$$

A_{nm}^s=(\nu_{nm}^{x})^2\frac{\omega_n}{e^{\beta\omega_n}-1}\frac{\omega_m}{1-e^{-\beta\omega_m}}\frac{(\omega_n+\omega_m)^2}{4\omega_n\omega_m}.

$$

The regular contribution is

$$

\Lambda_x^r(\omega)=\sum_{n,m}A_{nm}^r\frac{\Gamma_{nm}}{(\omega-\omega_n-\omega_m)^2+\Gamma_{nm}^2},

$$

where

$$

A_{nm}^r=\frac{1}{8}(\nu_{nm}^{x})^2\frac{\omega_n}{1-e^{-\beta\omega_n}}\frac{\omega_m}{1-e^{-\beta\omega_m}}\frac{(\omega_n-\omega_m)^2}{2\omega_n\omega_m}.

$$

Thus the singular Lorentzians are centered at $\omega_m-\omega_n$, while the regular Lorentzians are centered at $\omega_n+\omega_m$.

Returns
-------
np.ndarray of shape (5, N, N): singular coefficients, singular centers, regular coefficients, regular centers, and pair broadenings
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_spectral_components(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
) -> "np.ndarray":
    """Construct pairwise terms defining the harmonic heat-current spectrum.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix of shape (N, N).
    beta : float
        Positive inverse temperature.
    gamma_tr : float
        Positive mode-independent transport damping rate.

    Returns
    -------
    components : np.ndarray
        Array of shape (5, N, N). The slices contain the singular
        coefficients, singular centers, regular coefficients, regular
        centers, and pair broadenings, respectively.
    """
    return components

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_spectral_components(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
) -> "np.ndarray":
    omega = np.asarray(omega, dtype=float)
    nu = np.asarray(nu, dtype=float)

    wn = omega[:, None]
    wm = omega[None, :]
    nu2 = nu**2
    beta = float(beta)

    singular_coeff = (
        nu2
        * (wn / np.expm1(beta * wn))
        * (wm / (-np.expm1(-beta * wm)))
        * ((wn + wm) ** 2 / (4.0 * wn * wm))
    )

    regular_coeff = (
        0.125
        * nu2
        * (wn / (-np.expm1(-beta * wn)))
        * (wm / (-np.expm1(-beta * wm)))
        * ((wn - wm) ** 2 / (2.0 * wn * wm))
    )

    singular_center = wm - wn
    regular_center = wn + wm

    pair_width = np.full_like(
        singular_coeff,
        2.0 * float(gamma_tr),
        dtype=float,
    )

    return np.stack(
        (
            singular_coeff,
            singular_center,
            regular_coeff,
            regular_center,
            pair_width,
        ),
        axis=0,
    ).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """
import numpy as np
omega = np.array([0.75, 1.20, 1.85, 2.55], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.20, 0.55, 0.20, 0.10],
    [0.55, 1.00, 0.50, 0.18],
    [0.20, 0.50, 0.78, 0.40],
    [0.10, 0.18, 0.40, 0.58]
], dtype=float)
nu_gold = nu.copy()

beta = 1.5
gamma_tr = 0.09011247684
""",
            "call": "np.round(construct_spectral_components(omega, nu, beta, gamma_tr), 10)",
            "gold_call": "np.round(_oracle_construct_spectral_components(omega_gold, nu_gold, beta, gamma_tr), 10)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([1.0], dtype=float)
omega_gold = omega.copy()

nu = np.array([[0.8]], dtype=float)
nu_gold = nu.copy()

beta = 1.2
gamma_tr = 0.05
""",
            "call": "np.round(construct_spectral_components(omega, nu, beta, gamma_tr), 10)",
            "gold_call": "np.round(_oracle_construct_spectral_components(omega_gold, nu_gold, beta, gamma_tr), 10)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([1.00, 1.02, 1.80], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.0, 0.5, 0.2],
    [0.5, 0.9, 0.4],
    [0.2, 0.4, 0.7]
], dtype=float)
nu_gold = nu.copy()

beta = 2.0
gamma_tr = 0.04
""",
            "call": "np.round(construct_spectral_components(omega, nu, beta, gamma_tr), 10)",
            "gold_call": "np.round(_oracle_construct_spectral_components(omega_gold, nu_gold, beta, gamma_tr), 10)",
        },
    ]
