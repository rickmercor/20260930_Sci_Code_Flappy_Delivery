"""
Convert harmonic and reconstructed zero-frequency spectral densities into harmonic conductivity, total conductivity, and percentage conductivity change.

For the reduced isotropic system with $k_B=\hbar=V=1$, the zero-frequency energy-current spectral density determines the dc thermal conductivity.

If $\Lambda_h(0)$ is the fixed harmonic contribution and $\widetilde{\Lambda}(0)$ is the reconstructed additional contribution, then

$$

\kappa_h=\beta^2\Lambda_h(0),\qquad \kappa_{\mathrm{tot}}=\beta^2\left[\Lambda_h(0)+\widetilde{\Lambda}(0)\right].

$$

The percentage change relative to the harmonic reconstruction is

$$

\Delta_\kappa=100\frac{\kappa_{\mathrm{tot}}-\kappa_h}{\kappa_h}.

$$

Returns
-------
np.ndarray of length 3 containing [harmonic conductivity, total conductivity, percentage change]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_conductivity_change(
    beta: float,
    harmonic_lambda_zero: float,
    additional_lambda_zero: float,
) -> "np.ndarray":
    """Compute conductivity values from zero-frequency spectral densities.

    Parameters
    ----------
    beta : float
        Finite strictly positive inverse temperature.
    harmonic_lambda_zero : float
        Finite strictly positive harmonic spectral density at zero frequency.
    additional_lambda_zero : float
        Finite nonnegative additional spectral density at zero frequency.

    Returns
    -------
    result : np.ndarray
        One-dimensional array of length 3 containing harmonic conductivity,
        total conductivity, and percentage conductivity change, in that order.

    Raises
    ------
    ValueError
        If beta is not finite and strictly positive, if the harmonic
        zero-frequency spectral density is not finite and strictly positive,
        or if the additional zero-frequency spectral density is not finite
        and nonnegative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_conductivity_change(
    beta: float,
    harmonic_lambda_zero: float,
    additional_lambda_zero: float,
) -> "np.ndarray":
    """Reference implementation."""
    beta = float(beta)
    harmonic_lambda_zero = float(harmonic_lambda_zero)
    additional_lambda_zero = float(additional_lambda_zero)

    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    if (
        not np.isfinite(harmonic_lambda_zero)
        or harmonic_lambda_zero <= 0.0
    ):
        raise ValueError(
            "harmonic_lambda_zero must be finite and strictly positive"
        )
    if (
        not np.isfinite(additional_lambda_zero)
        or additional_lambda_zero < 0.0
    ):
        raise ValueError(
            "additional_lambda_zero must be finite and nonnegative"
        )

    kappa_h = beta**2 * harmonic_lambda_zero

    kappa_total = beta**2 * (
        harmonic_lambda_zero
        + additional_lambda_zero
    )

    delta_percent = 100.0 * (
        kappa_total - kappa_h
    ) / kappa_h

    return np.array(
        [
            kappa_h,
            kappa_total,
            delta_percent,
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
beta = 1.5
harmonic_lambda_zero = 5.2
additional_lambda_zero = 0.0041
""",
            "call": "compute_conductivity_change(beta, harmonic_lambda_zero, additional_lambda_zero)",
            "gold_call": "_oracle_compute_conductivity_change(beta, harmonic_lambda_zero, additional_lambda_zero)",
        },
        {
            "setup": """import numpy as np
beta = 2.0
harmonic_lambda_zero = 3.0
additional_lambda_zero = 0.0
""",
            "call": "compute_conductivity_change(beta, harmonic_lambda_zero, additional_lambda_zero)",
            "gold_call": "_oracle_compute_conductivity_change(beta, harmonic_lambda_zero, additional_lambda_zero)",
        },
        {
            "setup": """import numpy as np
beta = 0.5
harmonic_lambda_zero = 0.25
additional_lambda_zero = 0.75
""",
            "call": "compute_conductivity_change(beta, harmonic_lambda_zero, additional_lambda_zero)",
            "gold_call": "_oracle_compute_conductivity_change(beta, harmonic_lambda_zero, additional_lambda_zero)",
        },
        {
            "setup": """import numpy as np
beta = 1.5
harmonic_lambda_zero = 6.0
additional_lambda_zero = -0.01

def run_model():
    try:
        compute_conductivity_change(
            beta,
            harmonic_lambda_zero,
            additional_lambda_zero,
        )
    except ValueError:
        return 1.0
    return 0.0

def run_gold():
    try:
        _oracle_compute_conductivity_change(
            beta,
            harmonic_lambda_zero,
            additional_lambda_zero,
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
