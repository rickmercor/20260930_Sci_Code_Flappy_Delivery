"""
Evaluate the singular, regular, and total fitted heat-current spectral density at zero frequency and obtain the transport lifetime associated with the fitted damping rate.

The zero-frequency limit of the reconstructed heat-current spectrum determines the Green--Kubo conductivity.

The transport lifetime corresponding to the fitted damping rate is

$$

\tau^{tr}=\frac{1}{2\Gamma^{tr}}.

$$

At zero frequency, the fitted reference spectrum is

$$

\Lambda_x(0)=\Lambda_x^s(0)+ξ\Lambda_x^r(0).

$$

The regular contribution returned separately by this step is the unscaled $\Lambda_x^r(0)$ so that the role of the fitted factor $ξ$ remains explicit.

Returns
-------
np.ndarray of length 4 containing transport lifetime, singular zero-frequency spectrum, unscaled regular zero-frequency spectrum, and fitted total zero-frequency spectrum
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_zero_frequency_spectrum(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    xi: float,
) -> "np.ndarray":
    """Evaluate the fitted heat-current spectrum at zero frequency.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    gamma_tr : float
        Positive fitted transport damping rate.
    xi : float
        Scale applied to the regular spectral contribution.

    Returns
    -------
    result : np.ndarray
        Length-4 array containing the transport lifetime,
        singular zero-frequency spectrum, unscaled regular
        zero-frequency spectrum, and fitted total zero-frequency spectrum.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_zero_frequency_spectrum(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    xi: float,
) -> "np.ndarray":
    components = _oracle_construct_spectral_components(
        omega,
        nu,
        beta,
        gamma_tr,
    )

    singular_coeff = components[0]
    singular_center = components[1]
    regular_coeff = components[2]
    regular_center = components[3]
    pair_width = components[4]

    lambda_s = float(
        np.sum(
            singular_coeff
            * pair_width
            / (
                singular_center**2
                + pair_width**2
            )
        )
    )

    lambda_r = float(
        np.sum(
            regular_coeff
            * pair_width
            / (
                regular_center**2
                + pair_width**2
            )
        )
    )

    lambda_total = (
        lambda_s
        + float(xi) * lambda_r
    )

    tau_tr = (
        1.0
        / (2.0 * float(gamma_tr))
    )

    return np.array(
        [
            tau_tr,
            lambda_s,
            lambda_r,
            lambda_total,
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
xi = 1.40975379461
""",
            "call": "np.round(evaluate_zero_frequency_spectrum(omega, nu, beta, gamma_tr, xi), 9)",
            "gold_call": "np.round(_oracle_evaluate_zero_frequency_spectrum(omega_gold, nu_gold, beta, gamma_tr, xi), 9)",
        },
        {
            "setup": """
import numpy as np

omega = np.array([0.8, 1.4], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.0, 0.3],
    [0.3, 0.8]
], dtype=float)
nu_gold = nu.copy()

beta = 1.2
gamma_tr = 0.06
xi = 0.5
""",
            "call": "np.round(evaluate_zero_frequency_spectrum(omega, nu, beta, gamma_tr, xi), 9)",
            "gold_call": "np.round(_oracle_evaluate_zero_frequency_spectrum(omega_gold, nu_gold, beta, gamma_tr, xi), 9)",
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
xi = 2.3
""",
            "call": "np.round(evaluate_zero_frequency_spectrum(omega, nu, beta, gamma_tr, xi), 9)",
            "gold_call": "np.round(_oracle_evaluate_zero_frequency_spectrum(omega_gold, nu_gold, beta, gamma_tr, xi), 9)",
        },
    ]
