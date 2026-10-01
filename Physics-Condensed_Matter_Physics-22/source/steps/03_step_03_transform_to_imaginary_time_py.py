"""
Transform the singular and regular harmonic spectral contributions into their corresponding imaginary-time heat-current correlation components.

The real-frequency heat-current spectrum is related to the imaginary-time harmonic current correlation by

$$

C_{xx}^{h}(\tau)=\frac{1}{\pi}\int_0^\infty\Lambda_x(\omega)\left[e^{-\tau\omega}+e^{-(\beta-\tau)\omega}\right]d\omega.

$$

Because the reference spectrum is linear in the singular and regular contributions, the two imaginary-time components can be evaluated separately. The regular contribution remains unscaled in this step so that its scale factor $ξ$ can be determined by the later reconstruction fit.

Returns
-------
np.ndarray of shape (2, M): row 0 contains the singular imaginary-time correlation and row 1 contains the unscaled regular imaginary-time correlation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transform_to_imaginary_time(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    tau: "np.ndarray",
) -> "np.ndarray":
    """Transform the spectral components to imaginary-time correlations.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    gamma_tr : float
        Positive mode-independent transport damping rate.
    tau : np.ndarray
        One-dimensional array of imaginary times satisfying
        0 < tau_i < beta.

    Returns
    -------
    correlation : np.ndarray
        Array of shape (2, M). Row 0 contains the singular imaginary-time
        contribution and row 1 contains the unscaled regular contribution.
    """
    return correlation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import exp1


def _lorentz_exp_integral(
    decay: float,
    center: "np.ndarray",
    width: "np.ndarray",
) -> "np.ndarray":
    z_plus = center + 1j * width
    z_minus = center - 1j * width

    value = (
        np.exp(-decay * z_plus) * exp1(-decay * z_plus)
        - np.exp(-decay * z_minus) * exp1(-decay * z_minus)
    ) / (2j)

    return np.real(value)


def _oracle_transform_to_imaginary_time(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    tau: "np.ndarray",
) -> "np.ndarray":
    tau = np.asarray(tau, dtype=float)

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

    values = []

    for tau_i in tau:
        singular_integral = (
            _lorentz_exp_integral(
                float(tau_i),
                singular_center,
                pair_width,
            )
            + _lorentz_exp_integral(
                float(beta) - float(tau_i),
                singular_center,
                pair_width,
            )
        )

        regular_integral = (
            _lorentz_exp_integral(
                float(tau_i),
                regular_center,
                pair_width,
            )
            + _lorentz_exp_integral(
                float(beta) - float(tau_i),
                regular_center,
                pair_width,
            )
        )

        c_s = float(
            np.sum(singular_coeff * singular_integral) / np.pi
        )

        c_r = float(
            np.sum(regular_coeff * regular_integral) / np.pi
        )

        values.append((c_s, c_r))

    return np.asarray(values, dtype=float).T

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

tau = np.array(
    [0.060, 0.150, 0.270, 0.420, 0.585, 0.750],
    dtype=float,
)
tau_gold = tau.copy()
""",
            "call": "np.round(transform_to_imaginary_time(omega, nu, beta, gamma_tr, tau), 8)",
            "gold_call": "np.round(_oracle_transform_to_imaginary_time(omega_gold, nu_gold, beta, gamma_tr, tau_gold), 8)",
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

tau = np.array([0.60], dtype=float)
tau_gold = tau.copy()
""",
            "call": "np.round(transform_to_imaginary_time(omega, nu, beta, gamma_tr, tau), 8)",
            "gold_call": "np.round(_oracle_transform_to_imaginary_time(omega_gold, nu_gold, beta, gamma_tr, tau_gold), 8)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([0.75, 1.20], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.0, 0.3],
    [0.3, 0.8]
], dtype=float)
nu_gold = nu.copy()

beta = 1.5
gamma_tr = 0.09

tau = np.array([1.0e-6, 1.499999], dtype=float)
tau_gold = tau.copy()
""",
            "call": "np.round(transform_to_imaginary_time(omega, nu, beta, gamma_tr, tau), 8)",
            "gold_call": "np.round(_oracle_transform_to_imaginary_time(omega_gold, nu_gold, beta, gamma_tr, tau_gold), 8)",
        },
    ]
