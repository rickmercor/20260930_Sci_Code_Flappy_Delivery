"""
Evaluate the finite-temperature splitting and inverse-merging rates for a gluon at a specified momentum fraction.

Finite temperature modifies the underlying deep-LPM kernel through Bose-Einstein occupation factors. For a gluon mode of energy $E$, the occupation is $n(E)=1/(\exp(E/T)-1)$. The Bose-enhanced splitting channel is

$$

\Gamma_1(z,p)=\frac{1}{2}\Gamma(z,p)\left[1+n(zp)+n((1-z)p)\right],

$$

while the inverse merging channel is

$$

\Gamma_2(z,p)=\frac{1}{(1-z)^3}\Gamma\left(z,\frac{p}{1-z}\right)\left[n\left(\frac{zp}{1-z}\right)-n\left(\frac{p}{1-z}\right)\right].

$$

These two rates form the competing finite-temperature $1\leftrightarrow2$ shower channels.

Returns
-------
np.ndarray of shape (2,), containing the finite-temperature splitting rate Gamma_1 and merging rate Gamma_2 in GeV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_thermal_channel_rates(
    z: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> "np.ndarray":
    """Evaluate the finite-temperature splitting and merging rates.

    Parameters
    ----------
    z : float
        Daughter momentum fraction, with 0 < z < 1.
    momentum : float
        Gluon momentum in GeV.
    temperature : float
        Medium temperature in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.

    Returns
    -------
    rates : np.ndarray
        Array of shape (2,) containing [Gamma_1, Gamma_2] in GeV.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bose_occupation(energy: float, temperature: float) -> float:
    return float(1.0 / np.expm1(energy / temperature))


def _oracle_compute_thermal_channel_rates(
    z: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> "np.ndarray":
    zbar = 1.0 - z

    base_rate = _oracle_compute_deep_lpm_rate(
        z,
        momentum,
        alpha_s,
        c_a,
        qhat,
    )

    n_first = _bose_occupation(z * momentum, temperature)
    n_second = _bose_occupation(zbar * momentum, temperature)

    gamma_1 = 0.5 * base_rate * (1.0 + n_first + n_second)

    transformed_momentum = momentum / zbar
    transformed_rate = _oracle_compute_deep_lpm_rate(
        z,
        transformed_momentum,
        alpha_s,
        c_a,
        qhat,
    )

    n_incoming = _bose_occupation(z * momentum / zbar, temperature)
    n_parent = _bose_occupation(momentum / zbar, temperature)

    gamma_2 = transformed_rate * (n_incoming - n_parent) / zbar**3

    return np.array([gamma_1, gamma_2], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "",
            "call": "compute_thermal_channel_rates(0.338459206968295, 3.0, 0.5, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_thermal_channel_rates(0.338459206968295, 3.0, 0.5, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_thermal_channel_rates(0.5, 3.0, 0.5, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_thermal_channel_rates(0.5, 3.0, 0.5, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_thermal_channel_rates(0.3, 1.2, 0.5, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_thermal_channel_rates(0.3, 1.2, 0.5, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_thermal_channel_rates(0.4, 8.0, 0.25, 0.25, 3.0, 0.01)",
            "gold_call": "_oracle_compute_thermal_channel_rates(0.4, 8.0, 0.25, 0.25, 3.0, 0.01)",
        },
        {
            "setup": "",
            "call": "compute_thermal_channel_rates(0.45, 1.0, 0.8, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_thermal_channel_rates(0.45, 1.0, 0.8, 0.3, 3.0, 0.015)",
        },
    ]
