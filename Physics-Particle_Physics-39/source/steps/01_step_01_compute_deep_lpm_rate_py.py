"""
Evaluate the differential deep-LPM gluon splitting rate for a specified daughter momentum fraction and parent momentum.

In the high-energy limit of a gluon shower in a quark-gluon plasma, medium-induced radiation can be described using a splitting rate in the deep Landau-Pomeranchuk-Migdal regime. The rate depends on the daughter momentum fraction $z$, the parent momentum $p$, the strong coupling $\alpha_s$, the gluon color factor $C_A$, and the transverse-momentum broadening coefficient $\hat q$, and is

$$

\Gamma(z,p)=\frac{\alpha_s}{2\pi}\frac{C_A}{z(1-z)}\sqrt{\frac{C_A\hat q}{z(1-z)p}}.

$$

This rate provides the underlying kernel from which the finite-temperature splitting and merging channels are constructed.

Returns
-------
float, the differential deep-LPM gluon splitting rate for the specified momentum fraction and parent momentum
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_deep_lpm_rate(
    z: float,
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    """Evaluate the differential deep-LPM gluon splitting rate.

    Parameters
    ----------
    z : float
        Daughter momentum fraction, with 0 < z < 1.
    momentum : float
        Parent gluon momentum in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.

    Returns
    -------
    rate : float
        Differential deep-LPM splitting rate.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_deep_lpm_rate(
    z: float,
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    zbar = 1.0 - z
    prefactor = alpha_s * c_a / (2.0 * np.pi * z * zbar)
    scale = np.sqrt(c_a * qhat / (z * zbar * momentum))
    return float(prefactor * scale)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "",
            "call": "compute_deep_lpm_rate(0.338459206968295, 3.0, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_deep_lpm_rate(0.338459206968295, 3.0, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_deep_lpm_rate(0.5, 3.0, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_deep_lpm_rate(0.5, 3.0, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_deep_lpm_rate(0.2, 2.0, 0.25, 3.0, 0.02)",
            "gold_call": "_oracle_compute_deep_lpm_rate(0.2, 2.0, 0.25, 3.0, 0.02)",
        },
        {
            "setup": "",
            "call": "compute_deep_lpm_rate(0.7, 6.0, 0.2, 3.0, 0.01)",
            "gold_call": "_oracle_compute_deep_lpm_rate(0.7, 6.0, 0.2, 3.0, 0.01)",
        },
        {
            "setup": "",
            "call": "compute_deep_lpm_rate(0.37, 4.2, 0.28, 4.0, 0.017)",
            "gold_call": "_oracle_compute_deep_lpm_rate(0.37, 4.2, 0.28, 4.0, 0.017)",
        },
    ]
