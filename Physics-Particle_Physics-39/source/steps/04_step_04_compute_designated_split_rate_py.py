"""
Calculate the integrated rate of a specified discrete splitting subchannel in the quadrature representation of the shower.

Each quadrature node represents a discrete shower subchannel when the continuous momentum-fraction integral is replaced by a weighted sum. For a designated splitting node $z_\star$ with quadrature weight $w_\star$, the rate for that specific $1\to2$ event is

$$

r_\star=w_\star\Gamma_1(z_\star,p).

$$

Only the finite-temperature splitting rate contributes to this designated event, while all splitting and merging channels remain part of the competing Sudakov hazard.

Returns
-------
float, the integrated rate of the designated discrete splitting subchannel in GeV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_designated_split_rate(
    z_star: float,
    weight_star: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    """Evaluate the integrated rate of one designated splitting subchannel.

    Parameters
    ----------
    z_star : float
        Momentum fraction of the designated splitting node.
    weight_star : float
        Quadrature weight of the designated node.
    momentum : float
        Parent gluon momentum in GeV.
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
    rate : float
        Integrated rate of the designated splitting subchannel in GeV.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_designated_split_rate(
    z_star: float,
    weight_star: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    rates = _oracle_compute_thermal_channel_rates(
        z_star,
        momentum,
        temperature,
        alpha_s,
        c_a,
        qhat,
    )
    return float(weight_star * rates[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "",
            "call": "compute_designated_split_rate(0.338459206968295, 0.143588601149810, 3.0, 0.5, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_designated_split_rate(0.338459206968295, 0.143588601149810, 3.0, 0.5, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_designated_split_rate(0.5, 0.170666666666667, 3.0, 0.5, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_designated_split_rate(0.5, 0.170666666666667, 3.0, 0.5, 0.3, 3.0, 0.015)",
        },
        {
            "setup": "",
            "call": "compute_designated_split_rate(0.3, 0.12, 1.2, 0.5, 0.3, 3.0, 0.015)",
            "gold_call": "_oracle_compute_designated_split_rate(0.3, 0.12, 1.2, 0.5, 0.3, 3.0, 0.015)",
        },
    ]
