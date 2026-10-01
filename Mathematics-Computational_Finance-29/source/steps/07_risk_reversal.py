"""
Integrate the pipeline and report one number. config holds 'tenors' (tau_0 = 0 first), 'atm_vols', the model coefficients 'vov', 'rho', 'eta0', 'alpha0', 'delta0', the spot 's0', the two log-moneyness levels 'm_put' and 'm_call', and the quadrature settings 'n_nodes' and 'u_max'. Calibrate the displacement levels from the at-the-money term structure with displacement_levels, take the spot volatility as the first at-the-money volatility, check the assembled expansion at zero frequency, set the two strikes at the last tenor by the log-moneyness convention of equation (12) of the source with that tenor's at-the-money volatility, price both calls with fourier_call_price, invert them with implied_volatility, and return the implied volatility at m_put minus the implied volatility at m_call.

The reported number is the put-minus-call implied-volatility difference of the last tenor, the quantity the source's expansion is designed to capture across few-day tenors once the displacement has absorbed the at-the-money term structure.

Returns
-------
A single float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def risk_reversal(config):
    """Return the put-minus-call implied volatility difference at the last tenor."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_risk_reversal(config):
    import numpy as np
    tenors = [float(x) for x in config["tenors"]]
    atm = [float(x) for x in config["atm_vols"]]
    s0 = float(config["s0"])
    n_nodes, u_max = int(config["n_nodes"]), float(config["u_max"])
    if s0 <= 0.0 or n_nodes < 1 or u_max <= 0.0:
        raise ValueError("s0 must be positive, n_nodes at least 1 and u_max positive")
    shifts = _oracle_displacement_levels(tenors, atm)
    sigma0 = atm[0]
    tau = tenors[-1]
    # consistency gate: at zero frequency every correction block vanishes and the expansion
    # equals one, so the assembled pieces are checked before any price is formed
    levels = 1.0 + np.asarray(shifts, dtype=float) / sigma0
    F = _oracle_displacement_functionals(tenors, levels, tau)
    if F[0] <= 0.0:
        raise ValueError("the squared displacement profile must have positive integral")
    blocks = _oracle_expansion_blocks(0.0, tau, sigma0, config["vov"], config["rho"], config["eta0"],
                                      config["alpha0"], config["delta0"], F)
    psi0 = _oracle_continuous_cf_expansion(0.0, tau, sigma0, config["vov"], config["rho"], config["eta0"],
                                           config["alpha0"], config["delta0"], tenors, shifts)
    if np.max(np.abs(blocks)) != 0.0 or abs(psi0 - 1.0) > 1e-12:
        raise ValueError("expansion does not equal one at zero frequency")
    model = dict(sigma0=sigma0, vov=float(config["vov"]), rho=float(config["rho"]),
                 eta0=float(config["eta0"]), alpha0=float(config["alpha0"]),
                 delta0=float(config["delta0"]), tenors=tenors, shifts=shifts)
    ivs = []
    for m in (float(config["m_put"]), float(config["m_call"])):
        strike = s0 * np.exp(m * atm[-1] * np.sqrt(tau))
        price = _oracle_fourier_call_price(model, s0, strike, tau, n_nodes, u_max)
        ivs.append(_oracle_implied_volatility(price, s0, strike, tau))
    return float(ivs[0] - ivs[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 4.0/365.0, 5.0/365.0], atm_vols=[0.19, 0.215, 0.205, 0.23, 0.22], vov=0.6, rho=-0.65, eta0=0.4, alpha0=0.10, delta0=0.0741, s0=100.0, m_put=-2.0, m_call=2.0, n_nodes=4000, u_max=20.0)",
         "call": "risk_reversal(cfg)",
         "gold_call": "_oracle_risk_reversal(cfg)"},
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0],"
                  " atm_vols=[0.25, 0.24, 0.26], vov=0.4, rho=-0.5, eta0=0.2, alpha0=0.0, delta0=0.05,"
                  " s0=100.0, m_put=-1.5, m_call=1.5, n_nodes=3000, u_max=16.0)",
         "call": "risk_reversal(cfg)",
         "gold_call": "_oracle_risk_reversal(cfg)"},
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 5.0/365.0],"
                  " atm_vols=[0.18, 0.185, 0.19, 0.188], vov=0.4, rho=-0.7, eta0=0.15, alpha0=0.05, delta0=0.0504,"
                  " s0=2500.0, m_put=-1.0, m_call=1.0, n_nodes=4000, u_max=20.0)",
         "call": "risk_reversal(cfg)",
         "gold_call": "_oracle_risk_reversal(cfg)"},
    ]
