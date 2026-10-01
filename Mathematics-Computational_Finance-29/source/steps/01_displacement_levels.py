"""
Given the tenor grid tau_0 = 0 < tau_1 < ... < tau_n and the at-the-money implied volatilities sigma_BS(tau_1), ..., sigma_BS(tau_n), return the displacement levels a_0, a_1, ..., a_{n-1} of the piecewise-constant displacement of equation (4) of the source, with a_0 = 0 and the remaining levels fixed by the recursion of Remark 1 (the displaced Black-Scholes calibration to the at-the-money term structure). The spot volatility of that recursion is the first at-the-money volatility.

Remark 1 of the source shows that in the displaced Black-Scholes special case the at-the-money implied variance at each tenor is a tenor-weighted average of the squared displaced volatility over the tenor intervals, so the displacement levels can be read off the at-the-money term structure interval by interval. Absence of calendar arbitrage is what makes every level real.

Returns
-------
A float array of length n holding a_0 = 0, a_1, ..., a_{n-1} in interval order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def displacement_levels(tenors, atm_vols):
    """Return the piecewise-constant displacement levels implied by the ATM term structure."""
    return np.zeros(len(atm_vols))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_displacement_levels(tenors, atm_vols):
    import numpy as np
    t = np.asarray(tenors, dtype=float)
    v = np.asarray(atm_vols, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v) + 1 or len(v) < 1:
        raise ValueError("tenors must hold tau_0 and one entry per ATM volatility")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("tenors must start at 0 and increase strictly")
    if np.any(v <= 0.0):
        raise ValueError("ATM volatilities must be positive")
    sigma0 = float(v[0])
    out = np.zeros(len(v))
    for k in range(1, len(v)):
        inc = v[k] ** 2 * t[k + 1] - v[k - 1] ** 2 * t[k]
        if inc < 0.0:
            raise ValueError("calendar arbitrage: total implied variance decreases")
        out[k] = -sigma0 + np.sqrt(inc / (t[k + 1] - t[k]))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 4.0/365.0, 5.0/365.0], atm_vols=[0.19, 0.215, 0.205, 0.23, 0.22], vov=0.6, rho=-0.65, eta0=0.4, alpha0=0.10, delta0=0.0741, s0=100.0, m_put=-2.0, m_call=2.0, n_nodes=4000, u_max=20.0)",
         "call": "displacement_levels(cfg['tenors'], cfg['atm_vols'])",
         "gold_call": "_oracle_displacement_levels(cfg['tenors'], cfg['atm_vols'])"},
        {"setup": "import numpy as np",
         "call": "displacement_levels([0.0, 2.0/365.0], [0.24])",
         "gold_call": "_oracle_displacement_levels([0.0, 2.0/365.0], [0.24])"},
        {"setup": "import numpy as np",
         "call": "displacement_levels([0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 7.0/365.0], [0.15, 0.16, 0.18, 0.17])",
         "gold_call": "_oracle_displacement_levels([0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 7.0/365.0], [0.15, 0.16, 0.18, 0.17])"},
    ]
