"""
Assemble the second-order expansion of Theorem 1 of the source for the characteristic function of the standardised, demeaned continuous log-return at tenor tau (a tenor grid point) and complex frequency u: normalise the displacement levels shifts into the profile of Theorem 1, evaluate the displacement functionals and the six correction blocks, and combine them with the Gaussian leading factor exactly as the theorem states. u is a complex scalar or a one-dimensional array of frequencies, evaluated elementwise.

The standardisation of equation (3) of the source demeans the log-return with the time-zero drift and scales it by the spot volatility times the square root of the tenor, so the leading factor of the expansion is the Gaussian characteristic function with the tenor-averaged squared displacement profile as its variance.

Returns
-------
A single complex number for scalar u, or a one-dimensional complex array for a one-dimensional u.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def continuous_cf_expansion(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, tenors, shifts):
    """Return the Theorem 1 expansion of the standardised continuous log-return's characteristic function at u."""
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_continuous_cf_expansion(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, tenors, shifts):
    import numpy as np
    sh = np.asarray(shifts, dtype=float)
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")
    levels = 1.0 + sh / sigma0
    F = _oracle_displacement_functionals(tenors, levels, tau)
    blocks = _oracle_expansion_blocks(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, F)
    u = np.asarray(u, dtype=complex)
    lead = np.exp(-u * u / 2.0 * F[0] / tau)
    out = lead * (1.0 + np.sum(blocks, axis=0))
    return complex(out) if out.ndim == 0 else np.asarray(out, dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 4.0/365.0, 5.0/365.0], atm_vols=[0.19, 0.215, 0.205, 0.23, 0.22], vov=0.6, rho=-0.65, eta0=0.4, alpha0=0.10, delta0=0.0741, s0=100.0, m_put=-2.0, m_call=2.0, n_nodes=4000, u_max=20.0)\n"
                  "sh = _oracle_displacement_levels(cfg['tenors'], cfg['atm_vols'])\ns0v = cfg['atm_vols'][0]",
         "call": "continuous_cf_expansion(1.0, cfg['tenors'][-1], s0v, cfg['vov'], cfg['rho'], cfg['eta0'], cfg['alpha0'], cfg['delta0'], cfg['tenors'], sh)",
         "gold_call": "_oracle_continuous_cf_expansion(1.0, cfg['tenors'][-1], s0v, cfg['vov'], cfg['rho'], cfg['eta0'], cfg['alpha0'], cfg['delta0'], cfg['tenors'], sh)"},
        {"setup": "import numpy as np",
         "call": "continuous_cf_expansion(0.0, 0.01, 0.2, 0.5, -0.5, 0.3, 0.0, 0.05, [0.0, 0.01], [0.0])",
         "gold_call": "_oracle_continuous_cf_expansion(0.0, 0.01, 0.2, 0.5, -0.5, 0.3, 0.0, 0.05, [0.0, 0.01], [0.0])"},
        {"setup": "import numpy as np",
         "call": "continuous_cf_expansion(np.array([0.5, 2.0 - 0.02j, 4.0]), 0.011, 0.25, 0.9, -0.8, 0.6, 0.2, -0.1, [0.0, 0.004, 0.011], [0.0, 0.05])",
         "gold_call": "_oracle_continuous_cf_expansion(np.array([0.5, 2.0 - 0.02j, 4.0]), 0.011, 0.25, 0.9, -0.8, 0.6, 0.2, -0.1, [0.0, 0.004, 0.011], [0.0, 0.05])"},
    ]
