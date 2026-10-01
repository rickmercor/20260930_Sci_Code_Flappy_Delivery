"""
At Fourier frequency u and tenor tau, return the six correction blocks that appear inside the large parentheses of the expansion of Theorem 1 of the source, in their order of appearance, each as a complex number: the block driven by the price-side volatility-of-volatility, the block driven by delta_0, the block driven by alpha_0, the block driven by eta_0, the block quadratic in the price-side volatility-of-volatility, and the block quadratic in the orthogonal volatility-of-volatility. The price-side and orthogonal volatility-of-volatility coefficients follow from the spot volatility-of-volatility vov and the spot leverage rho as in equation (2) of the source. The displacement enters only through functionals, the array returned by displacement_functionals. u is a complex scalar or a one-dimensional array of frequencies; in the array case the blocks are evaluated elementwise.

Theorem 1 of the source expands the characteristic function of the standardised, demeaned continuous log-return to second order in the square root of the tenor. The leading factor is Gaussian; the six blocks carry the leverage, drift, volatility-of-volatility and volatility-of-volatility-of-volatility corrections, each with its own coefficient and its own displacement functional.

Returns
-------
A complex array of shape (6,) for scalar u, or (6, len(u)) for a one-dimensional u, holding the blocks in their order of appearance in Theorem 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def expansion_blocks(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, functionals):
    """Return the six correction blocks of the source's Theorem 1 at the frequency or frequencies u."""
    return np.zeros(6, dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_expansion_blocks(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, functionals):
    import numpy as np
    F = np.asarray(functionals, dtype=float)
    if F.shape != (8,):
        raise ValueError("functionals must be the length-8 array of displacement_functionals")
    if tau <= 0.0 or sigma0 <= 0.0 or vov < 0.0 or not (-1.0 <= rho <= 1.0):
        raise ValueError("invalid model coefficients")
    A0, A1, A2, A3, A4, B2, B3a, B3b = [float(x) for x in F]
    u = np.asarray(u, dtype=complex)
    if u.ndim > 1:
        raise ValueError("u must be a scalar or a one-dimensional array of frequencies")
    u2 = u * u
    beta0 = vov * rho
    betap0 = vov * np.sqrt(1.0 - rho * rho)
    skew = -1j * u ** 3 * beta0 / (sigma0 * tau ** 1.5) * A1
    dblk = -u2 * delta0 / (sigma0 * tau) * A2
    ablk = -u2 * alpha0 / (sigma0 * tau) * A3
    eblk = u ** 4 * eta0 / (sigma0 * tau ** 2) * A4
    bblk = -beta0 ** 2 * u2 / (8.0 * sigma0 ** 2 * tau) * (
        2.0 * tau ** 2 + 4.0 * u2 * A1 - 24.0 * u2 / tau * B2
        - 12.0 * u2 / tau * (B3a - 2.0 * u2 / tau * B3b))
    pblk = -betap0 ** 2 * u2 / (2.0 * sigma0 ** 2 * tau) * (tau ** 2 / 2.0 - 2.0 * u2 / tau * B3a)
    return np.array([skew, dblk, ablk, eblk, bblk, pblk], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 4.0/365.0, 5.0/365.0], atm_vols=[0.19, 0.215, 0.205, 0.23, 0.22], vov=0.6, rho=-0.65, eta0=0.4, alpha0=0.10, delta0=0.0741, s0=100.0, m_put=-2.0, m_call=2.0, n_nodes=4000, u_max=20.0)\n"
                  "sh = _oracle_displacement_levels(cfg['tenors'], cfg['atm_vols'])\n"
                  "s0v = cfg['atm_vols'][0]\n"
                  "F = _oracle_displacement_functionals(cfg['tenors'], 1.0 + sh / s0v, cfg['tenors'][-1])",
         "call": "expansion_blocks(1.5, cfg['tenors'][-1], s0v, cfg['vov'], cfg['rho'], cfg['eta0'], cfg['alpha0'], cfg['delta0'], F)",
         "gold_call": "_oracle_expansion_blocks(1.5, cfg['tenors'][-1], s0v, cfg['vov'], cfg['rho'], cfg['eta0'], cfg['alpha0'], cfg['delta0'], F)"},
        {"setup": "import numpy as np\nF = _oracle_displacement_functionals([0.0, 0.01], [1.0], 0.01)",
         "call": "expansion_blocks(0.0, 0.01, 0.2, 0.5, -0.5, 0.3, 0.0, 0.05, F)",
         "gold_call": "_oracle_expansion_blocks(0.0, 0.01, 0.2, 0.5, -0.5, 0.3, 0.0, 0.05, F)"},
        {"setup": "import numpy as np\nF = _oracle_displacement_functionals([0.0, 0.004, 0.011], [1.0, 1.3], 0.011)",
         "call": "expansion_blocks(np.array([0.5, 2.5 - 0.02j, 4.0]), 0.011, 0.25, 0.9, -0.8, 0.6, 0.2, -0.1, F)",
         "gold_call": "_oracle_expansion_blocks(np.array([0.5, 2.5 - 0.02j, 4.0]), 0.011, 0.25, 0.9, -0.8, 0.6, 0.2, -0.1, F)"},
    ]
