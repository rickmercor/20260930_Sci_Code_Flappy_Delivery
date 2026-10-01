"""
Price a European call with spot s0, strike strike and tenor tau (a tenor grid point) by the Fourier representation of equation (9) of the source, with zero interest rate and no jump component, using continuous_cf_expansion as the characteristic function. model is a dict with keys 'sigma0', 'vov', 'rho', 'eta0', 'alpha0', 'delta0', 'tenors' and 'shifts'. Evaluate each of the two integrals over the positive frequency axis with the midpoint rule on (0, u_max] with n_nodes equal cells.

Equation (9) of the source writes the call as the spot times one probability minus the strike times another, each probability an integral of the characteristic function against a phase fixed by the standardised log-moneyness. The first probability uses the characteristic function under a complex shift of its argument, with the normalisation the source prescribes.

Returns
-------
A single float, the call price.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fourier_call_price(model, s0, strike, tau, n_nodes, u_max):
    """Return the call price of equation (9) of the source under the expanded characteristic function."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fourier_call_price(model, s0, strike, tau, n_nodes, u_max):
    import numpy as np
    if s0 <= 0.0 or strike <= 0.0 or tau <= 0.0:
        raise ValueError("s0, strike and tau must be positive")
    n_nodes = int(n_nodes)
    if n_nodes < 1 or u_max <= 0.0:
        raise ValueError("n_nodes must be positive and u_max positive")
    sigma0 = float(model["sigma0"])
    args = (sigma0, model["vov"], model["rho"], model["eta0"], model["alpha0"], model["delta0"],
            model["tenors"], model["shifts"])
    du = u_max / n_nodes
    grid = (np.arange(n_nodes) + 0.5) * du
    sq = sigma0 * np.sqrt(tau)
    d2 = (np.log(s0) - np.log(strike) - 0.5 * sigma0 * sigma0 * tau) / sq
    shift = 1j * sq
    norm = _oracle_continuous_cf_expansion(-shift, tau, *args)
    psi_plain = _oracle_continuous_cf_expansion(grid, tau, *args)
    psi_shift = _oracle_continuous_cf_expansion(grid - shift, tau, *args)
    phase = np.exp(1j * grid * d2)
    p1 = 0.5 + du / np.pi * float(np.sum((phase * psi_shift / (1j * grid * norm)).real))
    p2 = 0.5 + du / np.pi * float(np.sum((phase * psi_plain / (1j * grid)).real))
    return float(s0 * p1 - strike * p2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 4.0/365.0, 5.0/365.0], atm_vols=[0.19, 0.215, 0.205, 0.23, 0.22], vov=0.6, rho=-0.65, eta0=0.4, alpha0=0.10, delta0=0.0741, s0=100.0, m_put=-2.0, m_call=2.0, n_nodes=4000, u_max=20.0)\n"
                  "sh = _oracle_displacement_levels(cfg['tenors'], cfg['atm_vols'])\n"
                  "model = dict(sigma0=cfg['atm_vols'][0], vov=cfg['vov'], rho=cfg['rho'], eta0=cfg['eta0'],"
                  " alpha0=cfg['alpha0'], delta0=cfg['delta0'], tenors=cfg['tenors'], shifts=sh)",
         "call": "fourier_call_price(model, cfg['s0'], 95.0, cfg['tenors'][-1], cfg['n_nodes'], cfg['u_max'])",
         "gold_call": "_oracle_fourier_call_price(model, cfg['s0'], 95.0, cfg['tenors'][-1], cfg['n_nodes'], cfg['u_max'])"},
        {"setup": "import numpy as np\n"
                  "model = dict(sigma0=0.2, vov=0.0, rho=0.0, eta0=0.0, alpha0=0.0, delta0=0.0, tenors=[0.0, 0.25], shifts=[0.0])",
         "call": "fourier_call_price(model, 100.0, 100.0, 0.25, 2000, 16.0)",
         "gold_call": "_oracle_fourier_call_price(model, 100.0, 100.0, 0.25, 2000, 16.0)"},
        {"setup": "import numpy as np\n"
                  "model = dict(sigma0=0.25, vov=0.9, rho=-0.8, eta0=0.6, alpha0=0.2, delta0=-0.1, tenors=[0.0, 0.004, 0.011], shifts=[0.0, 0.05])",
         "call": "fourier_call_price(model, 50.0, 52.0, 0.011, 1500, 12.0)",
         "gold_call": "_oracle_fourier_call_price(model, 50.0, 52.0, 0.011, 1500, 12.0)"},
    ]
