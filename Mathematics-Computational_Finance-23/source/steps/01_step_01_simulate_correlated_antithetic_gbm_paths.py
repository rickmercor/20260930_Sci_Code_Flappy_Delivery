"""
Generate deterministic two-asset risk-neutral GBM paths with correlation and antithetic pairing.

The paper's numerical experiments use risk-neutral geometric Brownian motion together with pseudo-random paths and corresponding antithetic variates. This supporting step fixes the exact training/evaluation path sets used by the primal and dual calculations.

Returns
-------
A NumPy array of shape `(steps + 1, n_paths, 2)` containing the correlated antithetic GBM paths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_correlated_antithetic_gbm_paths(s0: "np.ndarray", r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, seed: int) -> "np.ndarray":
    """Return correlated two-asset GBM paths with antithetic shocks.

    Parameters
    ----------
    s0 : np.ndarray
        Length-2 initial asset-price vector.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Common continuously compounded dividend yield.
    sigma : float
        Common annualized volatility.
    rho : float
        Instantaneous Brownian correlation.
    maturity : float
        Option maturity.
    steps : int
        Number of equal time intervals.
    n_paths : int
        Even total number of paths. The first half uses the generated correlated
        shocks and the second half uses their sign reversals.
    seed : int
        Seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    paths : np.ndarray
        Array of shape ``(steps + 1, n_paths, 2)``.
    """
    return paths

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_simulate_correlated_antithetic_gbm_paths(s0: "np.ndarray", r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, seed: int) -> "np.ndarray":
    s0_arr = np.asarray(s0, dtype=float)
    half = int(n_paths) // 2
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((int(steps), half, 2))
    corr = np.empty_like(raw)
    corr[..., 0] = raw[..., 0]
    corr[..., 1] = float(rho) * raw[..., 0] + np.sqrt(1.0 - float(rho) ** 2) * raw[..., 1]
    shocks = np.concatenate((corr, -corr), axis=1)
    dt = float(maturity) / int(steps)
    drift = (float(r) - float(q) - 0.5 * float(sigma) ** 2) * dt
    vol = float(sigma) * np.sqrt(dt)
    paths = np.empty((int(steps) + 1, int(n_paths), 2), dtype=float)
    paths[0] = s0_arr
    for t in range(int(steps)):
        paths[t + 1] = paths[t] * np.exp(drift + vol * shocks[t])
    return paths

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three deterministic path-generation cases."""
    return [
        {
            "setup": "import numpy as np\ns0=np.array([100.,100.]); r=.06; q=0.; sigma=.30; rho=.25; maturity=.75; steps=3; n_paths=16; seed=17",
            "call": "simulate_correlated_antithetic_gbm_paths(s0,r,q,sigma,rho,maturity,steps,n_paths,seed)",
            "gold_call": "_oracle_simulate_correlated_antithetic_gbm_paths(s0,r,q,sigma,rho,maturity,steps,n_paths,seed)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\ns0=np.array([96.,104.]); r=.04; q=.01; sigma=.22; rho=-.35; maturity=.6; steps=4; n_paths=24; seed=7",
            "call": "simulate_correlated_antithetic_gbm_paths(s0,r,q,sigma,rho,maturity,steps,n_paths,seed)",
            "gold_call": "_oracle_simulate_correlated_antithetic_gbm_paths(s0,r,q,sigma,rho,maturity,steps,n_paths,seed)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\ns0=np.array([110.,90.]); r=.02; q=0.; sigma=.18; rho=0.; maturity=.4; steps=2; n_paths=10; seed=31",
            "call": "simulate_correlated_antithetic_gbm_paths(s0,r,q,sigma,rho,maturity,steps,n_paths,seed)",
            "gold_call": "_oracle_simulate_correlated_antithetic_gbm_paths(s0,r,q,sigma,rho,maturity,steps,n_paths,seed)",
            "tol": 1e-13,
        },
    ]
