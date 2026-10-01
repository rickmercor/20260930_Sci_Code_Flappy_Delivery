"""
Accumulate the normalized sorting/projection gap path across polynomial basis enrichment.

Basis enrichment changes the independently evaluated stopping rule, the two unsorted martingale projections, and the sorted alpha branch. The task's final continuation diagnostic tracks these changes jointly in a four-component normalized gap space.

Returns
-------
A native Python float equal to the cumulative Euclidean path length across the requested degrees.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cumulative_sorting_basis_path(s0: "np.ndarray", strike: float, r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, train_seed: int, eval_seed: int, degrees: tuple) -> float:
    """Return the cumulative normalized sorting/projection path length.

    Parameters
    ----------
    s0 : np.ndarray
        Length-2 initial asset-price vector.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Common dividend yield.
    sigma : float
        Common annualized volatility.
    rho : float
        Instantaneous Brownian correlation.
    maturity : float
        Option maturity.
    steps : int
        Number of equally spaced exercise intervals after initiation.
    n_paths : int
        Even path count for each of the independent training and evaluation sets.
    train_seed : int
        RNG seed for policy training paths.
    eval_seed : int
        RNG seed for evaluation and dual paths.
    degrees : tuple
        Increasing sequence of polynomial degrees with at least two entries.

    Returns
    -------
    path_length : float
        Sum of Euclidean distances between consecutive four-component normalized
        gap vectors built from ``[L_u,U_alpha_u,U_beta_u,L_s,U_alpha_s]``.
    """
    return path_length

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cumulative_sorting_basis_path(s0: "np.ndarray", strike: float, r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, train_seed: int, eval_seed: int, degrees: tuple) -> float:
    train = _oracle_simulate_correlated_antithetic_gbm_paths(s0, r, q, sigma, rho, maturity, steps, n_paths, train_seed)
    eval_paths = _oracle_simulate_correlated_antithetic_gbm_paths(s0, r, q, sigma, rho, maturity, steps, n_paths, eval_seed)
    dt = float(maturity) / int(steps)
    rows = []
    for degree in degrees:
        rows.append(_oracle_sorting_aware_bound_vector(train, eval_paths, strike, r, dt, int(degree)))
    bounds = np.asarray(rows, dtype=float)
    lower_u = bounds[:, 0]
    lower_s = bounds[:, 3]
    gaps = np.column_stack((
        (bounds[:, 1] - lower_u) / lower_u,
        (bounds[:, 2] - lower_u) / lower_u,
        (bounds[:, 4] - lower_s) / lower_s,
        (lower_s - lower_u) / lower_u,
    ))
    return float(np.sum(np.linalg.norm(np.diff(gaps, axis=0), axis=1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three whole-pipeline orchestrator cases."""
    return [
        {
            "setup": "import numpy as np\ns0=np.array([100.,100.]); strike=100.; r=.06; q=0.; sigma=.30; rho=.25; maturity=.75; steps=8; n_paths=512; train_seed=17; eval_seed=41; degrees=(1,2,3,4)",
            "call": "cumulative_sorting_basis_path(s0,strike,r,q,sigma,rho,maturity,steps,n_paths,train_seed,eval_seed,degrees)",
            "gold_call": "_oracle_cumulative_sorting_basis_path(s0,strike,r,q,sigma,rho,maturity,steps,n_paths,train_seed,eval_seed,degrees)",
            "tol": 5e-10,
        },
        {
            "setup": "import numpy as np\ns0=np.array([96.,104.]); strike=100.; r=.04; q=0.; sigma=.24; rho=-.15; maturity=.6; steps=6; n_paths=256; train_seed=7; eval_seed=19; degrees=(1,2,3)",
            "call": "cumulative_sorting_basis_path(s0,strike,r,q,sigma,rho,maturity,steps,n_paths,train_seed,eval_seed,degrees)",
            "gold_call": "_oracle_cumulative_sorting_basis_path(s0,strike,r,q,sigma,rho,maturity,steps,n_paths,train_seed,eval_seed,degrees)",
            "tol": 5e-10,
        },
        {
            "setup": "import numpy as np\ns0=np.array([110.,110.]); strike=100.; r=.03; q=.01; sigma=.35; rho=.6; maturity=1.; steps=7; n_paths=384; train_seed=23; eval_seed=31; degrees=(1,2,3,4)",
            "call": "cumulative_sorting_basis_path(s0,strike,r,q,sigma,rho,maturity,steps,n_paths,train_seed,eval_seed,degrees)",
            "gold_call": "_oracle_cumulative_sorting_basis_path(s0,strike,r,q,sigma,rho,maturity,steps,n_paths,train_seed,eval_seed,degrees)",
            "tol": 5e-10,
        },
    ]
