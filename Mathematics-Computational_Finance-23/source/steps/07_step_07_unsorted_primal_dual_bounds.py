"""
Compute the unsorted primal lower estimate and both paper-specific dual upper estimates for one basis degree.

This step combines independent policy fitting/evaluation, alpha and beta martingales, and the pathwise dual maximum to expose the common three-price primal-dual bracket on the unsorted state.

Returns
-------
A length-3 NumPy vector `[lower, alpha_upper, beta_upper]`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def unsorted_primal_dual_bounds(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    """Return unsorted primal, alpha-dual, and beta-dual prices.

    Parameters
    ----------
    train_paths : np.ndarray
        Independent policy-training paths.
    eval_paths : np.ndarray
        Evaluation paths used for the fixed policy and both dual projections.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.

    Returns
    -------
    bounds : np.ndarray
        Length-3 vector ``[lower, alpha_upper, beta_upper]``.
    """
    return bounds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _dual_upper_price(paths: "np.ndarray", martingale: "np.ndarray", strike: float, r: float, dt: float) -> float:
    x = np.asarray(paths, dtype=float)
    M = np.asarray(martingale, dtype=float)
    steps = x.shape[0] - 1
    bank = np.exp(float(r) * float(dt) * np.arange(steps + 1, dtype=float))
    discounted_immediate = np.stack([_min_put_payoff(x[t], strike) / bank[t] for t in range(steps + 1)])
    return float(np.mean(np.max(discounted_immediate[1:] - M[1:], axis=0)))


def _oracle_unsorted_primal_dual_bounds(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    theta = _oracle_fit_min_put_primal_policy(train_paths, strike, r, dt, degree, False)
    H = _oracle_evaluate_min_put_payoff_process(eval_paths, theta, strike, r, dt, degree, False)
    M_alpha = _oracle_single_projection_alpha_martingale(eval_paths, H, strike, r, dt, degree, False)
    M_beta = _oracle_double_projection_beta_martingale(eval_paths, H, strike, r, dt, degree)
    lower = float(np.mean(H[0]))
    upper_alpha = _dual_upper_price(eval_paths, M_alpha, strike, r, dt)
    upper_beta = _dual_upper_price(eval_paths, M_beta, strike, r, dt)
    return np.array([lower, upper_alpha, upper_beta], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three complete unsorted bound cases."""
    return [
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,2); eval_paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,3); strike=100.; r=.05; dt=.6/5; degree=1",
            "call": "unsorted_primal_dual_bounds(train,eval_paths,strike,r,dt,degree)",
            "gold_call": "_oracle_unsorted_primal_dual_bounds(train,eval_paths,strike,r,dt,degree)",
            "tol": 2e-8,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,7); eval_paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,8); strike=100.; r=.04; dt=.75/6; degree=2",
            "call": "unsorted_primal_dual_bounds(train,eval_paths,strike,r,dt,degree)",
            "gold_call": "_oracle_unsorted_primal_dual_bounds(train,eval_paths,strike,r,dt,degree)",
            "tol": 2e-7,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,11); eval_paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,12); strike=102.; r=.03; dt=.8/7; degree=4",
            "call": "unsorted_primal_dual_bounds(train,eval_paths,strike,r,dt,degree)",
            "gold_call": "_oracle_unsorted_primal_dual_bounds(train,eval_paths,strike,r,dt,degree)",
            "tol": 5e-7,
        },
    ]
