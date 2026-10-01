"""
Extend the unsorted bound vector with the source-approved sorted primal and alpha bounds.

The source's min-put sorting treatment changes the state representation used by the primal and alpha regressions while intentionally retaining beta on the unsorted process. The resulting five prices capture both projection-order and sorting effects at one basis degree.

Returns
-------
A length-5 NumPy vector `[L_u, U_alpha_u, U_beta_u, L_s, U_alpha_s]`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sorting_aware_bound_vector(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    """Return unsorted bounds together with the source-approved sorted branch.

    Parameters
    ----------
    train_paths : np.ndarray
        Independent policy-training paths.
    eval_paths : np.ndarray
        Evaluation paths used for policy evaluation and dual fitting.
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
        Length-5 vector ``[L_u, U_alpha_u, U_beta_u, L_s, U_alpha_s]``.
    """
    return bounds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sorting_aware_bound_vector(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    unsorted = _oracle_unsorted_primal_dual_bounds(train_paths, eval_paths, strike, r, dt, degree)
    theta_sorted = _oracle_fit_min_put_primal_policy(train_paths, strike, r, dt, degree, True)
    H_sorted = _oracle_evaluate_min_put_payoff_process(eval_paths, theta_sorted, strike, r, dt, degree, True)
    M_alpha_sorted = _oracle_single_projection_alpha_martingale(eval_paths, H_sorted, strike, r, dt, degree, True)
    lower_sorted = float(np.mean(H_sorted[0]))
    upper_alpha_sorted = _dual_upper_price(eval_paths, M_alpha_sorted, strike, r, dt)
    return np.array([unsorted[0], unsorted[1], unsorted[2], lower_sorted, upper_alpha_sorted], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three sorting-aware bound cases."""
    return [
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,2); eval_paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,3); strike=100.; r=.05; dt=.6/5; degree=1",
            "call": "sorting_aware_bound_vector(train,eval_paths,strike,r,dt,degree)",
            "gold_call": "_oracle_sorting_aware_bound_vector(train,eval_paths,strike,r,dt,degree)",
            "tol": 2e-8,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,7); eval_paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,8); strike=100.; r=.04; dt=.75/6; degree=2",
            "call": "sorting_aware_bound_vector(train,eval_paths,strike,r,dt,degree)",
            "gold_call": "_oracle_sorting_aware_bound_vector(train,eval_paths,strike,r,dt,degree)",
            "tol": 2e-7,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,11); eval_paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,12); strike=102.; r=.03; dt=.8/7; degree=4",
            "call": "sorting_aware_bound_vector(train,eval_paths,strike,r,dt,degree)",
            "gold_call": "_oracle_sorting_aware_bound_vector(train,eval_paths,strike,r,dt,degree)",
            "tol": 5e-7,
        },
    ]
