"""
Construct the paper-specific single-projection alpha martingale on unsorted or sorted states.

Alpha projects the same one-period-discounted realized payoff on the common basis evaluated at the end and beginning of each timestep. Their fitted-value difference is accumulated with the paper's bank-account scaling to approximate the Doob martingale.

Returns
-------
A NumPy martingale array of shape `(steps + 1, n_paths)` with zero initial row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def single_projection_alpha_martingale(paths: "np.ndarray", H_D: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Return the source's alpha-algorithm martingale process.

    Parameters
    ----------
    paths : np.ndarray
        Evaluation paths of shape ``(steps + 1, n_paths, 2)``.
    H_D : np.ndarray
        Realized-payoff process from the same evaluation paths.
    strike : float
        Min-put strike used by the common basis.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.
    sort_state : bool
        Whether both adjacent-time basis evaluations use the pathwise-sorted state.

    Returns
    -------
    M_alpha : np.ndarray
        Martingale array of shape ``(steps + 1, n_paths)`` with a zero initial row.
    """
    return M_alpha

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_single_projection_alpha_martingale(paths: "np.ndarray", H_D: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    H = np.asarray(H_D, dtype=float)
    steps = x.shape[0] - 1
    n_paths = x.shape[1]
    disc = np.exp(-float(r) * float(dt))
    bank = np.exp(float(r) * float(dt) * np.arange(steps + 1, dtype=float))
    M = np.zeros((steps + 1, n_paths), dtype=float)
    for t in range(steps):
        regressand = disc * H[t + 1]
        Phi_next = _oracle_min_put_polynomial_basis(x[t + 1], strike, degree, sort_state)
        gamma = _ols(Phi_next, regressand)
        value_next = Phi_next @ gamma
        if t == 0:
            continuation = np.full(n_paths, np.mean(regressand), dtype=float)
        else:
            Phi_now = _oracle_min_put_polynomial_basis(x[t], strike, degree, sort_state)
            alpha = _ols(Phi_now, regressand)
            continuation = Phi_now @ alpha
        M[t + 1] = M[t] + (value_next - continuation) / bank[t]
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three alpha-martingale cases."""
    return [
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,2); paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,3); strike=100.; r=.05; dt=.6/5; degree=1; sort_state=False; theta=_oracle_fit_min_put_primal_policy(train,strike,r,dt,degree,sort_state); H_D=_oracle_evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "call": "single_projection_alpha_martingale(paths,H_D,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_single_projection_alpha_martingale(paths,H_D,strike,r,dt,degree,sort_state)",
            "tol": 2e-9,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,7); paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,8); strike=100.; r=.04; dt=.75/6; degree=2; sort_state=True; theta=_oracle_fit_min_put_primal_policy(train,strike,r,dt,degree,sort_state); H_D=_oracle_evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "call": "single_projection_alpha_martingale(paths,H_D,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_single_projection_alpha_martingale(paths,H_D,strike,r,dt,degree,sort_state)",
            "tol": 2e-8,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,11); paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,12); strike=102.; r=.03; dt=.8/7; degree=4; sort_state=False; theta=_oracle_fit_min_put_primal_policy(train,strike,r,dt,degree,sort_state); H_D=_oracle_evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "call": "single_projection_alpha_martingale(paths,H_D,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_single_projection_alpha_martingale(paths,H_D,strike,r,dt,degree,sort_state)",
            "tol": 2e-7,
        },
    ]
