"""
Replay the fixed policy on independent evaluation paths and recover the full realized-payoff process.

The regression-based dual algorithms require a backward-primal payoff cross section at every date. Replaying the already fitted policy on new paths supplies that process without refitting the stopping rule on its own evaluation sample. The replay applies the same exercise test as the fit: exercise only when the immediate min-put payoff is strictly positive and at least the continuation value implied by the supplied coefficients.

Returns
-------
A NumPy array of shape `(steps + 1, n_paths)` containing the realized payoff process and its time-zero discounted row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_min_put_payoff_process(paths: "np.ndarray", theta: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Evaluate a fixed min-put stopping policy on new paths.

    Parameters
    ----------
    paths : np.ndarray
        Evaluation paths of shape ``(steps + 1, n_paths, 2)``.
    theta : np.ndarray
        Backward continuation coefficients from the training sample.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree used by ``theta``.
    sort_state : bool
        Whether the fixed policy evaluates the sorted-state basis.

    Returns
    -------
    H_D : np.ndarray
        Array of shape ``(steps + 1, n_paths)``. Rows 1 through maturity are
        the source-style realized payoff process in each row's time units; row
        0 is the one-period-discounted row-1 payoff used for the time-zero
        lower estimate.
    """
    return H_D

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_min_put_payoff_process(paths: "np.ndarray", theta: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    th = np.asarray(theta, dtype=float)
    steps = x.shape[0] - 1
    H = np.zeros((steps + 1, x.shape[1]), dtype=float)
    H[steps] = _min_put_payoff(x[steps], strike)
    disc = np.exp(-float(r) * float(dt))
    for t in range(steps - 1, 0, -1):
        regressand = disc * H[t + 1]
        Phi = _oracle_min_put_polynomial_basis(x[t], strike, degree, sort_state)
        continuation = Phi @ th[t]
        immediate = _min_put_payoff(x[t], strike)
        exercise = (immediate > 0.0) & (immediate >= continuation)
        H[t] = np.where(exercise, immediate, regressand)
    H[0] = disc * H[1]
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three independent policy-evaluation cases."""
    return [
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,2); paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,3); strike=100.; r=.05; dt=.6/5; degree=1; sort_state=False; theta=_oracle_fit_min_put_primal_policy(train,strike,r,dt,degree,sort_state)",
            "call": "evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,7); paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,8); strike=100.; r=.04; dt=.75/6; degree=2; sort_state=True; theta=_oracle_fit_min_put_primal_policy(train,strike,r,dt,degree,sort_state)",
            "call": "evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "tol": 1e-9,
        },
        {
            "setup": "import numpy as np\ntrain=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,11); paths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,12); strike=102.; r=.03; dt=.8/7; degree=4; sort_state=False; theta=_oracle_fit_min_put_primal_policy(train,strike,r,dt,degree,sort_state)",
            "call": "evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_evaluate_min_put_payoff_process(paths,theta,strike,r,dt,degree,sort_state)",
            "tol": 2e-8,
        },
    ]
