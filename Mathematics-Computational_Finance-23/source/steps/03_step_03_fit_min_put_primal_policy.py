"""
Fit the backward primal stopping policy on the independent training sample.

The source begins from a stopping-time approximation whose continuation values are successive least-squares projections of one-period-discounted realized future payoffs. It uses all paths for the regressions and a separate path sample for later policy evaluation to remove primal foresight bias. Exercise is taken at a date only when the immediate min-put payoff is strictly positive and at least the fitted continuation value; otherwise the one-step-discounted realized payoff is carried back.

Returns
-------
A NumPy coefficient array of shape `(steps + 1, p)` for the requested basis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_min_put_primal_policy(paths: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Fit backward continuation coefficients on training paths.

    Parameters
    ----------
    paths : np.ndarray
        Training paths of shape ``(steps + 1, n_paths, 2)``.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.
    sort_state : bool
        Whether to sort the two asset prices pathwise before basis evaluation.

    Returns
    -------
    theta : np.ndarray
        Coefficient array of shape ``(steps + 1, p)`` with the fitted rows for
        dates 1 through ``steps - 1`` and zero rows at initiation and maturity.
    """
    return theta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _min_put_payoff(states: "np.ndarray", strike: float) -> "np.ndarray":
    x = np.asarray(states, dtype=float)
    return np.maximum(float(strike) - np.minimum(x[..., 0], x[..., 1]), 0.0)


def _ols(Phi: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    return np.linalg.lstsq(np.asarray(Phi, dtype=float), np.asarray(y, dtype=float), rcond=None)[0]


def _oracle_fit_min_put_primal_policy(paths: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    steps = x.shape[0] - 1
    p = len(_basis_exponents(int(degree)))
    theta = np.zeros((steps + 1, p), dtype=float)
    realized = _min_put_payoff(x[steps], strike)
    disc = np.exp(-float(r) * float(dt))
    for t in range(steps - 1, 0, -1):
        regressand = disc * realized
        Phi = _oracle_min_put_polynomial_basis(x[t], strike, degree, sort_state)
        theta[t] = _ols(Phi, regressand)
        continuation = Phi @ theta[t]
        immediate = _min_put_payoff(x[t], strike)
        exercise = (immediate > 0.0) & (immediate >= continuation)
        realized = np.where(exercise, immediate, regressand)
    return theta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three backward-policy cases."""
    return [
        {
            "setup": "import numpy as np\npaths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([100.,100.]),.05,0.,.24,.2,.6,5,96,2); strike=100.; r=.05; dt=.6/5; degree=1; sort_state=False",
            "call": "fit_min_put_primal_policy(paths,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_fit_min_put_primal_policy(paths,strike,r,dt,degree,sort_state)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\npaths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([96.,104.]),.04,.01,.27,-.15,.75,6,128,7); strike=100.; r=.04; dt=.75/6; degree=2; sort_state=True",
            "call": "fit_min_put_primal_policy(paths,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_fit_min_put_primal_policy(paths,strike,r,dt,degree,sort_state)",
            "tol": 1e-9,
        },
        {
            "setup": "import numpy as np\npaths=_oracle_simulate_correlated_antithetic_gbm_paths(np.array([108.,108.]),.03,0.,.32,.55,.8,7,192,11); strike=102.; r=.03; dt=.8/7; degree=4; sort_state=False",
            "call": "fit_min_put_primal_policy(paths,strike,r,dt,degree,sort_state)",
            "gold_call": "_oracle_fit_min_put_primal_policy(paths,strike,r,dt,degree,sort_state)",
            "tol": 2e-8,
        },
    ]
