"""
Solve the Black-Scholes equation for a European put with the implicit NSFD scheme on a uniform grid. Given the contract and market parameters, the truncated asset domain, the smoothing half-width and the numbers of asset intervals and time steps, return the numerical put value at every grid node and every time level.

The domain [0, S_max] x [0, T] in asset price and time to maturity is split into M equal asset intervals of width ds = S_max / M and N equal time steps of length dt = T / N. The initial row (t = 0) is the smoothed put payoff Psi(K - s_m) at every node, boundary nodes included. Each later row k = 1, ..., N is obtained by one implicit NSFD step from row k-1, using the coefficients built once from M, dt, r and sigma. The Dirichlet data of a put on the truncated domain are the value of a riskless claim on the strike at s = 0, u(0, t) = K exp(-r t), and zero at the far boundary, u(S_max, t) = 0.

Returns
-------
np.ndarray, float64 array of shape (N+1, M+1) with the NSFD put values at every time level and asset node
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_nsfd_put(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int, n_steps: int) -> "np.ndarray":
    """NSFD put values on the full space-time grid.

    Parameters
    ----------
    strike : float
        Strike K, finite and strictly positive, with K + eps < s_max.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.
    maturity : float
        Maturity T, finite and strictly positive.
    s_max : float
        Right end S_max of the asset domain, finite and larger than K + eps.
    eps : float
        Half-width of the payoff smoothing band, finite, positive and below K.
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    n_steps : int
        Number N of time steps, an integer of at least 1.

    Returns
    -------
    grid : np.ndarray
        Float64 array of shape (N+1, M+1); grid[k, m] approximates the put
        value at time to maturity k T / N and asset price m S_max / M.

    Raises
    ------
    ValueError
        If any parameter violates the conditions above.
    """
    return grid

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_nsfd_put(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int, n_steps: int) -> "np.ndarray":
    strike, maturity, s_max, eps = float(strike), float(maturity), float(s_max), float(eps)
    if not all(np.isfinite([strike, maturity, s_max, eps])):
        raise ValueError("strike, maturity, s_max and eps must be finite.")
    if strike <= 0.0 or maturity <= 0.0 or eps <= 0.0 or eps >= strike or strike + eps >= s_max:
        raise ValueError("need strike, maturity > 0, 0 < eps < strike and strike + eps < s_max.")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    dt = maturity / int(n_steps)
    lower, centre, upper, psi1 = _oracle_nsfd_coefficients(n_intervals, dt, rate, volatility)
    s = np.linspace(0.0, s_max, int(n_intervals) + 1)
    grid = np.empty((int(n_steps) + 1, s.size))
    grid[0] = _oracle_smooth_put_payoff(s, strike, eps)
    for k in range(1, int(n_steps) + 1):
        left = strike * np.exp(-float(rate) * k * dt)
        grid[k] = _oracle_nsfd_time_step(grid[k - 1], lower, centre, upper, psi1, left, 0.0)
    return grid

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Benchmark contract on a coarse 32 x 32 grid.
        {
            "setup": "import numpy as np\n",
            "call": "solve_nsfd_put(1.0, 0.05, 0.3, 0.5, 4.0, 1e-3, 32, 32)",
            "gold_call": "_oracle_solve_nsfd_put(1.0, 0.05, 0.3, 0.5, 4.0, 1e-3, 32, 32)",
            "tol": 1e-10,
        },
        # Unequal space and time resolution, strike not on a grid node, wider smoothing band.
        {
            "setup": "import numpy as np\n",
            "call": "solve_nsfd_put(1.1, 0.03, 0.45, 1.5, 5.0, 0.05, 40, 15)",
            "gold_call": "_oracle_solve_nsfd_put(1.1, 0.03, 0.45, 1.5, 5.0, 0.05, 40, 15)",
            "tol": 1e-10,
        },
        # Boundary: a single time step on a small grid.
        {
            "setup": "import numpy as np\n",
            "call": "solve_nsfd_put(2.0, 0.08, 0.2, 0.25, 6.0, 0.01, 6, 1)",
            "gold_call": "_oracle_solve_nsfd_put(2.0, 0.08, 0.2, 0.25, 6.0, 0.01, 6, 1)",
            "tol": 1e-12,
        },
        # Invalid: strike beyond the truncated domain.
        {
            "setup": "def code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: solve_nsfd_put(5.0, 0.05, 0.3, 0.5, 4.0, 1e-3, 16, 16))",
            "gold_call": "code(lambda: _oracle_solve_nsfd_put(5.0, 0.05, 0.3, 0.5, 4.0, 1e-3, 16, 16))",
        },
    ]
