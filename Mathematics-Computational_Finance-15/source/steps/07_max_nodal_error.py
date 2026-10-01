"""
Measure the discretisation error of a numerical European put solution by its maximum nodal error. Given the full space-time grid of numerical values and the parameters that produced it, return the largest absolute difference between the numerical value and the exact solution of the smoothed-payoff Black-Scholes problem over every node of the grid.

The scheme is started from the smoothed payoff Psi(K - s), so the exact solution of the problem it discretises is u(s, t) = exp(-r t) E[Psi(K - S_t) | S_0 = s], with t the time to maturity. Comparing with that solution, instead of the Black-Scholes price of the kinked payoff, separates the discretisation error from the smoothing error. The error norm is E = max over k = 0, ..., N and m = 0, ..., M of |v_m^k - u(s_m, t_k)|, with s_m = m S_max / M and t_k = k T / N; it runs over all time levels, including the initial one, and over the boundary nodes.

Returns
-------
float, the maximum absolute nodal error against the exact smoothed-payoff solution over the whole grid, as a Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def max_nodal_error(grid: "np.ndarray", strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float) -> float:
    """Maximum absolute error of a put grid against the exact smoothed-payoff solution.

    Parameters
    ----------
    grid : np.ndarray
        Finite array of shape (N+1, M+1) with N >= 1 and M >= 1; grid[k, m]
        is the numerical value at time to maturity k T / N and asset price
        m S_max / M.
    strike : float
        Strike K, finite and strictly positive.
    rate : float
        Risk-free rate r, finite.
    volatility : float
        Volatility sigma, finite and strictly positive.
    maturity : float
        Maturity T, finite and strictly positive.
    s_max : float
        Right end S_max of the asset domain, finite and strictly positive.
    eps : float
        Half-width of the payoff smoothing band, finite with 0 < eps < K.

    Returns
    -------
    error : float
        max over all nodes of |grid[k, m] - u(s_m, t_k)|, as a Python float.

    Raises
    ------
    ValueError
        If grid is not a finite two-dimensional array with at least two rows
        and two columns, or a scalar parameter violates the conditions above.
    """
    return error

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_max_nodal_error(grid: "np.ndarray", strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float) -> float:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a finite 2-D array with at least two rows and columns.")
    maturity, s_max = float(maturity), float(s_max)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(s_max) or s_max <= 0.0:
        raise ValueError("s_max must be finite and positive.")
    n_steps, n_intervals = grid.shape[0] - 1, grid.shape[1] - 1
    s = np.linspace(0.0, s_max, n_intervals + 1)
    error = 0.0
    for k in range(n_steps + 1):
        exact = _oracle_smoothed_put_exact(s, strike, rate, volatility, k * maturity / n_steps, eps)
        error = max(error, float(np.max(np.abs(grid[k] - exact))))
    return error

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Discounted intrinsic values on a 32-interval grid with a wide band: the error peaks near the strike.
        {
            "setup": "import numpy as np\ns = np.linspace(0.0, 4.0, 33)\nt = np.linspace(0.0, 1.0, 9)[:, None]\ng = np.maximum(1.0 - s[None, :], 0.0) * np.exp(-0.1 * t)\n",
            "call": "max_nodal_error(g, 1.0, 0.1, 0.3, 1.0, 4.0, 0.05)",
            "gold_call": "_oracle_max_nodal_error(g.copy(), 1.0, 0.1, 0.3, 1.0, 4.0, 0.05)",
            "tol": 1e-12,
        },
        # A constant grid on an unequal space-time resolution, whose error peaks where the exact price is extreme.
        {
            "setup": "import numpy as np\ng = np.full((6, 11), 0.2)\n",
            "call": "max_nodal_error(g, 1.2, 0.03, 0.4, 2.0, 3.0, 0.01)",
            "gold_call": "_oracle_max_nodal_error(g.copy(), 1.2, 0.03, 0.4, 2.0, 3.0, 0.01)",
            "tol": 1e-12,
        },
        # Boundary: the smallest grid, one time step and one asset interval.
        {
            "setup": "import numpy as np\ng = np.array([[1.0, 0.0], [0.97, 0.01]])\n",
            "call": "max_nodal_error(g, 1.0, 0.06, 0.25, 0.5, 2.0, 0.05)",
            "gold_call": "_oracle_max_nodal_error(g.copy(), 1.0, 0.06, 0.25, 0.5, 2.0, 0.05)",
            "tol": 1e-12,
        },
        # Invalid: a one-dimensional grid.
        {
            "setup": "import numpy as np\ndef code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: max_nodal_error(np.zeros(5), 1.0, 0.05, 0.3, 0.5, 4.0, 0.05))",
            "gold_call": "code(lambda: _oracle_max_nodal_error(np.zeros(5), 1.0, 0.05, 0.3, 0.5, 4.0, 0.05))",
        },
    ]
