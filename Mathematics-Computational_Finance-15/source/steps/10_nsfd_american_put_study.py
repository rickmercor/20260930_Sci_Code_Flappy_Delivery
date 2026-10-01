"""
Price an American put and its European counterpart with the NSFD scheme on one grid and report the accuracy of the European solution and the early-exercise premium. Given the contract and market parameters, the asset domain, the smoothing half-width and the number of asset intervals M (with N = M time steps), return the maximum nodal error of the European NSFD solution against the exact solution of the smoothed-payoff Black-Scholes problem, the European and American NSFD values at the strike at maturity, and their difference.

The European solution has an exact reference, the Black-Scholes solution with the smoothed payoff as initial data, so its maximum nodal error over the whole grid measures the discretisation error alone; the American put has no closed form, and on the same grid its value exceeds the European value by the early-exercise premium. Both values at the strike are read from the last time level (time to maturity T) by linear interpolation in the asset direction, which returns the nodal value when K is a grid node.

Returns
-------
np.ndarray, float64 array of shape (4,): [European maximum nodal error E against the exact smoothed-payoff solution, European value at s = K, American value at s = K, early-exercise premium], values at time to maturity T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nsfd_american_put_study(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int) -> "np.ndarray":
    """European error, European and American values at the strike, and the premium.

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
        Number M of asset intervals and of time steps, an integer of at least 2.

    Returns
    -------
    result : np.ndarray
        Float64 array of shape (4,): [E, european, american, premium], where E
        is the maximum nodal error of the European M x M solution against the
        exact smoothed-payoff solution, european and american are the European and American
        M x M values at time to maturity T linearly interpolated at s = K, and
        premium = american - european.

    Raises
    ------
    ValueError
        If any parameter violates the conditions above.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_nsfd_american_put_study(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int) -> "np.ndarray":
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or n_intervals < 2:
        raise ValueError("n_intervals must be an integer of at least 2.")
    m = int(n_intervals)
    european_grid = _oracle_solve_nsfd_put(strike, rate, volatility, maturity, s_max, eps, m, m)
    error = _oracle_max_nodal_error(european_grid, strike, rate, volatility, maturity, s_max, eps)
    american_grid = _oracle_solve_nsfd_american_put(strike, rate, volatility, maturity, s_max, eps, m, m)
    s = np.linspace(0.0, float(s_max), m + 1)
    european = float(np.interp(float(strike), s, european_grid[-1]))
    american = float(np.interp(float(strike), s, american_grid[-1]))
    return np.array([error, european, american, american - european])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # High-rate contract on a 64 x 64 grid with a wide smoothing band.
        {
            "setup": "import numpy as np\n",
            "call": "nsfd_american_put_study(1.0, 0.1, 0.3, 1.0, 4.0, 0.05, 64)",
            "gold_call": "_oracle_nsfd_american_put_study(1.0, 0.1, 0.3, 1.0, 4.0, 0.05, 64)",
            "tol": 1e-9,
        },
        # Longer maturity, higher volatility, strike off the grid.
        {
            "setup": "import numpy as np\n",
            "call": "nsfd_american_put_study(1.1, 0.03, 0.45, 1.5, 5.0, 0.05, 40)",
            "gold_call": "_oracle_nsfd_american_put_study(1.1, 0.03, 0.45, 1.5, 5.0, 0.05, 40)",
            "tol": 1e-9,
        },
        # Boundary: the smallest grid.
        {
            "setup": "import numpy as np\n",
            "call": "nsfd_american_put_study(1.0, 0.04, 0.4, 1.0, 8.0, 1e-3, 2)",
            "gold_call": "_oracle_nsfd_american_put_study(1.0, 0.04, 0.4, 1.0, 8.0, 1e-3, 2)",
            "tol": 1e-9,
        },
        # Invalid: grid too small.
        {
            "setup": "def code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: nsfd_american_put_study(1.0, 0.05, 0.3, 0.5, 4.0, 1e-3, 1))",
            "gold_call": "code(lambda: _oracle_nsfd_american_put_study(1.0, 0.05, 0.3, 0.5, 4.0, 1e-3, 1))",
        },
    ]
