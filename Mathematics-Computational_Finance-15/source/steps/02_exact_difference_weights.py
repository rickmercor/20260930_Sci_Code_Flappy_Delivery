"""
Compute the exact finite-difference weights of the Cauchy-Euler part of the Black-Scholes operator on a uniform asset grid. Given the number of asset intervals, the risk-free rate and the volatility, return the three weight arrays A1, A2 and A3 of the interior grid nodes m = 1, ..., M-1.

The time-independent part of the Black-Scholes operator, (sigma^2/2) s^2 u'' + r s u' - r u = 0, is an equidimensional (Cauchy-Euler) equation, so its two independent solutions are powers of s. A three-point relation between grid values at consecutive integer indices j, j+1, j+2 is exact for this equation when it is satisfied by both power solutions. Up to a common factor, its coefficients are the signed 2 x 2 minors obtained by expanding, along its first row, the 3 x 3 determinant whose first row is (u_{j+2}, u_{j+1}, u_j) and whose second and third rows hold the two power solutions, each written as a monomial x^lambda with unit coefficient and evaluated at x = j+2, j+1, j (the solution with the larger exponent in the second row). Because the solutions are homogeneous in s, the grid spacing only rescales each row and is dropped.



The weights are the magnitudes of these minors: the expansion equals A1_j u_{j+2} - A2_j u_{j+1} + A3_j u_j, with A1_j, A2_j and A3_j all positive when r > 0. The weights assigned to interior node m are those with j = m, built on the indices m, m+1, m+2; this keeps every power evaluation away from index 0, where the negative power is singular (a stencil starting at m - 1 would reach index 0 when m = 1).

Returns
-------
tuple of three np.ndarray, the exact-difference weights (A1, A2, A3), each float64 of shape (M-1,) at m = 1, ..., M-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_difference_weights(n_intervals: int, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Cauchy-Euler exact-difference weights A1, A2, A3 at m = 1, ..., M-1.

    Parameters
    ----------
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.

    Returns
    -------
    A1, A2, A3 : np.ndarray
        Float64 arrays of shape (M-1,), entry m-1 holding the weight at index m
        as defined in the step background.

    Raises
    ------
    ValueError
        If n_intervals is not an integer of at least 2, or rate or volatility
        is not finite and strictly positive.
    """
    return A1, A2, A3

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exact_difference_weights(n_intervals: int, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or n_intervals < 2:
        raise ValueError("n_intervals must be an integer of at least 2.")
    rate, volatility = float(rate), float(volatility)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be finite and positive.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive.")
    alpha = rate / (0.5 * volatility ** 2)
    m = np.arange(1, int(n_intervals), dtype=float)
    a1 = (m + 1.0) / m ** alpha - m / (m + 1.0) ** alpha
    a2 = (m + 2.0) / m ** alpha - m / (m + 2.0) ** alpha
    a3 = (m + 2.0) / (m + 1.0) ** alpha - (m + 1.0) / (m + 2.0) ** alpha
    return a1, a2, a3

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = "import numpy as np\ndef pack(out):\n    return np.concatenate([np.asarray(a, dtype=float) for a in out])\n"
    return [
        # Benchmark grid: 64 intervals, r = 0.05, sigma = 0.3 (alpha = 10/9).
        {
            "setup": pack,
            "call": "pack(exact_difference_weights(64, 0.05, 0.3))",
            "gold_call": "pack(_oracle_exact_difference_weights(64, 0.05, 0.3))",
            "tol": 1e-12,
        },
        # Large alpha (low volatility): the weights are dominated by the m^(-alpha) terms.
        {
            "setup": pack,
            "call": "pack(exact_difference_weights(12, 0.08, 0.1))",
            "gold_call": "pack(_oracle_exact_difference_weights(12, 0.08, 0.1))",
            "tol": 1e-9,
        },
        # Boundary: two intervals, so a single interior node m = 1.
        {
            "setup": pack,
            "call": "pack(exact_difference_weights(2, 0.02, 0.4))",
            "gold_call": "pack(_oracle_exact_difference_weights(2, 0.02, 0.4))",
            "tol": 1e-12,
        },
        # Invalid: zero rate.
        {
            "setup": "def code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: exact_difference_weights(8, 0.0, 0.3))",
            "gold_call": "code(lambda: _oracle_exact_difference_weights(8, 0.0, 0.3))",
        },
    ]
