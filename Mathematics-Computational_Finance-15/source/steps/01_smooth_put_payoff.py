"""
Evaluate the smoothed European put payoff used as the initial condition of the non-standard finite-difference scheme. Given asset prices, the strike and the half-width eps of the smoothing band, return phi(s) = Psi(K - s), where Psi replaces the kink of max(x, 0) at x = 0 by a polynomial on the band [-eps, eps].

The put payoff max(K - s, 0) has a kink at s = K, where its second derivative does not exist, and the Black-Scholes operator needs that derivative. The scheme therefore starts from a smoothed payoff: Psi(x) is 0 for x <= -eps, x for x >= eps, and a ninth-degree polynomial on [-eps, eps] whose value and first four derivatives match those of the two linear pieces at x = -eps and x = eps, so that Psi is four times continuously differentiable. These ten matching conditions determine the polynomial uniquely; Psi(x) - x/2 is an even function, so the polynomial contains only the constant term, the linear term x/2 and the even powers x^2, x^4, x^6 and x^8.

Returns
-------
np.ndarray, float64 array with the shape of s holding the smoothed put payoff Psi(K - s)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smooth_put_payoff(s: "np.ndarray", strike: float, eps: float) -> "np.ndarray":
    """Smoothed put payoff phi(s) = Psi(K - s).

    Parameters
    ----------
    s : np.ndarray
        Asset prices, any shape, all finite.
    strike : float
        Strike K, finite and strictly positive.
    eps : float
        Half-width of the smoothing band, finite and strictly positive.

    Returns
    -------
    phi : np.ndarray
        Float64 array with the shape of s: 0 where K - s <= -eps, K - s where
        K - s >= eps, and the C^4 ninth-degree matching polynomial of x = K - s
        on -eps < x < eps.

    Raises
    ------
    ValueError
        If s contains non-finite values, strike is not finite and positive,
        or eps is not finite and positive.
    """
    return phi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_smooth_put_payoff(s: "np.ndarray", strike: float, eps: float) -> "np.ndarray":
    s = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("s must be finite.")
    strike, eps = float(strike), float(eps)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    x = strike - s
    poly = (35.0 / 256.0 * eps + 0.5 * x + 35.0 / (64.0 * eps) * x ** 2 - 35.0 / (128.0 * eps ** 3) * x ** 4
            + 7.0 / (64.0 * eps ** 5) * x ** 6 - 5.0 / (256.0 * eps ** 7) * x ** 8)
    return np.where(x <= -eps, 0.0, np.where(x >= eps, x, poly))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Uniform grid on [0, 4] with 128 intervals: the strike 1 is a node inside the band.
        {
            "setup": "import numpy as np\ns = np.linspace(0.0, 4.0, 129)\n",
            "call": "smooth_put_payoff(s, 1.0, 1e-3)",
            "gold_call": "_oracle_smooth_put_payoff(s.copy(), 1.0, 1e-3)",
            "tol": 1e-12,
        },
        # Points spread across a wide band, including both band edges and the centre.
        {
            "setup": "import numpy as np\ns = np.array([0.7, 0.8, 0.85, 0.95, 1.0, 1.05, 1.15, 1.2, 1.3])\n",
            "call": "smooth_put_payoff(s, 1.0, 0.2)",
            "gold_call": "_oracle_smooth_put_payoff(s.copy(), 1.0, 0.2)",
            "tol": 1e-12,
        },
        # Edge: a two-dimensional array of prices around a strike of 2.5 with a narrow band.
        {
            "setup": "import numpy as np\ns = np.array([[2.49, 2.4995, 2.5], [2.5003, 2.51, 3.0]])\n",
            "call": "smooth_put_payoff(s, 2.5, 5e-4)",
            "gold_call": "_oracle_smooth_put_payoff(s.copy(), 2.5, 5e-4)",
            "tol": 1e-12,
        },
        # Invalid: non-positive band half-width.
        {
            "setup": "import numpy as np\ndef code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: smooth_put_payoff(np.array([1.0]), 1.0, 0.0))",
            "gold_call": "code(lambda: _oracle_smooth_put_payoff(np.array([1.0]), 1.0, 0.0))",
        },
    ]
