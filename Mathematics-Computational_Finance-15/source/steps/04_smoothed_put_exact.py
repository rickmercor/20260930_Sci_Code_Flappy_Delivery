"""
Evaluate the exact solution of the Black-Scholes equation whose initial condition is the smoothed put payoff. Given asset prices, the strike, the rate, the volatility, the time to maturity and the smoothing half-width, return u(s, tau) = exp(-r tau) E[Psi(K - S_tau) | S_0 = s], where S follows risk-neutral geometric Brownian motion and Psi is the C^4 smoothed version of max(x, 0) on the band [-eps, eps].

The numerical scheme is started from the smoothed payoff, so the problem it discretises is the Black-Scholes equation with initial data Psi(K - s), not the kinked payoff. Its exact solution is therefore the reference that isolates the discretisation error from the smoothing error. The returned values are exact to an absolute accuracy of 1e-12. At s = 0 the solution is K exp(-r tau) (the band lies at positive prices), and at tau = 0 it is Psi(K - s).

Returns
-------
np.ndarray, float64 array with the shape of s holding exp(-r tau) E[Psi(K - S_tau)], the exact Black-Scholes solution with the smoothed put payoff as initial data, to absolute accuracy 1e-12
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smoothed_put_exact(s: "np.ndarray", strike: float, rate: float, volatility: float, tau: float, eps: float) -> "np.ndarray":
    """Exact Black-Scholes solution with the smoothed put payoff as initial data.

    Parameters
    ----------
    s : np.ndarray
        Asset prices, any shape, finite and non-negative.
    strike : float
        Strike K, finite and strictly positive.
    rate : float
        Risk-free rate r, finite.
    volatility : float
        Volatility sigma, finite and strictly positive.
    tau : float
        Time to maturity, finite and non-negative.
    eps : float
        Half-width of the payoff smoothing band, finite, positive and below K.

    Returns
    -------
    price : np.ndarray
        Float64 array with the shape of s: exp(-rate tau) E[Psi(K - S_tau)]
        for s > 0 and tau > 0, K exp(-rate tau) at s = 0, and Psi(K - s) at
        tau = 0, with absolute error at most 1e-12.

    Raises
    ------
    ValueError
        If s is not finite and non-negative, strike is not finite and positive,
        rate is not finite, volatility is not finite and positive, tau is not
        finite and non-negative, or eps is not finite with 0 < eps < strike.
    """
    return price

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ndtr


def _bs_put(s, strike, rate, volatility, tau):
    price = np.full(s.shape, strike * np.exp(-rate * tau))
    pos = s > 0.0
    root = volatility * np.sqrt(tau)
    d1 = (np.log(s[pos] / strike) + (rate + 0.5 * volatility ** 2) * tau) / root
    price[pos] = strike * np.exp(-rate * tau) * ndtr(-(d1 - root)) - s[pos] * ndtr(-d1)
    return price


def _oracle_smoothed_put_exact(s: "np.ndarray", strike: float, rate: float, volatility: float, tau: float, eps: float) -> "np.ndarray":
    s = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("s must be finite and non-negative.")
    strike, rate, volatility, tau, eps = float(strike), float(rate), float(volatility), float(tau), float(eps)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative.")
    if not np.isfinite(eps) or eps <= 0.0 or eps >= strike:
        raise ValueError("eps must be finite with 0 < eps < strike.")
    if tau == 0.0:
        return _oracle_smooth_put_payoff(s, strike, eps)
    price = _bs_put(s, strike, rate, volatility, tau)
    pos = s > 0.0
    # Composite Gauss-Legendre in x = K - S on [-eps, 0] and [0, eps]; the kink of the integrand is a panel edge,
    # and the panels are narrow compared with the lognormal spread at the band.
    root = volatility * np.sqrt(tau)
    per_half = int(min(4096, max(8, np.ceil(4.0 * eps / ((strike - eps) * root)))))
    edges = np.linspace(-eps, eps, 2 * per_half + 1)
    xg, wg = np.polynomial.legendre.leggauss(16)
    half = 0.5 * np.diff(edges)
    x = (half[:, None] * xg[None, :] + 0.5 * (edges[:-1] + edges[1:])[:, None]).ravel()
    w = (half[:, None] * wg[None, :]).ravel()
    even = (35.0 / 256.0 * eps + 35.0 / (64.0 * eps) * x ** 2 - 35.0 / (128.0 * eps ** 3) * x ** 4
            + 7.0 / (64.0 * eps ** 5) * x ** 6 - 5.0 / (256.0 * eps ** 7) * x ** 8)
    bump = even - 0.5 * np.abs(x)
    big_s = strike - x
    sp = s[pos].reshape(-1, 1)
    z = (np.log(big_s[None, :] / sp) - (rate - 0.5 * volatility ** 2) * tau) / root
    density = np.exp(-0.5 * z ** 2) / (np.sqrt(2.0 * np.pi) * root * big_s[None, :])
    price[pos] += np.exp(-rate * tau) * (density * (bump * w)[None, :]).sum(axis=1)
    return price

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Grid on [0, 4] with 32 intervals after one step of 1/128: several nodes feel the wide band eps = 0.05.
        {
            "setup": "import numpy as np\ns = np.linspace(0.0, 4.0, 129)\n",
            "call": "smoothed_put_exact(s, 1.0, 0.1, 0.3, 1.0 / 128, 0.05)",
            "gold_call": "_oracle_smoothed_put_exact(s.copy(), 1.0, 0.1, 0.3, 1.0 / 128, 0.05)",
            "tol": 1e-12,
        },
        # Long maturity, narrow band, prices around the strike (two-dimensional input).
        {
            "setup": "import numpy as np\ns = np.array([[0.9, 0.99, 1.0], [1.01, 1.1, 1.5]])\n",
            "call": "smoothed_put_exact(s, 1.0, 0.05, 0.3, 1.0, 1e-3)",
            "gold_call": "_oracle_smoothed_put_exact(s.copy(), 1.0, 0.05, 0.3, 1.0, 1e-3)",
            "tol": 1e-12,
        },
        # Short maturity with a strike of 2 and a band comparable to the lognormal spread.
        {
            "setup": "import numpy as np\ns = np.linspace(1.8, 2.2, 17)\n",
            "call": "smoothed_put_exact(s, 2.0, 0.08, 0.25, 0.01, 0.04)",
            "gold_call": "_oracle_smoothed_put_exact(s.copy(), 2.0, 0.08, 0.25, 0.01, 0.04)",
            "tol": 1e-12,
        },
        # Boundary: tau = 0 returns the smoothed payoff, including the node s = 0.
        {
            "setup": "import numpy as np\ns = np.array([0.0, 0.97, 1.0, 1.02, 1.5])\n",
            "call": "smoothed_put_exact(s, 1.0, 0.1, 0.3, 0.0, 0.05)",
            "gold_call": "_oracle_smoothed_put_exact(s.copy(), 1.0, 0.1, 0.3, 0.0, 0.05)",
            "tol": 1e-12,
        },
        # Invalid: smoothing band wider than the strike.
        {
            "setup": "import numpy as np\ndef code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: smoothed_put_exact(np.array([1.0]), 1.0, 0.05, 0.3, 0.5, 1.2))",
            "gold_call": "code(lambda: _oracle_smoothed_put_exact(np.array([1.0]), 1.0, 0.05, 0.3, 0.5, 1.2))",
        },
    ]
