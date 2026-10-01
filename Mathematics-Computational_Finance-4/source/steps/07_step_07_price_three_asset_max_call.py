"""
Price a European call on the maximum of three assets exactly. Given the current prices of the three assets, the strike, the time to expiry, the risk-free rate, each asset's dividend yield and volatility and the correlation matrix of their Brownian motions, return the arbitrage-free price of the payoff max(max(S1, S2, S3) - K, 0) at expiry.

The three assets follow correlated geometric Brownian motions under the risk-neutral measure, each with its own continuous dividend yield and volatility, and the price is the discounted risk-neutral expectation of the payoff. The same call with early exercise permitted is the Bermudan option priced by the pipeline, so the true Bermudan value is at least this European value, which serves as the no-early-exercise benchmark. The expectation has an exact value, and it must be computed to double precision; sampling-based estimates and coarse numerical integration are not adequate. At zero time to expiry the price equals the payoff itself.

Returns
-------
np.ndarray of prices with shape spots.shape[:-1] (a 0-d array when spots has shape (3,))
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def price_three_asset_max_call(
    spots: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    dividend_yields: "np.ndarray",
    volatilities: "np.ndarray",
    correlation: "np.ndarray",
) -> "np.ndarray":
    """European call on max(S1, S2, S3) with strike K and time to expiry tau.

    Parameters
    ----------
    spots : np.ndarray
        Current prices of shape (..., 3), all finite and strictly positive;
        the last axis holds the three assets and any leading axes index
        independent price vectors.
    strike : float
        Strike K, finite and strictly positive.
    tau : float
        Time to expiry in years, finite and non-negative.
    rate : float
        Continuously compounded risk-free rate, finite.
    dividend_yields : np.ndarray
        Continuous dividend yields of the three assets, shape (3,), finite.
    volatilities : np.ndarray
        Volatilities of the three assets, shape (3,), finite and strictly
        positive.
    correlation : np.ndarray
        Correlation matrix of the three Brownian motions, shape (3, 3),
        finite, symmetric, with unit diagonal and positive definite.

    Returns
    -------
    price : np.ndarray
        Float64 prices with shape spots.shape[:-1]. For tau = 0 the price is
        max(max(spots over the last axis) - strike, 0).

    Raises
    ------
    ValueError
        If spots is not a finite, strictly positive array whose last axis has
        length 3, strike is not finite and positive, tau is not finite and
        non-negative, rate is not finite, dividend_yields is not a finite
        array of shape (3,), volatilities is not a finite, strictly positive
        array of shape (3,), or correlation is not a finite, symmetric,
        positive definite (3, 3) matrix with unit diagonal.
    """
    return price

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import integrate, special


def _bvn_owen(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho, via Owen's T function."""
    if h == 0.0 and k == 0.0:
        return 0.25 + np.arcsin(rho) / (2.0 * np.pi)
    s = np.sqrt(1.0 - rho * rho)
    ah = (k - rho * h) / (h * s) if h != 0.0 else np.copysign(np.inf, k)
    ak = (h - rho * k) / (k * s) if k != 0.0 else np.copysign(np.inf, h)
    beta = 0.5 if (h * k < 0.0 or (h * k == 0.0 and h + k < 0.0)) else 0.0
    return float(0.5 * special.ndtr(h) + 0.5 * special.ndtr(k)
                 - special.owens_t(h, ah) - special.owens_t(k, ak) - beta)


def _tvn(a, r):
    """P(X1 <= a1, X2 <= a2, X3 <= a3) for a standard trivariate normal with correlation matrix r.

    Conditions on X1 and integrates the exact bivariate distribution function of (X2, X3) given X1.
    """
    r12, r13, r23 = r[0, 1], r[0, 2], r[1, 2]
    s12, s13 = np.sqrt(1.0 - r12 * r12), np.sqrt(1.0 - r13 * r13)
    rc = (r23 - r12 * r13) / (s12 * s13)

    def _integrand(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi) * _bvn_owen((a[1] - r12 * x) / s12, (a[2] - r13 * x) / s13, rc)

    lower = min(-40.0, a[0] - 1.0)
    value, _ = integrate.quad(_integrand, lower, a[0], epsabs=1e-15, epsrel=1e-13, limit=500)
    return value


def _oracle_price_three_asset_max_call(
    spots: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    dividend_yields: "np.ndarray",
    volatilities: "np.ndarray",
    correlation: "np.ndarray",
) -> "np.ndarray":
    s = np.asarray(spots, dtype=float)
    q = np.asarray(dividend_yields, dtype=float)
    v = np.asarray(volatilities, dtype=float)
    c = np.asarray(correlation, dtype=float)
    if s.ndim < 1 or s.shape[-1] != 3 or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("spots must be a finite, strictly positive array whose last axis has length 3.")
    strike, tau, rate = float(strike), float(tau), float(rate)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("dividend_yields must be a finite array of shape (3,).")
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("volatilities must be a finite, strictly positive array of shape (3,).")
    if (c.shape != (3, 3) or not np.all(np.isfinite(c)) or not np.allclose(c, c.T, rtol=0.0, atol=1e-12)
            or not np.allclose(np.diag(c), 1.0, rtol=0.0, atol=1e-12) or np.min(np.linalg.eigvalsh(c)) <= 1e-12):
        raise ValueError("correlation must be a finite, symmetric, positive definite (3, 3) matrix with unit diagonal.")

    flat = s.reshape(-1, 3)
    if tau == 0.0:
        return np.maximum(flat.max(axis=1) - strike, 0.0).reshape(s.shape[:-1])

    root = np.sqrt(tau)
    pair = np.sqrt(np.maximum(v[:, None] ** 2 + v[None, :] ** 2 - 2.0 * c * np.outer(v, v), 0.0))
    prices = np.empty(flat.shape[0])
    for n, x in enumerate(flat):
        total = 0.0
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            upper = np.array([
                (np.log(x[i] / strike) + (rate - q[i] + 0.5 * v[i] ** 2) * tau) / (v[i] * root),
                (np.log(x[i] / x[j]) + (q[j] - q[i] + 0.5 * pair[i, j] ** 2) * tau) / (pair[i, j] * root),
                (np.log(x[i] / x[k]) + (q[k] - q[i] + 0.5 * pair[i, k] ** 2) * tau) / (pair[i, k] * root),
            ])
            corr = np.eye(3)
            corr[0, 1] = corr[1, 0] = (v[i] - c[i, j] * v[j]) / pair[i, j]
            corr[0, 2] = corr[2, 0] = (v[i] - c[i, k] * v[k]) / pair[i, k]
            corr[1, 2] = corr[2, 1] = (v[i] ** 2 - c[i, j] * v[i] * v[j] - c[i, k] * v[i] * v[k]
                                       + c[j, k] * v[j] * v[k]) / (pair[i, j] * pair[i, k])
            total += x[i] * np.exp(-q[i] * tau) * _tvn(upper, corr)
        below = -(np.log(x / strike) + (rate - q - 0.5 * v ** 2) * tau) / (v * root)
        total -= strike * np.exp(-rate * tau) * (1.0 - _tvn(below, c))
        prices[n] = total
    return prices.reshape(s.shape[:-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    codes = (
        "import numpy as np\n"
        "eye = np.eye(3)\n"
        "q = np.array([0.1, 0.1, 0.1])\n"
        "v = np.array([0.2, 0.2, 0.2])\n"
        "def code(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    invalid_args = [
        "np.array([100.0, -100.0, 100.0]), 100.0, 1.0, 0.05, q, v, eye",
        "np.array([100.0, 100.0]), 100.0, 1.0, 0.05, q, v, eye",
        "np.full(3, 100.0), 0.0, 1.0, 0.05, q, v, eye",
        "np.full(3, 100.0), 100.0, -1.0, 0.05, q, v, eye",
        "np.full(3, 100.0), 100.0, 1.0, 0.05, q, np.array([0.2, 0.0, 0.2]), eye",
        "np.full(3, 100.0), 100.0, 1.0, 0.05, q, v, np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])",
        "np.full(3, 100.0), 100.0, 1.0, 0.05, q, v, np.array([[1.0, 0.2, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])",
    ]
    return [
        # Three-asset benchmark: independent assets, S = K = 100, three years, 10% dividends, 20% volatility.
        {
            "setup": "import numpy as np\n",
            "call": "np.round(price_three_asset_max_call(np.full(3, 100.0), 100.0, 3.0, 0.05, np.full(3, 0.10), np.full(3, 0.20), np.eye(3)), 10)",
            "gold_call": "np.round(_oracle_price_three_asset_max_call(np.full(3, 100.0), 100.0, 3.0, 0.05, np.full(3, 0.10), np.full(3, 0.20), np.eye(3)), 10)",
            "tol": 1e-8,
        },
        # Several price vectors with unequal dividends and volatilities and a correlation matrix with mixed signs.
        {
            "setup": ("import numpy as np\n"
                      "spots = np.array([[95.0, 100.0, 108.0], [80.0, 85.0, 120.0], [130.0, 70.0, 95.0]])\n"
                      "corr = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]])\n"),
            "call": "np.round(price_three_asset_max_call(spots, 100.0, 1.0, 0.04, np.array([0.02, 0.0, 0.05]), np.array([0.25, 0.30, 0.20]), corr), 10)",
            "gold_call": "np.round(_oracle_price_three_asset_max_call(spots.copy(), 100.0, 1.0, 0.04, np.array([0.02, 0.0, 0.05]), np.array([0.25, 0.30, 0.20]), corr.copy()), 10)",
            "tol": 1e-8,
        },
        # Strongly correlated assets with similar volatilities, strike above the spots, half a year.
        {
            "setup": ("import numpy as np\n"
                      "corr = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])\n"),
            "call": "np.round(price_three_asset_max_call(np.array([100.0, 98.0, 103.0]), 105.0, 0.5, 0.03, np.array([0.01, 0.02, 0.0]), np.array([0.20, 0.22, 0.18]), corr), 10)",
            "gold_call": "np.round(_oracle_price_three_asset_max_call(np.array([100.0, 98.0, 103.0]), 105.0, 0.5, 0.03, np.array([0.01, 0.02, 0.0]), np.array([0.20, 0.22, 0.18]), corr.copy()), 10)",
            "tol": 1e-8,
        },
        # Negatively correlated assets, deep in and out of the money on a 2 x 2 grid of price vectors.
        {
            "setup": ("import numpy as np\n"
                      "spots = np.array([[[60.0, 55.0, 70.0], [150.0, 90.0, 80.0]], [[100.0, 100.0, 100.0], [85.0, 140.0, 120.0]]])\n"
                      "corr = np.array([[1.0, -0.4, -0.2], [-0.4, 1.0, -0.1], [-0.2, -0.1, 1.0]])\n"),
            "call": "np.round(price_three_asset_max_call(spots, 100.0, 2.0, 0.05, np.array([0.03, 0.03, 0.03]), np.array([0.35, 0.15, 0.25]), corr), 10)",
            "gold_call": "np.round(_oracle_price_three_asset_max_call(spots.copy(), 100.0, 2.0, 0.05, np.array([0.03, 0.03, 0.03]), np.array([0.35, 0.15, 0.25]), corr.copy()), 10)",
            "tol": 1e-8,
        },
        # Boundary: zero time to expiry returns the payoff.
        {
            "setup": "import numpy as np\nspots = np.array([[90.0, 95.0, 99.0], [90.0, 125.0, 110.0]])\n",
            "call": "np.round(price_three_asset_max_call(spots, 100.0, 0.0, 0.05, np.full(3, 0.1), np.full(3, 0.2), np.eye(3)), 10)",
            "gold_call": "np.round(_oracle_price_three_asset_max_call(spots.copy(), 100.0, 0.0, 0.05, np.full(3, 0.1), np.full(3, 0.2), np.eye(3)), 10)",
            "tol": 1e-8,
        },
        # Edge: very short expiry, where the price approaches the discounted-forward intrinsic value.
        {
            "setup": "import numpy as np\nspots = np.array([[120.0, 90.0, 100.0], [99.0, 100.0, 101.0]])\n",
            "call": "np.round(price_three_asset_max_call(spots, 100.0, 1e-3, 0.05, np.full(3, 0.1), np.full(3, 0.2), np.eye(3)), 10)",
            "gold_call": "np.round(_oracle_price_three_asset_max_call(spots.copy(), 100.0, 1e-3, 0.05, np.full(3, 0.1), np.full(3, 0.2), np.eye(3)), 10)",
            "tol": 1e-8,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: price_three_asset_max_call(%s))" % args,
            "gold_call": "code(lambda: _oracle_price_three_asset_max_call(%s))" % args,
        }
        for args in invalid_args
    ]
