"""
Price a European call on the maximum of two assets exactly. Given the two current prices, the strike, the time to expiry, the risk-free rate, each asset's dividend yield and volatility and the correlation between the assets, return the arbitrage-free price of the payoff max(max(S1, S2) - K, 0) at expiry.

The two assets follow correlated geometric Brownian motions under the risk-neutral measure, each with its own continuous dividend yield and volatility, and the price is the discounted risk-neutral expectation of the payoff. This expectation has an exact value, and the Bermudan regression later uses it as a basis feature, so it must be correct to double precision; a Monte Carlo or coarse quadrature estimate is not adequate. At zero time to expiry the price equals the payoff itself.

Returns
-------
np.ndarray of prices with the broadcast shape of s1 and s2 (a 0-d array when both are scalars)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def price_two_asset_max_call(
    s1: "np.ndarray",
    s2: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
) -> "np.ndarray":
    """European call on max(S1, S2) with strike K and time to expiry tau.

    Parameters
    ----------
    s1, s2 : np.ndarray
        Current prices of the two assets, strictly positive; scalars or
        arrays that broadcast together.
    strike : float
        Strike K, strictly positive.
    tau : float
        Time to expiry in years, non-negative.
    rate : float
        Continuously compounded risk-free rate.
    q1, q2 : float
        Continuous dividend yields of the two assets.
    sigma1, sigma2 : float
        Volatilities of the two assets, strictly positive.
    rho : float
        Correlation between the two assets' Brownian motions, in (-1, 1).

    Returns
    -------
    price : np.ndarray
        Float64 prices with the broadcast shape of s1 and s2. For tau = 0
        the price is max(max(s1, s2) - strike, 0).

    Raises
    ------
    ValueError
        If s1 and s2 do not broadcast together, any price is not finite and
        positive, strike is not finite and positive, tau is not finite and
        non-negative, rate, q1 or q2 is not finite, a volatility is not
        finite and positive, or rho is not in the open interval (-1, 1).
    """
    return price

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special


def _bivariate_normal_cdf(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho, via Owen's T function."""
    h, k = np.broadcast_arrays(np.atleast_1d(np.asarray(h, dtype=float)), np.atleast_1d(np.asarray(k, dtype=float)))
    s = np.sqrt(1.0 - rho * rho)
    out = np.empty(h.shape)
    both_zero = (h == 0.0) & (k == 0.0)
    out[both_zero] = 0.25 + np.arcsin(rho) / (2.0 * np.pi)
    m = ~both_zero
    hh, kk = h[m], k[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        ah = np.where(hh != 0.0, (kk - rho * hh) / (hh * s), np.copysign(np.inf, kk))
        ak = np.where(kk != 0.0, (hh - rho * kk) / (kk * s), np.copysign(np.inf, hh))
    beta = np.where((hh * kk < 0.0) | ((hh * kk == 0.0) & (hh + kk < 0.0)), 0.5, 0.0)
    out[m] = (0.5 * special.ndtr(hh) + 0.5 * special.ndtr(kk)
              - special.owens_t(hh, ah) - special.owens_t(kk, ak) - beta)
    return out


def _oracle_price_two_asset_max_call(
    s1: "np.ndarray",
    s2: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
) -> "np.ndarray":
    a1 = np.asarray(s1, dtype=float)
    a2 = np.asarray(s2, dtype=float)
    try:
        a1, a2 = np.broadcast_arrays(a1, a2)
    except ValueError as exc:
        raise ValueError("s1 and s2 must broadcast together.") from exc
    if not (np.all(np.isfinite(a1)) and np.all(np.isfinite(a2))) or np.any(a1 <= 0.0) or np.any(a2 <= 0.0):
        raise ValueError("Asset prices must be finite and positive.")
    strike, tau, rate = float(strike), float(tau), float(rate)
    q1, q2, sigma1, sigma2, rho = float(q1), float(q2), float(sigma1), float(sigma2), float(rho)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative.")
    if not (np.isfinite(rate) and np.isfinite(q1) and np.isfinite(q2)):
        raise ValueError("rate and dividend yields must be finite.")
    if not (np.isfinite(sigma1) and np.isfinite(sigma2)) or sigma1 <= 0.0 or sigma2 <= 0.0:
        raise ValueError("volatilities must be finite and positive.")
    if not np.isfinite(rho) or rho <= -1.0 or rho >= 1.0:
        raise ValueError("rho must lie in (-1, 1).")

    if tau == 0.0:
        return np.maximum(np.maximum(a1, a2) - strike, 0.0)

    b1, b2 = rate - q1, rate - q2
    sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2.0 * rho * sigma1 * sigma2)
    root = np.sqrt(tau)
    d = (np.log(a1 / a2) + (b1 - b2 + 0.5 * sigma * sigma) * tau) / (sigma * root)
    y1 = (np.log(a1 / strike) + (b1 + 0.5 * sigma1 * sigma1) * tau) / (sigma1 * root)
    y2 = (np.log(a2 / strike) + (b2 + 0.5 * sigma2 * sigma2) * tau) / (sigma2 * root)
    rho1 = (sigma1 - rho * sigma2) / sigma
    rho2 = (sigma2 - rho * sigma1) / sigma
    shape = a1.shape
    price = (a1.ravel() * np.exp((b1 - rate) * tau) * _bivariate_normal_cdf(y1.ravel(), d.ravel(), rho1)
             + a2.ravel() * np.exp((b2 - rate) * tau) * _bivariate_normal_cdf(y2.ravel(), -d.ravel() + sigma * root, rho2)
             - strike * np.exp(-rate * tau)
             * (1.0 - _bivariate_normal_cdf(-y1.ravel() + sigma1 * root, -y2.ravel() + sigma2 * root, rho)))
    return price.reshape(shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    codes = (
        "import numpy as np\n"
        "def code(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    invalid_args = [
        "-1.0, 100.0, 100.0, 1.0, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0",
        "100.0, 100.0, 0.0, 1.0, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0",
        "100.0, 100.0, 100.0, -1.0, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0",
        "100.0, 100.0, 100.0, 1.0, 0.05, 0.1, 0.1, 0.0, 0.2, 0.0",
        "100.0, 100.0, 100.0, 1.0, 0.05, 0.1, 0.1, 0.2, 0.2, 1.0",
        "np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 1.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0",
    ]
    return [
        # At-the-money benchmark point: S1 = S2 = K = 100, three years, dividends, independent assets.
        {
            "setup": "import numpy as np\n",
            "call": "np.round(price_two_asset_max_call(100.0, 100.0, 100.0, 3.0, 0.05, 0.10, 0.10, 0.20, 0.20, 0.0), 10)",
            "gold_call": "np.round(_oracle_price_two_asset_max_call(100.0, 100.0, 100.0, 3.0, 0.05, 0.10, 0.10, 0.20, 0.20, 0.0), 10)",
            "tol": 1e-8,
        },
        # Array inputs across moneyness, with the lower price in either position.
        {
            "setup": "import numpy as np\ns1 = np.array([90.0, 110.0, 80.0, 130.0])\ns2 = np.array([110.0, 90.0, 80.0, 70.0])\n",
            "call": "np.round(price_two_asset_max_call(s1, s2, 100.0, 1.0, 0.05, 0.10, 0.10, 0.20, 0.20, 0.0), 10)",
            "gold_call": "np.round(_oracle_price_two_asset_max_call(s1.copy(), s2.copy(), 100.0, 1.0, 0.05, 0.10, 0.10, 0.20, 0.20, 0.0), 10)",
            "tol": 1e-8,
        },
        # Correlated assets with unequal dividends and volatilities.
        {
            "setup": "import numpy as np\n",
            "call": "np.round(price_two_asset_max_call(np.array([95.0, 80.0]), np.array([105.0, 130.0]), 100.0, 2.0, 0.03, 0.0, 0.02, 0.30, 0.25, 0.4), 10)",
            "gold_call": "np.round(_oracle_price_two_asset_max_call(np.array([95.0, 80.0]), np.array([105.0, 130.0]), 100.0, 2.0, 0.03, 0.0, 0.02, 0.30, 0.25, 0.4), 10)",
            "tol": 1e-8,
        },
        # Boundary: zero time to expiry returns the payoff.
        {
            "setup": "import numpy as np\ns = np.array([90.0, 100.0, 120.0])\n",
            "call": "np.round(price_two_asset_max_call(s, s[::-1], 100.0, 0.0, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0), 10)",
            "gold_call": "np.round(_oracle_price_two_asset_max_call(s.copy(), s[::-1].copy(), 100.0, 0.0, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0), 10)",
            "tol": 1e-8,
        },
        # Edge: very short expiry, close to intrinsic value.
        {
            "setup": "import numpy as np\ns = np.array([90.0, 100.0, 120.0])\n",
            "call": "np.round(price_two_asset_max_call(s, s[::-1], 100.0, 1e-4, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0), 10)",
            "gold_call": "np.round(_oracle_price_two_asset_max_call(s.copy(), s[::-1].copy(), 100.0, 1e-4, 0.05, 0.1, 0.1, 0.2, 0.2, 0.0), 10)",
            "tol": 1e-8,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: price_two_asset_max_call(%s))" % args,
            "gold_call": "code(lambda: _oracle_price_two_asset_max_call(%s))" % args,
        }
        for args in invalid_args
    ]
