"""
Build the Bermudan max-call exercise payoffs and the 16-function regression basis on simulated multi-asset paths. Given the price paths of two or more assets, strike, maturity, risk-free rate, dividend yield and volatility, return the payoff of the call on the maximum of all assets at every date and the basis evaluated at every path and date.

Regression-based pricing of a Bermudan option approximates the continuation value by a linear combination of basis functions of the current state. For a call on the maximum of several assets the basis is built from the two largest asset prices at each date, found path by path, so the features follow the assets that currently drive the payoff rather than fixed asset labels. With x1 >= x2 the largest and second-largest prices divided by the strike, the basis contains a constant, the monomials of x1 and of x2 up to fourth order, the four cross terms x1 x2, x1^2 x2, x1 x2^2 and x1^2 x2^2, and the price of the European two-asset call on the maximum with the same strike and expiry T - t_j, divided by the strike, together with its square and cube.

The European price is the most informative feature because it already carries the option-like curvature of the continuation value. It is evaluated on the two largest prices, with the assets' common dividend yield, zero correlation and the common volatility; at the final date its expiry is zero and it equals max(x1 - 1, 0). The payoff itself uses the largest of all asset prices.

Returns
-------
tuple (payoffs (n_paths, n_steps + 1) float ndarray, basis (n_paths, n_steps + 1, 16) float ndarray): max-call exercise payoffs and the 16 regression features at every path and date
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_max_call_payoffs_and_basis(
    paths: "np.ndarray",
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Max-call payoffs and the 16-function basis on paths of two or more assets.

    Parameters
    ----------
    paths : np.ndarray
        Prices of shape (n_paths, n_steps + 1, n_assets) with n_assets >= 2,
        on the equally spaced grid t_j = j * maturity / n_steps, all finite and
        positive, n_steps >= 1.
    strike : float
        Strike K, strictly positive.
    maturity : float
        Maturity T in years, strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Common continuous dividend yield of the assets.
    volatility : float
        Common volatility of the assets, strictly positive.

    Returns
    -------
    payoffs : np.ndarray
        Shape (n_paths, n_steps + 1), the largest asset price minus K, floored
        at zero.
    basis : np.ndarray
        Shape (n_paths, n_steps + 1, 16), columns in the order
        [1, x1, x1^2, x1^3, x1^4, x2, x2^2, x2^3, x2^4, x1 x2, x1^2 x2,
        x1 x2^2, x1^2 x2^2, c, c^2, c^3], where x1 >= x2 are the largest and
        second-largest prices on that path and date divided by K (equal when
        two assets tie for the maximum) and c is the European two-asset
        max-call price on those two prices with expiry maturity - t_j
        (dividend yield q for both, zero correlation, common volatility)
        divided by K.

    Raises
    ------
    ValueError
        If paths is not a finite, positive array of shape (n_paths, n_times,
        n_assets) with n_times >= 2 and n_assets >= 2, strike or maturity is
        not finite and positive, rate or dividend_yield is not finite, or
        volatility is not finite and positive.
    """
    return payoffs, basis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_max_call_payoffs_and_basis(
    paths: "np.ndarray",
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> "tuple[np.ndarray, np.ndarray]":
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 2 or x.shape[2] < 2:
        raise ValueError("paths must have shape (n_paths, n_times, n_assets) with n_times >= 2 and n_assets >= 2.")
    if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("paths must be finite and positive.")
    strike, maturity = float(strike), float(maturity)
    rate, dividend_yield, volatility = float(rate), float(dividend_yield), float(volatility)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not (np.isfinite(rate) and np.isfinite(dividend_yield)):
        raise ValueError("rate and dividend_yield must be finite.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive.")

    n_paths, n_times, _ = x.shape
    n_steps = n_times - 1
    ranked = np.sort(x, axis=2)
    top, second = ranked[:, :, -1], ranked[:, :, -2]
    payoffs = np.maximum(top - strike, 0.0)
    x1, x2 = top / strike, second / strike
    euro = np.empty((n_paths, n_times))
    for j in range(n_times):
        tau = maturity - j * maturity / n_steps if j < n_steps else 0.0
        euro[:, j] = _oracle_price_two_asset_max_call(
            top[:, j], second[:, j], strike, max(tau, 0.0), rate, dividend_yield, dividend_yield,
            volatility, volatility, 0.0) / strike
    features = [np.ones_like(x1), x1, x1 ** 2, x1 ** 3, x1 ** 4, x2, x2 ** 2, x2 ** 3, x2 ** 4,
                x1 * x2, x1 ** 2 * x2, x1 * x2 ** 2, x1 ** 2 * x2 ** 2, euro, euro ** 2, euro ** 3]
    return payoffs, np.stack(features, axis=2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = (
        "import numpy as np\n"
        "def pack(out):\n"
        "    payoffs, basis = out\n"
        "    return np.concatenate((np.asarray(payoffs, dtype=np.float64).ravel(), np.asarray(basis, dtype=np.float64).ravel()))\n"
    )
    codes = (
        "import numpy as np\n"
        "good = np.full((2, 3, 3), 100.0)\n"
        "def code(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    invalid_args = [
        "np.full((2, 3, 1), 100.0), 100.0, 3.0, 0.05, 0.1, 0.2",
        "np.full((2, 1, 3), 100.0), 100.0, 3.0, 0.05, 0.1, 0.2",
        "-good, 100.0, 3.0, 0.05, 0.1, 0.2",
        "good, 0.0, 3.0, 0.05, 0.1, 0.2",
        "good, 100.0, 0.0, 0.05, 0.1, 0.2",
        "good, 100.0, 3.0, 0.05, 0.1, 0.0",
    ]
    return [
        # Three assets on two hand-written paths over three dates; the identity of the largest and
        # second-largest asset changes between dates.
        {
            "setup": pack + "paths = np.array([[[100.0, 100.0, 100.0], [112.0, 95.0, 104.0], [90.0, 125.0, 118.0]],\n"
                            "                  [[100.0, 100.0, 100.0], [85.0, 97.0, 99.0], [101.0, 99.5, 88.0]]])\n",
            "call": "pack(build_max_call_payoffs_and_basis(paths, 100.0, 3.0, 0.05, 0.10, 0.20))",
            "gold_call": "pack(_oracle_build_max_call_payoffs_and_basis(paths.copy(), 100.0, 3.0, 0.05, 0.10, 0.20))",
            "tol": 1e-9,
        },
        # Simulated benchmark-style paths: payoffs and basis for 200 three-asset paths on nine steps.
        {
            "setup": pack + "g = np.random.default_rng(99)\n"
                            "z = g.standard_normal((200, 9, 3))\n"
                            "dt = 3.0 / 9\n"
                            "paths = 100.0 * np.exp(np.concatenate((np.zeros((200, 1, 3)), np.cumsum((0.05 - 0.10 - 0.02) * dt + 0.2 * np.sqrt(dt) * z, axis=1)), axis=1))\n",
            "call": "pack(build_max_call_payoffs_and_basis(paths, 100.0, 3.0, 0.05, 0.10, 0.20))",
            "gold_call": "pack(_oracle_build_max_call_payoffs_and_basis(paths.copy(), 100.0, 3.0, 0.05, 0.10, 0.20))",
            "tol": 1e-9,
        },
        # Four assets with permuted labels: payoffs and basis depend only on the two largest prices.
        {
            "setup": pack + "g = np.random.default_rng(5)\n"
                            "z = g.standard_normal((30, 4, 4))\n"
                            "dt = 1.0 / 4\n"
                            "paths = 90.0 * np.exp(np.concatenate((np.zeros((30, 1, 4)), np.cumsum((0.03 - 0.02 - 0.5 * 0.35 ** 2) * dt + 0.35 * np.sqrt(dt) * z, axis=1)), axis=1))\n"
                            "def permuted(fn):\n"
                            "    return pack(fn(paths[:, :, [2, 0, 3, 1]].copy(), 95.0, 1.0, 0.03, 0.02, 0.35))\n",
            "call": "permuted(build_max_call_payoffs_and_basis)",
            "gold_call": "permuted(_oracle_build_max_call_payoffs_and_basis)",
            "tol": 1e-9,
        },
        # Two assets and a single step: the European feature at the final date equals the payoff divided by the strike.
        {
            "setup": pack + "paths = np.array([[[100.0, 100.0], [130.0, 80.0]], [[100.0, 100.0], [70.0, 99.0]]])\n",
            "call": "pack(build_max_call_payoffs_and_basis(paths, 100.0, 0.5, 0.05, 0.10, 0.20))",
            "gold_call": "pack(_oracle_build_max_call_payoffs_and_basis(paths.copy(), 100.0, 0.5, 0.05, 0.10, 0.20))",
            "tol": 1e-9,
        },
        # Ties: two assets share the maximum, so the largest and second-largest prices coincide.
        {
            "setup": pack + "paths = np.array([[[100.0, 100.0, 100.0], [120.0, 120.0, 80.0], [95.0, 110.0, 110.0]]])\n",
            "call": "pack(build_max_call_payoffs_and_basis(paths, 100.0, 2.0, 0.04, 0.05, 0.25))",
            "gold_call": "pack(_oracle_build_max_call_payoffs_and_basis(paths.copy(), 100.0, 2.0, 0.04, 0.05, 0.25))",
            "tol": 1e-9,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: build_max_call_payoffs_and_basis(%s))" % args,
            "gold_call": "code(lambda: _oracle_build_max_call_payoffs_and_basis(%s))" % args,
        }
        for args in invalid_args
    ]
