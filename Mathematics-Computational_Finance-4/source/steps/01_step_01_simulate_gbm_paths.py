"""
Simulate independent risk-neutral geometric Brownian motion paths for several assets on an equally spaced time grid. Given a common spot price, maturity, risk-free rate, dividend yield, volatility, number of time steps, number of paths, number of assets and a random seed, return the full price array including the initial prices.

Under the risk-neutral measure with a continuous dividend yield q, each asset follows dS = (r - q) S dt + sigma S dW, and the exact one-step update on a grid of spacing dt = T / n_steps is S_{t+dt} = S_t exp[(r - q - sigma^2 / 2) dt + sigma sqrt(dt) Z] with Z standard normal. The assets are independent, so each receives its own standard-normal innovations.



The draw layout is part of the specification because it fixes which normal variate drives which asset at which step. All innovations are drawn in one call as numpy.random.default_rng(seed).standard_normal((n_paths, n_steps, n_assets)), so entry [n, j, i] drives asset i of path n over step j.

Returns
-------
np.ndarray of shape (n_paths, n_steps + 1, n_assets): simulated prices, with column 0 equal to the spot price for every path and asset
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_steps: int,
    n_paths: int,
    n_assets: int,
    seed: int,
) -> "np.ndarray":
    """Simulate independent risk-neutral GBM paths for several assets.

    Parameters
    ----------
    spot : float
        Common initial price of every asset, strictly positive.
    maturity : float
        Horizon T in years, strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuous dividend yield q of every asset.
    volatility : float
        Common volatility sigma, non-negative.
    n_steps : int
        Number of equal time steps, at least one.
    n_paths : int
        Number of simulated paths, at least one.
    n_assets : int
        Number of independent assets, at least one.
    seed : int
        Seed for numpy.random.default_rng.

    Returns
    -------
    paths : np.ndarray
        Float64 array of shape (n_paths, n_steps + 1, n_assets). Innovations
        are drawn as default_rng(seed).standard_normal((n_paths, n_steps,
        n_assets)).

    Raises
    ------
    ValueError
        If spot or maturity is not finite and positive, rate or
        dividend_yield is not finite, volatility is not finite and
        non-negative, or n_steps, n_paths or n_assets is not an integer >= 1,
        or seed is not an integer.
    """
    return paths

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_steps: int,
    n_paths: int,
    n_assets: int,
    seed: int,
) -> "np.ndarray":
    for name, value in (("n_steps", n_steps), ("n_paths", n_paths), ("n_assets", n_assets)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    spot, maturity = float(spot), float(maturity)
    rate, dividend_yield, volatility = float(rate), float(dividend_yield), float(volatility)
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive.")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not (np.isfinite(rate) and np.isfinite(dividend_yield)):
        raise ValueError("rate and dividend_yield must be finite.")
    if not np.isfinite(volatility) or volatility < 0.0:
        raise ValueError("volatility must be finite and non-negative.")

    dt = maturity / int(n_steps)
    z = np.random.default_rng(int(seed)).standard_normal((int(n_paths), int(n_steps), int(n_assets)))
    increments = (rate - dividend_yield - 0.5 * volatility * volatility) * dt + volatility * np.sqrt(dt) * z
    log_paths = np.concatenate((np.zeros((int(n_paths), 1, int(n_assets))), np.cumsum(increments, axis=1)), axis=1)
    return spot * np.exp(log_paths)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    anchors = (
        "import numpy as np\n"
        "def anchors(fn):\n"
        "    p = fn(100.0, 3.0, 0.05, 0.10, 0.20, 9, 50000, 2, 314159)\n"
        "    idx = [(0, 1, 0), (0, 9, 1), (7, 4, 0), (123, 6, 1), (25000, 2, 0), (49999, 9, 1)]\n"
        "    return np.array([float(p.shape[0]), float(p.shape[1]), float(p.shape[2])]\n"
        "                    + [p[i, j, k] for i, j, k in idx])\n"
    )
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
        "-100.0, 3.0, 0.05, 0.1, 0.2, 9, 10, 2, 1",
        "100.0, 0.0, 0.05, 0.1, 0.2, 9, 10, 2, 1",
        "100.0, 3.0, 0.05, 0.1, -0.2, 9, 10, 2, 1",
        "100.0, 3.0, 0.05, 0.1, 0.2, 0, 10, 2, 1",
        "100.0, 3.0, 0.05, 0.1, 0.2, 9, 10, 0, 1",
        "100.0, 3.0, 0.05, 0.1, 0.2, 9, 10, 2, 1.5",
    ]
    return [
        # Small case: full array for two assets, three paths, two steps.
        {
            "setup": "import numpy as np\n",
            "call": "np.round(simulate_gbm_paths(100.0, 1.0, 0.05, 0.02, 0.3, 2, 3, 2, 42), 10)",
            "gold_call": "np.round(_oracle_simulate_gbm_paths(100.0, 1.0, 0.05, 0.02, 0.3, 2, 3, 2, 42), 10)",
            "tol": 1e-8,
        },
        # Benchmark sample: shape and representative entries of the 50,000-path training sample.
        {
            "setup": anchors,
            "call": "anchors(simulate_gbm_paths)",
            "gold_call": "anchors(_oracle_simulate_gbm_paths)",
            "tol": 1e-8,
        },
        # Zero volatility: deterministic growth at rate r - q.
        {
            "setup": "import numpy as np\n",
            "call": "np.round(simulate_gbm_paths(90.0, 3.0, 0.05, 0.10, 0.0, 3, 2, 3, 7), 10)",
            "gold_call": "np.round(_oracle_simulate_gbm_paths(90.0, 3.0, 0.05, 0.10, 0.0, 3, 2, 3, 7), 10)",
            "tol": 1e-8,
        },
        # Single asset, single step.
        {
            "setup": "import numpy as np\n",
            "call": "np.round(simulate_gbm_paths(50.0, 0.25, 0.01, 0.0, 0.4, 1, 4, 1, 2026), 10)",
            "gold_call": "np.round(_oracle_simulate_gbm_paths(50.0, 0.25, 0.01, 0.0, 0.4, 1, 4, 1, 2026), 10)",
            "tol": 1e-8,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: simulate_gbm_paths(%s))" % args,
            "gold_call": "code(lambda: _oracle_simulate_gbm_paths(%s))" % args,
        }
        for args in invalid_args
    ]
