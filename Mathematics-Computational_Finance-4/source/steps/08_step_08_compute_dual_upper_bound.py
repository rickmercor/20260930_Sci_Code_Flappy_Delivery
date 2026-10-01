"""
Run the complete primal-dual pipeline for the Bermudan call on the maximum of several independent assets and return the dual upper-bound estimate. Given the common spot price, strike, maturity, rate, dividend yield, volatility, number of assets (two or three), number of exercise intervals, and the sizes and seeds of the training and pricing samples, compose the earlier steps into one evaluation and construct the dual martingale within this step.

The training sample fixes the stopping rule and the independent pricing sample evaluates it: the continuation regressions fitted on the training sample are frozen, the frozen rule applied to the pricing sample gives the primal lower bound, and the same rule defines the realised value process of every pricing path at every date.



The dual needs a martingale close to the martingale part of the discounted option value, obtained without nested simulation. The single-projection (alpha) construction takes, for each step from one date to the next, one common regressand, the realised value at the later date expressed in time-zero money, and projects it by least squares over the pricing paths onto two information sets: the basis at the later date and the basis at the earlier date. The difference of the two fitted values is the martingale increment. The basis functions are strongly collinear, so the fits must use an orthogonal-decomposition least-squares solver such as numpy.linalg.lstsq; solving the normal equations loses the required accuracy. At time zero every path shares the same state, so the projection onto the time-zero information is the sample mean of the regressand. The martingale starts at zero and stays in time-zero money.



Each pricing path's dual value is the largest, over the exercise dates, of its time-zero-discounted payoff minus the martingale; the dual upper bound is the mean of these values and the primal-dual gap is the dual bound minus the primal bound. For any martingale that starts at zero this mean bounds the option value from above. The European value of the same call on the maximum of all the assets, without early exercise and with expiry T at the common spot, is reported alongside the bounds as the no-early-exercise benchmark: the true Bermudan value is at least this European value.

Returns
-------
tuple (discounted_exercise_payoffs (n_pricing_paths, n_steps + 1) float ndarray, dual_path_values (n_pricing_paths,) float ndarray, dual_upper_bound float, primal_dual_gap float, european_price float): time-zero-discounted payoffs of the pricing sample, pathwise dual values, their mean, the dual minus primal bound, and the time-zero European price of the call on the maximum of all the assets
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_dual_upper_bound(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_assets: int,
    n_steps: int,
    n_training_paths: int,
    training_seed: int,
    n_pricing_paths: int,
    pricing_seed: int,
) -> "tuple[np.ndarray, np.ndarray, float, float, float]":
    """Primal-dual evaluation of a Bermudan max-call on independent assets.

    Parameters
    ----------
    spot : float
        Common initial price of every asset.
    strike : float
        Strike K.
    maturity : float
        Maturity T in years; exercise dates are t_j = j T / n_steps,
        j = 1, ..., n_steps.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Common continuous dividend yield.
    volatility : float
        Common volatility.
    n_assets : int
        Number of independent assets, two or three.
    n_steps : int
        Number of equal exercise intervals.
    n_training_paths, n_pricing_paths : int
        Sizes of the training and the independent pricing samples.
    training_seed, pricing_seed : int
        Seeds of the two samples; each sample is simulated with the draw
        layout of simulate_gbm_paths.

    Returns
    -------
    discounted_exercise_payoffs : np.ndarray
        Shape (n_pricing_paths, n_steps + 1), each pricing path's payoff at
        every date discounted to time zero.
    dual_path_values : np.ndarray
        Shape (n_pricing_paths,), each path's largest discounted payoff minus
        martingale over the exercise dates j = 1, ..., n_steps.
    dual_upper_bound : float
        Mean of dual_path_values.
    primal_dual_gap : float
        dual_upper_bound minus the primal lower bound of the frozen policy.
    european_price : float
        Time-zero price of the European call on the maximum of all n_assets
        assets (no early exercise), each at the common spot with the common
        dividend yield and volatility, independent, with expiry maturity.

    Raises
    ------
    ValueError
        If spot, strike, maturity or volatility is not finite and positive,
        rate or dividend_yield is not finite, n_assets is not the integer 2 or
        3, n_steps, n_training_paths or n_pricing_paths is not a
        positive integer, or a seed is not a non-negative integer.
    """
    return discounted_exercise_payoffs, dual_path_values, dual_upper_bound, primal_dual_gap, european_price

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _alpha_martingale(basis, backward_values, maturity, rate):
    """Single-projection alpha martingale in time-zero money, shape (n_paths, n_steps + 1)."""
    b = np.asarray(basis, dtype=float)
    v = np.asarray(backward_values, dtype=float)
    n_paths, n_times, _ = b.shape
    n_steps = n_times - 1
    dt = float(maturity) / n_steps
    increments = np.empty((n_paths, n_steps))
    for j in range(n_steps):
        y = np.exp(-float(rate) * (j + 1) * dt) * v[:, j + 1]
        later = b[:, j + 1, :] @ np.linalg.lstsq(b[:, j + 1, :], y, rcond=None)[0]
        if j == 0:
            earlier = np.mean(y)
        else:
            earlier = b[:, j, :] @ np.linalg.lstsq(b[:, j, :], y, rcond=None)[0]
        increments[:, j] = later - earlier
    return np.concatenate((np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)), axis=1)


def _oracle_compute_dual_upper_bound(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_assets: int,
    n_steps: int,
    n_training_paths: int,
    training_seed: int,
    n_pricing_paths: int,
    pricing_seed: int,
) -> "tuple[np.ndarray, np.ndarray, float, float, float]":
    for name, value in (("spot", spot), ("strike", strike), ("maturity", maturity), ("volatility", volatility)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (("rate", rate), ("dividend_yield", dividend_yield)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if isinstance(n_assets, bool) or not isinstance(n_assets, (int, np.integer)) or n_assets not in (2, 3):
        raise ValueError("n_assets must be the integer 2 or 3.")
    for name, value in (("n_steps", n_steps), ("n_training_paths", n_training_paths), ("n_pricing_paths", n_pricing_paths)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("training_seed", training_seed), ("pricing_seed", pricing_seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    training_paths = _oracle_simulate_gbm_paths(
        spot, maturity, rate, dividend_yield, volatility, n_steps, n_training_paths, n_assets, training_seed)
    training_payoffs, training_basis = _oracle_build_max_call_payoffs_and_basis(
        training_paths, strike, maturity, rate, dividend_yield, volatility)
    coefficients, _, _ = _oracle_fit_backward_primal(training_payoffs, training_basis, maturity, rate)

    pricing_paths = _oracle_simulate_gbm_paths(
        spot, maturity, rate, dividend_yield, volatility, n_steps, n_pricing_paths, n_assets, pricing_seed)
    payoffs, basis = _oracle_build_max_call_payoffs_and_basis(
        pricing_paths, strike, maturity, rate, dividend_yield, volatility)
    _, _, primal_lower_bound = _oracle_apply_frozen_primal_policy(payoffs, basis, coefficients, maturity, rate)
    backward_values, _ = _oracle_build_backward_discounted_process(payoffs, basis, coefficients, maturity, rate)
    martingale = _alpha_martingale(basis, backward_values, maturity, rate)

    discount = np.exp(-float(rate) * float(maturity) / int(n_steps) * np.arange(int(n_steps) + 1))
    discounted_payoffs = payoffs * discount[None, :]
    dual_path_values = np.max(discounted_payoffs[:, 1:] - martingale[:, 1:], axis=1)
    dual_upper_bound = float(np.mean(dual_path_values))
    if int(n_assets) == 2:
        european_price = float(_oracle_price_two_asset_max_call(
            spot, spot, strike, maturity, rate, dividend_yield, dividend_yield, volatility, volatility, 0.0))
    else:
        european_price = float(_oracle_price_three_asset_max_call(
            np.full(3, float(spot)), strike, maturity, rate, np.full(3, float(dividend_yield)),
            np.full(3, float(volatility)), np.eye(3)))
    return (discounted_payoffs, dual_path_values, dual_upper_bound,
            float(dual_upper_bound - primal_lower_bound), european_price)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = (
        "import numpy as np\n"
        "def pack(out):\n"
        "    z, u, upper, gap, euro = out\n"
        "    return np.concatenate((np.asarray(z, dtype=np.float64).ravel(), np.asarray(u, dtype=np.float64).ravel(), [float(upper), float(gap), float(euro)]))\n"
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
        "-100.0, 100.0, 3.0, 0.05, 0.1, 0.2, 3, 9, 500, 1, 500, 2",
        "100.0, 100.0, 3.0, 0.05, 0.1, 0.0, 3, 9, 500, 1, 500, 2",
        "100.0, 100.0, 3.0, 0.05, 0.1, 0.2, 1, 9, 500, 1, 500, 2",
        "100.0, 100.0, 3.0, 0.05, 0.1, 0.2, 4, 9, 500, 1, 500, 2",
        "100.0, 100.0, 3.0, 0.05, 0.1, 0.2, 3, 0, 500, 1, 500, 2",
        "100.0, 100.0, 3.0, 0.05, 0.1, 0.2, 3, 9, 500, 1.5, 500, 2",
    ]
    return [
        # Benchmark contract on three assets with smaller samples: full outputs.
        {
            "setup": pack,
            "call": "pack(compute_dual_upper_bound(100.0, 100.0, 3.0, 0.05, 0.10, 0.20, 3, 9, 3000, 314159, 2500, 271828))",
            "gold_call": "pack(_oracle_compute_dual_upper_bound(100.0, 100.0, 3.0, 0.05, 0.10, 0.20, 3, 9, 3000, 314159, 2500, 271828))",
            "tol": 1e-7,
        },
        # Two assets, out-of-the-money start with fewer exercise dates.
        {
            "setup": pack,
            "call": "pack(compute_dual_upper_bound(90.0, 100.0, 1.5, 0.04, 0.08, 0.25, 2, 6, 2000, 11, 1800, 12))",
            "gold_call": "pack(_oracle_compute_dual_upper_bound(90.0, 100.0, 1.5, 0.04, 0.08, 0.25, 2, 6, 2000, 11, 1800, 12))",
            "tol": 1e-7,
        },
        # Three assets, in-the-money start without dividends.
        {
            "setup": pack,
            "call": "pack(compute_dual_upper_bound(110.0, 100.0, 1.0, 0.03, 0.0, 0.30, 3, 4, 1500, 7, 1500, 8))",
            "gold_call": "pack(_oracle_compute_dual_upper_bound(110.0, 100.0, 1.0, 0.03, 0.0, 0.30, 3, 4, 1500, 7, 1500, 8))",
            "tol": 1e-7,
        },
        # Boundary: one exercise interval, so the martingale has a single increment built from the time-zero sample mean.
        {
            "setup": pack,
            "call": "pack(compute_dual_upper_bound(100.0, 100.0, 1.0, 0.05, 0.10, 0.20, 2, 1, 400, 3, 400, 4))",
            "gold_call": "pack(_oracle_compute_dual_upper_bound(100.0, 100.0, 1.0, 0.05, 0.10, 0.20, 2, 1, 400, 3, 400, 4))",
            "tol": 1e-7,
        },
        # Zero rate and zero dividend on two assets: discounting is the identity, so the martingale units are tested directly.
        {
            "setup": pack,
            "call": "pack(compute_dual_upper_bound(100.0, 95.0, 2.0, 0.0, 0.0, 0.25, 2, 3, 800, 21, 700, 22))",
            "gold_call": "pack(_oracle_compute_dual_upper_bound(100.0, 95.0, 2.0, 0.0, 0.0, 0.25, 2, 3, 800, 21, 700, 22))",
            "tol": 1e-7,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: compute_dual_upper_bound(%s))" % args,
            "gold_call": "code(lambda: _oracle_compute_dual_upper_bound(%s))" % args,
        }
        for args in invalid_args
    ]
