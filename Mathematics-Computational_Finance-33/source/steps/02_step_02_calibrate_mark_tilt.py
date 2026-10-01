"""
Determine the mark tilt from a discounted cumulative-loss swap.

Let $m(\theta)$ be the tilted mean mark and $a=\kappa-\beta m(\theta)>0$.

The first moment of intensity obeys



$$

\frac{d}{dt}\mathbb E[\lambda_t]=\kappa\bar\lambda-a\mathbb E[\lambda_t].

$$



Writing $\lambda_\infty=\kappa\bar\lambda/a$, a swap settled at $T_s$ has value



$$

S(\theta)=e^{-rT_s}m(\theta)\left[\lambda_\infty T_s+

(\lambda_0-\lambda_\infty)\frac{1-e^{-aT_s}}{a}\right].

$$



Calibrate on the untruncated process, using a closed interval with admissible,

subcritical endpoints and a bracketed quote. Both endpoint quotes are allowed.

Require $\bar\lambda+\lambda_0>0$, so the expected accumulated increment is

strictly increasing with the tilted mean and calibration is unique.

The numerical root must have absolute error below $10^{-11}$ in inverse loss units.

Returns
-------
Calibrated mark tilt in inverse loss units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_mark_tilt(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap_maturity: float,
    swap_price: float,
    bracket: np.ndarray,
) -> float:
    r"""Determine the mark tilt from a discounted cumulative-loss swap.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (M,), normalized positive probabilities.
    shapes : np.ndarray
        Shape (M,), positive Gamma shapes.
    rates : np.ndarray
        Shape (M,), positive rates.
    market : np.ndarray
        Shape (5,), ordered kappa, baseline intensity, initial intensity, excitation
        beta, interest rate; kappa > 0, others nonnegative, and baseline plus initial
        intensity positive.
    swap_maturity : float
        Positive settlement time in years.
    swap_price : float
        Nonnegative discounted price in loss-payment currency units.
    bracket : np.ndarray
        Shape (2,), strictly increasing finite tilt endpoints; every endpoint satisfies
        theta < min(rates) and kappa > beta times tilted mean.

    Returns
    -------
    theta : float
        Calibrated mark tilt in inverse loss units.

    Raises
    ------
    ValueError
        If mixture or market inputs violate their stated domains, the bracket is
        inadmissible or non-subcritical, or the quote is not bracketed.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_calibrate_mark_tilt(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap_maturity: float,
    swap_price: float,
    bracket: np.ndarray,
) -> float:
    market = _checked_market(market)
    if market[1] + market[2] <= 0:
        raise ValueError("calibration requires positive arrival activity")
    maturity = _real_scalar(swap_maturity, "swap_maturity")
    quote = _real_scalar(swap_price, "swap_price")
    bounds = _real_array(bracket, "bracket")
    if maturity <= 0 or quote < 0 or bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError("invalid maturity, quote, or bracket")

    def _residual(theta):
        mixture = _oracle_tilt_mark_distribution(probabilities, shapes, rates, theta)
        mean = np.sum(mixture[:, 0] * mixture[:, 1] / mixture[:, 2])
        kappa, baseline, initial, beta, rate = market
        decay = kappa - beta * mean
        if decay <= 0:
            raise ValueError("calibration bracket must be subcritical")
        equilibrium = kappa * baseline / decay
        integrated = (
            equilibrium * maturity
            + (initial - equilibrium) * (-np.expm1(-decay * maturity)) / decay
        )
        return float(np.exp(-rate * maturity) * mean * integrated - quote)

    left, right = _residual(bounds[0]), _residual(bounds[1])
    if left == 0:
        return float(bounds[0])
    if right == 0:
        return float(bounds[1])
    if left * right > 0:
        raise ValueError("swap quote is outside bracket")
    return float(brentq(_residual, bounds[0], bounds[1], xtol=1e-13, rtol=1e-14))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
market = np.array([8.0, 2.0, 2.7, 1.1, 0.02])
q = 2.2
T = 0.5
bracket = np.array([0.0, 0.35])
""",
            "call": "calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
            "gold_call": "_oracle_calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
p = np.array([1.0])
k = np.array([1.0])
b = np.array([2.0])
market = np.array([2.0, 1.0, 1.0, 0.0, 0.0])
q = 0.25
T = 0.5
bracket = np.array([0.0, 0.5])
""",
            "call": "calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
            "gold_call": "_oracle_calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
p = np.array([1.0])
k = np.array([1.0])
b = np.array([2.0])
market = np.array([2.0, 1.0, 1.0, 0.0, 0.0])
q = 1 / 3
T = 0.5
bracket = np.array([0.0, 0.5])
""",
            "call": "calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
            "gold_call": "_oracle_calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
p = np.array([0.3, 0.7])
k = np.array([1.0, 2.0])
b = np.array([2.0, 4.0])
market = np.array([3.0, 1.0, 2.0, 0.5, 0.01])
q = 0.3
T = 0.4
bracket = np.array([-0.5, 0.8])
""",
            "call": "calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
            "gold_call": "_oracle_calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
p = np.array([1.0])
k = np.array([1.0])
b = np.array([2.0])
market = np.array([2.0, 1.0, 1.0, 0.0, 0.0])
q = 20.0
T = 0.5
bracket = np.array([0.0, 0.5])

def _model_exception():
    try:
        calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
p = np.array([1.0])
k = np.array([1.0])
b = np.array([2.0])
market = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
q = 0.0
T = 0.5
bracket = np.array([0.0, 0.5])

def _model_exception():
    try:
        calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_calibrate_mark_tilt(p.copy(), k.copy(), b.copy(), market.copy(), T, q, bracket.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
