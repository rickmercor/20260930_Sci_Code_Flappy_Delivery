"""
Return the normalized, exponentially reweighted Gamma mixture.

An exponential reweighting changes both component probabilities and component rates.

For original probabilities $p_m$, shapes $k_m$, rates $b_m$, and tilt $\theta$, define



$$

a_m=p_m\left(\frac{b_m}{b_m-\theta}\right)^{k_m},\qquad

p_m^\theta=\frac{a_m}{\sum_\ell a_\ell},\qquad b_m^\theta=b_m-\theta.

$$



The event intensity is unchanged. The output rows retain the input component order;

each row contains probability, shape, and rate. Shapes are dimensionless, rates and

$\theta$ have inverse-loss units. All original probabilities must be positive and

sum to one within $10^{-12}$; no inactive components are included.

Returns
-------
Shape (M, 3), columns probability, shape, and tilted rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tilt_mark_distribution(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    theta: float,
) -> np.ndarray:
    r"""Return the normalized, exponentially reweighted Gamma mixture.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (M,), positive component probabilities.
    shapes : np.ndarray
        Shape (M,), positive Gamma shapes.
    rates : np.ndarray
        Shape (M,), positive original rates in inverse loss units.
    theta : float
        Finite tilt, strictly below every original rate.

    Returns
    -------
    mixture : np.ndarray
        Shape (M, 3), columns probability, shape, and tilted rate.

    Raises
    ------
    ValueError
        If arrays are empty, non-real, non-finite, differently shaped, nonpositive, or
        unnormalized, or theta is inadmissible.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _real_array(value, name):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _real_scalar(value, name):
    result = _real_array(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(result)


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_mixture(probabilities, shapes, rates):
    arrays = [
        _real_array(x, name)
        for x, name in zip(
            (probabilities, shapes, rates), ("probabilities", "shapes", "rates")
        )
    ]
    if any(x.ndim != 1 or x.size == 0 or np.any(x <= 0) for x in arrays):
        raise ValueError("mixture inputs must be positive nonempty vectors")
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("mixture vectors must have equal shapes")
    if abs(np.sum(arrays[0]) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")
    return arrays


def _checked_market(market):
    market = _real_array(market, "market")
    if market.shape != (5,):
        raise ValueError("market must contain kappa, baseline, initial, beta, rate")
    if market[0] <= 0 or np.any(market[1:] < 0):
        raise ValueError("kappa must be positive; other market entries nonnegative")
    return market


def _oracle_tilt_mark_distribution(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    theta: float,
) -> np.ndarray:
    p, k, b = _checked_mixture(probabilities, shapes, rates)
    theta = _real_scalar(theta, "theta")
    if theta >= np.min(b):
        raise ValueError("theta must be below every rate")
    log_weights = np.log(p) + k * (np.log(b) - np.log(b - theta))
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    if np.any(weights <= 0):
        raise ValueError("tilted probabilities underflow")
    return np.column_stack((weights, k, b - theta))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
theta = 0.2
""",
            "call": "tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)",
            "gold_call": "_oracle_tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)",
        },
        {
            "setup": """import numpy as np
p = np.array([1.0])
k = np.array([1.0])
b = np.array([2.0])
theta = 0.0
""",
            "call": "tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)",
            "gold_call": "_oracle_tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)",
        },
        {
            "setup": """import numpy as np
p = np.array([0.25, 0.75])
k = np.array([0.5, 3.0])
b = np.array([1.0, 5.0])
theta = -0.4
""",
            "call": "tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)",
            "gold_call": "_oracle_tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)",
        },
        {
            "setup": """import numpy as np
p = np.array([0.6, 0.4])
k = np.array([2.0, 6.0])
b = np.array([4.0, 2.5])
theta = 2.5

def _model_exception():
    try:
        tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_tilt_mark_distribution(p.copy(), k.copy(), b.copy(), theta)
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
