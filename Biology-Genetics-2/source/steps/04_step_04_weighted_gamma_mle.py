"""
Fit the shape and rate of a gamma density to a weighted sample by maximum

likelihood, solving the shape equation by bisection on a fixed bracket.

The per-tree observation is a positive, right-skewed quantity, and a gamma

density with shape a and rate b is the natural two-parameter family for it. The

same fit is needed three times over the course of the analysis: once for the

bulk of the genome-wide distribution, once for the flagged upper tail, and once

per iteration of expectation maximisation, where the membership weights are

posterior state probabilities rather than zeros and ones. Maximising the weighted

log-likelihood leaves the rate in closed form once the shape is known, and

reduces the shape to a one-dimensional root problem whose left side decreases

monotonically, so the root is unique whenever the weighted sample has dispersion

in the log domain and there is no finite maximum-likelihood shape when it does

not. Bisection on a fixed bracket then reaches machine precision in a fixed

number of halvings without depending on a starting guess, which is what makes the

fit reproducible across all three of its uses.

Returns
-------
tuple of two native Python floats, the gamma shape and the gamma rate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weighted_gamma_mle(
    observations: np.ndarray,
    weights: np.ndarray,
    bracket: tuple = (1e-6, 1e6),
    n_bisect: int = 200,
) -> tuple:
    """Return the maximum-likelihood gamma shape and rate of a weighted sample.

    Parameters
    ----------
    observations : np.ndarray
        Strictly positive observations of shape (m,).
    weights : np.ndarray
        Non-negative membership weights of shape (m,).
    bracket : tuple
        Lower and upper bound of the bisection bracket for the shape.
    n_bisect : int
        Number of bisection halvings.

    Returns
    -------
    result : tuple
        The maximum-likelihood shape followed by the maximum-likelihood rate.

    Raises
    ------
    ValueError
        If observations or weights are not one dimensional or have different
        lengths, if any observation is not strictly positive, if any weight is
        negative, if the weights sum to zero, if n_bisect is not positive, if the
        bracket is not an increasing pair of positive numbers, or if the weighted
        sample has no dispersion in the log domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import digamma


def _oracle_weighted_gamma_mle(
    observations: np.ndarray,
    weights: np.ndarray,
    bracket: tuple = (1e-6, 1e6),
    n_bisect: int = 200,
) -> tuple:
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if observations.ndim != 1 or weights.ndim != 1:
        raise ValueError("observations and weights must be one dimensional")
    if observations.size != weights.size:
        raise ValueError("observations and weights must have the same length")
    if np.any(observations <= 0.0):
        raise ValueError("observations must be strictly positive")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    if int(n_bisect) < 1:
        raise ValueError("n_bisect must be a positive integer")
    low, high = float(bracket[0]), float(bracket[1])
    if not (0.0 < low < high):
        raise ValueError("bracket must be an increasing pair of positive numbers")

    mean = float((weights * observations).sum() / total)
    mean_log = float((weights * np.log(observations)).sum() / total)
    s = np.log(mean) - mean_log
    if s <= 0.0:
        raise ValueError("the weighted sample has no dispersion in the log domain")

    def _shape_equation(a):
        return np.log(a) - digamma(a) - s

    for _ in range(int(n_bisect)):
        mid = 0.5 * (low + high)
        if _shape_equation(mid) > 0.0:
            low = mid
        else:
            high = mid
    shape = 0.5 * (low + high)
    rate = shape / mean
    return float(shape), float(rate)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_OBS = np.array([
    22100.0, 19500.0, 22000.0, 15200.0, 13800.0, 11300.0, 23500.0, 19200.0,
    14100.0, 6000.0, 8500.0, 8100.0, 6200.0, 20500.0, 138300.0, 148500.0,
    134400.0, 142800.0, 46000.0, 18300.0,
])
_TAIL_INDICATOR = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
])
'''
    return [
        # --- Normal scenario: the bulk of the distribution ---
        {
            "setup": fixture + """observations = _OBS.copy()
weights = 1.0 - _TAIL_INDICATOR
""",
            "call": "list(weighted_gamma_mle(observations, weights))",
            "gold_call": "list(_oracle_weighted_gamma_mle(observations, weights))",
        },
        # --- Boundary case: fractional weights concentrated on the upper tail ---
        {
            "setup": fixture + """observations = _OBS.copy()
weights = 0.02 + 0.96 * _TAIL_INDICATOR
""",
            "call": "list(weighted_gamma_mle(observations, weights))",
            "gold_call": "list(_oracle_weighted_gamma_mle(observations, weights))",
        },
        # --- Edge case: a sample with no dispersion has no finite shape ---
        {
            "setup": fixture + """observations = np.full(6, 7500.0)
weights = np.ones(6)
def run_model():
    try:
        weighted_gamma_mle(observations, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_weighted_gamma_mle(observations, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
