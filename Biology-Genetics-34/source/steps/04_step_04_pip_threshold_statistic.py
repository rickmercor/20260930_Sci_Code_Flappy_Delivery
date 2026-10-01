"""
Invert the posterior inclusion probability of a single variant to find the chi-square statistic at which a given PIP threshold is first reached.

Summing the posterior odds of the non-null components against the null gives PIP = S / (1 + S) with S = sum_k A_k * exp(B_k * z), where z is the one-degree-of-freedom chi-square statistic of the variant, so the PIP rises monotonically from the prior odds against the null at z = 0 towards one. A threshold on the probability is therefore a single threshold on the statistic, found by bracketing the crossing and bisecting on it, with the sum accumulated in log space because B_k * z reaches several hundred for strong signals.

Returns
-------
float: the chi-square statistic at which the PIP threshold is reached, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pip_threshold_statistic(curve_constants: np.ndarray, alpha: float) -> float:
    """Find the chi-square statistic at which the PIP threshold is reached.

    Parameters
    ----------
    curve_constants : np.ndarray
        Shape ``(n_components, 2)`` array with the amplitude in column 0 and
        the rate in column 1, one row per non-null mixture component.
    alpha : float
        Posterior inclusion probability threshold (0 < alpha < 1).

    Returns
    -------
    z_threshold : float
        Smallest non-negative chi-square statistic whose posterior inclusion
        probability reaches ``alpha``, as a native Python float.

    Raises
    ------
    ValueError
        If ``curve_constants`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if ``alpha`` is not a
        real number strictly between 0 and 1; or if the threshold is not
        reached by a chi-square statistic of 1e9, which no admissible set of
        constants should require.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pip_threshold_statistic(curve_constants: np.ndarray, alpha: float) -> float:
    import numpy as np

    constants = np.asarray(curve_constants, dtype=float)
    if constants.ndim != 2 or constants.shape[1] != 2 or constants.shape[0] < 1:
        raise ValueError("curve_constants must have shape (n_components, 2)")
    if not np.all(np.isfinite(constants)) or np.any(constants <= 0.0):
        raise ValueError("curve_constants must be finite and > 0")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError("alpha must be a real number")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    log_amplitude = np.log(constants[:, 0])
    rate = constants[:, 1]

    # Work with the log posterior odds: the threshold PIP = S / (1 + S) is met
    # exactly when log S reaches log(alpha / (1 - alpha)), and the odds
    # themselves would overflow long before the statistic runs out of range.
    log_target = float(np.log(alpha / (1.0 - alpha)))

    def _log_odds_gap(z):
        terms = log_amplitude + rate * z
        shift = float(np.max(terms))
        return shift + float(np.log(np.sum(np.exp(terms - shift)))) - log_target

    # A variant with no evidence at all already carries the prior odds, so a
    # threshold below that is met immediately.
    if _log_odds_gap(0.0) >= 0.0:
        return 0.0

    # Double the upper end until the crossing is bracketed. The rates are
    # positive, so this terminates unless the constants are degenerate.
    lo = 0.0
    hi = 1.0
    while _log_odds_gap(hi) < 0.0:
        hi *= 2.0
        if hi > 1.0e9:
            raise ValueError("the PIP threshold is not reachable for these constants")

    # The gap is strictly increasing, so bisection converges to the crossing;
    # 200 halvings take the bracket well below double precision.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _log_odds_gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: with a single non-null component the inversion is
        # closed form, z = log(alpha / ((1 - alpha) * A)) / B, so both
        # crossings can be written down without the oracle. Probing two
        # thresholds rejects a solver that returns a bracket end, and rejects
        # an implementation that inverts 1 - PIP instead of PIP.
        {
            "setup": """import numpy as np
constants = np.array([[2.5e-4, 0.45]])
def exact(a):
    return float(np.log(a / ((1.0 - a) * 2.5e-4)) / 0.45)
EXPECTED = float(exact(0.9) + 10.0 * exact(0.5))
""",
            "call": ("float(pip_threshold_statistic(constants, 0.9)"
                     " + 10.0 * pip_threshold_statistic(constants, 0.5))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component curve at the usual reporting threshold,
        # where every component contributes to the sum ---
        {
            "setup": """import numpy as np
constants = np.array([[2.5e-3, 0.4100], [7.0e-5, 0.4700],
                      [1.4e-6, 0.4950], [2.0e-8, 0.4990]])
""",
            "call": "pip_threshold_statistic(constants, 0.9)",
            "gold_call": "_oracle_pip_threshold_statistic(constants, 0.9)",
        },
        # --- Boundary: a threshold below the prior odds against the null, where
        # no single-variant evidence at all is required ---
        {
            "setup": """import numpy as np
constants = np.array([[0.6, 0.3], [0.2, 0.45]])
""",
            "call": "pip_threshold_statistic(constants, 0.4)",
            "gold_call": "_oracle_pip_threshold_statistic(constants, 0.4)",
        },
        # --- Edge: an extremely stringent threshold on a very sparse prior,
        # which pushes the crossing far out and would overflow a direct sum ---
        {
            "setup": """import numpy as np
constants = np.array([[4.0e-9, 0.4999], [1.0e-11, 0.5]])
""",
            "call": "pip_threshold_statistic(constants, 0.999)",
            "gold_call": "_oracle_pip_threshold_statistic(constants, 0.999)",
        },
        # --- Invalid: a threshold of one, which is never attained ---
        {
            "setup": """import numpy as np
constants = np.array([[1.0e-3, 0.5]])
def run_model():
    try:
        pip_threshold_statistic(constants, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pip_threshold_statistic(constants, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative amplitude among the constants ---
        {
            "setup": """import numpy as np
constants = np.array([[-1.0e-3, 0.5], [1.0e-5, 0.49]])
def run_model():
    try:
        pip_threshold_statistic(constants, 0.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pip_threshold_statistic(constants, 0.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
