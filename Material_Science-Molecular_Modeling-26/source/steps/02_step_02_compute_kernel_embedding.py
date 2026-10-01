"""
Evaluate Matern-1/2 kernel means for one or many query points over the full collective-variable range or over an ordered collection of subintervals.

Bayesian quadrature places a Gaussian-process prior on the integrand and then integrates that prior analytically, so every quantity the method reports is built from integrals of the covariance function rather than from the integrand itself. The first of these is the kernel mean, the covariance between the value of the integrand at one point and the value of the integral over the whole range. It is what converts a Gaussian process over the free-energy gradient into a Gaussian distribution over the free energy, and it appears in the posterior mean of the integral, in its posterior variance, and in the acquisition function that chooses the next window.




The covariance here is Matern-1/2, that is the exponential kernel of the given process variance and lengthscale; it corresponds to a nowhere-differentiable prior sample path, which is the appropriate regularity assumption for a nucleation mean force that changes abruptly near the stable states. Quadrature uses the unnormalised Lebesgue measure on the finite interval, so the embedding returned here is the plain integral of the covariance between the query point and every point of the integration interval, not an expectation under a probability density; the normalized alternative would rescale every value by the width of the range.




Two properties fix what a correct implementation must produce. The evaluation point need not lie inside the integration interval, and the two situations give different one-sided behaviour, so an expression derived only for an interior query point is not sufficient. And because integration is additive over a partition, the values returned for the rows of any non-overlapping partition of the range must sum to the value returned for the whole range, at every query point. The multi-interval form is what lets a free-energy path be assembled from local increments: rows identify integration intervals and columns identify GP query points.

Returns
-------
float or np.ndarray: kernel means with scalar/full-range calls collapsed and explicit interval/query axes retained as documented.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_kernel_embedding(query: np.ndarray, lower: float = 0.0, upper: float = 288.0,
                             variance: float = 0.25, lengthscale: float = 20.0,
                             intervals: np.ndarray = None) -> np.ndarray:
    """Evaluate exponential-kernel means for queries and integration intervals.

    Parameters
    ----------
    query : float or np.ndarray
        Scalar query or non-empty one-dimensional array of query points. All
        entries must lie in the closed global domain [lower, upper].
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    intervals : np.ndarray, optional
        Array of shape (m, 2) whose rows are ordered integration bounds
        [a_j, b_j] inside the global domain, with a_j < b_j. If omitted, the
        single interval [lower, upper] is used without retaining an interval
        axis in the result.

    Raises
    ------
    ValueError
        If a scalar parameter is not finite, query is not a scalar or a
        non-empty one-dimensional numeric array, upper is not greater than
        lower, variance or lengthscale is non-positive, a query lies outside
        [lower, upper], or intervals is not a finite (m, 2) array of strictly
        ordered bounds contained in that domain.

    Returns
    -------
    embedding : float or np.ndarray
        With intervals omitted, a native float for a scalar query or an array
        of shape (n,) for n queries. With intervals supplied, an array of
        shape (m,) for a scalar query or (m, n) for n queries; rows retain the
        input interval order and columns retain the query order.
    """
    return embedding  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_boolean(value) -> bool:
    """Return True for a boolean scalar or a numpy boolean."""
    import numpy as np

    return isinstance(value, (bool, np.bool_))


def _oracle_compute_kernel_embedding(query: np.ndarray, lower: float = 0.0,
                                     upper: float = 288.0, variance: float = 0.25,
                                     lengthscale: float = 20.0,
                                     intervals: np.ndarray = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    lower = float(lower)
    upper = float(upper)
    variance = float(variance)
    lengthscale = float(lengthscale)

    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if variance <= 0.0:
        raise ValueError("variance must be > 0")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0")
    if _is_boolean(query):
        raise ValueError("query must contain real numbers")
    try:
        queries = np.asarray(query, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("query must contain real numbers")
    scalar_query = queries.ndim == 0
    if queries.ndim > 1 or queries.size == 0:
        raise ValueError("query must be scalar or a non-empty one-dimensional array")
    queries = queries.reshape(-1)
    if not np.all(np.isfinite(queries)):
        raise ValueError("query must contain only finite entries")
    if np.any(queries < lower) or np.any(queries > upper):
        raise ValueError("every query must lie inside the integration range")

    explicit_intervals = intervals is not None
    if intervals is None:
        bounds = np.array([[lower, upper]], dtype=float)
    else:
        try:
            bounds = np.asarray(intervals, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("intervals must contain real numbers")
        if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
            raise ValueError("intervals must have shape (m, 2) with m >= 1")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("intervals must contain only finite entries")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("every interval must have a lower bound below its upper bound")
        if np.any(bounds[:, 0] < lower) or np.any(bounds[:, 1] > upper):
            raise ValueError("every interval must lie inside the integration range")

    a = bounds[:, 0, None]
    b = bounds[:, 1, None]
    q = queries[None, :]
    span_factor = -np.expm1(-(b - a) / lengthscale)
    left_value = (np.exp(-(a - q) / lengthscale) * span_factor)
    right_value = (np.exp(-(q - b) / lengthscale) * span_factor)
    inside_value = (-np.expm1(-(q - a) / lengthscale)
                    - np.expm1(-(b - q) / lengthscale))
    dimensionless = np.where(q < a, left_value,
                             np.where(q > b, right_value, inside_value))
    result = variance * lengthscale * dimensionless

    if not explicit_intervals:
        result = result[0]
        if scalar_query:
            return float(result[0])
        return result
    if scalar_query:
        return result[:, 0]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: an interior point many lengthscales from both ends (normal scenario) ---
        {
            "setup": """import numpy as np
query = 142.545454545454547
""",
            "call": "compute_kernel_embedding(query)",
            "gold_call": "_oracle_compute_kernel_embedding(query)",
        },
        # --- Valid: a point one lengthscale from the lower end, where the correction bites ---
        {
            "setup": """import numpy as np
query = 20.0
""",
            "call": "compute_kernel_embedding(query)",
            "gold_call": "_oracle_compute_kernel_embedding(query)",
        },
        # --- Boundary: the lower end of the range itself ---
        {
            "setup": """import numpy as np
query = 0.0
""",
            "call": "compute_kernel_embedding(query)",
            "gold_call": "_oracle_compute_kernel_embedding(query)",
        },
        # --- Boundary: the upper end of the range itself ---
        {
            "setup": """import numpy as np
query = 288.0
""",
            "call": "compute_kernel_embedding(query)",
            "gold_call": "_oracle_compute_kernel_embedding(query)",
        },
        # --- Edge: a range short compared with the lengthscale, so nothing saturates ---
        {
            "setup": """import numpy as np
query = 1.25
lower = -2.0
upper = 3.5
variance = 1.75
lengthscale = 40.0
""",
            "call": "compute_kernel_embedding(query, lower, upper, variance, lengthscale)",
            "gold_call": "_oracle_compute_kernel_embedding(query, lower, upper, variance, lengthscale)",
        },
        # --- Valid: a vector of queries on the full benchmark range ---
        {
            "setup": """import numpy as np
queries = np.array([0.0, 20.0, 144.0, 288.0])
""",
            "call": "compute_kernel_embedding(queries)",
            "gold_call": "_oracle_compute_kernel_embedding(queries)",
        },
        # --- Valid: one query viewed from intervals to its left, around it and to its right ---
        {
            "setup": """import numpy as np
intervals = np.array([[0.0, 25.0], [25.0, 90.0], [90.0, 288.0]])
query = 70.0
""",
            "call": "compute_kernel_embedding(query, intervals=intervals)",
            "gold_call": "_oracle_compute_kernel_embedding(query, intervals=intervals)",
        },
        # --- Valid: the full interval-by-query kernel-mean matrix ---
        {
            "setup": """import numpy as np
intervals = np.array([[0.0, 17.0], [17.0, 111.0], [111.0, 288.0]])
queries = np.array([1.6, 52.0, 142.5, 285.0])
""",
            "call": "compute_kernel_embedding(queries, intervals=intervals)",
            "gold_call": "_oracle_compute_kernel_embedding(queries, intervals=intervals)",
        },
        # --- Boundary/additivity: partition rows sum to the full-range embedding ---
        {
            "setup": """import numpy as np
cuts = np.array([0.0, 31.0, 90.0, 173.0, 288.0])
intervals = np.column_stack((cuts[:-1], cuts[1:]))
queries = np.array([0.0, 31.0, 144.0, 288.0])
""",
            "call": "np.concatenate((compute_kernel_embedding(queries, intervals=intervals).sum(axis=0), compute_kernel_embedding(queries)))",
            "gold_call": "np.concatenate((_oracle_compute_kernel_embedding(queries, intervals=intervals).sum(axis=0), _oracle_compute_kernel_embedding(queries)))",
        },
        # --- Invalid: a query outside the integration range ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_kernel_embedding(300.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kernel_embedding(300.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: malformed interval bounds ---
        {
            "setup": """import numpy as np
intervals = np.array([[0.0, 20.0], [40.0, 30.0]])
def run_model():
    try:
        compute_kernel_embedding(np.array([10.0, 50.0]), intervals=intervals)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kernel_embedding(np.array([10.0, 50.0]), intervals=intervals)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive lengthscale ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_kernel_embedding(100.0, 0.0, 288.0, 0.25, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kernel_embedding(100.0, 0.0, 288.0, 0.25, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an inverted integration range ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_kernel_embedding(10.0, 288.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kernel_embedding(10.0, 288.0, 0.0)
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
