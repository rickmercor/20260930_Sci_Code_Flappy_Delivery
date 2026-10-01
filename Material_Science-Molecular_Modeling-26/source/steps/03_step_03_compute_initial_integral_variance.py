"""
Evaluate the prior covariance of one full-range free-energy integral or of an ordered collection of subinterval integrals under the Matern-1/2 process.

The prior variance of the integral is the covariance function integrated twice against the integration measure. More generally, integrating over two different subintervals gives their prior cross-covariance. Keeping those cross terms is essential when a full free-energy difference is assembled by adding local increments: the variance of the sum is the sum of every entry of the increment covariance matrix, not merely the sum of its diagonal. Computing the matrix in closed form rather than by grid quadrature keeps the convergence diagnostic free of reconstruction-grid error.




The quantity wanted here is therefore a double integral of the same Matern-1/2 covariance used elsewhere in the pipeline, taken under the same unnormalised Lebesgue measure: the entry for a pair of intervals is the covariance function integrated once over each of them. One implementation route evaluates that double integral in closed form; the interval geometry is not restricted, so whatever route is taken must cover disjoint, touching, overlapping and nested pairs alike and must return a symmetric positive-semidefinite matrix.




Three checks pin a correct result. Summing every entry of the matrix for a non-overlapping partition of the range must reproduce the variance of the integral over the whole range, since the variance of a sum of increments includes the cross terms. On a single interval two regimes are visible: an interval long compared with the lengthscale gives variance growing linearly in its width, while a short one approaches the fully correlated limit in which the variance is the process variance times the square of the width. And no pairwise entry may exceed the geometric mean of the two corresponding diagonal entries.

Returns
-------
float or np.ndarray: full-range prior variance, or the ordered covariance matrix of requested subinterval integrals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_initial_integral_variance(lower: float = 0.0, upper: float = 288.0,
                                      variance: float = 0.25,
                                      lengthscale: float = 20.0,
                                      intervals: np.ndarray = None) -> np.ndarray:
    """Evaluate prior covariances between free-energy-gradient integrals.

    Parameters
    ----------
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    intervals : np.ndarray, optional
        Array of shape (m, 2) containing ordered subinterval bounds inside
        [lower, upper]. Intervals may be disjoint, touching, overlapping or
        nested, and their input order is retained. If omitted, compute only
        the variance of the full-range integral.

    Raises
    ------
    ValueError
        If a scalar parameter is not finite, upper is not greater than lower,
        variance or lengthscale is non-positive, or intervals is not a finite
        non-empty array of shape (m, 2) with strictly ordered bounds contained
        in [lower, upper].

    Returns
    -------
    prior_variance : float or np.ndarray
        A native float when intervals is omitted. Otherwise, the symmetric
        array of shape (m, m) whose (i, j) entry is the prior covariance
        between the integrals over intervals i and j.
    """
    return prior_variance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _oracle_compute_initial_integral_variance(lower: float = 0.0, upper: float = 288.0,
                                              variance: float = 0.25,
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

    if intervals is None:
        span = upper - lower
        return float(2.0 * variance * lengthscale
                     * (span + lengthscale * np.expm1(-span / lengthscale)))

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

    a = bounds[:, 0]
    b = bounds[:, 1]

    def _second_primitive(delta):
        distance = np.abs(np.asarray(delta, dtype=np.longdouble))
        ell = np.longdouble(lengthscale)
        amplitude = np.longdouble(variance)
        return amplitude * ell * ell * (distance / ell + np.exp(-distance / ell))

    result = (_second_primitive(b[:, None] - a[None, :])
              - _second_primitive(a[:, None] - a[None, :])
              - _second_primitive(b[:, None] - b[None, :])
              + _second_primitive(a[:, None] - b[None, :]))
    result = np.asarray(result, dtype=float)
    return 0.5 * (result + result.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark range, many lengthscales wide (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_initial_integral_variance()",
            "gold_call": "_oracle_compute_initial_integral_variance()",
        },
        # --- Valid: the same range modelled with a longer lengthscale ---
        {
            "setup": """import numpy as np
lengthscale = 60.0
""",
            "call": "compute_initial_integral_variance(0.0, 288.0, 0.25, lengthscale)",
            "gold_call": "_oracle_compute_initial_integral_variance(0.0, 288.0, 0.25, lengthscale)",
        },
        # --- Boundary: an interval exactly one lengthscale wide ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_initial_integral_variance(0.0, 20.0, 0.25, 20.0)",
            "gold_call": "_oracle_compute_initial_integral_variance(0.0, 20.0, 0.25, 20.0)",
        },
        # --- Edge: a range far shorter than the lengthscale, the fully correlated limit ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_initial_integral_variance(-1.0, 1.0, 3.0, 5000.0)",
            "gold_call": "_oracle_compute_initial_integral_variance(-1.0, 1.0, 3.0, 5000.0)",
            "tol": 1e-7,
        },
        # --- Valid: covariance matrix for disjoint, overlapping and nested increments ---
        {
            "setup": """import numpy as np
intervals = np.array([[0.0, 70.0], [70.0, 140.0],
                      [35.0, 190.0], [90.0, 120.0]])
""",
            "call": "compute_initial_integral_variance(intervals=intervals)",
            "gold_call": "_oracle_compute_initial_integral_variance(intervals=intervals)",
        },
        # --- Boundary/additivity: a partition covariance recovers the full-range variance ---
        {
            "setup": """import numpy as np
cuts = np.array([0.0, 31.0, 90.0, 173.0, 288.0])
intervals = np.column_stack((cuts[:-1], cuts[1:]))
""",
            "call": "np.array([compute_initial_integral_variance(intervals=intervals).sum(), compute_initial_integral_variance()])",
            "gold_call": "np.array([_oracle_compute_initial_integral_variance(intervals=intervals).sum(), _oracle_compute_initial_integral_variance()])",
        },
        # --- Edge: input order and repeated or nested intervals are retained ---
        {
            "setup": """import numpy as np
intervals = np.array([[120.0, 220.0], [5.0, 15.0],
                      [120.0, 220.0], [0.0, 288.0]])
""",
            "call": "compute_initial_integral_variance(0.0, 288.0, 0.7, 11.0, intervals)",
            "gold_call": "_oracle_compute_initial_integral_variance(0.0, 288.0, 0.7, 11.0, intervals)",
        },
        # --- Invalid: a degenerate range of zero width ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_initial_integral_variance(5.0, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_initial_integral_variance(5.0, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive process variance ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_initial_integral_variance(0.0, 288.0, -0.25, 20.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_initial_integral_variance(0.0, 288.0, -0.25, 20.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: one subinterval extends beyond the declared domain ---
        {
            "setup": """import numpy as np
intervals = np.array([[0.0, 80.0], [200.0, 300.0]])
def run_model():
    try:
        compute_initial_integral_variance(intervals=intervals)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_initial_integral_variance(intervals=intervals)
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
