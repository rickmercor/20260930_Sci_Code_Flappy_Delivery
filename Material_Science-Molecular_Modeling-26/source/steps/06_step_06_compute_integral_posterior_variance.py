"""
Combine steps 02 and 03 with the observation covariance to evaluate the full-range posterior integral variance or the posterior covariance matrix of local free-energy increments.

The posterior variance of the integral is what makes this a probabilistic numerical method rather than a quadrature rule: it reports how much uncertainty about the free-energy difference survives the windows already run, and it is the quantity the sample-placement strategy is designed to reduce. For several local increments, the same conditioning rule acts on the complete prior interval-covariance matrix. Its off-diagonal entries cannot be discarded: conditioned increments remain correlated, and the posterior variance of their sum is the sum of every matrix entry. None of these quantities depends on the measured force values, only on the window locations and their noise levels.




What this step computes is the prior interval covariance supplied by the earlier step, reduced by the part of it that the existing noisy observations already explain. Both delegated ingredients are required for the intervals asked for here: the prior covariance of those interval integrals, and the kernel means relating each interval integral to each measured window. The result must remain symmetric and positive-semidefinite, must never exceed the corresponding prior entry, and must approach the prior as the observation noise grows without bound.




Used as a stopping rule the diagnostic must be made dimensionless, since the variance carries the square of the units of the free energy and scales with the width of the range. Comparing the posterior standard deviation with the magnitude of the current integral estimate gives a scale-free convergence measure; a run that exhausts its budget before that measure falls below tolerance is budget-limited rather than converged, which is an outcome worth being able to detect.

Returns
-------
float or np.ndarray: full-range posterior integral variance, or the posterior covariance matrix of requested subinterval integrals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_integral_posterior_variance(centers: np.ndarray, lower: float = 0.0,
                                        upper: float = 288.0, variance: float = 0.25,
                                        lengthscale: float = 20.0,
                                        noise: np.ndarray = 1.0e-3,
                                        intervals: np.ndarray = None) -> np.ndarray:
    """Evaluate posterior covariance of one or many integral functionals.

    Obtain kernel means from ``compute_kernel_embedding`` and prior integral
    covariance from ``compute_initial_integral_variance`` rather than
    recreating either previous step.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance or an array of shape
        (n,) with one variance per window.
    intervals : np.ndarray, optional
        Array of shape (m, 2) with ordered integration intervals inside the
        domain. If omitted, compute the scalar full-range variance.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, upper is not greater than lower,
        variance or lengthscale is non-positive, noise cannot be broadcast to
        shape (n,) or contains a negative/non-finite entry, centers is empty
        or non-finite, a centre lies outside the range, intervals is invalid,
        a delegated prior/embedding result has the wrong shape, or the
        observation covariance is singular.

    Returns
    -------
    integral_variance : float or np.ndarray
        A native float when intervals is omitted. Otherwise, the symmetric
        posterior covariance matrix of shape (m, m), preserving interval
        order.
    """
    return integral_variance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _oracle_compute_integral_posterior_variance(centers: np.ndarray, lower: float = 0.0,
                                                upper: float = 288.0, variance: float = 0.25,
                                                lengthscale: float = 20.0,
                                                noise: np.ndarray = 1.0e-3,
                                                intervals: np.ndarray = None) -> np.ndarray:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper), ("variance", variance),
                        ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    centers = np.asarray(centers, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    if noise_array.ndim == 0:
        noise_array = np.full(centers.size, float(noise_array))
    elif noise_array.shape != centers.shape:
        raise ValueError("noise must be scalar or have shape (n,)")
    if not np.all(np.isfinite(noise_array)) or np.any(noise_array < 0.0):
        raise ValueError("noise variances must be finite and non-negative")

    embedding = np.asarray(_oracle_compute_kernel_embedding(
        centers, float(lower), float(upper), float(variance),
        float(lengthscale), intervals), dtype=float)
    prior = np.asarray(_oracle_compute_initial_integral_variance(
        float(lower), float(upper), float(variance),
        float(lengthscale), intervals), dtype=float)

    if intervals is None:
        if embedding.shape != centers.shape or prior.ndim != 0:
            raise ValueError("a delegated scalar integral result has the wrong shape")
    else:
        try:
            interval_count = np.asarray(intervals).shape[0]
        except (AttributeError, IndexError):
            raise ValueError("intervals is invalid")
        if (embedding.shape != (interval_count, centers.size)
                or prior.shape != (interval_count, interval_count)):
            raise ValueError("a delegated interval result has the wrong shape")
    if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(prior)):
        raise ValueError("a delegated integral result is non-finite")

    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_array)

    try:
        if intervals is None:
            explained = float(embedding @ np.linalg.solve(gram, embedding))
        else:
            explained = embedding @ np.linalg.solve(gram, embedding.T)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    result = prior - explained
    if intervals is None:
        return float(result)
    result = np.asarray(result, dtype=float)
    return 0.5 * (result + result.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the four initialisation windows of the benchmark (normal scenario) ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
""",
            "call": "compute_integral_posterior_variance(centers)",
            "gold_call": "_oracle_compute_integral_posterior_variance(centers)",
        },
        # --- Valid: a denser thirteen-window design spanning the same range ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 20.363636363636363, 37.818181818181820,
                    72.727272727272734, 107.636363636363640, 142.545454545454547,
                    177.454545454545453, 212.363636363636374, 247.272727272727280,
                    264.727272727272748, 284.16, 285.0])
""",
            "call": "compute_integral_posterior_variance(centers)",
            "gold_call": "_oracle_compute_integral_posterior_variance(centers)",
        },
        # --- Boundary: one window only, so the reduction is a single rank-one term ---
        {
            "setup": """import numpy as np
centers = np.array([144.0])
""",
            "call": "compute_integral_posterior_variance(centers)",
            "gold_call": "_oracle_compute_integral_posterior_variance(centers)",
        },
        # --- Edge: a dense low-noise design on a range shorter than the lengthscale ---
        {
            "setup": """import numpy as np
centers = np.linspace(0.0, 10.0, 6)
""",
            "call": "compute_integral_posterior_variance(centers, 0.0, 10.0, 1.0, 4.0, 1.0e-6)",
            "gold_call": "_oracle_compute_integral_posterior_variance(centers, 0.0, 10.0, 1.0, 4.0, 1.0e-6)",
        },
        # --- Valid: posterior covariance for disjoint, overlapping and nested increments ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 37.8, 107.6, 177.5, 247.3, 285.0])
intervals = np.array([[0.0, 70.0], [70.0, 140.0],
                      [35.0, 190.0], [90.0, 120.0]])
""",
            "call": "compute_integral_posterior_variance(centers, intervals=intervals)",
            "gold_call": "_oracle_compute_integral_posterior_variance(centers, intervals=intervals)",
        },
        # --- Boundary/additivity: partition covariance recovers full posterior variance ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.7, 142.5, 212.4, 284.16, 285.0])
cuts = np.array([0.0, 31.0, 90.0, 173.0, 288.0])
intervals = np.column_stack((cuts[:-1], cuts[1:]))
""",
            "call": "np.array([compute_integral_posterior_variance(centers, intervals=intervals).sum(), compute_integral_posterior_variance(centers)])",
            "gold_call": "np.array([_oracle_compute_integral_posterior_variance(centers, intervals=intervals).sum(), _oracle_compute_integral_posterior_variance(centers)])",
        },
        # --- Edge: heteroscedastic noise propagates into every covariance entry ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 18.0, 55.0, 97.0, 144.0])
noise = np.array([1.0e-5, 2.0e-3, 5.0e-4, 4.0e-3, 1.0e-4])
intervals = np.array([[0.0, 50.0], [50.0, 100.0], [100.0, 144.0]])
""",
            "call": "compute_integral_posterior_variance(centers, 0.0, 144.0, 0.6, 17.0, noise, intervals)",
            "gold_call": "_oracle_compute_integral_posterior_variance(centers, 0.0, 144.0, 0.6, 17.0, noise, intervals)",
        },
        # --- Invalid: a window centre outside the integration range ---
        {
            "setup": """import numpy as np
centers = np.array([-5.0, 100.0])
def run_model():
    try:
        compute_integral_posterior_variance(centers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_integral_posterior_variance(centers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an empty design ---
        {
            "setup": """import numpy as np
centers = np.array([])
def run_model():
    try:
        compute_integral_posterior_variance(centers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_integral_posterior_variance(centers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: heteroscedastic noise does not match the window count ---
        {
            "setup": """import numpy as np
centers = np.array([10.0, 100.0, 200.0])
noise = np.array([1.0e-3, 2.0e-3])
def run_model():
    try:
        compute_integral_posterior_variance(centers, noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_integral_posterior_variance(centers, noise=noise)
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
