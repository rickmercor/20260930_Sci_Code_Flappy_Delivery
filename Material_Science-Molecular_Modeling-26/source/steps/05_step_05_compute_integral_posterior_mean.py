"""
Use the kernel embeddings from step 02 to evaluate posterior means for the full free-energy difference or for an ordered collection of local free-energy increments.

Integrating the Gaussian-process posterior mean over the collective-variable range gives the posterior mean of the integral, which for a mean-force integrand is the free-energy difference between the two ends of the range. Because integration is linear, the result is not a quadrature over predicted values but a fixed weighted sum of the measured mean forces, with weights obtained by pushing the kernel means through the inverse observation covariance. Applying the same weights to a matrix of subinterval kernel means gives local free-energy increments whose sum over a partition is exactly the full-range posterior mean.




The consequence worth stating explicitly is that this is not a quadrature over predicted values on any grid. The posterior mean of the integral is a linear functional of the measured forces alone, and the weight each measured force carries is obtained by combining the kernel means supplied by the embedding step with the same conditioning the surrogate uses, so the delegated embedding must be requested for exactly the intervals asked for here and used as returned. No reconstruction grid appears anywhere in this step.




The zero prior mean leaves a visible signature. Each weight decays with the distance from a window to the rest of the design, so a range that is sparsely covered relative to the lengthscale contributes less than its true share and the magnitude of the estimated free-energy difference is systematically pulled towards zero. This shrinkage, not the discretisation of any grid, is the dominant systematic error of the quadrature at small sample budgets, and it relaxes as windows accumulate. A correct result also satisfies an additivity check: the values returned for a non-overlapping partition of the range must sum to the value returned for the whole range.

Returns
-------
float or np.ndarray: full-range posterior integral mean, or one posterior mean per requested subinterval.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_integral_posterior_mean(centers: np.ndarray, forces: np.ndarray,
                                    lower: float = 0.0, upper: float = 288.0,
                                    variance: float = 0.25, lengthscale: float = 20.0,
                                    noise: np.ndarray = 1.0e-3,
                                    intervals: np.ndarray = None) -> np.ndarray:
    """Estimate one full-range or several local free-energy differences.

    Obtain the required kernel means by calling the previously implemented
    ``compute_kernel_embedding`` function.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
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
        (n,) with one variance per measured window.
    intervals : np.ndarray, optional
        Array of shape (m, 2) containing ordered integration intervals inside
        [lower, upper]. If omitted, integrate once over the full range.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, upper is not greater than lower,
        variance or lengthscale is non-positive, noise cannot be broadcast to
        shape (n,) or contains a negative/non-finite entry, the window arrays
        are empty, mismatched, or non-finite, a centre lies outside the range,
        intervals is invalid, a delegated embedding has the wrong shape, or
        the observation covariance is singular.

    Returns
    -------
    integral_mean : float or np.ndarray
        A native float when intervals is omitted. Otherwise, an array of
        shape (m,) containing one posterior integral mean per input interval.
    """
    return integral_mean  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _oracle_compute_integral_posterior_mean(centers: np.ndarray, forces: np.ndarray,
                                            lower: float = 0.0, upper: float = 288.0,
                                            variance: float = 0.25, lengthscale: float = 20.0,
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
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")
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
    expected_shape = centers.shape if intervals is None else (np.asarray(intervals).shape[0], centers.size)
    if embedding.shape != expected_shape or not np.all(np.isfinite(embedding)):
        raise ValueError("the delegated kernel embedding has the wrong shape")

    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_array)

    try:
        weights = np.linalg.solve(gram, forces)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    result = embedding @ weights
    if intervals is None:
        return float(result)
    return np.asarray(result, dtype=float)

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
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_integral_posterior_mean(centers, forces)",
            "gold_call": "_oracle_compute_integral_posterior_mean(centers, forces)",
        },
        # --- Valid: a denser design covering the barrier region as well ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 37.818181818181820, 72.727272727272734,
                    142.545454545454547, 212.363636363636374, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, 0.066906927, -0.014481221,
                   -0.087001808, -0.124326953, -0.148868367933, -0.149108533322])
""",
            "call": "compute_integral_posterior_mean(centers, forces)",
            "gold_call": "_oracle_compute_integral_posterior_mean(centers, forces)",
        },
        # --- Boundary: one window only, at the centre of the range ---
        {
            "setup": """import numpy as np
centers = np.array([144.0])
forces = np.array([-0.0871])
""",
            "call": "compute_integral_posterior_mean(centers, forces)",
            "gold_call": "_oracle_compute_integral_posterior_mean(centers, forces)",
        },
        # --- Edge: a constant integrand on a short range, where the estimate is nearly exact ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
forces = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
""",
            "call": "compute_integral_posterior_mean(centers, forces, 0.0, 10.0, 4.0, 6.0, 1.0e-10)",
            "gold_call": "_oracle_compute_integral_posterior_mean(centers, forces, 0.0, 10.0, 4.0, 6.0, 1.0e-10)",
        },
        # --- Valid: local increments over disjoint, overlapping and nested intervals ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 37.8, 107.6, 177.5, 247.3, 285.0])
forces = np.array([0.395264592, 0.06695, -0.05291, -0.10846, -0.13713, -0.149108533])
intervals = np.array([[0.0, 90.0], [90.0, 180.0],
                      [35.0, 210.0], [120.0, 150.0]])
""",
            "call": "compute_integral_posterior_mean(centers, forces, intervals=intervals)",
            "gold_call": "_oracle_compute_integral_posterior_mean(centers, forces, intervals=intervals)",
        },
        # --- Boundary/additivity: partition increments sum to the full-range estimate ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.7, 142.5, 212.4, 284.16, 285.0])
forces = np.array([0.395264592, 0.357711253, -0.01446, -0.08698,
                   -0.12434, -0.148868368, -0.149108533])
cuts = np.array([0.0, 31.0, 90.0, 173.0, 288.0])
intervals = np.column_stack((cuts[:-1], cuts[1:]))
""",
            "call": "np.append(compute_integral_posterior_mean(centers, forces, intervals=intervals), compute_integral_posterior_mean(centers, forces))",
            "gold_call": "np.append(_oracle_compute_integral_posterior_mean(centers, forces, intervals=intervals), _oracle_compute_integral_posterior_mean(centers, forces))",
        },
        # --- Edge: heteroscedastic window noise changes the interval weights ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 18.0, 55.0, 97.0, 144.0])
forces = np.array([0.3, 0.17, 0.02, -0.06, -0.09])
noise = np.array([1.0e-5, 2.0e-3, 5.0e-4, 4.0e-3, 1.0e-4])
intervals = np.array([[0.0, 50.0], [50.0, 100.0], [100.0, 144.0]])
""",
            "call": "compute_integral_posterior_mean(centers, forces, 0.0, 144.0, 0.6, 17.0, noise, intervals)",
            "gold_call": "_oracle_compute_integral_posterior_mean(centers, forces, 0.0, 144.0, 0.6, 17.0, noise, intervals)",
        },
        # --- Invalid: a window centre outside the integration range ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 300.0])
forces = np.array([0.4, -0.15])
def run_model():
    try:
        compute_integral_posterior_mean(centers, forces)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_integral_posterior_mean(centers, forces)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite measured force ---
        {
            "setup": """import numpy as np
centers = np.array([10.0, 100.0])
forces = np.array([0.4, np.nan])
def run_model():
    try:
        compute_integral_posterior_mean(centers, forces)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_integral_posterior_mean(centers, forces)
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
forces = np.array([0.2, 0.0, -0.1])
noise = np.array([1.0e-3, 2.0e-3])
def run_model():
    try:
        compute_integral_posterior_mean(centers, forces, noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_integral_posterior_mean(centers, forces, noise=noise)
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
