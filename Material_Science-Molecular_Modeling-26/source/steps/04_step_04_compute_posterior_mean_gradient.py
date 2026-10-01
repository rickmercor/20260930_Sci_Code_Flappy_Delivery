"""
Evaluate the Gaussian-process posterior mean of the free-energy gradient at one query or an ordered query batch, conditioned on mean forces with scalar or window-specific noise.

The surrogate for the free-energy gradient is a zero-mean Gaussian process with an exponential covariance and an additive white-noise term that absorbs the finite-sampling error of each restrained window. Conditioning it on the measured mean forces gives a posterior mean that interpolates the observations and relaxes back towards the prior mean between them. Two consequences matter for what follows. First, the interpolant is the only object that carries information about the shape of the landscape into the free-energy reconstruction, so its accuracy between windows is exactly what the sample-placement strategy is trying to buy. Second, because the prior mean is zero, the posterior mean is shrunk towards zero wherever the data are sparse relative to the lengthscale, and that shrinkage biases every integral built on top of it.




Each measured mean force is treated as the latent gradient at that window plus independent additive white noise of the stated variance, which may differ from window to window because restrained simulations can have unequal effective sample sizes. That noise term is what makes the conditioning well posed when two windows are placed close together, and it is also what stops the interpolant from chasing sampling noise in the measured forces. The covariance is the same Matern-1/2 kernel used throughout the pipeline. A query batch shares one solve for the GP weights; only the query-to-window cross-covariance changes from row to row, so a batch must not be looped over one query at a time when a single solve would serve.

Returns
-------
float or np.ndarray: one posterior mean for a scalar query, otherwise one value per ordered query.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_posterior_mean_gradient(centers: np.ndarray, forces: np.ndarray, query: np.ndarray,
                                    variance: float = 0.25, lengthscale: float = 20.0,
                                    noise: np.ndarray = 1.0e-3) -> np.ndarray:
    """Evaluate posterior mean gradients at one point or a query batch.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    query : float or np.ndarray
        Scalar query or non-empty one-dimensional array of query values;
        input order and repetitions are preserved.
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance or an array of shape
        (n,) with one variance per measured window.

    Raises
    ------
    ValueError
        If variance or lengthscale is not a finite positive scalar, query is
        not a finite scalar or non-empty one-dimensional array, noise is not
        a finite non-negative scalar or an array of shape (n,), the window
        arrays are empty, mismatched, or non-finite, or their observation
        covariance matrix is singular.

    Returns
    -------
    posterior_mean : float or np.ndarray
        A native Python float for a scalar query; otherwise an array matching
        the one-dimensional query shape.
    """
    return posterior_mean  # placeholder

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


def _oracle_compute_posterior_mean_gradient(centers: np.ndarray, forces: np.ndarray,
                                            query: np.ndarray,
                                            variance: float = 0.25, lengthscale: float = 20.0,
                                            noise: np.ndarray = 1.0e-3) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
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

    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_array)
    cross = float(variance) * np.exp(
        -np.abs(queries[:, None] - centers[None, :]) / float(lengthscale))

    try:
        weights = np.linalg.solve(gram, forces)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    result = cross @ weights
    if scalar_query:
        return float(result[0])
    return np.asarray(result, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: interpolation between two well-separated windows (normal scenario) ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 142.545454545454547, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.086, -0.148868367933, -0.149108533322])
query = 72.0
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query)",
        },
        # --- Valid: evaluation on top of an existing window, where the ridge matters ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
query = 3.0
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query)",
        },
        # --- Boundary: a single window, where the posterior mean is a pure exponential decay ---
        {
            "setup": """import numpy as np
centers = np.array([64.0])
forces = np.array([0.00062])
query = 200.0
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query)",
        },
        # --- Edge: a noiseless model with two nearly coincident windows ---
        {
            "setup": """import numpy as np
centers = np.array([50.0, 50.0009, 210.0])
forces = np.array([0.02, 0.0201, -0.12])
query = 130.0
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query, 0.25, 20.0, 1.0e-8)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query, 0.25, 20.0, 1.0e-8)",
        },
        # --- Valid: an ordered query batch including repeated observations ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.0, 142.5, 220.0, 284.16, 285.0])
forces = np.array([0.395264592, 0.357711253, -0.0139, -0.0870,
                   -0.1276, -0.148868368, -0.149108533])
query = np.array([288.0, 3.0, 90.0, 3.0, 0.0])
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query)",
        },
        # --- Boundary: a whole reconstruction grid is handled in one shared solve ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 25.0, 80.0, 144.0])
forces = np.array([0.3, 0.13, -0.025, -0.087])
query = np.linspace(0.0, 144.0, 41)
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query, 0.6, 17.0, 1.0e-4)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query, 0.6, 17.0, 1.0e-4)",
        },
        # --- Edge: heteroscedastic noise changes influence across the batch ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 18.0, 55.0, 97.0, 144.0])
forces = np.array([0.3, 0.17, 0.02, -0.06, -0.09])
query = np.array([0.0, 18.0, 55.0, 97.0, 144.0, 72.0])
noise = np.array([1.0e-5, 2.0e-3, 5.0e-4, 4.0e-3, 1.0e-4])
""",
            "call": "compute_posterior_mean_gradient(centers, forces, query, 0.6, 17.0, noise)",
            "gold_call": "_oracle_compute_posterior_mean_gradient(centers, forces, query, 0.6, 17.0, noise)",
        },
        # --- Invalid: mismatched numbers of centres and forces ---
        {
            "setup": """import numpy as np
centers = np.array([1.0, 2.0, 3.0])
forces = np.array([0.1, 0.2])
def run_model():
    try:
        compute_posterior_mean_gradient(centers, forces, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_posterior_mean_gradient(centers, forces, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an empty set of windows ---
        {
            "setup": """import numpy as np
centers = np.array([])
forces = np.array([])
def run_model():
    try:
        compute_posterior_mean_gradient(centers, forces, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_posterior_mean_gradient(centers, forces, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative white-noise variance ---
        {
            "setup": """import numpy as np
centers = np.array([1.0, 2.0])
forces = np.array([0.1, 0.2])
def run_model():
    try:
        compute_posterior_mean_gradient(centers, forces, 1.5, 0.25, 20.0, -1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_posterior_mean_gradient(centers, forces, 1.5, 0.25, 20.0, -1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a two-dimensional query batch is not part of the contract ---
        {
            "setup": """import numpy as np
centers = np.array([1.0, 2.0])
forces = np.array([0.1, 0.2])
query = np.array([[1.0, 1.5]])
def run_model():
    try:
        compute_posterior_mean_gradient(centers, forces, query)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_posterior_mean_gradient(centers, forces, query)
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
