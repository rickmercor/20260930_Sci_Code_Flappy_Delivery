"""
Reconstruct the free-energy profile by calling step 04 once for the complete uniform-grid posterior mean gradient, integrating it, and gathering one node or an ordered node batch.

Umbrella integration recovers a free-energy profile by integrating a mean-force estimate, and here the estimate being integrated is the Gaussian-process posterior mean rather than the raw window measurements. The surrogate is evaluated on a uniform grid spanning the collective-variable range and accumulated with the composite trapezoidal rule, so the profile is defined on the same nodes at which it will later be compared with a reference. The constant of integration is fixed by shifting the whole profile so that its minimum sits at zero, which is the standard convention that makes two profiles comparable without prejudging where the reference state lies.




The grid is uniform and spans the range endpoint to endpoint, and the profile is accumulated from the lower end so that every node carries the integral of the predicted gradient from that lower end up to it. The whole profile is then shifted so that its minimum sits at zero, which is the standard convention that makes two profiles comparable without prejudging where the reference state lies; the shift is applied after accumulation, over the complete grid, so the returned value at a node depends on the profile everywhere and not only on the nodes requested.




Two properties of this construction are worth keeping in view. Because the posterior mean is shrunk towards zero where the design is sparse, the reconstructed profile is systematically flattened relative to the truth, and the flattening accumulates along the direction of integration. And because the profile enters the sample-placement strategy as the exploitation term, the reconstruction is not merely an output of the run; it feeds back into where the next window is placed.

Returns
-------
float or np.ndarray: reconstructed profile value for a scalar node, otherwise one value per requested node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_free_energy_profile_value(centers: np.ndarray, forces: np.ndarray, node: np.ndarray,
                                      n_grid: int = 100, lower: float = 0.0,
                                      upper: float = 288.0, variance: float = 0.25,
                                      lengthscale: float = 20.0,
                                      noise: np.ndarray = 1.0e-3) -> np.ndarray:
    """Evaluate the reconstructed profile at one node or a node batch.

    Obtain the complete grid of gradient predictions in one call to the
    previously implemented ``compute_posterior_mean_gradient`` function.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    node : int or np.ndarray
        Scalar index or non-empty one-dimensional integer array. Every index
        must be in the range 0 to n_grid - 1; order and repetitions are
        preserved in the returned values.
    n_grid : int
        Number of uniformly spaced nodes spanning the range (n_grid >= 2).
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

    Raises
    ------
    ValueError
        If a scalar argument is not finite, node is not a scalar or
        one-dimensional integer array, n_grid is not an integer or is less
        than 2, an index is outside the grid, upper is not greater than lower,
        a covariance hyperparameter is invalid, noise is invalid, the window
        arrays are empty, mismatched, or non-finite, the delegated grid
        prediction has the wrong shape, or the observation covariance is
        singular.

    Returns
    -------
    profile_value : float or np.ndarray
        A native Python float for a scalar node, otherwise an array matching
        node, measured from the minimum of the reconstructed profile.
    """
    return profile_value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_integer_scalar(value) -> bool:
    """Return True for an integer scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def _oracle_compute_free_energy_profile_value(centers: np.ndarray, forces: np.ndarray,
                                              node: np.ndarray,
                                              n_grid: int = 100, lower: float = 0.0,
                                              upper: float = 288.0, variance: float = 0.25,
                                              lengthscale: float = 20.0,
                                              noise: np.ndarray = 1.0e-3) -> np.ndarray:
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
    if not _is_integer_scalar(n_grid):
        raise ValueError("n_grid must be an integer")
    try:
        nodes = np.asarray(node)
    except (TypeError, ValueError):
        raise ValueError("node must contain integers")
    if (nodes.ndim > 1 or nodes.size < 1 or
            np.issubdtype(nodes.dtype, np.bool_) or
            not np.issubdtype(nodes.dtype, np.integer)):
        raise ValueError("node must be a scalar or non-empty one-dimensional integer array")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    if np.any(nodes < 0) or np.any(nodes >= int(n_grid)):
        raise ValueError("every node must index a grid point")

    centers = np.asarray(centers, dtype=float).ravel()
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")

    grid = np.linspace(float(lower), float(upper), int(n_grid))
    predicted = np.asarray(_oracle_compute_posterior_mean_gradient(
        centers, forces, grid, float(variance), float(lengthscale), noise),
        dtype=float)
    if predicted.shape != grid.shape or not np.all(np.isfinite(predicted)):
        raise ValueError("the delegated grid prediction has the wrong shape")
    spacing = grid[1] - grid[0]
    raw = np.concatenate(([0.0], np.cumsum(0.5 * (predicted[1:] + predicted[:-1]) * spacing)))
    profile = raw - raw.min()

    values = profile[nodes]
    if values.ndim == 0:
        return float(values)
    return values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the barrier node of a partially acquired design (normal scenario) ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.727272727272734, 142.545454545454547,
                    212.363636363636374, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.014481221, -0.087001808,
                   -0.124326953, -0.148868367933, -0.149108533322])
""",
            "call": "compute_free_energy_profile_value(centers, forces, 22)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, 22)",
        },
        # --- Valid: the same design evaluated on the descending flank ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.727272727272734, 142.545454545454547,
                    212.363636363636374, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.014481221, -0.087001808,
                   -0.124326953, -0.148868367933, -0.149108533322])
""",
            "call": "compute_free_energy_profile_value(centers, forces, 70)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, 70)",
        },
        # --- Boundary: the last grid node of a design whose minimum sits at the first node ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_free_energy_profile_value(centers, forces, 99)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, 99)",
        },
        # --- Boundary: the coarsest admissible grid, a single trapezoid ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_free_energy_profile_value(centers, forces, 1, 2)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, 1, 2)",
        },
        # --- Edge: a monotonically rising profile, whose minimum sits at the first node ---
        {
            "setup": """import numpy as np
centers = np.array([2.0, 60.0, 120.0, 180.0, 240.0, 286.0])
forces = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
""",
            "call": "compute_free_energy_profile_value(centers, forces, 50)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, 50)",
        },
        # --- Valid batch: unsorted and repeated reconstruction nodes ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 72.0, 144.0, 216.0, 285.0])
forces = np.array([0.395264592035, -0.0139, -0.0876, -0.1257, -0.149108533322])
nodes = np.array([99, 0, 49, 22, 49, 1])
""",
            "call": "compute_free_energy_profile_value(centers, forces, nodes)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, nodes)",
        },
        # --- Boundary batch: request the entire profile in a single call ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 142.5, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.087001808,
                   -0.148868367933, -0.149108533322])
nodes = np.arange(37, dtype=int)
""",
            "call": "compute_free_energy_profile_value(centers, forces, nodes, 37)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, nodes, 37)",
        },
        # --- Edge: heteroscedastic training noise feeds through the delegated batch prediction ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 18.0, 55.0, 97.0, 144.0])
forces = np.array([0.3, 0.17, 0.02, -0.06, -0.09])
noise = np.array([1.0e-5, 2.0e-3, 5.0e-4, 4.0e-3, 1.0e-4])
nodes = np.array([40, 0, 17, 40, 8])
""",
            "call": "compute_free_energy_profile_value(centers, forces, nodes, 41, 0.0, 144.0, 0.6, 17.0, noise)",
            "gold_call": "_oracle_compute_free_energy_profile_value(centers, forces, nodes, 41, 0.0, 144.0, 0.6, 17.0, noise)",
        },
        # --- Invalid: a node index outside the grid ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
def run_model():
    try:
        compute_free_energy_profile_value(centers, forces, 100)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_free_energy_profile_value(centers, forces, 100)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a grid too coarse to define a trapezoid ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
def run_model():
    try:
        compute_free_energy_profile_value(centers, forces, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_free_energy_profile_value(centers, forces, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a node batch containing non-integer values ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
nodes = np.array([0.0, 1.0, 2.5])
def run_model():
    try:
        compute_free_energy_profile_value(centers, forces, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_free_energy_profile_value(centers, forces, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: delegated heteroscedastic noise has the wrong length ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 100.0])
forces = np.array([0.4, 0.36, -0.04])
noise = np.array([1.0e-3, 2.0e-3])
def run_model():
    try:
        compute_free_energy_profile_value(centers, forces, 5, noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_free_energy_profile_value(centers, forces, 5, noise=noise)
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
