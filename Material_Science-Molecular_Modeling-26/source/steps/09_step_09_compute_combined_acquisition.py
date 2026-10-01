"""
Compose the batched outputs of steps 07 and 08, convert variance reduction to reduction per unit sampling cost, normalize over the complete candidate grid, and return combined acquisition scores at one node or a batch of nodes.

Pure integral-variance reduction is blind to the values measured so far, so on its own it spreads windows over the collective-variable range without regard to where the physically interesting states are. Adding a term built from the current free-energy reconstruction restores that sensitivity: candidates in predicted low-free-energy regions are promoted, which concentrates effort on the basins and on the path between them rather than on high, rarely visited parts of the landscape.




The two terms have incompatible units and wildly different dynamic ranges, so each is min-max rescaled onto the unit interval before they are mixed, using the smallest and largest value that term takes across the complete candidate grid. A term that is constant across the grid carries no information and rescales to zero everywhere rather than to an undefined ratio. The rescaling is what makes the weight meaningful, so it must precede the mixing and must be taken over the whole grid even when only a few nodes are asked for.




The exploitation term enters with a negative sign, because a high reconstructed free energy is a reason not to sample, and it is weighted by the supplied weight while the variance-reduction term takes the complementary weight. At zero weight the criterion reduces to pure variance reduction; as the weight grows the design collapses towards the predicted minima. Because the variance-reduction term is value-blind, the free-energy term can redirect the design at any nonzero weight whenever its weighted differences overturn the variance-reduction ranking; increasing the weight makes such redirection more likely.




Existing windows may have unequal sampling errors, and the expected error of a prospective window may vary across the menu. Both noise descriptions therefore have to survive the composition: existing-window noise conditions both terms, whereas candidate noise belongs to the prospective measurement and reaches only the variance-reduction term. A candidate-noise vector is always aligned with the complete grid, never with the subset of nodes requested.




When candidate windows have unequal computational costs, the value of an uncertainty-reducing observation is its raw reduction divided by the cost of acquiring it. This cost-sensitive rate, rather than the raw reduction, is the quantity rescaled over the complete grid and mixed with the free-energy term. All costs must be strictly positive; a zero cost would make the rate undefined, and negative costs have no resource-allocation meaning.

Returns
-------
float or np.ndarray: combined score for a scalar node, otherwise one score per requested node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_combined_acquisition(centers: np.ndarray, forces: np.ndarray,
                                 node: np.ndarray,
                                 n_grid: int = 100, weight: float = 0.45,
                                 lower: float = 0.0, upper: float = 288.0,
                                 variance: float = 0.25, lengthscale: float = 20.0,
                                 noise: np.ndarray = 1.0e-3,
                                 candidate_noise: np.ndarray = None,
                                 candidate_cost: np.ndarray = None) -> np.ndarray:
    """Evaluate combined acquisition scores at one node or a node batch.

    Obtain the complete-grid IVR and profile vectors by calling the previously
    implemented ``compute_ivr_acquisition`` and
    ``compute_free_energy_profile_value`` functions. Normalize over the full
    grid before selecting the requested node values.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
    node : int or np.ndarray
        Scalar index or non-empty one-dimensional integer array. Every index
        must lie from 0 through n_grid - 1; order and repetitions are
        preserved.
    n_grid : int
        Number of uniformly spaced candidate nodes spanning the range
        (n_grid >= 2).
    weight : float
        Weight of the exploitation term, in the closed interval [0, 1].
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar observation-noise variance shared by the existing
        windows or an array of shape (n,) giving one variance per window.
    candidate_noise : float or np.ndarray, optional
        Non-negative scalar prospective noise variance or an array of shape
        (n_grid,) giving one variance per complete-grid candidate. If omitted,
        the scalar value of noise is reused. It must be supplied explicitly
        when noise is window-specific. A candidate-noise array always refers
        to the complete grid, even when node requests only a subset.
    candidate_cost : float or np.ndarray, optional
        Strictly positive scalar acquisition cost or an array of shape
        (n_grid,) aligned with the complete candidate grid. If omitted, unit
        costs are used. Cost divides raw variance reduction before the
        complete-grid min-max rescaling.

    Raises
    ------
    ValueError
        If a scalar argument is not finite, node is not a scalar or
        one-dimensional integer array, n_grid is not an integer or is less
        than 2, an index is outside the grid, weight is outside [0, 1], the
        range or covariance hyperparameters are invalid, the window arrays are
        empty, mismatched, or non-finite, observation or candidate noise has
        an invalid shape or value, candidate cost has an invalid shape or
        non-positive entry, a centre lies outside the range, or a delegated
        batched result is invalid.

    Returns
    -------
    score : float or np.ndarray
        A native Python float for a scalar node, otherwise an array matching
        node with one combined score per requested index.
    """
    return score  # placeholder

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


def _oracle_compute_combined_acquisition(centers: np.ndarray, forces: np.ndarray,
                                         node: np.ndarray,
                                         n_grid: int = 100, weight: float = 0.45,
                                         lower: float = 0.0, upper: float = 288.0,
                                         variance: float = 0.25, lengthscale: float = 20.0,
                                         noise: np.ndarray = 1.0e-3,
                                         candidate_noise: np.ndarray = None,
                                         candidate_cost: np.ndarray = None) -> np.ndarray:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("weight", weight), ("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
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
    if float(weight) < 0.0 or float(weight) > 1.0:
        raise ValueError("weight must lie in [0, 1]")
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
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    noise_was_scalar = noise_array.ndim == 0
    if noise_was_scalar:
        observation_noise = np.full(centers.size, float(noise_array))
    elif noise_array.shape == centers.shape:
        observation_noise = noise_array.astype(float, copy=False)
    else:
        raise ValueError("noise must be scalar or have shape (n,)")
    if (not np.all(np.isfinite(observation_noise)) or
            np.any(observation_noise < 0.0)):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with window-specific noise")
        candidate_noise_array = np.asarray(float(noise_array))
    else:
        try:
            raw_candidate_noise = np.asarray(candidate_noise)
            if np.issubdtype(raw_candidate_noise.dtype, np.bool_):
                raise ValueError("candidate_noise must contain real numbers")
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    if candidate_noise_array.ndim == 0:
        candidate_noise_grid = np.full(int(n_grid), float(candidate_noise_array))
    elif candidate_noise_array.shape == (int(n_grid),):
        candidate_noise_grid = candidate_noise_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_noise must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_noise_grid)) or
            np.any(candidate_noise_grid < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    if candidate_cost is None:
        candidate_cost_array = np.asarray(1.0)
    else:
        try:
            raw_candidate_cost = np.asarray(candidate_cost)
            if np.issubdtype(raw_candidate_cost.dtype, np.bool_):
                raise ValueError("candidate_cost must contain real numbers")
            candidate_cost_array = np.asarray(candidate_cost, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_cost must contain real numbers")
        if candidate_cost_array.ndim > 1:
            raise ValueError("candidate_cost must be scalar or one-dimensional")
    if candidate_cost_array.ndim == 0:
        candidate_cost_grid = np.full(int(n_grid), float(candidate_cost_array))
    elif candidate_cost_array.shape == (int(n_grid),):
        candidate_cost_grid = candidate_cost_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_cost must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_cost_grid)) or
            np.any(candidate_cost_grid <= 0.0)):
        raise ValueError("candidate costs must be finite and strictly positive")

    def _normalise(values):
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    grid = np.linspace(float(lower), float(upper), int(n_grid))
    ivr = np.asarray(_oracle_compute_ivr_acquisition(
        centers, grid, lower, upper, variance, lengthscale,
        observation_noise, candidate_noise_grid), dtype=float)
    profile = np.asarray(_oracle_compute_free_energy_profile_value(
        centers, forces, np.arange(int(n_grid), dtype=int), int(n_grid),
        lower, upper, variance, lengthscale, observation_noise), dtype=float)
    if (ivr.shape != grid.shape or profile.shape != grid.shape or
            not np.all(np.isfinite(ivr)) or not np.all(np.isfinite(profile))):
        raise ValueError("the delegated batched acquisition inputs are invalid")

    cost_sensitive_ivr = ivr / candidate_cost_grid
    combined = (-float(weight) * _normalise(profile)
                + (1.0 - float(weight)) * _normalise(cost_sensitive_ivr))

    values = combined[nodes]
    if values.ndim == 0:
        return float(values)
    return values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the node the benchmark run picks first (normal scenario) ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_combined_acquisition(centers, forces, 48)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, 48)",
        },
        # --- Valid: a node near the barrier, where the exploitation term subtracts most ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_combined_acquisition(centers, forces, 22)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, 22)",
        },
        # --- Valid: pure variance reduction, with the exploitation term switched off ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_combined_acquisition(centers, forces, 22, 100, 0.0)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, 22, 100, 0.0)",
        },
        # --- Boundary: pure exploitation, where the geometric term drops out entirely ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "compute_combined_acquisition(centers, forces, 99, 100, 1.0)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, 99, 100, 1.0)",
        },
        # --- Edge: a flat integrand, where the exploitation term is constant and normalises away ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.0, 0.0, 0.0, 0.0])
""",
            "call": "compute_combined_acquisition(centers, forces, 20, 30, 0.5)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, 20, 30, 0.5)",
        },
        # --- Valid batch: score an unsorted, repeated subset after full-grid normalization ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.0, 144.0, 216.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.0139, -0.0876,
                   -0.1257, -0.148868367933, -0.149108533322])
nodes = np.array([99, 0, 49, 22, 49, 75])
""",
            "call": "compute_combined_acquisition(centers, forces, nodes)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, nodes)",
        },
        # --- Valid batch: heterogeneous existing and full-menu candidate noise ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 48.0, 144.0, 240.0, 285.0])
forces = np.array([0.395264592035, 0.0618, -0.0876, -0.1342, -0.149108533322])
noise = np.array([1.0e-5, 2.0e-3, 4.0e-4, 3.0e-3, 8.0e-5])
candidate_noise = np.geomspace(2.0e-5, 5.0e-3, 41)
nodes = np.array([40, 3, 19, 3, 27, 0])
""",
            "call": "compute_combined_acquisition(centers, forces, nodes, 41, 0.35, 0.0, 288.0, 0.25, 20.0, noise, candidate_noise)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, nodes, 41, 0.35, 0.0, 288.0, 0.25, 20.0, noise, candidate_noise)",
        },
        # --- Boundary batch: return the complete pure-IVR acquisition vector ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 142.5, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.087001808,
                   -0.148868367933, -0.149108533322])
nodes = np.arange(41, dtype=int)
""",
            "call": "compute_combined_acquisition(centers, forces, nodes, 41, 0.0)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, nodes, 41, 0.0)",
        },
        # --- Invalid: an exploitation weight outside the unit interval ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
def run_model():
    try:
        compute_combined_acquisition(centers, forces, 10, 100, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_acquisition(centers, forces, 10, 100, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: heterogeneous existing noise without prospective noise ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 144.0, 285.0])
forces = np.array([0.395264592035, -0.0876, -0.149108533322])
noise = np.array([1.0e-5, 2.0e-3, 8.0e-5])
def run_model():
    try:
        compute_combined_acquisition(centers, forces, 10, noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_acquisition(centers, forces, 10, noise=noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a node index outside the candidate grid ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
def run_model():
    try:
        compute_combined_acquisition(centers, forces, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_acquisition(centers, forces, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Valid: nonuniform costs alter the IVR rate before normalization ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
candidate_cost = 1.0 + 0.8 * np.linspace(0.0, 1.0, 100) ** 2
nodes = np.array([12, 48, 73, 91])
""",
            "call": "compute_combined_acquisition(centers, forces, nodes, candidate_cost=candidate_cost)",
            "gold_call": "_oracle_compute_combined_acquisition(centers, forces, nodes, candidate_cost=candidate_cost)",
        },
        # --- Invalid: zero candidate cost makes reduction per cost undefined ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
candidate_cost = np.ones(100)
candidate_cost[20] = 0.0
def run_model():
    try:
        compute_combined_acquisition(centers, forces, 20, candidate_cost=candidate_cost)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_acquisition(centers, forces, 20, candidate_cost=candidate_cost)
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
        compute_combined_acquisition(centers, forces, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_acquisition(centers, forces, nodes)
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
