"""
Choose the next affordable restrained-window value from the batched complete-grid cost-sensitive acquisition returned by step 09, masking sampled or unaffordable nodes and applying the deterministic lower-index tie-break.

The acquisition step is what turns a fixed set of umbrella windows into an adaptive protocol. The combined score is evaluated at every node of the candidate grid and the largest one wins, so the grid is not a discretisation of the free energy but the finite menu of window positions the protocol is allowed to choose from. Its resolution therefore bounds how finely the design can be refined, and coarsening it changes the run in a way that refining the reconstruction grid would not.




Nodes that already carry a window are removed from the menu, and the winner is the grid node with the largest surviving score. Without that exclusion the criterion can return a location it has already measured: the reduction in integral variance offered by repeating an observation is small but strictly positive once a white-noise term is present, and in a landscape where one basin dominates the exploitation term that residual can still be the largest score on the grid. Removing sampled nodes makes each iteration strictly informative and guarantees the run terminates with as many distinct windows as it has queries. If the exclusion leaves nothing behind, there is no admissible window to run and the request cannot be satisfied.




Exclusion happens after the scores have been formed, so it changes which node wins but never the value any node is given; a node's score does not depend on which other nodes remain available. Ties among equal scores are broken towards the lower end of the range, which is the convention that makes the whole protocol reproducible. When prospective sampling precision varies with position, its full-grid noise vector stays aligned with the candidate menu while sampled nodes are masked; it is never shortened to match the requested or remaining nodes.




Resource feasibility is another mask, not another normalization. The cost-sensitive rate is normalized over the full candidate grid first, and only then are nodes whose acquisition cost exceeds the remaining budget removed. This preserves one score scale throughout the run while guaranteeing that a selected window can actually be paid for.

Returns
-------
float: the collective-variable value of the next restrained window, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_next_umbrella_center(centers: np.ndarray, forces: np.ndarray, n_grid: int = 100,
                                weight: float = 0.45, lower: float = 0.0,
                                upper: float = 288.0, variance: float = 0.25,
                                lengthscale: float = 20.0,
                                noise: np.ndarray = 1.0e-3,
                                candidate_noise: np.ndarray = None,
                                candidate_cost: np.ndarray = None,
                                max_cost: float = None) -> float:
    """Choose the collective-variable value of the next restrained window.

    Obtain all complete-grid scores in one batched call to the previously
    implemented ``compute_combined_acquisition`` function; this step supplies
    only sampled-node masking and the deterministic tie-break.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    forces : np.ndarray
        Array of shape (n,) holding the gradient estimate carried by each of
        those windows.
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
        (n_grid,) aligned with the complete candidate grid. If omitted, the
        scalar value of noise is reused; it is required when noise is
        window-specific.
    candidate_cost : float or np.ndarray, optional
        Strictly positive scalar acquisition cost or an array of shape
        (n_grid,) aligned with the complete candidate grid. If omitted, unit
        costs are used.
    max_cost : float, optional
        Largest acquisition cost still affordable. If omitted, no
        affordability mask is applied. A value of zero is allowed and leaves
        no strictly positive-cost candidate.

    Raises
    ------
    ValueError
        If n_grid is not an integer or is less than 2, a scalar argument is
        not finite, weight is outside [0, 1], the range or covariance
        hyperparameters are invalid, the window arrays are empty, mismatched,
        or non-finite, observation or candidate noise has an invalid shape or
        value, candidate cost or max_cost is invalid, a centre lies outside
        the range, the covariance matrix is singular, the batched
        combined-acquisition result is invalid, or no unsampled affordable
        candidate remains.

    Returns
    -------
    next_center : float
        The collective-variable value at which the next window should be
        restrained, as a native Python float.
    """
    return next_center  # placeholder

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


def _oracle_select_next_umbrella_center(centers: np.ndarray, forces: np.ndarray, n_grid: int = 100,
                                        weight: float = 0.45, lower: float = 0.0,
                                        upper: float = 288.0, variance: float = 0.25,
                                        lengthscale: float = 20.0,
                                        noise: np.ndarray = 1.0e-3,
                                        candidate_noise: np.ndarray = None,
                                        candidate_cost: np.ndarray = None,
                                        max_cost: float = None) -> float:
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
    if max_cost is not None:
        if (not _is_real_scalar(max_cost) or not np.isfinite(float(max_cost))
                or float(max_cost) < 0.0):
            raise ValueError("max_cost must be a finite non-negative real number")

    grid = np.linspace(float(lower), float(upper), int(n_grid))
    score = np.asarray(_oracle_compute_combined_acquisition(
        centers, forces, np.arange(int(n_grid), dtype=int), int(n_grid),
        weight, lower, upper, variance, lengthscale, observation_noise,
        candidate_noise_grid, candidate_cost_grid), dtype=float)
    if score.shape != grid.shape or not np.all(np.isfinite(score)):
        raise ValueError("the batched combined acquisition is invalid")
    for center in centers:
        score[np.isclose(grid, center, rtol=0.0, atol=1.0e-9)] = -np.inf
    if max_cost is not None:
        score[candidate_cost_grid > float(max_cost) + 1.0e-12] = -np.inf
    if not np.isfinite(score).any():
        raise ValueError("no unsampled affordable candidate remains")

    return float(grid[int(np.argmax(score))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the first acquisition of the benchmark run (normal scenario) ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "select_next_umbrella_center(centers, forces)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces)",
        },
        # --- Valid: a later acquisition, choosing between several gaps of similar width ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.727272727272734, 142.545454545454547,
                    212.363636363636374, 247.272727272727280, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.014481221, -0.087001808,
                   -0.124326953, -0.137502864, -0.148868367933, -0.149108533322])
""",
            "call": "select_next_umbrella_center(centers, forces)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces)",
        },
        # --- Valid: the same state with pure variance reduction instead ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.727272727272734, 142.545454545454547,
                    212.363636363636374, 247.272727272727280, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.014481221, -0.087001808,
                   -0.124326953, -0.137502864, -0.148868367933, -0.149108533322])
""",
            "call": "select_next_umbrella_center(centers, forces, 100, 0.0)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces, 100, 0.0)",
        },
        # --- Boundary: a coarse candidate grid whose nodes are all already sampled but one ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 96.0, 192.0])
forces = np.array([0.4333, -0.0532, -0.1160])
""",
            "call": "select_next_umbrella_center(centers, forces, 4)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces, 4)",
        },
        # --- Edge: heavy exploitation, which pulls the choice to the lowest predicted node ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
""",
            "call": "select_next_umbrella_center(centers, forces, 100, 0.9)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces, 100, 0.9)",
        },
        # --- Edge: heterogeneous current noise and position-dependent candidate noise ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 48.0, 144.0, 240.0, 285.0])
forces = np.array([0.395264592035, 0.0618, -0.0876, -0.1342, -0.149108533322])
noise = np.array([1.0e-5, 2.0e-3, 4.0e-4, 3.0e-3, 8.0e-5])
candidate_noise = np.geomspace(2.0e-5, 5.0e-3, 41)
""",
            "call": "select_next_umbrella_center(centers, forces, 41, 0.35, 0.0, 288.0, 0.25, 20.0, noise, candidate_noise)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces, 41, 0.35, 0.0, 288.0, 0.25, 20.0, noise, candidate_noise)",
        },
        # --- Valid: affordability masks expensive nodes only after scoring ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
forces = np.array([0.395264592035, 0.357711252640, -0.148868367933, -0.149108533322])
candidate_cost = 1.0 + 0.8 * np.linspace(0.0, 1.0, 100) ** 2
""",
            "call": "select_next_umbrella_center(centers, forces, candidate_cost=candidate_cost, max_cost=1.12)",
            "gold_call": "_oracle_select_next_umbrella_center(centers, forces, candidate_cost=candidate_cost, max_cost=1.12)",
        },
        # --- Invalid: the remaining budget funds no positive-cost candidate ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
def run_model():
    try:
        select_next_umbrella_center(centers, forces, max_cost=0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_umbrella_center(centers, forces, max_cost=0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: no candidate node is left unsampled ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 96.0, 192.0, 288.0])
forces = np.array([0.4333, -0.0532, -0.1160, -0.1499])
def run_model():
    try:
        select_next_umbrella_center(centers, forces, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_umbrella_center(centers, forces, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a candidate grid with fewer than two nodes ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
forces = np.array([0.4, 0.36])
def run_model():
    try:
        select_next_umbrella_center(centers, forces, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_umbrella_center(centers, forces, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: candidate noise is not aligned with the complete grid ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 144.0, 285.0])
forces = np.array([0.395264592035, -0.0876, -0.149108533322])
candidate_noise = np.ones(12) * 1.0e-3
def run_model():
    try:
        select_next_umbrella_center(centers, forces, 41, candidate_noise=candidate_noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_umbrella_center(centers, forces, 41, candidate_noise=candidate_noise)
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
