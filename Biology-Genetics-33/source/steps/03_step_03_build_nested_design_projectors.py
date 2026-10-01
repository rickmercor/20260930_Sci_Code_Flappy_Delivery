"""
Build, for a two-level family design with unequal family sizes, the pair of symmetric matrices that isolate the between-family and the within-family variation while annihilating the population mean.

In a nested design each level is separated out by a symmetric matrix that annihilates every level coarser than it. The two such matrices here carry the family-level and the individual-level signal respectively.

Returns
-------
np.ndarray of shape (2, n, n), float: the between-family and within-family level-increment matrices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_nested_design_projectors(family_sizes: np.ndarray,
                                   include_intercept: bool = True) -> np.ndarray:
    """Build the two level-increment matrices of an unbalanced full-sibling design.

    Individuals are ordered family by family, so the first ``family_sizes[0]``
    rows belong to family zero, the next ``family_sizes[1]`` rows to family one,
    and so on.

    Parameters
    ----------
    family_sizes : np.ndarray
        One-dimensional array of the number of individuals in each family; every
        entry is an integer value of at least one.
    include_intercept : bool
        Whether an unknown population mean common to all individuals is carried
        as a fixed effect and must be annihilated by both matrices.

    Returns
    -------
    projectors : np.ndarray
        Array of shape ``(2, n, n)`` with ``n`` the total number of individuals.
        Entry ``0`` is the matrix that isolates the between-family level and
        entry ``1`` the matrix that isolates the within-family level.

    Raises
    ------
    ValueError
        If ``family_sizes`` is not a one-dimensional, non-empty array of finite
        integer values, or if any entry is smaller than one.
    """
    return projectors  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_nested_design_projectors(family_sizes: np.ndarray,
                                           include_intercept: bool = True) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    sizes = np.asarray(family_sizes, dtype=float)
    if sizes.ndim != 1 or sizes.size < 1:
        raise ValueError("family_sizes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sizes)):
        raise ValueError("family_sizes must contain only finite entries")
    if not np.allclose(sizes, np.round(sizes), rtol=0.0, atol=1e-12):
        raise ValueError("family_sizes entries must be integer valued")
    sizes = np.asarray(np.round(sizes), dtype=np.int64)
    if np.any(sizes < 1):
        raise ValueError("every family must contain at least one individual")

    n_families = int(sizes.size)
    n_individuals = int(sizes.sum())

    # Membership matrix of the family level; the individual level is the identity.
    membership = np.zeros((n_individuals, n_families), dtype=float)
    start = 0
    for family, size in enumerate(sizes):
        membership[start:start + int(size), family] = 1.0
        start += int(size)

    # Orthogonal projection onto the family column space. Its Gram matrix is
    # diagonal, so the projection is a block of reciprocal family sizes.
    family_projection = membership @ (membership.T / sizes[:, None].astype(float))

    if bool(include_intercept):
        # The population mean spans the constant vector, which lies inside the
        # family column space, so its projection is subtracted from the coarser
        # increment only.
        mean_projection = np.full((n_individuals, n_individuals),
                                  1.0 / float(n_individuals), dtype=float)
    else:
        mean_projection = np.zeros((n_individuals, n_individuals), dtype=float)

    between = family_projection - mean_projection
    within = np.eye(n_individuals, dtype=float) - family_projection

    projectors = np.stack([between, within])
    # Symmetrise to remove the asymmetry that floating point accumulation leaves.
    return 0.5 * (projectors + np.transpose(projectors, (0, 2, 1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the repeating one-two-three family pattern of the testbed,
        #     truncated to a size the test can carry (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
""",
            "call": "sig(build_nested_design_projectors(family_sizes, True), 1.0)",
            "gold_call": "sig(_oracle_build_nested_design_projectors(family_sizes, True), 1.0)",
        },
        # --- Valid: the same design without a population mean, which shifts only
        #     the between-family matrix ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
""",
            "call": "sig(build_nested_design_projectors(family_sizes, False), 1.0)",
            "gold_call": "sig(_oracle_build_nested_design_projectors(family_sizes, False), 1.0)",
        },
        # --- Valid: a strongly unbalanced design, where equal-weight averaging
        #     across families would be visible ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([7, 1, 1, 2], dtype=np.int64)
""",
            "call": "sig(build_nested_design_projectors(family_sizes, True), 1.0)",
            "gold_call": "sig(_oracle_build_nested_design_projectors(family_sizes, True), 1.0)",
        },
        # --- Boundary: singleton families only, for which the within-family level
        #     carries no information at all ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([1, 1, 1, 1], dtype=np.int64)
""",
            "call": "sig(build_nested_design_projectors(family_sizes, True), 1.0)",
            "gold_call": "sig(_oracle_build_nested_design_projectors(family_sizes, True), 1.0)",
        },
        # --- Edge: a single family, where the between-family increment collapses
        #     to zero once the population mean is removed ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([4], dtype=np.int64)
""",
            "call": "sig(build_nested_design_projectors(family_sizes, True), 1.0)",
            "gold_call": "sig(_oracle_build_nested_design_projectors(family_sizes, True), 1.0)",
        },
        # --- Edge: the two ranks, which pin the level decomposition independently
        #     of the entries of the matrices ---
        {
            "setup": """import numpy as np
family_sizes = np.array([2, 3, 1, 4, 2], dtype=np.int64)
def ranks(fn):
    out = np.asarray(fn(family_sizes, True), dtype=float)
    return round(float(np.linalg.matrix_rank(out[0], tol=1e-9)
                       + 100.0 * np.linalg.matrix_rank(out[1], tol=1e-9)), 6)
""",
            "call": "ranks(build_nested_design_projectors)",
            "gold_call": "ranks(_oracle_build_nested_design_projectors)",
        },
        # --- Invalid: a family with no individuals ---
        {
            "setup": """import numpy as np
family_sizes = np.array([2, 0, 3], dtype=np.int64)
def run_model():
    try:
        build_nested_design_projectors(family_sizes, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_nested_design_projectors(family_sizes, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fractional family sizes ---
        {
            "setup": """import numpy as np
family_sizes = np.array([2.0, 1.5, 3.0])
def run_model():
    try:
        build_nested_design_projectors(family_sizes, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_nested_design_projectors(family_sizes, True)
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
