"""
Assemble the dense matrix representing one multigrid V-cycle applied to the effective stiffness, which is the approximate inverse used inside the preconditioner.

The preconditioner never inverts the effective stiffness. It replaces the inverse by the action of a single multigrid V-cycle, and everything the spectral theory says about the method is expressed in terms of how close that action is to the true inverse. Because the quantity to be measured downstream is spectral rather than algorithmic, the V-cycle is materialised here as an explicit matrix: applying the cycle to each unit vector in turn gives the operator whose spectral relationship with the effective stiffness is the object of interest. In a production solver the same operator is applied matrix-free at O(n) cost per application; forming it densely changes nothing about what it computes.

The hierarchy is built from the aggregation and the smoothed prolongator of the two preceding steps. Those of the finest level are supplied as arguments, so that the level on which the mechanism's own rigid-body structure still exists is transferred exactly as those steps produced it; every coarser level is coarsened by calling the same two step functions again on the Galerkin coarse operator. The hierarchy therefore has no private copy of the coarsening rule, and a change in either step propagates into this operator and into every number that follows from it.

One V-cycle at a given level consists of a pre-smoothing step applied to a zero initial guess, formation of the residual, restriction of that residual by the transpose of the prolongator, a recursive V-cycle on the Galerkin coarse operator formed as the prolongator transpose times the level operator times the prolongator, prolongation of the coarse correction back and addition to the current iterate, and finally a post-smoothing step applied to the corrected iterate against the same right-hand side. The recursion terminates when the level operator is small enough or the level limit is reached, and there the operator is inverted exactly. One pre-smoothing and one post-smoothing sweep are used at every level, which is the usual balance between cost per cycle and error attenuation.

The smoother is not the same at all levels, because the type of degree of freedom is not the same. At the finest level the unknowns are the six coordinates of each rigid body, three translational and three rotational, and these are coupled so tightly through the inertia tensor and the joint stiffnesses that a pointwise smoother cannot relax them independently; the splitting into diagonal, strictly lower and strictly upper parts is therefore taken block-wise, one block per body, so that each body is relaxed exactly. Aggregation destroys that structure, so at every coarser level the same smoother is used with scalar blocks. In both cases the smoother is the symmetric form built from the two triangular splittings, which is symmetric whenever the level operator is, so that the whole V-cycle operator is symmetric and the spectral-equivalence statement of the next step is meaningful.

Applying the cycle to a zero initial guess with right-hand side b returns the smoother acting on b, and the corrected iterate is post-smoothed against the same b, not against a zero right-hand side; smoothing a corrected iterate against zero would drive it towards the origin and destroy the coarse-grid correction rather than clean it up.

Returns
-------
np.ndarray of shape (n, n), float: the dense matrix representation of one multigrid V-cycle, an approximate inverse of the input operator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_vcycle_operator(stiffness: np.ndarray, aggregates: np.ndarray,
                             prolongator: np.ndarray, theta: float, omega: float,
                             block_size: int, n_min: int,
                             max_levels: int) -> np.ndarray:
    """Assemble the dense operator of one multigrid V-cycle.

    Parameters
    ----------
    stiffness : np.ndarray
        Symmetric positive-definite operator of shape (n, n).
    aggregates : np.ndarray
        Finest-level aggregation of shape (n,), as produced by the aggregation
        step, holding consecutive aggregate indices starting at zero.
    prolongator : np.ndarray
        Finest-level transfer operator of shape (n, n_coarse), as produced by
        the prolongator step from the same aggregation.
    theta : float
        Strength-of-connection threshold, 0 < theta <= 1.
    omega : float
        Damping factor of the prolongator smoothing sweep, 0 < omega <= 1.
    block_size : int
        Size of the smoother blocks on the finest level (block_size >= 1);
        coarser levels always use scalar blocks.
    n_min : int
        Level order at or below which the operator is inverted exactly
        (n_min >= 1).
    max_levels : int
        Maximum number of coarsening steps before an exact solve
        (max_levels >= 0).

    Returns
    -------
    vcycle_operator : np.ndarray
        Matrix of shape (n, n) whose action on a vector equals one V-cycle
        applied to that vector as right-hand side with a zero initial guess.
    """
    return vcycle_operator  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_vcycle_operator(stiffness: np.ndarray, aggregates: np.ndarray,
                                     prolongator: np.ndarray, theta: float, omega: float,
                                     block_size: int, n_min: int,
                                     max_levels: int) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the coarsening steps 03 and 04 so that the hierarchy keeps no
    #    private copy of the coarsening rule. Preference order: (1) already in
    #    the executing namespace, (2) loaded from a sibling sub-problem file.
    #    The gold path never falls back to a public (submitted) implementation:
    #    binding one would make the differential comparison vacuous, so an
    #    unresolvable step raises instead.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        roots = []
        if "__file__" in namespace:
            roots.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        roots.append(os.getcwd())
        if sys.argv and sys.argv[0]:
            roots.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        # Each root, its sub_problems/ child and its parent (and that parent's
        # sub_problems/) are searched, so the gold resolves whether the harness
        # runs from the task root, from sub_problems/, or from a copy of this
        # file placed one level away from its siblings.
        for root in roots:
            parent = os.path.dirname(root)
            search_dirs += [root, os.path.join(root, "sub_problems"),
                            parent, os.path.join(parent, "sub_problems")]
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    aggregate_of = _resolve_step(
        "_oracle_build_strength_aggregates", "*build_strength_aggregates*.py")
    prolongator_of = _resolve_step(
        "_oracle_build_smoothed_prolongator", "*build_smoothed_prolongator*.py")

    def _symmetric_smoother(matrix, size):
        """Return the symmetric smoother built from the two triangular splittings."""
        n_dof = matrix.shape[0]
        width = size if (size > 1 and n_dof % size == 0) else 1
        block_of = np.arange(n_dof) // width
        diagonal_part = np.where(block_of[:, None] == block_of[None, :], matrix, 0.0)
        lower_part = np.where(block_of[:, None] > block_of[None, :], matrix, 0.0)
        upper_part = np.where(block_of[:, None] < block_of[None, :], matrix, 0.0)
        return (np.linalg.inv(diagonal_part + lower_part)
                + np.linalg.inv(diagonal_part + upper_part)
                - np.linalg.inv(diagonal_part))

    def _vcycle(matrix, transfer, level):
        """Recursively assemble the V-cycle operator of a level."""
        n_dof = matrix.shape[0]
        if n_dof <= n_min or level >= max_levels:
            return np.linalg.inv(matrix)

        if transfer is None:
            labels = aggregate_of(matrix, theta)
            if int(np.asarray(labels).max()) + 1 >= n_dof:
                return np.linalg.inv(matrix)
            transfer = prolongator_of(matrix, labels, omega)
        transfer = np.asarray(transfer, dtype=float)
        if transfer.shape[1] >= n_dof:
            return np.linalg.inv(matrix)

        coarse = transfer.T @ matrix @ transfer
        smoother = _symmetric_smoother(matrix, block_size if level == 0 else 1)
        coarse_inverse = _vcycle(coarse, None, level + 1)

        identity = np.eye(n_dof)
        # Pre-smooth from a zero guess, correct on the coarse level, then
        # post-smooth the corrected iterate against the same right-hand side.
        cycle = smoother
        coarse_correction = transfer @ coarse_inverse @ transfer.T
        cycle = cycle + coarse_correction @ (identity - matrix @ cycle)
        cycle = cycle + smoother @ (identity - matrix @ cycle)
        return cycle

    if not (isinstance(theta, (int, float)) and np.isfinite(theta)
            and 0.0 < float(theta) <= 1.0):
        raise ValueError("theta must be a finite number in the half-open interval (0, 1]")
    if not (isinstance(omega, (int, float)) and np.isfinite(omega)
            and 0.0 < float(omega) <= 1.0):
        raise ValueError("omega must be a finite number in the half-open interval (0, 1]")
    for name, value, floor in (("block_size", block_size, 1), ("n_min", n_min, 1),
                               ("max_levels", max_levels, 0)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    matrix = np.asarray(stiffness, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("stiffness must be a square 2D array of order n >= 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("stiffness must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0.0):
        raise ValueError("stiffness must be symmetric")

    labels = np.asarray(aggregates).ravel()
    if labels.size != matrix.shape[0]:
        raise ValueError("aggregates must have one entry per unknown of stiffness")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("aggregates must be an integer array")
    transfer = np.asarray(prolongator, dtype=float)
    if transfer.ndim != 2 or transfer.shape[0] != matrix.shape[0]:
        raise ValueError("prolongator must have one row per unknown of stiffness")
    if transfer.shape[1] != int(labels.max()) + 1:
        raise ValueError("prolongator must have one column per aggregate")
    if not np.all(np.isfinite(transfer)):
        raise ValueError("prolongator must be finite")

    theta = float(theta)
    omega = float(omega)
    block_size = int(block_size)
    n_min = int(n_min)
    max_levels = int(max_levels)

    try:
        return _vcycle(matrix, transfer, 0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("a level operator of the hierarchy is singular") from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark-like block operator with a rigid-body block smoother ---
        {
            "setup": """import numpy as np
n_bodies = 10
n = 6 * n_bodies
stiffness = np.zeros((n, n))
for b in range(n_bodies):
    mass = 1.0 + 0.5 * (b % 5)
    rows = np.arange(6 * b, 6 * b + 6)
    stiffness[rows, rows] += np.array([mass, mass, mass, 0.1 * mass, 0.125 * mass, 0.15 * mass]) * 4.0e6
for k in range(n_bodies):
    i, j = k, (k + 1) % n_bodies
    k_edge = 1.0e8 * 10.0 ** (3.0 * np.cos(2.0 * np.pi * k / n_bodies))
    for d in range(6):
        ii, jj = 6 * i + d, 6 * j + d
        stiffness[ii, ii] += k_edge
        stiffness[jj, jj] += k_edge
        stiffness[ii, jj] -= k_edge
        stiffness[jj, ii] -= k_edge
theta, omega, block_size, n_min, max_levels = 0.25, 2.0 / 3.0, 6, 24, 4
aggregates = _oracle_build_strength_aggregates(stiffness, theta)
prolongator = _oracle_build_smoothed_prolongator(stiffness, aggregates, omega)
""",
            "call": "assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
            "gold_call": "_oracle_assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
        },
        # --- Valid: scalar smoother and an aggressive strength threshold ---
        {
            "setup": """import numpy as np
n = 48
stiffness = np.diag(np.full(n, 4.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
theta, omega, block_size, n_min, max_levels = 0.75, 2.0 / 3.0, 1, 8, 3
aggregates = _oracle_build_strength_aggregates(stiffness, theta)
prolongator = _oracle_build_smoothed_prolongator(stiffness, aggregates, omega)
""",
            "call": "assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
            "gold_call": "_oracle_assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
        },
        # --- Boundary: level limit of zero, so the operator is inverted exactly ---
        {
            "setup": """import numpy as np
n = 16
stiffness = np.diag(np.full(n, 3.0)) - np.diag(0.5 * np.ones(n - 1), 1) - np.diag(0.5 * np.ones(n - 1), -1)
theta, omega, block_size, n_min, max_levels = 0.25, 2.0 / 3.0, 1, 4, 0
aggregates = _oracle_build_strength_aggregates(stiffness, theta)
prolongator = _oracle_build_smoothed_prolongator(stiffness, aggregates, omega)
""",
            "call": "assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
            "gold_call": "_oracle_assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
        },
        # --- Edge: strongly heterogeneous coefficients over many levels ---
        {
            "setup": """import numpy as np
n = 36
w = 10.0 ** np.linspace(-3.0, 3.0, n - 1)
stiffness = np.zeros((n, n))
for k in range(n - 1):
    stiffness[k, k] += w[k]
    stiffness[k + 1, k + 1] += w[k]
    stiffness[k, k + 1] -= w[k]
    stiffness[k + 1, k] -= w[k]
stiffness += np.eye(n) * 1.0e-2
theta, omega, block_size, n_min, max_levels = 0.25, 2.0 / 3.0, 1, 6, 5
aggregates = _oracle_build_strength_aggregates(stiffness, theta)
prolongator = _oracle_build_smoothed_prolongator(stiffness, aggregates, omega)
""",
            "call": "assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
            "gold_call": "_oracle_assemble_vcycle_operator(stiffness, aggregates, prolongator, theta, omega, block_size, n_min, max_levels)",
        },
        # --- Invalid: prolongator column count disagrees with the aggregation ---
        {
            "setup": """import numpy as np
stiffness = np.eye(6) * 2.0
aggregates = np.array([0, 0, 1, 1, 2, 2])
prolongator = np.ones((6, 2))
def run_model():
    try:
        assemble_vcycle_operator(stiffness, aggregates, prolongator, 0.25, 2.0 / 3.0, 1, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_vcycle_operator(stiffness, aggregates, prolongator, 0.25, 2.0 / 3.0, 1, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: damping factor outside its admissible interval ---
        {
            "setup": """import numpy as np
stiffness = np.eye(6) * 2.0
aggregates = np.array([0, 0, 1, 1, 2, 2])
prolongator = np.zeros((6, 3))
prolongator[np.arange(6), aggregates] = 1.0
def run_model():
    try:
        assemble_vcycle_operator(stiffness, aggregates, prolongator, 0.25, 1.5, 1, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_vcycle_operator(stiffness, aggregates, prolongator, 0.25, 1.5, 1, 2, 2)
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
