"""
Runs one iteration's column phase of the serial march end to end: forms the residual-derived column sampling weight from the current auxiliary vector, draws a block of column indices from it, and updates the auxiliary vector using the sampled columns, returning only the updated vector.

Column-phase weight formation, sampling, and update for the serial march.

The column phase of an iteration takes the current auxiliary residual vector and the coefficient matrix, and produces the auxiliary vector the same iteration's row phase will use; this is the only point in the march at which the auxiliary vector changes. It proceeds in three parts, all internal to this step: first, a residual-derived weight is formed for every column from the matrix and the current auxiliary vector, using the column's own precomputed squared norm as the sole per-column normalizer; second, a block of the requested size is drawn from the resulting weight vector using the sampling convention pinned elsewhere in this task (sequential draws from the supplied generator, with each drawn index's weight zeroed before the next draw); and third, the sampled block is used, together with the incoming auxiliary vector alone, to produce the updated vector -- no quantity from the solution side of the march enters this update, and it is applied once to the whole vector rather than index by index. The sampled block is not guaranteed to have linearly independent columns, and this step returns a well-defined vector even when it does not. Neither the intermediate column weight nor the drawn indices are exposed by this step's return value; only the fully updated auxiliary vector is.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def column_phase_update(A: np.ndarray, z: np.ndarray, col_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Run one iteration's column-phase weight formation, sampling, and update.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    z : numpy.ndarray
        (m,) current auxiliary residual vector.
    col_sq_norms : numpy.ndarray
        (n,) squared Euclidean norm of every column of A, precomputed by
        the caller (must not be recomputed from A inside this function).
    block_size : int
        Number of distinct column indices to sample this iteration (must
        be a positive integer no larger than the number of columns with
        strictly positive column sampling weight).
    rng : numpy.random.Generator
        A live NumPy random Generator object, consumed for this
        iteration's column-index draw using the pinned sampling
        convention. Must be the SAME generator object threaded across
        every call within a march, never a freshly seeded generator per
        call.

    Returns
    -------
    z_next : numpy.ndarray
        (m,) the auxiliary vector after this iteration's column-phase
        weight formation, sampling, and update have all been applied.

    Raises
    ------
    ValueError
        If A is not 2-D, if z's length does not match A's number of
        rows, if col_sq_norms's length does not match A's number of
        columns, if any entry of col_sq_norms is not strictly positive,
        if block_size is not a positive integer, if every resulting
        column weight is exactly zero, or if fewer than block_size
        columns have strictly positive weight.
    """
    return z_next

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sample_block(weights, block_size, rng):
    """Sequential inverse-CDF sampling, the same convention pinned for
    sample_index_block, reproduced locally so this step is
    self-contained."""
    weights = np.asarray(weights, dtype=np.float64)
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(weights > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )
    w = weights.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen


def _oracle_column_phase_update(A: np.ndarray, z: np.ndarray, col_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of column_phase_update."""
    A = np.asarray(A, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    col_sq_norms = np.asarray(col_sq_norms, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    if col_sq_norms.shape != (n,):
        raise ValueError("col_sq_norms must have length equal to A's number of columns")
    if np.any(col_sq_norms <= 0.0):
        raise ValueError("every entry of col_sq_norms must be strictly positive")

    eps_z = (A.T @ z) ** 2 / col_sq_norms
    if float(np.sum(eps_z)) <= 0.0:
        raise ValueError("degenerate auxiliary vector: every column weight is zero")

    col_idx = _sample_block(eps_z, block_size, rng)
    A_U = A[:, col_idx]
    step = np.linalg.lstsq(A_U, z, rcond=None)[0]
    return z - A_U @ step

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a normal instance (norm of z_next, sensitive to the
    weight formula, the sampled columns, and the projection direction
    all at once), a discrimination case where the sampled column block
    is rank-deficient (two duplicate columns), which the minimum-norm
    solve must still resolve to a well-defined vector, a boundary case
    where the single sampled column is an exact multiple of z (z_next
    must be numerically zero), and edge cases (a non-positive
    col_sq_norms entry, block_size exceeding the number of positive
    column weights).
    """
    return [
        {
            # normal: 4x3 A, block_size=1, fixed-seed generator; the
            # returned norm is sensitive to the weight formula (which
            # column gets drawn) and to the projection direction.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(column_phase_update("
                "np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0], [2.0, -1.0, 1.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), "
                "np.array([1.0**2+0.0**2+1.0**2+2.0**2, 0.0**2+1.0**2+1.0**2+1.0**2, 2.0**2+1.0**2+0.0**2+1.0**2]), "
                "1, np.random.default_rng(3))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_column_phase_update("
                "np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0], [2.0, -1.0, 1.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), "
                "np.array([1.0**2+0.0**2+1.0**2+2.0**2, 0.0**2+1.0**2+1.0**2+1.0**2, 2.0**2+1.0**2+0.0**2+1.0**2]), "
                "1, np.random.default_rng(3))))"
            ),
        },
        {
            # discrimination: two of the three sampled columns are
            # identical (rank-deficient sampled block), so a naive
            # normal-equations solve raises on the singular Gram matrix
            # while the pinned minimum-norm solve is well defined --
            # this catches a broken solve by exception, not merely by
            # value drift.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(column_phase_update("
                "np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [0.0, 0.0, 1.0]]), "
                "np.array([1.0, 1.0, 1.0]), np.array([5.0, 5.0, 2.0]), "
                "2, np.random.default_rng(0))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_column_phase_update("
                "np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [0.0, 0.0, 1.0]]), "
                "np.array([1.0, 1.0, 1.0]), np.array([5.0, 5.0, 2.0]), "
                "2, np.random.default_rng(0))))"
            ),
        },
        {
            # boundary: z is an exact multiple of the single column that
            # gets sampled (verified to be column 0 at this seed), so it
            # lies entirely in that column's span -- z_next must be
            # numerically zero (fails hard under a sign or scaling bug).
            "setup": "import numpy as np",
            "call": (
                "float(np.max(np.abs(column_phase_update("
                "np.array([[2.0, 0.0], [0.0, 1.0], [4.0, -1.0]]), "
                "np.array([2.0, 0.0, 4.0]), np.array([20.0, 2.0]), "
                "1, np.random.default_rng(0)))))"
            ),
            "gold_call": (
                "float(np.max(np.abs(_oracle_column_phase_update("
                "np.array([[2.0, 0.0], [0.0, 1.0], [4.0, -1.0]]), "
                "np.array([2.0, 0.0, 4.0]), np.array([20.0, 2.0]), "
                "1, np.random.default_rng(0)))))"
            ),
        },
        {
            # edge: a non-positive col_sq_norms entry is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": (
                "_guard(lambda: column_phase_update("
                "np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([1.0, 1.0]), "
                "np.array([2.0, 0.0]), 1, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_column_phase_update("
                "np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([1.0, 1.0]), "
                "np.array([2.0, 0.0]), 1, np.random.default_rng(1)))"
            ),
        },
        {
            # edge: block_size exceeds the number of strictly positive
            # column weights
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": (
                "_guard(lambda: column_phase_update("
                "np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "
                "np.array([1.0, 1.0]), 2, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_column_phase_update("
                "np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "
                "np.array([1.0, 1.0]), 2, np.random.default_rng(1)))"
            ),
        },
    ]
