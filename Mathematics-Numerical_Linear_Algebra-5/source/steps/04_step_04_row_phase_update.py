"""
Runs one iteration's row phase of the serial march end to end: forms the residual-derived row sampling weight from the solution estimate and the auxiliary vector as it stands when this step is called, draws a block of row indices from it, and updates the solution estimate with a block least-squares step over the sampled rows.

Row-phase weight formation, sampling, and update for the serial march.

The row phase of an iteration takes the current solution estimate, the coefficient matrix and right-hand side, and the auxiliary residual vector exactly as it stands when the row phase runs, and produces the next solution estimate; it is the only point in the march at which the solution estimate changes. It proceeds in three parts, all internal to this step: first, a residual-derived weight is formed for every row from how badly the current solution estimate and the supplied auxiliary vector together fail to satisfy that row's equation, using the row's own precomputed squared norm as the sole per-row normalizer; second, a block of the requested size is drawn from the resulting weight vector using the sampling convention pinned elsewhere in this task; and third, the sampled block is used, together with the right-hand side, the supplied auxiliary vector, and the current solution estimate, to compute a block least-squares correction that is added to the solution estimate. The march that calls this step is what determines which value of the auxiliary vector this step receives -- this step itself takes that vector exactly as given and does not decide, or assume, whether it is the value from before or after any other phase's update. As with the column phase, the minimum-norm least-squares solve is used so the update stays well-defined even when the sampled row block happens not to be full rank. Neither the intermediate row weight nor the drawn indices are exposed by this step's return value; only the fully updated solution estimate is.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def row_phase_update(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, row_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Run one iteration's row-phase weight formation, sampling, and update.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    z : numpy.ndarray
        (m,) the auxiliary residual vector as it stands when the row
        phase runs (the caller decides which value of the auxiliary
        vector this is; this function does not assume or check it).
    x : numpy.ndarray
        (n,) current solution estimate.
    row_sq_norms : numpy.ndarray
        (m,) squared Euclidean norm of every row of A, precomputed by
        the caller (must not be recomputed from A inside this
        function).
    block_size : int
        Number of distinct row indices to sample this iteration (must
        be a positive integer no larger than the number of rows with
        strictly positive row sampling weight).
    rng : numpy.random.Generator
        A live NumPy random Generator object, consumed for this
        iteration's row-index draw using the pinned sampling
        convention. Must be the SAME generator object threaded across
        every call within a march, never a freshly seeded generator per
        call.

    Returns
    -------
    x_next : numpy.ndarray
        (n,) the solution estimate after this iteration's row-phase
        weight formation, sampling, and block least-squares update have
        all been applied.

    Raises
    ------
    ValueError
        If A is not 2-D, if b's or z's length does not match A's number
        of rows, if x's length does not match A's number of columns, if
        row_sq_norms's length does not match A's number of rows, if any
        entry of row_sq_norms is not strictly positive, if block_size is
        not a positive integer, if every resulting row weight is
        exactly zero, or if fewer than block_size rows have strictly
        positive weight.
    """
    return x_next

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


def _oracle_row_phase_update(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, row_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of row_phase_update."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    row_sq_norms = np.asarray(row_sq_norms, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    if x.shape != (n,):
        raise ValueError("x must have length equal to A's number of columns")
    if row_sq_norms.shape != (m,):
        raise ValueError("row_sq_norms must have length equal to A's number of rows")
    if np.any(row_sq_norms <= 0.0):
        raise ValueError("every entry of row_sq_norms must be strictly positive")

    resid = b - z - A @ x
    eps_x = resid ** 2 / row_sq_norms
    if float(np.sum(eps_x)) <= 0.0:
        raise ValueError("degenerate row residual: every row weight is zero")

    row_idx = _sample_block(eps_x, block_size, rng)
    A_J = A[row_idx, :]
    rhs = b[row_idx] - z[row_idx] - A_J @ x
    step = np.linalg.lstsq(A_J, rhs, rcond=None)[0]
    return x + step

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a normal instance (norm of x_next, sensitive to the
    weight formula, the sampled rows, and the update direction all at
    once), a discrimination case with duplicate sampled rows (rank
    deficient block, resolved by the minimum-norm solve rather than by
    an exception), a boundary case where one row already satisfies the
    system exactly (so it carries zero weight and can never be sampled,
    forcing the sampler onto the other row), and edge cases (a
    non-positive row_sq_norms entry, block_size exceeding the number of
    positive row weights).
    """
    return [
        {
            # normal: 4x3 A, generic b/z/x, block_size=2.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(row_phase_update("
                "np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0], [-1.0, 2.0, 1.0], [0.0, -1.0, 2.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), np.array([0.5, 0.5, -0.5, 0.0]), "
                "np.array([1.0, -1.0, 0.5]), "
                "np.array([2.0, 1.25, 6.0, 5.0]), 2, np.random.default_rng(9))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_row_phase_update("
                "np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0], [-1.0, 2.0, 1.0], [0.0, -1.0, 2.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), np.array([0.5, 0.5, -0.5, 0.0]), "
                "np.array([1.0, -1.0, 0.5]), "
                "np.array([2.0, 1.25, 6.0, 5.0]), 2, np.random.default_rng(9))))"
            ),
        },
        {
            # discrimination: rows 0 and 1 of A are identical, so a
            # sampled block covering both is rank-deficient -- a naive
            # normal-equations solve raises on the singular Gram matrix
            # while the pinned minimum-norm solve is well defined.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(row_phase_update("
                "np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]), "
                "np.array([2.0, 2.0, 1.0]), np.array([0.0, 0.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([2.0, 2.0, 1.0]), "
                "2, np.random.default_rng(0))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_row_phase_update("
                "np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]), "
                "np.array([2.0, 2.0, 1.0]), np.array([0.0, 0.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([2.0, 2.0, 1.0]), "
                "2, np.random.default_rng(0))))"
            ),
        },
        {
            # boundary: row 0 already satisfies b_0 - z_0 == A_0 @ x
            # exactly, so it carries zero weight and can never be
            # sampled -- with block_size=1 the sampler is forced onto
            # row 1, and the resulting least-squares correction must
            # drive row 1's own residual to numerically zero (fails hard
            # under a sign bug in the correction or a formula that lets
            # a zero-weight row get sampled anyway).
            "setup": "import numpy as np",
            "call": (
                "float(abs(3.0 - 0.0 - float(np.array([0.0, 1.0]) @ row_phase_update("
                "np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([4.0, 3.0]), "
                "np.array([1.0, 0.0]), np.array([1.0, 1.0]), "
                "np.array([5.0, 1.0]), 1, np.random.default_rng(0)))))"
            ),
            "gold_call": (
                "float(abs(3.0 - 0.0 - float(np.array([0.0, 1.0]) @ _oracle_row_phase_update("
                "np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([4.0, 3.0]), "
                "np.array([1.0, 0.0]), np.array([1.0, 1.0]), "
                "np.array([5.0, 1.0]), 1, np.random.default_rng(0)))))"
            ),
        },
        {
            # edge: a non-positive row_sq_norms entry is invalid
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
                "_guard(lambda: row_phase_update("
                "np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([1.0, 0.0]), 1, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_row_phase_update("
                "np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([1.0, 0.0]), 1, np.random.default_rng(1)))"
            ),
        },
        {
            # edge: block_size exceeds the number of strictly positive
            # row weights
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
                "_guard(lambda: row_phase_update("
                "np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([1.0, 1.0]), 2, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_row_phase_update("
                "np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([1.0, 1.0]), 2, np.random.default_rng(1)))"
            ),
        },
    ]
