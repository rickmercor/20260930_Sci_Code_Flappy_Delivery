"""
Computes one iteration's row-phase update of the solution estimate for the hybrid-parallel march, applying the row phase separately within each rank of the fixed rank partition, each rank drawing its own row-index block from the shared sampling stream restricted to its own rows, in increasing rank order, to produce the iteration's updated solution estimate.

Row-phase update of the solution estimate for the hybrid-parallel march.

Each iteration's row phase in the parallel formulation runs separately within each rank of the fixed, contiguous rank partition named in the configuration: rank by rank, in increasing rank order, each rank draws its own row-index block from the march's shared sampling stream, with that draw restricted to the rows belonging to that rank alone, independent of every other rank's draw, exactly as the draw order and partition are specified for the parallel march as a whole. What is not stated here is how each rank forms its own row sampling weight from its own portion of the problem, how each rank's own block least-squares correction is computed from its own sampled rows, and how the ranks' independently computed corrections are combined into the iteration's single updated solution estimate; derive these from the phase name, the partition, the draw order, and the role the single-rank row phase's own update plays in that formulation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parallel_row_phase(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, n_ranks: int, block_size_per_rank: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one iteration's row-phase update in the hybrid-parallel march.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    z : numpy.ndarray
        (m,) the auxiliary residual vector as it stands when the row
        phase runs, using the same row ordering as A and reflecting the
        fixed, contiguous rank partition named in the configuration.
    x : numpy.ndarray
        (n,) current solution estimate.
    n_ranks : int
        Number of ranks the row dimension is partitioned into. Must be
        a positive integer no greater than A's number of rows; ranks
        own a fixed, contiguous partition of the rows, determined by
        the rule named in the configuration.
    block_size_per_rank : int
        Number of distinct row indices each rank samples this
        iteration, from its own rows only. Must be a positive integer
        no larger than the number of rows, within any single rank's
        block, with strictly positive row sampling weight.
    rng : numpy.random.Generator
        The parallel march's own shared sampling stream (the same
        sampling convention as the single-rank formulation's draws),
        consumed for every rank's row-index draw this iteration, rank
        by rank in increasing rank order.

    Returns
    -------
    x_next : numpy.ndarray
        (n,) the solution estimate after this iteration's parallel
        row-phase update has been applied.

    Raises
    ------
    ValueError
        If A is not 2-D, if b's or z's length does not match A's number
        of rows, if x's length does not match A's number of columns, if
        n_ranks is not a positive integer no greater than A's number of
        rows, if block_size_per_rank is not a positive integer, or if
        any rank's own row sampling weights are all exactly zero or
        number fewer than block_size_per_rank strictly positive
        entries.
    """
    return x_next

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _row_block_bounds(A, n_ranks):
    """Contiguous, nonzero-count-balanced row partition of A's rows into
    n_ranks blocks, in index order (S22: same-named function redefined
    elsewhere must be byte-identical). Row i's nonzero count c_i is
    computed once over the whole matrix; C = sum(c_i) is the total
    nonzero count. The running prefix sum S(i) = c_0 + ... + c_i is a
    single monotone pass over the whole matrix, taken from row 0 and
    never reset at a block boundary. For p = 0, ..., n_ranks - 2 (in
    increasing p order), boundary p closes immediately after the first
    row index i such that S(i) >= (p + 1) * C / n_ranks (i.e. the first
    row whose global cumulative nonzero count reaches or exceeds a
    (p + 1) / n_ranks share of the whole matrix's total nonzero count);
    the final rank takes every remaining row. Non-degeneracy guard,
    applied left to right in increasing p order: if the scan would ever
    produce an empty block, the boundary is pulled forward by one row so
    every rank owns at least one row."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if not isinstance(n_ranks, (int, np.integer)) or int(n_ranks) < 1:
        raise ValueError("n_ranks must be a positive integer")
    n_ranks = int(n_ranks)
    if m < n_ranks:
        raise ValueError(f"m={m} rows fewer than n_ranks={n_ranks}")
    c = np.count_nonzero(A, axis=1).astype(np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("A has no nonzero entries; cannot balance the row partition")
    S = np.cumsum(c)
    bounds = []
    lo = 0
    for p in range(n_ranks - 1):
        target = (p + 1) * C / n_ranks
        i = int(np.searchsorted(S, target, side="left"))
        i = min(i, m - 1)
        hi = i + 1
        if hi <= lo:
            hi = lo + 1
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, m))
    return bounds


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


def _oracle_parallel_row_phase(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, n_ranks: int, block_size_per_rank: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of parallel_row_phase."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    if x.shape != (n,):
        raise ValueError("x must have length equal to A's number of columns")
    bounds = _row_block_bounds(A, n_ranks)
    n_ranks = int(n_ranks)

    updates = []
    for (lo, hi) in bounds:
        A_p = A[lo:hi, :]
        b_p = b[lo:hi]
        z_p = z[lo:hi]
        row_sq_norms_p = np.sum(A_p ** 2, axis=1)
        if np.any(row_sq_norms_p <= 0.0):
            raise ValueError("every row of A must have strictly positive squared norm")
        resid_p = b_p - z_p - A_p @ x
        eps_x_p = resid_p ** 2 / row_sq_norms_p
        if float(np.sum(eps_x_p)) <= 0.0:
            raise ValueError("degenerate row residual on one rank: every row weight is zero")

        idx_p = _sample_block(eps_x_p, block_size_per_rank, rng)
        A_Jp = A_p[idx_p, :]
        rhs = b_p[idx_p] - z_p[idx_p] - A_Jp @ x
        step_p = np.linalg.lstsq(A_Jp, rhs, rcond=None)[0]
        updates.append(step_p)

    x_next = x + np.sum(updates, axis=0) / n_ranks
    return x_next

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a normal case (8x4 A, n_ranks=2), a boundary case where
    n_ranks=1 collapses the rank partition to a single block, so every
    rank's own local update reduces exactly to the single-rank row-phase
    update's formula applied over all rows at once (a degenerate
    partition is a correctness anchor, not a separate mechanism -- this
    also incidentally catches "corrections summed instead of averaged",
    since n_ranks=1 makes summing and averaging coincide and only the
    normal case at n_ranks=2 can separate them), a discrimination case
    on a sparse instance whose row-wise nonzero counts are deliberately
    unequal, so the fixed rank partition differs from what an
    equal-row-count partition would produce (this case fails under an
    equal-row-count partition, since that wrong partition changes which
    rows fall on which side of the rank boundary, and therefore which
    rows compete for each rank's own draw), and edge cases (n_ranks
    greater than the row count, a degenerate rank whose own residual is
    exactly zero everywhere).
    """
    return [
        {
            # normal: 8x4 A, n_ranks=2 (4 rows per rank), one row sampled
            # per rank; sensitive to both ranks' own sampled rows and to
            # the 1/P averaging of their independent corrections.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(parallel_row_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0], "
                "[1.5, -0.5, 0.5, 1.0], [-0.5, 1.5, 1.0, 0.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0, 0.3, -0.7]), "
                "np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.05, 0.15]), "
                "np.array([0.5, -0.5, 0.2, 0.1]), 2, 1, np.random.default_rng(6))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_parallel_row_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0], "
                "[1.5, -0.5, 0.5, 1.0], [-0.5, 1.5, 1.0, 0.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0, 0.3, -0.7]), "
                "np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.05, 0.15]), "
                "np.array([0.5, -0.5, 0.2, 0.1]), 2, 1, np.random.default_rng(6))))"
            ),
        },
        {
            # boundary: n_ranks=1 -- the single "rank" owns every row,
            # so the parallel update must coincide exactly with the
            # serial row-phase update's own formula over the whole
            # 4x2 instance (the discrimination axis this anchors is
            # "sum vs 1/P average", since at n_ranks=1 both coincide but
            # would diverge from the correct answer under a formula
            # that always divides by, or always multiplies by, a fixed
            # P regardless of n_ranks).
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(parallel_row_phase("
                "np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0], [-1.0, 2.0, 1.0], [0.0, -1.0, 2.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), np.array([0.5, 0.5, -0.5, 0.0]), "
                "np.array([1.0, -1.0, 0.5]), 1, 2, np.random.default_rng(9))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_parallel_row_phase("
                "np.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0], [-1.0, 2.0, 1.0], [0.0, -1.0, 2.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), np.array([0.5, 0.5, -0.5, 0.0]), "
                "np.array([1.0, -1.0, 0.5]), 1, 2, np.random.default_rng(9))))"
            ),
        },
        {
            # discrimination: 6x3 A, deliberately unequal row-wise nonzero
            # counts (row-wise nonzero counts [3, 3, 1, 1, 1, 2], total 11),
            # n_ranks=2, one row sampled per rank. The pinned
            # nonzero-balanced partition places the boundary at row index 2
            # ([(0, 2), (2, 6)]), while an equal-row-count partition would
            # place it at row index 3 ([(0, 3), (3, 6)]); with this rng seed
            # the two partitions give mathematically distinct results
            # (~0.5449442959737131 vs ~0.4301162633521313), so this case
            # fails under an equal-row-count partition.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(parallel_row_phase("
                "np.array([[1.0, 0.5, -0.5], [0.5, -1.0, 1.5], "
                "[0.0, 2.0, 0.0], [0.0, 0.0, 1.5], "
                "[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]), "
                "np.array([1.0, -0.5, 2.0, 0.7, 0.3, -0.2]), "
                "np.array([0.2, -0.1, 0.4, 0.0, 0.1, -0.2]), "
                "np.array([0.5, -0.5, 0.2]), 2, 1, np.random.default_rng(6))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_parallel_row_phase("
                "np.array([[1.0, 0.5, -0.5], [0.5, -1.0, 1.5], "
                "[0.0, 2.0, 0.0], [0.0, 0.0, 1.5], "
                "[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]), "
                "np.array([1.0, -0.5, 2.0, 0.7, 0.3, -0.2]), "
                "np.array([0.2, -0.1, 0.4, 0.0, 0.1, -0.2]), "
                "np.array([0.5, -0.5, 0.2]), 2, 1, np.random.default_rng(6))))"
            ),
        },
        {
            # edge: n_ranks greater than A's row count
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
                "_guard(lambda: parallel_row_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0]), "
                "np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "
                "np.array([0.5, -0.5, 0.2, 0.1]), 7, 1, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_parallel_row_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0]), "
                "np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "
                "np.array([0.5, -0.5, 0.2, 0.1]), 7, 1, np.random.default_rng(1)))"
            ),
        },
        {
            # edge: one rank's own residual is exactly zero everywhere
            # (rank 0's two rows already satisfy b - z == A @ x exactly),
            # degenerating that rank's own row weight vector to all zero
            # -- a global (pooled-across-ranks) weight vector would not
            # be degenerate here, so this also anchors "rank-local"
            # weight normalization.
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
                "_guard(lambda: parallel_row_phase("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]), "
                "np.array([1.0, 0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0, 0.0]), "
                "np.array([1.0, 0.0]), 2, 1, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_parallel_row_phase("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]), "
                "np.array([1.0, 0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0, 0.0]), "
                "np.array([1.0, 0.0]), 2, 1, np.random.default_rng(1)))"
            ),
        },
    ]
