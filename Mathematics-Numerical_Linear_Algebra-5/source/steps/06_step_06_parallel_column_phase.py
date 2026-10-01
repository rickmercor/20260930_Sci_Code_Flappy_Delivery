"""
Computes one iteration's column-phase update of the auxiliary residual vector for the hybrid-parallel march, using the same residual-derived column sampling weight and the same shared draw as the single-rank formulation, then producing the updated auxiliary vector across the fixed rank partition.

Column-phase update of the auxiliary residual vector for the hybrid-parallel march.

Each iteration's column phase in the parallel formulation begins exactly as it does in the single-rank formulation: the residual-derived weight of column j is formed from the current auxiliary residual vector, taken as a whole across every rank, using the identical formula the single-rank formulation uses, and the iteration's column-index block is drawn from the resulting distribution using the march's shared sampling stream. Neither this weight formation nor this draw depends on the rank partition at all -- they use the coefficient matrix and the auxiliary vector as a whole, exactly as the single-rank column phase does, and this part is not something the parallel formulation changes. What the parallel formulation does change is what happens next: the auxiliary vector's row dimension is split across the fixed, contiguous rank partition named in the configuration, and the sampled column block is then used to produce the iteration's updated auxiliary vector. Exactly how the sampled columns and the rank partition are used to form the iteration's updated auxiliary vector, including whether any part of that update is computed separately per rank and whether or how such parts interact before the updated vector is assembled, is not stated here; derive it from the phase name, the partition, the draw order, and the role the single-rank column phase's own update plays in that formulation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parallel_column_phase(A: np.ndarray, z: np.ndarray, n_ranks: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one iteration's column-phase update in the hybrid-parallel march.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    z : numpy.ndarray
        (m,) current auxiliary residual vector, using the same row
        ordering as A.
    n_ranks : int
        Number of ranks the row dimension is partitioned into. Must be
        a positive integer no greater than A's number of rows; ranks
        own a fixed, contiguous partition of the rows, determined by
        the rule named in the configuration.
    block_size : int
        Number of distinct column indices to sample this iteration.
        Must be a positive integer no larger than the number of
        columns with strictly positive column sampling weight.
    rng : numpy.random.Generator
        The parallel march's own shared sampling stream (the same
        sampling convention as the single-rank formulation's draws),
        consumed for this iteration's column-index draw.

    Returns
    -------
    z_next : numpy.ndarray
        (m,) the auxiliary vector after this iteration's column-phase
        update has been applied, in the same row ordering as z.

    Raises
    ------
    ValueError
        If A is not 2-D, if z's length does not match A's number of
        rows, if n_ranks is not a positive integer no greater than A's
        number of rows, if block_size is not a positive integer no
        larger than the number of columns with strictly positive
        column sampling weight, or if any column of A has a zero
        squared Euclidean norm.
    """
    return z_next

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


def _oracle_parallel_column_phase(A: np.ndarray, z: np.ndarray, n_ranks: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of parallel_column_phase."""
    A = np.asarray(A, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    bounds = _row_block_bounds(A, n_ranks)

    col_sq_norms = np.sum(A ** 2, axis=0)
    if np.any(col_sq_norms <= 0.0):
        raise ValueError("every column of A must have strictly positive squared norm")
    eps_z = (A.T @ z) ** 2 / col_sq_norms
    col_idx = _sample_block(eps_z, block_size, rng)

    z_next = z.copy()
    for (lo, hi) in bounds:
        A_p = A[lo:hi, :]
        z_p = z[lo:hi]
        A_p_U = A_p[:, col_idx]
        z_sol_p = np.linalg.lstsq(A_p_U, z_p, rcond=None)[0]
        z_next[lo:hi] = z_p - A_p_U @ z_sol_p
    return z_next

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a discrimination case that separates the parallel
    formulation's rank-local update from the single-rank formulation's
    global projection (n_ranks=2 vs the mathematically distinct
    global-projection value on the same A, z, and sampled columns), a
    boundary case where n_ranks=1 collapses the rank partition to a
    single block so the result must coincide exactly with the
    single-rank global projection, a discrimination case on a sparse
    instance whose row-wise nonzero counts are deliberately unequal, so
    the fixed rank partition differs from what an equal-row-count
    partition would produce (this case fails under an equal-row-count
    partition, since that wrong partition changes which rows fall on
    which side of the rank boundary and therefore changes the returned
    vector), and edge cases (n_ranks greater than the number of rows, a
    z whose length does not match A's row count).
    """
    return [
        {
            # discrimination: A is 4x2 split into 2 ranks of 2 rows each;
            # with the sampled column fixed by the rng draw, the
            # rank-local update (each rank projects its own 2x1 local
            # block) gives a different z_next than the single-rank
            # formulation's global projection would on the same A, z,
            # and sampled column -- separates "each rank computes its
            # own local least-squares correction" from "one global
            # least-squares correction applied to the whole vector".
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(parallel_column_phase("
                "np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]), "
                "np.array([1.0, 1.0, 1.0, 1.0]), 2, 1, np.random.default_rng(5))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_parallel_column_phase("
                "np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]), "
                "np.array([1.0, 1.0, 1.0, 1.0]), 2, 1, np.random.default_rng(5))))"
            ),
        },
        {
            # boundary: n_ranks=1, so the single "rank" owns every row --
            # the rank-local update must coincide exactly with the
            # single-rank formulation's global projection on the same
            # A, z, and sampled columns (a degenerate partition is a
            # correctness anchor, not a separate mechanism).
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(parallel_column_phase("
                "np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0], [2.0, -1.0, 1.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), 1, 2, np.random.default_rng(9))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_parallel_column_phase("
                "np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0], [2.0, -1.0, 1.0]]), "
                "np.array([1.0, 2.0, -1.0, 0.5]), 1, 2, np.random.default_rng(9))))"
            ),
        },
        {
            # discrimination: 4x3 A, deliberately unequal row-wise nonzero
            # counts (row 0 dense with 3 nonzeros, rows 1-3 each with a
            # single nonzero: nonzero counts [3, 1, 1, 1], total 6), n_ranks=2.
            # The pinned nonzero-balanced partition places the boundary at
            # row index 1 ([(0, 1), (1, 4)]), while an equal-row-count
            # partition would place it at row index 2 ([(0, 2), (2, 4)]);
            # with this rng seed the two partitions give mathematically
            # distinct rank-local column projections (1.25 vs
            # ~1.5109891579009156), so this case fails under an
            # equal-row-count partition.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(parallel_column_phase("
                "np.array([[1.0, 0.5, -0.5], [0.0, 2.0, 0.0], "
                "[0.0, 0.0, 1.5], [1.0, 0.0, 0.0]]), "
                "np.array([1.0, 0.5, -1.0, 0.75]), 2, 1, np.random.default_rng(1))))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_parallel_column_phase("
                "np.array([[1.0, 0.5, -0.5], [0.0, 2.0, 0.0], "
                "[0.0, 0.0, 1.5], [1.0, 0.0, 0.0]]), "
                "np.array([1.0, 0.5, -1.0, 0.75]), 2, 1, np.random.default_rng(1))))"
            ),
        },
        {
            # edge: n_ranks greater than A's number of rows
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
                "_guard(lambda: parallel_column_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0]), 7, 2, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_parallel_column_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0]), 7, 2, np.random.default_rng(1)))"
            ),
        },
        {
            # edge: z's length does not match A's number of rows
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
                "_guard(lambda: parallel_column_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]), "
                "np.array([1.0, -2.0, 0.5]), 2, 2, np.random.default_rng(1)))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_parallel_column_phase("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0]]), "
                "np.array([1.0, -2.0, 0.5]), 2, 2, np.random.default_rng(1)))"
            ),
        },
    ]
