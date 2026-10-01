"""
Runs the full hybrid-parallel RGDBEK march for a fixed number of iterations: every iteration runs the parallel column phase and then runs the parallel row phase against the auxiliary vector the column phase just produced, never against its pre-update value, with a single random number generator threaded through every sampling draw of every rank of the whole march.

The full hybrid-parallel double-block extended Kaczmarz march.

Starting from the auxiliary residual vector equal to the right-hand side and the solution estimate equal to the zero vector, exactly as the serial march does, this step repeats the same fixed two-phase iteration for the same fixed number of steps, over the same fixed rank partition throughout. Every iteration first runs the parallel column phase, producing an updated auxiliary vector from the current one, and then runs the parallel row phase using that just-produced, already-updated auxiliary vector -- never the value the auxiliary vector held before the column phase ran -- together with the current solution estimate, to produce the next solution estimate; this is the same timing rule the serial march follows, carried over unchanged to the parallel formulation. The column-block size is the same fixed count used for the whole matrix at once, since the column phase's weight formation and draw are shared across every rank; the row-block size is the fixed per-rank count each rank draws from its own rows. A single random number generator, distinct from the serial march's own generator, is constructed once before this march's loop begins and threaded through every draw of every iteration: the shared column-index draw, then every rank's own row-index draw in increasing rank order, iteration after iteration, so the entire sequence of draws made by every rank over the whole march is a deterministic function of this march's own seed alone.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def march_parallel_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int, n_ranks: int = 4) -> dict:
    """Run the full hybrid-parallel RGDBEK march for a fixed iteration budget.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    eta : float
        Block-size fraction (must satisfy 0 < eta <= 1); the shared
        column block size is max(1, round(eta * n)) and each rank's own
        row block size is max(1, round(eta * m / n_ranks)), the same
        fixed count for every rank regardless of that rank's own row
        share.
    iterations : int
        Number of fixed iterations to run (must be a positive integer).
    march_seed : int
        Seed for a SINGLE `numpy.random.default_rng(march_seed)`
        generator constructed once before the loop and threaded through
        every sampling draw of every rank of the whole march, the
        shared column draw before any rank's row draw within every
        iteration, and every rank's row draw in increasing rank order
        (must be a non-negative integer). This must be a separate
        generator object from the one the serial march uses.
    n_ranks : int
        Number of ranks the row dimension is partitioned into (must be
        a positive integer no greater than A's number of rows; ranks
        own a fixed, contiguous partition of the rows, determined by
        the rule named in the configuration). Defaults to the
        benchmark's pinned rank count.

    Returns
    -------
    result : dict
        x : (n,) float array, the final solution estimate x_T^par.
        z : (m,) float array, the final auxiliary residual vector z_T^par.
        error_history : list of float, length `iterations`, the
            Euclidean norm of the solution-estimate update
            ||x_{k+1} - x_k|| at every iteration, in order (diagnostic
            only).

    Raises
    ------
    ValueError
        If A is not 2-D, if b's length does not match A's number of
        rows, if eta is not in (0, 1], if iterations is not a positive
        integer, if march_seed is not a non-negative integer, or if
        n_ranks is not a positive integer no greater than A's number of
        rows.
    """
    return result

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


def _oracle_march_parallel_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int, n_ranks: int = 4) -> dict:
    """Reference implementation of march_parallel_rgdbek: chains
    _oracle_parallel_column_phase and _oracle_parallel_row_phase into the
    fixed two-phase iteration, feeding the row phase the auxiliary vector
    the column phase just produced (z_{k+1}, not the pre-update z_k), the
    same timing rule the serial march follows."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if not (isinstance(eta, (int, float, np.floating, np.integer)) and 0.0 < float(eta) <= 1.0):
        raise ValueError("eta must satisfy 0 < eta <= 1")
    if not isinstance(iterations, (int, np.integer)) or int(iterations) < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(march_seed, (int, np.integer)) or int(march_seed) < 0:
        raise ValueError("march_seed must be a non-negative integer")
    _row_block_bounds(A, n_ranks)  # validates n_ranks against A up front
    n_ranks = int(n_ranks)

    eta = float(eta)
    iterations = int(iterations)
    march_seed = int(march_seed)
    n_eta = max(1, round(eta * n))
    d_eta = max(1, round(eta * m / n_ranks))

    rng = np.random.default_rng(march_seed)
    x = np.zeros(n)
    z = b.copy()
    error_history = []

    for _k in range(iterations):
        z_next = _oracle_parallel_column_phase(A, z, n_ranks, n_eta, rng)
        # The row phase consumes z_next (the auxiliary vector the column
        # phase just produced this same iteration), never the pre-update z.
        x_next = _oracle_parallel_row_phase(A, b, z_next, x, n_ranks, d_eta, rng)

        error_history.append(float(np.linalg.norm(x_next - x)))
        x = x_next
        z = z_next

    return {"x": x, "z": z, "error_history": error_history}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Tiny instances keep every case cheap for the mutation battery. >= 3
    cases: a normal small parallel march (norm of the final x, sensitive
    to the whole chain of shared and per-rank sampling and updates), a
    determinism check (two independent generators with the same seed
    must give bit-identical results), and edge cases (eta outside its
    valid range, n_ranks greater than the row count).
    """
    return [
        {
            # normal: 8x4 A, n_ranks=2, 3 iterations.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(march_parallel_rgdbek("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0], "
                "[1.5, -0.5, 0.5, 1.0], [-0.5, 1.5, 1.0, 0.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0, 0.3, -0.7]), "
                "0.5, 3, 5, n_ranks=2)['x']))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_march_parallel_rgdbek("
                "np.array([[1.0, 2.0, 0.0, -1.0], [0.5, -1.0, 2.0, 1.0], "
                "[2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.5, 2.0], "
                "[1.0, 1.0, -1.0, 0.0], [0.0, 2.0, 1.0, -1.0], "
                "[1.5, -0.5, 0.5, 1.0], [-0.5, 1.5, 1.0, 0.0]]), "
                "np.array([1.0, -2.0, 0.5, 1.5, -1.0, 2.0, 0.3, -0.7]), "
                "0.5, 3, 5, n_ranks=2)['x']))"
            ),
        },
        {
            # determinism: bit-identical rerun of an 8x3 instance, n_ranks=4,
            # with the same march_seed must reproduce the same sum of x.
            "setup": (
                "import numpy as np\n"
                "_A8 = np.array([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [2.0, 0.5, 1.0], "
                "[-1.0, 1.5, 0.5], [1.5, -0.5, -1.0], [0.7, 1.0, 1.2], "
                "[0.3, -0.2, 1.1], [1.1, 0.4, -0.6]])\n"
                "_b8 = np.array([1.0, 0.5, 2.0, -1.0, 0.3, 1.5, -0.4, 0.9])"
            ),
            "call": (
                "float(np.sum(march_parallel_rgdbek(_A8, _b8, 0.34, 4, 42, n_ranks=4)['x']) "
                "- np.sum(march_parallel_rgdbek(_A8, _b8, 0.34, 4, 42, n_ranks=4)['x']))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_march_parallel_rgdbek(_A8, _b8, 0.34, 4, 42, n_ranks=4)['x']) "
                "- np.sum(_oracle_march_parallel_rgdbek(_A8, _b8, 0.34, 4, 42, n_ranks=4)['x']))"
            ),
        },
        {
            # edge: eta outside (0, 1] is invalid
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
                "_guard(lambda: march_parallel_rgdbek("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]), "
                "np.array([1.0, 1.0, 2.0, 0.0]), 1.5, 2, 1, n_ranks=2))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_march_parallel_rgdbek("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]), "
                "np.array([1.0, 1.0, 2.0, 0.0]), 1.5, 2, 1, n_ranks=2))"
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
                "_guard(lambda: march_parallel_rgdbek("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]), "
                "np.array([1.0, 1.0, 2.0, 0.0]), 0.5, 2, 1, n_ranks=5))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_march_parallel_rgdbek("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]), "
                "np.array([1.0, 1.0, 2.0, 0.0]), 0.5, 2, 1, n_ranks=5))"
            ),
        },
    ]
