"""
Runs the full serial RGDBEK march for a fixed number of iterations: every iteration runs the column phase and then runs the row phase against the auxiliary vector the column phase just produced, never against its pre-update value, with a single random number generator threaded through every sampling draw of the whole march.

The full serial double-block extended Kaczmarz march.

Starting from the auxiliary residual vector equal to the right-hand side and the solution estimate equal to the zero vector, this step repeats the same fixed two-phase iteration for a fixed number of steps, with no tolerance-based stopping rule. Every iteration first runs the column phase, producing an updated auxiliary vector from the current one, and then runs the row phase using that just-produced, already-updated auxiliary vector -- never the value the auxiliary vector held before the column phase ran -- together with the current solution estimate, to produce the next solution estimate. This ordering is not incidental: building the row phase's weights from the stale, pre-column-phase auxiliary vector instead uses information the algorithm has already moved past, changes which rows get sampled, and can degrade or even fully break the row weight vector for particular instances, so the row phase always consumes the freshly produced auxiliary vector, not the one the iteration started with. Both sampling draws within the same iteration, and across every iteration of the march, consume the same single random number generator object, in the fixed order column block first and row block second; using a single generator threaded through the whole march, rather than a fresh generator constructed at any point along the way, is what makes the drawn index sequence -- and therefore the final result -- a deterministic function of the march seed alone.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def march_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int) -> dict:
    """Run the full serial RGDBEK march for a fixed iteration budget.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    eta : float
        Block-size fraction (must satisfy 0 < eta <= 1); the column block
        size is max(1, round(eta * n)) and the row block size is
        max(1, round(eta * m)).
    iterations : int
        Number of fixed iterations to run (must be a positive integer).
    march_seed : int
        Seed for a SINGLE `numpy.random.default_rng(march_seed)`
        generator constructed once before the loop and threaded through
        every sampling draw of the whole march, columns before rows
        within every iteration (must be a non-negative integer).

    Returns
    -------
    result : dict
        x : (n,) float array, the final solution estimate x_T.
        z : (m,) float array, the final auxiliary residual vector z_T.
        error_history : list of float, length `iterations`, the
            Euclidean norm of the solution-estimate update
            ||x_{k+1} - x_k|| at every iteration, in order (diagnostic
            only).

    Raises
    ------
    ValueError
        If A is not 2-D, if b's length does not match A's number of
        rows, if eta is not in (0, 1], if iterations is not a positive
        integer, or if march_seed is not a non-negative integer.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_march_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int) -> dict:
    """Reference implementation of march_rgdbek: chains
    _oracle_column_phase_update and _oracle_row_phase_update into the
    fixed two-phase iteration, feeding the row phase the auxiliary
    vector the column phase just produced (z_{k+1}, not the pre-update
    z_k)."""
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

    eta = float(eta)
    iterations = int(iterations)
    march_seed = int(march_seed)
    n_eta = max(1, round(eta * n))
    m_eta = max(1, round(eta * m))

    rng = np.random.default_rng(march_seed)
    col_sq_norms = np.sum(A ** 2, axis=0)
    row_sq_norms = np.sum(A ** 2, axis=1)

    x = np.zeros(n)
    z = b.copy()
    error_history = []

    for _k in range(iterations):
        z_next = _oracle_column_phase_update(A, z, col_sq_norms, n_eta, rng)
        # The row phase consumes z_next (the auxiliary vector the column
        # phase just produced this same iteration), never the pre-update z.
        x_next = _oracle_row_phase_update(A, b, z_next, x, row_sq_norms, m_eta, rng)

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
    cases: a normal small march (norm of the final x, sensitive to the
    whole chain of sampling and updates), a determinism check (two
    independent generators with the same seed must give bit-identical
    results), and edge cases (invalid eta, invalid iterations).
    """
    return [
        {
            # normal: small consistent instance, 3 iterations.
            "setup": "import numpy as np",
            "call": (
                "float(np.linalg.norm(march_rgdbek("
                "np.array([[2.0, 0.0, 1.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, -1.0, 2.0]]), "
                "np.array([3.0, 0.0, 2.0, 1.0, 2.0]), 0.5, 3, 5)['x']))"
            ),
            "gold_call": (
                "float(np.linalg.norm(_oracle_march_rgdbek("
                "np.array([[2.0, 0.0, 1.0], [0.0, 1.0, -1.0], [1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, -1.0, 2.0]]), "
                "np.array([3.0, 0.0, 2.0, 1.0, 2.0]), 0.5, 3, 5)['x']))"
            ),
        },
        {
            # boundary/determinism: bit-identical rerun of a small,
            # densely-populated 6x3 instance (no zero rows or columns, so
            # no configuration here can hit the degenerate-weights guard)
            # with the same march_seed must reproduce the same sum of x
            # (checks that the generator is the sole source of randomness
            # and consumed identically both times).
            "setup": (
                "import numpy as np\n"
                "_A5 = np.array([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0], [2.0, 0.5, 1.0], "
                "[-1.0, 1.5, 0.5], [1.5, -0.5, -1.0], [0.7, 1.0, 1.2]])\n"
                "_b5 = np.array([1.0, 0.5, 2.0, -1.0, 0.3, 1.5])"
            ),
            "call": (
                "float(np.sum(march_rgdbek(_A5, _b5, 0.34, 5, 42)['x']) "
                "- np.sum(march_rgdbek(_A5, _b5, 0.34, 5, 42)['x']))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_march_rgdbek(_A5, _b5, 0.34, 5, 42)['x']) "
                "- np.sum(_oracle_march_rgdbek(_A5, _b5, 0.34, 5, 42)['x']))"
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
                "_guard(lambda: march_rgdbek(np.array([[1.0, 0.0], [0.0, 1.0]]), "
                "np.array([1.0, 1.0]), 1.5, 2, 1))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_march_rgdbek(np.array([[1.0, 0.0], [0.0, 1.0]]), "
                "np.array([1.0, 1.0]), 1.5, 2, 1))"
            ),
        },
        {
            # edge: a non-positive iteration count is invalid
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
                "_guard(lambda: march_rgdbek(np.array([[1.0, 0.0], [0.0, 1.0]]), "
                "np.array([1.0, 1.0]), 0.5, 0, 1))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_march_rgdbek(np.array([[1.0, 0.0], [0.0, 1.0]]), "
                "np.array([1.0, 1.0]), 0.5, 0, 1))"
            ),
        },
    ]
