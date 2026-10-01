"""
Computes the convergence diagnostics that compare a finished serial march and a finished parallel march against the instance's exact targets, and against each other: the relative solution error of each march, the ratio of the parallel march's relative solution error to the serial march's, and the auxiliary-residual error ratio of each march measured against the auxiliary vector's known initial value.

Reading off both marches' results against the instance's exact targets and against each other.

Once the serial march and the parallel march have each produced a final solution estimate and a final auxiliary residual vector from the same instance, the same fixed iteration budget, and the same zero-vector initial solution estimate, this step compares every one of those four vectors against the exact targets the pinned instance guarantees: the true least-squares solution and the true optimal residual. Each march's relative solution error is the Euclidean distance from that march's final solution estimate to the true solution, divided by the Euclidean norm of the true solution itself; dividing the parallel march's relative solution error by the serial march's gives the ratio this task reports as its final answer. Each march's auxiliary-residual error ratio compares that march's final auxiliary vector to the true optimal residual, relative to how far the right-hand side itself started from that same target -- the same starting point for both marches, since both are initialized with the auxiliary vector equal to the right-hand side -- and isolates how well that march's own column phase drove the auxiliary vector toward its limit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def convergence_diagnostics(A: np.ndarray, b: np.ndarray, x_serial: np.ndarray, x_parallel: np.ndarray, z_serial: np.ndarray, z_parallel: np.ndarray, x_star: np.ndarray, r_opt: np.ndarray) -> dict:
    """Compute the convergence diagnostics of a finished serial and parallel march.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector, used as the shared reference point
        for both marches' auxiliary-residual-error-ratio denominators
        (both marches are initialized with the auxiliary vector equal
        to b).
    x_serial : numpy.ndarray
        (n,) final solution estimate from the serial march.
    x_parallel : numpy.ndarray
        (n,) final solution estimate from the parallel march.
    z_serial : numpy.ndarray
        (m,) final auxiliary residual vector from the serial march.
    z_parallel : numpy.ndarray
        (m,) final auxiliary residual vector from the parallel march.
    x_star : numpy.ndarray
        (n,) exact least-squares solution.
    r_opt : numpy.ndarray
        (m,) exact optimal residual.

    Returns
    -------
    diagnostics : dict
        rel_error_serial : float, ||x_serial - x_star||_2 / ||x_star||_2.
        rel_error_parallel : float, ||x_parallel - x_star||_2 / ||x_star||_2.
        error_ratio : float, rel_error_parallel / rel_error_serial.
        z_error_ratio_serial : float, ||z_serial - r_opt||_2 / ||b - r_opt||_2.
        z_error_ratio_parallel : float, ||z_parallel - r_opt||_2 / ||b - r_opt||_2.

    Raises
    ------
    ValueError
        If A is not 2-D, if b's or r_opt's length does not match A's
        number of rows, if x_serial's, x_parallel's, or x_star's length
        does not match A's number of columns, if z_serial's or
        z_parallel's length does not match A's number of rows, if
        x_star has zero norm, if b equals r_opt exactly (a degenerate
        z-error-ratio denominator), or if x_serial equals x_star exactly
        (a degenerate error_ratio denominator).
    """
    return diagnostics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_convergence_diagnostics(A: np.ndarray, b: np.ndarray, x_serial: np.ndarray, x_parallel: np.ndarray, z_serial: np.ndarray, z_parallel: np.ndarray, x_star: np.ndarray, r_opt: np.ndarray) -> dict:
    """Reference implementation of convergence_diagnostics."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_serial = np.asarray(x_serial, dtype=np.float64)
    x_parallel = np.asarray(x_parallel, dtype=np.float64)
    z_serial = np.asarray(z_serial, dtype=np.float64)
    z_parallel = np.asarray(z_parallel, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    r_opt = np.asarray(r_opt, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if r_opt.shape != (m,):
        raise ValueError("r_opt must have length equal to A's number of rows")
    if z_serial.shape != (m,):
        raise ValueError("z_serial must have length equal to A's number of rows")
    if z_parallel.shape != (m,):
        raise ValueError("z_parallel must have length equal to A's number of rows")
    if x_serial.shape != (n,):
        raise ValueError("x_serial must have length equal to A's number of columns")
    if x_parallel.shape != (n,):
        raise ValueError("x_parallel must have length equal to A's number of columns")
    if x_star.shape != (n,):
        raise ValueError("x_star must have length equal to A's number of columns")
    x_star_norm = np.linalg.norm(x_star)
    if x_star_norm == 0.0:
        raise ValueError("x_star must have nonzero norm")
    z0_minus_r = b - r_opt
    denom = np.linalg.norm(z0_minus_r)
    if denom == 0.0:
        raise ValueError("b must not equal r_opt exactly (degenerate z-error-ratio denominator)")

    rel_error_serial = float(np.linalg.norm(x_serial - x_star) / x_star_norm)
    if rel_error_serial == 0.0:
        raise ValueError("x_serial must not equal x_star exactly (degenerate error_ratio denominator)")
    rel_error_parallel = float(np.linalg.norm(x_parallel - x_star) / x_star_norm)
    error_ratio = rel_error_parallel / rel_error_serial
    z_error_ratio_serial = float(np.linalg.norm(z_serial - r_opt) / denom)
    z_error_ratio_parallel = float(np.linalg.norm(z_parallel - r_opt) / denom)

    return {
        "rel_error_serial": rel_error_serial,
        "rel_error_parallel": rel_error_parallel,
        "error_ratio": error_ratio,
        "z_error_ratio_serial": z_error_ratio_serial,
        "z_error_ratio_parallel": z_error_ratio_parallel,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a normal small instance (sum of all five diagnostics,
    sensitive to every formula), a boundary case where the parallel
    march's outputs exactly equal the exact targets while the serial
    march's do not (rel_error_parallel and z_error_ratio_parallel must
    be exactly zero, while error_ratio stays a well-defined nonzero
    number), and edge cases (zero-norm x_star, b == r_opt, and
    x_serial == x_star exactly, the new degenerate case this step's
    error_ratio introduces).
    """
    return [
        {
            # normal: 3x2 instance with nonzero errors on both marches.
            "setup": "import numpy as np",
            "call": (
                "float(sum(convergence_diagnostics("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0, 2.5]), "
                "np.array([0.8, 0.9]), np.array([0.6, 1.2]), "
                "np.array([0.1, 0.1, 0.3]), np.array([0.2, -0.1, 0.4]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0, 0.5])).values()))"
            ),
            "gold_call": (
                "float(sum(_oracle_convergence_diagnostics("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0, 2.5]), "
                "np.array([0.8, 0.9]), np.array([0.6, 1.2]), "
                "np.array([0.1, 0.1, 0.3]), np.array([0.2, -0.1, 0.4]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0, 0.5])).values()))"
            ),
        },
        {
            # boundary: x_parallel == x_star and z_parallel == r_opt
            # exactly, while the serial march's outputs are not exact --
            # rel_error_parallel and z_error_ratio_parallel must both be
            # exactly zero (fails hard under a sign or swapped-argument
            # bug), while error_ratio remains well-defined at 0.
            "setup": "import numpy as np",
            "call": (
                "convergence_diagnostics("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0, 2.5]), "
                "np.array([0.8, 0.9]), np.array([1.0, 1.0]), "
                "np.array([0.1, 0.1, 0.3]), np.array([0.0, 0.0, 0.5]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0, 0.5]))['rel_error_parallel'] "
                "+ convergence_diagnostics("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0, 2.5]), "
                "np.array([0.8, 0.9]), np.array([1.0, 1.0]), "
                "np.array([0.1, 0.1, 0.3]), np.array([0.0, 0.0, 0.5]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0, 0.5]))['z_error_ratio_parallel']"
            ),
            "gold_call": (
                "_oracle_convergence_diagnostics("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0, 2.5]), "
                "np.array([0.8, 0.9]), np.array([1.0, 1.0]), "
                "np.array([0.1, 0.1, 0.3]), np.array([0.0, 0.0, 0.5]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0, 0.5]))['rel_error_parallel'] "
                "+ _oracle_convergence_diagnostics("
                "np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0, 2.5]), "
                "np.array([0.8, 0.9]), np.array([1.0, 1.0]), "
                "np.array([0.1, 0.1, 0.3]), np.array([0.0, 0.0, 0.5]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0, 0.5]))['z_error_ratio_parallel']"
            ),
        },
        {
            # edge: x_star with zero norm is invalid
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
                "_guard(lambda: convergence_diagnostics("
                "np.eye(2), np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.5, 0.5])))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_convergence_diagnostics("
                "np.eye(2), np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.0, 0.0]), "
                "np.array([0.0, 0.0]), np.array([0.5, 0.5])))"
            ),
        },
        {
            # edge: b == r_opt exactly is invalid (degenerate z-ratio
            # denominator)
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
                "_guard(lambda: convergence_diagnostics("
                "np.eye(2), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), "
                "np.array([1.0, 1.0]), np.array([1.0, 1.0]), "
                "np.array([1.0, 1.0]), np.array([1.0, 1.0])))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_convergence_diagnostics("
                "np.eye(2), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), "
                "np.array([1.0, 1.0]), np.array([1.0, 1.0]), "
                "np.array([1.0, 1.0]), np.array([1.0, 1.0])))"
            ),
        },
        {
            # edge: x_serial == x_star exactly is invalid (degenerate
            # error_ratio denominator) -- the new degenerate case this
            # step's ratio introduces beyond the original diagnostics.
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
                "_guard(lambda: convergence_diagnostics("
                "np.eye(2), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), "
                "np.array([0.0, 0.0]), np.array([0.1, 0.1]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0])))"
            ),
            "gold_call": (
                "_guard(lambda: _oracle_convergence_diagnostics("
                "np.eye(2), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), "
                "np.array([0.0, 0.0]), np.array([0.1, 0.1]), "
                "np.array([1.0, 1.0]), np.array([0.0, 0.0])))"
            ),
        },
    ]
