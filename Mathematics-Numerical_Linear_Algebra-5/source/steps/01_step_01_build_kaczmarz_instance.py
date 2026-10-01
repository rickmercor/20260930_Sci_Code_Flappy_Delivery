"""
Builds a fully pinned, deterministically inconsistent least-squares instance for the double-block extended Kaczmarz march: a sparse random coefficient matrix, a right-hand side whose noise component is exactly orthogonal to the matrix's column space, and the derived scalars every later step needs to sample and solve against it.

Deterministic instance construction for an inconsistent least-squares system.

Extended Kaczmarz methods are benchmarked against systems that are exactly inconsistent by design, so that the method's convergence toward the true least-squares solution and toward the true optimal residual can both be checked against known targets rather than against numbers estimated after the fact. This step draws a sparse random coefficient matrix and a ground-truth solution, then builds a noise vector that is exactly orthogonal to the coefficient matrix's column space and scales it to a fixed fraction of the noiseless signal, so that the resulting right-hand side decomposes into a component the system can reach exactly and a component it can never reach at all. Because the matrix has full column rank and the noise is exactly orthogonal to its range, that construction guarantees, without any iterative solve, that the unique least-squares solution equals the ground-truth vector and that the optimal residual equals the noise vector itself; every later step consumes the matrix, the right-hand side, and these two exact targets, plus the column- and row-wise squared norms used throughout the sampling and solving steps.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_kaczmarz_instance(seed: int = 20260920, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10) -> dict:
    """Build a pinned inconsistent least-squares instance for RGDBEK.

    Parameters
    ----------
    seed : int
        Seed for `numpy.random.default_rng`, consumed in the fixed draw
        order values, mask, x_true, w (must be a non-negative integer).
    m : int
        Number of rows of the coefficient matrix (must be a positive
        integer with m >= n).
    n : int
        Number of columns of the coefficient matrix (must be a positive
        integer with n <= m).
    density : float
        Target fraction of nonzero entries of the coefficient matrix
        (must satisfy 0 < density <= 1).
    noise_ratio : float
        Ratio of the noise vector's Euclidean norm to the Euclidean norm
        of the noiseless signal A @ x_true (must be > 0).

    Returns
    -------
    instance : dict
        A : (m, n) float array, the sparse random coefficient matrix.
        b : (m,) float array, the right-hand side A @ x_star + r_opt.
        x_star : (n,) float array, the exact least-squares solution
            (equal to the ground-truth vector by construction).
        r_opt : (m,) float array, the exact optimal residual
            b - A @ x_star (orthogonal to the range of A by construction).
        col_sq_norms : (n,) float array, squared Euclidean norm of every
            column of A.
        row_sq_norms : (m,) float array, squared Euclidean norm of every
            row of A.
        b_norm : float, the Euclidean norm of b.

    Raises
    ------
    ValueError
        If seed is not a non-negative integer, if m or n is not a
        positive integer or m < n, if density is not in (0, 1], if
        noise_ratio is not a positive real number, or if the drawn noise
        vector has exactly zero norm (a degenerate instance).
    """
    return instance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_kaczmarz_instance(seed: int = 20260920, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10) -> dict:
    """Reference implementation of build_kaczmarz_instance."""
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(m, (int, np.integer)) or int(m) < 1:
        raise ValueError("m must be a positive integer")
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    m = int(m)
    n = int(n)
    if m < n:
        raise ValueError("m must be >= n (A must have full column rank)")
    if not (isinstance(density, (int, float, np.floating, np.integer)) and 0.0 < float(density) <= 1.0):
        raise ValueError("density must satisfy 0 < density <= 1")
    if not (isinstance(noise_ratio, (int, float, np.floating, np.integer)) and float(noise_ratio) > 0.0):
        raise ValueError("noise_ratio must be a positive real number")

    seed = int(seed)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((m, n))
    mask = rng.random((m, n)) < float(density)
    A = values * mask
    x_true = rng.standard_normal(n)
    w = rng.standard_normal(m)

    xi = np.linalg.lstsq(A, w, rcond=None)[0]
    r = w - A @ xi
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        raise ValueError("degenerate instance: noise vector has zero norm")
    r = r * (float(noise_ratio) * np.linalg.norm(A @ x_true) / r_norm)
    b = A @ x_true + r

    col_sq_norms = np.sum(A ** 2, axis=0)
    row_sq_norms = np.sum(A ** 2, axis=1)
    b_norm = float(np.linalg.norm(b))

    return {
        "A": A,
        "b": b,
        "x_star": x_true,
        "r_opt": r,
        "col_sq_norms": col_sq_norms,
        "row_sq_norms": row_sq_norms,
        "b_norm": b_norm,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: normal scenario (benchmark 600x150), a consistency check
    (A @ x_star + r_opt == b, r_opt orthogonal to range(A)), a boundary
    case (small, well-conditioned 8x2 instance with m comfortably greater
    than n), and edge cases (invalid density, m < n).
    """
    return [
        {
            # benchmark config: b_norm plus the norms of the two exact
            # targets x_star and r_opt.
            "setup": "import numpy as np",
            "call": (
                "float(build_kaczmarz_instance()['b_norm']) "
                "+ float(np.linalg.norm(build_kaczmarz_instance()['x_star'])) "
                "+ float(np.linalg.norm(build_kaczmarz_instance()['r_opt']))"
            ),
            "gold_call": (
                "float(_oracle_build_kaczmarz_instance()['b_norm']) "
                "+ float(np.linalg.norm(_oracle_build_kaczmarz_instance()['x_star'])) "
                "+ float(np.linalg.norm(_oracle_build_kaczmarz_instance()['r_opt']))"
            ),
        },
        {
            # consistency check: b == A @ x_star + r_opt exactly, and
            # r_opt orthogonal to range(A) (both to machine precision).
            "setup": "import numpy as np",
            "call": (
                "float(np.max(np.abs("
                "build_kaczmarz_instance()['A'] @ build_kaczmarz_instance()['x_star'] "
                "+ build_kaczmarz_instance()['r_opt'] - build_kaczmarz_instance()['b']"
                ")))"
            ),
            "gold_call": (
                "float(np.max(np.abs("
                "_oracle_build_kaczmarz_instance()['A'] @ _oracle_build_kaczmarz_instance()['x_star'] "
                "+ _oracle_build_kaczmarz_instance()['r_opt'] - _oracle_build_kaczmarz_instance()['b']"
                ")))"
            ),
        },
        {
            # boundary: small, well-conditioned 8x2 instance (m
            # comfortably greater than n, unlike a square m=n boundary,
            # which forces the noise construction through pure LAPACK
            # round-off and can violate the function's own orthogonality
            # contract) -- sum of the squared-norm arrays plus the
            # shapes-derived scalar.
            "setup": "import numpy as np",
            "call": (
                "float(np.sum(build_kaczmarz_instance(seed=1, m=8, n=2, density=1.0, noise_ratio=0.1)['col_sq_norms'])) "
                "+ float(np.sum(build_kaczmarz_instance(seed=1, m=8, n=2, density=1.0, noise_ratio=0.1)['row_sq_norms']))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_build_kaczmarz_instance(seed=1, m=8, n=2, density=1.0, noise_ratio=0.1)['col_sq_norms'])) "
                "+ float(np.sum(_oracle_build_kaczmarz_instance(seed=1, m=8, n=2, density=1.0, noise_ratio=0.1)['row_sq_norms']))"
            ),
        },
        {
            # edge: density outside (0, 1] is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: build_kaczmarz_instance(m=20, n=10, density=1.5))",
            "gold_call": "_guard(lambda: _oracle_build_kaczmarz_instance(m=20, n=10, density=1.5))",
        },
        {
            # edge: m < n is invalid (A cannot have full column rank)
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: build_kaczmarz_instance(m=5, n=10, density=0.5))",
            "gold_call": "_guard(lambda: _oracle_build_kaczmarz_instance(m=5, n=10, density=0.5))",
        },
    ]
