"""
Compute the MINBERR-NE backward-error history from the paper's Golub--Kahan reduction.

MINBERR-NE minimizes backward error over the normal Krylov space. Golub--Kahan bidiagonalization converts each growing problem into an upper-bidiagonal smallest-singular-vector calculation, and the paper's normal-equation scale recovers the full-space iterate.

Returns
-------
np.ndarray, a length-iterations vector containing the full-space MINBERR-NE backward errors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_normal_equation_history(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the MINBERR-NE backward-error history.

    Start the Golub--Kahan process with ``u_1 = b / ||b||_2`` and
    ``alpha_1 v_1 = A.T @ u_1``. For every one-based step ``j``, append
    ``u_(j+1)`` and ``v_(j+1)`` using

    ``beta_(j+1) u_(j+1) = A @ v_j - alpha_j u_j`` and
    ``alpha_(j+1) v_(j+1) = A.T @ u_(j+1) - beta_(j+1) v_j``.

    Before taking each residual norm and normalizing, perform one full
    modified Gram--Schmidt pass against all previously stored vectors of
    the same (left or right) basis, in their creation order. Include the
    initial left update against ``u_1``. The post-reorthogonalization norms
    define the nonnegative ``alpha`` and ``beta`` coefficients.

    After forming ``beta_(j+2) u_(j+2)`` in the same way, build the
    ``(j+1)``-by-``(j+1)`` upper-bidiagonal reduced matrix whose diagonal is
    ``beta_2, ..., beta_(j+2)`` and whose superdiagonal is
    ``alpha_2, ..., alpha_(j+1)``. Call
    ``approximate_smallest_singular_pair`` on this matrix with seed
    ``seed + j`` and the supplied fixed iteration count. If ``y`` is the
    returned right singular-vector estimate and ``V`` contains
    ``v_1, ..., v_(j+1)``, use the paper's scale

    ``scale = ||b||_2**2 / ((b.T @ A) @ (V @ y))``

    and evaluate the full matrix-only relative backward error of
    ``x = V @ (scale * y)``. Do not replace the Golub--Kahan reduction by a
    normal-equations eigensolve, SVD, or dense least-squares solve.

    Parameters
    ----------
    A : np.ndarray
        Finite nonzero square matrix of shape ``(n, n)``.
    b : np.ndarray
        Finite nonzero vector of shape ``(n,)``.
    iterations : int
        Number of history entries, with ``1 <= iterations < n``.
    inverse_steps : int
        Positive fixed iteration count for every reduced problem.
    seed : int
        Base integer seed; reduced problem ``j`` uses ``seed + j``.

    Returns
    -------
    np.ndarray
        Array of shape ``(iterations,)`` containing the full-space
        MINBERR-NE backward errors.

    Raises
    ------
    ValueError
        If ``A`` is not a nonempty square matrix, if ``A`` or ``b`` is
        non-finite or their shapes are incompatible, if ``A`` or ``b`` is
        zero, if ``iterations`` does not satisfy ``1 <= iterations < n``, if
        ``inverse_steps`` is not a positive integer, if ``seed`` is not an
        integer, or if the Golub--Kahan recursion breaks down at any step
        (initial right/left vector or a later right/left update) or the
        MINBERR-NE direction cannot be scaled.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_normal_equation_history(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the deterministic MINBERR-NE history."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite with compatible shapes")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or not 1 <= iterations < n
    ):
        raise ValueError("iterations must satisfy 1 <= iterations < n")
    if (
        isinstance(inverse_steps, (bool, np.bool_))
        or not isinstance(inverse_steps, (int, np.integer))
        or inverse_steps < 1
    ):
        raise ValueError("inverse_steps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    norm_A = float(np.linalg.norm(A, 2))
    norm_b = float(np.linalg.norm(b))
    if norm_A == 0.0 or norm_b == 0.0:
        raise ValueError("A and b must be nonzero")
    breakdown_tolerance = 64.0 * np.finfo(float).eps * max(1.0, norm_A)

    u = b / norm_b
    v_raw = A.T @ u
    alpha = float(np.linalg.norm(v_raw))
    if alpha <= breakdown_tolerance:
        raise ValueError("Golub--Kahan broke down at the initial right vector")
    v = v_raw / alpha

    u_next_raw = A @ v - alpha * u
    u_next_raw -= u * (u @ u_next_raw)
    beta_next = float(np.linalg.norm(u_next_raw))
    if beta_next <= breakdown_tolerance:
        raise ValueError("Golub--Kahan broke down at the initial left update")
    u_next = u_next_raw / beta_next

    V = np.empty((n, iterations + 1), dtype=float)
    V[:, 0] = v
    alphas = [alpha]
    reduced_diagonal = [beta_next]
    history = []
    U_basis = [u.copy(), u_next.copy()]

    for j in range(1, iterations + 1):
        v_next_raw = A.T @ u_next - beta_next * v
        for q in V[:, :j].T:
            v_next_raw -= q * (q @ v_next_raw)
        alpha_next = float(np.linalg.norm(v_next_raw))
        if alpha_next <= breakdown_tolerance:
            raise ValueError("Golub--Kahan broke down in a right update")
        v_next = v_next_raw / alpha_next
        V[:, j] = v_next
        alphas.append(alpha_next)

        following_u_raw = A @ v_next - alpha_next * u_next
        for q in U_basis:
            following_u_raw -= q * (q @ following_u_raw)
        following_beta = float(np.linalg.norm(following_u_raw))
        if following_beta <= breakdown_tolerance:
            raise ValueError("Golub--Kahan broke down in a left update")
        following_u = following_u_raw / following_beta
        reduced_diagonal.append(following_beta)
        U_basis.append(following_u.copy())

        dimension = j + 1
        reduced = np.diag(np.asarray(reduced_diagonal[:dimension]))
        reduced += np.diag(np.asarray(alphas[1:dimension]), k=1)
        pair = _oracle_approximate_smallest_singular_pair(
            reduced,
            int(inverse_steps),
            int(seed) + j,
        )
        y = pair[1:]
        direction = V[:, :dimension] @ y
        denominator = float((b @ A) @ direction)
        denominator_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(b @ A)))
        )
        if abs(denominator) <= denominator_tolerance:
            raise ValueError("the MINBERR-NE direction cannot be scaled")
        x = direction * (norm_b**2 / denominator)
        history.append(_oracle_compute_relative_backward_error(A, b, x))

        u = u_next
        u_next = following_u
        v = v_next
        alpha = alpha_next
        beta_next = following_beta

    return np.asarray(history)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return numerical, invalid-input, and frozen layout-regression cases."""
    return [
        {
            "setup": """import numpy as np
A = np.diag([4.0, 2.0, 0.75, 0.2])
b = np.array([1.0, -0.5, 0.25, 2.0])
iterations = 1
inverse_steps = 6
seed = 3
""",
            "call": "compute_normal_equation_history(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_normal_equation_history(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(9, 1.0e5)
A, b = packed[:, :-1], packed[:, -1]
iterations = 4
inverse_steps = 14
seed = 31
""",
            "call": "compute_normal_equation_history(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_normal_equation_history(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(16, 1.0e12)
A, b = packed[:, :-1], packed[:, -1]
iterations = 7
inverse_steps = 30
seed = 11729
""",
            "call": "compute_normal_equation_history(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_normal_equation_history(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(12, 1.0e8)
A, b = packed[:, :-1], packed[:, -1]
iterations = 5
inverse_steps = 18
seed = 59
forbidden_names = ("solve", "lstsq", "inv", "pinv", "svd", "eig", "eigh", "cholesky")
saved_linalg = {name: getattr(np.linalg, name) for name in forbidden_names}
def forbidden(*args, **kwargs):
    raise RuntimeError("dense solve or spectral routine is forbidden")
def run_with_restrictions(function):
    try:
        for name in forbidden_names:
            setattr(np.linalg, name, forbidden)
        return function(A, b, iterations, inverse_steps, seed)
    finally:
        for name, implementation in saved_linalg.items():
            setattr(np.linalg, name, implementation)
""",
            "call": "run_with_restrictions(compute_normal_equation_history)",
            "gold_call": "run_with_restrictions(_oracle_compute_normal_equation_history)",
        },
        {
            "setup": """import numpy as np
A = np.ones((3, 2))
b = np.ones(3)
def run_model():
    try:
        compute_normal_equation_history(A, b, 1, 4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_normal_equation_history(A, b, 1, 4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(16, 1.0e12)
A, b = packed[:, :-1], packed[:, -1]
layouts = (A.copy(order='C'), A.copy(order='F'), np.repeat(A, 2, axis=1)[:, ::2])
""",
            "call": "np.concatenate([compute_normal_equation_history(M, b.copy(), 7, 30, 11729) for M in layouts])",
            "gold_call": "np.tile(np.array([0.14786535504131615, 0.023330690370914463, 0.0036807925993580966, 0.000580602304635574, 0.00009155824718331016, 0.000014431839624013653, 0.000002273121135119345]), 3)",
        },
    ]
