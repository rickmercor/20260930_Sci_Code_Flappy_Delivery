"""
Compute the fixed-update, direct-MINBERR, and MINBERR-NE backward-error histories.

The baseline uses fixed Richardson updates. Direct MINBERR uses the first-row-deleted symmetric recurrence and stops at the first shifted-Cholesky rejection, while MINBERR-NE uses a Golub--Kahan reduction of the normal Krylov space for the same number of retained entries.

Returns
-------
np.ndarray, an array with shape (3, m) containing the fixed-update, direct-MINBERR, and MINBERR-NE histories
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_error_histories(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return three finite-iteration backward-error histories.

    Row 0 uses ``x_0 = 0`` and the fixed updates
    ``x_j = x_(j-1) + (b - A x_(j-1)) / ||A||_2``.

    Row 1 uses the symmetric recurrence through each dimension ``j``. At
    every dimension, call ``extract_reduced_operator(Tj, 1e-5)`` and use its
    reduced operator in ``approximate_smallest_singular_pair`` with seed
    ``seed + j``. If ``v`` is the returned direction, set
    ``scale = ||b||_2 / (Tj[0, :] @ v)`` and evaluate the full-space error of
    ``Qj[:, :j] @ (scale * v)``. Include the first prefix whose shifted
    Cholesky certificate rejects a pivot, then stop both rows 0 and 1 there.

    Row 2 must be the first ``m`` entries returned by
    ``compute_normal_equation_history(A, b, m, inverse_steps, seed + 10000)``,
    where ``m`` is the number of retained direct-MINBERR prefixes. This keeps
    the normal-equation Golub--Kahan comparison independent of the symmetric
    inverse-iteration starts while using the same number of entries.

    Parameters
    ----------
    A : np.ndarray
        Finite symmetric matrix of shape ``(n, n)`` with nonzero 2-norm.
    b : np.ndarray
        Finite nonzero vector of shape ``(n,)``.
    iterations : int
        Maximum number of history entries, with ``1 <= iterations < n``.
    inverse_steps : int
        Positive fixed iteration count for every reduced problem.
    seed : int
        Base integer seed for the reduced iterations.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, m)``, where ``m <= iterations`` is the first
        rejected symmetric prefix or ``iterations`` if no prefix rejects.
        The rows contain fixed-update, direct-MINBERR, and MINBERR-NE errors.

    Raises
    ------
    ValueError
        If ``A`` is not a nonempty square matrix, if ``A`` or ``b`` is
        non-finite or their shapes are incompatible, if ``A`` is not
        symmetric, if ``A`` or ``b`` is zero, if ``iterations`` does not
        satisfy ``1 <= iterations < n``, if ``inverse_steps`` is not a
        positive integer, if ``seed`` is not an integer, or if the shifted
        Cholesky certificate/factor state is invalid or the direct-MINBERR
        direction cannot be scaled.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_error_histories(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the three deterministic backward-error histories."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite with compatible shapes")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
        raise ValueError("A must be symmetric")
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

    Q, T = _oracle_build_orthogonal_recurrence(A, b, int(iterations))
    fixed_errors = []
    direct_errors = []
    x_fixed = np.zeros(n, dtype=float)

    for j in range(1, iterations + 1):
        x_fixed = x_fixed + (b - A @ x_fixed) / norm_A
        fixed_errors.append(_oracle_compute_relative_backward_error(A, b, x_fixed))

        Qj = Q[:, : j + 1]
        Tj = T[: j + 1, :j]
        reduced, certificate, factor = _oracle_extract_reduced_operator(Tj, 1.0e-5)
        if certificate.shape != (j + 2,) or not np.all(np.isfinite(certificate)):
            raise ValueError("invalid shifted-Cholesky certificate")
        if factor.shape != (j, j) or not np.all(np.isfinite(factor)):
            raise ValueError("invalid shifted-Cholesky factor state")
        pair = _oracle_approximate_smallest_singular_pair(
            reduced,
            int(inverse_steps),
            int(seed) + j,
        )
        v = pair[1:]
        denominator = float(Tj[0, :] @ v)
        denominator_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(Tj[0, :])))
        )
        if abs(denominator) <= denominator_tolerance:
            raise ValueError("the direct MINBERR direction cannot be scaled")
        x_direct = Qj[:, :j] @ ((norm_b / denominator) * v)
        direct_errors.append(_oracle_compute_relative_backward_error(A, b, x_direct))
        if int(certificate[0]) < j:
            break

    retained = len(direct_errors)
    normal_errors = _oracle_compute_normal_equation_history(
        A,
        b,
        retained,
        int(inverse_steps),
        int(seed) + 10000,
    )
    return np.vstack(
        (
            np.asarray(fixed_errors),
            np.asarray(direct_errors),
            np.asarray(normal_errors),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return short, ill-conditioned, long, dependency, and invalid histories."""
    return [
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(6, 50.0)
A, b = packed[:, :-1], packed[:, -1]
iterations = 2
inverse_steps = 6
seed = 3
""",
            "call": "compute_error_histories(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_error_histories(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(10, 1.0e8)
A, b = packed[:, :-1], packed[:, -1]
iterations = 5
inverse_steps = 18
seed = 41
""",
            "call": "compute_error_histories(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_error_histories(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(15, 1.0e12)
A, b = packed[:, :-1], packed[:, -1]
iterations = 8
inverse_steps = 30
seed = 1729
""",
            "call": "compute_error_histories(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_error_histories(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(9, 1.0e6)
A, b = packed[:, :-1], packed[:, -1]
iterations = 4
inverse_steps = 14
seed = 23
""",
            "call": "compute_error_histories(A, b, iterations, inverse_steps, seed)",
            "gold_call": "_oracle_compute_error_histories(A, b, iterations, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
A = np.eye(4)
b = np.ones(4)
def run_model():
    try:
        compute_error_histories(A, b, 4, 5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_error_histories(A, b, 4, 5, 0)
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
