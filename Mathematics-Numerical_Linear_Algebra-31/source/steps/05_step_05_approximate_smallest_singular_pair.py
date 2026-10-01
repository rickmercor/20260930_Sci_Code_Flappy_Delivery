"""
Approximate the smallest right singular pair of the paper's banded reduced operator without using dense factorizations.

Deleting the first row of the extended symmetric recurrence produces an upper-triangular matrix with two superdiagonals. Algorithm 4.1 applies inverse iteration through solves with this factor and its transpose; explicit banded substitutions keep each inverse step linear in the Krylov dimension. A fixed seed, iteration count, and orientation rule make the numerical result reproducible.

Returns
-------
np.ndarray, a vector [sigma, v...] with shape (k+1,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def approximate_smallest_singular_pair(
    reduced: np.ndarray,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return ``[sigma, v...]`` after structured inverse iterations.

    ``reduced`` is the paper's square matrix obtained by deleting the first
    row of the extended symmetric recurrence. It is upper triangular with at
    most two nonzero superdiagonals. Draw ``v`` from
    ``np.random.default_rng(seed).standard_normal(k)`` and normalize it.

    Repeat exactly ``inverse_steps`` times. First solve
    ``reduced.T @ z = v`` by forward substitution using only the diagonal and
    two subdiagonals of ``reduced.T``. Then solve ``reduced @ v = z`` by
    backward substitution using only the diagonal and two superdiagonals of
    ``reduced``. Normalize after every pair of solves. Do not call a dense
    linear solver, matrix inverse, eigensolver, singular-value decomposition,
    or Cholesky factorization.

    After the iterations, orient ``v`` so its first component whose magnitude
    exceeds ``1e-14`` is positive, and set
    ``sigma = ||reduced @ v||_2``.

    Parameters
    ----------
    reduced : np.ndarray
        Finite nonsingular upper-triangular matrix of shape ``(k, k)`` with
        upper bandwidth two.
    inverse_steps : int
        Positive number of fixed iterations. Boolean values are invalid.
    seed : int
        Seed passed to ``np.random.default_rng``. Boolean values are invalid.

    Returns
    -------
    np.ndarray
        Vector of shape ``(k+1,)`` containing ``sigma`` followed by the
        deterministically oriented unit vector ``v``.

    Raises
    ------
    ValueError
        If ``reduced`` is not a nonempty, finite, square matrix, if it is
        not upper triangular with upper bandwidth two, or if it is singular;
        if ``inverse_steps`` is not a positive integer (booleans included);
        if ``seed`` is not an integer (booleans included); or if an inverse
        iteration produces an invalid (non-finite or non-normalizable)
        vector.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_approximate_smallest_singular_pair(
    reduced: np.ndarray,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the fixed-iteration smallest-singular-pair approximation."""
    np = __import__("numpy")

    reduced = np.asarray(reduced, dtype=float)
    if reduced.ndim != 2 or reduced.shape[0] != reduced.shape[1] or reduced.shape[0] == 0:
        raise ValueError("reduced must be a nonempty square matrix")
    if not np.all(np.isfinite(reduced)):
        raise ValueError("reduced must be finite")
    if (
        isinstance(inverse_steps, (bool, np.bool_))
        or not isinstance(inverse_steps, (int, np.integer))
        or inverse_steps < 1
    ):
        raise ValueError("inverse_steps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    scale = max(1.0, float(np.linalg.norm(reduced, ord=np.inf)))
    structure_tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(np.abs(np.tril(reduced, k=-1)) > structure_tolerance):
        raise ValueError("reduced must be upper triangular")
    if np.any(np.abs(np.triu(reduced, k=3)) > structure_tolerance):
        raise ValueError("reduced must have upper bandwidth two")
    diagonal = np.diag(reduced)
    if np.any(np.abs(diagonal) <= structure_tolerance):
        raise ValueError("reduced must be nonsingular")

    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal(reduced.shape[0])
    v = v / np.linalg.norm(v)
    for _ in range(int(inverse_steps)):
        z = np.empty_like(v)
        for i in range(reduced.shape[0]):
            rhs = v[i]
            if i >= 1:
                rhs -= reduced[i - 1, i] * z[i - 1]
            if i >= 2:
                rhs -= reduced[i - 2, i] * z[i - 2]
            z[i] = rhs / diagonal[i]

        next_v = np.empty_like(v)
        for i in range(reduced.shape[0] - 1, -1, -1):
            rhs = z[i]
            if i + 1 < reduced.shape[0]:
                rhs -= reduced[i, i + 1] * next_v[i + 1]
            if i + 2 < reduced.shape[0]:
                rhs -= reduced[i, i + 2] * next_v[i + 2]
            next_v[i] = rhs / diagonal[i]
        v = next_v
        norm_v = float(np.linalg.norm(v))
        if not np.isfinite(norm_v) or norm_v == 0.0:
            raise ValueError("inverse iteration produced an invalid vector")
        v = v / norm_v

    nonzero = np.flatnonzero(np.abs(v) > 1e-14)
    if nonzero.size and v[nonzero[0]] < 0.0:
        v = -v
    sigma = float(np.linalg.norm(reduced @ v))
    return np.concatenate(([sigma], v))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar, bidiagonal, recurrence-derived, restricted, and invalid cases."""
    return [
        {
            "setup": "import numpy as np\nreduced = np.array([[0.375]])\ninverse_steps = 1\nseed = 2",
            "call": "approximate_smallest_singular_pair(reduced, inverse_steps, seed)",
            "gold_call": "_oracle_approximate_smallest_singular_pair(reduced, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
reduced = np.array([[1.5, 0.2, 0.0], [0.0, 0.45, 0.1], [0.0, 0.0, 0.08]])
inverse_steps = 8
seed = 17
""",
            "call": "approximate_smallest_singular_pair(reduced, inverse_steps, seed)",
            "gold_call": "_oracle_approximate_smallest_singular_pair(reduced, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(11, 1.0e9)
A, b = packed[:, :-1], packed[:, -1]
_, T = _oracle_build_orthogonal_recurrence(A, b, 6)
reduced, _, _ = _oracle_extract_reduced_operator(T, 0.0)
inverse_steps = 20
seed = 1729
""",
            "call": "approximate_smallest_singular_pair(reduced, inverse_steps, seed)",
            "gold_call": "_oracle_approximate_smallest_singular_pair(reduced, inverse_steps, seed)",
        },
        {
            "setup": """import numpy as np
packed = _oracle_construct_controlled_psd_system(14, 1.0e11)
A, b = packed[:, :-1], packed[:, -1]
_, T = _oracle_build_orthogonal_recurrence(A, b, 8)
reduced, _, _ = _oracle_extract_reduced_operator(T, 0.0)
inverse_steps = 24
seed = 311
forbidden_names = ("solve", "inv", "pinv", "svd", "eig", "eigh", "eigvals", "eigvalsh", "cholesky")
saved_linalg = {name: getattr(np.linalg, name) for name in forbidden_names}
def forbidden(*args, **kwargs):
    raise RuntimeError("dense spectral or factorization routine is forbidden")
def run_with_restrictions(function):
    try:
        for name in forbidden_names:
            setattr(np.linalg, name, forbidden)
        return function(reduced, inverse_steps, seed)
    finally:
        for name, implementation in saved_linalg.items():
            setattr(np.linalg, name, implementation)
""",
            "call": "run_with_restrictions(approximate_smallest_singular_pair)",
            "gold_call": "run_with_restrictions(_oracle_approximate_smallest_singular_pair)",
        },
        {
            "setup": """import numpy as np
reduced = np.array([[1.0, 2.0], [2.0, 4.0]])
def run_model():
    try:
        approximate_smallest_singular_pair(reduced, 5, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_approximate_smallest_singular_pair(reduced, 5, 0)
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
