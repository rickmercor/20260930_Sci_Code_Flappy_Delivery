"""
the nearest Euclidean orthogonal matrix T (thin polar factor)

The orthogonal factor of the polar decomposition is the unique Frobenius

minimizer among matrices with orthonormal columns for a full-column-rank tall

matrix, and a minimizer in the spectral norm. This makes it the reference

answer against which any approximate or metric-modified notion of

orthogonalization is judged. It also factors the original matrix into an

orthonormal part and a symmetric positive definite part, separating the

rotation from the stretch. Computing it exactly is more costly than the

approximate alternatives used at scale, which is precisely why cheaper

surrogates are of interest.

Returns
-------
ndarray of shape (m, n), float64: the nearest Euclidean orthogonal matrix T to A, i.e. the orthogonal polar factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def form_nearest_euclidean_orthogonal(A):
    """A: (m, n) tall matrix of full column rank with m >= n. Returns (m, n)
    float64: the unique Frobenius-nearest matrix with orthonormal columns
    (a spectral-norm minimizer as well). Raises ValueError for m < n,
    non-finite entries, or a rank-deficient A."""
    return np.zeros_like(np.asarray(A, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np





def _oracle_form_nearest_euclidean_orthogonal(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if m < n:
        raise ValueError("require m >= n")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    U, sigma, Yt = np.linalg.svd(A, full_matrices=False)
    if sigma[0] <= 0.0 or sigma[-1] <= max(m, n) * np.finfo(np.float64).eps * sigma[0]:
        raise ValueError("A must have full column rank")
    return U @ Yt

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'A = construct_kernel_matrix(128, 8)', "call": 'form_nearest_euclidean_orthogonal(A)', "gold_call": '_oracle_form_nearest_euclidean_orthogonal(A)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(6, 6, 0.5)', "call": 'form_nearest_euclidean_orthogonal(A)', "gold_call": '_oracle_form_nearest_euclidean_orthogonal(A)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(5, 1)', "call": 'form_nearest_euclidean_orthogonal(A)', "gold_call": '_oracle_form_nearest_euclidean_orthogonal(A)', "tol": 1e-10},
    ]
