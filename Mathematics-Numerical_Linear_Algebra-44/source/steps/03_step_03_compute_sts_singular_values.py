"""
the S^T S-singular values of A.

The ordinary singular value decomposition is tied to the Euclidean inner

product. Van Loan showed that the decomposition generalizes when the ambient

metric is replaced by one induced by a symmetric positive (semi)definite

matrix, producing a family of generalized singular values attached to that

metric. When the metric comes from a sketching operator, these generalized

values are the stationary values of the ratio between the sketched norm of the

image and the Euclidean norm of the argument, and they inherit the rank

information of the original matrix: they vanish exactly where the ordinary

singular values do. They are therefore the natural spectral summary of a matrix

as seen through the sketch, and they are what a sketched algorithm actually

operates on.

Returns
-------
ndarray of shape (n,), float64: the S^T S-singular values of A in nonincreasing order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_sts_singular_values(A, S):
    """A: (m, n) tall matrix with m >= n; S: (s, m) sketch operator with
    s >= n. Returns (n,) float64: the S^T S-singular values of A in
    nonincreasing order. Raises ValueError on shape mismatch, s < n, or
    non-finite input."""
    return np.zeros(A.shape[1])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_sts_singular_values(A, S):
    A = np.asarray(A, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if A.ndim != 2 or S.ndim != 2:
        raise ValueError("A and S must be 2D")
    m, n = A.shape
    s, m2 = S.shape
    if m != m2:
        raise ValueError("S must have m columns matching the rows of A")
    if m < n:
        raise ValueError("require m >= n")
    if s < n:
        raise ValueError("require sketch dimension s >= n")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(S)):
        raise ValueError("A and S must be finite")
    theta = np.linalg.svd(S @ A, compute_uv=False)
    out = np.zeros(n, dtype=np.float64)
    out[: min(theta.size, n)] = theta[: min(theta.size, n)]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(32, 4)\nS = build_sketch_operator(32, 4, 37, 5)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(16, 2)\nS = build_sketch_operator(16, 8, 17, 3)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(96, 6)\nS = build_sketch_operator(96, 24, 97, 13)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(80, 5)\nS = build_sketch_operator(80, 20, 83, 17)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
        # Must be SVD(S A), not SVD(A)
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\ntheta = compute_sts_singular_values(A, S)\nsigma = np.linalg.svd(A, compute_uv=False)', "call": 'float(abs(theta[0] - sigma[0]) > 1e-3)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\ntheta = compute_sts_singular_values(A, S)', "call": 'float(theta[0])', "gold_call": '0.891292371607', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(64, 6)\nS = build_sketch_operator(64, 16, 67, 11)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(112, 7)\nS = build_sketch_operator(112, 28, 113, 15)', "call": 'compute_sts_singular_values(A, S)', "gold_call": '_oracle_compute_sts_singular_values(A, S)', "tol": 1e-10},
    ]
