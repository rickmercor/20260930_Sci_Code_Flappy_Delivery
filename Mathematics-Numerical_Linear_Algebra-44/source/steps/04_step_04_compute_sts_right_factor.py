"""
the right orthogonal factor V of the S^T S-SVD.

Return the ordinary orthogonal right factor of the sketched matrix under

this task's deterministic column-sign convention (diagonal-preferring,

positive determinant). Raw library SVD signs or a largest-magnitude-only

rule alone are not accepted.

Returns
-------
ndarray of shape (n, n), float64: the orthogonal right factor V under the instance column-sign convention.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_sts_right_factor(A, S):
    """A: (m, n) tall matrix; S: (s, m) sketch operator with s >= n. Returns
    (n, n) float64: the orthogonal right factor V from the S^T S-SVD of A,
    with diagonal-preferring column signs and det(V) = +1. Raises
    ValueError on shape mismatch, s < n, or a rank-deficient sketch S A."""
    return np.zeros((A.shape[1], A.shape[1]))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fix_sts_right_factor_signs(V):
    V = np.array(V, dtype=np.float64, copy=True)
    n = V.shape[1]
    eps = np.finfo(np.float64).eps
    for j in range(n):
        col = V[:, j]
        scale = np.linalg.norm(col)
        thr = 10.0 * eps * max(scale, 1.0)
        if abs(col[j]) > thr:
            if col[j] < 0.0:
                V[:, j] = -col
        else:
            k = int(np.argmax(np.abs(col)))
            if col[k] < 0.0:
                V[:, j] = -col
    if np.linalg.det(V) < 0.0:
        V[:, -1] *= -1.0
    return V


def _oracle_compute_sts_right_factor(A, S):
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

    theta = _oracle_compute_sts_singular_values(A, S)
    # Exact <= 0.0 misses numerically rank-deficient sketches (tiny positive
    # singular values from roundoff). Use a relative machine-eps threshold.
    if theta[0] <= 0.0 or theta[-1] <= max(m, n) * np.finfo(np.float64).eps * theta[0]:
        raise ValueError("S A must have full column rank")

    _, _, Vt = np.linalg.svd(S @ A, full_matrices=False)
    V = np.array(Vt.T[:, :n], dtype=np.float64)
    return _fix_sts_right_factor_signs(V)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as nP

def test_cases():
    # Hard cases: diagonal-prefer + det=+1 disagrees with largest-magnitude-
    # positive and with first-entry-positive on several columns.
    return [
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(32, 4)\nS = build_sketch_operator(32, 4, 37, 5)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(16, 2)\nS = build_sketch_operator(16, 8, 17, 3)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(64, 6)\nS = build_sketch_operator(64, 16, 67, 11)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(48, 3)\nS = build_sketch_operator(48, 12, 53, 7)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(96, 6)\nS = build_sketch_operator(96, 24, 97, 13)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(80, 5)\nS = build_sketch_operator(80, 20, 83, 17)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(72, 5)\nS = build_sketch_operator(72, 18, 73, 11)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(56, 4)\nS = build_sketch_operator(56, 14, 59, 5)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(40, 4)\nS = build_sketch_operator(40, 10, 41, 9)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        # det(V) must be +1 after the two-stage fix
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)', "call": 'float(np.linalg.det(V))', "gold_call": '1.0', "tol": 1e-09},
        # Largest-magnitude-positive alone disagrees on this instance
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\n_,_,Vt = np.linalg.svd(S @ A, full_matrices=False)\nV_max = np.array(Vt.T[:, :8], dtype=np.float64)\nfor j in range(8):\n    col = V_max[:, j]\n    k = int(np.argmax(np.abs(col)))\n    if col[k] < 0.0:\n        V_max[:, j] = -col', "call": 'float(np.linalg.norm(V - V_max) > 0.1)', "gold_call": '1.0', "tol": 0.0},
        # First-entry-positive disagrees on this instance
        {"setup": 'A = construct_kernel_matrix(96, 6)\nS = build_sketch_operator(96, 24, 97, 13)\nV = compute_sts_right_factor(A, S)\n_,_,Vt = np.linalg.svd(S @ A, full_matrices=False)\nV_first = np.array(Vt.T[:, :6], dtype=np.float64)\nfor j in range(6):\n    if V_first[0, j] < 0.0:\n        V_first[:, j] *= -1.0', "call": 'float(np.linalg.norm(V - V_first) > 0.1)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'A = construct_kernel_matrix(112, 7)\nS = build_sketch_operator(112, 28, 113, 15)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(88, 5)\nS = build_sketch_operator(88, 22, 89, 9)', "call": 'compute_sts_right_factor(A, S)', "gold_call": '_oracle_compute_sts_right_factor(A, S)', "tol": 1e-09},
        # Euclidean right factor of A is not the S^T S right factor
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\n_,_,VtA = np.linalg.svd(A, full_matrices=False)\nVA = VtA.T', "call": 'float(np.linalg.norm(V - VA) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        # Diagonal-prefer without the det=+1 stage disagrees on the gold instance
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\n_,_,Vt = np.linalg.svd(S @ A, full_matrices=False)\nV_diag = np.array(Vt.T[:, :8], dtype=np.float64)\neps = np.finfo(np.float64).eps\nfor j in range(8):\n    col = V_diag[:, j]\n    thr = 10.0 * eps * max(np.linalg.norm(col), 1.0)\n    if abs(col[j]) > thr:\n        if col[j] < 0.0:\n            V_diag[:, j] = -col\n    else:\n        k = int(np.argmax(np.abs(col)))\n        if col[k] < 0.0:\n            V_diag[:, j] = -col', "call": 'float(np.linalg.norm(V - V_diag) > 0.1)', "gold_call": '1.0', "tol": 0.0},
    ]
