"""
the realized subspace-embedding distortion of S on Range(A).

Nominal embedding tolerances are design parameters. For a fixed matrix and

sketch, the distortion actually incurred on Range(A) is the smallest eps

satisfying the Problem statement's squared-norm inequality on that subspace.

Recover any equivalent computational form from the primary / browsing sources;

coordinatewise or ambient substitutes are not accepted.

Returns
-------
float: the realized spectral embedding distortion eps*, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def realized_embedding_distortion(A, S):
    """A: (m, n) tall full-column-rank matrix; S: (s, m) sketch operator with
    s >= n. Returns float: the realized spectral embedding distortion of S
    on Range(A) (the tightest eps for the Problem statement's squared-norm
    inequality on that subspace). Raises ValueError on shape mismatch,
    s < n, or a rank-deficient A."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_realized_embedding_distortion(A, S):
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

    sigma = np.linalg.svd(A, compute_uv=False)
    if sigma[0] <= 0.0 or sigma[-1] <= max(m, n) * np.finfo(np.float64).eps * sigma[0]:
        raise ValueError("A must have full column rank")

    # The Euclidean polar factor has orthonormal columns spanning Range(A),
    # so it is a valid (basis-independent) choice of B.
    B = _oracle_form_nearest_euclidean_orthogonal(A)
    SB = S @ B
    G = SB.T @ SB - np.eye(n)
    return float(np.linalg.norm(G, 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(32, 4)\nS = build_sketch_operator(32, 32, 37, 5)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(16, 2)\nS = build_sketch_operator(16, 8, 17, 3)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(64, 6)\nS = build_sketch_operator(64, 32, 67, 11)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(48, 5)\nS = build_sketch_operator(48, 24, 53, 7)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(96, 6)\nS = build_sketch_operator(96, 24, 97, 13)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(80, 5)\nS = build_sketch_operator(80, 20, 83, 17)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        # Gold instance literal (independent of oracle-vs-oracle)
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '0.415147471896', "tol": 1e-10},
        # Coordinatewise max |||Sv||^2-1| on a QR basis is NOT eps* here
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)\nQ, _ = np.linalg.qr(A)\neps = realized_embedding_distortion(A, S)\ncoord = max(abs(np.linalg.norm(S @ Q[:, k]) ** 2 - 1.0) for k in range(8))', "call": 'float(abs(eps - coord) > 0.1)', "gold_call": '1.0', "tol": 0.0},
        # Ambient ||S^T S - I|| is NOT eps* (s < m)
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)\neps = realized_embedding_distortion(A, S)\nambient = float(np.linalg.norm(S.T @ S - np.eye(144), 2))', "call": 'float(abs(eps - ambient) > 0.2)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'A = construct_kernel_matrix(112, 7)\nS = build_sketch_operator(112, 28, 113, 15)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        {"setup": 'A = construct_kernel_matrix(88, 5)\nS = build_sketch_operator(88, 22, 89, 9)', "call": 'realized_embedding_distortion(A, S)', "gold_call": '_oracle_realized_embedding_distortion(A, S)', "tol": 1e-10},
        # Prop-1 certificate eps/(1-eps) is not eps* itself
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)\neps = realized_embedding_distortion(A, S)\ncert = eps / (1.0 - eps)', "call": 'float(abs(eps - cert) > 0.2)', "gold_call": '1.0', "tol": 0.0},
        # Unsquared max |||Sv||-1| on a QR basis is not eps*
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)\nQ, _ = np.linalg.qr(A)\neps = realized_embedding_distortion(A, S)\nunsquared = max(abs(np.linalg.norm(S @ Q[:, k]) - 1.0) for k in range(8))', "call": 'float(abs(eps - unsquared) > 0.2)', "gold_call": '1.0', "tol": 0.0},
        # Frobenius Gram defect is not the spectral eps*
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)\nQ, _ = np.linalg.qr(A)\neps = realized_embedding_distortion(A, S)\nfrob = float(np.linalg.norm(Q.T @ (S.T @ S) @ Q - np.eye(8), \"fro\"))', "call": 'float(abs(eps - frob) > 0.05)', "gold_call": '1.0', "tol": 0.0},
        # max_i |sigma_i(SA)/sigma_i(A) - 1| is not eps*
        {"setup": 'A = construct_kernel_matrix(144, 8)\nS = build_sketch_operator(144, 72, 149, 23)\neps = realized_embedding_distortion(A, S)\nsig = np.linalg.svd(A, compute_uv=False)\nth = np.linalg.svd(S @ A, compute_uv=False)\nratio = float(np.max(np.abs(th / sig - 1.0)))', "call": 'float(abs(eps - ratio) > 0.05)', "gold_call": '1.0', "tol": 0.0},
    ]
