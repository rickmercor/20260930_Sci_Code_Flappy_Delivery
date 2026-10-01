"""
the nearest S^T S-orthogonal matrix P = W V^T.

A classical matrix nearness problem asks for the matrix with orthonormal

columns closest to a given tall matrix. Once orthonormality is measured in a

sketched metric, the problem transports to the range-constrained feasible set

from the primary source. Recover that source's construction of the nearest

S^T S-orthogonal factor; the right factor V is an input and is not re-signed

here.

Returns
-------
ndarray of shape (m, n), float64: the nearest S^T S-orthogonal matrix P to A.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def form_nearest_sts_orthogonal(A, S, V):
    """A: (m, n) tall matrix; S: (s, m) sketch operator with s >= n; V: (n, n)
    right factor from the S^T S-SVD under the instance sign convention.
    Returns (m, n) float64: the unique Frobenius-nearest matrix to A among
    factors with S^T S-orthonormal columns spanning Range(A). Raises
    ValueError on shape mismatch, s < n, a rank-deficient sketch, or a V
    that violates the instance sign convention."""
    return np.zeros_like(np.asarray(A, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_form_nearest_sts_orthogonal(A, S, V):
    A = np.asarray(A, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if A.ndim != 2 or S.ndim != 2 or V.ndim != 2:
        raise ValueError("A, S, and V must be 2D")
    m, n = A.shape
    s, m2 = S.shape
    if m != m2:
        raise ValueError("S must have m columns matching the rows of A")
    if m < n:
        raise ValueError("require m >= n")
    if s < n:
        raise ValueError("require sketch dimension s >= n")
    if V.shape != (n, n):
        raise ValueError("V must be (n, n)")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(S)) or not np.all(np.isfinite(V)):
        raise ValueError("A, S, and V must be finite")

    # Sign convention is owned by step 04. Refuse largest-magnitude-only or
    # library-default patterns here so a wrong V cannot silently pass.
    V_canon = _fix_sts_right_factor_signs(V)
    if not np.allclose(V, V_canon, rtol=0.0, atol=1e-10):
        raise ValueError(
            "V columns must already satisfy the right-factor step's two-stage "
            "sign convention (diagonal-preferring, then det(V)=+1)"
        )

    theta = _oracle_compute_sts_singular_values(A, S)
    if theta[0] <= 0.0 or theta[-1] <= max(m, n) * np.finfo(np.float64).eps * theta[0]:
        raise ValueError("S A must have full column rank")

    # Left factor in R^{m x n}; never the s-row factor U_1 of S A.
    W = (A @ V) / theta[None, :]
    if W.shape != (m, n):
        raise ValueError("left factor W must have shape (m, n)")
    return W @ V.T

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(32, 4)\nS = build_sketch_operator(32, 4, 37, 5)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(16, 2)\nS = build_sketch_operator(16, 8, 17, 3)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(64, 6)\nS = build_sketch_operator(64, 32, 67, 11)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(48, 5)\nS = build_sketch_operator(48, 24, 53, 7)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(96, 6)\nS = build_sketch_operator(96, 24, 97, 13)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(80, 5)\nS = build_sketch_operator(80, 20, 83, 17)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(56, 4)\nS = build_sketch_operator(56, 14, 59, 5)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        # Wrong route: Euclidean polar of A must disagree with P (inline SVD;
        # step 06 is not yet in the harness namespace at this step)
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nU, _, Yt = np.linalg.svd(A, full_matrices=False)\nT = U @ Yt', "call": 'float(np.linalg.norm(P - T))', "gold_call": '0.580531495744', "tol": 1e-09},
        # Largest-magnitude-only (no diagonal/det stages) must be rejected
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\n_,_,Vt = np.linalg.svd(S @ A, full_matrices=False)\nV_max = np.array(Vt.T[:, :8], dtype=np.float64)\nfor j in range(8):\n    col = V_max[:, j]\n    k = int(np.argmax(np.abs(col)))\n    if col[k] < 0.0:\n        V_max[:, j] = -col\ndef _check_maxmag_only():\n    try:\n        form_nearest_sts_orthogonal(A, S, V_max)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _check_maxmag_only_gold():\n    try:\n        _oracle_form_nearest_sts_orthogonal(A, S, V_max)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2', "call": '_check_maxmag_only()', "gold_call": '_check_maxmag_only_gold()', "tol": 0.0},
        {"setup": 'A = construct_kernel_matrix(96, 6)\nS = build_sketch_operator(96, 24, 97, 13)\nV = compute_sts_right_factor(A, S)\nV_bad = V.copy()\nV_bad[:, 0] *= -1.0\ndef _check_bad_signs():\n    try:\n        form_nearest_sts_orthogonal(A, S, V_bad)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _check_bad_signs_gold():\n    try:\n        _oracle_form_nearest_sts_orthogonal(A, S, V_bad)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2', "call": '_check_bad_signs()', "gold_call": '_check_bad_signs_gold()', "tol": 0.0},
        {"setup": 'A = construct_kernel_matrix(80, 5)\nS = build_sketch_operator(80, 20, 83, 17)\nV = compute_sts_right_factor(A, S)\nV_bad = V.copy()\nV_bad[:, -1] *= -1.0\ndef _check_bad_signs2():\n    try:\n        form_nearest_sts_orthogonal(A, S, V_bad)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _check_bad_signs2_gold():\n    try:\n        _oracle_form_nearest_sts_orthogonal(A, S, V_bad)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2', "call": '_check_bad_signs2()', "gold_call": '_check_bad_signs2_gold()', "tol": 0.0},
        # STS orthonormality of P (paper constraint)
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nSP = S @ P', "call": 'float(np.linalg.norm(SP.T @ SP - np.eye(8)))', "gold_call": '0.0', "tol": 1e-09},
        # Gram closed form must match W V^T route
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nG = A.T @ (S.T @ S) @ A\nevals, evecs = np.linalg.eigh(G)\nP2 = A @ (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T)', "call": 'float(np.linalg.norm(P - P2))', "gold_call": '0.0', "tol": 1e-08},
        # Wrong route: lift sketched polar by S.T (stays wrong vs ambient P)
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nU1, _, Vt = np.linalg.svd(S @ A, full_matrices=False)\nP_lift = S.T @ (U1 @ Vt)', "call": 'float(np.linalg.norm(P - P_lift) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        # Wrong route: unscaled A V V^T is not the STS polar
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nP_raw = A @ V @ V.T', "call": 'float(np.linalg.norm(P - P_raw) > 0.05)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'A = construct_kernel_matrix(112, 7)\nS = build_sketch_operator(112, 28, 113, 15)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        {"setup": 'A = construct_kernel_matrix(88, 5)\nS = build_sketch_operator(88, 22, 89, 9)\nV = compute_sts_right_factor(A, S)', "call": 'form_nearest_sts_orthogonal(A, S, V)', "gold_call": '_oracle_form_nearest_sts_orthogonal(A, S, V)', "tol": 1e-09},
        # Wrong route: use U1 (s-row left factor of SA) as if it were W
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nU1, th, Vt = np.linalg.svd(S @ A, full_matrices=False)\nP_u1 = U1 @ Vt', "call": 'float(P_u1.shape[0] != P.shape[0] or np.linalg.norm(P - P_u1) > 1.0)', "gold_call": '1.0', "tol": 0.0},
        # Wrong route: Euclidean polar of SA (wrong metric and shape)
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nU1, _, Vt = np.linalg.svd(S @ A, full_matrices=False)\nP_sa = U1 @ Vt', "call": 'float(P_sa.shape != P.shape or np.linalg.norm(P[:64] - P_sa) > 0.5)', "gold_call": '1.0', "tol": 0.0},
        # Null-space freedom: P + Z with S Z = 0 is not the range-constrained minimizer
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nZ = np.random.default_rng(0).normal(size=P.shape)\nZ = Z - S.T @ np.linalg.lstsq(S @ S.T, S @ Z, rcond=None)[0]\nP_free = P + 0.1 * Z', "call": 'float(np.linalg.norm(P - P_free) > 0.05)', "gold_call": '1.0', "tol": 0.0},
    ]
