"""
the sketched-polar scaled gap ratio Xi.

Assemble the Problem statement's dimensionless scalar from P, T, Q_P,

and eps*: the spectral cross-gap over the own-polar gap, scaled by the

reciprocal of the Proposition-1 certificate at eps*. Chain the

right-factor step into the nearest-STS step.

Returns
-------
float: the requested dimensionless scalar for the instance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def orchestrate_sketched_polar_audit(m, n, s, p, q, h=None):
    """m, n, s, p, q, h: instance parameters for the kernel matrix and
    deterministic sketch. Returns float: the dimensionless scalar from the
    Problem statement for this instance. Must obtain V from the S^T S
    right-factor step and pass it into the nearest-STS step. Raises
    ValueError on invalid parameters, a rank-deficient sketch, a distortion
    outside (0, 1), or a V that violates the instance sign convention."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h=None):
    A = _oracle_construct_kernel_matrix(m, n, h)
    S = _oracle_build_sketch_operator(m, s, p, q)

    theta = _oracle_compute_sts_singular_values(A, S)
    if theta[0] <= 0.0 or theta[-1] <= max(A.shape) * np.finfo(np.float64).eps * theta[0]:
        raise ValueError("the sketch destroys the column rank of A")

    # V is computed once and consumed by the STS polar step. Wrong signs fail
    # there; a pipeline that never calls the right-factor step is incomplete.
    V = _oracle_compute_sts_right_factor(A, S)
    P = _oracle_form_nearest_sts_orthogonal(A, S, V)
    T = _oracle_form_nearest_euclidean_orthogonal(A)
    eps = _oracle_realized_embedding_distortion(A, S)
    if not np.isfinite(eps) or not (0.0 < eps < 1.0):
        raise ValueError("the realized distortion must lie strictly in (0, 1)")

    gap = float(np.linalg.norm(P - T, 2))
    Qp = _oracle_form_nearest_euclidean_orthogonal(P)
    own = float(np.linalg.norm(P - Qp, 2))
    if own <= 0.0 or not np.isfinite(own):
        raise ValueError("the Higham polar gap of P must be positive")
    return (gap / own) * (1.0 - eps) / eps

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'm, n, s, p, q = 128, 8, 64, 131, 79\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '_oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "tol": 1e-08},
        {"setup": 'm, n, s, p, q = 64, 6, 32, 67, 11\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '_oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "tol": 1e-08},
        {"setup": 'm, n, s, p, q = 24, 3, 12, 29, 5\nh = 0.4', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '_oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "tol": 1e-08},
        {"setup": 'm, n, s, p, q = 96, 6, 24, 97, 13\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '_oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "tol": 1e-08},
        {"setup": 'm, n, s, p, q = 80, 5, 20, 83, 17\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '_oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "tol": 1e-08},
        {"setup": 'm, n, s, p, q = 56, 4, 14, 59, 5\nh = 0.35', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '_oracle_orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "tol": 1e-08},
        # Independent gold literals (not oracle-vs-oracle)
        {"setup": 'm, n, s, p, q = 128, 8, 64, 131, 79\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '1.175583703626', "tol": 1e-09},
        {"setup": 'm, n, s, p, q = 112, 7, 28, 113, 15\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '0.285493790701', "tol": 1e-09},
        {"setup": 'm, n, s, p, q = 88, 5, 22, 89, 9\nh = None', "call": 'orchestrate_sketched_polar_audit(m, n, s, p, q, h)', "gold_call": '0.507980015554', "tol": 1e-09},
        # Old Gamma = gap*(1-eps)/eps is NOT Xi
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nT = form_nearest_euclidean_orthogonal(A)\neps = realized_embedding_distortion(A, S)\ngap = float(np.linalg.norm(P - T, 2))\nxi = orchestrate_sketched_polar_audit(128, 8, 64, 131, 79, None)', "call": 'float(abs(xi - gap * (1.0 - eps) / eps) > 0.2)', "gold_call": '1.0', "tol": 0.0},
        # Inverted certificate on the ratio
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nT = form_nearest_euclidean_orthogonal(A)\nQp = form_nearest_euclidean_orthogonal(P)\neps = realized_embedding_distortion(A, S)\ngap = float(np.linalg.norm(P - T, 2))\nown = float(np.linalg.norm(P - Qp, 2))\nxi = orchestrate_sketched_polar_audit(128, 8, 64, 131, 79, None)', "call": 'float(abs(xi - (gap / own) * eps / (1.0 - eps)) > 0.05)', "gold_call": '1.0', "tol": 0.0},
        # Certificate-scaled Higham gap alone is not Xi
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nQp = form_nearest_euclidean_orthogonal(P)\neps = realized_embedding_distortion(A, S)\nown = float(np.linalg.norm(P - Qp, 2))\nxi = orchestrate_sketched_polar_audit(128, 8, 64, 131, 79, None)', "call": 'float(abs(xi - own * (1.0 - eps) / eps) > 0.2)', "gold_call": '1.0', "tol": 0.0},
        # gap/eps is not Xi
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nT = form_nearest_euclidean_orthogonal(A)\neps = realized_embedding_distortion(A, S)\ngap = float(np.linalg.norm(P - T, 2))\nxi = orchestrate_sketched_polar_audit(128, 8, 64, 131, 79, None)', "call": 'float(abs(xi - gap / eps) > 0.2)', "gold_call": '1.0', "tol": 0.0},
        # Bare ratio must use a tight threshold (Xi - gap/own ~ 0.029)
        {"setup": 'A = construct_kernel_matrix(128, 8)\nS = build_sketch_operator(128, 64, 131, 79)\nV = compute_sts_right_factor(A, S)\nP = form_nearest_sts_orthogonal(A, S, V)\nT = form_nearest_euclidean_orthogonal(A)\nQp = form_nearest_euclidean_orthogonal(P)\ngap = float(np.linalg.norm(P - T, 2))\nown = float(np.linalg.norm(P - Qp, 2))\nxi = orchestrate_sketched_polar_audit(128, 8, 64, 131, 79, None)', "call": 'float(abs(xi - gap / own) > 0.02)', "gold_call": '1.0', "tol": 0.0},
    ]
