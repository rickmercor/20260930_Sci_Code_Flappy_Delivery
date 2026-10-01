#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def construct_kernel_matrix(m, n, h=None):
    if isinstance(m, bool) or isinstance(n, bool):
        raise ValueError("m and n must be integers")
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    if m < n:
        raise ValueError("require m >= n (tall matrix)")
    if h is None:
        h = 1.0 if n == 1 else 1.0 / (n - 1)
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a positive finite float")
    t = np.linspace(0.0, 1.0, int(m))
    c = np.linspace(0.0, 1.0, int(n))
    A = np.exp(-((t[:, None] - c[None, :]) ** 2) / (2.0 * h * h)) / np.sqrt(float(m))
    return np.asarray(A, dtype=np.float64)

import numpy as np
from scipy.fft import dct


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    f = 3
    while f * f <= v:
        if v % f == 0:
            return False
        f += 2
    return True


def build_sketch_operator(m, s, p, q):
    for name, v in (("m", m), ("s", s), ("p", p), ("q", q)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    m, s, p, q = int(m), int(s), int(p), int(q)
    if m < 1 or s < 1:
        raise ValueError("m and s must be positive")
    if s > m:
        raise ValueError("require s <= m")
    if p <= m or not _is_prime(p):
        raise ValueError("p must be a prime strictly greater than m")
    if q < 1:
        raise ValueError("q must be positive")

    residues = {(k * k) % p for k in range(1, p)}
    signs = np.array(
        [1.0 if (i % p) in residues else -1.0 for i in range(1, m + 1)],
        dtype=np.float64,
    )
    rows = np.array([(q * k) % m for k in range(s)], dtype=int)
    if len(np.unique(rows)) != s:
        raise ValueError("stride q does not produce s distinct rows modulo m")

    F = dct(np.eye(m, dtype=np.float64), type=2, axis=0, norm="ortho")
    S = np.sqrt(m / s) * (F[rows, :] * signs[None, :])
    return np.asarray(S, dtype=np.float64)

import numpy as np


def compute_sts_singular_values(A, S):
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


def compute_sts_right_factor(A, S):
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

    theta = compute_sts_singular_values(A, S)
    # Exact <= 0.0 misses numerically rank-deficient sketches (tiny positive
    # singular values from roundoff). Use a relative machine-eps threshold.
    if theta[0] <= 0.0 or theta[-1] <= max(m, n) * np.finfo(np.float64).eps * theta[0]:
        raise ValueError("S A must have full column rank")

    _, _, Vt = np.linalg.svd(S @ A, full_matrices=False)
    V = np.array(Vt.T[:, :n], dtype=np.float64)
    return _fix_sts_right_factor_signs(V)

import numpy as np

def form_nearest_sts_orthogonal(A, S, V):
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

    theta = compute_sts_singular_values(A, S)
    if theta[0] <= 0.0 or theta[-1] <= max(m, n) * np.finfo(np.float64).eps * theta[0]:
        raise ValueError("S A must have full column rank")

    # Left factor in R^{m x n}; never the s-row factor U_1 of S A.
    W = (A @ V) / theta[None, :]
    if W.shape != (m, n):
        raise ValueError("left factor W must have shape (m, n)")
    return W @ V.T

import numpy as np





def form_nearest_euclidean_orthogonal(A):
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

import numpy as np

def realized_embedding_distortion(A, S):
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
    B = form_nearest_euclidean_orthogonal(A)
    SB = S @ B
    G = SB.T @ SB - np.eye(n)
    return float(np.linalg.norm(G, 2))

import numpy as np

def orchestrate_sketched_polar_audit(m, n, s, p, q, h=None):
    A = construct_kernel_matrix(m, n, h)
    S = build_sketch_operator(m, s, p, q)

    theta = compute_sts_singular_values(A, S)
    if theta[0] <= 0.0 or theta[-1] <= max(A.shape) * np.finfo(np.float64).eps * theta[0]:
        raise ValueError("the sketch destroys the column rank of A")

    # V is computed once and consumed by the STS polar step. Wrong signs fail
    # there; a pipeline that never calls the right-factor step is incomplete.
    V = compute_sts_right_factor(A, S)
    P = form_nearest_sts_orthogonal(A, S, V)
    T = form_nearest_euclidean_orthogonal(A)
    eps = realized_embedding_distortion(A, S)
    if not np.isfinite(eps) or not (0.0 < eps < 1.0):
        raise ValueError("the realized distortion must lie strictly in (0, 1)")

    gap = float(np.linalg.norm(P - T, 2))
    Qp = form_nearest_euclidean_orthogonal(P)
    own = float(np.linalg.norm(P - Qp, 2))
    if own <= 0.0 or not np.isfinite(own):
        raise ValueError("the Higham polar gap of P must be positive")
    return (gap / own) * (1.0 - eps) / eps
SCICODE_GOLD_EOF
