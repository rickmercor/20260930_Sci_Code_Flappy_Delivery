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

def construct_krylov_operator(n: int = 14) -> np.ndarray:
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    K = np.zeros((n, n), dtype=np.float64)
    for i1 in range(1, n + 1):
        i = i1 - 1
        K[i, i] = 1.0 + 0.05 * i1
        if i + 1 < n:
            K[i, i + 1] = 0.8
            K[i + 1, i] = 0.01
        if i + 2 < n:
            K[i, i + 2] = 0.5
    return K

import numpy as np

def truncated_arnoldi_basis(A, b, m, k):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be nonempty square")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n or not np.all(np.isfinite(b)):
        raise ValueError("b must be finite length n")
    if not isinstance(m, (int, np.integer)) or not (1 <= int(m) <= n):
        raise ValueError("invalid m")
    if not isinstance(k, (int, np.integer)) or not (1 <= int(k) <= int(m)):
        raise ValueError("invalid k")
    if np.linalg.norm(b) == 0:
        raise ValueError("b must be nonzero")
    m = int(m)
    k = int(k)
    V = np.empty((n, m), dtype=np.float64)
    M = np.empty((n, m), dtype=np.float64)
    V[:, 0] = b / np.linalg.norm(b)
    M[:, 0] = A @ V[:, 0]
    for j in range(1, m):
        w = M[:, j - 1].copy()
        for i in range(max(0, j - k), j):
            w -= V[:, i] * float(V[:, i] @ M[:, j - 1])
        nw = float(np.linalg.norm(w))
        if not np.isfinite(nw) or nw <= np.finfo(np.float64).eps * np.linalg.norm(M[:, j - 1]):
            raise ValueError("Arnoldi breakdown")
        V[:, j] = w / nw
        M[:, j] = A @ V[:, j]
    return V, M

import numpy as np

def qdeim_indices(V, tie_tol=1e-12):
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2 or min(V.shape) < 1 or not np.all(np.isfinite(V)):
        raise ValueError("V must be finite nonempty 2D")
    n, m = V.shape
    if n < m or np.linalg.matrix_rank(V) != m:
        raise ValueError("V must have full column rank with n>=m")
    if not isinstance(tie_tol, (int, float, np.integer, np.floating)) or not np.isfinite(tie_tol) or float(tie_tol) < 0:
        raise ValueError("invalid tie_tol")
    tol = float(tie_tol)
    B = V.T.copy()
    selected = []
    Q = []
    for _ in range(m):
        residuals = {}
        norms = {}
        for j in range(n):
            if j in selected:
                continue
            r = B[:, j].copy()
            for q in Q:
                r -= q * float(q @ r)
            for q in Q:
                r -= q * float(q @ r)
            residuals[j] = r
            norms[j] = float(np.linalg.norm(r))
        mx = max(norms.values())
        cand = [j for j, v in norms.items() if abs(v - mx) <= tol]
        j = min(cand)
        r = residuals[j]
        nr = float(np.linalg.norm(r))
        if nr <= np.finfo(np.float64).eps:
            raise ValueError("rank-deficient pivot")
        selected.append(j)
        Q.append(r / nr)
    return np.asarray(selected, dtype=np.int64) + 1

import numpy as np

def gappypod_e_oversample(V, initial_indices, s, tie_tol=1e-12):
    V = np.asarray(V, dtype=np.float64)
    idx = np.asarray(initial_indices)
    if V.ndim != 2 or min(V.shape) < 1 or not np.all(np.isfinite(V)):
        raise ValueError("V must be finite nonempty 2D")
    n, m = V.shape
    if n < m or np.linalg.matrix_rank(V) != m:
        raise ValueError("V must be full column rank")
    if idx.ndim != 1 or idx.size != m or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("initial_indices must contain m integers")
    idx = idx.astype(np.int64)
    if np.unique(idx).size != m or np.any(idx < 1) or np.any(idx > n):
        raise ValueError("invalid initial indices")
    if np.linalg.matrix_rank(V[idx - 1, :]) != m:
        raise ValueError("initial sampled basis must be full rank")
    if not isinstance(s, (int, np.integer)) or not (m <= int(s) <= n):
        raise ValueError("invalid s")
    if not isinstance(tie_tol, (int, float, np.integer, np.floating)) or not np.isfinite(tie_tol) or float(tie_tol) < 0:
        raise ValueError("invalid tie_tol")
    chosen = list((idx - 1).tolist())
    tol = float(tie_tol)
    while len(chosen) < int(s):
        SV = V[chosen, :]
        _, sing, Vh = np.linalg.svd(SV, full_matrices=False)
        g = float(sing[-2] ** 2 - sing[-1] ** 2) if m > 1 else 0.0
        W = Vh @ V.T
        y = np.sum(W * W, axis=0)
        disc = np.maximum((g + y) ** 2 - 4.0 * g * (W[-1, :] ** 2), 0.0)
        score = g + y - np.sqrt(disc)
        score[chosen] = -np.inf
        mx = float(np.max(score))
        cand = np.flatnonzero(np.abs(score - mx) <= tol)
        if cand.size == 0:
            raise ValueError("no admissible oversampling row")
        chosen.append(int(cand[0]))
    return np.asarray(chosen, dtype=np.int64) + 1

import numpy as np

def deterministic_sgmres_solve(V, M, b, indices):
    V = np.asarray(V, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    idx = np.asarray(indices)
    if V.ndim != 2 or M.shape != V.shape or not np.all(np.isfinite(V)) or not np.all(np.isfinite(M)):
        raise ValueError("V and M must be finite and same shape")
    n, m = V.shape
    if np.linalg.matrix_rank(V) != m or np.linalg.matrix_rank(M) != m:
        raise ValueError("V and M must be full column rank")
    if b.ndim != 1 or b.shape[0] != n or not np.all(np.isfinite(b)):
        raise ValueError("invalid b")
    if idx.ndim != 1 or idx.size < m or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("invalid indices")
    idx = idx.astype(np.int64)
    if np.unique(idx).size != idx.size or np.any(idx < 1) or np.any(idx > n):
        raise ValueError("invalid indices")
    SM = M[idx - 1, :]
    if np.linalg.matrix_rank(SM) != m:
        raise ValueError("sampled image must be full rank")
    Q, R = np.linalg.qr(SM, mode='reduced')
    y = np.linalg.solve(R, Q.T @ b[idx - 1])
    r = b - M @ y
    return y.astype(np.float64), float(r @ r)

import numpy as np

def deterministic_residual_inflation(M, b, residual_sq):
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if M.ndim != 2 or min(M.shape) < 1 or not np.all(np.isfinite(M)):
        raise ValueError("invalid M")
    n, m = M.shape
    if np.linalg.matrix_rank(M) != m:
        raise ValueError("M must be full column rank")
    if b.ndim != 1 or b.shape[0] != n or not np.all(np.isfinite(b)):
        raise ValueError("invalid b")
    if not isinstance(residual_sq, (int, float, np.integer, np.floating)) or not np.isfinite(residual_sq) or float(residual_sq) < 0:
        raise ValueError("invalid residual_sq")
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    r = b - M @ x
    opt = float(r @ r)
    if opt <= 0:
        raise ValueError("optimal residual must be positive")
    return float(residual_sq) / opt, opt

import numpy as np

def random_orthonormal_benchmark(n, r, ell, field="real"):
    if not all(isinstance(x, (int, np.integer)) for x in (n, r, ell)):
        raise ValueError("dimensions must be integers")
    n, r, ell = int(n), int(r), int(ell)
    if n < 1 or r < 1 or ell < 1 or r >= n or ell > n:
        raise ValueError("invalid dimensions")
    if field not in ("real", "complex"):
        raise ValueError("invalid field")
    if ell == n:
        return 1.0
    alpha = 1 if field == "real" else 0
    if r >= ell - alpha:
        raise ValueError("expectation is not finite")
    return float(1.0 + ((n - ell) / (n - r)) * (r / (ell - r - alpha)))

import numpy as np


def _sphere_quadratic_maximum(H, g, tau):
    """Dual solution of the equality-constrained trust-region problem."""
    d, U = np.linalg.eigh(H)
    h = U.T @ g
    top = float(d[-1])
    gaps = top - d
    eps = 64 * np.finfo(float).eps
    top_mask = gaps <= eps * max(1.0, float(np.max(np.abs(d))))
    lower = ~top_mask
    hard_norm = np.linalg.norm(h[lower] / gaps[lower])
    if np.linalg.norm(h[top_mask]) <= eps * max(1.0, np.linalg.norm(g)) and hard_norm <= tau:
        # Add the missing radius in the entire top eigenspace. Its
        # orientation is irrelevant to the optimal value.
        return float(top * tau**2 + np.sum(h[lower]**2 / gaps[lower]))
    lo = 0.0
    hi = max(1.0, float(np.linalg.norm(h) / tau))
    # Solve for the shift above the top eigenvalue, avoiding cancellation.
    while np.linalg.norm(h / (gaps + hi)) > tau:
        hi *= 2.0
    for _ in range(100):
        shift = (lo + hi) / 2.0
        if np.linalg.norm(h / (gaps + shift)) > tau:
            lo = shift
        else:
            hi = shift
    shift = hi
    return float((top + shift) * tau**2 + np.sum(h**2 / (gaps + shift)))


def fractional_sphere_maximum(A, g, c, B, tau):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    g = np.asarray(g, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be nonempty square")
    if B.shape != A.shape or g.shape != (A.shape[0],):
        raise ValueError("incompatible shapes")
    if not all(np.all(np.isfinite(x)) for x in (A, B, g)):
        raise ValueError("nonfinite array")
    if not np.isscalar(c) or not np.isscalar(tau) or not np.isfinite(c) or not np.isfinite(tau) or tau < 0:
        raise ValueError("invalid scalar")
    for X in (A, B):
        if np.max(np.abs(X - X.T)) > 1e-12 * max(1.0, np.max(np.abs(X))):
            raise ValueError("matrix must be symmetric")
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    if np.linalg.eigvalsh(B)[0] < -1e-12 * max(1.0, np.linalg.norm(B, 2)):
        raise ValueError("B must be positive semidefinite")
    if tau == 0:
        return float(c)
    bound = abs(c) + 2 * tau * np.linalg.norm(g) + tau**2 * np.linalg.norm(A, 2)
    lo, hi = -float(bound) - 1.0, float(bound) + 1.0
    for _ in range(80):
        rho = (lo + hi) / 2
        f = c - rho + _sphere_quadratic_maximum(A - rho * B, g, tau)
        if f > 0:
            lo = rho
        else:
            hi = rho
    return float((lo + hi) / 2)

import numpy as np

def deterministic_random_benchmark_ratio(K, b, m=5, k=1, s=7, tie_tol=1e-12, tau=0.65, weights=None):
    if isinstance(K, (bool, np.bool_)):
        raise ValueError("K must be a square matrix or a positive integer")
    if isinstance(K, (int, np.integer)):
        K = construct_krylov_operator(int(K))
    elif np.asarray(K).ndim != 2:
        raise ValueError("K must be a square matrix or a positive integer")
    V, M = truncated_arnoldi_basis(K, b, m, k)
    p0 = qdeim_indices(V, tie_tol)
    p = gappypod_e_oversample(V, p0, s, tie_tol)
    _, residual_sq = deterministic_sgmres_solve(V, M, b, p)
    rho_det, opt = deterministic_residual_inflation(M, b, residual_sq)
    rho_orth = random_orthonormal_benchmark(V.shape[0], V.shape[1], s, "real")
    if not np.isscalar(tau) or not np.isfinite(tau) or tau < 0:
        raise ValueError("invalid uncertainty radius")
    n = V.shape[0]
    weights = 1.0 + 0.1 * np.arange(1, n + 1) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite positive length n")
    if tau == 0:
        return float(fractional_sphere_maximum(np.zeros((1, 1)), np.zeros(1), rho_det, np.zeros((1, 1)), 0.0) / rho_orth)
    if n - m - 1 < 1:
        raise ValueError("positive radius requires a nonempty tangent space")
    b = np.asarray(b, dtype=float)
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    u = (b - M @ x) / np.sqrt(opt)
    Q = np.linalg.qr(np.column_stack((M, u)), mode="complete")[0]
    Z = Q[:, m + 1:]
    C = np.linalg.cholesky(Z.T @ (weights[:, None] * Z))
    T = np.linalg.solve(C, Z.T).T
    rows = p - 1
    # F maps residual perturbations to their oblique fitted component.
    F = M @ np.linalg.lstsq(M[rows], T[rows], rcond=None)[0]
    a = M @ np.linalg.lstsq(M[rows], u[rows], rcond=None)[0]
    B = T.T @ T
    A = B + F.T @ F
    g = F.T @ a
    rho_robust = fractional_sphere_maximum(A, g, rho_det, B, tau)
    return float(rho_robust / rho_orth)
SCICODE_GOLD_EOF
