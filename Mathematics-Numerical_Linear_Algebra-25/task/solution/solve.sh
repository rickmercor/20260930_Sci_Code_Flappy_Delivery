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

def construct_clustered_ls_data(
    m: int,
    n: int,
    s: np.ndarray,
    seed: int,
) -> np.ndarray:
    np = __import__("numpy")
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    if m < 1 or n < 1 or m < n:
        raise ValueError("require m >= n >= 1")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(seed))
    U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = U @ np.diag(s) @ V.T
    b = rng.standard_normal(m)
    return np.hstack([A, b[:, None]])

import numpy as np

def form_sketched_range(A: np.ndarray, n_sketch: int, seed: int) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array with at least one row and column")
    if not isinstance(n_sketch, (int, np.integer)) or int(n_sketch) < 1:
        raise ValueError("n_sketch must be an integer >= 1")
    rng = np.random.default_rng(int(seed))
    S = rng.standard_normal((int(n_sketch), A.shape[0]))
    return S @ A

import numpy as np

def _lu_row_pivots(M: np.ndarray, k: int) -> np.ndarray:
    np = __import__("numpy")
    A = np.array(M, dtype=float, copy=True)
    m, n = A.shape
    piv = np.arange(m)
    for i in range(min(m, n)):
        j = i + int(np.argmax(np.abs(A[i:, i])))
        if j != i:
            A[[i, j]] = A[[j, i]]
            piv[[i, j]] = piv[[j, i]]
        pivot = A[i, i]
        if abs(pivot) > 0.0:
            A[i + 1 :, i] /= pivot
            A[i + 1 :, i + 1 :] -= np.outer(A[i + 1 :, i], A[i, i + 1 :])
    return piv[:k]


def select_cur_indices(A: np.ndarray, Y: np.ndarray, ell: int) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if A.ndim != 2 or Y.ndim != 2:
        raise ValueError("A and Y must be 2D")
    m, n = A.shape
    if Y.shape[1] != n:
        raise ValueError("Y must have n columns matching A")
    if not isinstance(ell, (int, np.integer)) or int(ell) < 1:
        raise ValueError("ell must be an integer >= 1")
    ell = int(ell)
    if ell > min(m, n, Y.shape[0]):
        raise ValueError("ell cannot exceed min(m, n, n_sketch)")
    J = _lu_row_pivots(Y.T, ell)
    I = _lu_row_pivots(A[:, J], ell)
    return np.concatenate([I, J]).astype(float)

import numpy as np

def _unpack_indices(index_vector: np.ndarray, m: int, n: int):
    np = __import__("numpy")
    idx = np.asarray(index_vector, dtype=float).reshape(-1)
    if idx.size < 2 or idx.size % 2 != 0:
        raise ValueError("index_vector must have even positive length")
    ell = idx.size // 2
    I = np.rint(idx[:ell]).astype(int)
    J = np.rint(idx[ell:]).astype(int)
    if np.any(I < 0) or np.any(I >= m) or np.any(J < 0) or np.any(J >= n):
        raise ValueError("indices out of range")
    if len(np.unique(I)) != ell or len(np.unique(J)) != ell:
        raise ValueError("I and J must each contain ell distinct indices")
    return I, J, ell


def cur_core_matrix(A: np.ndarray, index_vector: np.ndarray) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array")
    I, J, _ = _unpack_indices(index_vector, A.shape[0], A.shape[1])
    Aij = A[np.ix_(I, J)]
    return np.linalg.pinv(Aij)

import numpy as np

def _unpack_indices(index_vector: np.ndarray, m: int, n: int):
    np = __import__("numpy")
    idx = np.asarray(index_vector, dtype=float).reshape(-1)
    if idx.size < 2 or idx.size % 2 != 0:
        raise ValueError("index_vector must have even positive length")
    ell = idx.size // 2
    I = np.rint(idx[:ell]).astype(int)
    J = np.rint(idx[ell:]).astype(int)
    if np.any(I < 0) or np.any(I >= m) or np.any(J < 0) or np.any(J >= n):
        raise ValueError("indices out of range")
    if len(np.unique(I)) != ell or len(np.unique(J)) != ell:
        raise ValueError("I and J must each contain ell distinct indices")
    return I, J, ell


def cur_captured_singular_values(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array")
    I, J, ell = _unpack_indices(index_vector, A.shape[0], A.shape[1])
    U = np.asarray(U, dtype=float)
    if U.shape != (ell, ell) or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite array of shape (ell, ell)")

    C = A[:, J]
    R = A[I, :]
    gram = C.T @ C
    try:
        T_C = np.linalg.cholesky(gram).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("C^T C must be SPD") from exc
    _, T_R = np.linalg.qr(R.T, mode="reduced")
    M = T_C @ U @ T_R.T
    sigma = np.linalg.svd(M, compute_uv=False, full_matrices=False)
    return np.asarray(sigma, dtype=float)

import numpy as np

def _unpack_indices(index_vector: np.ndarray, m: int, n: int):
    np = __import__("numpy")
    idx = np.asarray(index_vector, dtype=float).reshape(-1)
    if idx.size < 2 or idx.size % 2 != 0:
        raise ValueError("index_vector must have even positive length")
    ell = idx.size // 2
    I = np.rint(idx[:ell]).astype(int)
    J = np.rint(idx[ell:]).astype(int)
    if np.any(I < 0) or np.any(I >= m) or np.any(J < 0) or np.any(J >= n):
        raise ValueError("indices out of range")
    if len(np.unique(I)) != ell or len(np.unique(J)) != ell:
        raise ValueError("I and J must each contain ell distinct indices")
    return I, J, ell


def build_spectral_pinv_factors(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
    sigma: np.ndarray,
    mu: float,
) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array")
    if not np.isfinite(mu) or float(mu) < 0.0:
        raise ValueError("mu must be a finite number >= 0")
    I, J, ell = _unpack_indices(index_vector, A.shape[0], A.shape[1])

    U = np.asarray(U, dtype=float)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if U.shape != (ell, ell) or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite array of shape (ell, ell)")
    if sigma.shape != (ell,) or np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must contain ell positive finite values")

    C = A[:, J]
    R = A[I, :]
    gram = C.T @ C
    try:
        T_C = np.linalg.cholesky(gram).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("C^T C must be SPD") from exc
    Q_R, T_R = np.linalg.qr(R.T, mode="reduced")
    M = T_C @ U @ T_R.T
    _, sigma_from_U, Vh = np.linalg.svd(M, full_matrices=False)
    if not np.allclose(sigma, sigma_from_U, rtol=1e-10, atol=1e-12):
        raise ValueError("sigma is inconsistent with A, index_vector, and U")

    Vhat = Q_R @ Vh.T
    gamma = np.sqrt(sigma**2 + float(mu) ** 2)
    inverse_scale = gamma[-1] / gamma - 1.0
    return np.vstack([inverse_scale, Vhat])

import numpy as np

def _apply_pinv_factors(pinv_factors: np.ndarray, vec: np.ndarray) -> np.ndarray:
    inverse_scale = pinv_factors[0]
    Vhat = pinv_factors[1:]
    return vec + Vhat @ (inverse_scale * (Vhat.T @ vec))


def _lsqr_gk(_matvec, _rmatvec, rhs: np.ndarray, niter: int, dim: int) -> np.ndarray:
    np = __import__("numpy")
    beta = float(np.linalg.norm(rhs))
    if beta == 0.0:
        return np.zeros(dim)
    u = rhs / beta
    tmp = _rmatvec(u)
    alpha = float(np.linalg.norm(tmp))
    if alpha == 0.0:
        return np.zeros(dim)
    v = tmp / alpha
    w = v.copy()
    x = np.zeros(dim)
    phibar = beta
    rhobar = alpha
    for _ in range(niter):
        u = _matvec(v) - alpha * u
        beta = float(np.linalg.norm(u))
        if beta == 0.0:
            break
        u = u / beta
        v = _rmatvec(u) - beta * v
        alpha = float(np.linalg.norm(v))
        if alpha == 0.0:
            rho = np.hypot(rhobar, beta)
            c = rhobar / rho
            phi = c * phibar
            x = x + (phi / rho) * w
            break
        v = v / alpha
        rho = np.hypot(rhobar, beta)
        c = rhobar / rho
        s = beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
    return x


def preconditioned_lsqr(
    A: np.ndarray,
    b: np.ndarray,
    pinv_factors: np.ndarray,
    mu: float,
    niter: int,
) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    pinv_factors = np.asarray(pinv_factors, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have shape (m,)")
    if (
        pinv_factors.ndim != 2
        or pinv_factors.shape[0] != n + 1
        or pinv_factors.shape[1] < 1
        or pinv_factors.shape[1] > n
        or not np.all(np.isfinite(pinv_factors))
    ):
        raise ValueError("pinv_factors must be finite with shape (n+1, ell)")
    Vhat = pinv_factors[1:]
    gram = Vhat.T @ Vhat
    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=1e-10, atol=1e-10):
        raise ValueError("the captured right basis must have orthonormal columns")
    if not np.isfinite(mu) or float(mu) < 0.0:
        raise ValueError("mu must be a finite number >= 0")
    if not isinstance(niter, (int, np.integer)) or int(niter) < 1:
        raise ValueError("niter must be an integer >= 1")

    mu_f = float(mu)

    def _pinv_map(vec):
        return _apply_pinv_factors(pinv_factors, vec)

    A_mu = np.vstack([A, mu_f * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])

    def _matvec(y):
        return A_mu @ _pinv_map(y)

    def _rmatvec(u):
        return _pinv_map(A_mu.T @ u)

    y = _lsqr_gk(_matvec, _rmatvec, b_aug, int(niter), n)
    return _pinv_map(y)

import numpy as np

def run_cur_spectral_lsqr(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    n_sketch: int,
    sketch_seed: int,
    ell: int,
    mu: float,
    niter: int,
) -> float:
    data = construct_clustered_ls_data(m, n, s, data_seed)
    A = data[:, :-1]
    b = data[:, -1]
    Y = form_sketched_range(A, n_sketch, sketch_seed)
    index_vector = select_cur_indices(A, Y, ell)
    U = cur_core_matrix(A, index_vector)
    sigma = cur_captured_singular_values(A, index_vector, U)
    pinv_factors = build_spectral_pinv_factors(A, index_vector, U, sigma, mu)
    x = preconditioned_lsqr(A, b, pinv_factors, mu, niter)
    return float(x[0])
SCICODE_GOLD_EOF
