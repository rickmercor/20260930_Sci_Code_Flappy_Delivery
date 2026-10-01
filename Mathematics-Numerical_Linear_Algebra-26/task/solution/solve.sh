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

def schrodinger_rhs(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Reference implementation."""
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError('A and B must be square matrices of the same size')
    if not (isinstance(alpha, (int, float)) and float(alpha) >= 0):
        raise ValueError('alpha must be >= 0')
    return 0.5j * (B @ A + A @ B) + 1j * float(alpha) * (A * A * A)

import numpy as np

def rk4_integrate(A0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float) -> np.ndarray:
    """Reference implementation."""
    A0 = np.asarray(A0, dtype=complex)
    if h <= 0:
        raise ValueError('h must be > 0')
    if t_end < t0:
        raise ValueError('t_end must be >= t0')
    n_steps = int(round((t_end - t0) / h))
    A = A0.copy()
    for _ in range(n_steps):
        k1 = rhs_fn(A)
        k2 = rhs_fn(A + 0.5 * h * k1)
        k3 = rhs_fn(A + 0.5 * h * k2)
        k4 = rhs_fn(A + h * k3)
        A = A + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return A

import numpy as np

def qdeim_select(U: np.ndarray) -> np.ndarray:
    """Reference implementation of QDEIM (Algorithm 1)."""
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError('U must be a 2D array')
    m, r = U.shape
    if r > m:
        raise ValueError('U must have at most as many columns as rows')
    if r == 0:
        return np.array([], dtype=int)
    U_work = U.astype(complex).copy()
    indices = np.empty(r, dtype=int)
    for k in range(r):
        row_norms = np.linalg.norm(U_work, axis=1).real
        pk = int(np.argmax(row_norms))
        indices[k] = pk
        u = U_work[pk, :].conj()
        u = u / np.linalg.norm(u)
        proj = U_work @ u
        U_work = U_work - np.outer(proj, u.conj())
    return indices

import numpy as np

def oblique_project(U: np.ndarray, V: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)
    Z = np.asarray(Z, dtype=complex)
    n, r = U.shape
    I_U = qdeim_select(U)
    I_V = qdeim_select(V)
    L_U = np.linalg.inv(U[I_U, :])
    R_V = np.linalg.solve(V[I_V, :].conj().T, V.conj().T)
    Z_rows = Z[I_U, :]
    Z_cols = Z[:, I_V]
    Z_sub = Z[np.ix_(I_U, I_V)]
    return U @ L_U @ (Z_rows - Z_sub @ R_V) + Z_cols @ R_V

import numpy as np
from typing import Tuple

def prk2_qdeim_step(U: np.ndarray, s: np.ndarray, Vh: np.ndarray, rhs_fn, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    U = np.asarray(U, dtype=complex)
    s = np.asarray(s, dtype=float)
    Vh = np.asarray(Vh, dtype=complex)
    Y = U @ np.diag(s) @ Vh
    V = Vh.conj().T
    F1 = rhs_fn(Y)
    P1 = oblique_project(U, V, F1)
    Z2 = Y + h * P1
    U2, s2, Vh2 = np.linalg.svd(Z2, full_matrices=False)
    U2, s2, Vh2 = (U2[:, :r], s2[:r], Vh2[:r, :])
    Y2 = U2 @ np.diag(s2) @ Vh2
    V2 = Vh2.conj().T
    F2 = rhs_fn(Y2)
    P2 = oblique_project(U2, V2, F2)
    Y_new = Y + 0.5 * h * (P1 + P2)
    U_n, s_n, Vh_n = np.linalg.svd(Y_new, full_matrices=False)
    return (U_n[:, :r], s_n[:r], Vh_n[:r, :])

import numpy as np
from typing import Tuple

def prk2_qdeim_integrate(U0: np.ndarray, s0: np.ndarray, Vh0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    if h <= 0:
        raise ValueError('h must be > 0')
    if t_end < t0:
        raise ValueError('t_end must be >= t0')
    n_steps = int(round((t_end - t0) / h))
    U, s, Vh = (np.asarray(U0, dtype=complex).copy(), np.asarray(s0, dtype=float).copy(), np.asarray(Vh0, dtype=complex).copy())
    for _ in range(n_steps):
        U, s, Vh = prk2_qdeim_step(U, s, Vh, rhs_fn, h, r)
    return (U, s, Vh)

import numpy as np

def kron_qdeim_indices(left: np.ndarray, right: np.ndarray, core: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    core = np.asarray(core)
    if left.ndim != 2 or right.ndim != 2 or core.ndim != 2:
        raise ValueError('left, right, and core must be matrices')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or not np.all(np.isfinite(core)):
        raise ValueError('all inputs must be finite')
    m1, r = left.shape
    m2, r_right = right.shape
    if r < 1 or m1 < r or m2 < r or r_right != r or core.shape != (r * r, r):
        raise ValueError('incompatible Kronecker sampling dimensions')
    eye = np.eye(r)
    if not np.allclose(left.conj().T @ left, eye, rtol=1e-11, atol=1e-12):
        raise ValueError('left must have orthonormal columns')
    if not np.allclose(right.conj().T @ right, eye, rtol=1e-11, atol=1e-12):
        raise ValueError('right must have orthonormal columns')
    if not np.allclose(core.conj().T @ core, eye, rtol=1e-11, atol=1e-12):
        raise ValueError('core must have orthonormal columns')
    left_indices = qdeim_select(left)
    right_indices = qdeim_select(right)
    sampled = np.kron(left[left_indices, :], right[right_indices, :]) @ core
    qhat, triangular = np.linalg.qr(sampled, mode='reduced')
    diagonal = np.diag(triangular)
    scale = max(1.0, float(np.linalg.norm(sampled, ord=2)))
    if np.any(np.abs(diagonal) <= 1e-12 * scale):
        raise ValueError('the sampled Kronecker core must have full column rank')
    phases = diagonal / np.abs(diagonal)
    qhat = qhat * phases
    pair_indices = qdeim_select(qhat)
    first = pair_indices // r
    second = pair_indices % r
    full = left_indices[first] * m2 + right_indices[second]
    if np.unique(full).size != r:
        raise RuntimeError('the mapped Kronecker indices must be distinct')
    return np.asarray(full, dtype=int)

import numpy as np
from typing import Dict, Any, Tuple

def dlra_deim_pipeline(n: int, alpha: float, r: int, h_ref: float, h_dlra: float, t0_pre: float, T: float) -> Dict[str, Any]:
    """Reference implementation."""
    sigma = 0.15 * n
    mu1, mu2 = (0.7 * n, 0.4 * n)
    nu1, nu2 = (0.6 * n, 0.3 * n)
    j = np.arange(1, n + 1, dtype=float)
    k = np.arange(1, n + 1, dtype=float)
    A0 = (
        np.exp(-((j[:, None] - mu1) ** 2 + (k[None, :] - nu1) ** 2) / sigma ** 2)
        + np.exp(-((j[:, None] - mu2) ** 2 + (k[None, :] - nu2) ** 2) / sigma ** 2)
    ).astype(complex)
    B = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    rhs_fn = lambda A: schrodinger_rhs(A, B, alpha)
    A_t0 = rk4_integrate(A0, rhs_fn, 0.0, t0_pre, h_ref)
    A_ref = rk4_integrate(A_t0, rhs_fn, t0_pre, T, h_ref)
    U0, s0, Vh0 = np.linalg.svd(A_t0, full_matrices=False)
    _tensor_core = np.eye(int(r) * int(r), int(r), dtype=float)
    _tensor_indices = kron_qdeim_indices(U0[:, :int(r)], Vh0[:int(r), :].T, _tensor_core)
    if _tensor_indices.shape != (int(r),) or np.unique(_tensor_indices).size != int(r):
        raise RuntimeError('the two-stage Kronecker QDEIM certificate failed')
    U0, s0, Vh0 = (U0[:, :r], s0[:r], Vh0[:r, :])
    U_N, s_N, Vh_N = prk2_qdeim_integrate(U0, s0, Vh0, rhs_fn, t0_pre, T, h_dlra, r)
    Y_N = U_N @ np.diag(s_N) @ Vh_N
    n_steps = int(round((T - t0_pre) / h_dlra))
    rel_error = float(np.linalg.norm(A_ref - Y_N, 'fro') / np.linalg.norm(A_ref, 'fro'))
    answer = int(np.floor(-np.log10(rel_error)))
    return {'rel_error': rel_error, 'answer': answer, 'n_steps': n_steps}
SCICODE_GOLD_EOF
