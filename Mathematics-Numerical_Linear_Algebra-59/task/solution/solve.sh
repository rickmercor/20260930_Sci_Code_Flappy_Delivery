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


def extract_bse_eigenvector(R: np.ndarray, C: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square 2D array")
    if not np.allclose(R, R.T, atol=1e-8):
        raise ValueError("R must be symmetric")
    if not np.allclose(C, C.T, atol=1e-8):
        raise ValueError("C must be symmetric")
    m = R.shape[0]

    H = np.block([[R, C], [-C, -R]])
    eigvals, eigvecs = np.linalg.eig(H)

    candidates = []
    for i in range(len(eigvals)):
        lam = eigvals[i]
        if abs(lam.imag) < 1e-8 and alpha < lam.real < beta:
            candidates.append(i)
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one eigenvalue in ({alpha},{beta}), found {len(candidates)}")
    idx = candidates[0]
    vec = eigvecs[:, idx]
    if np.max(np.abs(vec.imag)) > 1e-8 * max(1.0, np.max(np.abs(vec.real))):
        raise ValueError("eigenvector has non-negligible imaginary part")
    vec = vec.real

    k = np.argmax(np.abs(vec))
    if vec[k] < 0:
        vec = -vec

    vec34 = vec / np.linalg.norm(vec)
    v32 = vec34[:32]
    v = v32 / np.linalg.norm(v32)
    return v

import numpy as np


def build_perturbed_matrix(v: np.ndarray, N: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    N = np.asarray(N, dtype=float)
    if v.shape != (32,):
        raise ValueError("v must have shape (32,)")
    if N.shape != (32, 32):
        raise ValueError("N must have shape (32,32)")
    if not np.allclose(N, -N.T, atol=1e-8):
        raise ValueError("N must be skew-symmetric")
    A = 3.0 * np.outer(v, v) + np.eye(32) + N
    return A

import numpy as np
from scipy.linalg import polar


def _helper_riemannian_grad(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    B = Q.conj().T @ A @ Q
    S = B - np.diag(np.diag(B))
    comm = B.conj().T @ S - S @ B.conj().T
    skew_comm = 0.5 * (comm - comm.conj().T)
    return 2.0 * Q @ skew_comm


def _helper_retract(Q: np.ndarray, tangent_dir: np.ndarray) -> np.ndarray:
    Up, _ = polar(Q + tangent_dir)
    return Up


def _helper_single_run(A: np.ndarray, Q0: np.ndarray, n_iters: int, step0: float):
    Q = Q0.copy()
    step = step0
    for _ in range(n_iters):
        grad = _helper_riemannian_grad(A, Q)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        Q_new = _helper_retract(Q, -step * grad)
        B_new = Q_new.conj().T @ A @ Q_new
        f_new = np.linalg.norm(B_new - np.diag(np.diag(B_new))) ** 2
        B_old = Q.conj().T @ A @ Q
        f_old = np.linalg.norm(B_old - np.diag(np.diag(B_old))) ** 2
        if f_new <= f_old:
            Q = Q_new
        if f_new < f_old:
            step = min(step * 1.02, 0.2)
        else:
            step *= 0.5
            if step < 1e-12:
                break
    B = Q.conj().T @ A @ Q
    f = float(np.linalg.norm(B - np.diag(np.diag(B))) ** 2)
    return np.diag(B).copy(), f


def closest_normal_diagonal(A: np.ndarray) -> np.ndarray:
    # The procedure's fixed hyperparameters, as pinned in the step description.
    n_iters, step0, n_restarts, base_seed = 4000, 0.05, 40, 200000
    A = np.asarray(A)
    if A.shape != (32, 32):
        raise ValueError("A must have shape (32,32)")
    n = A.shape[0]
    best_diag, best_f = None, np.inf
    for r in range(n_restarts):
        rng = np.random.default_rng(base_seed + r)
        Q0, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        diag_vals, f = _helper_single_run(A, Q0, n_iters, step0)
        if f < best_f:
            best_f, best_diag = f, diag_vals
    e = np.sort(np.real(best_diag))[::-1]
    return e

import numpy as np


def build_base_matrix(e: np.ndarray, U: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    U = np.asarray(U, dtype=float)
    if e.shape != (32,):
        raise ValueError("e must have shape (32,)")
    if U.shape != (32, 32):
        raise ValueError("U must have shape (32,32)")
    if not np.allclose(U @ U.T, np.eye(32), atol=1e-5):
        raise ValueError("U must be orthogonal")
    s = e - e.min() + 0.5
    A0 = U @ np.diag(s) @ U.T
    A0 = 0.5 * (A0 + A0.T)
    return A0

import numpy as np


def build_matrix_A1(A0: np.ndarray, M1: np.ndarray) -> np.ndarray:
    A0 = np.asarray(A0, dtype=float)
    M1 = np.asarray(M1, dtype=float)
    if A0.shape != (32, 32) or M1.shape != (32, 32):
        raise ValueError("A0 and M1 must have shape (32,32)")
    n = 32
    X = A0 + 2.0 * np.eye(n) + 0.1 * M1
    X = 0.5 * (X + X.T)
    min_eig = np.linalg.eigvalsh(X).min()
    A1 = X + (abs(min(0.0, min_eig)) + 0.5) * np.eye(n)
    return A1

import numpy as np


def build_matrix_A3(M2: np.ndarray) -> np.ndarray:
    M2 = np.asarray(M2, dtype=float)
    if M2.shape != (32, 32):
        raise ValueError("M2 must have shape (32,32)")
    n = 32
    X = 0.3 * np.eye(n) + 0.05 * M2
    X = 0.5 * (X + X.T)
    min_eig = np.linalg.eigvalsh(X).min()
    A3 = X + (abs(min(0.0, min_eig)) + 0.2) * np.eye(n)
    return A3

import numpy as np


def build_matrix_A2(A3: np.ndarray, M3: np.ndarray) -> np.ndarray:
    A3 = np.asarray(A3, dtype=float)
    M3 = np.asarray(M3, dtype=float)
    if A3.shape != (32, 32) or M3.shape != (32, 32):
        raise ValueError("A3 and M3 must have shape (32,32)")
    n = 32
    A2_0 = A3 + 1.5 * np.eye(n) + 0.05 * M3
    A2_0 = 0.5 * (A2_0 + A2_0.T)
    d = np.linalg.eigvalsh(A2_0 - A3).min()
    if d < 0:
        A2 = A2_0 - d * np.eye(n) + 1e-6 * np.eye(n)
    else:
        A2 = A2_0
    return A2

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def generalized_eigenvalue_definite_pencil(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != (32, 32) or Y.shape != (32, 32):
        raise ValueError("X and Y must have shape (32,32)")
    if not np.allclose(Y, Y.T, atol=1e-8):
        raise ValueError("Y must be symmetric")
    try:
        L = cholesky(Y, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError("Y must be positive definite")
    W = solve_triangular(L, X, lower=True)
    Z = solve_triangular(L, W.T, lower=True).T
    Z = 0.5 * (Z + Z.T)
    lam_max = float(np.linalg.eigvalsh(Z).max())
    return lam_max

import numpy as np

# NOTE for Studio authoring: the model-facing compute_stability_product (once
# implemented by a candidate) should call the PUBLIC function names from
# sub_problems 01-08 (extract_bse_eigenvector, build_perturbed_matrix, etc.),
# since only those are available to it. The GOLD compute_stability_product
# below must instead call the -prefixed versions of those same steps --
# calling the public names there would let a candidate's own buggy step 1-8
# implementations leak into what is supposed to be the independent reference
# answer, letting a broken pipeline pass step 9's comparison.


def compute_stability_product(R: np.ndarray, C: np.ndarray, alpha: float, beta: float,
                                       N: np.ndarray, U: np.ndarray, M1: np.ndarray,
                                       M2: np.ndarray, M3: np.ndarray) -> float:
    for name, mat, shape in (("N", N, (32, 32)), ("U", U, (32, 32)), ("M1", M1, (32, 32)),
                              ("M2", M2, (32, 32)), ("M3", M3, (32, 32))):
        arr = np.asarray(mat, dtype=float)
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.allclose(N, -np.asarray(N).T, atol=1e-8):
        raise ValueError("N must be skew-symmetric")
    if not np.allclose(U @ U.T, np.eye(32), atol=1e-5):
        raise ValueError("U must be orthogonal")

    v = extract_bse_eigenvector(R, C, alpha, beta)
    A = build_perturbed_matrix(v, N)
    e = closest_normal_diagonal(A)
    A0 = build_base_matrix(e, U)
    A1 = build_matrix_A1(A0, M1)
    A3 = build_matrix_A3(M2)
    A2 = build_matrix_A2(A3, M3)
    # lambda_max(A,B) follows arXiv:2607.15636's own convention (Section 2, Theorem
    # 3.1): the largest lambda solving det(lambda*A - B) = 0, equivalently
    # det(B - lambda*A) = 0. generalized_eigenvalue_definite_pencil(X, Y) solves
    # det(X - lambda*Y) = 0, so lambda_max(A1,A0) [det(A0 - lambda*A1) = 0] is
    # called as (A0, A1), and lambda_max(A2,A3) [det(A3 - lambda*A2) = 0] as (A3, A2).
    lam1 = generalized_eigenvalue_definite_pencil(A0, A1)
    lam2 = generalized_eigenvalue_definite_pencil(A3, A2)
    return float(lam1 * lam2)
SCICODE_GOLD_EOF
