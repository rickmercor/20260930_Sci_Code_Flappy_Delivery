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
def generate_deterministic_inputs(
    seed: int, p: int, m: int, k: int
) -> tuple:
    if not (isinstance(seed, (int, np.integer)) and seed >= 0):
        raise ValueError("seed must be a non-negative integer")

    if not (isinstance(p, (int, np.integer)) and p >= 1):
        raise ValueError("p must be a positive integer")

    if not (isinstance(m, (int, np.integer)) and m >= p):
        raise ValueError("m must be an integer >= p")

    if not (isinstance(k, (int, np.integer)) and k >= 1):
        raise ValueError("k must be a positive integer")

    rng = np.random.default_rng(int(seed))

    G = rng.standard_normal((int(p), int(p)))
    E = G @ G.T + float(p) * np.eye(int(p), dtype=float)

    C = rng.standard_normal((int(p), int(m)))
    Omega = rng.standard_normal((int(m), int(k)))

    return (
        E.astype(float),
        C.astype(float),
        Omega.astype(float),
    )

import numpy as np
def solve_diagonal_and_exact_actions(
    E: np.ndarray,
    C: np.ndarray,
    Omega: np.ndarray,
) -> tuple:
    E = np.asarray(E, dtype=float)
    C = np.asarray(C, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError("E must be square")

    if not np.allclose(E, E.T, rtol=0.0, atol=1e-12):
        raise ValueError("E must be symmetric")

    try:
        np.linalg.cholesky(E)
    except np.linalg.LinAlgError as exc:
        raise ValueError("E must be SPD") from exc

    if C.ndim != 2 or C.shape[0] != E.shape[0]:
        raise ValueError("C shape is incompatible with E")

    if Omega.ndim != 2 or Omega.shape[0] != C.shape[1]:
        raise ValueError("Omega shape is incompatible with C")

    E_D = np.diag(np.diag(E))
    C_Omega = C @ Omega

    Y = np.linalg.solve(E, C_Omega)
    Y_D = np.linalg.solve(E_D, C_Omega)

    return Y.astype(float), Y_D.astype(float)

import numpy as np
def form_sample_matrix(
    C: np.ndarray,
    Y: np.ndarray,
    Y_D: np.ndarray,
) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Y_D = np.asarray(Y_D, dtype=float)

    if C.ndim != 2 or Y.ndim != 2 or Y_D.ndim != 2:
        raise ValueError("C, Y, and Y_D must be 2D arrays")

    if Y.shape != Y_D.shape:
        raise ValueError("Y and Y_D must have the same shape")

    if C.shape[0] != Y.shape[0]:
        raise ValueError("C and Y have incompatible row dimensions")

    W = C.T @ (Y - Y_D)
    return W.astype(float)

import numpy as np
def thin_qr_orthonormal_basis(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array")

    m, k = W.shape

    if m < k:
        raise ValueError("W must satisfy m >= k")

    V, R = np.linalg.qr(W, mode="reduced")

    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0

    V = V * signs

    return V.astype(float)

import numpy as np
def form_regularized_core(
    W: np.ndarray,
    V: np.ndarray,
    Omega: np.ndarray,
    eps: float,
) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    V = np.asarray(V, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    if W.ndim != 2 or V.ndim != 2 or Omega.ndim != 2:
        raise ValueError("W, V, and Omega must be 2D arrays")

    if W.shape != V.shape or W.shape != Omega.shape:
        raise ValueError("W, V, and Omega must have identical shapes")

    if not (isinstance(eps, (int, float)) and float(eps) > 0.0):
        raise ValueError("eps must be positive")

    k = W.shape[1]

    Z = Omega.T @ W
    M = V.T @ W

    H = M @ np.linalg.solve(
        Z + float(eps) * np.eye(k, dtype=float), M.T
    )

    H = 0.5 * (H + H.T)

    return H.astype(float)

import numpy as np
def assemble_low_rank_correction(
    V: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    H = np.asarray(H, dtype=float)

    if V.ndim != 2:
        raise ValueError("V must be 2D")

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")

    if V.shape[1] != H.shape[0]:
        raise ValueError("V and H have incompatible dimensions")

    Delta_hat = V @ H @ V.T

    Delta_hat = 0.5 * (Delta_hat + Delta_hat.T)

    return Delta_hat.astype(float)

import numpy as np

def differentiate_correction_sketch(
    W: np.ndarray,
    Omega: np.ndarray,
    W_s: np.ndarray,
    W_t: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    eps: float,
) -> np.ndarray:
    W, Omega, W_s, W_t, P, Q = [
        np.asarray(a, dtype=float) for a in (W, Omega, W_s, W_t, P, Q)
    ]
    if any(a.ndim != 2 or a.shape != W.shape for a in (W, Omega, W_s, W_t, P, Q)):
        raise ValueError("all six arrays must be 2D with identical shapes")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    Z = Omega.T @ W
    Z_s = P.T @ W + Omega.T @ W_s
    Z_t = Q.T @ W + Omega.T @ W_t
    Z_st = P.T @ W_t + Q.T @ W_s
    B = np.linalg.solve(Z + eps * np.eye(W.shape[1]), np.eye(W.shape[1]))
    B_s = -B @ Z_s @ B
    B_t = -B @ Z_t @ B
    B_st = B @ Z_s @ B @ Z_t @ B + B @ Z_t @ B @ Z_s @ B - B @ Z_st @ B
    D_s = W_s @ B @ W.T + W @ B_s @ W.T + W @ B @ W_s.T
    D_t = W_t @ B @ W.T + W @ B_t @ W.T + W @ B @ W_t.T
    D_st = (
        W_s @ B_t @ W.T + W_s @ B @ W_t.T
        + W_t @ B_s @ W.T + W @ B_st @ W.T + W @ B_s @ W_t.T
        + W_t @ B @ W_s.T + W @ B_t @ W_s.T
    )
    derivatives = np.stack((D_s, D_t, D_st))
    return 0.5 * (derivatives + derivatives.transpose(0, 2, 1))

import numpy as np

def mixed_generalized_projector_derivative(
    D: np.ndarray,
    T: np.ndarray,
    D_s: np.ndarray,
    T_s: np.ndarray,
    D_t: np.ndarray,
    T_t: np.ndarray,
    D_st: np.ndarray,
    T_st: np.ndarray,
    r: int,
) -> np.ndarray:
    arrays = [np.asarray(x, dtype=float) for x in (D, T, D_s, T_s, D_t, T_t, D_st, T_st)]
    D = arrays[0]
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("matrix inputs must be nonempty and square")
    n = D.shape[0]
    for x in arrays:
        if x.shape != D.shape or not np.all(np.isfinite(x)) or not np.allclose(x, x.T, atol=1e-12, rtol=0):
            raise ValueError("matrix inputs must have the same shape and be finite and symmetric")
    if not isinstance(r, (int, np.integer)) or not 1 <= r < n:
        raise ValueError("r must be an integer satisfying 1 <= r < n")
    D, T, D_s, T_s, D_t, T_t, D_st, T_st = [0.5 * (x + x.T) for x in arrays]
    try:
        L = np.linalg.cholesky(T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("T must be SPD") from exc
    S = np.linalg.solve(L, D)
    S = np.linalg.solve(L, S.T).T
    eigenvalues, U = np.linalg.eigh(0.5 * (S + S.T))
    V = np.linalg.solve(L.T, U)
    V_inv = U.T @ L.T
    selected = np.zeros(n)
    selected[-r:] = 1.0

    A = np.linalg.solve(T, D)
    A_s = np.linalg.solve(T, D_s - T_s @ A)
    A_t = np.linalg.solve(T, D_t - T_t @ A)
    A_st = np.linalg.solve(T, D_st - T_st @ A - T_s @ A_t - T_t @ A_s)
    E_s, E_t, E_st = [V_inv @ x @ V for x in (A_s, A_t, A_st)]
    cross = selected[:, None] != selected[None, :]
    gaps = eigenvalues[:, None] - eigenvalues[None, :]
    factors = np.zeros((n, n))
    factors[cross] = (selected[:, None] - selected[None, :])[cross] / gaps[cross]
    P_s = factors * E_s
    P_t = factors * E_t
    commutator = (
        E_s @ P_t - P_t @ E_s + E_t @ P_s - P_s @ E_t
        + E_st * selected[None, :] - selected[:, None] * E_st
    )
    P_st = np.zeros((n, n))
    P_st[cross] = -commutator[cross] / gaps[cross]
    products = P_s @ P_t + P_t @ P_s
    inside = (selected[:, None] == 1) & (selected[None, :] == 1)
    outside = (selected[:, None] == 0) & (selected[None, :] == 0)
    P_st[inside] = -products[inside]
    P_st[outside] = products[outside]
    return V @ P_st @ V_inv

import numpy as np

def full_pipeline_mixed_sketch_sensitivity(
    seed: int,
    p: int,
    m: int,
    k: int,
    eps: float,
    r: int,
) -> float:

    if not (isinstance(seed, (int, np.integer)) and seed >= 0):
        raise ValueError("seed must be a non-negative integer")

    if not (isinstance(p, (int, np.integer)) and p >= 1):
        raise ValueError("p must be positive")

    if not (isinstance(m, (int, np.integer)) and m >= p):
        raise ValueError("m must be >= p")

    if not (isinstance(k, (int, np.integer)) and 1 <= k <= m):
        raise ValueError("k must satisfy 1 <= k <= m")

    if not (isinstance(eps, (int, float)) and np.isfinite(eps) and float(eps) > 0.0):
        raise ValueError("eps must be positive")

    if not isinstance(r, (int, np.integer)) or not 1 <= r < m:
        raise ValueError("r must be an integer satisfying 1 <= r < m")

    # Step 01: generate the deterministic inputs.
    E, C, Omega = generate_deterministic_inputs(
        seed,
        p,
        m,
        k,
    )

    # Step 02: consume Step 01 output.
    Y, Y_D = solve_diagonal_and_exact_actions(
        E,
        C,
        Omega,
    )

    # Step 03: consume Step 02 output.
    W = form_sample_matrix(
        C,
        Y,
        Y_D,
    )

    # Step 04: consume Step 03 output.
    V = thin_qr_orthonormal_basis(
        W,
    )

    # Step 05: consume Steps 03, 04, and the original sketch.
    H = form_regularized_core(
        W,
        V,
        Omega,
        eps,
    )

    # Step 06: consume Step 04 and Step 05 outputs.
    Delta_hat = assemble_low_rank_correction(
        V,
        H,
    )

    ij = np.arange(1, m + 1)[:, None] * np.arange(1, k + 1)[None, :]
    P = np.sin(ij) / np.sqrt(float(m))
    Q = np.cos(ij) / np.sqrt(float(m))
    Y_s, Y_Ds = solve_diagonal_and_exact_actions(E, C, P)
    Y_t, Y_Dt = solve_diagonal_and_exact_actions(E, C, Q)
    W_s = form_sample_matrix(C, Y_s, Y_Ds)
    W_t = form_sample_matrix(C, Y_t, Y_Dt)
    D_s, D_t, D_st = differentiate_correction_sketch(
        W, Omega, W_s, W_t, P, Q, eps
    )
    T = np.eye(m) + W @ W.T
    T_s = W_s @ W.T + W @ W_s.T
    T_t = W_t @ W.T + W @ W_t.T
    T_st = W_s @ W_t.T + W_t @ W_s.T
    projector_st = mixed_generalized_projector_derivative(
        Delta_hat, T, D_s, T_s, D_t, T_t, D_st, T_st, r
    )
    return float(np.linalg.norm(projector_st, ord="fro"))
SCICODE_GOLD_EOF
