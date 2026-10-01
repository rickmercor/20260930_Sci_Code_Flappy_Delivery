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

def skew_symmetry_residual(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    if n < 2:
        raise ValueError("n must be at least 2")
    if nu <= 0:
        raise ValueError("nu must be positive")
    if c < 0:
        raise ValueError("c must be nonnegative")
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    struct = float(
        np.linalg.norm(H - H.T, ord="fro") + np.linalg.norm(S + S.T, ord="fro")
    )
    if struct > 1e-10:
        raise ValueError("dissipative split fails symmetry structure checks")
    
    return float(np.linalg.norm(S, ord="fro"))

import numpy as np


def h_skew_symmetry_residual(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    # Structural validation via prior step (raises if invalid).
    s_mag = skew_symmetry_residual(n, nu, b_adv, c)
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    # Cross-step check: the skew block must match the step-01 fingerprint.
    if abs(float(np.linalg.norm(S, ord="fro")) - s_mag) > 1e-9 * (1.0 + abs(s_mag)):
        raise ValueError("skew block disagrees with the preceding step")
    L = np.linalg.cholesky(H)
    Y = np.linalg.solve(L, S)
    K = np.linalg.solve(L, Y.T).T
    skew_res = float(np.linalg.norm(K + K.T, ord="fro"))
    if skew_res > 1e-8:
        raise ValueError("H-congruence is not skew within tolerance")
    # Nonzero fingerprint: stubs returning 0.0 fail whenever b_adv != 0.
    return float(np.linalg.norm(K, ord="fro"))

import numpy as np

def spectral_width_lambda(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    k_mag = h_skew_symmetry_residual(n, nu, b_adv, c)
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    # Generalized pencil S v = mu H v  <=>  eigenvalues of H^{-1}S.
    eigvals = np.linalg.eigvals(np.linalg.solve(H, S))
    imag_parts = []
    for mu in eigvals:
        if abs(np.real(mu)) <= 1e-8 * (1.0 + abs(mu)):
            imag_parts.append(abs(np.imag(mu)))
    if not imag_parts:
        return 0.0
    lam = float(max(imag_parts))
    # Cross-step check: H^{-1}S and K are similar, so their spectra coincide and
    # the spectral width cannot exceed the step-02 congruence magnitude.
    if lam > k_mag + 1e-8 * (1.0 + abs(k_mag)):
        raise ValueError("spectral width exceeds the congruence magnitude")
    return lam

import numpy as np

def rapoport_residual_reduction_factor(
    n: int, nu: float, b_adv: float, c: float
) -> float:
    lam = spectral_width_lambda(n, nu, b_adv, c)
    if lam < 0:
        raise ValueError("spectral width must be nonnegative")
    root = np.sqrt(1.0 + lam**2)
    rho_r = float(lam / (root + 1.0))
    rho_w = float((root - 1.0) / (root + 1.0))
    if lam > 1e-14 and abs(rho_r - rho_w) < 1e-14:
        raise ValueError("Rapoport and Widlund factors collapsed unexpectedly")
    if lam > 1e-14 and rho_r <= rho_w + 1e-15:
        raise ValueError("Rapoport factor must exceed Widlund factor for lambda>0")
    return rho_r

import numpy as np

def widlund_even_iterate_bound(
    n: int, nu: float, b_adv: float, c: float, k: int
) -> float:
    if k < 2 or k % 2 != 0:
        raise ValueError("k must be a positive even integer for Eq. (2.3)")
    lam = spectral_width_lambda(n, nu, b_adv, c)
    rho_r = rapoport_residual_reduction_factor(n, nu, b_adv, c)
    root = np.sqrt(1.0 + lam**2)
    rho_w = (root - 1.0) / (root + 1.0)
    if lam > 1e-14 and abs(rho_w - rho_r) < 1e-14:
        raise ValueError("Widlund and Rapoport factors collapsed unexpectedly")
    if lam > 1e-14 and rho_w >= rho_r - 1e-15:
        raise ValueError("Widlund factor must be strictly below Rapoport factor for lambda>0")
    m = k // 2
    return float(2.0 * (rho_w**m))

import numpy as np
def widlund_relative_h_error(
    n: int,
    nu: float,
    b_adv: float,
    c: float,
    b: np.ndarray,
    k: int,
) -> float:
    if n < 2:
        raise ValueError("n must be at least 2")
    if nu <= 0:
        raise ValueError("nu must be positive")
    if c < 0:
        raise ValueError("c must be nonnegative")
    if k < 1:
        raise ValueError("k must be at least 1")
    bound = None
    if k % 2 == 0:
        bound = widlund_even_iterate_bound(n, nu, b_adv, c, k)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("invalid even-iterate bound from the preceding step")
    b = np.asarray(b, dtype=float).reshape(n)
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    A = H + S
    b_hat = np.linalg.solve(H, b)
    r0h = float(np.sqrt(max(b_hat @ H @ b_hat, 0.0)))
    if r0h < 1e-30:
        return 0.0

    # H-MGS Krylov basis for M = H^{-1}S applied matrix-free via H-solves.
    V = np.zeros((n, k))
    V[:, 0] = b_hat / r0h
    dim = 1
    for j in range(1, k):
        w = np.linalg.solve(H, S @ V[:, j - 1])
        for i in range(j):
            w = w - float(w @ H @ V[:, i]) * V[:, i]
        beta = float(np.sqrt(max(w @ H @ w, 0.0)))
        if beta < 1e-14:
            break
        V[:, j] = w / beta
        dim = j + 1
    V = V[:, :dim]
    MV = np.column_stack([np.linalg.solve(H, S @ V[:, j]) for j in range(dim)])
    T = V.T @ (H @ MV)
    # Force skew + zero diagonal (H-skew Galerkin projection).
    T = 0.5 * (T - T.T)
    np.fill_diagonal(T, 0.0)
    rhs = np.zeros(dim)
    rhs[0] = r0h
    y = np.linalg.solve(np.eye(dim) + T, rhs)
    x_k = V @ y
    x_exact = np.linalg.solve(A, b)
    denom = float(np.sqrt(x_exact @ H @ x_exact))
    if denom < 1e-30:
        return 0.0
    diff = x_k - x_exact
    err = float(np.sqrt(max(diff @ H @ diff, 0.0)) / denom)
    if not np.isfinite(err):
        raise ValueError("non-finite Widlund relative error")
    # Cross-step check: a vanishing even-iterate bound forces a vanishing error,
    # since both are driven by the same spectral width.
    if bound is not None and bound == 0.0 and err > 1e-12:
        raise ValueError("zero even-iterate bound with nonzero realized error")
    return err

import numpy as np
def condensed_ocp_hessian_entry(
    n: int, nu: float, b_adv: float, c: float, mu: float
) -> float:
    if mu <= 0:
        raise ValueError("mu must be positive")
    s_mag = skew_symmetry_residual(n, nu, b_adv, c)
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    A = H + S
    # Cross-step check: the split used here must carry the step-01 skew block.
    if abs(float(np.linalg.norm(A - H, ord="fro")) - s_mag) > 1e-9 * (1.0 + abs(s_mag)):
        raise ValueError("split operator disagrees with the preceding step")
    e0 = np.zeros(n)
    e0[0] = 1.0
    # Two-solve form e0^T A^{-T} A^{-1} e0 + mu  (paper Sec. 4.2 with B=C=I).
    z = np.linalg.solve(A, e0)
    w = np.linalg.solve(A.T, z)
    return float(e0 @ w + mu)

import numpy as np

def orchestrate_dissipative_preconditioning_pipeline(
    n: int,
    nu: float,
    b_adv: float,
    c: float,
    b: np.ndarray,
    k: int,
    mu: float,
) -> float:
    b = np.asarray(b, dtype=float).reshape(n)
    if k < 2 or k % 2 != 0:
        raise ValueError("orchestrator requires even k for Eq. (2.3)")

    s_mag = skew_symmetry_residual(n, nu, b_adv, c)
    k_mag = h_skew_symmetry_residual(n, nu, b_adv, c)
    if not np.isfinite(s_mag) or s_mag < -1e-15:
        raise ValueError("invalid skew-block magnitude")
    if not np.isfinite(k_mag) or k_mag < -1e-15:
        raise ValueError("invalid H-congruence magnitude")
    if abs(b_adv) > 1e-14 and (s_mag <= 0.0 or k_mag <= 0.0):
        raise ValueError("nonzero advection must yield positive structural magnitudes")

    lam = spectral_width_lambda(n, nu, b_adv, c)
    rho_r = rapoport_residual_reduction_factor(n, nu, b_adv, c)
    if not np.isfinite(lam) or not np.isfinite(rho_r):
        raise ValueError("non-finite spectral diagnostics")
    if rho_r < 0.0 or rho_r > 1.0 + 1e-12:
        raise ValueError("Rapoport factor out of range")

    bound = widlund_even_iterate_bound(n, nu, b_adv, c, k)
    err = widlund_relative_h_error(n, nu, b_adv, c, b, k)
    k00 = condensed_ocp_hessian_entry(n, nu, b_adv, c, mu)

    if bound == 0.0:
        if err > 1e-12:
            raise ValueError("zero Widlund bound with nonzero error")
        ratio = 0.0
    else:
        ratio = err / bound

    tau = float(ratio + k00)
    if not np.isfinite(tau):
        raise ValueError("non-finite tau")
    return tau
SCICODE_GOLD_EOF
