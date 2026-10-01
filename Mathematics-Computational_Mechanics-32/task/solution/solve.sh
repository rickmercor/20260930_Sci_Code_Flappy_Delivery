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


def _gauss_jacobi(n, a, b):
    """Golub-Welsch nodes/weights for the weight (1-x)^a (1+x)^b on [-1, 1], numpy only."""
    import math
    import numpy as np
    n = int(n)
    if n < 1 or a <= -1.0 or b <= -1.0:
        raise ValueError("need n >= 1 and a, b > -1")
    k = np.arange(n, dtype=np.float64)
    ab = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = (b * b - a * a) / ((2.0 * k + ab) * (2.0 * k + ab + 2.0))
    diag[0] = (b - a) / (ab + 2.0)
    kk = np.arange(1, n, dtype=np.float64)
    num = 4.0 * kk * (kk + a) * (kk + b) * (kk + ab)
    den = (2.0 * kk + ab) ** 2 * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0)
    off2 = num / den
    if n > 1 and abs(ab) < 1e-300:            # k = 1 with a + b = 0 is the 0/0 case
        off2[0] = 4.0 * (1.0 + a) * (1.0 + b) / 12.0
    J = np.diag(diag)
    if n > 1:
        off = np.sqrt(off2)
        J += np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    mu0 = 2.0 ** (ab + 1.0) * math.gamma(a + 1.0) * math.gamma(b + 1.0) / math.gamma(ab + 2.0)
    w = mu0 * V[0, :] ** 2
    return x, w


def nonlocal_symbol(xi, alpha, delta):
    """Eigenvalue of L_delta on the plane wave e^{i xi x} for the kernel (5.1).

    lambda_d(xi) = 2 int_{-d}^{d} gamma_d(s) (1 - cos(xi s)) ds
                 = 4 c int_0^d s^(2-a) [2 sin^2(xi s/2) / s^2] ds,   c = (3-a)/(2 d^(3-a)),
    computed with a 200-point Gauss-Jacobi rule for the weight s^(2-a) (exact to round-off
    for the smooth bracket), which handles 0 < alpha < 3 uniformly.  lambda_d(xi) -> xi^2 as
    delta -> 0 because int_{-d}^{d} s^2 gamma_d = 1.
    """
    import numpy as np
    xi = float(xi)
    alpha = float(alpha)
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not (0.0 < alpha < 3.0):
        raise ValueError("alpha must lie in (0, 3)")
    beta = 2.0 - alpha
    c = (3.0 - alpha) / (2.0 * delta ** (3.0 - alpha))
    x, w = _gauss_jacobi(200, 0.0, beta)
    s = 0.5 * delta * (x + 1.0)
    ws = c * w * (0.5 * delta) ** (1.0 + beta)          # includes c s^(2-a) ds
    bracket = 2.0 * np.sin(0.5 * xi * s) ** 2 / s ** 2   # (1 - cos(xi s)) / s^2, smooth
    return float(4.0 * np.sum(ws * bracket))

import numpy as np


def _legendre_values(k, xi):
    """P_0..P_k at the points xi (shape (len(xi), k+1)), three-term recurrence."""
    import numpy as np
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    out = np.zeros((xi.size, k + 1))
    out[:, 0] = 1.0
    if k >= 1:
        out[:, 1] = xi
    for m in range(1, k):
        out[:, m + 1] = ((2 * m + 1) * xi * out[:, m] - m * out[:, m - 1]) / (m + 1)
    return out


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def l2_projection(fun, N, k):
    """Modal Legendre coefficients of the L2 projection P_h fun onto V_h^k, Eq (3.1).

    Cells I_j = ((j-1)h, jh), h = 1/N, local coordinate xi = 2(x - x_j)/h in [-1,1], basis
    P_0..P_k (Legendre, NOT normalised).  Coefficient m of cell j is
    (2m+1)/2 * int_{-1}^{1} fun P_m dxi, evaluated with a (k+8)-point Gauss-Legendre rule.
    Returns a flat array ordered cell-major: entry j*(k+1)+m.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k)
    h = 1.0 / N
    xg, wg = np.polynomial.legendre.leggauss(k + 8)
    P = _legendre_values(k, xg)
    scale = (2.0 * np.arange(k + 1) + 1.0) / 2.0
    coef = np.zeros(N * (k + 1))
    for j in range(N):
        x = (j + 0.5) * h + 0.5 * h * xg
        f = np.asarray(fun(x), dtype=np.float64)
        if f.shape != x.shape:
            raise ValueError("fun must map an array of points to an array of the same shape")
        coef[j * (k + 1):(j + 1) * (k + 1)] = scale * (P.T @ (wg * f))
    return coef

import numpy as np


def _legendre_values(k, xi):
    """P_0..P_k at the points xi (shape (len(xi), k+1)), three-term recurrence."""
    import numpy as np
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    out = np.zeros((xi.size, k + 1))
    out[:, 0] = 1.0
    if k >= 1:
        out[:, 1] = xi
    for m in range(1, k):
        out[:, m + 1] = ((2 * m + 1) * xi * out[:, m] - m * out[:, m - 1]) / (m + 1)
    return out


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def shift_projection_matrix(N, k, s):
    """Matrix of S(s): coefficients of P_h[u_h(. + s)] on the periodic mesh, s > 0.

    This is the exact building block of H_j in (2.5): for x in I_j, x + s lies in cell
    j+m on [x_{j-1/2}, x_{j+1/2} - r] and in cell j+m+1 on the rest, where s = m h + r,
    0 <= r < h.  Each piece is a polynomial, so the two integrals are evaluated EXACTLY with
    a (k+2)-point Gauss-Legendre rule on the sub-interval (never a single rule across the
    crossing point).  With rho = 2r/h, block A multiplies the coefficients of cell j+m and
    block B those of cell j+m+1 (indices mod N):
        A = Mhat^-1 int_{-1}^{1-rho} phi(xi) phi(xi+rho)^T dxi,
        B = Mhat^-1 int_{1-rho}^{1}  phi(xi) phi(xi+rho-2)^T dxi.
    S(h) is the pure cell permutation; S(s) preserves constants.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); s = float(s)
    if s <= 0.0:
        raise ValueError("s must be positive")
    h = 1.0 / N
    m = int(np.floor(s / h + 1e-12))
    r = s - m * h
    if r < 0.0:
        r = 0.0
    rho = 2.0 * r / h
    xg, wg = np.polynomial.legendre.leggauss(k + 2)
    Minv = np.diag((2.0 * np.arange(k + 1) + 1.0) / 2.0)

    def block(lo, hi, shift):
        if hi - lo <= 0.0:
            return np.zeros((k + 1, k + 1))
        xi = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        PA = _legendre_values(k, xi)
        PB = _legendre_values(k, xi + shift)
        return Minv @ (PA.T @ (w[:, None] * PB))

    A = block(-1.0, 1.0 - rho, rho)
    B = block(1.0 - rho, 1.0, rho - 2.0)
    n = k + 1
    S = np.zeros((N * n, N * n))
    for j in range(N):
        ja = (j + m) % N
        jb = (j + m + 1) % N
        S[j * n:(j + 1) * n, ja * n:(ja + 1) * n] += A
        S[j * n:(j + 1) * n, jb * n:(jb + 1) * n] += B
    return S

import numpy as np


def _gauss_jacobi(n, a, b):
    """Golub-Welsch nodes/weights for the weight (1-x)^a (1+x)^b on [-1, 1], numpy only."""
    import math
    import numpy as np
    n = int(n)
    if n < 1 or a <= -1.0 or b <= -1.0:
        raise ValueError("need n >= 1 and a, b > -1")
    k = np.arange(n, dtype=np.float64)
    ab = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = (b * b - a * a) / ((2.0 * k + ab) * (2.0 * k + ab + 2.0))
    diag[0] = (b - a) / (ab + 2.0)
    kk = np.arange(1, n, dtype=np.float64)
    num = 4.0 * kk * (kk + a) * (kk + b) * (kk + ab)
    den = (2.0 * kk + ab) ** 2 * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0)
    off2 = num / den
    if n > 1 and abs(ab) < 1e-300:            # k = 1 with a + b = 0 is the 0/0 case
        off2[0] = 4.0 * (1.0 + a) * (1.0 + b) / 12.0
    J = np.diag(diag)
    if n > 1:
        off = np.sqrt(off2)
        J += np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    mu0 = 2.0 ** (ab + 1.0) * math.gamma(a + 1.0) * math.gamma(b + 1.0) / math.gamma(ab + 2.0)
    w = mu0 * V[0, :] ** 2
    return x, w


def _mass_diag(N, k):
    """Diagonal of the mass matrix of the Legendre modal basis on N uniform cells of (0,1)."""
    import numpy as np
    h = 1.0 / N
    return np.tile(h / (2.0 * np.arange(k + 1) + 1.0), N)


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def nonlocal_dg_operator(N, k, alpha, delta, shift_matrix):
    """Dense matrix B_h of the semi-discrete scheme (2.6): d^2 u_h / dt^2 = B_h u_h.

    shift_matrix(N, k, s) is the previous step (the orchestrator passes its oracle).

    From (2.4)-(2.6) with q_h(.;s) = H(s) u_h, H(s) = (S(s) - I)/s (L2 projection of the
    forward quotient), and K_j acting on q_h through the backward quotient, Lemma 2.3 gives
    K(q_h, v; s) = -H(v, q_h; s) = -(H(s) v, q_h)_M, hence
        B_h = -2 int_0^delta s^2 gamma_d(s) M^-1 H(s)^T M H(s) ds,
    M-symmetric and negative semidefinite with (B_h u, u)_M = -2 int s^2 gamma ||H(s)u||_M^2.
    The s-integrand is piecewise smooth with breakpoints at multiples of h and behaves like
    s^(2-alpha) at 0, so it is integrated with a 24-point Gauss-Jacobi rule (weight s^(2-a))
    on [0, min(h,delta)] and 24-point Gauss-Legendre rules on every following interval of
    length h (last one truncated at delta).  Result is converged to round-off.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); alpha = float(alpha); delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not (0.0 < alpha < 3.0):
        raise ValueError("alpha must lie in (0, 3)")
    if not callable(shift_matrix):
        raise ValueError("shift_matrix must be callable")
    h = 1.0 / N
    n = k + 1
    dim = N * n
    c = (3.0 - alpha) / (2.0 * delta ** (3.0 - alpha))
    beta = 2.0 - alpha
    # quadrature in s: nodes and weights that already include s^2 gamma_d(s) ds
    b0 = min(h, delta)
    xj, wj = _gauss_jacobi(24, 0.0, beta)
    nodes = [0.5 * b0 * (xj + 1.0)]
    weights = [c * wj * (0.5 * b0) ** (1.0 + beta)]
    edges = [b0]
    while edges[-1] < delta - 1e-14:
        edges.append(min(edges[-1] + h, delta))
    xg, wg = np.polynomial.legendre.leggauss(24)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sq = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        nodes.append(sq)
        weights.append(0.5 * (hi - lo) * wg * c * sq ** beta)
    sq = np.concatenate(nodes)
    wq = np.concatenate(weights)
    Md = _mass_diag(N, k)
    I = np.eye(dim)
    B = np.zeros((dim, dim))
    for s, w in zip(sq, wq):
        H = (np.asarray(shift_matrix(N, k, s), dtype=np.float64) - I) / s
        B -= 2.0 * w * (H.T @ (Md[:, None] * H))
    return B / Md[:, None]

import numpy as np


def _mass_diag(N, k):
    """Diagonal of the mass matrix of the Legendre modal basis on N uniform cells of (0,1)."""
    import numpy as np
    h = 1.0 / N
    return np.tile(h / (2.0 * np.arange(k + 1) + 1.0), N)


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def discrete_energy(u_new, u_old, ht, N, k, B):
    """Fully discrete energy of Theorem 4.1 for the pair (u^{n+1}, u^n).

    E = ||(u^{n+1} - u^n)/h_t||^2_{L2} + int_0^d s^2 gamma (||q^{n+1}||^2 + ||q^n||^2) ds,
    with q^n = H(s) u^n, and int s^2 gamma ||H(s) u||^2 ds = -(B_h u, u)_M / 2, so
    E = ||(u^{n+1}-u^n)/h_t||_M^2 - [(B u^{n+1}, u^{n+1})_M + (B u^n, u^n)_M] / 2.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); ht = float(ht)
    if ht <= 0.0:
        raise ValueError("ht must be positive")
    u_new = np.asarray(u_new, dtype=np.float64).ravel()
    u_old = np.asarray(u_old, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    dim = N * (k + 1)
    if u_new.size != dim or u_old.size != dim or B.shape != (dim, dim):
        raise ValueError("u_new, u_old and B must match N*(k+1)")
    Md = _mass_diag(N, k)
    v = (u_new - u_old) / ht
    kin = float(np.sum(Md * v * v))
    pot = -0.5 * (float(u_new @ (Md * (B @ u_new))) + float(u_old @ (Md * (B @ u_old))))
    return kin + pot

import numpy as np


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def crank_nicolson_march(u0, ht, nsteps, N, k, B):
    """Crank-Nicolson scheme (4.1) for d^2u/dt^2 = B u with u_t(0) = 0, nsteps steps.

    (u^{n+1} - 2u^n + u^{n-1})/h_t^2 = B (u^{n+1} + u^{n-1})/2   (mean-value operator on q_h)
    <=> (I - h_t^2 B/2) u^{n+1} = 2 u^n - (I - h_t^2 B/2) u^{n-1}.
    Start-up (declared in the task, the paper is silent): the ghost value u^{-1} := u^{1}
    encodes u_t(x,0) = 0, so the n = 0 equation gives (I - h_t^2 B/2) u^1 = u^0.
    Returns a (2, dim) array: row 0 is u^{nsteps}, row 1 is u^{nsteps-1}.
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); ht = float(ht)
    if ht <= 0.0:
        raise ValueError("ht must be positive")
    if int(nsteps) != nsteps or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")
    nsteps = int(nsteps)
    u0 = np.asarray(u0, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    dim = N * (k + 1)
    if u0.size != dim or B.shape != (dim, dim):
        raise ValueError("u0 and B must match N*(k+1)")
    C = np.eye(dim) - 0.5 * ht * ht * B
    Cinv = np.linalg.inv(C)
    u_prev = u0.copy()
    u_cur = Cinv @ u0
    for _ in range(1, nsteps):
        u_next = Cinv @ (2.0 * u_cur - C @ u_prev)
        u_prev, u_cur = u_cur, u_next
    return np.vstack([u_cur, u_prev])

import numpy as np


def _legendre_values(k, xi):
    """P_0..P_k at the points xi (shape (len(xi), k+1)), three-term recurrence."""
    import numpy as np
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    out = np.zeros((xi.size, k + 1))
    out[:, 0] = 1.0
    if k >= 1:
        out[:, 1] = xi
    for m in range(1, k):
        out[:, m + 1] = ((2 * m + 1) * xi * out[:, m] - m * out[:, m - 1]) / (m + 1)
    return out


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def gauss_lobatto_l2_error(coef, fun, N, k):
    """e_u of Section 5: (k+3)-point Gauss-Lobatto rule per cell.

    e_u = ( sum_j sum_i (h/2) w_i (fun(x_i^j) - u_h(x_i^j))^2 )^(1/2), with the Lobatto
    nodes +-1 and the roots of P'_{k+2}, weights 2 / ((k+3)(k+2) P_{k+2}(x_i)^2).
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k)
    coef = np.asarray(coef, dtype=np.float64).ravel()
    n = k + 1
    if coef.size != N * n:
        raise ValueError("coef must have N*(k+1) entries")
    npts = k + 3
    Pn = np.polynomial.legendre.Legendre.basis(npts - 1)
    inner = np.real(Pn.deriv().roots())
    xi = np.sort(np.concatenate(([-1.0], inner, [1.0])))
    Pval = _legendre_values(npts - 1, xi)
    w = 2.0 / (npts * (npts - 1) * Pval[:, npts - 1] ** 2)
    P = _legendre_values(k, xi)
    h = 1.0 / N
    err2 = 0.0
    for j in range(N):
        x = (j + 0.5) * h + 0.5 * h * xi
        f = np.asarray(fun(x), dtype=np.float64)
        if f.shape != x.shape:
            raise ValueError("fun must map an array of points to an array of the same shape")
        uh = P @ coef[j * n:(j + 1) * n]
        err2 += 0.5 * h * float(np.sum(w * (f - uh) ** 2))
    return float(np.sqrt(err2))

import numpy as np


def _gauss_jacobi(n, a, b):
    """Golub-Welsch nodes/weights for the weight (1-x)^a (1+x)^b on [-1, 1], numpy only."""
    import math
    import numpy as np
    n = int(n)
    if n < 1 or a <= -1.0 or b <= -1.0:
        raise ValueError("need n >= 1 and a, b > -1")
    k = np.arange(n, dtype=np.float64)
    ab = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = (b * b - a * a) / ((2.0 * k + ab) * (2.0 * k + ab + 2.0))
    diag[0] = (b - a) / (ab + 2.0)
    kk = np.arange(1, n, dtype=np.float64)
    num = 4.0 * kk * (kk + a) * (kk + b) * (kk + ab)
    den = (2.0 * kk + ab) ** 2 * (2.0 * kk + ab + 1.0) * (2.0 * kk + ab - 1.0)
    off2 = num / den
    if n > 1 and abs(ab) < 1e-300:            # k = 1 with a + b = 0 is the 0/0 case
        off2[0] = 4.0 * (1.0 + a) * (1.0 + b) / 12.0
    J = np.diag(diag)
    if n > 1:
        off = np.sqrt(off2)
        J += np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    mu0 = 2.0 ** (ab + 1.0) * math.gamma(a + 1.0) * math.gamma(b + 1.0) / math.gamma(ab + 2.0)
    w = mu0 * V[0, :] ** 2
    return x, w


def _legendre_values(k, xi):
    """P_0..P_k at the points xi (shape (len(xi), k+1)), three-term recurrence."""
    import numpy as np
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    out = np.zeros((xi.size, k + 1))
    out[:, 0] = 1.0
    if k >= 1:
        out[:, 1] = xi
    for m in range(1, k):
        out[:, m + 1] = ((2 * m + 1) * xi * out[:, m] - m * out[:, m - 1]) / (m + 1)
    return out


def _mass_diag(N, k):
    """Diagonal of the mass matrix of the Legendre modal basis on N uniform cells of (0,1)."""
    import numpy as np
    h = 1.0 / N
    return np.tile(h / (2.0 * np.arange(k + 1) + 1.0), N)


def _check_space(N, k):
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")


def nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T):
    """Orchestrator: e_u(T) of the DG-CN solution for u0 = sin(2 pi x) + cos(4 pi x)/2.

    Exact solution: L_d acts on e^{i xi x} as lambda_d(xi) (step 1), so with u_t(x,0) = 0
    u(x,t) = cos(w1 t) sin(2 pi x) + cos(w2 t) cos(4 pi x)/2, w_m = sqrt(lambda_d(2 pi m)).
    Chain: L2 projection of u0 (step 2) -> B_h (step 4, fed the shift matrices of step 3) -> CN march (step 6)
    -> energy check with step 5 (conservation to 1e-9 relative, else ValueError)
    -> e_u with the Gauss-Lobatto rule (step 7).
    """
    import numpy as np
    _check_space(N, k)
    N = int(N); k = int(k); alpha = float(alpha); delta = float(delta)
    ht = float(ht); T = float(T)
    if ht <= 0.0 or T <= 0.0:
        raise ValueError("ht and T must be positive")
    nsteps = int(round(T / ht))
    if nsteps < 1 or abs(nsteps * ht - T) > 1e-9 * max(1.0, T):
        raise ValueError("T must be a positive integer multiple of ht")
    w1 = np.sqrt(nonlocal_symbol(2.0 * np.pi, alpha, delta))
    w2 = np.sqrt(nonlocal_symbol(4.0 * np.pi, alpha, delta))

    def u0(x):
        return np.sin(2.0 * np.pi * x) + 0.5 * np.cos(4.0 * np.pi * x)

    def uT(x):
        return (np.cos(w1 * T) * np.sin(2.0 * np.pi * x)
                + 0.5 * np.cos(w2 * T) * np.cos(4.0 * np.pi * x))

    c0 = l2_projection(u0, N, k)
    B = nonlocal_dg_operator(N, k, alpha, delta, shift_projection_matrix)
    pair = crank_nicolson_march(c0, ht, nsteps, N, k, B)
    first = crank_nicolson_march(c0, ht, 1, N, k, B)
    E1 = discrete_energy(first[0], first[1], ht, N, k, B)
    EN = discrete_energy(pair[0], pair[1], ht, N, k, B)
    if abs(EN - E1) > 1e-9 * max(1.0, abs(E1)):
        raise ValueError("discrete energy is not conserved")
    return gauss_lobatto_l2_error(pair[0], uT, N, k)
SCICODE_GOLD_EOF
