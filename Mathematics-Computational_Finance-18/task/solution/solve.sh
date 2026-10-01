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
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def regularized_kernel(alpha: float, delta_star: float, t: "np.ndarray") -> "np.ndarray":
    """Eq (19): translated fractional kernel K_{a,d*}(t) = (t + d*)^{-a} / Gamma(1-a)."""
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must satisfy 0 < alpha < 1/2")
    if delta_star <= 0.0:
        raise ValueError("delta_star must be positive")
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("t must be a non-empty 1-D array")
    if np.any(t < 0.0):
        raise ValueError("t must be nonnegative")
    # TRANSLATION, not clipping: the argument is shifted, which keeps the kernel
    # completely monotone with a nonnegative representing measure while making the
    # zero-lag response finite. Clipping and truncation also make it finite and are
    # the natural guesses, but neither retains that representation.
    return (t + delta_star) ** (-alpha) / gamma(1.0 - alpha)

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def bernstein_density(alpha: float, delta_star: float, x: "np.ndarray") -> "np.ndarray":
    """Density of the representing measure of the TRANSLATED kernel.

    K_{a,d*}(t) = int_0^inf exp(-x t) w(x) dx  with
        w(x) = exp(-x d*) x^{a-1} / (Gamma(a) Gamma(1-a)).
    The exp(-x d*) factor is exactly what the translation contributes.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must satisfy 0 < alpha < 1/2")
    if delta_star <= 0.0:
        raise ValueError("delta_star must be positive")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("x must be a non-empty 1-D array")
    if np.any(x <= 0.0):
        raise ValueError("x must be strictly positive")
    return np.exp(-x * delta_star) * x ** (alpha - 1.0) / (gamma(alpha) * gamma(1.0 - alpha))

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def block_quadrature_nodes(alpha: float, m: int, n_quad: int) -> tuple:
    """Eq (25): nodes and DENSITY-RELATIVE weights for the prescribed 3-block split.

    Blocks are [0, 2^-m], [2^-m, 1] and [1, inf), carrying Gauss-Jacobi, Gauss-Legendre
    and Gauss-Laguerre respectively, n_quad points each.

    Weights q are returned relative to the representing density: the caller forms
    omega = q * w(x). On the first block the Jacobi rule integrates the x^{a-1} endpoint
    singularity exactly, so its raw weight is divided back out by x^{1-a} to leave a
    density-relative weight.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must satisfy 0 < alpha < 1/2")
    if int(m) != m or m < 0:
        raise ValueError("m must be a nonnegative integer")
    if int(n_quad) != n_quad or n_quad < 1:
        raise ValueError("n_quad must be a positive integer")
    m, n_quad = int(m), int(n_quad)
    c = 2.0 ** (-m)

    # [0, c]: x = c(1+u)/2 turns int_0^c f(x) x^{a-1} dx into
    # (c/2)^a int_{-1}^1 f(.) (1+u)^{a-1} du, which is Gauss-Jacobi with beta = a-1.
    u, wu = roots_jacobi(n_quad, 0.0, alpha - 1.0)
    x1 = c * (1.0 + u) / 2.0
    q1 = wu * (c / 2.0) ** alpha * x1 ** (1.0 - alpha)

    # [c, 1]: smooth -> Gauss-Legendre
    v, wv = roots_legendre(n_quad)
    x2 = c + (1.0 - c) * (v + 1.0) / 2.0
    q2 = wv * (1.0 - c) / 2.0

    # [1, inf): the density decays exponentially -> Gauss-Laguerre on x = 1 + y
    y, wy = roots_laguerre(n_quad)
    x3 = 1.0 + y
    q3 = wy * np.exp(y)

    return np.concatenate([x1, x2, x3]), np.concatenate([q1, q2, q3])

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def lift_fidelity_error_bp(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int) -> float:
    """Lift-fidelity error in basis points: 1e4 * (omega^T G omega / int_0^dt K^2 - 1).

    Calls step 3 (nodes and density-relative weights), step 2 (density at the nodes,
    omega = q * w(x)) and step 1 (kernel on the time grid). The one-step Volterra
    variance is the composite Simpson integral of the squared kernel on the uniform
    grid; the factor increments over the step share one Brownian path, so their
    covariance is G_lm = (1 - exp(-(x_l + x_m) dt)) / (x_l + x_m), and the lift's
    one-step variance is the quadratic form omega^T G omega.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if int(n_grid) != n_grid or n_grid < 3 or int(n_grid) % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 3")
    n_grid = int(n_grid)

    x, q = block_quadrature_nodes(alpha, m, n_quad)
    omega = q * bernstein_density(alpha, delta_star, x)

    t = np.linspace(0.0, float(dt), n_grid)
    kernel = regularized_kernel(alpha, delta_star, t)
    f = kernel * kernel
    h = float(dt) / (n_grid - 1)
    exact = h / 3.0 * (f[0] + f[-1] + 4.0 * np.sum(f[1:-1:2]) + 2.0 * np.sum(f[2:-1:2]))

    s = x[:, None] + x[None, :]
    cov = (1.0 - np.exp(-s * dt)) / s
    lift = float(omega @ cov @ omega)

    return float(1.0e4 * (lift / exact - 1.0))

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def lifted_moment_blocks(x: "np.ndarray", omega: "np.ndarray", kappa: float, theta: float, v0: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray") -> tuple:
    """Affine drift and jump blocks of the lifted state z = (U_1..U_N, lam_S, lam_V).

    Continuous drift a(z) = A z + c: each factor decays at its own rate, is pulled by
    kappa (theta - V) with V = v0 + sum_k omega_k U_k, and each intensity reverts at its
    decay rate to its baseline. Jumps: a price jump raises the intensities by the first
    column of eta; a variance jump adds its mark J to every factor (the exponential sum
    reproduces the kernel) and raises the intensities by the second column of eta.
    With J exponential of mean mu_v, E[J] = mu_v and E[J^2] = 2 mu_v^2.
    Returns (A, c, j_s, j_v, Q_s, Q_v): mean jump vectors j_k = E[jump of z] for type k
    and second-moment products Q_k = E[jump_i jump_j] for type k.
    """
    x = np.asarray(x, dtype=np.float64); omega = np.asarray(omega, dtype=np.float64)
    lam_inf = np.asarray(lam_inf, dtype=np.float64); eta = np.asarray(eta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if x.ndim != 1 or x.size < 1 or omega.shape != x.shape:
        raise ValueError("x and omega must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(omega))) or np.any(x <= 0.0):
        raise ValueError("rates must be finite and strictly positive")
    for name, val in (("kappa", kappa), ("theta", theta), ("v0", v0), ("mu_v", mu_v)):
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(name + " must be finite and nonnegative")
    if lam_inf.shape != (2,) or eta.shape != (2, 2) or beta.shape != (2,):
        raise ValueError("lam_inf and beta must have shape (2,), eta shape (2, 2)")
    if not (np.all(np.isfinite(lam_inf)) and np.all(np.isfinite(eta)) and np.all(np.isfinite(beta))):
        raise ValueError("Hawkes inputs must be finite")
    if np.any(lam_inf < 0.0) or np.any(eta < 0.0) or np.any(beta <= 0.0):
        raise ValueError("baselines and excitations must be nonnegative, decay rates positive")
    N = x.size; n = N + 2
    A = np.zeros((n, n)); c = np.zeros(n)
    A[:N, :N] = -np.diag(x) - kappa * np.tile(omega, (N, 1))
    c[:N] = kappa * (theta - v0)
    A[N, N] = -beta[0]; A[N + 1, N + 1] = -beta[1]
    c[N] = beta[0] * lam_inf[0]; c[N + 1] = beta[1] * lam_inf[1]
    j_s = np.zeros(n); j_s[N] = eta[0, 0]; j_s[N + 1] = eta[1, 0]
    j_v = np.zeros(n); j_v[:N] = mu_v; j_v[N] = eta[0, 1]; j_v[N + 1] = eta[1, 1]
    Q_s = np.outer(j_s, j_s)
    Q_v = np.outer(j_v, j_v); Q_v[:N, :N] = 2.0 * mu_v * mu_v
    return A, c, j_s, j_v, Q_s, Q_v

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def lifted_moment_generator(A: "np.ndarray", c: "np.ndarray", xi: float, omega: "np.ndarray", v0: float, j_s: "np.ndarray", j_v: "np.ndarray", Q_s: "np.ndarray", Q_v: "np.ndarray") -> "np.ndarray":
    """Generator of the closed linear system for the first and second moments of z.

    Coordinates of the moment vector y: the second moments S_ij = E[z_i z_j] for i <= j in
    row-major order (n(n+1)/2 entries), then the first moments m_i = E[z_i] (n entries),
    then the constant 1. Returns G with dy/dt = G y.
    For each pair (i, j): dS_ij/dt = E[z_i a_j + z_j a_i] + E[sigma_i sigma_j]
    + sum over jump types k of E[lam_k (z_i dz_j + z_j dz_i + dz_i dz_j)], where a = A z + c,
    sigma_i sigma_j = xi^2 (v0 + omega . U) for factor pairs and 0 otherwise (every factor is
    driven by the same Brownian increment), lam_S = z_N, lam_V = z_{N+1}, and the jump
    expectations use j_k and Q_k. First moments: dm/dt = A m + c + m_N j_s + m_{N+1} j_v.
    """
    A = np.asarray(A, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    j_s = np.asarray(j_s, dtype=np.float64); j_v = np.asarray(j_v, dtype=np.float64)
    Q_s = np.asarray(Q_s, dtype=np.float64); Q_v = np.asarray(Q_v, dtype=np.float64)
    n = A.shape[0] if A.ndim == 2 else -1
    N = omega.size
    if A.shape != (n, n) or n != N + 2 or c.shape != (n,) or j_s.shape != (n,) or j_v.shape != (n,) or Q_s.shape != (n, n) or Q_v.shape != (n, n):
        raise ValueError("block shapes must be consistent with n = len(omega) + 2")
    if not all(np.all(np.isfinite(v)) for v in (A, c, omega, j_s, j_v, Q_s, Q_v)):
        raise ValueError("blocks must be finite")
    if not np.isfinite(xi) or xi < 0.0 or not np.isfinite(v0) or v0 < 0.0:
        raise ValueError("xi and v0 must be finite and nonnegative")
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i, n):
            idx[(i, j)] = k
            k += 1
    nS = k
    dim = nS + n + 1

    def I(i, j):
        return idx[(i, j)] if i <= j else idx[(j, i)]

    Gm = np.zeros((dim, dim))
    lam_S, lam_V = N, N + 1
    for i in range(n):
        Gm[nS + i, nS:nS + n] += A[i]
        Gm[nS + i, dim - 1] += c[i]
        Gm[nS + i, nS + lam_S] += j_s[i]
        Gm[nS + i, nS + lam_V] += j_v[i]
    for i in range(n):
        for j in range(i, n):
            r = I(i, j)
            for kk in np.nonzero(A[j])[0]:
                Gm[r, I(i, kk)] += A[j, kk]
            for kk in np.nonzero(A[i])[0]:
                Gm[r, I(j, kk)] += A[i, kk]
            Gm[r, nS + i] += c[j]
            Gm[r, nS + j] += c[i]
            if i < N and j < N:
                Gm[r, dim - 1] += xi * xi * v0
                Gm[r, nS:nS + N] += xi * xi * omega
            for lam_idx, jk, Qk in ((lam_S, j_s, Q_s), (lam_V, j_v, Q_v)):
                if jk[j] != 0.0:
                    Gm[r, I(i, lam_idx)] += jk[j]
                if jk[i] != 0.0:
                    Gm[r, I(j, lam_idx)] += jk[i]
                if Qk[i, j] != 0.0:
                    Gm[r, nS + lam_idx] += Qk[i, j]
    return Gm

import numpy as np
from scipy.linalg import expm
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def variance_intensity_correlation(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int, kappa: float, theta: float, v0: float, xi: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray", horizon: float) -> float:
    """ORCHESTRATOR. Correlation between the variance and the variance-jump intensity at
    the horizon under the lifted Volterra-Hawkes dynamics, from the exact moments.

    Calls every earlier step: 3 and 2 for the exponential sum, 4 for the lift-fidelity
    admissibility check, 5 for the drift and jump blocks, 6 for the moment generator, and 1
    for the kernel's zero-lag value, which the exponential-sum weights must approximate to
    within one percent (a second admissibility check). Initial state: all factors at zero and
    both intensities at their baselines, so the initial second moments are the outer product
    of the initial means.
    """
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be a positive finite time")
    x, q = block_quadrature_nodes(alpha, m, n_quad)
    omega = q * bernstein_density(alpha, delta_star, x)
    fidelity = lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)
    if not np.isfinite(fidelity) or abs(fidelity) > 100.0:
        raise ValueError("the exponential-sum lift is not admissible at this configuration")
    k0 = float(regularized_kernel(alpha, delta_star, np.array([0.0]))[0])
    if abs(float(omega.sum()) / k0 - 1.0) > 0.01:
        raise ValueError("the exponential-sum weights do not reproduce the zero-lag kernel")
    A, c, j_s, j_v, Q_s, Q_v = lifted_moment_blocks(x, omega, kappa, theta, v0, mu_v, lam_inf, eta, beta)
    Gm = lifted_moment_generator(A, c, xi, omega, v0, j_s, j_v, Q_s, Q_v)
    N = x.size
    n = N + 2
    nS = n * (n + 1) // 2
    m0 = np.concatenate([np.zeros(N), np.asarray(lam_inf, dtype=np.float64)])
    y0 = np.zeros(nS + n + 1)
    k = 0
    for i in range(n):
        for j in range(i, n):
            y0[k] = m0[i] * m0[j]
            k += 1
    y0[nS:nS + n] = m0
    y0[-1] = 1.0
    yT = expm(Gm * float(horizon)) @ y0
    mT = yT[nS:nS + n]
    ST = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            ST[i, j] = ST[j, i] = yT[k]
            k += 1
    cov = ST - np.outer(mT, mT)
    wv = np.concatenate([omega, [0.0, 0.0]])
    var_v = float(wv @ cov @ wv)
    var_l = float(cov[N + 1, N + 1])
    cov_vl = float(wv @ cov[:, N + 1])
    if not (var_v > 0.0 and var_l > 0.0):
        raise ValueError("degenerate variance at the horizon")
    return cov_vl / np.sqrt(var_v * var_l)
SCICODE_GOLD_EOF
