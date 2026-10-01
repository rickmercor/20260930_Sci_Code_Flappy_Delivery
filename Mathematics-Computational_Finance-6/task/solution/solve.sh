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
import math


def lift_parameters(H: float, N: int) -> tuple:
    """Reference implementation of lift_parameters."""
    if not isinstance(H, (int, float, np.integer, np.floating)) or not (0.0 < float(H) < 0.5):
        raise ValueError("H must be a real number with 0 < H < 0.5")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or int(N) < 1:
        raise ValueError("N must be an integer >= 1")
    H = float(H)
    N = int(N)
    rN = 1.0 + 10.0 * N ** (-0.9)
    n = np.arange(1, N + 1)
    omega = ((rN ** (0.5 - H) - 1.0)
             * rN ** ((H - 0.5) * (1.0 + 0.5 * N))
             / (math.gamma(H + 0.5) * math.gamma(1.5 - H))
             * rN ** ((0.5 - H) * n))
    x = ((0.5 - H) / (1.5 - H)
         * (rN ** (1.5 - H) - 1.0) / (rN ** (0.5 - H) - 1.0)
         * rN ** (n - 1.0 - N / 2.0))
    return omega, x

import numpy as np
import math


def state_matrix(lam: float, omega: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference implementation of state_matrix."""
    if not isinstance(lam, (int, float, np.integer, np.floating)) or isinstance(lam, bool):
        raise ValueError("lam must be a real number >= 0")
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a real finite number >= 0")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    if om.ndim != 1 or xs.ndim != 1:
        raise ValueError("omega and x must be one-dimensional arrays")
    if om.shape[0] != xs.shape[0] or om.shape[0] < 1:
        raise ValueError("omega and x must be one-dimensional arrays of the same length >= 1")
    n = xs.shape[0]
    return -lam * np.outer(np.ones(n), om) - np.diag(xs)

import numpy as np
import math


def conditional_mean(lam: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Reference implementation of conditional_mean."""
    np = __import__("numpy")
    math = __import__("math")
    expm = __import__("scipy.linalg", fromlist=["expm"]).expm
    exprel = __import__("scipy.special", fromlist=["exprel"]).exprel

    def curve_integral(elapsed):
        # G0(s,s+elapsed), evaluated without subtracting almost equal terms.
        a = -xs * elapsed
        small = abs(a) < 0.05
        phi2 = np.empty_like(a)
        aa = a[small]
        value = np.full_like(aa, 1.0 / math.factorial(12))
        for j in range(11, 1, -1):
            value = 1.0 / math.factorial(j) + aa * value
        phi2[small] = value
        phi2[~small] = (np.expm1(a[~small]) - a[~small]) / a[~small]**2
        initial = v0 + lam * theta * np.sum(om * s * exprel(-xs * s))
        return initial * elapsed + lam * theta * np.sum(om * np.exp(-xs*s) * elapsed**2 * phi2)
    if not isinstance(n_grid, (int, np.integer)) or isinstance(n_grid, bool) or int(n_grid) < 2:
        raise ValueError("n_grid must be an integer >= 2")
    for name, value in (("lam", lam), ("v0", v0), ("theta", theta)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a real finite number")
    for name, value in (("s", s), ("t", t)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number")
    if not (float(t) > float(s)):
        raise ValueError("t must be strictly greater than s")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    if om.ndim != 1 or xs.ndim != 1 or om.shape[0] != xs.shape[0] or om.shape[0] < 1:
        raise ValueError("omega and x must be one-dimensional arrays of the same length >= 1")
    n = om.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("A must be a square matrix of shape (N, N)")
    if U_s.ndim != 1 or U_s.shape[0] != n:
        raise ValueError("U_s must be a one-dimensional array of length N")

    lam = float(lam); v0 = float(v0); theta = float(theta)
    s = float(s); t = float(t); n_grid = int(n_grid)
    one = np.ones(n)
    dt = t - s
    h = dt / (n_grid - 1)
    xi = np.zeros(n)
    for k in range(n_grid - 1):
        uk = s + k * h
        g1 = curve_integral(k * h)
        k1 = A @ xi - lam * g1 * one
        g2 = curve_integral((k + 0.5) * h)
        k2 = A @ (xi + 0.5 * h * k1) - lam * g2 * one
        g3 = curve_integral((k + 0.5) * h)
        k3 = A @ (xi + 0.5 * h * k2) - lam * g3 * one
        g4 = curve_integral((k + 1) * h)
        k4 = A @ (xi + h * k3) - lam * g4 * one
        xi = xi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = A
    augmented[:n, n] = U_s
    eta = expm(augmented * dt)[:n, n]
    mu = eta + xi
    alpha = float(om @ mu + curve_integral(dt))
    return alpha, mu

import numpy as np
import math


def cross_moment(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Reference implementation of cross_moment."""
    np = __import__("numpy")
    math = __import__("math")
    expm = __import__("scipy.linalg", fromlist=["expm"]).expm
    exprel = __import__("scipy.special", fromlist=["exprel"]).exprel

    def curve_integral(elapsed):
        # G0(s,s+elapsed), evaluated without subtracting almost equal terms.
        a = -xs * elapsed
        small = abs(a) < 0.05
        phi2 = np.empty_like(a)
        aa = a[small]
        value = np.full_like(aa, 1.0 / math.factorial(12))
        for j in range(11, 1, -1):
            value = 1.0 / math.factorial(j) + aa * value
        phi2[small] = value
        phi2[~small] = (np.expm1(a[~small]) - a[~small]) / a[~small]**2
        initial = v0 + lam * theta * np.sum(om * s * exprel(-xs * s))
        return initial * elapsed + lam * theta * np.sum(om * np.exp(-xs*s) * elapsed**2 * phi2)
    if not isinstance(n_grid, (int, np.integer)) or isinstance(n_grid, bool) or int(n_grid) < 2:
        raise ValueError("n_grid must be an integer >= 2")
    for name, value in (("lam", lam), ("nu", nu), ("v0", v0), ("theta", theta)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a real finite number")
    for name, value in (("s", s), ("t", t)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number")
    if not (float(t) > float(s)):
        raise ValueError("t must be strictly greater than s")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    if om.ndim != 1 or xs.ndim != 1 or om.shape[0] != xs.shape[0] or om.shape[0] < 1:
        raise ValueError("omega and x must be one-dimensional arrays of the same length >= 1")
    n = om.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("A must be a square matrix of shape (N, N)")
    if U_s.ndim != 1 or U_s.shape[0] != n:
        raise ValueError("U_s must be a one-dimensional array of length N")

    lam = float(lam); nu = float(nu); v0 = float(v0); theta = float(theta)
    s = float(s); t = float(t); n_grid = int(n_grid)
    one = np.ones(n)
    dt = t - s
    h = dt / (n_grid - 1)

    # joint RK4 for the conditional-mean forcing (xi) and the companion (psi)
    def _rhs_t6_04(uu, yy):
        xi, psi = yy[:n], yy[n:]
        g = curve_integral(uu)
        return np.concatenate([A @ xi - lam * g * one,
                               A @ psi + nu * (one * (om @ xi) + g) * one])

    y = np.zeros(2 * n)
    for k in range(n_grid - 1):
        uk = k * h
        k1 = _rhs_t6_04(uk, y)
        k2 = _rhs_t6_04(uk + 0.5 * h, y + 0.5 * h * k1)
        k3 = _rhs_t6_04(uk + 0.5 * h, y + 0.5 * h * k2)
        k4 = _rhs_t6_04(uk + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    xi, psi = y[:n], y[n:]

    # y'=Ay+U_s, chi'=A chi+nu*1*omega*y integrates the
    # homogeneous covariance contribution from its defining integral.
    augmented = np.zeros((2*n + 1, 2*n + 1))
    augmented[:n, :n] = A
    augmented[:n, -1] = U_s
    augmented[n:2*n, :n] = nu * np.outer(one, om)
    augmented[n:2*n, n:2*n] = A
    chi = expm(augmented * dt)[n:2*n, -1]
    kappa = chi + psi
    EXZ = float(om @ kappa)
    if not math.isfinite(EXZ) or abs(EXZ) < 1e-300:
        raise ValueError("the conditional cross-moment must be finite and non-zero")
    return EXZ, kappa

import numpy as np
import math


def _g0_t6_05(t, omega, x, lam, v0, theta):
    """Initial variance curve evaluated at t."""
    return float(v0 + lam * theta * np.sum(omega / x * (1.0 - np.exp(-x * t))))


def _real_scalar_05(value) -> bool:
    """True when value is a real scalar (not a bool, not an array)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def projection_slope(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, U_s: np.ndarray, s: float, t: float, mu: np.ndarray, alpha: float, kappa: np.ndarray, EXZ: float) -> tuple:
    """Reference implementation of projection_slope."""
    for name, value in (("lam", lam), ("nu", nu), ("v0", v0), ("theta", theta)):
        if not _real_scalar_05(value) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a real finite number")
    for name, value in (("s", s), ("t", t)):
        if not _real_scalar_05(value):
            raise ValueError(f"{name} must be a real number")
    if not _real_scalar_05(alpha) or float(alpha) == 0.0:
        raise ValueError("alpha must be a real number != 0")
    if not _real_scalar_05(EXZ) or float(EXZ) == 0.0:
        raise ValueError("EXZ must be a real number != 0")
    om = np.asarray(omega, dtype=float)
    xs = np.asarray(x, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    for name, arr in (("omega", om), ("x", xs), ("U_s", U_s), ("mu", mu), ("kappa", kappa)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
    if not (om.shape[0] == xs.shape[0] == U_s.shape[0] == mu.shape[0] == kappa.shape[0]):
        raise ValueError("omega, x, U_s, mu, and kappa must all have the same length")
    if om.shape[0] < 1:
        raise ValueError("omega, x, U_s, mu, and kappa must all have the same length")

    lam = float(lam); nu = float(nu); v0 = float(v0); theta = float(theta)
    s = float(s); t = float(t)
    alpha = float(alpha); EXZ = float(EXZ)

    beta = EXZ / alpha
    betaL = nu * float(np.sum(om)) / float(np.sum(om * (xs * kappa / EXZ + lam)))
    X0 = mu - alpha * (kappa / EXZ)
    c = float(om @ U_s - om @ (xs * X0) + _g0_t6_05(t, om, xs, lam, v0, theta))
    C0 = c - nu * float(np.sum(om)) * alpha / beta
    feasible = bool(beta > 0.0 and beta <= betaL and C0 >= 0.0)
    return beta, betaL, C0, c, feasible

import numpy as np
import math


def _real_scalar_05(value) -> bool:
    """True when value is a real scalar (not a bool, not an array)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def constrained_slope(nu: float, omega: np.ndarray, alpha: float, c: float, beta: float, betaL: float, C0: float) -> float:
    """Reference implementation of constrained_slope."""
    if not _real_scalar_05(alpha) or not (float(alpha) > 0.0):
        raise ValueError("alpha must be a real number > 0")
    if not _real_scalar_05(c) or float(c) == 0.0:
        raise ValueError("c must be a real number != 0")
    if not _real_scalar_05(nu) or not (float(nu) > 0.0):
        raise ValueError("nu must be a real number > 0")
    om = np.asarray(omega, dtype=float)
    if om.ndim != 1 or om.size < 1:
        raise ValueError("omega must be a one-dimensional array with at least one entry")
    for name, value in (("beta", beta), ("betaL", betaL), ("C0", C0)):
        if not _real_scalar_05(value):
            raise ValueError(f"{name} must be a real number")
    alpha = float(alpha)
    c = float(c)
    nu = float(nu)
    beta = float(beta)
    betaL = float(betaL)
    C0 = float(C0)
    feasible = (beta > 0.0) and (beta <= betaL) and (C0 >= 0.0)
    if feasible:
        return beta
    return nu * alpha * float(np.sum(om)) / c

import numpy as np
import math


def _real_scalar_07(value) -> bool:
    """True when value is a real scalar (not a bool, not an array)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _as_vector_07(value, name: str) -> np.ndarray:
    """Coerce value to a one-dimensional float array, or raise ValueError."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(f"{name} must be a one-dimensional array with at least one entry")
    return arr


def simulate_step(alpha: float, beta_tilde: float, mu: np.ndarray, kappa: np.ndarray, EXZ: float, x: np.ndarray, omega: np.ndarray, lam: float, nu: float, U_s: np.ndarray, z_k: float, u_k: float, g0_t: float) -> tuple:
    """Reference implementation of simulate_step."""
    if not _real_scalar_07(alpha) or not (float(alpha) > 0.0):
        raise ValueError("alpha must be a real number > 0")
    if not _real_scalar_07(beta_tilde) or not (float(beta_tilde) > 0.0):
        raise ValueError("beta_tilde must be a real number > 0")
    if not _real_scalar_07(EXZ) or float(EXZ) == 0.0:
        raise ValueError("EXZ must be a real number != 0")
    mu_v = _as_vector_07(mu, "mu")
    kappa_v = _as_vector_07(kappa, "kappa")
    x_v = _as_vector_07(x, "x")
    omega_v = _as_vector_07(omega, "omega")
    U_s_v = _as_vector_07(U_s, "U_s")
    n = mu_v.size
    if not (kappa_v.size == n and x_v.size == n and omega_v.size == n and U_s_v.size == n):
        raise ValueError("mu, kappa, x, omega and U_s must all have the same length")
    for name, value in (("lam", lam), ("nu", nu), ("z_k", z_k), ("u_k", u_k), ("g0_t", g0_t)):
        if not _real_scalar_07(value):
            raise ValueError(f"{name} must be a real number")
    alpha = float(alpha)
    beta_tilde = float(beta_tilde)
    EXZ = float(EXZ)
    lam = float(lam)
    nu = float(nu)
    z_k = float(z_k)
    u_k = float(u_k)
    g0_t = float(g0_t)
    gamma = (alpha / beta_tilde) ** 2
    V = z_k ** 2
    X1 = (alpha + alpha ** 2 * V / (2.0 * gamma)
          - (alpha / (2.0 * gamma)) * math.sqrt(4.0 * alpha * gamma * V + alpha ** 2 * V ** 2))
    Xhat = X1 if u_k <= alpha / (alpha + X1) else alpha ** 2 / X1
    Zhat = (Xhat - alpha) / beta_tilde
    Xn = mu_v + (kappa_v / EXZ) * (Xhat - alpha)
    U_next = U_s_v - x_v * Xn - lam * Xhat + nu * Zhat
    V_next = float(omega_v @ U_next) + g0_t
    return Xhat, Zhat, U_next, V_next

import numpy as np
import math


def _g0_t6_05(t, omega, x, lam, v0, theta):
    """Initial variance curve evaluated at t."""
    return float(v0 + lam * theta * np.sum(omega / x * (1.0 - np.exp(-x * t))))


def run_clp_march(H: float = 0.3, N: int = 5, lam: float = 0.25, nu: float = 0.1, v0: float = 0.02, theta: float = 0.5, T: float = 5.0, M: int = 10, seed: int = 12345, n_grid: int = 4001) -> float:
    """Reference implementation of run_clp_march: chains
    lift_parameters, state_matrix, conditional_mean,
    cross_moment, projection_slope, constrained_slope,
    and simulate_step step by step over the horizon, accumulating the
    realized variance increment."""
    if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or int(M) < 1:
        raise ValueError("M must be an integer >= 1")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(T, (int, float, np.integer, np.floating)) or isinstance(T, bool) or not (float(T) > 0.0):
        raise ValueError("T must be a real number > 0")
    if not isinstance(n_grid, (int, np.integer)) or isinstance(n_grid, bool) or int(n_grid) < 2:
        raise ValueError("n_grid must be an integer >= 2")
    M = int(M)
    seed = int(seed)
    T = float(T)
    n_grid = int(n_grid)

    omega, x = lift_parameters(H, N)
    A = state_matrix(lam, omega, x)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(M)
    u = rng.random(M)

    dt = T / M
    U = np.zeros(N)
    X_T = 0.0
    for k in range(M):
        s = k * dt
        t = (k + 1) * dt
        alpha, mu = conditional_mean(lam, v0, theta, omega, x, A, s, t, U, n_grid)
        EXZ, kappa = cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U, n_grid)
        beta, betaL, C0, c, _feasible = projection_slope(lam, nu, v0, theta, omega, x, U, s, t, mu, alpha, kappa, EXZ)
        beta_tilde = constrained_slope(nu, omega, alpha, c, beta, betaL, C0)
        g0_t = _g0_t6_05(t, omega, x, lam, v0, theta)
        Xhat, Zhat, U_next, V_next = simulate_step(alpha, beta_tilde, mu, kappa, EXZ, x, omega, lam, nu, U, float(z[k]), float(u[k]), g0_t)
        U = U_next
        X_T += Xhat
    return float(X_T)
SCICODE_GOLD_EOF
