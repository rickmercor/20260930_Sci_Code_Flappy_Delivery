#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def laguerre_scale(alpha, gam, nu, rho, v0, theta, big_t):
    from math import sqrt, gamma as _gamma_fn
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam <= 0.0 or nu <= 0.0:
        raise ValueError("gam and nu must be positive")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if big_t <= 0.0:
        raise ValueError("big_t must be positive")
    return (sqrt(1.0 - rho * rho) / (gam * nu)
            * (gam * theta * big_t
               + v0 * big_t ** (1.0 - alpha) / _gamma_fn(2.0 - alpha)))

def scaled_laguerre_exponential(sigma, n_quad, a):
    import numpy as np
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if a <= 0.0:
        raise ValueError("a must be positive")
    n_quad = int(n_quad)
    if n_quad < 1:
        raise ValueError("n_quad must be >= 1")
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    u_n = x / sigma
    w_n = w / sigma
    return 2.0 * float(np.sum(w_n * np.exp(x) * np.exp(-a * u_n)))

def _s03_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s03_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s03_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s03_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s03_F(xi, h[n + 1], gam, nu, rho)
    return h


def riccati_terminal_re(u, damp, level, alpha, gam, nu, rho, big_t,
                                m0):
    import numpy as np
    if u < 0.0:
        raise ValueError("u must be non-negative")
    if damp > 0.0:
        raise ValueError("damp must be <= 0")
    level = int(level)
    if level < 0:
        raise ValueError("level must be >= 0")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam < 0.0 or nu <= 0.0:
        raise ValueError("gam must be >= 0 and nu > 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if big_t <= 0.0 or int(m0) < 1:
        raise ValueError("big_t must be > 0 and m0 >= 1")
    num_steps = int(m0) * 2 ** level
    dt = big_t / num_steps
    xi = np.array([u + 1j * damp])
    h = _s03_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    return float(h[-1, 0].real)

def _s04_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s04_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s04_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s04_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s04_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s04_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s04_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s04_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def cf_exponent_re(u, damp, level, alpha, gam, nu, rho, v0, theta,
                           s0, r, big_t, m0):
    import numpy as np
    if u < 0.0:
        raise ValueError("u must be non-negative")
    if damp > 0.0:
        raise ValueError("damp must be <= 0")
    level = int(level)
    if level < 0:
        raise ValueError("level must be >= 0")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam < 0.0 or nu <= 0.0:
        raise ValueError("gam must be >= 0 and nu > 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or big_t <= 0.0 or int(m0) < 1:
        raise ValueError("s0, big_t must be positive and m0 >= 1")
    xi = np.array([u + 1j * damp])
    gv = _s04_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                       big_t, int(m0))
    return float(gv[0].real)

def _s05_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s05_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s05_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s05_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s05_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s05_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s05_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s05_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s05_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s05_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def fourier_integrand(u, damp, level, alpha, gam, nu, rho, v0,
                              theta, s0, strike, r, big_t, m0):
    if u < 0.0:
        raise ValueError("u must be non-negative")
    if damp >= -1.0:
        raise ValueError("damp must be < -1 for the call payoff transform")
    level = int(level)
    if level < 0:
        raise ValueError("level must be >= 0")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam < 0.0 or nu <= 0.0:
        raise ValueError("gam must be >= 0 and nu > 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or strike <= 0.0 or big_t <= 0.0 or int(m0) < 1:
        raise ValueError("s0, strike, big_t must be positive and m0 >= 1")
    vals = _s05_integrand(u, damp, level, alpha, gam, nu, rho, v0, theta,
                          s0, strike, r, big_t, int(m0))
    return float(vals[0])

def _s06_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s06_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s06_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s06_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s06_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s06_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s06_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s06_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s06_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s06_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def _s06_lag_nodes(n_quad, sigma):
    """Nodes/weights of the n_quad-point Gauss-Laguerre rule with weight
    e^{-sigma u} on (0, inf): standard nodes/weights rescaled by 1/sigma."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return x / sigma, w / sigma


def _s06_quad_apply(vals, n_quad, sigma):
    """Q_N^sigma[g] = 2 * sum_n w_n * e^{sigma u_n} * g(u_n) (half line,
    factor 2 absorbed)."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return 2.0 * float(np.sum((w / sigma) * np.exp(x) * vals))


def _s06_level_price(level, n_quad, sigma, damp, alpha, gam, nu, rho, v0,
                   theta, s0, strike, r, big_t, m0):
    """Single-level scaled Gauss-Laguerre price V_(N, level)."""
    un, _ = _s06_lag_nodes(n_quad, sigma)
    gv = _s06_integrand(un, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                      strike, r, big_t, m0)
    return _s06_quad_apply(gv, n_quad, sigma)


def single_level_price(level, n_quad, sigma, damp, alpha, gam, nu,
                               rho, v0, theta, s0, strike, r, big_t, m0):
    level = int(level)
    n_quad = int(n_quad)
    if level < 0 or n_quad < 1:
        raise ValueError("level must be >= 0 and n_quad >= 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if damp >= -1.0:
        raise ValueError("damp must be < -1")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam < 0.0 or nu <= 0.0:
        raise ValueError("gam must be >= 0 and nu > 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or strike <= 0.0 or big_t <= 0.0 or int(m0) < 1:
        raise ValueError("s0, strike, big_t must be positive and m0 >= 1")
    return float(_s06_level_price(level, n_quad, sigma, damp, alpha, gam,
                                  nu, rho, v0, theta, s0, strike, r, big_t,
                                  int(m0)))

def _s07_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s07_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s07_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s07_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s07_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s07_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s07_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s07_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s07_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s07_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def _s07_lag_nodes(n_quad, sigma):
    """Nodes/weights of the n_quad-point Gauss-Laguerre rule with weight
    e^{-sigma u} on (0, inf): standard nodes/weights rescaled by 1/sigma."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return x / sigma, w / sigma


def _s07_quad_apply(vals, n_quad, sigma):
    """Q_N^sigma[g] = 2 * sum_n w_n * e^{sigma u_n} * g(u_n) (half line,
    factor 2 absorbed)."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return 2.0 * float(np.sum((w / sigma) * np.exp(x) * vals))


def _s07_level_price(level, n_quad, sigma, damp, alpha, gam, nu, rho, v0,
                   theta, s0, strike, r, big_t, m0):
    """Single-level scaled Gauss-Laguerre price V_(N, level)."""
    un, _ = _s07_lag_nodes(n_quad, sigma)
    gv = _s07_integrand(un, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                      strike, r, big_t, m0)
    return _s07_quad_apply(gv, n_quad, sigma)


def select_level(eps_disc, p, n_pilot, sigma, damp, alpha, gam, nu,
                         rho, v0, theta, s0, strike, r, big_t, m0):
    from math import ceil, log2
    if eps_disc <= 0.0 or p <= 0.0:
        raise ValueError("eps_disc and p must be positive")
    n_pilot = int(n_pilot)
    if n_pilot < 1:
        raise ValueError("n_pilot must be >= 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if damp >= -1.0:
        raise ValueError("damp must be < -1")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam < 0.0 or nu <= 0.0:
        raise ValueError("gam must be >= 0 and nu > 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or strike <= 0.0 or big_t <= 0.0 or int(m0) < 1:
        raise ValueError("s0, strike, big_t must be positive and m0 >= 1")
    v1 = _s07_level_price(1, n_pilot, sigma, damp, alpha, gam, nu, rho, v0,
                          theta, s0, strike, r, big_t, int(m0))
    v0p = _s07_level_price(0, n_pilot, sigma, damp, alpha, gam, nu, rho,
                           v0, theta, s0, strike, r, big_t, int(m0))
    d1 = abs(v1 - v0p)
    if d1 == 0.0:
        return 1.0
    raw = 1.0 + log2(d1 / ((2.0 ** p - 1.0) * eps_disc)) / p
    return float(max(1, int(ceil(raw))))

def _s08_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx, beta, p,
                    big_t, m0):
    """Allocation of quadrature points: level-zero and correction counts.

    One half of eps_quad is assigned to the level-zero term and one half to
    the corrections.  Correction prefactors are propagated from the first
    correction level, A_l = a1 * (dt_l / dt_1)^p.  Per-point correction cost
    is c_l = W_l + W_{l-1} with W_l = dt_l^{-beta}.  Real-valued minimisers
    are rounded up (ceiling)."""
    import numpy as np
    from math import ceil
    dt = np.array([big_t / (m0 * 2 ** l) for l in range(big_l + 1)])
    w_cost = dt ** (-beta)
    n0_star = (2.0 * a0 / eps_quad) ** (2.0 / s0_idx)
    a_l = np.array([a1 * (dt[l] / dt[1]) ** p for l in range(1, big_l + 1)])
    c_l = np.array([w_cost[l] + w_cost[l - 1] for l in range(1, big_l + 1)])
    ssum = float(np.sum(a_l ** (2.0 / (s_idx + 2.0))
                        * c_l ** (s_idx / (s_idx + 2.0))))
    n_star = ((a_l / c_l) ** (2.0 / (s_idx + 2.0))
              * ((2.0 / eps_quad) * ssum) ** (2.0 / s_idx))
    n0 = int(ceil(n0_star - 1e-12))
    nl = [max(1, int(ceil(v - 1e-12))) for v in n_star]
    return n0, nl


def allocate_points_total(big_l, eps_quad, a0, s0_idx, a1, s_idx,
                                  beta, p, big_t, m0):
    big_l = int(big_l)
    if big_l < 1:
        raise ValueError("big_l must be >= 1")
    if eps_quad <= 0.0:
        raise ValueError("eps_quad must be positive")
    if a0 <= 0.0 or a1 <= 0.0:
        raise ValueError("a0 and a1 must be positive")
    if s0_idx < 1.0 or s_idx < 1.0:
        raise ValueError("s0_idx and s_idx must be >= 1")
    if beta <= 0.0 or p <= 0.0:
        raise ValueError("beta and p must be positive")
    if big_t <= 0.0 or int(m0) < 1:
        raise ValueError("big_t must be > 0 and m0 >= 1")
    n0, nl = _s08_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx,
                               beta, p, big_t, int(m0))
    return float(n0 + sum(nl))

def _s09_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s09_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s09_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s09_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s09_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s09_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s09_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s09_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s09_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s09_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def _s09_lag_nodes(n_quad, sigma):
    """Nodes/weights of the n_quad-point Gauss-Laguerre rule with weight
    e^{-sigma u} on (0, inf): standard nodes/weights rescaled by 1/sigma."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return x / sigma, w / sigma


def _s09_quad_apply(vals, n_quad, sigma):
    """Q_N^sigma[g] = 2 * sum_n w_n * e^{sigma u_n} * g(u_n) (half line,
    factor 2 absorbed)."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return 2.0 * float(np.sum((w / sigma) * np.exp(x) * vals))


def _s09_level_price(level, n_quad, sigma, damp, alpha, gam, nu, rho, v0,
                   theta, s0, strike, r, big_t, m0):
    """Single-level scaled Gauss-Laguerre price V_(N, level)."""
    un, _ = _s09_lag_nodes(n_quad, sigma)
    gv = _s09_integrand(un, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                      strike, r, big_t, m0)
    return _s09_quad_apply(gv, n_quad, sigma)


def _s09_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx, beta, p,
                    big_t, m0):
    """Allocation of quadrature points: level-zero and correction counts.

    One half of eps_quad is assigned to the level-zero term and one half to
    the corrections.  Correction prefactors are propagated from the first
    correction level, A_l = a1 * (dt_l / dt_1)^p.  Per-point correction cost
    is c_l = W_l + W_{l-1} with W_l = dt_l^{-beta}.  Real-valued minimisers
    are rounded up (ceiling)."""
    import numpy as np
    from math import ceil
    dt = np.array([big_t / (m0 * 2 ** l) for l in range(big_l + 1)])
    w_cost = dt ** (-beta)
    n0_star = (2.0 * a0 / eps_quad) ** (2.0 / s0_idx)
    a_l = np.array([a1 * (dt[l] / dt[1]) ** p for l in range(1, big_l + 1)])
    c_l = np.array([w_cost[l] + w_cost[l - 1] for l in range(1, big_l + 1)])
    ssum = float(np.sum(a_l ** (2.0 / (s_idx + 2.0))
                        * c_l ** (s_idx / (s_idx + 2.0))))
    n_star = ((a_l / c_l) ** (2.0 / (s_idx + 2.0))
              * ((2.0 / eps_quad) * ssum) ** (2.0 / s_idx))
    n0 = int(ceil(n0_star - 1e-12))
    nl = [max(1, int(ceil(v - 1e-12))) for v in n_star]
    return n0, nl


def rough_heston_multilevel_price(eps, damp, alpha, gam, nu, rho,
                                          v0, theta, s0, strike, r, big_t,
                                          m0, n_pilot, a0, s0_idx, a1,
                                          s_idx, beta):
    import numpy as np
    from math import ceil, log2
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if damp >= -1.0:
        raise ValueError("damp must be < -1")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam <= 0.0 or nu <= 0.0:
        raise ValueError("gam and nu must be positive")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or strike <= 0.0 or big_t <= 0.0:
        raise ValueError("s0, strike, big_t must be positive")
    m0 = int(m0)
    n_pilot = int(n_pilot)
    if m0 < 1 or n_pilot < 1:
        raise ValueError("m0 and n_pilot must be >= 1")
    if a0 <= 0.0 or a1 <= 0.0 or s0_idx < 1.0 or s_idx < 1.0:
        raise ValueError("invalid quadrature-model constants")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    # empirical rate convention of the pipeline
    p_rate = 1.0 + alpha
    eps_disc = 0.5 * eps
    eps_quad = 0.5 * eps

    # step-1 oracle: weight scale
    sigma = laguerre_scale(alpha, gam, nu, rho, v0, theta, big_t)

    # step-2 oracle: scaled-rule invariant  Q_N^sigma[e^-sigma u] = 2/sigma
    inv = scaled_laguerre_exponential(sigma, 8, sigma)
    if abs(inv - 2.0 / sigma) > 1e-9:
        raise ValueError("scaled-rule invariant violated")

    # step-6 oracle: pilot level prices; pilot indicator
    v1p = single_level_price(1, n_pilot, sigma, damp, alpha, gam,
                                     nu, rho, v0, theta, s0, strike, r,
                                     big_t, m0)
    v0p = single_level_price(0, n_pilot, sigma, damp, alpha, gam,
                                     nu, rho, v0, theta, s0, strike, r,
                                     big_t, m0)
    d1 = abs(v1p - v0p)

    # step-7 oracle: selected level, cross-checked
    big_l = int(select_level(eps_disc, p_rate, n_pilot, sigma,
                                     damp, alpha, gam, nu, rho, v0, theta,
                                     s0, strike, r, big_t, m0))
    if d1 == 0.0:
        big_l_int = 1
    else:
        big_l_int = max(1, int(ceil(
            1.0 + log2(d1 / ((2.0 ** p_rate - 1.0) * eps_disc)) / p_rate)))
    if big_l != big_l_int:
        raise ValueError("level-selection cross-check failed")

    # step-8 oracle: allocation, cross-checked against internal vector
    n0, nl = _s09_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx,
                               beta, p_rate, big_t, m0)
    total = allocate_points_total(big_l, eps_quad, a0, s0_idx, a1,
                                          s_idx, beta, p_rate, big_t, m0)
    if int(total) != n0 + sum(nl):
        raise ValueError("allocation cross-check failed")

    # step-3/4/5 oracles: probe cross-checks against the local trace
    u_probe = 1.5
    href = riccati_terminal_re(u_probe, damp, 1, alpha, gam, nu,
                                       rho, big_t, m0)
    htr = _s09_adams(np.array([u_probe + 1j * damp]), 2 * m0,
                     big_t / (2 * m0), alpha, gam, nu, rho)[-1, 0].real
    if abs(href - htr) > 1e-9:
        raise ValueError("Riccati probe cross-check failed")
    gref = cf_exponent_re(u_probe, damp, 1, alpha, gam, nu, rho,
                                  v0, theta, s0, r, big_t, m0)
    gtr = _s09_exponent(np.array([u_probe + 1j * damp]), 1, alpha, gam, nu,
                        rho, v0, theta, s0, r, big_t, m0)[0].real
    if abs(gref - gtr) > 1e-9:
        raise ValueError("exponent probe cross-check failed")
    iref = fourier_integrand(u_probe, damp, 1, alpha, gam, nu, rho,
                                     v0, theta, s0, strike, r, big_t, m0)
    itr = float(_s09_integrand(u_probe, damp, 1, alpha, gam, nu, rho, v0,
                               theta, s0, strike, r, big_t, m0)[0])
    if abs(iref - itr) > 1e-9:
        raise ValueError("integrand probe cross-check failed")

    # telescoped multilevel price (local re-trace of nodal values)
    u0, _ = _s09_lag_nodes(n0, sigma)
    g0 = _s09_integrand(u0, damp, 0, alpha, gam, nu, rho, v0, theta, s0,
                        strike, r, big_t, m0)
    price = _s09_quad_apply(g0, n0, sigma)
    for lev in range(1, big_l + 1):
        un, _ = _s09_lag_nodes(nl[lev - 1], sigma)
        g_hi = _s09_integrand(un, damp, lev, alpha, gam, nu, rho, v0,
                              theta, s0, strike, r, big_t, m0)
        g_lo = _s09_integrand(un, damp, lev - 1, alpha, gam, nu, rho, v0,
                              theta, s0, strike, r, big_t, m0)
        price += _s09_quad_apply(g_hi - g_lo, nl[lev - 1], sigma)
    return float(price)
SCICODE_GOLD_EOF
