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


def adams_convolution_weights(alpha, dt, M):
    def _rows(alpha, dt, M):
        ca = dt ** alpha / (alpha * (alpha + 1.0))
        cb = dt ** alpha / alpha
        a = np.zeros((M + 1, M + 1))
        b = np.zeros((M + 1, M + 1))
        for jp in range(1, M + 1):
            j = jp - 1
            a[jp, 0] = ca * (j ** (alpha + 1.0)
                             - (j - alpha) * (j + 1.0) ** alpha)
            if j >= 1:
                k = np.arange(1, j + 1)
                a[jp, 1:j + 1] = ca * ((j - k + 2.0) ** (alpha + 1.0)
                                       + (j - k) ** (alpha + 1.0)
                                       - 2.0 * (j - k + 1.0) ** (alpha + 1.0))
            a[jp, jp] = ca
            k = np.arange(0, j + 1)
            b[jp, 0:j + 1] = cb * ((j + 1.0 - k) ** alpha - (j - k) ** alpha)
        return a, b

    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    M = int(M)
    if M < 1:
        raise ValueError("M must be at least 1")
    a, b = _rows(float(alpha), float(dt), M)
    return np.concatenate((a.reshape(-1), b.reshape(-1)))

import numpy as np


def solve_riccati_nodal(w_packed, xi_re, xi_im, alpha, dt, M, gam, nu,
                                rho):
    def _rhs(xi, h, gam, nu, rho):
        return (-0.5 * (xi * xi + 1j * xi)
                + gam * (1j * xi * rho * nu - 1.0) * h
                + 0.5 * (gam * nu) ** 2 * h * h)

    def _gamma(x):
        c = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
             -176.61502916214059, 12.507343278686905, -0.13857109526572012,
             9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return np.pi / (np.sin(np.pi * x) * _gamma(1.0 - x))
        x -= 1.0
        a = 0.99999999999980993
        t = x + 7.5
        for i, ci in enumerate(c):
            a += ci / (x + i + 1.0)
        return np.sqrt(2.0 * np.pi) * t ** (x + 0.5) * np.exp(-t) * a

    M = int(M)
    if M < 1:
        raise ValueError("M must be at least 1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    w = np.asarray(w_packed, dtype=float).reshape(-1)
    if w.size != 2 * (M + 1) ** 2:
        raise ValueError("w_packed does not match M")

    n = M + 1
    a = w[:n * n].reshape(n, n)
    b = w[n * n:].reshape(n, n)
    gam, nu, rho = float(gam), float(nu), float(rho)
    xi = float(xi_re) + 1j * float(xi_im)
    ga = _gamma(float(alpha))

    h = np.zeros(n, dtype=complex)
    Fv = np.zeros(n, dtype=complex)
    Fv[0] = _rhs(xi, 0.0 + 0.0j, gam, nu, rho)
    for jp in range(1, n):
        j = jp - 1
        hp = (b[jp, 0:j + 1] @ Fv[0:j + 1]) / ga
        h[jp] = (a[jp, 0:j + 1] @ Fv[0:j + 1]
                 + a[jp, jp] * _rhs(xi, hp, gam, nu, rho)) / ga
        Fv[jp] = _rhs(xi, h[jp], gam, nu, rho)
    return np.concatenate((h.real, h.imag))

import numpy as np


def characteristic_exponent(h_packed, xi_re, xi_im, dt, X0, r, gam, nu,
                                    rho, theta, V0):
    def _rhs(xi, h, gam, nu, rho):
        return (-0.5 * (xi * xi + 1j * xi)
                + gam * (1j * xi * rho * nu - 1.0) * h
                + 0.5 * (gam * nu) ** 2 * h * h)

    def _trapezoid(v, dt):
        return dt * (0.5 * v[0] + v[1:-1].sum() + 0.5 * v[-1])

    v = np.asarray(h_packed, dtype=float).reshape(-1)
    if v.size % 2 != 0 or v.size < 4:
        raise ValueError("h_packed must hold at least two nodes in [Re; Im] form")
    n = v.size // 2
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")

    h = v[:n] + 1j * v[n:]
    dt = float(dt)
    T = (n - 1) * dt
    xi = float(xi_re) + 1j * float(xi_im)
    J = float(theta) * float(gam) * h + float(V0) * _rhs(
        xi, h, float(gam), float(nu), float(rho))
    G = 1j * xi * (float(X0) + float(r) * T) + _trapezoid(J, dt)
    return np.array([G.real, G.imag])

import numpy as np


def laguerre_scaling(T, alpha, gam, nu, rho, theta, V0):
    def _gamma(x):
        g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
             -176.61502916214059, 12.507343278686905, -0.13857109526572012,
             9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return np.pi / (np.sin(np.pi * x) * _gamma(1.0 - x))
        x -= 1.0
        a = 0.99999999999980993
        t = x + 7.5
        for i, c in enumerate(g):
            a += c / (x + i + 1.0)
        return np.sqrt(2.0 * np.pi) * t ** (x + 0.5) * np.exp(-t) * a

    T, alpha = float(T), float(alpha)
    gam, nu, rho = float(gam), float(nu), float(rho)
    theta, V0 = float(theta), float(V0)
    if abs(rho) >= 1.0:
        raise ValueError("the scale degenerates at |rho| = 1")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if gam * nu == 0.0:
        raise ValueError("gam*nu must be non-zero")
    pref = np.sqrt(1.0 - rho * rho) / (gam * nu)
    bracket = gam * theta * T + V0 * T ** (1.0 - alpha) / _gamma(2.0 - alpha)
    return float(pref * bracket)

import numpy as np


def scaled_laguerre_rule(N, sigma):
    def _standard(N):
        k = np.arange(N, dtype=float)
        diag = 2.0 * k + 1.0
        off = -(k[1:])
        J = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        x, V = np.linalg.eigh(J)
        w = V[0, :] ** 2
        order = np.argsort(x)
        return x[order], w[order]

    N = int(N)
    if N < 1:
        raise ValueError("N must be at least 1")
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    x, w = _standard(N)
    return np.concatenate((x / sigma, w / sigma))

import numpy as np


def fourier_integrand(u, G_re, G_im, R, K, r, T):
    def _payoff_hat(xi, K):
        return -(K ** (1.0 - 1j * xi)) / (xi * xi + 1j * xi)

    u = np.asarray(u, dtype=float).reshape(-1)
    gr = np.asarray(G_re, dtype=float).reshape(-1)
    gi = np.asarray(G_im, dtype=float).reshape(-1)
    if u.size != gr.size or u.size != gi.size:
        raise ValueError("u, G_re and G_im must have the same length")
    R, K, r, T = float(R), float(K), float(r), float(T)
    if R >= -1.0:
        raise ValueError("payoff admissibility requires R < -1")
    if K <= 0.0:
        raise ValueError("K must be positive")
    xi = u + 1j * R
    return np.exp(-r * T) / (2.0 * np.pi) * np.real(
        np.exp(gr + 1j * gi) * _payoff_hat(xi, K))

import numpy as np


def select_discretization_level(D1, eps_disc, p):
    D1, eps_disc, p = float(D1), float(eps_disc), float(p)
    if D1 <= 0.0:
        raise ValueError("D1 must be positive")
    if eps_disc <= 0.0:
        raise ValueError("eps_disc must be positive")
    if p <= 0.0:
        raise ValueError("p must be positive")
    inner = D1 / ((2.0 ** p - 1.0) * eps_disc)
    return float(max(1, int(np.ceil(1.0 + np.log2(inner) / p))))

import numpy as np


def allocate_quadrature_points(A, s0, s, eps_quad, W):
    A = np.asarray(A, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float).reshape(-1)
    s0, s, eps_quad = float(s0), float(s), float(eps_quad)
    if A.size != W.size:
        raise ValueError("A and W must have the same length")
    if A.size < 1:
        raise ValueError("A must hold at least the level-zero constant")
    if eps_quad <= 0.0:
        raise ValueError("eps_quad must be positive")
    if s0 <= 0.0 or s <= 0.0:
        raise ValueError("smoothness indices must be positive")
    if np.any(A <= 0.0):
        raise ValueError("quadrature constants must be positive")

    L = A.size - 1
    N0 = (2.0 * A[0] / eps_quad) ** (2.0 / s0)
    out = np.empty(L + 1)
    out[0] = np.ceil(N0)
    if L >= 1:
        pair = W[1:] + W[:-1]
        Ak = A[1:]
        total = np.sum(Ak ** (2.0 / (s + 2.0)) * pair ** (s / (s + 2.0)))
        lead = ((2.0 / eps_quad) * total) ** (2.0 / s)
        out[1:] = np.ceil((Ak / pair) ** (2.0 / (s + 2.0)) * lead)
    return np.maximum(out, 1.0)

import numpy as np


def run_multilevel_pricer(model, contract, numerics):
    def _level_integrand(u, level, alpha, dt0, T, gam, nu, rho, theta, V0,
                         X0, r, R):
        dt = dt0 * 2.0 ** (-level)
        M = int(round(T / dt))
        w = adams_convolution_weights(alpha, dt, M)
        gr = np.empty(u.size)
        gi = np.empty(u.size)
        for i in range(u.size):
            hp = solve_riccati_nodal(w, float(u[i]), R, alpha, dt, M,
                                             gam, nu, rho)
            G = characteristic_exponent(hp, float(u[i]), R, dt, X0, r,
                                                gam, nu, rho, theta, V0)
            gr[i] = G[0]
            gi[i] = G[1]
        return gr, gi

    model = np.asarray(model, dtype=float).reshape(-1)
    contract = np.asarray(contract, dtype=float).reshape(-1)
    numerics = np.asarray(numerics, dtype=float).reshape(-1)
    if model.size != 6 or contract.size != 5 or numerics.size != 8:
        raise ValueError("model, contract and numerics must have sizes 6, 5, 8")

    alpha, gam, nu, rho, V0, theta = model
    S0, K, T, r, R = contract
    dt0, eps, nbar, A0, s0, CA, s, beta = numerics
    nbar = int(nbar)
    if S0 <= 0.0:
        raise ValueError("S0 must be positive")
    X0 = float(np.log(S0))
    p = 1.0 + alpha
    eps_disc = eps_quad = 0.5 * eps

    sigma = laguerre_scaling(T, alpha, gam, nu, rho, theta, V0)

    def _value(N, level):
        rule = scaled_laguerre_rule(int(N), sigma)
        n = int(N)
        u, wq = rule[:n], rule[n:]
        gr, gi = _level_integrand(u, level, alpha, dt0, T, gam, nu, rho,
                                  theta, V0, X0, r, R)
        g = fourier_integrand(u, gr, gi, R, K, r, T)
        return 2.0 * float(np.sum(wq * np.exp(sigma * u) * g))

    D1 = abs(_value(nbar, 1) - _value(nbar, 0))
    L = int(select_discretization_level(D1, eps_disc, p))

    dt = np.array([dt0 * 2.0 ** (-l) for l in range(L + 1)])
    W = dt ** (-beta)
    A = np.concatenate(([A0], CA * dt[1:] ** p))
    N = allocate_quadrature_points(A, s0, s, eps_quad, W)

    total = _value(int(N[0]), 0)
    for l in range(1, L + 1):
        n = int(N[l])
        rule = scaled_laguerre_rule(n, sigma)
        u, wq = rule[:n], rule[n:]
        gr_hi, gi_hi = _level_integrand(u, l, alpha, dt0, T, gam, nu, rho,
                                        theta, V0, X0, r, R)
        gr_lo, gi_lo = _level_integrand(u, l - 1, alpha, dt0, T, gam, nu, rho,
                                        theta, V0, X0, r, R)
        diff = (fourier_integrand(u, gr_hi, gi_hi, R, K, r, T)
                - fourier_integrand(u, gr_lo, gi_lo, R, K, r, T))
        total += 2.0 * float(np.sum(wq * np.exp(sigma * u) * diff))
    return float(total)
SCICODE_GOLD_EOF
