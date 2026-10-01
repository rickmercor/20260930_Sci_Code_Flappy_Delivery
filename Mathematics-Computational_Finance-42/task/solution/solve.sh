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


def harmonic_fluctuation_width(omega, T, sigma):
    omega, T, sigma = float(omega), float(T), float(sigma)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    f = 0.5 * omega * T
    if f < 1.0e-3:
        bracket = f / 3.0 - f ** 3 / 45.0 + 2.0 * f ** 5 / 945.0
    else:
        bracket = 1.0 / np.tanh(f) - 1.0 / f
    return float(sigma * sigma / (2.0 * omega) * bracket)

import numpy as np


def variational_frequency(xbar, T, k, sigma, lam):
    def _width(omega, T, sigma):
        f = 0.5 * omega * T
        if f < 1.0e-3:
            br = f / 3.0 - f ** 3 / 45.0 + 2.0 * f ** 5 / 945.0
        else:
            br = 1.0 / np.tanh(f) - 1.0 / f
        return sigma * sigma / (2.0 * omega) * br

    xbar, T, k = float(xbar), float(T), float(k)
    sigma, lam = float(sigma), float(lam)
    if T <= 0.0:
        raise ValueError("T must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    omega = k
    for _ in range(10000):
        a = _width(omega, T, sigma)
        new = np.sqrt(k * k + sigma * sigma * lam * np.exp(0.5 * a + xbar))
        if abs(new - omega) <= 1.0e-15 * max(1.0, abs(new)):
            omega = new
            break
        omega = new
    return float(omega)

import numpy as np


def linear_coefficient_integrals(xbar, omega, T, k, sigma, theta):
    xbar, omega, T = float(xbar), float(omega), float(T)
    k, sigma, theta = float(k), float(sigma), float(theta)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    g = ((omega * omega - k * k) / sigma ** 2
         + k * k * (xbar - theta) / sigma ** 2)
    th = np.tanh(0.5 * omega * T)
    ghat = g * T
    r_sum = 2.0 * g / omega * th
    r_zero = g / omega * th
    r_cross = g * g / omega * (0.5 * T - th / omega)
    return np.array([g, ghat, r_sum, r_zero, r_cross])

import numpy as np


def displacement_and_correction(reduced, omega, T, sigma):
    v = np.asarray(reduced, dtype=float).reshape(-1)
    omega, T, sigma = float(omega), float(T), float(sigma)
    if v.size != 5:
        raise ValueError("reduced must hold five entries")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    ghat, r_sum, r_cross = v[1], v[2], v[4]
    f = 0.5 * omega * T
    gap = r_sum - ghat
    delta = sigma ** 2 / (2.0 * omega * f) * gap
    corr = sigma ** 2 / omega * (r_cross - 0.25 / f * gap * gap)
    d_gamma = sigma ** 2 / (2.0 * omega) * (r_sum / np.tanh(f) - ghat / f)
    return np.array([delta, corr, d_gamma])

import numpy as np


def log_trial_normalisation(xbar, omega, alpha, corr, T, k, sigma,
                                    lam, theta):
    xbar, omega, alpha, corr = float(xbar), float(omega), float(alpha), float(corr)
    T, k, sigma = float(T), float(k), float(sigma)
    lam, theta = float(lam), float(theta)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    f = 0.5 * omega * T
    w_int = T * (k * k * (theta - xbar) ** 2 / (2.0 * sigma ** 2)
                 + lam * np.exp(0.5 * alpha + xbar)
                 + (k * k - omega * omega) / (2.0 * sigma ** 2) * alpha)
    log_fsinh = np.log(2.0 * f) - f - np.log1p(-np.exp(-2.0 * f))
    return float(-0.5 * np.log(2.0 * np.pi * alpha)
                 - 0.5 * np.log(2.0 * np.pi * T * sigma ** 2)
                 + log_fsinh - w_int + corr)

import numpy as np


def endpoint_coefficients(xbar, x0, delta, reduced, alpha, omega,
                                  log_N, T, k, sigma, theta):
    v = np.asarray(reduced, dtype=float).reshape(-1)
    if v.size != 5:
        raise ValueError("reduced must hold five entries")
    xbar, x0, delta = float(xbar), float(x0), float(delta)
    alpha, omega, log_N = float(alpha), float(omega), float(log_N)
    T, k, sigma, theta = float(T), float(k), float(sigma), float(theta)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    f = 0.5 * omega * T
    coth = (1.0 + np.exp(-2.0 * f)) / (1.0 - np.exp(-2.0 * f))
    r_sum, r_zero = v[2], v[3]
    A = (1.0 / (8.0 * alpha) + omega * coth / (4.0 * sigma ** 2)
         + k / (2.0 * sigma ** 2))
    B = ((x0 - xbar + delta) / (2.0 * alpha) + r_zero
         + k * (x0 - theta) / sigma ** 2)
    C = (-(x0 - xbar + delta) ** 2 / (2.0 * alpha)
         - (x0 - xbar) * r_sum + k * T / 2.0)
    log_weight = 0.5 * np.log(np.pi / A) + log_N + C + B * B / (4.0 * A)
    return np.array([A, B, C, x0 - B / (2.0 * A), log_weight])

import numpy as np


def average_point_grid(T, x0, k, sigma, theta, nq, span):
    T, x0, k = float(T), float(x0), float(k)
    sigma, theta, span = float(sigma), float(theta), float(span)
    nq = int(nq)
    if T <= 0.0:
        raise ValueError("T must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if nq < 1:
        raise ValueError("nq must be at least 1")
    if span <= 0.0:
        raise ValueError("span must be positive")
    m = theta + (x0 - theta) * (1.0 - np.exp(-k * T)) / (k * T)
    s = sigma * np.sqrt(T / 3.0)
    lo, hi = m - span * s, m + span * s
    x, w = np.polynomial.legendre.leggauss(nq)
    return np.concatenate((0.5 * (hi - lo) * x + 0.5 * (hi + lo),
                           0.5 * (hi - lo) * w))

import numpy as np


def assemble_expectation(log_weight, centre, A, quad_weight):
    ln = np.asarray(log_weight, dtype=float).reshape(-1)
    ce = np.asarray(centre, dtype=float).reshape(-1)
    aa = np.asarray(A, dtype=float).reshape(-1)
    ww = np.asarray(quad_weight, dtype=float).reshape(-1)
    if not (ln.size == ce.size == aa.size == ww.size):
        raise ValueError("all four inputs must have the same length")
    if ln.size == 0:
        raise ValueError("inputs must be non-empty")
    if np.any(aa <= 0.0):
        raise ValueError("entries of A must be positive")

    def _total(logs):
        mx = logs.max()
        return float(np.exp(mx) * np.sum(ww * np.exp(logs - mx)))

    return np.array([_total(ln), _total(ln + ce + 1.0 / (4.0 * aa))])

import numpy as np


def par_cds_spread(model, contract, numerics):
    def _node(xbar, x0, T, k, sigma, theta, lam):
        omega = float(variational_frequency(xbar, T, k, sigma, lam))
        alpha = float(harmonic_fluctuation_width(omega, T, sigma))
        red = linear_coefficient_integrals(xbar, omega, T, k, sigma,
                                                   theta)
        dsp = displacement_and_correction(red, omega, T, sigma)
        lgn = log_trial_normalisation(xbar, omega, alpha, float(dsp[1]),
                                              T, k, sigma, lam, theta)
        return endpoint_coefficients(xbar, x0, float(dsp[0]), red,
                                             alpha, omega, lgn, T, k, sigma,
                                             theta)

    m = np.asarray(model, dtype=float).reshape(-1)
    c = np.asarray(contract, dtype=float).reshape(-1)
    q = np.asarray(numerics, dtype=float).reshape(-1)
    if m.size != 4 or c.size != 5 or q.size != 3:
        raise ValueError("model, contract and numerics must have sizes 4, 5, 3")
    k, sigma, theta, x0 = float(m[0]), float(m[1]), float(m[2]), float(m[3])
    dtau, rec, rate, lam = float(c[0]), float(c[2]), float(c[3]), float(c[4])
    nper = int(round(float(c[1])))
    nq, span = int(round(float(q[0]))), float(q[1])
    if dtau <= 0.0:
        raise ValueError("dtau must be positive")
    if nper < 1:
        raise ValueError("nper must be at least 1")

    taus = dtau * np.arange(nper + 1)
    surv = np.empty(nper + 1)
    inten = np.empty(nper + 1)
    surv[0], inten[0] = 1.0, np.exp(x0)
    for i in range(1, nper + 1):
        T = float(taus[i])
        grid = average_point_grid(T, x0, k, sigma, theta, nq, span)
        u, w = grid[:nq], grid[nq:]
        lw = np.empty(nq)
        ce = np.empty(nq)
        aa = np.empty(nq)
        for j in range(nq):
            e = _node(float(u[j]), x0, T, k, sigma, theta, lam)
            aa[j], ce[j], lw[j] = float(e[0]), float(e[3]), float(e[4])
        val = assemble_expectation(lw, ce, aa, w)
        surv[i], inten[i] = float(val[0]), float(val[1])

    disc = np.exp(-rate * taus)
    annuity = float(np.sum(dtau * disc[1:] * surv[1:]))
    y = disc * inten
    prot = (1.0 - rec) * float(dtau * (0.5 * y[0] + y[1:-1].sum()
                                       + 0.5 * y[-1]))
    return float(1.0e4 * prot / annuity)
SCICODE_GOLD_EOF
