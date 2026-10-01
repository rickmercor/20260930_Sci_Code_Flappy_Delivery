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


def riccati_interval_advance(b0_re, b0_im, u1_re, u1_im, xi, rho, kappa, s):
    xi, rho, kappa, s = float(xi), float(rho), float(kappa), float(s)
    if xi <= 0.0:
        raise ValueError("xi must be positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must lie in [-1, 1]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if s < 0.0:
        raise ValueError("s must be non-negative")
    b0 = np.asarray(b0_re, dtype=float) + 1j * np.asarray(b0_im, dtype=float)
    u1 = np.asarray(u1_re, dtype=float) + 1j * np.asarray(u1_im, dtype=float)
    b0, u1 = np.broadcast_arrays(b0, u1)
    if s == 0.0:
        zero = np.zeros(b0.shape)
        return np.array([b0.real, b0.imag, zero, zero])
    aa = 0.5 * xi * xi
    bb = 1j * u1 * rho * xi - kappa
    cc = -0.5 * (u1 * u1 + 1j * u1)
    d = np.sqrt(bb * bb - 4.0 * aa * cc + 0j)
    lam1 = (-bb + d) / (xi * xi)
    lam2 = (-bb - d) / (xi * xi)
    R = (b0 - lam2) / (b0 - lam1)
    e = np.exp(-d * s)
    B = (lam2 - lam1 * R * e) / (1.0 - R * e)
    I = lam2 * s - (2.0 / (xi * xi)) * np.log((1.0 - R * e) / (1.0 - R))
    return np.array([B.real, B.imag, I.real, I.imag])

import numpy as np


def characteristic_exponents(u1_re, u1_im, b0_re, b0_im, t, T, model, term):
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    if term.size != 13:
        raise ValueError("term must have thirteen entries")
    t, T = float(t), float(T)
    if T < t:
        raise ValueError("T must not precede t")
    kappa = float(model[0])
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    edge = term[0:4]
    xis, rhos, thetas = term[4:7], term[7:10], term[10:13]
    if np.any(np.diff(edge) <= 0.0):
        raise ValueError("interval edges must be strictly increasing")
    if np.any(xis <= 0.0):
        raise ValueError("every volatility of variance must be positive")

    def rate_integral(level, amp, decay, lo, hi):
        if abs(decay) < 1.0e-12:
            return (level + amp) * (hi - lo)
        return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))

    u1 = np.asarray(u1_re, dtype=float) + 1j * np.asarray(u1_im, dtype=float)
    B = np.asarray(b0_re, dtype=float) + 1j * np.asarray(b0_im, dtype=float)
    u1, B = np.broadcast_arrays(u1, B)
    B = np.array(B, dtype=complex)
    A = np.zeros(B.shape, dtype=complex)
    segs = []
    for n in range(3):
        lo, hi = max(float(edge[n]), t), min(float(edge[n + 1]), T)
        if hi > lo + 1.0e-14:
            segs.append((lo, hi, float(xis[n]), float(rhos[n]), float(thetas[n])))
    for (lo, hi, xi, rho, theta) in segs[::-1]:
        r = riccati_interval_advance(B.real, B.imag, u1.real, u1.imag,
                                             xi, rho, kappa, hi - lo)
        carry = (rate_integral(model[2], model[3], model[4], lo, hi)
                 - rate_integral(model[5], model[6], model[7], lo, hi))
        A = A + 1j * u1 * carry + kappa * theta * (r[2] + 1j * r[3])
        B = r[0] + 1j * r[1]
    return np.array([A.real, A.imag, B.real, B.imag])

import numpy as np


def simpson_frequency_rule(omega_max, n):
    omega_max = float(omega_max)
    n = int(n)
    if omega_max <= 0.0:
        raise ValueError("omega_max must be positive")
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be an even integer of at least two")
    h = 2.0 * omega_max / n
    om = -omega_max + h * np.arange(n + 1)
    w = np.empty(n + 1)
    w[0] = 1.0
    w[-1] = 1.0
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return np.concatenate((om, w * h / 3.0))

import numpy as np


def truncated_payoff_transform(omega, alpha, xstar, rd, rf):
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1)
    alpha, xstar = float(alpha), float(xstar)
    rd, rf = float(rd), float(rf)
    if omega.size == 0:
        raise ValueError("omega must not be empty")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    z = 1j * omega + alpha
    g = rd * np.exp(z * xstar) / z - rf * np.exp((1.0 + z) * xstar) / (1.0 + z)
    return np.concatenate((g.real, g.imag))

import numpy as np


def exercise_surface_coefficients(u, T2, A0, B0, model):
    model = np.asarray(model, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    u, T2, A0, B0 = float(u), float(T2), float(A0), float(B0)
    if T2 <= 0.0:
        raise ValueError("T2 must be positive")
    if u > T2:
        raise ValueError("u must not exceed T2")
    rd_T2 = float(model[2] + model[3] * np.exp(-model[4] * T2))
    rf_T2 = float(model[5] + model[6] * np.exp(-model[7] * T2))
    if rd_T2 <= 0.0 or rf_T2 <= 0.0:
        raise ValueError("both rates at T2 must be positive")
    root = np.sqrt(T2 - u)
    return np.array([np.log(rd_T2 / rf_T2) - A0 * root, -B0 * root])

import numpy as np


def spot_weighted_variance_exponents(q_re, q_im, t0, model, term):
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    if term.size != 13:
        raise ValueError("term must have thirteen entries")
    t0 = float(t0)
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")

    def rate_integral(level, amp, decay, lo, hi):
        if abs(decay) < 1.0e-12:
            return (level + amp) * (hi - lo)
        return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))

    q = np.asarray(q_re, dtype=float) + 1j * np.asarray(q_im, dtype=float)
    q = np.atleast_1d(q) if np.ndim(q) else q
    shape = np.shape(q)
    e = characteristic_exponents(np.zeros(shape), -np.ones(shape), np.real(q), np.imag(q),
                                         0.0, t0, model, term)
    carry = (rate_integral(model[2], model[3], model[4], 0.0, t0)
             - rate_integral(model[5], model[6], model[7], 0.0, t0))
    return np.array([e[0] - carry, e[1], e[2], e[3]])

import numpy as np


def forward_start_kernel(omega, alpha, t0, u, a, b, model, term):
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1)
    alpha, t0, u = float(alpha), float(t0), float(u)
    a, b = float(a), float(b)
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    if omega.size == 0:
        raise ValueError("omega must not be empty")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if u < t0:
        raise ValueError("u must not precede t0")
    if model.size != 8 or term.size != 13:
        raise ValueError("model must have eight entries and term thirteen")
    m = omega.size
    v0 = float(model[1])
    rd_u = float(model[2] + model[3] * np.exp(-model[4] * u))
    rf_u = float(model[5] + model[6] * np.exp(-model[7] * u))
    g1 = truncated_payoff_transform(omega, alpha, a, rd_u, 0.0)
    g2 = truncated_payoff_transform(omega, alpha, a, 0.0, rf_u)
    leg1 = g1[:m] + 1j * g1[m:]
    leg2 = g2[:m] + 1j * g2[m:]
    z = 1j * omega + alpha
    u1 = -omega + 1j * alpha
    total = np.zeros(m, dtype=complex)
    for leg, shift in ((leg1, 0.0), (leg2, 1.0)):
        c0 = (shift + z) * b
        e = characteristic_exponents(u1.real, u1.imag, c0.real, c0.imag, t0, u, model, term)
        h = spot_weighted_variance_exponents(e[2], e[3], t0, model, term)
        total = total + leg * np.exp((e[0] + 1j * e[1]) + (h[0] + 1j * h[1])
                                     + (h[2] + 1j * h[3]) * v0)
    return np.concatenate((total.real, total.imag))

import numpy as np


def exercise_region_expectation(kernel_flat, rule_flat, alpha, x_src):
    kernel_flat = np.asarray(kernel_flat, dtype=float).reshape(-1)
    rule_flat = np.asarray(rule_flat, dtype=float).reshape(-1)
    alpha, x_src = float(alpha), float(x_src)
    if kernel_flat.size % 2 != 0 or rule_flat.size % 2 != 0:
        raise ValueError("both packed arrays must have even length")
    m = kernel_flat.size // 2
    k = rule_flat.size // 2
    if m != k:
        raise ValueError("kernel and rule must hold the same number of nodes")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    w = kernel_flat[:m] + 1j * kernel_flat[m:]
    om, ow = rule_flat[:k], rule_flat[k:]
    val = np.exp(-alpha * x_src) / (2.0 * np.pi) * np.sum(
        ow * np.exp(-1j * om * x_src) * w)
    return float(val.real)

import numpy as np


def premium_time_rule(T1, T2, nstep, t0, model):
    model = np.asarray(model, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    T1, T2, t0 = float(T1), float(T2), float(t0)
    nstep = int(nstep)
    if T2 <= T1:
        raise ValueError("T2 must exceed T1")
    if nstep < 1:
        raise ValueError("nstep must be at least one")
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if T1 < t0:
        raise ValueError("the window must not open before t0")
    level, amp, decay = float(model[2]), float(model[3]), float(model[4])
    h = (T2 - T1) / nstep
    u = T1 + h * np.arange(nstep + 1)
    if abs(decay) < 1.0e-12:
        integ = (level + amp) * (u - t0)
    else:
        integ = level * (u - t0) + (amp / decay) * (np.exp(-decay * t0) - np.exp(-decay * u))
    w = np.full(nstep + 1, h)
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.concatenate((u, w * np.exp(-integ)))

import numpy as np


def forward_start_flexible_forward_value(model, term, contract, numerics):
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    contract = np.asarray(contract, dtype=float).reshape(-1)
    numerics = np.asarray(numerics, dtype=float).reshape(-1)
    if model.size != 8 or term.size != 13:
        raise ValueError("model must have eight entries and term thirteen")
    if contract.size != 5:
        raise ValueError("contract must have five entries")
    if numerics.size != 4:
        raise ValueError("numerics must have four entries")
    t0, T1, T2, A0, B0 = [float(c) for c in contract]
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if T1 < t0:
        raise ValueError("the window must not open before t0")
    if T2 <= T1:
        raise ValueError("T2 must exceed T1")
    alpha, omega_max = float(numerics[0]), float(numerics[1])
    n_omega, nstep = int(numerics[2]), int(numerics[3])

    def rate_integral(level, amp, decay, lo, hi):
        if abs(decay) < 1.0e-12:
            return (level + amp) * (hi - lo)
        return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))

    rd_int = lambda lo, hi: rate_integral(model[2], model[3], model[4], lo, hi)
    rf_int = lambda lo, hi: rate_integral(model[5], model[6], model[7], lo, hi)
    carry = rd_int(t0, T2) - rf_int(t0, T2)
    ratio = np.exp(carry)
    x_fix = -carry
    rule = simpson_frequency_rule(omega_max, n_omega)
    om = rule[:rule.size // 2]
    tr = premium_time_rule(T1, T2, nstep, t0, model)
    half = tr.size // 2
    nodes, weights = tr[:half], tr[half:]
    prem = 0.0
    for i in range(half):
        ab = exercise_surface_coefficients(float(nodes[i]), T2, A0, B0, model)
        ker = forward_start_kernel(om, alpha, t0, float(nodes[i]),
                                           float(ab[0]), float(ab[1]), model, term)
        prem += weights[i] * exercise_region_expectation(ker, rule, alpha, x_fix)
    fixing_scale = ratio * np.exp(-rf_int(0.0, t0))
    terminal = fixing_scale * np.exp(-rd_int(t0, T2)) - np.exp(-rf_int(0.0, T2))
    return float(1.0e4 * (terminal + fixing_scale * prem))
SCICODE_GOLD_EOF
