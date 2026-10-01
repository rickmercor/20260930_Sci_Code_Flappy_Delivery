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


def pn_hamiltonian(P, Q, eta, eps):
    # Oracle: non-spinning ADM Hamiltonian through 2PN in reduced variables (G = M = 1).
    P = np.asarray(P, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    if P.shape != Q.shape or P.size not in (2, 3):
        raise ValueError("P and Q must be 1-D arrays of equal length 2 or 3")
    if not (0.0 < eta <= 0.25):
        raise ValueError("eta must satisfy 0 < eta <= 0.25")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    r = float(np.sqrt(Q @ Q))
    if r == 0.0:
        raise ValueError("separation |Q| must be positive")
    p2 = float(P @ P)
    s = float(Q @ P) / r
    H_N = 0.5 * p2 - 1.0 / r
    H_1 = ((3.0 * eta - 1.0) / 8.0 * p2 ** 2
           - 0.5 * ((3.0 + eta) * p2 + eta * s ** 2) / r
           + 0.5 / r ** 2)
    c1 = 5.0 - 20.0 * eta - 3.0 * eta ** 2
    H_2 = ((1.0 - 5.0 * eta + 5.0 * eta ** 2) / 16.0 * p2 ** 3
           + (c1 * p2 ** 2 - 2.0 * eta ** 2 * s ** 2 * p2 - 3.0 * eta ** 2 * s ** 4) / (8.0 * r)
           + 0.5 * ((5.0 + 8.0 * eta) * p2 + 3.0 * eta * s ** 2) / r ** 2
           - (1.0 + 3.0 * eta) / (4.0 * r ** 3))
    return float(H_N + eps * H_1 + eps ** 2 * H_2)

import numpy as np


def pn_perturbation_gradients(P, Q, eta, eps):
    # Oracle: analytic gradients of H_1 = H_1PN + eps*H_2PN written through (p2, s, r),
    # p2 = P.P, s = N.P, r = |Q|; chain rule dp2/dP = 2P, ds/dP = N, dr/dQ = N, ds/dQ = (P - sN)/r.
    P = np.asarray(P, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    if P.shape != Q.shape or P.size not in (2, 3):
        raise ValueError("P and Q must be 1-D arrays of equal length 2 or 3")
    if not (0.0 < eta <= 0.25):
        raise ValueError("eta must satisfy 0 < eta <= 0.25")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    r = float(np.sqrt(Q @ Q))
    if r == 0.0:
        raise ValueError("separation |Q| must be positive")
    N = Q / r
    p2 = float(P @ P)
    s = float(N @ P)
    a1 = (3.0 * eta - 1.0) / 8.0
    b1 = (1.0 - 5.0 * eta + 5.0 * eta ** 2) / 16.0
    c1 = 5.0 - 20.0 * eta - 3.0 * eta ** 2
    e2 = eta ** 2
    d_p2 = 2.0 * a1 * p2 - (3.0 + eta) / (2.0 * r)
    d_s = -eta * s / r
    d_r = ((3.0 + eta) * p2 + eta * s ** 2) / (2.0 * r ** 2) - 1.0 / r ** 3
    d_p2 += eps * (3.0 * b1 * p2 ** 2 + (2.0 * c1 * p2 - 2.0 * e2 * s ** 2) / (8.0 * r)
                   + (5.0 + 8.0 * eta) / (2.0 * r ** 2))
    d_s += eps * ((-4.0 * e2 * s * p2 - 12.0 * e2 * s ** 3) / (8.0 * r) + 3.0 * eta * s / r ** 2)
    d_r += eps * (-(c1 * p2 ** 2 - 2.0 * e2 * s ** 2 * p2 - 3.0 * e2 * s ** 4) / (8.0 * r ** 2)
                  - ((5.0 + 8.0 * eta) * p2 + 3.0 * eta * s ** 2) / r ** 3
                  + 3.0 * (1.0 + 3.0 * eta) / (4.0 * r ** 4))
    g = 2.0 * d_p2 * P + d_s * N
    f = -(d_r * N + d_s * (P - s * N) / r)
    return np.concatenate([f, g])

import numpy as np


def kepler_flow(P, Q, t):
    # Oracle: universal-variable Kepler propagator (mu = 1) with Lagrange f, g coefficients.
    # The universal Kepler equation F(chi) = t has dF/dchi = r(chi) > 0, so F is strictly
    # increasing; the root is bracketed first and then refined by Newton steps safeguarded
    # with bisection, which converges for elliptic, parabolic and hyperbolic arcs and any sign of t.
    def stumpff(z):
        if z > 1e-6:
            sz = np.sqrt(z)
            return (1.0 - np.cos(sz)) / z, (sz - np.sin(sz)) / sz ** 3
        if z < -1e-6:
            sz = np.sqrt(-z)
            return (np.cosh(sz) - 1.0) / (-z), (np.sinh(sz) - sz) / sz ** 3
        return (0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0,
                1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0)

    v0 = np.array(P, dtype=float).ravel()
    x0 = np.array(Q, dtype=float).ravel()
    if v0.shape != x0.shape or v0.size not in (2, 3):
        raise ValueError("P and Q must be 1-D arrays of equal length 2 or 3")
    r0 = float(np.sqrt(x0 @ x0))
    if r0 == 0.0:
        raise ValueError("separation |Q| must be positive")
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    if t == 0.0:
        return np.concatenate([v0, x0])
    sigma0 = float(x0 @ v0)
    alpha = 2.0 / r0 - float(v0 @ v0)

    def kepler_residual(chi):
        z = alpha * chi * chi
        if z < -1.0e4:
            # hyperbolic overshoot: cosh/sinh would overflow, and F is far beyond t in the sign of chi
            return np.copysign(np.inf, chi), np.inf
        C, S = stumpff(z)
        F = sigma0 * chi * chi * C + (1.0 - alpha * r0) * chi ** 3 * S + r0 * chi - t
        dF = sigma0 * chi * (1.0 - z * S) + (1.0 - alpha * r0) * chi * chi * C + r0
        return F, dF

    # bracket the root: F(0) = -t, and F is increasing
    step = t / r0
    if t > 0.0:
        lo, hi = 0.0, step
        while kepler_residual(hi)[0] < 0.0:
            lo, hi = hi, 2.0 * hi
    else:
        lo, hi = step, 0.0
        while kepler_residual(lo)[0] > 0.0:
            lo, hi = 2.0 * lo, lo
    chi = 0.5 * (lo + hi)
    for _ in range(300):
        F, dF = kepler_residual(chi)
        if F > 0.0:
            hi = chi
        else:
            lo = chi
        new = chi - F / dF if np.isfinite(F) and np.isfinite(dF) else 0.5 * (lo + hi)
        if not (lo < new < hi):
            new = 0.5 * (lo + hi)
        if abs(new - chi) <= 4e-16 * max(1.0, abs(new)):
            chi = new
            break
        chi = new
    z = alpha * chi * chi
    C, S = stumpff(z)
    f = 1.0 - chi * chi / r0 * C
    g = t - chi ** 3 * S
    x1 = f * x0 + g * v0
    r1 = float(np.sqrt(x1 @ x1))
    fdot = chi / (r1 * r0) * (z * S - 1.0)
    gdot = 1.0 - chi * chi / r1 * C
    v1 = fdot * x0 + gdot * v0
    return np.concatenate([v1, x1])

import numpy as np


def doubled_flow_A(z, tau, fg):
    # Oracle: under eps*H_1(p, y) the pair (p, y) is frozen, so the flow is an exact linear drift:
    # q <- q + tau * g(p, y),  x <- x + tau * f(p, y).
    z = np.array(z, dtype=float).ravel()
    fg = np.asarray(fg, dtype=float).ravel()
    if z.size not in (8, 12):
        raise ValueError("z must have length 8 (planar) or 12 (3-D)")
    d = z.size // 4
    if fg.size != 2 * d:
        raise ValueError("fg must have length len(z)//2")
    f, g = fg[:d], fg[d:]
    out = z.copy()
    out[d:2 * d] = z[d:2 * d] + tau * g
    out[2 * d:3 * d] = z[2 * d:3 * d] + tau * f
    return out

import numpy as np


def doubled_flow_B(z, tau, fg):
    # Oracle: under eps*H_1(x, q) the pair (x, q) is frozen, so the flow is an exact linear kick:
    # p <- p + tau * f(x, q),  y <- y + tau * g(x, q).
    z = np.array(z, dtype=float).ravel()
    fg = np.asarray(fg, dtype=float).ravel()
    if z.size not in (8, 12):
        raise ValueError("z must have length 8 (planar) or 12 (3-D)")
    d = z.size // 4
    if fg.size != 2 * d:
        raise ValueError("fg must have length len(z)//2")
    f, g = fg[:d], fg[d:]
    out = z.copy()
    out[0:d] = z[0:d] + tau * f
    out[3 * d:4 * d] = z[3 * d:4 * d] + tau * g
    return out

import numpy as np


def doubled_projection(z, n, lam0, mu0):
    # Oracle: Liang & Mei projection P (their Eq. 2.6) with parity-alternating weights.
    z = np.asarray(z, dtype=float).ravel()
    if z.size not in (8, 12):
        raise ValueError("z must have length 8 (planar) or 12 (3-D)")
    if int(n) != n or int(n) < 1:
        raise ValueError("step number n must be an integer >= 1")
    if not (0.0 < lam0 < 1.0 and 0.0 < mu0 < 1.0):
        raise ValueError("lam0 and mu0 must lie in (0, 1)")
    d = z.size // 4
    p, q, x, y = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]
    if int(n) % 2 == 1:
        a, b = lam0, mu0
    else:
        a, b = mu0, lam0
    pt = a * p + (1.0 - a) * x
    qt = b * q + (1.0 - b) * y
    return np.concatenate([pt, qt, pt, qt])

import numpy as np


def ds4_schedule(h, eps):
    # Oracle: Yoshida triple-jump of the symmetric map C(h/2) B(eps h/2) A(eps h) B(eps h/2) C(h/2)
    # with weights (lam, 1 - 2 lam, lam), lam = 1/(2 - 2**(1/3)); adjacent C flows merged.
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    lam = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w = [lam, 1.0 - 2.0 * lam, lam]
    out = [w[0] * h / 2.0]
    for i, wi in enumerate(w):
        out.extend([wi * eps * h / 2.0, wi * eps * h, wi * eps * h / 2.0])
        nxt = w[i + 1] if i < 2 else 0.0
        out.append((wi + nxt) * h / 2.0)
    return np.array(out, dtype=float)

import numpy as np


def run_ds4_pn_energy_error(beta, eps, P0, Q0, h, n_steps, lam0, mu0):
    # Oracle (orchestrator): chains the seven earlier sub-problem oracles (-prefixed twins).
    if beta <= 0.0:
        raise ValueError("mass ratio beta must be positive")
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    if int(n_steps) != n_steps or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    eta = beta / (1.0 + beta) ** 2
    P0 = np.asarray(P0, dtype=float).ravel()
    Q0 = np.asarray(Q0, dtype=float).ravel()
    d = P0.size
    H_start = pn_hamiltonian(P0, Q0, eta, eps)
    sched = ds4_schedule(h, eps)
    ops = "CBABCBABCBABC"
    z = np.concatenate([P0, Q0, P0, Q0])
    for n in range(1, int(n_steps) + 1):
        for op, dt in zip(ops, sched):
            p, q, x, y = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]
            if op == "C":
                a = kepler_flow(p, q, dt)
                b = kepler_flow(x, y, dt)
                z = np.concatenate([a[:d], a[d:], b[:d], b[d:]])
            elif op == "A":
                z = doubled_flow_A(z, dt, pn_perturbation_gradients(p, y, eta, eps))
            else:
                z = doubled_flow_B(z, dt, pn_perturbation_gradients(x, q, eta, eps))
        z = doubled_projection(z, n, lam0, mu0)
    H_end = pn_hamiltonian(z[:d], z[d:2 * d], eta, eps)
    return float(abs(H_end - H_start) / abs(H_start))
SCICODE_GOLD_EOF
