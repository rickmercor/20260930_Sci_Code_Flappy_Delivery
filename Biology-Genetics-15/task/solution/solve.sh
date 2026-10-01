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
from functools import lru_cache
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=None)
def _cached_leggauss_nodes_weights(n_nodes: int) -> "tuple[np.ndarray, np.ndarray]":
    """Return cached Gauss-Legendre nodes and weights for the requested order."""
    x, w = leggauss(int(n_nodes))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def locus_stationary_moments(delta: float, alpha: float, dom: float, theta_plus: float, theta_minus: float,
                                     sel_ratio: float, n_nodes: int) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(delta) or not _num(dom):
        raise ValueError("delta and dom must be finite numbers")
    if not _num(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a finite number > 0")
    if not _num(theta_plus) or float(theta_plus) <= 0.0 or not _num(theta_minus) or float(theta_minus) <= 0.0:
        raise ValueError("theta_plus and theta_minus must be finite numbers > 0")
    if not _num(sel_ratio) or float(sel_ratio) < 0.0:
        raise ValueError("sel_ratio must be a finite number >= 0")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer >= 2")
    d, a, D, tp, tm, s = float(delta), float(alpha), float(dom), float(theta_plus), float(theta_minus), float(sel_ratio)
    # Wright's stationary density in the source's scaling (theta = 2 N mu, time in 2N generations, sel_ratio = 2 N / omega^2):
    #   p^(2 theta_plus - 1) (1 - p)^(2 theta_minus - 1) exp(2 int_0^p xi(u) du),
    #   xi(u) = sel_ratio [ -beta(u) delta + beta(u)^2 (u - 1/2) + 2 alpha D u (1 - u) ],  beta(u) = alpha + D (1 - 2u).
    # 2 int_0^p xi = sel_ratio * S(p), S a quartic with the coefficients below.
    c4 = 2.0 * D * D
    c3 = -4.0 * a * D - 4.0 * D * D
    c2 = a * a + 6.0 * a * D + 3.0 * D * D + 2.0 * D * d
    c1 = -a * a - 2.0 * a * D - D * D - 2.0 * a * d - 2.0 * D * d
    # Quadrature: split at 1/2; on [0, 1/2] substitute p = u^(1/A) / 2, on [1/2, 1] substitute 1 - p = v^(1/B) / 2,
    # with A = 2 theta_plus and B = 2 theta_minus, which turns each endpoint weight into a constant; Gauss-Legendre
    # on u and v in (0, 1) with n_nodes nodes each. Gauss-Jacobi rules on the raw weight lose accuracy at these exponents.
    A, B = 2.0 * tp, 2.0 * tm
    x, w = _cached_leggauss_nodes_weights(int(n_nodes))
    u = 0.5 * (x + 1.0)
    w = 0.5 * w
    p_left = 0.5 * u ** (1.0 / A)
    p_right = 1.0 - 0.5 * u ** (1.0 / B)
    log_w_left = A * np.log(0.5) - np.log(A) + (B - 1.0) * np.log1p(-p_left) + np.log(w)
    log_w_right = B * np.log(0.5) - np.log(B) + (A - 1.0) * np.log(p_right) + np.log(w)
    p = np.concatenate([p_left, p_right])
    log_w = np.concatenate([log_w_left, log_w_right])
    expo = log_w + s * (((c4 * p + c3) * p + c2) * p + c1) * p
    f = np.exp(expo - expo.max())
    z = f.sum()
    pq = p * (1.0 - p)
    beta = a + D * (1.0 - 2.0 * p)
    return np.array([(f * p).sum() / z, (f * pq).sum() / z, (f * beta * beta * pq).sum() / z, (f * pq * pq).sum() / z])

import numpy as np


def _class_arrays(alpha, dom, theta_plus, theta_minus, counts):
    arrs = []
    for name, v in (("alpha", alpha), ("dom", dom), ("theta_plus", theta_plus), ("theta_minus", theta_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        a = np.asarray(v, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        arrs.append(a)
    if isinstance(counts, (str, bytes)):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    c = np.asarray(counts)
    if c.ndim != 1 or c.size < 1 or c.dtype.kind not in "iu" or np.any(c < 1):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    if len({a.size for a in arrs} | {c.size}) != 1:
        raise ValueError("the class arrays must all have the same length")
    a, d, tp, tm = arrs
    if np.any(a <= 0.0) or np.any(tp <= 0.0) or np.any(tm <= 0.0):
        raise ValueError("alpha, theta_plus and theta_minus entries must be > 0")
    return a, d, tp, tm, c.astype(int)


def architecture_mean_and_variances(delta: float, sel_ratio: float, alpha: "np.ndarray", dom: "np.ndarray",
                                            theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray",
                                            n_nodes: int) -> "np.ndarray":
    a, d, tp, tm, c = _class_arrays(alpha, dom, theta_plus, theta_minus, counts)
    mean = va = vd = 0.0
    for k in range(a.size):
        m = locus_stationary_moments(delta, float(a[k]), float(d[k]), float(tp[k]), float(tm[k]), sel_ratio, n_nodes)
        mean += c[k] * 2.0 * (a[k] * m[0] + d[k] * m[1])       # expected genotypic value 2 alpha p + 2 D p(1-p)
        va += c[k] * 2.0 * m[2]                                # 2 p(1-p) beta(p)^2
        vd += c[k] * 4.0 * d[k] * d[k] * m[3]                  # (2 p(1-p) D)^2
    return np.array([float(mean), float(va), float(vd)])

import numpy as np
from scipy.optimize import brentq


def _class_arrays(alpha, dom, theta_plus, theta_minus, counts):
    arrs = []
    for name, v in (("alpha", alpha), ("dom", dom), ("theta_plus", theta_plus), ("theta_minus", theta_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        a = np.asarray(v, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        arrs.append(a)
    if isinstance(counts, (str, bytes)):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    c = np.asarray(counts)
    if c.ndim != 1 or c.size < 1 or c.dtype.kind not in "iu" or np.any(c < 1):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    if len({a.size for a in arrs} | {c.size}) != 1:
        raise ValueError("the class arrays must all have the same length")
    a, d, tp, tm = arrs
    if np.any(a <= 0.0) or np.any(tp <= 0.0) or np.any(tm <= 0.0):
        raise ValueError("alpha, theta_plus and theta_minus entries must be > 0")
    return a, d, tp, tm, c.astype(int)


def equilibrium_trait_deviation(sel_ratio: float, eta: float, alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray", n_nodes: int, tol: float) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))
    if not _num(eta): raise ValueError("eta must be a finite number")
    if not _num(tol) or float(tol) <= 0.0: raise ValueError("tol must be a finite number > 0")
    a, d, tp, tm, c = _class_arrays(alpha, dom, theta_plus, theta_minus, counts)
    e = float(eta)
    genotype_values = np.stack([np.zeros_like(a), a + d, 2.0 * a], axis=0)
    z_min = float(np.sum(c * np.min(genotype_values, axis=0)))
    z_max = float(np.sum(c * np.max(genotype_values, axis=0)))
    def _gap(delta):
        mean = architecture_mean_and_variances(delta, sel_ratio, a, d, tp, tm, c, n_nodes)[0]
        return mean - e - delta
    return float(brentq(_gap, z_min - e, z_max - e, xtol=float(tol), rtol=4.0 * np.finfo(float).eps, maxiter=500))

import numpy as np


def _class_arrays(alpha, dom, theta_plus, theta_minus, counts):
    arrs = []
    for name, v in (("alpha", alpha), ("dom", dom), ("theta_plus", theta_plus), ("theta_minus", theta_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        a = np.asarray(v, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        arrs.append(a)
    if isinstance(counts, (str, bytes)):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    c = np.asarray(counts)
    if c.ndim != 1 or c.size < 1 or c.dtype.kind not in "iu" or np.any(c < 1):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    if len({a.size for a in arrs} | {c.size}) != 1:
        raise ValueError("the class arrays must all have the same length")
    a, d, tp, tm = arrs
    if np.any(a <= 0.0) or np.any(tp <= 0.0) or np.any(tm <= 0.0):
        raise ValueError("alpha, theta_plus and theta_minus entries must be > 0")
    return a, d, tp, tm, c.astype(int)


def neutral_variance_components(alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray",
                                        theta_minus: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    a, d, tp, tm, c = _class_arrays(alpha, dom, theta_plus, theta_minus, counts)
    # Beta(2 theta_plus, 2 theta_minus) moments: E[p^i (1-p)^j] = prod_{r<i}(A+r) prod_{r<j}(B+r) / prod_{r<i+j}(A+B+r)
    A, B = 2.0 * tp, 2.0 * tm

    def _mom(i, j):
        num = np.ones_like(A)
        for r in range(i):
            num = num * (A + r)
        for r in range(j):
            num = num * (B + r)
        den = np.ones_like(A)
        for r in range(i + j):
            den = den * (A + B + r)
        return num / den

    e_pq, e_p2q, e_p3q, e_p2q2 = _mom(1, 1), _mom(2, 1), _mom(3, 1), _mom(2, 2)
    # beta(p) = (a + D) - 2 D p:  E[beta^2 p q] = (a+D)^2 E[pq] - 4 D (a+D) E[p^2 q] + 4 D^2 E[p^3 q]
    e_b2pq = (a + d) ** 2 * e_pq - 4.0 * d * (a + d) * e_p2q + 4.0 * d * d * e_p3q
    va = float(np.sum(c * 2.0 * e_b2pq))
    vd = float(np.sum(c * 4.0 * d * d * e_p2q2))
    return np.array([va, vd])

import numpy as np


def _class_arrays(alpha, dom, theta_plus, theta_minus, counts):
    arrs = []
    for name, v in (("alpha", alpha), ("dom", dom), ("theta_plus", theta_plus), ("theta_minus", theta_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        a = np.asarray(v, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        arrs.append(a)
    if isinstance(counts, (str, bytes)):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    c = np.asarray(counts)
    if c.ndim != 1 or c.size < 1 or c.dtype.kind not in "iu" or np.any(c < 1):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    if len({a.size for a in arrs} | {c.size}) != 1:
        raise ValueError("the class arrays must all have the same length")
    a, d, tp, tm = arrs
    if np.any(a <= 0.0) or np.any(tp <= 0.0) or np.any(tm <= 0.0):
        raise ValueError("alpha, theta_plus and theta_minus entries must be > 0")
    return a, d, tp, tm, c.astype(int)


def _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts):
    """Natural units to the source's scaling: theta = 2 N mu, with N the number of diploid individuals."""
    if isinstance(n_diploid, bool) or not isinstance(n_diploid, (int, np.integer)) or int(n_diploid) < 1:
        raise ValueError("n_diploid must be an integer >= 1")
    for name, v in (("mu_plus", mu_plus), ("mu_minus", mu_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers > 0")
        m = np.asarray(v, dtype=float)
        if m.ndim != 1 or m.size < 1 or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers > 0")
    two_n = 2.0 * int(n_diploid)
    tp = two_n * np.asarray(mu_plus, dtype=float)
    tm = two_n * np.asarray(mu_minus, dtype=float)
    return _class_arrays(alpha, dom, tp, tm, counts)


def variance_ratio_curve(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                                 dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: "np.ndarray",
                                 n_nodes: int, tol: float) -> "np.ndarray":
    a, d, tp, tm, c = _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts)
    if isinstance(omega_inv2, (str, bytes)):
        raise ValueError("omega_inv2 must be a one-dimensional array of finite numbers >= 0")
    g = np.asarray(omega_inv2, dtype=float)
    if g.ndim != 1 or g.size < 1 or not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("omega_inv2 must be a one-dimensional array of finite numbers >= 0")
    neutral = neutral_variance_components(a, d, tp, tm, c)
    s0 = float(neutral[0] + neutral[1])
    two_n = 2.0 * int(n_diploid)
    out = np.empty(g.size)
    for i, w in enumerate(g):
        sel_ratio = two_n * float(w)                    # the source's selection-drift ratio, 2 N / omega^2
        delta = equilibrium_trait_deviation(sel_ratio, eta, a, d, tp, tm, c, n_nodes, tol)
        mv = architecture_mean_and_variances(delta, sel_ratio, a, d, tp, tm, c, n_nodes)
        out[i] = (mv[1] + mv[2]) / s0
    return out

import numpy as np


def _class_arrays(alpha, dom, theta_plus, theta_minus, counts):
    arrs = []
    for name, v in (("alpha", alpha), ("dom", dom), ("theta_plus", theta_plus), ("theta_minus", theta_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        a = np.asarray(v, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
        arrs.append(a)
    if isinstance(counts, (str, bytes)):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    c = np.asarray(counts)
    if c.ndim != 1 or c.size < 1 or c.dtype.kind not in "iu" or np.any(c < 1):
        raise ValueError("counts must be a one-dimensional array of positive integers")
    if len({a.size for a in arrs} | {c.size}) != 1:
        raise ValueError("the class arrays must all have the same length")
    a, d, tp, tm = arrs
    if np.any(a <= 0.0) or np.any(tp <= 0.0) or np.any(tm <= 0.0):
        raise ValueError("alpha, theta_plus and theta_minus entries must be > 0")
    return a, d, tp, tm, c.astype(int)


def _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts):
    """Natural units to the source's scaling: theta = 2 N mu, with N the number of diploid individuals."""
    if isinstance(n_diploid, bool) or not isinstance(n_diploid, (int, np.integer)) or int(n_diploid) < 1:
        raise ValueError("n_diploid must be an integer >= 1")
    for name, v in (("mu_plus", mu_plus), ("mu_minus", mu_minus)):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers > 0")
        m = np.asarray(v, dtype=float)
        if m.ndim != 1 or m.size < 1 or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError(f"{name} must be a one-dimensional array of finite numbers > 0")
    two_n = 2.0 * int(n_diploid)
    tp = two_n * np.asarray(mu_plus, dtype=float)
    tm = two_n * np.asarray(mu_minus, dtype=float)
    return _class_arrays(alpha, dom, tp, tm, counts)


def equilibrium_state_at_strength(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                                          dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: float,
                                          n_nodes: int, tol: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(omega_inv2) or float(omega_inv2) <= 0.0:
        raise ValueError("omega_inv2 must be a finite number > 0")
    a, d, tp, tm, c = _scaled_architecture(n_diploid, mu_plus, mu_minus, alpha, dom, counts)
    sel_ratio = 2.0 * int(n_diploid) * float(omega_inv2)
    delta = equilibrium_trait_deviation(sel_ratio, eta, a, d, tp, tm, c, n_nodes, tol)
    mv = architecture_mean_and_variances(delta, sel_ratio, a, d, tp, tm, c, n_nodes)
    p_first = locus_stationary_moments(delta, float(a[0]), float(d[0]), float(tp[0]), float(tm[0]), sel_ratio, n_nodes)[0]
    # Ornstein-Uhlenbeck relaxation of the mean trait's deviation: the rate is the additive variance times the
    # selection strength per generation (the source's rate in units of 2N generations, divided by 2N)
    rate_per_generation = mv[1] * float(omega_inv2)
    return np.array([float(delta), float(mv[1]), float(mv[2] / (mv[1] + mv[2])), float(p_first), float(1.0 / rate_per_generation)])

import numpy as np


def relaxation_time_at_variance_crossing(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray",
                                                 alpha: "np.ndarray", dom: "np.ndarray", counts: "np.ndarray", eta: float,
                                                 omega_inv2_grid: "np.ndarray", n_nodes: int, tol: float) -> float:
    if isinstance(omega_inv2_grid, (str, bytes)):
        raise ValueError("omega_inv2_grid must be a one-dimensional array of finite numbers > 0")
    g = np.asarray(omega_inv2_grid, dtype=float)
    if g.ndim != 1 or g.size < 1 or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError("omega_inv2_grid must be a one-dimensional array of finite numbers > 0")
    ratio = variance_ratio_curve(n_diploid, mu_plus, mu_minus, alpha, dom, counts, eta, g, n_nodes, tol)
    below = np.flatnonzero(ratio < 1.0)
    if below.size == 0:
        raise ValueError("no grid strength gives a variance ratio strictly below 1")
    state = equilibrium_state_at_strength(n_diploid, mu_plus, mu_minus, alpha, dom, counts, eta,
                                                  float(g[below[0]]), n_nodes, tol)
    return float(state[4])
SCICODE_GOLD_EOF
