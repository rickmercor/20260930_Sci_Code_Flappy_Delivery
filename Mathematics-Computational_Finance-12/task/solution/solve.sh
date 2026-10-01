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
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def _h_integrate(ws, wt, dens, ys, bs_pos, bt_pos, bs_neg, bt_neg):
    """Integrate over the whole real line.

    bs_* are the integrand on the small nodes and are divided by y**2 here,
    because the y**(1 - a_ts) part of the kernel sits in the Jacobi weight.
    bt_* are the integrand on the tail nodes, used directly.
    """
    return (
        np.sum(ws * dens["sp"] * bs_pos / ys ** 2)
        + np.sum(wt * dens["tp"] * bt_pos)
        + np.sum(ws * dens["sm"] * bs_neg / ys ** 2)
        + np.sum(wt * dens["tm"] * bt_neg)
    )


def levy_shape_integrals(p: float, M: float, G: float, a_ts: float,
                                 a_exc: float) -> 'np.ndarray':
    """Reference implementation."""
    if not (0.0 < float(p) < 1.0):
        raise ValueError("p must lie in (0,1)")
    if float(M) <= 0.0 or float(G) <= 0.0:
        raise ValueError("M and G must be positive")
    if not (0.0 < float(a_ts) < 2.0):
        raise ValueError("a_ts must lie in (0,2)")
    if float(a_exc) <= 0.0:
        raise ValueError("a_exc must be positive")
    if float(M) <= 1.0:
        raise ValueError(
            "M must exceed 1 for a finite drift-restriction integral")

    ys, ws, yt, wt = _h_nodes(float(a_ts))
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)

    g_small = -np.expm1(-float(a_exc) * ys ** 2)
    g_tail = -np.expm1(-float(a_exc) * yt ** 2)
    mean_exc = _h_integrate(ws, wt, dens, ys,
                            g_small, g_tail, g_small, g_tail)

    drift = _h_integrate(
        ws, wt, dens, ys,
        np.expm1(ys) - ys, np.exp(yt) - 1.0,
        np.expm1(-ys) + ys, np.exp(-yt) - 1.0)

    entropy = _h_integrate(
        ws, wt, dens, ys,
        ys * np.expm1(ys) - (np.expm1(ys) - ys),
        yt * np.exp(yt) - np.exp(yt) + 1.0,
        -ys * np.expm1(-ys) - (np.expm1(-ys) + ys),
        -yt * np.exp(-yt) - np.exp(-yt) + 1.0)

    return np.array([mean_exc, drift, entropy], dtype=float)

import math

import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def _h_rhs_activity(u_arg, psi, chiJ, kappa, eta, ys, yt, dens,
                    w_sp, w_tp, w_sm, w_tm):
    """Right-hand side of the activity coefficient equation, vectorised."""
    iu = 1j * u_arg
    e_sp = np.exp(np.multiply.outer(iu, ys)
                  + eta * np.multiply.outer(psi, dens["gs"]))
    e_tp = np.exp(np.multiply.outer(iu, yt)
                  + eta * np.multiply.outer(psi, dens["gt"]))
    e_sm = np.exp(np.multiply.outer(-iu, ys)
                  + eta * np.multiply.outer(psi, dens["gs"]))
    e_tm = np.exp(np.multiply.outer(-iu, yt)
                  + eta * np.multiply.outer(psi, dens["gt"]))
    jump = (np.einsum("ij,j->i", e_sp - 1.0 - np.multiply.outer(iu, ys), w_sp)
            + np.einsum("ij,j->i", e_tp - 1.0, w_tp)
            + np.einsum("ij,j->i", e_sm - 1.0 + np.multiply.outer(iu, ys), w_sm)
            + np.einsum("ij,j->i", e_tm - 1.0, w_tm))
    return -kappa * psi - iu * chiJ + jump


def _h_rhs_constant(u_arg, psi, kappa, lam_bar, r, sigma):
    """Right-hand side of the constant coefficient equation."""
    return (1j * u_arg * (r - 0.5 * sigma ** 2)
            - 0.5 * sigma ** 2 * u_arg * u_arg
            + kappa * lam_bar * psi)


def activity_riccati_transform(u_line: 'np.ndarray', alpha: float,
                                       T: float, chiJ: float, p: float,
                                       M: float, G: float, a_ts: float,
                                       a_exc: float, kappa: float,
                                       lam_bar: float, eta: float,
                                       lam0: float, r: float, sigma: float,
                                       S0: float) -> 'np.ndarray':
    """Reference implementation."""
    u = np.asarray(u_line, dtype=float)
    if u.ndim != 1 or u.size < 1:
        raise ValueError("u_line must be a non-empty one-dimensional array")
    if np.any(u < 0.0):
        raise ValueError("u_line entries must be non-negative")
    if u.size > 1 and np.any(np.diff(u) <= 0.0):
        raise ValueError("u_line must be strictly increasing")
    if float(T) <= 0.0:
        raise ValueError("T must be positive")
    if float(lam0) < 0.0:
        raise ValueError("lam0 must be non-negative")
    if float(sigma) < 0.0:
        raise ValueError("sigma must be non-negative")
    if float(S0) <= 0.0:
        raise ValueError("S0 must be positive")

    ys, ws, yt, wt = _h_nodes(float(a_ts))
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)
    w_sp = ws * dens["sp"] / ys ** 2
    w_tp = wt * dens["tp"]
    w_sm = ws * dens["sm"] / ys ** 2
    w_tm = wt * dens["tm"]

    u_arg = -u - 1j * float(alpha)
    n_steps = 200
    h = float(T) / n_steps
    psi = np.zeros_like(u_arg)
    phi = np.zeros_like(u_arg)

    for _ in range(n_steps):
        k1p = _h_rhs_activity(u_arg, psi, chiJ, kappa, eta, ys, yt, dens,
                              w_sp, w_tp, w_sm, w_tm)
        k1c = _h_rhs_constant(u_arg, psi, kappa, lam_bar, r, sigma)
        k2p = _h_rhs_activity(u_arg, psi + 0.5 * h * k1p, chiJ, kappa, eta,
                              ys, yt, dens, w_sp, w_tp, w_sm, w_tm)
        k2c = _h_rhs_constant(u_arg, psi + 0.5 * h * k1p, kappa, lam_bar,
                              r, sigma)
        k3p = _h_rhs_activity(u_arg, psi + 0.5 * h * k2p, chiJ, kappa, eta,
                              ys, yt, dens, w_sp, w_tp, w_sm, w_tm)
        k3c = _h_rhs_constant(u_arg, psi + 0.5 * h * k2p, kappa, lam_bar,
                              r, sigma)
        k4p = _h_rhs_activity(u_arg, psi + h * k3p, chiJ, kappa, eta,
                              ys, yt, dens, w_sp, w_tp, w_sm, w_tm)
        k4c = _h_rhs_constant(u_arg, psi + h * k3p, kappa, lam_bar, r, sigma)
        phi = phi + h / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        psi = psi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        if not np.all(np.isfinite(psi)) or np.max(psi.real) > 50.0:
            raise ValueError(
                "coefficient system unbounded: damping level outside the "
                "admissible strip")

    value = np.exp(1j * u_arg * math.log(float(S0)) + phi + psi * float(lam0))
    if not np.all(np.isfinite(value)):
        raise ValueError(
            "coefficient system unbounded: damping level outside the "
            "admissible strip")
    return np.vstack([value.real, value.imag])

import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def moment_growth_driver(psi_grid: 'np.ndarray', alpha: float,
                                 chiJ: float, p: float, M: float, G: float,
                                 a_ts: float, a_exc: float, kappa: float,
                                 eta: float) -> 'np.ndarray':
    """Reference implementation."""
    xs = np.asarray(psi_grid, dtype=float)
    if xs.ndim != 1 or xs.size < 1:
        raise ValueError("psi_grid must be a non-empty one-dimensional array")
    if np.any(xs < 0.0):
        raise ValueError("psi_grid entries must be non-negative")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
    if float(eta) < 0.0:
        raise ValueError("eta must be non-negative")

    a = float(alpha)
    ys, ws, yt, wt = _h_nodes(float(a_ts), n_tail=224, y_cut=20.0)
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)
    w_sp = ws * dens["sp"] / ys ** 2
    w_tp = wt * dens["tp"]
    w_sm = ws * dens["sm"] / ys ** 2
    w_tm = wt * dens["tm"]

    exc_s = float(eta) * np.multiply.outer(xs, dens["gs"])
    exc_t = float(eta) * np.multiply.outer(xs, dens["gt"])
    jump = ((np.exp(a * ys + exc_s) - 1.0 - a * ys) @ w_sp
            + (np.exp(a * yt + exc_t) - 1.0) @ w_tp
            + (np.exp(-a * ys + exc_s) - 1.0 + a * ys) @ w_sm
            + (np.exp(-a * yt + exc_t) - 1.0) @ w_tm)
    return -float(kappa) * xs - a * float(chiJ) + jump

import math

import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def _h_make_driver(chiJ, kappa, eta, ys, yt, dens, w_sp, w_tp, w_sm, w_tm):
    """Return the scalar driver as a function of (x, alpha), vectorised in x."""

    def _driver(x, a):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        exc_s = eta * np.multiply.outer(xa, dens["gs"])
        exc_t = eta * np.multiply.outer(xa, dens["gt"])
        jump = ((np.exp(a * ys + exc_s) - 1.0 - a * ys) @ w_sp
                + (np.exp(a * yt + exc_t) - 1.0) @ w_tp
                + (np.exp(-a * ys + exc_s) - 1.0 + a * ys) @ w_sm
                + (np.exp(-a * yt + exc_t) - 1.0) @ w_tm)
        out = -kappa * xa - a * chiJ + jump
        return out if np.ndim(x) else float(out[0])

    return _driver


def _h_blowup_time(driver, a, eta, x_cap=18.0, n_scan=4001):
    """Time at which the activity coefficient escapes, infinite if it cannot."""
    xs = np.linspace(0.0, x_cap, n_scan)
    fv = driver(xs, a)
    if fv.min() <= 0.0:
        return np.inf
    inv = 1.0 / fv
    h = xs[1] - xs[0]
    lower = h / 3.0 * (inv[0] + inv[-1]
                       + 4.0 * inv[1:-1:2].sum() + 2.0 * inv[2:-2:2].sum())
    xg, wg = roots_legendre(64)
    t_max = math.exp(-eta * x_cap)
    xt = 0.5 * t_max * xg + 0.5 * t_max
    wt_ = 0.5 * t_max * wg
    upper = float(np.sum(wt_ / (eta * xt * driver(-np.log(xt) / eta, a))))
    return lower + upper


def transform_strip_boundary(T: float, chiJ: float, p: float,
                                     M: float, G: float, a_ts: float,
                                     a_exc: float, kappa: float,
                                     eta: float) -> float:
    """Reference implementation."""
    if float(T) <= 0.0:
        raise ValueError("T must be positive")
    if float(M) <= 1.0:
        raise ValueError("M must exceed 1")
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
    if float(eta) < 0.0:
        raise ValueError("eta must be non-negative")

    ys, ws, yt, wt = _h_nodes(float(a_ts), n_tail=224, y_cut=20.0)
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)
    driver = _h_make_driver(float(chiJ), float(kappa), float(eta), ys, yt,
                            dens,
                            ws * dens["sp"] / ys ** 2, wt * dens["tp"],
                            ws * dens["sm"] / ys ** 2, wt * dens["tm"])

    lo = 1.0 + 1e-6
    hi = float(M) - 1e-6

    if float(eta) == 0.0:
        # the driver is affine and decreasing, so it always has a root and the
        # moment is finite at every exponent the search considers
        return float(hi)
    if _h_blowup_time(driver, hi, float(eta)) == np.inf:
        return float(hi)

    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if _h_blowup_time(driver, mid, float(eta)) > float(T):
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

import math

import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def _h_exponential_moment(alpha, T, chiJ, p, M, G, a_ts, a_exc, kappa,
                          lam_bar, eta, lam0, r, sigma, S0):
    """Exponential moment of the terminal price at the given exponent."""
    ys, ws, yt, wt = _h_nodes(a_ts)
    dens = _h_density(p, M, G, a_ts, a_exc, ys, yt)
    w_sp = ws * dens["sp"] / ys ** 2
    w_tp = wt * dens["tp"]
    w_sm = ws * dens["sm"] / ys ** 2
    w_tm = wt * dens["tm"]
    u_arg = np.array([-1j * float(alpha)])

    def _rhs_activity(psi):
        iu = 1j * u_arg
        e_sp = np.exp(np.multiply.outer(iu, ys)
                      + eta * np.multiply.outer(psi, dens["gs"]))
        e_tp = np.exp(np.multiply.outer(iu, yt)
                      + eta * np.multiply.outer(psi, dens["gt"]))
        e_sm = np.exp(np.multiply.outer(-iu, ys)
                      + eta * np.multiply.outer(psi, dens["gs"]))
        e_tm = np.exp(np.multiply.outer(-iu, yt)
                      + eta * np.multiply.outer(psi, dens["gt"]))
        jump = ((e_sp - 1.0 - np.multiply.outer(iu, ys)) @ w_sp
                + (e_tp - 1.0) @ w_tp
                + (e_sm - 1.0 + np.multiply.outer(iu, ys)) @ w_sm
                + (e_tm - 1.0) @ w_tm)
        return -kappa * psi - iu * chiJ + jump

    def _rhs_constant(psi):
        return (1j * u_arg * (r - 0.5 * sigma ** 2)
                - 0.5 * sigma ** 2 * u_arg * u_arg + kappa * lam_bar * psi)

    n_steps = 4000
    h = float(T) / n_steps
    psi = np.zeros_like(u_arg)
    phi = np.zeros_like(u_arg)
    for _ in range(n_steps):
        k1p, k1c = _rhs_activity(psi), _rhs_constant(psi)
        k2p = _rhs_activity(psi + 0.5 * h * k1p)
        k2c = _rhs_constant(psi + 0.5 * h * k1p)
        k3p = _rhs_activity(psi + 0.5 * h * k2p)
        k3c = _rhs_constant(psi + 0.5 * h * k2p)
        k4p = _rhs_activity(psi + h * k3p)
        k4c = _rhs_constant(psi + h * k3p)
        phi = phi + h / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        psi = psi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        if not np.all(np.isfinite(psi)) or np.max(psi.real) > 50.0:
            raise ValueError(
                "damping level lies outside the admissible strip")
    value = np.exp(1j * u_arg * math.log(float(S0)) + phi + psi * float(lam0))
    if not np.all(np.isfinite(value)):
        raise ValueError("damping level lies outside the admissible strip")
    return float(abs(value[0]))


def damping_parameter(alpha_expl: float, frac: float, K: float,
                              T: float, chiJ: float, p: float, M: float,
                              G: float, a_ts: float, a_exc: float,
                              kappa: float, lam_bar: float, eta: float,
                              lam0: float, r: float, sigma: float,
                              S0: float) -> 'np.ndarray':
    """Reference implementation."""
    if not (0.0 < float(frac) < 1.0):
        raise ValueError("frac must lie in (0,1)")
    if float(alpha_expl) <= 1.0:
        raise ValueError("alpha_expl must exceed 1")
    if float(K) <= 0.0:
        raise ValueError("K must be positive")

    alpha = 1.0 + float(frac) * (float(alpha_expl) - 1.0)
    payoff_mass = abs(float(K) ** (1.0 - alpha) / (alpha * (alpha - 1.0)))
    density_mass = _h_exponential_moment(
        alpha, float(T), float(chiJ), float(p), float(M), float(G),
        float(a_ts), float(a_exc), float(kappa), float(lam_bar), float(eta),
        float(lam0), float(r), float(sigma), float(S0))
    return np.array([alpha, payoff_mass * density_mass], dtype=float)

import numpy as np


def _h_frequency_grid(m, NQ):
    """Symmetric frequency grid and trapezoidal weights at resolution m."""
    half_width = 2.0 ** m * np.pi
    nodes = np.linspace(-half_width, half_width, int(NQ))
    step = nodes[1] - nodes[0]
    weights = np.full(int(NQ), step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def _h_damped_call_transform(nodes, alpha, K):
    """Fourier transform of the damped call payoff on the frequency grid."""
    s = float(alpha) + 1j * nodes
    return float(K) ** (1.0 - s) / (s * (s - 1.0))


def payoff_coefficients(alpha: float, m: int, lam_lo: int,
                                lam_hi: int, NQ: int,
                                K: float) -> 'np.ndarray':
    """Reference implementation."""
    if int(lam_lo) > int(lam_hi):
        raise ValueError("lam_lo must not exceed lam_hi")
    if int(NQ) < 3 or int(NQ) % 2 == 0:
        raise ValueError("NQ must be an odd integer of at least 3")
    if float(alpha) <= 1.0:
        raise ValueError(
            "alpha must exceed 1 for an integrable damped call payoff")
    if float(K) <= 0.0:
        raise ValueError("K must be positive")

    nodes, weights = _h_frequency_grid(int(m), int(NQ))
    transform = _h_damped_call_transform(nodes, float(alpha), float(K))
    indices = np.arange(int(lam_lo), int(lam_hi) + 1)
    phase = np.exp(1j * np.outer(indices / 2.0 ** int(m), nodes))
    scale = 2.0 ** (-int(m) / 2.0) / (2.0 * np.pi)
    return scale * np.real(phase @ (weights * transform))

import numpy as np


def _h_frequency_grid(m, NQ):
    """Symmetric frequency grid and trapezoidal weights at resolution m."""
    half_width = 2.0 ** m * np.pi
    nodes = np.linspace(-half_width, half_width, int(NQ))
    step = nodes[1] - nodes[0]
    weights = np.full(int(NQ), step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def _h_mirror(packed):
    """Rebuild the full-line transform from its non-negative half."""
    half = packed[0] + 1j * packed[1]
    return np.concatenate([np.conjugate(half[:0:-1]), half])


def density_coefficients(transform_packed: 'np.ndarray', alpha: float,
                                 m: int, lam_lo: int, lam_hi: int,
                                 NQ: int) -> 'np.ndarray':
    """Reference implementation."""
    packed = np.asarray(transform_packed, dtype=float)
    if packed.ndim != 2 or packed.shape[0] != 2:
        raise ValueError("transform_packed must have exactly two rows")
    if int(NQ) < 3 or int(NQ) % 2 == 0:
        raise ValueError("NQ must be an odd integer of at least 3")
    if packed.shape[1] != (int(NQ) + 1) // 2:
        raise ValueError(
            "transform_packed column count inconsistent with NQ")
    if int(lam_lo) > int(lam_hi):
        raise ValueError("lam_lo must not exceed lam_hi")

    nodes, weights = _h_frequency_grid(int(m), int(NQ))
    transform = _h_mirror(packed)
    indices = np.arange(int(lam_lo), int(lam_hi) + 1)
    phase = np.exp(1j * np.outer(indices / 2.0 ** int(m), nodes))
    scale = 2.0 ** (-int(m) / 2.0) / (2.0 * np.pi)
    return scale * np.real(phase @ (weights * transform))

import math

import numpy as np


def damped_swift_price(cfg: dict) -> float:
    """Reference implementation."""
    c = cfg

    constants = levy_shape_integrals(c["p"], c["M"], c["G"],
                                             c["a_ts"], c["a_exc"])
    chi_j = float(constants[1])
    if not c["eta"] * float(constants[0]) < c["kappa"]:
        raise ValueError("mean-subcriticality condition violated")

    alpha_expl = transform_strip_boundary(
        c["T"], chi_j, c["p"], c["M"], c["G"], c["a_ts"], c["a_exc"],
        c["kappa"], c["eta"])

    selection = damping_parameter(
        alpha_expl, c["frac"], c["K"], c["T"], chi_j, c["p"], c["M"], c["G"],
        c["a_ts"], c["a_exc"], c["kappa"], c["lam_bar"], c["eta"], c["lam0"],
        c["r"], c["sigma"], c["S0"])
    alpha = float(selection[0])

    driver = moment_growth_driver(
        np.linspace(0.0, 20.0, 201), alpha, chi_j, c["p"], c["M"], c["G"],
        c["a_ts"], c["a_exc"], c["kappa"], c["eta"])
    explosive = alpha_expl < c["M"] - 1e-3
    if (not explosive) and float(driver.min()) > 0.0:
        raise ValueError(
            "driver is strictly positive at the selected damping level "
            "although the boundary equals the tempering rate")

    half_grid = np.linspace(0.0, 2.0 ** c["m"] * np.pi, (c["NQ"] + 1) // 2)
    packed = activity_riccati_transform(
        half_grid, alpha, c["T"], chi_j, c["p"], c["M"], c["G"], c["a_ts"],
        c["a_exc"], c["kappa"], c["lam_bar"], c["eta"], c["lam0"], c["r"],
        c["sigma"], c["S0"])

    payoff = payoff_coefficients(alpha, c["m"], c["lam_lo"],
                                         c["lam_hi"], c["NQ"], c["K"])
    density = density_coefficients(packed, alpha, c["m"], c["lam_lo"],
                                           c["lam_hi"], c["NQ"])

    return float(math.exp(-c["r"] * c["T"]) * np.dot(density, payoff))
SCICODE_GOLD_EOF
