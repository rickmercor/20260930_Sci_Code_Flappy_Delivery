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
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _morse_prime(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return 2.0 * a_le * e * (1.0 - e)


def _langevin_inverse(y):
    """Inverse Langevin function L^-1(y) for 0 < y < 1 by root bracketing."""
    hi = 10.0 / (1.0 - y) + 10.0
    return brentq(lambda b: 1.0 / np.tanh(b) - 1.0 / b - y, 1e-9, hi, xtol=1e-14, rtol=4.0 * np.finfo(float).eps)


def _frc_geometry(l, phi, n_bonds):
    """Contour length R_max = N l cos(phi/2) and Kuhn length l_k = 2 l cos(phi/2) / (1 - cos phi) of a FRC."""
    return n_bonds * l * np.cos(0.5 * phi), 2.0 * l * np.cos(0.5 * phi) / (1.0 - np.cos(phi))


def _incomplete_beta_minus_one(x, a, n_nodes=80):
    """B(x; a, -1) = int_0^x t^(a-1) (1-t)^-2 dt for 0 < x < 1, by parts: x^a/(1-x) - (a-1) int_0^x t^(a-1)/(1-t) dt,
    with the remaining log-singular integral split as -ln(1-x) - int_0^x (1 - t^(a-1))/(1-t) dt (smooth, Gauss-Legendre)."""
    xs, ws = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * x * (xs + 1.0)
    smooth = np.sum(0.5 * x * ws * (1.0 - t ** (a - 1.0)) / (1.0 - t))
    return x ** a / (1.0 - x) - (a - 1.0) * (-np.log1p(-x) - smooth)


def _incomplete_beta_minus_one_da(x, a, n_nodes=80):
    """d/da B(x; a, -1) = int_0^x t^(a-1) ln t (1-t)^-2 dt for 0 < x < 1, by parts: x^(a-1) ln x / (1-x) - int_0^x g(t)/(1-t) dt
    with g(t) = t^(a-2) [(a-1) ln t + 1], g(1) = 1, the log-singular remainder split as -ln(1-x) + int_0^x (g(t) - 1)/(1-t) dt."""
    xs, ws = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * x * (xs + 1.0)
    g = t ** (a - 2.0) * ((a - 1.0) * np.log(t) + 1.0)
    smooth = np.sum(0.5 * x * ws * (g - 1.0) / (1.0 - t))
    return x ** (a - 1.0) * np.log(x) / (1.0 - x) - (-np.log1p(-x) + smooth)


def _frc_force(r, l, phi, n_bonds, k_t):
    """Explicit FRC force-extension relation of the elasticity source (its Eq. 8), energies in D_e."""
    r_max, l_k = _frc_geometry(l, phi, n_bonds)
    rs = r / r_max
    b = _langevin_inverse(rs)
    return k_t / l_k * (b + 0.5 * rs ** 2 / (1.0 - rs) ** 2 * (1.0 - rs ** (l_k / l - 1.0)))


def _frc_free_energy(r, l, phi, n_bonds, k_t):
    """Entropic free energy of the FRC (elasticity source, Eq. 9); 1e6 outside the physical domain."""
    r_max, l_k = _frc_geometry(l, phi, n_bonds)
    rs = r / r_max
    if not (1e-9 < rs < 1.0 - 1e-7):
        return 1e6
    b = _langevin_inverse(rs)
    log_sinh = b + np.log1p(-np.exp(-2.0 * b)) - np.log(2.0)
    inc_beta = _incomplete_beta_minus_one(rs, 2.0 + l_k / l)                    # B(rs; 2 + l_k/l, -1)
    return k_t * r_max / l_k * (rs * b + np.log(b) - log_sinh + (1.0 + rs * (1.0 - rs)) / (2.0 * (1.0 - rs))
                                + np.log(1.0 - rs) - 0.5 * inc_beta)


def _dfrc_total(p, r, n_bonds, beta_de, a_le, beta_kphi, phi_e):
    """Helmholtz free energy Psi(l, phi; r) of the dFRC with Morse bonds and harmonic bending, in D_e."""
    l, phi = float(p[0]), float(p[1])
    if l <= 0.5 or l > 3.0 or phi <= 0.05 or phi >= np.pi - 0.05:
        return 1e6
    k_t = 1.0 / beta_de
    return (n_bonds * _morse(l, a_le) + (n_bonds - 1) * 0.5 * beta_kphi * k_t * (phi - phi_e) ** 2
            + _frc_free_energy(r, l, phi, n_bonds, k_t))


def _dfrc_gradient(p, r, n_bonds, beta_de, a_le, beta_kphi, phi_e):
    """Analytic gradient (dPsi/dl, dPsi/dphi) of the dFRC free energy, in D_e per l_e and D_e per radian. Psi_ent depends on
    l only through r* = r / R_max, so dPsi_ent/dl = -(r / l) f with f from Eq. 8; the phi-derivative also needs
    d(l_k/l)/dphi and the derivative of the incomplete beta function with respect to its first parameter."""
    l, phi = float(p[0]), float(p[1])
    k_t = 1.0 / beta_de
    r_max, l_k = _frc_geometry(l, phi, n_bonds)
    x = r / r_max
    if l <= 0.5 or l > 3.0 or phi <= 0.05 or phi >= np.pi - 0.05 or not (1e-9 < x < 1.0 - 1e-7):
        return [1e6, 1e6]
    eta = _langevin_inverse(x)
    a = 2.0 + l_k / l                                                          # incomplete-beta parameter, a function of phi only
    g_x = eta + 0.5 * x ** 2 / (1.0 - x) ** 2 * (1.0 - x ** (a - 3.0))          # dG/dr* = f l_k / (k_B T), Eq. 8
    f = k_t / l_k * g_x
    log_sinh = eta + np.log1p(-np.exp(-2.0 * eta)) - np.log(2.0)
    g_val = (x * eta + np.log(eta) - log_sinh + (1.0 + x * (1.0 - x)) / (2.0 * (1.0 - x)) + np.log1p(-x)
             - 0.5 * _incomplete_beta_minus_one(x, a))                          # Psi_ent = k_B T (R_max / l_k) G
    g_a = -0.5 * _incomplete_beta_minus_one_da(x, a)
    prefac, dprefac = 0.5 * n_bonds * (1.0 - np.cos(phi)), 0.5 * n_bonds * np.sin(phi)   # R_max / l_k and its phi-derivative
    dx_dphi = 0.5 * x * np.tan(0.5 * phi)
    c, sn = np.cos(0.5 * phi), np.sin(0.5 * phi)
    da_dphi = (-sn * (1.0 - np.cos(phi)) - 2.0 * c * np.sin(phi)) / (1.0 - np.cos(phi)) ** 2
    dpsi_dl = n_bonds * _morse_prime(l, a_le) - (r / l) * f
    dpsi_dphi = ((n_bonds - 1) * beta_kphi * k_t * (phi - phi_e)
                 + k_t * (dprefac * g_val + prefac * (g_x * dx_dphi + g_a * da_dphi)))
    return [dpsi_dl, dpsi_dphi]


def _newton_polish(gradient, p0, args, n_iter=12, h=1e-6):
    """Newton iterations on a stationarity system with a central-difference Jacobian of the analytic gradient; returns the
    point and the size of the last step (a converged optimum has a last step far below 1e-12)."""
    p = np.array(p0, dtype=float)
    step = np.inf
    for _ in range(n_iter):
        g0 = np.array(gradient(p, *args))
        jac = np.zeros((p.size, p.size))
        for k in range(p.size):
            d = np.zeros(p.size)
            d[k] = h
            jac[:, k] = (np.array(gradient(p + d, *args)) - np.array(gradient(p - d, *args))) / (2.0 * h)
        delta = np.linalg.solve(jac, -g0)
        p = p + delta
        step = float(np.max(np.abs(delta)))
        if step < 1e-15:
            break
    return p, step


def _fjc_total(p, r, n_bonds, beta_de, a_le):
    """Free energy of the extensible freely jointed reduction of the dFRC (l_k = l, R_max = N l), in D_e."""
    l = float(p[0]) if np.ndim(p) else float(p)
    rs = r / (n_bonds * l)
    if l <= 0.5 or l > 3.0 or not (1e-9 < rs < 1.0 - 1e-7):
        return 1e6
    b = _langevin_inverse(rs)
    log_sinh = b + np.log1p(-np.exp(-2.0 * b)) - np.log(2.0)
    return n_bonds * _morse(l, a_le) + n_bonds / beta_de * (rs * b + np.log(b) - log_sinh)


def _fjc_gradient(p, r, n_bonds, beta_de, a_le):
    """Analytic dPsi/dl of the freely jointed reduction: dPsi_ent/dl = -(r / l) f with f = k_B T eta / l."""
    l = float(p[0]) if np.ndim(p) else float(p)
    rs = r / (n_bonds * l)
    if l <= 0.5 or l > 3.0 or not (1e-9 < rs < 1.0 - 1e-7):
        return [1e6]
    eta = _langevin_inverse(rs)
    return [n_bonds * _morse_prime(l, a_le) - (r / l) * eta / (beta_de * l)]


def dfrc_chain_force(stretch_ratio: float, n_bonds: int, beta_de: float, a_le: float, beta_kphi: float,
                             phi_e: float) -> tuple:
    for name, value in (("stretch_ratio", stretch_ratio), ("beta_de", beta_de), ("a_le", a_le)):
        _check_positive(name, value)
    if not isinstance(n_bonds, (int, np.integer)) or n_bonds < 2:
        raise ValueError("n_bonds must be an integer >= 2")
    if not np.isfinite(beta_kphi) or beta_kphi < 0.0:
        raise ValueError("beta_kphi must be finite and non-negative")
    if not np.isfinite(phi_e) or phi_e <= 0.0 or phi_e >= np.pi:
        raise ValueError("phi_e must lie strictly inside (0, pi)")
    k_t = 1.0 / beta_de
    r = stretch_ratio * n_bonds * np.cos(0.5 * phi_e)                      # r in units of l_e, R_0 = N l_e cos(phi_e/2)
    if beta_kphi == 0.0:                                                      # freely jointed limit: l_k = l, R_max = N l
        fj_args = (r, int(n_bonds), beta_de, a_le)
        res = minimize(_fjc_total, x0=[max(1.0, r / n_bonds * 1.001)], args=fj_args, method="Nelder-Mead",
                       options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))
        sol, step = _newton_polish(_fjc_gradient, res.x, fj_args)
        if not step < 1e-12:
            raise ValueError("the freely jointed optimum did not converge to 1e-12")
        l_star = float(sol[0])
        rs = r / (n_bonds * l_star)
        return l_star, 0.0, float(k_t / l_star * _langevin_inverse(rs))
    args = (r, int(n_bonds), beta_de, a_le, beta_kphi, phi_e)
    start = [max(1.0, 1.1 * stretch_ratio), phi_e]                             # feasible start (r* < 1)
    res = minimize(_dfrc_total, x0=start, args=args, method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))          # locate the basin
    sol, step = _newton_polish(_dfrc_gradient, res.x, args)                    # Newton on the analytic gradient: 1e-12 in both variables
    if not step < 1e-12:
        raise ValueError("the dFRC optimum did not converge to 1e-12 in both variables")
    l_star, phi_star = float(sol[0]), float(sol[1])
    if _dfrc_total(sol, *args) >= 1e5:
        raise ValueError("no physical dFRC optimum for the supplied stretch ratio")
    return l_star, phi_star, float(_frc_force(r, l_star, phi_star, int(n_bonds), k_t))

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _log_trapezoid(log_f, x):
    """log of the trapezoidal integral of exp(log_f) over x (log_f may contain -inf)."""
    dx = np.diff(x)
    log_w = np.log(np.concatenate(([dx[0] / 2.0], (dx[:-1] + dx[1:]) / 2.0, [dx[-1] / 2.0])))
    e = log_f + log_w
    m = e.max()
    return float(m + np.log(np.exp(e - m).sum()))


def collinear_reference_rate(f_red: float, beta_de: float, a_le: float, prefactor: float,
                                     n_l: int = 20001) -> tuple:
    for name, value in (("f_red", f_red), ("beta_de", beta_de), ("a_le", a_le), ("prefactor", prefactor)):
        _check_positive(name, value)
    if not isinstance(n_l, (int, np.integer)) or n_l < 3:
        raise ValueError("n_l must be an integer >= 3")
    if f_red >= 0.5 * a_le:
        raise ValueError("f_red must be below the collinear critical force a_le / 2")
    s = np.sqrt(1.0 - 2.0 * f_red / a_le)
    l_min = 1.0 - np.log((1.0 + s) / 2.0) / a_le          # tilted-Morse stationary points (S95)
    l_bar = 1.0 - np.log((1.0 - s) / 2.0) / a_le

    w_min = _morse(l_min, a_le) - f_red * l_min
    barrier = float(_morse(l_bar, a_le) - f_red * l_bar - w_min)
    l = np.linspace(0.0, l_bar, int(n_l))
    log_int = _log_trapezoid(-beta_de * (_morse(l, a_le) - f_red * l - w_min), l)
    rate = float(prefactor * np.exp(-beta_de * barrier - log_int))
    return float(l_min), float(l_bar), barrier, rate

import numpy as np
from scipy.optimize import brentq, minimize


def bending_kernel(theta: "np.ndarray", beta_kphi: float, phi_e: float, n_omega: int = 512) -> "np.ndarray":
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 1 or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a one-dimensional finite array")
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        raise ValueError("theta must lie in [0, pi]")
    if not np.isfinite(beta_kphi) or beta_kphi < 0.0:
        raise ValueError("beta_kphi must be finite and non-negative")
    if not np.isfinite(phi_e) or phi_e < 0.0 or phi_e > np.pi:
        raise ValueError("phi_e must lie in [0, pi]")
    if not isinstance(n_omega, (int, np.integer)) or n_omega < 1:
        raise ValueError("n_omega must be a positive integer")
    omega = (np.arange(int(n_omega)) + 0.5) * (2.0 * np.pi / int(n_omega))   # midpoint rule, periodic
    ct, st = np.cos(theta), np.sin(theta)
    cos_phi = ct[:, None, None] * ct[None, :, None] + st[:, None, None] * st[None, :, None] * np.cos(omega)[None, None, :]
    phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
    return np.exp(-0.5 * beta_kphi * (phi - phi_e) ** 2).sum(axis=2) * (2.0 * np.pi / int(n_omega))

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _log_trapezoid(log_f, x):
    """log of the trapezoidal integral of exp(log_f) over x (log_f may contain -inf)."""
    dx = np.diff(x)
    log_w = np.log(np.concatenate(([dx[0] / 2.0], (dx[:-1] + dx[1:]) / 2.0, [dx[-1] / 2.0])))
    e = log_f + log_w
    m = e.max()
    return float(m + np.log(np.exp(e - m).sum()))


def log_intact_weight(theta: "np.ndarray", l_thr: float, f_red: float, beta_de: float, a_le: float,
                              n_l: int = 4001) -> "np.ndarray":
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 1 or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a one-dimensional finite array")
    if np.any(theta <= 0.0) or np.any(theta >= np.pi):
        raise ValueError("theta must lie strictly inside (0, pi)")
    for name, value in (("l_thr", l_thr), ("f_red", f_red), ("beta_de", beta_de), ("a_le", a_le)):
        _check_positive(name, value)
    if not isinstance(n_l, (int, np.integer)) or n_l < 3:
        raise ValueError("n_l must be an integer >= 3")
    l = np.linspace(0.0, l_thr, int(n_l))
    with np.errstate(divide="ignore"):
        log_l2 = 2.0 * np.log(l)                                  # -inf at l = 0: zero measure there
    expo = (-beta_de * _morse(l, a_le) + log_l2)[None, :] + beta_de * f_red * l[None, :] * np.cos(theta)[:, None]
    out = np.empty(theta.size)
    for k in range(theta.size):
        out[k] = _log_trapezoid(expo[k], l)
    return np.log(np.sin(theta)) + out

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _check_grid(theta, w_theta):
    theta = np.asarray(theta, dtype=float)
    w_theta = np.asarray(w_theta, dtype=float)
    if theta.ndim != 1 or theta.size < 2 or w_theta.shape != theta.shape:
        raise ValueError("theta and w_theta must be one-dimensional arrays of equal length >= 2")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(w_theta))):
        raise ValueError("theta and w_theta must be finite")
    if np.any(theta <= 0.0) or np.any(theta >= np.pi) or np.any(np.diff(theta) <= 0.0) or np.any(w_theta <= 0.0):
        raise ValueError("theta must be strictly increasing inside (0, pi) with positive weights")
    return theta, w_theta


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _morse_prime(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return 2.0 * a_le * e * (1.0 - e)


def _angular_terms(theta, w_theta, log_w, f_red, beta_de, l):
    """For each l: log of  int w(theta) exp(beta f l cos theta) sin theta dtheta  and the cos-average under it."""
    l = np.atleast_1d(np.asarray(l, dtype=float))
    c = np.cos(theta)
    base = log_w + np.log(np.sin(theta)) + np.log(w_theta)
    e = base[None, :] + beta_de * f_red * l[:, None] * c[None, :]
    m = e.max(axis=1)
    z = np.exp(e - m[:, None])
    s = z.sum(axis=1)
    return m + np.log(s), (z * c[None, :]).sum(axis=1) / s


def _pmf_value(l, theta, w_theta, log_w, f_red, beta_de, a_le):
    """W(l) in D_e for one bond's log angular weight; accepts a scalar or an array of lengths (in l_e)."""
    la, _ = _angular_terms(theta, w_theta, log_w, f_red, beta_de, l)
    out = _morse(np.atleast_1d(l), a_le) - (2.0 * np.log(np.atleast_1d(l)) + la) / beta_de
    return float(out[0]) if np.ndim(l) == 0 else out


def _pmf_slope(l, theta, w_theta, log_w, f_red, beta_de, a_le):
    """W'(l) in D_e / l_e; accepts a scalar or an array of lengths."""
    _, cm = _angular_terms(theta, w_theta, log_w, f_red, beta_de, l)
    out = _morse_prime(np.atleast_1d(l), a_le) - (2.0 / np.atleast_1d(l)) / beta_de - f_red * cm
    return float(out[0]) if np.ndim(l) == 0 else out


def _messages_at(theta, w_theta, kernel, log_i, bond_index):
    """Normalized forward and backward message shapes arriving at bond `bond_index` (S72-S76)."""
    n_bonds = log_i.shape[0]
    left = np.full(theta.size, 1.0 / np.pi)
    for i in range(bond_index):                                    # bonds 1 .. bond_index (0-based i)
        shift = log_i[i].max()
        prop = kernel @ (left * np.exp(log_i[i] - shift) * w_theta)
        left = prop / (prop * w_theta).sum()
    right = np.full(theta.size, 1.0 / np.pi)
    for i in range(n_bonds - 1, bond_index, -1):                   # bonds N .. bond_index + 2
        shift = log_i[i].max()
        prop = kernel.T @ (right * np.exp(log_i[i] - shift) * w_theta)
        right = prop / (prop * w_theta).sum()
    return left, right


def constrained_pmf(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                            bond_index: int, l_values: "np.ndarray", f_red: float, beta_de: float,
                            a_le: float) -> "np.ndarray":
    theta, w_theta = _check_grid(theta, w_theta)
    kernel = np.asarray(kernel, dtype=float)
    log_i = np.asarray(log_i, dtype=float)
    n = theta.size
    if kernel.shape != (n, n) or not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("kernel must be a finite non-negative (n_theta, n_theta) array")
    if log_i.ndim != 2 or log_i.shape[1] != n or log_i.shape[0] < 1 or np.any(np.isnan(log_i)) or np.any(log_i == np.inf):
        raise ValueError("log_i must be an (n_bonds, n_theta) array without NaN or +inf")
    if not isinstance(bond_index, (int, np.integer)) or bond_index < 0 or bond_index >= log_i.shape[0]:
        raise ValueError("bond_index must be an integer in [0, n_bonds)")
    l_values = np.atleast_1d(np.asarray(l_values, dtype=float))
    if l_values.ndim != 1 or l_values.size < 1 or not np.all(np.isfinite(l_values)) or np.any(l_values < 0.0):
        raise ValueError("l_values must be a one-dimensional array of finite non-negative lengths")
    for name, value in (("f_red", f_red), ("beta_de", beta_de), ("a_le", a_le)):
        _check_positive(name, value)
    left, right = _messages_at(theta, w_theta, kernel, log_i, int(bond_index))
    weight = left * right
    norm = (weight * w_theta).sum()
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("angular weight lost normalization: messages underflowed")
    with np.errstate(divide="ignore"):
        log_w = np.log(np.pi * weight / norm)                      # mean-normalized angular weight (S107)
    args = (theta, w_theta, log_w, f_red, beta_de, a_le)
    with np.errstate(divide="ignore"):
        pmf = _pmf_value(l_values, *args) - _pmf_value(np.array([1.0]), *args)[0]
        slope = _pmf_slope(l_values, *args)
    return np.vstack([pmf, slope])

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def pmf_stationary_points(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                                  bond_index: int, f_red: float, beta_de: float, a_le: float,
                                  l_lo: float = 0.8, l_hi: float = 6.0, n_scan: int = 2601) -> tuple:
    for name, value in (("l_lo", l_lo), ("l_hi", l_hi)):
        _check_positive(name, value)
    if l_hi <= l_lo:
        raise ValueError("l_hi must exceed l_lo")
    if not isinstance(n_scan, (int, np.integer)) or n_scan < 3:
        raise ValueError("n_scan must be an integer >= 3")
    grid = np.linspace(l_lo, l_hi, int(n_scan))
    profile = constrained_pmf(theta, w_theta, kernel, log_i, bond_index, grid, f_red, beta_de, a_le)
    d = profile[1]
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    mins = [i for i in idx if d[i] < 0.0]
    if not mins:
        raise ValueError("no bonded minimum of the potential of mean force on the scan interval")
    i_min = mins[0]
    maxs = [i for i in idx if d[i] > 0.0 and i > i_min]
    if not maxs:
        raise ValueError("no barrier top of the potential of mean force: force at or above the critical force")

    slope = lambda l: float(constrained_pmf(theta, w_theta, kernel, log_i, bond_index, np.array([l]),
                                                    f_red, beta_de, a_le)[1, 0])
    l_min = brentq(slope, grid[i_min], grid[i_min + 1], xtol=1e-13, rtol=4.0 * np.finfo(float).eps)
    l_bar = brentq(slope, grid[maxs[0]], grid[maxs[0] + 1], xtol=1e-13, rtol=4.0 * np.finfo(float).eps)
    ends = constrained_pmf(theta, w_theta, kernel, log_i, bond_index, np.array([l_min, l_bar]),
                                   f_red, beta_de, a_le)[0]
    return float(l_min), float(l_bar), float(ends[1] - ends[0])

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _check_grid(theta, w_theta):
    theta = np.asarray(theta, dtype=float)
    w_theta = np.asarray(w_theta, dtype=float)
    if theta.ndim != 1 or theta.size < 2 or w_theta.shape != theta.shape:
        raise ValueError("theta and w_theta must be one-dimensional arrays of equal length >= 2")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(w_theta))):
        raise ValueError("theta and w_theta must be finite")
    if np.any(theta <= 0.0) or np.any(theta >= np.pi) or np.any(np.diff(theta) <= 0.0) or np.any(w_theta <= 0.0):
        raise ValueError("theta must be strictly increasing inside (0, pi) with positive weights")
    return theta, w_theta


def self_consistent_thresholds(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", f_red: float,
                                       beta_de: float, a_le: float, n_bonds: int, l_init: float,
                                       tol: float = 1e-10, max_iter: int = 50, n_l: int = 4001) -> "np.ndarray":
    theta, w_theta = _check_grid(theta, w_theta)
    if not isinstance(n_bonds, (int, np.integer)) or n_bonds < 1:
        raise ValueError("n_bonds must be a positive integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    for name, value in (("l_init", l_init), ("tol", tol)):
        _check_positive(name, value)
    thresholds = np.full(int(n_bonds), float(l_init))
    for _ in range(int(max_iter)):
        log_i = np.array([log_intact_weight(theta, thresholds[i], f_red, beta_de, a_le, n_l)
                          for i in range(int(n_bonds))])
        new = np.array([pmf_stationary_points(theta, w_theta, kernel, log_i, i, f_red, beta_de, a_le)[1]
                        for i in range(int(n_bonds))])
        change = float(np.max(np.abs(new - thresholds)))
        thresholds = new
        if change < tol:
            return thresholds
    raise ValueError("self-consistent thresholds did not converge within max_iter iterations")

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _log_trapezoid(log_f, x):
    """log of the trapezoidal integral of exp(log_f) over x (log_f may contain -inf)."""
    dx = np.diff(x)
    log_w = np.log(np.concatenate(([dx[0] / 2.0], (dx[:-1] + dx[1:]) / 2.0, [dx[-1] / 2.0])))
    e = log_f + log_w
    m = e.max()
    return float(m + np.log(np.exp(e - m).sum()))


def bond_tst_rate(l_values: "np.ndarray", pmf_values: "np.ndarray", beta_de: float, prefactor: float) -> float:
    l_values = np.asarray(l_values, dtype=float)
    pmf_values = np.asarray(pmf_values, dtype=float)
    if l_values.ndim != 1 or l_values.size < 3 or pmf_values.shape != l_values.shape:
        raise ValueError("l_values and pmf_values must be one-dimensional arrays of equal length >= 3")
    if not np.all(np.isfinite(l_values)) or l_values[0] != 0.0 or np.any(np.diff(l_values) <= 0.0):
        raise ValueError("l_values must increase strictly from 0 to the barrier top")
    if np.any(np.isnan(pmf_values)) or np.any(pmf_values == -np.inf) or not np.all(np.isfinite(pmf_values[1:])):
        raise ValueError("pmf_values must be finite (the value at l = 0 may be +inf)")
    for name, value in (("beta_de", beta_de), ("prefactor", prefactor)):
        _check_positive(name, value)
    w_ref = float(np.min(pmf_values[1:]))
    log_f = np.where(np.isfinite(pmf_values), -beta_de * (pmf_values - w_ref), -np.inf)
    log_int = _log_trapezoid(log_f, l_values)
    return float(prefactor * np.exp(-beta_de * (pmf_values[-1] - w_ref) - log_int))

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def mean_scission_time(n_bonds: int, stretch_ratio: float, d_e: float, a: float, l_e: float, k_phi: float,
                               valence_angle_deg: float, temperature: float, mass: float,
                               k_b: float = 1.380649e-23, n_theta: int = 400, n_omega: int = 512,
                               n_well: int = 20001) -> float:
    if not isinstance(n_bonds, (int, np.integer)) or n_bonds < 2:
        raise ValueError("n_bonds must be an integer >= 2")
    for name, value in (("stretch_ratio", stretch_ratio), ("d_e", d_e), ("a", a), ("l_e", l_e),
                        ("temperature", temperature), ("mass", mass), ("k_b", k_b)):
        _check_positive(name, value)
    if not np.isfinite(k_phi) or k_phi < 0.0:
        raise ValueError("k_phi must be finite and non-negative")
    if not np.isfinite(valence_angle_deg) or valence_angle_deg <= 0.0 or valence_angle_deg >= 180.0:
        raise ValueError("valence_angle_deg must lie strictly inside (0, 180)")
    if not isinstance(n_theta, (int, np.integer)) or n_theta < 2:
        raise ValueError("n_theta must be an integer >= 2")
    if not isinstance(n_well, (int, np.integer)) or n_well < 3:
        raise ValueError("n_well must be an integer >= 3")
    beta = 1.0 / (k_b * temperature)
    beta_de, a_le = beta * d_e, a * l_e
    beta_kphi = beta * k_phi
    phi_e = np.pi - np.deg2rad(valence_angle_deg)      # angle between successive bond vectors = pi - valence angle
    f_red = dfrc_chain_force(stretch_ratio, int(n_bonds), beta_de, a_le, beta_kphi, phi_e)[2]
    pref_anchor = 1.0 / np.sqrt(2.0 * np.pi * mass * beta) / l_e           # bond 1: one mobile atom, mass m (S40)
    pref_inner = 1.0 / np.sqrt(2.0 * np.pi * (mass / 2.0) * beta) / l_e    # bonds >= 2: relative coordinate, m/2
    x, w = np.polynomial.legendre.leggauss(int(n_theta))
    theta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w
    l_bar_1d = collinear_reference_rate(f_red, beta_de, a_le, pref_inner)[1]
    kernel = bending_kernel(theta, beta_kphi, phi_e, n_omega)
    thresholds = self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le,
                                                    int(n_bonds), l_bar_1d)
    log_i = np.array([log_intact_weight(theta, thresholds[i], f_red, beta_de, a_le)
                      for i in range(int(n_bonds))])
    total = 0.0
    for i in range(int(n_bonds)):
        l_min, l_bar, _ = pmf_stationary_points(theta, w_theta, kernel, log_i, i, f_red, beta_de, a_le)
        well = np.linspace(0.0, l_bar, int(n_well))
        profile = constrained_pmf(theta, w_theta, kernel, log_i, i, well, f_red, beta_de, a_le)[0]
        prefactor = pref_anchor if i == 0 else pref_inner
        total += bond_tst_rate(well, profile, beta_de, prefactor)
    return float(1.0 / total / 3600.0)                                       # mean time to first scission in hours
SCICODE_GOLD_EOF
