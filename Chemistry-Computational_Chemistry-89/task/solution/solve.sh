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
from math import comb


def _jet_mul(f, g):
    """Leibniz product of two derivative jets [h, h', h'', h''', h''''']."""
    out = np.zeros(np.broadcast(f, g).shape)
    for n in range(5):
        out[n] = sum(comb(n, k) * f[k] * g[n - k] for k in range(n + 1))
    return out


def _jet_compose(f, g):
    """Jet of G(f(x)) from the jet f and outer derivatives g[k] = G^(k)(f(x))."""
    f1, f2, f3, f4 = f[1], f[2], f[3], f[4]
    h = np.empty(np.broadcast(f, g).shape)
    h[0] = g[0]
    h[1] = g[1] * f1
    h[2] = g[1] * f2 + g[2] * f1 ** 2
    h[3] = g[1] * f3 + 3.0 * g[2] * f1 * f2 + g[3] * f1 ** 3
    h[4] = (g[1] * f4 + g[2] * (4.0 * f1 * f3 + 3.0 * f2 ** 2)
            + 6.0 * g[3] * f1 ** 2 * f2 + g[4] * f1 ** 4)
    return h


def _jet_exp(f):
    e = np.exp(f[0])
    return _jet_compose(f, np.array([e, e, e, e, e]))


def _jet_sqrt(f):
    s = np.sqrt(f[0])
    return _jet_compose(f, np.array([s, 0.5 / s, -0.25 / s ** 3,
                                     0.375 / s ** 5, -0.9375 / s ** 7]))


def _jet_inv(f):
    r = 1.0 / f[0]
    return _jet_compose(f, np.array([r, -r ** 2, 2.0 * r ** 3,
                                     -6.0 * r ** 4, 24.0 * r ** 5]))


def potential_derivative_tensor(points: "np.ndarray", order: int, params: dict) -> "np.ndarray":
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 4:
        raise ValueError("order must be an integer in 0..4")
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1:
        raise ValueError("points must have shape (n, 2) with n >= 1")
    x, y = pts[:, 0], pts[:, 1]
    a, V0, V_inf = float(params["a"]), float(params["V0"]), float(params["V_inf"])
    m, we = float(params["m"]), float(params["omega_e"])
    c_inf, c_0, sig = float(params["chi_inf"]), float(params["chi_0"]), float(params["sigma_e"])

    # Eckart part: sech^2(u) and the logistic step, differentiated in u = x/a.
    u = x / a
    t = np.tanh(u)
    s = 1.0 / np.cosh(u) ** 2
    sech2 = [s, -2.0 * s * t, s * (6.0 * t ** 2 - 2.0), s * t * (16.0 - 24.0 * t ** 2),
             s * (16.0 - 120.0 * t ** 2 + 120.0 * t ** 4)]
    step = [0.5 * (1.0 + t), 0.5 * s, -s * t, s * (3.0 * t ** 2 - 1.0),
            s * t * (8.0 - 12.0 * t ** 2)]
    v_eck = [(V0 * sech2[n] + V_inf * step[n]) / a ** n for n in range(5)]

    # Jets of chi(x), D(x) = omega_e / (4 chi) and alpha(x) = sqrt(2 m omega_e chi).
    gauss = np.exp(-x ** 2 / (2.0 * sig ** 2))
    z = x / sig
    hermite = [np.ones_like(x), -z / sig, (z ** 2 - 1.0) / sig ** 2,
               -(z ** 3 - 3.0 * z) / sig ** 3, (z ** 4 - 6.0 * z ** 2 + 3.0) / sig ** 4]
    chi = np.array([c_inf + (c_0 - c_inf) * gauss]
                   + [(c_0 - c_inf) * gauss * hermite[n] for n in range(1, 5)])
    depth = 0.25 * we * _jet_inv(chi)
    alpha = _jet_sqrt(2.0 * m * we * chi)

    # V_M = D - 2 D exp(-alpha y) + D exp(-2 alpha y); d^j/dy^j acts on the exponentials.
    mixed = {}
    for i in range(5):
        for j in range(5 - i):
            val = v_eck[i] if j == 0 else np.zeros_like(x)
            if j == 0:
                val = val + depth[i]
            for k, coef in ((1, -2.0), (2, 1.0)):
                amp = depth
                for _ in range(j):
                    amp = _jet_mul(amp, -k * alpha)
                val = val + coef * _jet_mul(amp, _jet_exp(-k * y * alpha))[i]
            mixed[(i, j)] = val

    out = np.empty((pts.shape[0],) + (2,) * order)
    for idx in np.ndindex(*((2,) * order)):
        n_y = sum(idx)
        out[(slice(None),) + idx] = mixed[(order - n_y, n_y)]
    return out

import numpy as np


def _half_ring_gradient_hessian(z, beta, n_beads, params):
    """Gradient and Hessian of the half-ring potential sum_i V + (k/2) sum |dz|^2."""
    half = z.shape[0]
    k_spring = float(params["m"]) * (n_beads / beta) ** 2
    grad = potential_derivative_tensor(z, 1, params)
    grad[:-1] += k_spring * (z[:-1] - z[1:])
    grad[1:] += k_spring * (z[1:] - z[:-1])
    hess = np.zeros((2 * half, 2 * half))
    blocks = potential_derivative_tensor(z, 2, params)
    for i in range(half):
        hess[2 * i:2 * i + 2, 2 * i:2 * i + 2] = blocks[i]
    lap = np.diag(np.r_[1.0, np.full(half - 2, 2.0), 1.0])
    lap -= np.diag(np.ones(half - 1), 1) + np.diag(np.ones(half - 1), -1)
    hess += k_spring * np.kron(lap, np.eye(2))
    return grad.ravel(), hess


def _barrier_top(params):
    """Location of the maximum of V(x, 0), refined by Newton iterations."""
    a = float(params["a"])
    xs = np.linspace(-10.0 * a, 10.0 * a, 4001)
    energy = potential_derivative_tensor(np.column_stack([xs, np.zeros_like(xs)]), 0, params)
    xb = xs[int(np.argmax(energy))]
    for _ in range(50):
        pt = np.array([[xb, 0.0]])
        g = potential_derivative_tensor(pt, 1, params)[0, 0]
        h = potential_derivative_tensor(pt, 2, params)[0, 0, 0]
        dx = -g / h
        xb += dx
        if abs(dx) < 1e-14 * max(1.0, abs(xb)):
            break
    return xb


def ring_polymer_instanton(beta: float, n_beads: int, params: dict) -> "np.ndarray":
    if (isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer))
            or n_beads < 4 or n_beads % 2):
        raise ValueError("n_beads must be an even integer >= 4")
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("beta must be positive")
    a = float(params["a"])
    half = n_beads // 2
    xb = _barrier_top(params)
    theta = np.pi * (np.arange(half) + 0.5) / half
    z = np.column_stack([xb + a * np.cos(theta), np.zeros(half)])

    max_step = 0.3 * a
    # Characteristic force: spring stiffness times the barrier width plus the barrier force.
    f_ref = float(params["m"]) * (n_beads / beta) ** 2 * a + (abs(float(params["V0"])) + abs(float(params["V_inf"]))) / a
    for _ in range(500):
        grad, hess = _half_ring_gradient_hessian(z, beta, n_beads, params)
        w, v = np.linalg.eigh(hess)
        gt = v.T @ grad
        if np.max(np.abs(grad)) < 1e-6 * f_ref:
            coeff = -gt / w                      # Newton polish onto the nearby stationary point
        else:
            coeff = -gt / np.abs(w)              # eigenvector following: descend along all modes
            coeff[0] = gt[0] / abs(w[0])         # ... except ascend along the lowest one
        step = v @ coeff
        norm = np.linalg.norm(step)
        if norm > max_step:
            step *= max_step / norm
        z = z + step.reshape(half, 2)
        if norm < 1e-10 * a and np.max(np.abs(grad)) < 1e-10 * f_ref:
            break
    grad, _ = _half_ring_gradient_hessian(z, beta, n_beads, params)
    ordered = bool(np.all(np.diff(z[:, 0]) < 0.0))
    if z[0, 0] - z[-1, 0] < 1e-6 * a or not ordered or np.max(np.abs(grad)) > 1e-10 * f_ref:
        raise ValueError("no delocalized instanton found (beta at or below crossover?)")
    return np.vstack([z, z[::-1]])

import numpy as np


def _cyclic_laplacian(n):
    """Second-difference matrix of a closed ring of n beads."""
    eye = np.eye(n)
    return 2.0 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)


def leading_order_instanton_rate(beads: "np.ndarray", beta: float, params: dict) -> float:
    q = np.asarray(beads, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] < 4:
        raise ValueError("beads must have shape (N, 2) with N >= 4")
    n = q.shape[0]
    m, we = float(params["m"]), float(params["omega_e"])
    beta_n = float(beta) / n
    dq = np.roll(q, -1, axis=0) - q
    stretch = float(np.sum(dq ** 2))
    u_n = float(np.sum(potential_derivative_tensor(q, 0, params))) + 0.5 * m * stretch / beta_n ** 2
    b_n = m * stretch

    hess = np.kron(_cyclic_laplacian(n), np.eye(2)) * (m / beta_n ** 2)
    blocks = potential_derivative_tensor(q, 2, params)
    for i in range(n):
        hess[2 * i:2 * i + 2, 2 * i:2 * i + 2] += blocks[i]
    lam = np.linalg.eigvalsh(hess / m)
    lam = np.delete(lam, int(np.argmin(np.abs(lam))))
    eta = np.sqrt(np.abs(lam))

    log_kz = (-np.log(beta_n) + 0.5 * np.log(b_n / (2.0 * np.pi * beta_n))
              - np.sum(np.log(beta_n * eta)) - beta_n * u_n)
    w_k = np.sqrt(we ** 2 + (2.0 * np.sin(np.pi * np.arange(n) / n) / beta_n) ** 2)
    log_z = 0.5 * np.log(m / (2.0 * np.pi * beta)) - np.sum(np.log(beta_n * w_k))
    return float(np.exp(log_kz - log_z))

import numpy as np


def ring_polymer_propagator(curvatures: "np.ndarray", beta: float, m: float) -> "np.ndarray":
    c = np.asarray(curvatures, dtype=float)
    if c.ndim != 1 or c.size < 3:
        raise ValueError("curvatures must be a one-dimensional array with N >= 3")
    beta, m = float(beta), float(m)
    if not (beta > 0.0 and m > 0.0):
        raise ValueError("beta and m must be positive")
    dtau = beta / c.size
    jac = (m / dtau) * _cyclic_laplacian(c.size) + dtau * np.diag(c)
    try:
        chol = np.linalg.cholesky(jac)
    except np.linalg.LinAlgError:
        raise ValueError("action Hessian is not positive definite")
    inv_chol = np.linalg.solve(chol, np.eye(c.size))
    g = inv_chol.T @ inv_chol
    return 0.5 * (g + g.T)

import numpy as np


def anharmonic_fluctuation_correction(propagator: "np.ndarray", cubic: "np.ndarray", quartic: "np.ndarray", beta: float) -> float:
    g = np.asarray(propagator, dtype=float)
    t3 = np.asarray(cubic, dtype=float)
    t4 = np.asarray(quartic, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("propagator must be a square matrix")
    n = g.shape[0]
    if t3.shape != (n,) or t4.shape != (n,):
        raise ValueError("cubic and quartic must be one-dimensional arrays of length N")
    if not float(beta) > 0.0:
        raise ValueError("beta must be positive")
    dtau = float(beta) / n
    tv, qv, d = dtau * t3, dtau * t4, np.diag(g)
    quartic_term = -0.125 * np.sum(qv * d ** 2)
    td = tv * d
    cubic_term = 0.125 * (td @ g @ td) + (tv @ (g ** 3) @ tv) / 12.0
    return float(quartic_term + cubic_term)

import numpy as np


def reactant_partition_correction(beta: float, n_beads: int, params: dict) -> float:
    if isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer)) or n_beads < 3:
        raise ValueError("n_beads must be an integer >= 3")
    m, we, chi = float(params["m"]), float(params["omega_e"]), float(params["chi_inf"])
    if not (float(beta) > 0.0 and we > 0.0 and chi > 0.0):
        raise ValueError("beta, omega_e and chi_inf must be positive")
    alpha = np.sqrt(2.0 * m * we * chi)
    k2 = m * we ** 2
    g = ring_polymer_propagator(np.full(n_beads, k2), beta, m)
    return float(anharmonic_fluctuation_correction(
        g, np.full(n_beads, -3.0 * k2 * alpha), np.full(n_beads, 7.0 * k2 * alpha ** 2), beta))

import numpy as np
from scipy.optimize import brentq


def _eckart_action_derivative(e, order, gam, V0, V_inf):
    """n-th energy derivative (n = 1..4) of W(E) = 2 gam - gam [sqrt(E/V0) + sqrt((E - V_inf)/V0)]."""
    coef = {1: 0.5, 2: -0.25, 3: 0.375, 4: -0.9375}[order]
    return -gam / np.sqrt(V0) * coef * (e ** (0.5 - order) + (e - V_inf) ** (0.5 - order))


def eckart_tunneling_correction(beta: float, V0: float, V_inf: float, a: float, m: float) -> float:
    beta, V0, V_inf, a, m = (float(v) for v in (beta, V0, V_inf, a, m))
    if not (V0 > 0.0 and a > 0.0 and m > 0.0):
        raise ValueError("V0, a and m must be positive")
    if not -4.0 * V0 < V_inf < 4.0 * V0:
        raise ValueError("V_inf must lie in (-4 V0, 4 V0)")
    gam = np.pi * np.sqrt(2.0 * m * a ** 2 * V0)
    e_top = (4.0 * V0 + V_inf) ** 2 / (16.0 * V0)
    e_low = max(0.0, V_inf)
    beta_c = -_eckart_action_derivative(e_top, 1, gam, V0, V_inf)
    if not beta > beta_c:
        raise ValueError("beta must exceed the crossover value for a deep-tunneling instanton")
    period = lambda e: -_eckart_action_derivative(e, 1, gam, V0, V_inf) - beta
    lo = e_low + (e_top - e_low) * 1e-15
    while period(lo) < 0.0:
        lo = e_low + (lo - e_low) * 1e-6
    e_st = brentq(period, lo, e_top, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    w2, w3, w4 = (_eckart_action_derivative(e_st, k, gam, V0, V_inf) for k in (2, 3, 4))
    shape = -0.125 * w4 / w2 ** 2 + (5.0 / 24.0) * w3 ** 2 / w2 ** 3
    return float(shape + np.pi ** 2 / (4.0 * gam))

import numpy as np


def corrected_instanton_rate(beta: float, n_beads: int, params: dict) -> float:
    au_velocity_cm_s = 2.18769126364e8
    m = float(params["m"])
    beads = ring_polymer_instanton(beta, n_beads, params)
    k0 = leading_order_instanton_rate(beads, beta, params)
    curv = potential_derivative_tensor(beads, 2, params)[:, 1, 1]
    cubic = potential_derivative_tensor(beads, 3, params)[:, 1, 1, 1]
    quartic = potential_derivative_tensor(beads, 4, params)[:, 1, 1, 1, 1]
    g = ring_polymer_propagator(curv, beta, m)
    gamma_perp = anharmonic_fluctuation_correction(g, cubic, quartic, beta)
    gamma_react = reactant_partition_correction(beta, n_beads, params)
    gamma_path = eckart_tunneling_correction(
        beta, params["V0"], params["V_inf"], params["a"], m)
    return float(au_velocity_cm_s * k0 * np.exp(gamma_path + gamma_perp - gamma_react))
SCICODE_GOLD_EOF
