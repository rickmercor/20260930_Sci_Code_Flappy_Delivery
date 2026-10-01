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


def excited_surface_derivatives(points: "np.ndarray", displacement: float, surface_params: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    pts = np.asarray(points, dtype=float)
    par = np.asarray(surface_params, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1 or not np.all(np.isfinite(pts)):
        raise ValueError("points must be a finite (N, 2) array with N >= 1")
    if par.size != 5 or not np.all(np.isfinite(par)):
        raise ValueError("surface_params must hold five finite numbers")
    omega_e, chi, omega_b, gamma, delta = par
    if omega_e <= 0.0 or chi <= 0.0 or omega_b <= 0.0:
        raise ValueError("omega_e, chi and omega_b must be strictly positive")
    depth = omega_e / (4.0 * chi)
    a = np.sqrt(2.0 * omega_e * chi)
    wb2 = omega_b * omega_b
    x = pts[:, 0] - float(displacement)
    y = pts[:, 1] - delta
    e = np.exp(-a * x)
    f = np.exp(-gamma * x)
    out = np.empty((pts.shape[0], 6))
    out[:, 0] = depth * (1.0 - e) ** 2 + 0.5 * wb2 * f * y * y
    out[:, 1] = 2.0 * depth * a * e * (1.0 - e) - 0.5 * gamma * wb2 * f * y * y
    out[:, 2] = wb2 * f * y
    out[:, 3] = 2.0 * depth * a * a * e * (2.0 * e - 1.0) + 0.5 * gamma * gamma * wb2 * f * y * y
    out[:, 4] = -gamma * wb2 * f * y
    out[:, 5] = wb2 * f
    return out

import numpy as np


def exact_autocorrelation(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, n_steps: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    x1 = np.asarray(q1_axis, dtype=float)
    x2 = np.asarray(q2_axis, dtype=float)
    for ax in (x1, x2):
        if ax.ndim != 1 or ax.size < 4:
            raise ValueError("each axis must be one-dimensional with at least 4 points")
        steps = np.diff(ax)
        if steps[0] <= 0.0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            raise ValueError("each axis must be evenly spaced and increasing")
    if not time_step > 0.0 or int(n_steps) < 0:
        raise ValueError("time_step must be positive and n_steps non-negative")
    w1, w2, theta = (float(v) for v in np.asarray(ground_params, dtype=float).ravel()[:3])
    if w1 <= 0.0 or w2 <= 0.0:
        raise ValueError("ground-state frequencies must be strictly positive")
    h1 = x1[1] - x1[0]
    h2 = x2[1] - x2[0]
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    width = rot @ np.diag([w1, w2]) @ rot.T
    quad = width[0, 0] * X1 * X1 + 2.0 * width[0, 1] * X1 * X2 + width[1, 1] * X2 * X2
    psi0 = np.pi ** -0.5 * (w1 * w2) ** 0.25 * np.exp(-0.5 * quad) + 0j
    pot = excited_surface_derivatives(np.column_stack([X1.ravel(), X2.ravel()]), displacement, surface_params)[:, 0]
    half_v = np.exp(-0.5j * time_step * pot.reshape(X1.shape))
    k1 = 2.0 * np.pi * np.fft.fftfreq(x1.size, d=h1)
    k2 = 2.0 * np.pi * np.fft.fftfreq(x2.size, d=h2)
    kin = np.exp(-0.5j * time_step * (k1[:, None] ** 2 + k2[None, :] ** 2))
    weight = np.conj(psi0) * h1 * h2
    n = int(n_steps)
    out = np.empty((n + 1, 2))
    psi = psi0.copy()
    val = np.sum(weight * psi)
    out[0] = (val.real, val.imag)
    for i in range(n):
        psi = half_v * np.fft.ifft2(kin * np.fft.fft2(half_v * psi))
        val = np.sum(weight * psi)
        out[i + 1] = (val.real, val.imag)
    return out

import numpy as np


def _surface_row(point, displacement: float, par: "np.ndarray"):
    """Surface row [V, grad, Hessian entries] at one point and the Hessian as a 2 x 2 matrix."""
    import numpy as np
    row = excited_surface_derivatives(np.asarray([point], dtype=float), displacement, par)[0]
    return row, np.array([[row[3], row[4]], [row[4], row[5]]])


def _gaussian_series(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int):
    """Centre q, momentum p, width matrices Q and P and phase S of psi_t = pi^(-1/2) det(Q)^(-1/2)
    exp(i[(1/2) x^T P Q^(-1) x + p^T x + S]) at every step, with det(Q)^(1/2) continued along the steps."""
    import numpy as np
    mode = int(hessian_mode)
    if mode not in (0, 1, 2, 3) or mode != hessian_mode:
        raise ValueError("hessian_mode must be 0, 1, 2 or 3")
    if not time_step > 0.0 or int(n_steps) < 0:
        raise ValueError("time_step must be positive and n_steps non-negative")
    w1, w2, theta = (float(v) for v in np.asarray(ground_params, dtype=float).ravel()[:3])
    if w1 <= 0.0 or w2 <= 0.0:
        raise ValueError("ground-state frequencies must be strictly positive")
    par = np.asarray(surface_params, dtype=float).ravel()
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    Q = rot @ np.diag([w1 ** -0.5, w2 ** -0.5]) + 0j
    P = 1j * (rot @ np.diag([w1 ** 0.5, w2 ** 0.5]))
    q = np.zeros(2)
    p = np.zeros(2)
    S = 0.0
    if mode == 1:
        fixed = _surface_row([displacement, par[4]], displacement, par)[1]
    elif mode == 2:
        fixed = _surface_row([0.0, 0.0], displacement, par)[1]
    elif mode == 3:
        fixed = rot @ np.diag([w1 * w1, w2 * w2]) @ rot.T
    else:
        fixed = None
    w_outer = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w_inner = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
    n = int(n_steps)
    qs = np.empty((n + 1, 2))
    ps = np.empty((n + 1, 2))
    Qs = np.empty((n + 1, 2, 2), dtype=complex)
    Ps = np.empty((n + 1, 2, 2), dtype=complex)
    Ss = np.empty(n + 1)
    roots = np.empty(n + 1, dtype=complex)
    qs[0], ps[0], Qs[0], Ps[0], Ss[0] = q, p, Q, P, S
    roots[0] = np.sqrt(np.linalg.det(Q))
    for i in range(n):
        for w in (w_outer, w_inner, w_outer):
            h = w * time_step
            q = q + 0.5 * h * p
            Q = Q + 0.5 * h * P
            S += 0.25 * h * float(p @ p)
            row, local = _surface_row(q, displacement, par)
            K = local if fixed is None else fixed
            p = p - h * row[1:3]
            P = P - h * (K @ Q)
            S -= h * row[0]
            q = q + 0.5 * h * p
            Q = Q + 0.5 * h * P
            S += 0.25 * h * float(p @ p)
        qs[i + 1], ps[i + 1], Qs[i + 1], Ps[i + 1], Ss[i + 1] = q, p, Q, P, S
        r = np.sqrt(np.linalg.det(Q))
        roots[i + 1] = r if abs(r - roots[i]) <= abs(r + roots[i]) else -r
    return qs, ps, Qs, Ps, Ss, roots


def thawed_gaussian_autocorrelation(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    q, p, Q, P, S, roots = _gaussian_series(displacement, surface_params, ground_params, hessian_mode, time_step, n_steps)
    w1, w2, theta = (float(v) for v in np.asarray(ground_params, dtype=float).ravel()[:3])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    A0 = 1j * (rot @ np.diag([w1, w2]) @ rot.T)
    norm0 = np.pi ** -0.5 * (w1 * w2) ** 0.25
    A = P @ np.linalg.inv(Q)
    M = -0.5j * (A - np.conj(A0)[None, :, :])
    b = 1j * (p - np.einsum("nij,nj->ni", A, q))
    const = 1j * (0.5 * np.einsum("ni,nij,nj->n", q, A, q) - np.einsum("ni,ni->n", p, q) + S)
    sqrt_det_m = np.prod(np.sqrt(np.linalg.eigvals(M)), axis=1)
    quad = np.einsum("ni,ni->n", b, np.linalg.solve(M, b[:, :, None])[:, :, 0])
    val = norm0 * np.pi ** 0.5 / (roots * sqrt_det_m) * np.exp(0.25 * quad + const)
    return np.column_stack([val.real, val.imag])

import numpy as np


def thawed_gaussian_energy(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    par = np.asarray(surface_params, dtype=float).ravel()
    if par.size != 5 or not np.all(np.isfinite(par)):
        raise ValueError("surface_params must hold five finite numbers")
    omega_e, chi, omega_b, gamma, delta = par
    if omega_e <= 0.0 or chi <= 0.0 or omega_b <= 0.0:
        raise ValueError("omega_e, chi and omega_b must be strictly positive")
    q, p, Q, P, S, roots = _gaussian_series(displacement, surface_params, ground_params, hessian_mode, time_step, n_steps)
    depth = omega_e / (4.0 * chi)
    a = np.sqrt(2.0 * omega_e * chi)
    sigma = 0.5 * np.real(Q @ np.conj(np.transpose(Q, (0, 2, 1))))
    pi_mom = 0.5 * np.real(P @ np.conj(np.transpose(P, (0, 2, 1))))
    kinetic = 0.5 * np.sum(p ** 2, axis=1) + 0.5 * np.trace(pi_mom, axis1=1, axis2=2)
    mx = q[:, 0] - float(displacement)
    my = q[:, 1] - delta
    s11 = sigma[:, 0, 0]
    s12 = sigma[:, 0, 1]
    s22 = sigma[:, 1, 1]
    # E[exp(-k x)] = exp(-k mu_x + k^2 s11 / 2); under that tilt y has mean mu_y - k s12 and variance s22
    e1 = np.exp(-a * mx + 0.5 * a * a * s11)
    e2 = np.exp(-2.0 * a * mx + 2.0 * a * a * s11)
    eg = np.exp(-gamma * mx + 0.5 * gamma * gamma * s11)
    potential = depth * (1.0 - 2.0 * e1 + e2) + 0.5 * omega_b * omega_b * eg * ((my - gamma * s12) ** 2 + s22)
    return kinetic + potential

import numpy as np


def spectral_contrast_cosine(approx_autocorrelation: "np.ndarray", reference_autocorrelation: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    ca = np.asarray(approx_autocorrelation, dtype=float)
    cr = np.asarray(reference_autocorrelation, dtype=float)
    if ca.ndim != 2 or ca.shape != cr.shape or ca.shape[1] != 2 or ca.shape[0] < 2:
        raise ValueError("autocorrelations must share a shape (N, 2) with N >= 2")
    if not (np.all(np.isfinite(ca)) and np.all(np.isfinite(cr))):
        raise ValueError("autocorrelations must be finite")
    if not time_step > 0.0 or not broadening_time > 0.0:
        raise ValueError("time_step and broadening_time must be strictly positive")
    za = ca[:, 0] + 1j * ca[:, 1]
    zr = cr[:, 0] + 1j * cr[:, 1]
    t = time_step * np.arange(za.size)
    damp2 = np.exp(-(t / broadening_time) ** 2)
    # int I_1 I_2 domega = pi Re int_0^inf C_1 conj(C_2) exp(-t^2 / tau^2) dt
    cross = np.pi * np.trapezoid(np.real(za * np.conj(zr)) * damp2, t)
    norm_a = np.pi * np.trapezoid(np.abs(za) ** 2 * damp2, t)
    norm_r = np.pi * np.trapezoid(np.abs(zr) ** 2 * damp2, t)
    if norm_a <= 0.0 or norm_r <= 0.0:
        raise ValueError("lineshapes must have nonzero norm")
    return np.array([cross / np.sqrt(norm_a * norm_r), norm_a, norm_r])

import numpy as np


def method_diagnostics(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if not time_step > 0.0 or not broadening_time > 0.0:
        raise ValueError("time_step and broadening_time must be strictly positive")
    n = int(round(6.0 * broadening_time / time_step))
    if n < 1:
        raise ValueError("the propagation must contain at least one step")
    reference = exact_autocorrelation(displacement, surface_params, ground_params, q1_axis, q2_axis, time_step, n)
    cosines = []
    excursions = {}
    for mode in (0, 1, 2, 3):
        auto = thawed_gaussian_autocorrelation(displacement, surface_params, ground_params, mode, time_step, n)
        cosines.append(float(spectral_contrast_cosine(auto, reference, time_step, broadening_time)[0]))
        if mode in (0, 1):
            energy = thawed_gaussian_energy(displacement, surface_params, ground_params, mode, time_step, n)
            excursions[mode] = float(np.max(np.abs(energy - energy[0])))
    return np.array(cosines + [excursions[1], excursions[0]])

import numpy as np
from scipy.optimize import brentq


def _crossover_gap(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float) -> float:
    """Vertical minus adiabatic spectral contrast cosine at one displacement."""
    diag = method_diagnostics(displacement, surface_params, ground_params, q1_axis, q2_axis, time_step, broadening_time)
    return float(diag[2] - diag[1])


def vertical_adiabatic_crossover(surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float, d_min: float, d_max: float, n_scan: int) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not d_max > d_min or int(n_scan) < 2:
        raise ValueError("require d_max > d_min and n_scan >= 2")
    grid = np.linspace(d_min, d_max, int(n_scan))
    args = (surface_params, ground_params, q1_axis, q2_axis, time_step, broadening_time)
    previous = _crossover_gap(grid[0], *args)
    if previous >= 0.0:
        raise ValueError("the scan starts at or beyond the crossover")
    for k in range(1, grid.size):
        current = _crossover_gap(grid[k], *args)
        if current >= 0.0:
            return float(brentq(lambda d: _crossover_gap(d, *args), grid[k - 1], grid[k], xtol=1e-8))
    raise ValueError("no crossover within the scanned displacements")
SCICODE_GOLD_EOF
