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


def stiffened_gas_pressure(rho: "np.ndarray", e: "np.ndarray", gamma: float, pi_: float) -> "np.ndarray":
    rho = np.asarray(rho, dtype=np.float64); e = np.asarray(e, dtype=np.float64)
    if gamma <= 1.0 or pi_ < 0.0:
        raise ValueError("need gamma > 1 and pi >= 0")
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    return (gamma - 1.0) * rho * e - gamma * pi_

import numpy as np


def interface_closures(phi: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", rho1: "np.ndarray", rho2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray") -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64)
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("volume fraction must lie in [0, 1]")
    m1 = phi * np.asarray(rho1, dtype=np.float64)
    m2 = (1.0 - phi) * np.asarray(rho2, dtype=np.float64)
    if np.any(m1 + m2 <= 0.0):
        raise ValueError("mixture density must be positive")
    pI = phi * np.asarray(p1, dtype=np.float64) + (1.0 - phi) * np.asarray(p2, dtype=np.float64)
    uI = (m1 * np.asarray(u1, dtype=np.float64) + m2 * np.asarray(u2, dtype=np.float64)) / (m1 + m2)
    return np.stack([pI, uI])

import numpy as np


def interface_distance(phi: "np.ndarray", eps: float, delta: float) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64)
    if eps <= 0.0 or not (0.0 <= delta < 0.5):
        raise ValueError("need eps > 0 and delta in [0, 0.5)")
    lo = np.maximum(phi - delta, 0.0) + 1.0e-100
    hi = np.maximum((1.0 - delta) - phi, 0.0) + 1.0e-100
    return eps * np.log(lo / hi)

import numpy as np


def _ddx(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def _ddy(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def interface_normal(psi: "np.ndarray", h: float) -> "np.ndarray":
    psi = np.asarray(psi, dtype=np.float64)
    if psi.ndim != 2:
        raise ValueError("psi must be a two-dimensional field")
    if h <= 0.0:
        raise ValueError("h must be positive")
    gx = _ddx(psi, h); gy = _ddy(psi, h)
    g = np.sqrt(gx * gx + gy * gy)
    g = np.where(g > 0.0, g, 1.0)
    return np.stack([gx / g, gy / g])

import numpy as np


def _ddx(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def _ddy(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def interface_regularization_flux(phi: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional field")
    if Gamma < 0.0:
        raise ValueError("the regularisation velocity cannot be negative")
    psi = interface_distance(phi, eps, delta)
    n = interface_normal(psi, h)
    sharp = ((1.0 - 2.0 * delta) ** 2 / 4.0) * (1.0 - np.tanh(psi / (2.0 * eps)) ** 2)
    ax = Gamma * (eps * (1.0 - 2.0 * delta) * _ddx(phi, h) - sharp * n[0])
    ay = Gamma * (eps * (1.0 - 2.0 * delta) * _ddy(phi, h) - sharp * n[1])
    return np.stack([ax, ay])

import numpy as np


def _ddx(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def _ddy(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def volume_fraction_rhs(phi: "np.ndarray", uI: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64); uI = np.asarray(uI, dtype=np.float64)
    if phi.ndim != 2 or uI.shape != (2,) + phi.shape:
        raise ValueError("phi must be a two-dimensional field and uI its (2, N, N) velocity")
    a = interface_regularization_flux(phi, eps, delta, Gamma, h)
    div_u = _ddx(uI[0], h) + _ddy(uI[1], h)
    return -(_ddx(phi * uI[0], h) + _ddy(phi * uI[1], h)) + phi * div_u + _ddx(a[0], h) + _ddy(a[1], h)

import numpy as np


def _ddx(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def _ddy(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def phasic_mass_rhs(phi: "np.ndarray", rho: "np.ndarray", u: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, phase_one: bool) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64); rho = np.asarray(rho, dtype=np.float64); u = np.asarray(u, dtype=np.float64)
    if phi.ndim != 2 or u.shape != (2,) + phi.shape:
        raise ValueError("phi must be a two-dimensional field and u its (2, N, N) velocity")
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    a = interface_regularization_flux(phi, eps, delta, Gamma, h)
    vf = phi if phase_one else (1.0 - phi)
    sgn = 1.0 if phase_one else -1.0
    conv = _ddx(vf * rho * u[0], h) + _ddy(vf * rho * u[1], h)
    return -conv + sgn * (_ddx(rho * a[0], h) + _ddy(rho * a[1], h))

import numpy as np


def advance(phi: "np.ndarray", m1: "np.ndarray", m2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, dt: float) -> "np.ndarray":
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    phi = np.asarray(phi, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64); m2 = np.asarray(m2, dtype=np.float64)
    u1 = np.asarray(u1, dtype=np.float64); u2 = np.asarray(u2, dtype=np.float64)
    if phi.ndim != 2 or u1.shape != (2,) + phi.shape or u2.shape != (2,) + phi.shape:
        raise ValueError("phi must be a two-dimensional field and u1, u2 its (2, N, N) velocities")
    if m1.shape != phi.shape or m2.shape != phi.shape or np.any(m1 <= 0.0) or np.any(m2 <= 0.0):
        raise ValueError("the phasic masses must be positive fields shaped like phi")

    def _rhs(ph, a1, a2):
        r1 = a1 / np.maximum(ph, delta)
        r2 = a2 / np.maximum(1.0 - ph, delta)
        uIx = interface_closures(ph, p1, p2, r1, r2, u1[0], u2[0])[1]
        uIy = interface_closures(ph, p1, p2, r1, r2, u1[1], u2[1])[1]
        uI = np.stack([uIx, uIy])
        return (volume_fraction_rhs(ph, uI, eps, delta, Gamma, h),
                phasic_mass_rhs(ph, r1, u1, eps, delta, Gamma, h, True),
                phasic_mass_rhs(ph, r2, u2, eps, delta, Gamma, h, False), uI)

    k = _rhs(phi, m1, m2)
    p_1 = phi + dt * k[0]
    a_1, b_1 = m1 + dt * k[1], m2 + dt * k[2]
    k2 = _rhs(p_1, a_1, b_1)
    p_n = 0.5 * (phi + p_1 + dt * k2[0])
    a_n = 0.5 * (m1 + a_1 + dt * k2[1])
    b_n = 0.5 * (m2 + b_1 + dt * k2[2])
    return np.stack([p_n, a_n, b_n, k[3][0], k[3][1]])

import numpy as np


def _ddx(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * h)


def _ddy(f: "np.ndarray", h: float) -> "np.ndarray":
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * h)


def _configuration() -> "tuple[dict, list]":
    """Material data and the three (N, R0, alpha, cx, cy, steps) configurations."""
    pars = dict(delta=1.0e-2, ceps=2.0, cfl=0.25, rho1=1.0e3, rho2=1.2, p0=1.0e5,
                gamma1=4.4, pi1=6.0e8, gamma2=1.4, pi2=0.0)
    cfgs = [(96, 0.15, 0.4, 0.5, 0.5, 160), (128, 0.12, 0.7, 0.25, 0.5, 100), (80, 0.18, 0.25, 0.3, 0.3, 107)]
    return pars, cfgs


def vortex_audit(vel_scale: float) -> "np.ndarray":
    if isinstance(vel_scale, bool) or not np.isfinite(vel_scale) or vel_scale <= 0.0:
        raise ValueError("vel_scale must be positive and finite")
    P, cfgs = _configuration()
    delta = P["delta"]
    rows = []
    for (N, R0, alpha, cx, cy, nstep) in cfgs:
        h = 1.0 / N
        eps = P["ceps"] * h
        x = (np.arange(N) + 0.5) * h
        X, Y = np.meshgrid(x, x, indexing="ij")
        wx = (X - cx + 0.5) % 1.0 - 0.5
        wy = (Y - cy + 0.5) % 1.0 - 0.5
        d = R0 - np.sqrt(wx * wx + wy * wy)
        phi = delta + (1.0 - 2.0 * delta) * 0.5 * (1.0 + np.tanh(d / (2.0 * eps)))
        phi0 = phi.copy()
        ux = np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
        uy = -np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
        u2 = float(vel_scale) * np.stack([ux, uy])
        u1 = alpha * u2
        Gamma = float(max(np.max(np.hypot(u1[0], u1[1])), np.max(np.hypot(u2[0], u2[1]))))
        dt = P["cfl"] * h / max(Gamma, 1.0e-30)
        p1 = np.full((N, N), P["p0"]); p2 = np.full((N, N), P["p0"])
        r1 = np.full((N, N), P["rho1"]); r2 = np.full((N, N), P["rho2"])
        e1 = (p1 + P["gamma1"] * P["pi1"]) / ((P["gamma1"] - 1.0) * r1)
        e2 = (p2 + P["gamma2"] * P["pi2"]) / ((P["gamma2"] - 1.0) * r2)
        if (np.max(np.abs(stiffened_gas_pressure(r1, e1, P["gamma1"], P["pi1"]) - p1)) > 1e-6 * P["p0"]
                or np.max(np.abs(stiffened_gas_pressure(r2, e2, P["gamma2"], P["pi2"]) - p2)) > 1e-6 * P["p0"]):
            raise ValueError("the equation of state does not invert consistently")
        a0 = interface_regularization_flux(phi, eps, delta, Gamma, h)
        res = float(np.max(np.hypot(a0[0], a0[1])))
        m1 = phi * r1; m2 = (1.0 - phi) * r2
        m0 = float(np.sum(m1 + m2)) * h * h
        for _ in range(nstep):
            out = advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt)
            phi, m1, m2 = out[0], out[1], out[2]
        if abs(float(np.sum(m1 + m2)) * h * h - m0) / m0 > 1.0e-10:
            raise ValueError("the regularisation is not in divergence form: mixture mass drifted")
        if not np.all(np.isfinite(phi)) or np.min(phi) < 0.0 or np.max(phi) > 1.0:
            raise ValueError("the volume fraction is not a finite field in [0, 1]")
        gx = _ddx(phi, h); gy = _ddy(phi, h)
        area = float(np.sum(phi)) * h * h
        area0 = float(np.sum(phi0)) * h * h
        L = float(np.sum(phi * (1.0 - phi))) * h * h
        moved = float(np.sum(np.abs(phi - phi0))) * h * h
        rows.append([res * 1.0e6, area / area0, float(np.max(np.hypot(gx, gy))), float(np.min(phi)), moved * 1.0e3, L * 1.0e3])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
