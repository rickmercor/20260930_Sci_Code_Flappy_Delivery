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


def response_spectrum(k: "np.ndarray", sigma: float) -> "np.ndarray":
    k = np.asarray(k, dtype=float)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return np.exp(-0.25 * (k * sigma) ** 2)

import numpy as np


def width_acceleration(a: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    a = float(a); power = float(power); gamma = float(gamma)
    alpha = float(alpha); sigma = float(sigma)
    if a <= 0.0 or power <= 0.0 or sigma <= 0.0:
        raise ValueError("a, power and sigma must be positive")
    return float(1.0 / a ** 3
                 - 2.0 * alpha ** 2 * a
                 - power * gamma / (np.sqrt(2.0 * np.pi) * a ** 2)
                 - 2.0 * power * a / (np.sqrt(np.pi) * (2.0 * a ** 2 + sigma ** 2) ** 1.5))

import numpy as np


def width_curvature(a: float, power: float, gamma: float, alpha: float,
                            sigma: float) -> float:
    """Curvature of the effective potential at width a: minus d(a'')/da."""
    a = float(a); power = float(power); sigma = float(sigma)
    if a <= 0.0 or power <= 0.0:
        raise ValueError("a and power must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return float(3.0 / a ** 4
                 + 2.0 * alpha ** 2
                 - 2.0 * power * gamma / (np.sqrt(2.0 * np.pi) * a ** 3)
                 + 2.0 * power / np.sqrt(np.pi)
                   * (sigma ** 2 - 4.0 * a ** 2) / (2.0 * a ** 2 + sigma ** 2) ** 2.5)

import numpy as np


def potential_change(a1: float, a2: float, power: float, gamma: float, alpha: float,
                             sigma: float) -> float:
    """Change in the effective potential between two widths: V(a2) - V(a1), with a'' = -dV/da."""
    a1 = float(a1); a2 = float(a2); power = float(power); sigma = float(sigma)
    if a1 <= 0.0 or a2 <= 0.0 or power <= 0.0 or sigma <= 0.0:
        raise ValueError("a1, a2, power and sigma must be positive")

    def _v(a: float) -> float:
        return (1.0 / (2.0 * a ** 2)
                + alpha ** 2 * a ** 2
                - power * gamma / (np.sqrt(2.0 * np.pi) * a)
                - power / (np.sqrt(np.pi) * np.sqrt(2.0 * a ** 2 + sigma ** 2)))

    return float(_v(a2) - _v(a1))

import numpy as np


def equilibrium_width(power: float, gamma: float, alpha: float, sigma: float,
                              lo: float = 1e-3, hi: float = 50.0) -> float:
    lo = float(lo); hi = float(hi)
    if not (0.0 < lo < hi):
        raise ValueError("require 0 < lo < hi")
    f = lambda a: width_acceleration(a, power, gamma, alpha, sigma)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("no sign change for the equilibrium condition on [lo, hi]")
    for _ in range(200):                      # bisection: deterministic, no scipy
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return float(0.5 * (lo + hi))

import numpy as np


def adiabatic_reference(z: float, zf: float, power: float, gamma: float,
                                alpha: float, sigma_i: float, sigma_f: float) -> float:
    """Equilibrium width at the response length a linear ramp has reached at z."""
    z = float(z); zf = float(zf)
    if zf <= 0.0:
        raise ValueError("zf must be positive")
    s = min(max(z / zf, 0.0), 1.0)
    sigma = sigma_i + (sigma_f - sigma_i) * s
    if sigma <= 0.0:
        raise ValueError("the ramped response length must stay positive")
    return equilibrium_width(power, gamma, alpha, sigma)

import numpy as np


def minimum_jerk_width(z: float, zf: float, a_i: float, a_f: float) -> "np.ndarray":
    z = float(z); zf = float(zf)
    if zf <= 0.0:
        raise ValueError("zf must be positive")
    s = min(max(z / zf, 0.0), 1.0)
    a = a_i + (a_f - a_i) * (10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5)
    add = (a_f - a_i) * (60.0 * s - 180.0 * s ** 2 + 120.0 * s ** 3) / zf ** 2
    return np.array([a, add], dtype=float)

import numpy as np


def inverse_control(knob: str, a: float, add: float, power: float,
                            gamma: float, alpha: float, sigma: float) -> float:
    """Invert Eq. (7) for one control knob so that the width follows (a, a'')."""
    a = float(a); add = float(add); power = float(power)
    if a <= 0.0 or power <= 0.0:
        raise ValueError("a and power must be positive")
    if knob == "sigma":
        rhs = (1.0 / a ** 3 - 2.0 * alpha ** 2 * a
               - power * gamma / (np.sqrt(2.0 * np.pi) * a ** 2) - add)
        if rhs <= 0.0:
            raise ValueError("no positive nonlocal length reproduces this trajectory")
        val = (2.0 * power * a / (np.sqrt(np.pi) * rhs)) ** (2.0 / 3.0) - 2.0 * a ** 2
        if val <= 0.0:
            raise ValueError("no positive nonlocal length reproduces this trajectory")
        return float(np.sqrt(val))
    if knob == "gamma":
        rest = (1.0 / a ** 3 - 2.0 * alpha ** 2 * a
                - 2.0 * power * a / (np.sqrt(np.pi) * (2.0 * a ** 2 + sigma ** 2) ** 1.5) - add)
        return float(rest * np.sqrt(2.0 * np.pi) * a ** 2 / power)
    if knob == "alpha2":
        rest = (1.0 / a ** 3 - power * gamma / (np.sqrt(2.0 * np.pi) * a ** 2)
                - 2.0 * power * a / (np.sqrt(np.pi) * (2.0 * a ** 2 + sigma ** 2) ** 1.5) - add)
        return float(rest / (2.0 * a))
    raise ValueError("knob must be 'sigma', 'gamma' or 'alpha2'")

import numpy as np


def _grid(npts: int, half_width: float) -> "np.ndarray":
    npts = int(npts); half_width = float(half_width)
    if npts < 8 or npts % 2 != 0:
        raise ValueError("npts must be an even integer of at least 8")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    dx = 2.0 * half_width / npts
    x = -half_width + dx * np.arange(npts)
    k = 2.0 * np.pi * np.fft.fftfreq(npts, d=dx)
    return np.stack([x, k])

def propagate_field(u0: "np.ndarray", npts: int, half_width: float, controls: "np.ndarray",
                      dz: float, gamma: float, alpha: float, sigma: float, knob: str) -> "np.ndarray":
    """Symmetric split-step march; controls[j] is the knob value on step j."""
    controls = np.asarray(controls, dtype=float)
    dz = float(dz)
    if dz <= 0.0 or controls.ndim != 1 or controls.size < 1:
        raise ValueError("dz must be positive and controls a non-empty 1-D array")
    g = _grid(npts, half_width)
    x, k = g[0], g[1]
    u = np.asarray(u0, dtype=complex).copy()
    if u.shape != x.shape:
        raise ValueError("u0 must have the same length as the grid")
    half = np.exp(-0.25j * dz * k ** 2)
    for c in controls:
        s, gm, al = sigma, gamma, alpha
        if knob == "sigma":
            s = c
        elif knob == "gamma":
            gm = c
        elif knob == "alpha2":
            al = np.sqrt(c) if c >= 0.0 else 0.0
        else:
            raise ValueError("knob must be 'sigma', 'gamma' or 'alpha2'")
        rhat = response_spectrum(k, s)
        u = np.fft.ifft(half * np.fft.fft(u))
        a2 = np.abs(u) ** 2
        phase = np.real(np.fft.ifft(rhat * np.fft.fft(a2))) + gm * a2 - (al ** 2 if knob != "alpha2" else c) * x ** 2
        u = u * np.exp(1j * dz * phase)
        u = np.fft.ifft(half * np.fft.fft(u))
    return u

import numpy as np


def _grid(npts: int, half_width: float) -> "np.ndarray":
    npts = int(npts); half_width = float(half_width)
    if npts < 8 or npts % 2 != 0:
        raise ValueError("npts must be an even integer of at least 8")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    dx = 2.0 * half_width / npts
    x = -half_width + dx * np.arange(npts)
    k = 2.0 * np.pi * np.fft.fftfreq(npts, d=dx)
    return np.stack([x, k])

def stationary_soliton(npts: int, half_width: float, power: float, gamma: float,
                               alpha: float, sigma: float, dtau: float, nsteps: int) -> "np.ndarray":
    """Imaginary-time relaxation at fixed power; returns the real, positive profile."""
    dtau = float(dtau); nsteps = int(nsteps)
    if dtau <= 0.0 or nsteps < 1:
        raise ValueError("dtau must be positive and nsteps a positive integer")
    if power <= 0.0:
        raise ValueError("power must be positive")
    g = _grid(npts, half_width)
    x, k = g[0], g[1]
    dx = x[1] - x[0]
    rhat = response_spectrum(k, sigma)
    u = np.exp(-0.5 * x ** 2).astype(complex)
    u *= np.sqrt(power / (np.sum(np.abs(u) ** 2) * dx))
    half = np.exp(-0.25 * dtau * k ** 2)
    for _ in range(nsteps):
        u = np.fft.ifft(half * np.fft.fft(u))
        a2 = np.abs(u) ** 2
        phase = np.real(np.fft.ifft(rhat * np.fft.fft(a2))) + gamma * a2 - alpha ** 2 * x ** 2
        u = u * np.exp(dtau * phase)
        u = np.fft.ifft(half * np.fft.fft(u))
        u *= np.sqrt(power / (np.sum(np.abs(u) ** 2) * dx))
    return np.abs(u)

import numpy as np


def beam_width(u: "np.ndarray", x: "np.ndarray", dx: float) -> float:
    u = np.asarray(u, dtype=complex); x = np.asarray(x, dtype=float)
    dx = float(dx)
    if u.shape != x.shape:
        raise ValueError("u and x must have the same shape")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    a2 = np.abs(u) ** 2
    p = np.sum(a2) * dx
    if p <= 0.0:
        raise ValueError("the field must carry positive power")
    xc = np.sum(x * a2) * dx / p
    return float(np.sqrt(2.0 * np.sum((x - xc) ** 2 * a2) * dx / p))

import numpy as np


def overlap_fidelity(u: "np.ndarray", target: "np.ndarray", dx: float) -> float:
    u = np.asarray(u, dtype=complex); target = np.asarray(target, dtype=complex)
    dx = float(dx)
    if u.shape != target.shape:
        raise ValueError("u and target must have the same shape")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    nu = np.sum(np.abs(u) ** 2) * dx
    nt = np.sum(np.abs(target) ** 2) * dx
    if nu <= 0.0 or nt <= 0.0:
        raise ValueError("both fields must carry positive power")
    ov = np.sum(np.conj(target) * u) * dx
    return float(np.abs(ov) ** 2 / (nu * nt))

import numpy as np


def confinement_floor(zf: float, power: float, gamma: float, alpha: float,
                              sigma_i: float, sigma_f: float) -> float:
    """Smallest waveguide strength the confinement-knob shortcut of length zf demands."""
    zf = float(zf)
    if zf <= 0.0:
        raise ValueError("zf must be positive")
    a_i = equilibrium_width(power, gamma, alpha, sigma_i)
    a_f = equilibrium_width(power, gamma, alpha, sigma_f)

    val = lambda z: inverse_control("alpha2",
                                           *minimum_jerk_width(z, zf, a_i, a_f),
                                           power, gamma, alpha, sigma_i)
    zs = np.linspace(0.0, zf, 401)
    vs = np.array([val(z) for z in zs])
    j = int(np.argmin(vs))
    lo = zs[max(j - 1, 0)]
    hi = zs[min(j + 1, zs.size - 1)]
    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    for _ in range(200):                      # golden section: no scipy, deterministic
        if hi - lo <= 1e-12:
            break
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        if val(c) < val(d):
            hi = d
        else:
            lo = c
    return float(min(val(0.5 * (lo + hi)), vs[0], vs[-1]))

import numpy as np


def antiguiding_onset(power: float, gamma: float, alpha: float, sigma_i: float,
                              sigma_f: float, lo: float = 1e-2, hi: float = 20.0) -> float:
    """Shortest shortcut length whose confinement profile stays everywhere guiding."""
    lo = float(lo); hi = float(hi)
    if not (0.0 < lo < hi):
        raise ValueError("require 0 < lo < hi")
    f = lambda z: confinement_floor(z, power, gamma, alpha, sigma_i, sigma_f)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("no antiguiding onset is bracketed by [lo, hi]")
    for _ in range(200):                      # bisection: deterministic, no scipy
        if hi - lo <= 1e-12:
            break
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return float(0.5 * (lo + hi))

import numpy as np


def _grid(npts: int, half_width: float) -> "np.ndarray":
    npts = int(npts); half_width = float(half_width)
    if npts < 8 or npts % 2 != 0:
        raise ValueError("npts must be an even integer of at least 8")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    dx = 2.0 * half_width / npts
    x = -half_width + dx * np.arange(npts)
    k = 2.0 * np.pi * np.fft.fftfreq(npts, d=dx)
    return np.stack([x, k])

def shortcut_audit(refine: int) -> "np.ndarray":
    """End-to-end audit of the three inverse-engineered shortcut protocols.

    Slots, in order: the final beam width under the nonlocal-length knob, under the
    Kerr knob and under the confinement knob; the initial and target equilibrium
    widths; the final width under a linear nonlocal-length ramp; the adiabatic
    reference width at the midpoint of that ramp; the adiabatic-following ratio; the
    effective-potential change; the fidelity of the nonlocal-length protocol; and the shortest confinement-knob protocol length that stays everywhere guiding.
    """
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    power, gamma, alpha = 12.0, 0.2, 0.15
    sigma_i, sigma_f, zf = 4.0, 0.8, 2.2
    npts, half_width = 1024 * refine, 30.0
    nz = 2000 * refine

    g = _grid(npts, half_width)
    x, k = g[0], g[1]
    dx = x[1] - x[0]
    a_i = equilibrium_width(power, gamma, alpha, sigma_i)
    a_f = equilibrium_width(power, gamma, alpha, sigma_f)

    amp = np.sqrt(power / (np.sqrt(np.pi) * a_i))
    u0 = (amp * np.exp(-x ** 2 / (2.0 * a_i ** 2))).astype(complex)

    dz = zf / nz
    zc = (np.arange(nz) + 0.5) * dz
    widths = []
    u_sigma = None
    for knob in ("sigma", "gamma", "alpha2"):
        ctrl = np.array([inverse_control(knob,
                                                 *minimum_jerk_width(z, zf, a_i, a_f),
                                                 power, gamma, alpha, sigma_i) for z in zc])
        u = propagate_field(u0, npts, half_width, ctrl, dz, gamma, alpha, sigma_i, knob)
        widths.append(beam_width(u, x, dx))
        if knob == "sigma":
            u_sigma = u

    ramp = sigma_i + (sigma_f - sigma_i) * zc / zf
    u_ramp = propagate_field(u0, npts, half_width, ramp, dz, gamma, alpha, sigma_i, "sigma")
    a_ref = adiabatic_reference(0.5 * zf, zf, power, gamma, alpha, sigma_i, sigma_f)

    zs = np.linspace(0.0, zf, 2001)
    hs = zs[1] - zs[0]
    ac = np.array([adiabatic_reference(z, zf, power, gamma, alpha, sigma_i, sigma_f)
                   for z in zs])
    acdd = (ac[2:] - 2.0 * ac[1:-1] + ac[:-2]) / hs ** 2
    s_mid = sigma_i + (sigma_f - sigma_i) * zs[1:-1] / zf
    om2 = np.array([width_curvature(a, power, gamma, alpha, s)
                    for a, s in zip(ac[1:-1], s_mid)])
    ratio = float(np.max(np.abs(acdd) / (om2 * ac[1:-1])))

    depth = potential_change(a_i, a_f, power, gamma, alpha, sigma_i)

    target = stationary_soliton(npts, half_width, power, gamma, alpha,
                                        sigma_f, 0.01, 2000)
    fid = overlap_fidelity(u_sigma, target, dx)

    onset = antiguiding_onset(power, gamma, alpha, sigma_i, sigma_f)

    return np.array([widths[0], widths[1], widths[2], a_i, a_f,
                     beam_width(u_ramp, x, dx), a_ref, ratio, depth, fid,
                     onset], dtype=float)
SCICODE_GOLD_EOF
