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
from scipy.integrate import solve_ivp


def extract_radial_parameters(de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    for nm, v in (("de", de), ("re", re), ("c1", c1), ("c2", c2)):
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{nm} must be a positive finite number")
    if abs(c1 - 6.0) < 1e-12:
        raise ValueError("c1 must not equal 6: the prefactor de/(c1-6) is singular")
    pref = de / (c1 - 6.0)
    return np.array([pref,
                     2.0 * (3.0 - c2) * pref,
                     -(4.0 * c2 - c1 * c2 + c1) * pref,
                     -(c1 - 6.0) * c2 * pref], float)

import numpy as np
from scipy.integrate import solve_ivp


def radial_potential(r: "np.ndarray", de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    a = extract_radial_parameters(de, re, c1, c2)
    r = np.asarray(r)
    if not np.iscomplexobj(r):
        r = r.astype(float)
        if np.any(r <= 0.0):
            raise ValueError("r must be strictly positive")
    x = r / re
    return a[1] * np.exp(c1 * (1.0 - x)) + a[2] * x ** -6 + a[3] * x ** -4

import numpy as np
from scipy.integrate import solve_ivp


def angular_potential(x: "np.ndarray", y: "np.ndarray", z: "np.ndarray", ve: float, alpha: float, re: float, b: float) -> "np.ndarray":
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    cplx = np.iscomplexobj(x) or np.iscomplexobj(y) or np.iscomplexobj(z)
    if not cplx:
        x = x.astype(float); y = y.astype(float); z = z.astype(float)
    if not np.isfinite(ve) or not np.isfinite(alpha) or not np.isfinite(b):
        raise ValueError("ve, alpha and b must be finite")
    if b < 0.0:
        raise ValueError("b must be non-negative")
    r2 = x * x + y * y + z * z
    if not cplx and np.any(r2 <= 0.0):
        raise ValueError("the origin is not in the domain")
    r = np.sqrt(r2)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    return v0 * ((x * x + y * y) / r2 + b * (x ** 3 - 3.0 * x * y * y) / r ** 3)

import numpy as np
from scipy.integrate import solve_ivp


def hamiltonian(state: "np.ndarray", params: dict) -> float:
    s = np.asarray(state, float) if not np.iscomplexobj(state) else np.asarray(state)
    if s.shape != (6,):
        raise ValueError("state must have exactly six components (X,Y,Z,pX,pY,pZ)")
    for k in ("ix", "iz", "m", "de", "re", "c1", "c2", "ve", "alpha", "b"):
        if k not in params:
            raise ValueError(f"params is missing '{k}'")
    ix, iz, m = params["ix"], params["iz"], params["m"]
    if ix <= 0 or iz <= 0 or m <= 0:
        raise ValueError("ix, iz and m must be positive")
    R, p = s[:3], s[3:]
    lx = R[1] * p[2] - R[2] * p[1]
    ly = R[2] * p[0] - R[0] * p[2]
    lz = R[0] * p[1] - R[1] * p[0]
    r = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
    vch = radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    ang = angular_potential(R[0], R[1], R[2], params["ve"], params["alpha"],
                                    params["re"], params["b"])
    return (lx * lx + ly * ly) / (2.0 * ix) + lz * lz / (2.0 * iz) \
           + (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) / (2.0 * m) + vch + ang

import numpy as np
from scipy.integrate import solve_ivp


def vector_field(state: "np.ndarray", params: dict) -> "np.ndarray":
    s = np.asarray(state, float)
    if s.shape != (6,):
        raise ValueError("state must have exactly six components")
    out = np.empty(6); h = 1e-30
    for i in range(6):
        sc = s.astype(complex); sc[i] += 1j * h
        d = hamiltonian(sc, params).imag / h
        if i < 3:
            out[i + 3] = -d
        else:
            out[i - 3] = d
    return out

import numpy as np
from scipy.integrate import solve_ivp


def _planar_field(y, params):
    """(r, theta, p_r, p_theta) with p_phi = 0; regular because the cot^2 term drops."""
    r, th, pr, pth = y
    ix, m = params["ix"], params["m"]
    de, re, c1, c2 = params["de"], params["re"], params["c1"], params["c2"]
    ve, alpha, b = params["ve"], params["alpha"], params["b"]
    s, c = np.sin(th), np.cos(th)
    a = extract_radial_parameters(de, re, c1, c2)
    x = r / re
    dvch = (a[1] * np.exp(c1 * (1.0 - x)) * (-c1 / re)
            + a[2] * (-6.0) * x ** -7 / re + a[3] * (-4.0) * x ** -5 / re)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    dv0 = v0 * (-2.0 * alpha * (r - re))
    return np.array([pr / m,
                     pth * (1.0 / ix + 1.0 / (m * r * r)),
                     pth * pth / (m * r ** 3) - dvch - dv0 * (s * s + b * s ** 3),
                     -v0 * (2.0 * s * c + 3.0 * b * s * s * c)], float)

def _planar_pth(r, th, pr, energy, params):
    ix, m = params["ix"], params["m"]
    s = np.sin(th)
    vch = radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    v0 = params["ve"] * np.exp(-params["alpha"] * (r - params["re"]) ** 2)
    rest = energy - pr * pr / (2.0 * m) - vch - v0 * (s * s + params["b"] * s ** 3)
    coef = 1.0 / (2.0 * ix) + 1.0 / (2.0 * m * r * r)
    if rest <= 0.0:
        raise ValueError("the requested energy is not attainable at this (r, p_r)")
    return np.sqrt(rest / coef)

def planar_return_map(r: float, pr: float, energy: float, params: dict) -> "np.ndarray":
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("r must be a positive finite number")
    if not np.isfinite(pr):
        raise ValueError("pr must be finite")
    pth = _planar_pth(r, 0.0, pr, energy, params)
    ev = lambda t, y, q: y[1] - 2.0 * np.pi
    ev.terminal = True; ev.direction = 1.0
    s = solve_ivp(lambda t, y, q: _planar_field(y, q), [0.0, 30.0],
                  [r, 0.0, pr, pth], args=(params,), events=ev,
                  method='DOP853', rtol=1e-11, atol=1e-11)
    if not s.t_events[0].size:
        raise ValueError("the trajectory did not complete a full turn in theta")
    t = float(s.t_events[0][0]); y = s.y_events[0][0]
    return np.array([y[0], y[2], t], float)

import numpy as np
from scipy.integrate import solve_ivp


def _planar_field(y, params):
    """(r, theta, p_r, p_theta) with p_phi = 0; regular because the cot^2 term drops."""
    r, th, pr, pth = y
    ix, m = params["ix"], params["m"]
    de, re, c1, c2 = params["de"], params["re"], params["c1"], params["c2"]
    ve, alpha, b = params["ve"], params["alpha"], params["b"]
    s, c = np.sin(th), np.cos(th)
    a = extract_radial_parameters(de, re, c1, c2)
    x = r / re
    dvch = (a[1] * np.exp(c1 * (1.0 - x)) * (-c1 / re)
            + a[2] * (-6.0) * x ** -7 / re + a[3] * (-4.0) * x ** -5 / re)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    dv0 = v0 * (-2.0 * alpha * (r - re))
    return np.array([pr / m,
                     pth * (1.0 / ix + 1.0 / (m * r * r)),
                     pth * pth / (m * r ** 3) - dvch - dv0 * (s * s + b * s ** 3),
                     -v0 * (2.0 * s * c + 3.0 * b * s * s * c)], float)

def _planar_pth(r, th, pr, energy, params):
    ix, m = params["ix"], params["m"]
    s = np.sin(th)
    vch = radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    v0 = params["ve"] * np.exp(-params["alpha"] * (r - params["re"]) ** 2)
    rest = energy - pr * pr / (2.0 * m) - vch - v0 * (s * s + params["b"] * s ** 3)
    coef = 1.0 / (2.0 * ix) + 1.0 / (2.0 * m * r * r)
    if rest <= 0.0:
        raise ValueError("the requested energy is not attainable at this (r, p_r)")
    return np.sqrt(rest / coef)

def locate_periodic_orbit(r_guess: float, pr_guess: float, energy: float, params: dict) -> "np.ndarray":
    x = np.array([float(r_guess), float(pr_guess)])
    for _ in range(60):
        out = planar_return_map(x[0], x[1], energy, params)
        f = out[:2] - x
        if np.max(np.abs(f)) < 1e-12:
            return np.array([x[0], x[1], out[2]], float)
        J = np.empty((2, 2)); h = 1e-7
        for i in range(2):
            xp = x.copy(); xp[i] += h
            op = planar_return_map(xp[0], xp[1], energy, params)
            J[:, i] = ((op[:2] - xp) - f) / h
        dx = np.linalg.solve(J, -f)
        lam = 1.0
        while lam > 1e-5:
            xn = x + lam * dx
            try:
                on = planar_return_map(xn[0], xn[1], energy, params)
            except ValueError:
                lam *= 0.5; continue
            if np.max(np.abs(on[:2] - xn)) < np.max(np.abs(f)):
                break
            lam *= 0.5
        x = x + lam * dx
    raise ValueError("the Newton iteration did not converge to a periodic orbit")

import numpy as np
from scipy.integrate import solve_ivp


def _planar_field(y, params):
    """(r, theta, p_r, p_theta) with p_phi = 0; regular because the cot^2 term drops."""
    r, th, pr, pth = y
    ix, m = params["ix"], params["m"]
    de, re, c1, c2 = params["de"], params["re"], params["c1"], params["c2"]
    ve, alpha, b = params["ve"], params["alpha"], params["b"]
    s, c = np.sin(th), np.cos(th)
    a = extract_radial_parameters(de, re, c1, c2)
    x = r / re
    dvch = (a[1] * np.exp(c1 * (1.0 - x)) * (-c1 / re)
            + a[2] * (-6.0) * x ** -7 / re + a[3] * (-4.0) * x ** -5 / re)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    dv0 = v0 * (-2.0 * alpha * (r - re))
    return np.array([pr / m,
                     pth * (1.0 / ix + 1.0 / (m * r * r)),
                     pth * pth / (m * r ** 3) - dvch - dv0 * (s * s + b * s ** 3),
                     -v0 * (2.0 * s * c + 3.0 * b * s * s * c)], float)

def _planar_pth(r, th, pr, energy, params):
    ix, m = params["ix"], params["m"]
    s = np.sin(th)
    vch = radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    v0 = params["ve"] * np.exp(-params["alpha"] * (r - params["re"]) ** 2)
    rest = energy - pr * pr / (2.0 * m) - vch - v0 * (s * s + params["b"] * s ** 3)
    coef = 1.0 / (2.0 * ix) + 1.0 / (2.0 * m * r * r)
    if rest <= 0.0:
        raise ValueError("the requested energy is not attainable at this (r, p_r)")
    return np.sqrt(rest / coef)

def orbit_action(r0: float, pr0: float, energy: float, params: dict) -> float:
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError("r0 must be a positive finite number")
    if not np.isfinite(pr0):
        raise ValueError("pr0 must be finite")
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")
    out = locate_periodic_orbit(r0, pr0, energy, params)
    r, pr, T = out
    ix, m = params["ix"], params["m"]
    pth = _planar_pth(r, 0.0, pr, energy, params)
    def _aug(t, y, q):
        rr, th, p_r, p_th = y[:4]
        kin = p_th ** 2 * (1.0 / (2.0 * ix) + 1.0 / (2.0 * m * rr * rr)) + p_r * p_r / (2.0 * m)
        return np.concatenate([_planar_field(y[:4], q), [2.0 * kin]])
    s = solve_ivp(_aug, [0.0, T], [r, 0.0, pr, pth, 0.0], args=(params,),
                  method='DOP853', rtol=1e-12, atol=1e-12, max_step=0.5)
    return float(s.y[4, -1])

import numpy as np
from scipy.integrate import solve_ivp


def monodromy_matrix(state: "np.ndarray", params: dict, period: float) -> "np.ndarray":
    y0 = np.asarray(state, float)
    if y0.shape != (6,):
        raise ValueError("state must have exactly six components")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be a positive finite number")
    def _jac(y):
        J = np.empty((6, 6)); eps = 1e-6
        for i in range(6):
            yp = y.copy(); ym = y.copy(); yp[i] += eps; ym[i] -= eps
            J[:, i] = (vector_field(yp, params) - vector_field(ym, params)) / (2.0 * eps)
        return J
    def _aug(t, z, q):
        y = z[:6]; P = z[6:].reshape(6, 6)
        return np.concatenate([vector_field(y, q), (_jac(y) @ P).ravel()])
    z0 = np.concatenate([y0, np.eye(6).ravel()])
    s = solve_ivp(_aug, [0.0, period], z0, args=(params,), method='DOP853', rtol=1e-11, atol=1e-11)
    return s.y[6:, -1].reshape(6, 6)

import numpy as np
from scipy.integrate import solve_ivp


def _planar_field(y, params):
    """(r, theta, p_r, p_theta) with p_phi = 0; regular because the cot^2 term drops."""
    r, th, pr, pth = y
    ix, m = params["ix"], params["m"]
    de, re, c1, c2 = params["de"], params["re"], params["c1"], params["c2"]
    ve, alpha, b = params["ve"], params["alpha"], params["b"]
    s, c = np.sin(th), np.cos(th)
    a = extract_radial_parameters(de, re, c1, c2)
    x = r / re
    dvch = (a[1] * np.exp(c1 * (1.0 - x)) * (-c1 / re)
            + a[2] * (-6.0) * x ** -7 / re + a[3] * (-4.0) * x ** -5 / re)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    dv0 = v0 * (-2.0 * alpha * (r - re))
    return np.array([pr / m,
                     pth * (1.0 / ix + 1.0 / (m * r * r)),
                     pth * pth / (m * r ** 3) - dvch - dv0 * (s * s + b * s ** 3),
                     -v0 * (2.0 * s * c + 3.0 * b * s * s * c)], float)

def _planar_pth(r, th, pr, energy, params):
    ix, m = params["ix"], params["m"]
    s = np.sin(th)
    vch = radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    v0 = params["ve"] * np.exp(-params["alpha"] * (r - params["re"]) ** 2)
    rest = energy - pr * pr / (2.0 * m) - vch - v0 * (s * s + params["b"] * s ** 3)
    coef = 1.0 / (2.0 * ix) + 1.0 / (2.0 * m * r * r)
    if rest <= 0.0:
        raise ValueError("the requested energy is not attainable at this (r, p_r)")
    return np.sqrt(rest / coef)

def roaming_stability_report(energy: float, b: float, n_steps: int) -> "np.ndarray":
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy must be a positive finite number")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("b must be a non-negative finite number")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be at least one")
    ix = 2.373409
    base = dict(ix=ix, iz=2.0 * ix, m=0.9445, de=47.0, re=1.1,
                c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0)
    # the FR1 seed at the anchor point E = 0.5, b = 0 is found by a bracketed scan
    seed = None
    for rr in np.linspace(3.55, 3.75, 41):
        try:
            out = locate_periodic_orbit(rr, 0.0, 0.5, base)
        except ValueError:
            continue
        if 3.0 < out[0] < 3.9:
            seed = np.array([out[0], out[1]]); break
    if seed is None:
        raise ValueError("could not locate the FR1 anchor orbit")
    # continue first in b at E = 0.5, then in energy at the requested b
    ns = int(n_steps)
    for bb in np.linspace(0.0, float(b), ns + 1)[1:]:
        params = dict(base); params["b"] = float(bb)
        o = locate_periodic_orbit(seed[0], seed[1], 0.5, params)
        seed = np.array([o[0], o[1]])
    params = dict(base); params["b"] = float(b)
    for ee in np.linspace(0.5, float(energy), ns + 1)[1:]:
        o = locate_periodic_orbit(seed[0], seed[1], float(ee), params)
        seed = np.array([o[0], o[1]])
    orb = locate_periodic_orbit(seed[0], seed[1], energy, params)
    r0, pr0, T = orb
    # the located point must be a genuine fixed point of the return map
    back = planar_return_map(r0, pr0, energy, params)
    if abs(back[0] - r0) > 1e-8 or abs(back[1] - pr0) > 1e-8:
        raise ValueError("the continuation did not end on a periodic orbit")
    W = orbit_action(r0, pr0, energy, params)
    # rebuild the Cartesian state and check it against the energy and the potentials
    coeff = extract_radial_parameters(params["de"], params["re"], params["c1"], params["c2"])
    if not np.isfinite(coeff[0]):
        raise ValueError("the radial coefficients are not finite")
    pth = _planar_pth(r0, 0.0, pr0, energy, params)
    y0 = np.array([0.0, 0.0, r0, pth / r0, 0.0, pr0])
    vrad = radial_potential(r0, params["de"], params["re"], params["c1"], params["c2"])
    vang = angular_potential(y0[0], y0[1], y0[2], params["ve"], params["alpha"],
                                     params["re"], params["b"])
    if not np.isfinite(float(vrad) + float(vang)):
        raise ValueError("the potential is not finite on the orbit")
    if abs(hamiltonian(y0, params) - energy) > 1e-8:
        raise ValueError("the reconstructed state does not sit at the requested energy")
    if abs(vector_field(y0, params)[1]) > 1e-12:
        raise ValueError("the reaction plane is not invariant at the start of the orbit")
    M = monodromy_matrix(y0, params, T)
    mt = M[np.ix_([1, 4], [1, 4])]
    inp = M[np.ix_([0, 2, 3, 5], [0, 2, 3, 5])]
    trace = float(np.trace(mt))
    lam_in = float(np.max(np.abs(np.linalg.eigvals(inp))))
    return np.array([trace, r0, pr0, T, W, lam_in], float)
SCICODE_GOLD_EOF
