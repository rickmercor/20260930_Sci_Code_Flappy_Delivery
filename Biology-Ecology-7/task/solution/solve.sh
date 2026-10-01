#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math

import math


def single_pest_orbit(a: float, b: float, h: float, tau: float) -> float:
    a = float(a); b = float(b); h = float(h); tau = float(tau)
    for v in (a, b, h, tau):
        if not math.isfinite(v):
            raise ValueError("inputs must be finite")
    if a <= 0.0 or b <= 0.0 or tau <= 0.0 or h <= -1.0:
        raise ValueError("require a > 0, b > 0, tau > 0 and h > -1")
    if a * tau + math.log1p(h) <= 0.0:
        raise ValueError("no positive tau-periodic orbit: a*tau + ln(1+h) <= 0")
    E = math.exp(a * tau)
    return float((a / b) * ((1.0 + h) * E - 1.0) / (E - 1.0))

import math
import numpy as np
from scipy.integrate import solve_ivp

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq

def _s02_params(params):
    """Validate the parameter dictionary and return float arrays (a, B, e, c, om, d, h)."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "e", "c", "om", "d", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    e = np.array(params["e"], dtype=float).ravel()
    c = np.array(params["c"], dtype=float).ravel()
    om = np.array(params["om"], dtype=float).ravel()
    d = float(params["d"])
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or e.shape != (3,) or c.shape != (3,) or om.shape != (3,) or h.shape != (4,):
        raise ValueError("expected a, e, c, om of length 3, B of shape 3x3 and h of length 4")
    for arr in (a, B, e, c, om, h):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite parameter")
    if not math.isfinite(d):
        raise ValueError("non-finite parameter d")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(e <= 0.0) or np.any(c <= 0.0) or d <= 0.0:
        raise ValueError("a_i, B_ii, e_i, c_i and d must be positive")
    if np.any(B < 0.0) or np.any(om < 0.0):
        raise ValueError("B_ij and om_i must be non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, e, c, om, d, h


def _s02_rates(z, P):
    """Per-capita growth rates (f_A, f_B, f_C, f_P) at the state z = (N_A, N_B, N_C, N_P)."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    D = 1.0 + om @ n
    f = np.empty(4)
    f[:3] = a - B @ n - e * z[3] / D
    f[3] = -d + (c @ n) / D
    return f


def _s02_jac(z, P):
    """Jacobian matrix d f_i / d z_j of the per-capita rates."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    w = z[3]
    D = 1.0 + om @ n
    J = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            J[i, j] = -B[i, j] + e[i] * w * om[j] / D ** 2
        J[i, 3] = -e[i] / D
    num = c @ n
    for j in range(3):
        J[3, j] = (c[j] * D - num * om[j]) / D ** 2
    return J


def _s02_rhs_var(t, y, P):
    """Flow, variational (fundamental-matrix) equations and running integrals of the per-capita rates."""
    z = y[:4]
    Phi = y[4:20].reshape(4, 4)
    f = _s02_rates(z, P)
    A = np.diag(f) + z[:, None] * _s02_jac(z, P)
    return np.concatenate([z * f, (A @ Phi).ravel(), f])


def _s02_map_full(z0, tau, P):
    """One-interval map: post-spray state, its Jacobian and the rate integrals over one interval."""
    y0 = np.concatenate([np.asarray(z0, dtype=float), np.eye(4).ravel(), np.zeros(4)])
    sol = solve_ivp(_s02_rhs_var, (0.0, tau), y0, method="DOP853", rtol=1e-13, atol=1e-16, args=(P,))
    if not sol.success:
        raise ValueError("integration failed: %s" % sol.message)
    y = sol.y[:, -1]
    g = 1.0 + P[6]
    return g * y[:4], g[:, None] * y[4:20].reshape(4, 4), y[20:24]


def _s02_face_orbit(present, guess, tau, P, tol=1e-12, maxit=60):
    """Newton shooting for the tau-periodic orbit on the face spanned by `present` (list of indices)."""
    present = list(present)
    k = len(present)
    z = np.zeros(4)
    z[present] = np.asarray(guess, dtype=float)
    if np.any(z[present] <= 0.0):
        raise ValueError("starting guess must be positive on the face")
    for _ in range(maxit):
        pz, Dpi, L = _s02_map_full(z, tau, P)
        F = pz[present] - z[present]
        if np.max(np.abs(F)) <= tol:
            return z, Dpi, L
        step = np.linalg.solve(Dpi[np.ix_(present, present)] - np.eye(k), -F)
        lam = 1.0
        while np.any(z[present] + lam * step <= 0.0):
            lam *= 0.5
            if lam < 1e-8:
                raise ValueError("Newton shooting left the positive face")
        z[present] = z[present] + lam * step
    raise ValueError("Newton shooting did not converge to the required residual")


def _s02_rates_on_orbit(z0, tau, P):
    """Pulse-adjusted long-term per-unit-time growth rates (1/tau)[int f_i dt + ln(1+h_i)] along the orbit through z0."""
    pz, Dpi, L = _s02_map_full(z0, tau, P)
    return (L + np.log1p(P[6])) / tau


def stroboscopic_map(params: dict, z0: object, tau: float, which: int) -> float:
    P = _s02_params(params)
    z = np.array(z0, dtype=float).ravel()
    tau = float(tau)
    if z.shape != (4,) or not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("z0 must be four finite non-negative densities")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    which = int(which)
    if which < 0 or which > 7:
        raise ValueError("which must be in 0..7")
    pz, Dpi, L = _s02_map_full(z, tau, P)
    if which < 4:
        return float(pz[which])
    return float(L[which - 4])

import math
import numpy as np

import math
import numpy as np


def _s03_params(params):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or h.shape != (4,):
        raise ValueError("expected a of length 3, B of shape 3x3 and h of length 4")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(B)) and np.all(np.isfinite(h))):
        raise ValueError("non-finite parameter")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(B < 0.0):
        raise ValueError("a_i and B_ii must be positive, B_ij non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, h


def pest_face_mean_density(params: dict, present: object, tau: float, which: int) -> float:
    a, B, h = _s03_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    T = [int(i) for i in present]
    if len(T) == 0 or len(set(T)) != len(T) or any(i < 0 or i > 2 for i in T):
        raise ValueError("present must be a non-empty set of distinct strain indices in {0, 1, 2}")
    which = int(which)
    if which not in T:
        raise ValueError("which must be one of the present strains")
    at = a + np.log1p(h[:3]) / tau
    mean = np.linalg.solve(B[np.ix_(T, T)], at[T])
    if np.any(mean <= 0.0):
        raise ValueError("this face carries no positive tau-periodic orbit (a period average would be non-positive)")
    return float(mean[T.index(which)])

import math
import numpy as np
from scipy.integrate import solve_ivp

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq

def _s04_params(params):
    """Validate the parameter dictionary and return float arrays (a, B, e, c, om, d, h)."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "e", "c", "om", "d", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    e = np.array(params["e"], dtype=float).ravel()
    c = np.array(params["c"], dtype=float).ravel()
    om = np.array(params["om"], dtype=float).ravel()
    d = float(params["d"])
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or e.shape != (3,) or c.shape != (3,) or om.shape != (3,) or h.shape != (4,):
        raise ValueError("expected a, e, c, om of length 3, B of shape 3x3 and h of length 4")
    for arr in (a, B, e, c, om, h):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite parameter")
    if not math.isfinite(d):
        raise ValueError("non-finite parameter d")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(e <= 0.0) or np.any(c <= 0.0) or d <= 0.0:
        raise ValueError("a_i, B_ii, e_i, c_i and d must be positive")
    if np.any(B < 0.0) or np.any(om < 0.0):
        raise ValueError("B_ij and om_i must be non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, e, c, om, d, h


def _s04_rates(z, P):
    """Per-capita growth rates (f_A, f_B, f_C, f_P) at the state z = (N_A, N_B, N_C, N_P)."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    D = 1.0 + om @ n
    f = np.empty(4)
    f[:3] = a - B @ n - e * z[3] / D
    f[3] = -d + (c @ n) / D
    return f


def _s04_jac(z, P):
    """Jacobian matrix d f_i / d z_j of the per-capita rates."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    w = z[3]
    D = 1.0 + om @ n
    J = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            J[i, j] = -B[i, j] + e[i] * w * om[j] / D ** 2
        J[i, 3] = -e[i] / D
    num = c @ n
    for j in range(3):
        J[3, j] = (c[j] * D - num * om[j]) / D ** 2
    return J


def _s04_rhs_var(t, y, P):
    """Flow, variational (fundamental-matrix) equations and running integrals of the per-capita rates."""
    z = y[:4]
    Phi = y[4:20].reshape(4, 4)
    f = _s04_rates(z, P)
    A = np.diag(f) + z[:, None] * _s04_jac(z, P)
    return np.concatenate([z * f, (A @ Phi).ravel(), f])


def _s04_map_full(z0, tau, P):
    """One-interval map: post-spray state, its Jacobian and the rate integrals over one interval."""
    y0 = np.concatenate([np.asarray(z0, dtype=float), np.eye(4).ravel(), np.zeros(4)])
    sol = solve_ivp(_s04_rhs_var, (0.0, tau), y0, method="DOP853", rtol=1e-13, atol=1e-16, args=(P,))
    if not sol.success:
        raise ValueError("integration failed: %s" % sol.message)
    y = sol.y[:, -1]
    g = 1.0 + P[6]
    return g * y[:4], g[:, None] * y[4:20].reshape(4, 4), y[20:24]


def _s04_face_orbit(present, guess, tau, P, tol=1e-12, maxit=60):
    """Newton shooting for the tau-periodic orbit on the face spanned by `present` (list of indices)."""
    present = list(present)
    k = len(present)
    z = np.zeros(4)
    z[present] = np.asarray(guess, dtype=float)
    if np.any(z[present] <= 0.0):
        raise ValueError("starting guess must be positive on the face")
    for _ in range(maxit):
        pz, Dpi, L = _s04_map_full(z, tau, P)
        F = pz[present] - z[present]
        if np.max(np.abs(F)) <= tol:
            return z, Dpi, L
        step = np.linalg.solve(Dpi[np.ix_(present, present)] - np.eye(k), -F)
        lam = 1.0
        while np.any(z[present] + lam * step <= 0.0):
            lam *= 0.5
            if lam < 1e-8:
                raise ValueError("Newton shooting left the positive face")
        z[present] = z[present] + lam * step
    raise ValueError("Newton shooting did not converge to the required residual")


def _s04_rates_on_orbit(z0, tau, P):
    """Pulse-adjusted long-term per-unit-time growth rates (1/tau)[int f_i dt + ln(1+h_i)] along the orbit through z0."""
    pz, Dpi, L = _s04_map_full(z0, tau, P)
    return (L + np.log1p(P[6])) / tau


def face_orbit_post_pulse(params: dict, present: object, guess: object, tau: float, which: int) -> float:
    P = _s04_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    T = [int(i) for i in present]
    if len(T) == 0 or len(set(T)) != len(T) or any(i < 0 or i > 3 for i in T):
        raise ValueError("present must be a non-empty set of distinct species indices in {0, 1, 2, 3}")
    which = int(which)
    if which not in T:
        raise ValueError("which must be one of the present species")
    g = np.array(guess, dtype=float).ravel()
    if g.shape != (len(T),) or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError("guess must give a positive finite density for every present species")
    z, Dpi, L = _s04_face_orbit(T, g, tau, P)
    if np.any(z[T] <= 0.0):
        raise ValueError("the located fixed point is not a positive orbit of the face")
    return float(z[which])

import math
import numpy as np
from scipy.integrate import solve_ivp

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq

def _s05_params(params):
    """Validate the parameter dictionary and return float arrays (a, B, e, c, om, d, h)."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "e", "c", "om", "d", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    e = np.array(params["e"], dtype=float).ravel()
    c = np.array(params["c"], dtype=float).ravel()
    om = np.array(params["om"], dtype=float).ravel()
    d = float(params["d"])
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or e.shape != (3,) or c.shape != (3,) or om.shape != (3,) or h.shape != (4,):
        raise ValueError("expected a, e, c, om of length 3, B of shape 3x3 and h of length 4")
    for arr in (a, B, e, c, om, h):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite parameter")
    if not math.isfinite(d):
        raise ValueError("non-finite parameter d")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(e <= 0.0) or np.any(c <= 0.0) or d <= 0.0:
        raise ValueError("a_i, B_ii, e_i, c_i and d must be positive")
    if np.any(B < 0.0) or np.any(om < 0.0):
        raise ValueError("B_ij and om_i must be non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, e, c, om, d, h


def _s05_rates(z, P):
    """Per-capita growth rates (f_A, f_B, f_C, f_P) at the state z = (N_A, N_B, N_C, N_P)."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    D = 1.0 + om @ n
    f = np.empty(4)
    f[:3] = a - B @ n - e * z[3] / D
    f[3] = -d + (c @ n) / D
    return f


def _s05_jac(z, P):
    """Jacobian matrix d f_i / d z_j of the per-capita rates."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    w = z[3]
    D = 1.0 + om @ n
    J = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            J[i, j] = -B[i, j] + e[i] * w * om[j] / D ** 2
        J[i, 3] = -e[i] / D
    num = c @ n
    for j in range(3):
        J[3, j] = (c[j] * D - num * om[j]) / D ** 2
    return J


def _s05_rhs_var(t, y, P):
    """Flow, variational (fundamental-matrix) equations and running integrals of the per-capita rates."""
    z = y[:4]
    Phi = y[4:20].reshape(4, 4)
    f = _s05_rates(z, P)
    A = np.diag(f) + z[:, None] * _s05_jac(z, P)
    return np.concatenate([z * f, (A @ Phi).ravel(), f])


def _s05_map_full(z0, tau, P):
    """One-interval map: post-spray state, its Jacobian and the rate integrals over one interval."""
    y0 = np.concatenate([np.asarray(z0, dtype=float), np.eye(4).ravel(), np.zeros(4)])
    sol = solve_ivp(_s05_rhs_var, (0.0, tau), y0, method="DOP853", rtol=1e-13, atol=1e-16, args=(P,))
    if not sol.success:
        raise ValueError("integration failed: %s" % sol.message)
    y = sol.y[:, -1]
    g = 1.0 + P[6]
    return g * y[:4], g[:, None] * y[4:20].reshape(4, 4), y[20:24]


def _s05_face_orbit(present, guess, tau, P, tol=1e-12, maxit=60):
    """Newton shooting for the tau-periodic orbit on the face spanned by `present` (list of indices)."""
    present = list(present)
    k = len(present)
    z = np.zeros(4)
    z[present] = np.asarray(guess, dtype=float)
    if np.any(z[present] <= 0.0):
        raise ValueError("starting guess must be positive on the face")
    for _ in range(maxit):
        pz, Dpi, L = _s05_map_full(z, tau, P)
        F = pz[present] - z[present]
        if np.max(np.abs(F)) <= tol:
            return z, Dpi, L
        step = np.linalg.solve(Dpi[np.ix_(present, present)] - np.eye(k), -F)
        lam = 1.0
        while np.any(z[present] + lam * step <= 0.0):
            lam *= 0.5
            if lam < 1e-8:
                raise ValueError("Newton shooting left the positive face")
        z[present] = z[present] + lam * step
    raise ValueError("Newton shooting did not converge to the required residual")


def _s05_rates_on_orbit(z0, tau, P):
    """Pulse-adjusted long-term per-unit-time growth rates (1/tau)[int f_i dt + ln(1+h_i)] along the orbit through z0."""
    pz, Dpi, L = _s05_map_full(z0, tau, P)
    return (L + np.log1p(P[6])) / tau


def stroboscopic_floquet_radius(params: dict, present: object, z_star: object, tau: float) -> float:
    P = _s05_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    T = [int(i) for i in present]
    if len(T) == 0 or len(set(T)) != len(T) or any(i < 0 or i > 3 for i in T):
        raise ValueError("present must be a non-empty set of distinct species indices in {0, 1, 2, 3}")
    z = np.array(z_star, dtype=float).ravel()
    if z.shape != (4,) or not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("z_star must be four finite non-negative densities")
    absent = [i for i in range(4) if i not in T]
    if np.any(z[absent] != 0.0) or np.any(z[T] <= 0.0):
        raise ValueError("z_star must be positive exactly on the present species")
    pz, Dpi, L = _s05_map_full(z, tau, P)
    if np.max(np.abs(pz[T] - z[T])) > 1e-8:
        raise ValueError("z_star is not a fixed point of the one-interval map on this face")
    ev = np.linalg.eigvals(Dpi[np.ix_(T, T)])
    return float(np.max(np.abs(ev)))

import math
import numpy as np
from scipy.integrate import solve_ivp

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq

def _s06_params(params):
    """Validate the parameter dictionary and return float arrays (a, B, e, c, om, d, h)."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "e", "c", "om", "d", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    e = np.array(params["e"], dtype=float).ravel()
    c = np.array(params["c"], dtype=float).ravel()
    om = np.array(params["om"], dtype=float).ravel()
    d = float(params["d"])
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or e.shape != (3,) or c.shape != (3,) or om.shape != (3,) or h.shape != (4,):
        raise ValueError("expected a, e, c, om of length 3, B of shape 3x3 and h of length 4")
    for arr in (a, B, e, c, om, h):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite parameter")
    if not math.isfinite(d):
        raise ValueError("non-finite parameter d")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(e <= 0.0) or np.any(c <= 0.0) or d <= 0.0:
        raise ValueError("a_i, B_ii, e_i, c_i and d must be positive")
    if np.any(B < 0.0) or np.any(om < 0.0):
        raise ValueError("B_ij and om_i must be non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, e, c, om, d, h


def _s06_rates(z, P):
    """Per-capita growth rates (f_A, f_B, f_C, f_P) at the state z = (N_A, N_B, N_C, N_P)."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    D = 1.0 + om @ n
    f = np.empty(4)
    f[:3] = a - B @ n - e * z[3] / D
    f[3] = -d + (c @ n) / D
    return f


def _s06_jac(z, P):
    """Jacobian matrix d f_i / d z_j of the per-capita rates."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    w = z[3]
    D = 1.0 + om @ n
    J = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            J[i, j] = -B[i, j] + e[i] * w * om[j] / D ** 2
        J[i, 3] = -e[i] / D
    num = c @ n
    for j in range(3):
        J[3, j] = (c[j] * D - num * om[j]) / D ** 2
    return J


def _s06_rhs_var(t, y, P):
    """Flow, variational (fundamental-matrix) equations and running integrals of the per-capita rates."""
    z = y[:4]
    Phi = y[4:20].reshape(4, 4)
    f = _s06_rates(z, P)
    A = np.diag(f) + z[:, None] * _s06_jac(z, P)
    return np.concatenate([z * f, (A @ Phi).ravel(), f])


def _s06_map_full(z0, tau, P):
    """One-interval map: post-spray state, its Jacobian and the rate integrals over one interval."""
    y0 = np.concatenate([np.asarray(z0, dtype=float), np.eye(4).ravel(), np.zeros(4)])
    sol = solve_ivp(_s06_rhs_var, (0.0, tau), y0, method="DOP853", rtol=1e-13, atol=1e-16, args=(P,))
    if not sol.success:
        raise ValueError("integration failed: %s" % sol.message)
    y = sol.y[:, -1]
    g = 1.0 + P[6]
    return g * y[:4], g[:, None] * y[4:20].reshape(4, 4), y[20:24]


def _s06_face_orbit(present, guess, tau, P, tol=1e-12, maxit=60):
    """Newton shooting for the tau-periodic orbit on the face spanned by `present` (list of indices)."""
    present = list(present)
    k = len(present)
    z = np.zeros(4)
    z[present] = np.asarray(guess, dtype=float)
    if np.any(z[present] <= 0.0):
        raise ValueError("starting guess must be positive on the face")
    for _ in range(maxit):
        pz, Dpi, L = _s06_map_full(z, tau, P)
        F = pz[present] - z[present]
        if np.max(np.abs(F)) <= tol:
            return z, Dpi, L
        step = np.linalg.solve(Dpi[np.ix_(present, present)] - np.eye(k), -F)
        lam = 1.0
        while np.any(z[present] + lam * step <= 0.0):
            lam *= 0.5
            if lam < 1e-8:
                raise ValueError("Newton shooting left the positive face")
        z[present] = z[present] + lam * step
    raise ValueError("Newton shooting did not converge to the required residual")


def _s06_rates_on_orbit(z0, tau, P):
    """Pulse-adjusted long-term per-unit-time growth rates (1/tau)[int f_i dt + ln(1+h_i)] along the orbit through z0."""
    pz, Dpi, L = _s06_map_full(z0, tau, P)
    return (L + np.log1p(P[6])) / tau


def pulsed_invasion_rate(params: dict, z0: object, i: int, tau: float) -> float:
    P = _s06_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    i = int(i)
    if i < 0 or i > 3:
        raise ValueError("species index must be in 0..3")
    z = np.array(z0, dtype=float).ravel()
    if z.shape != (4,) or not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("z0 must be four finite non-negative densities")
    pz, Dpi, L = _s06_map_full(z, tau, P)
    if np.max(np.abs(pz - z)) > 1e-8:
        raise ValueError("z0 is not the post-spray state of a tau-periodic orbit")
    return float((L[i] + math.log1p(P[6][i])) / tau)

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq

def _s07_params(params):
    """Validate the parameter dictionary and return float arrays (a, B, e, c, om, d, h)."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "e", "c", "om", "d", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    e = np.array(params["e"], dtype=float).ravel()
    c = np.array(params["c"], dtype=float).ravel()
    om = np.array(params["om"], dtype=float).ravel()
    d = float(params["d"])
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or e.shape != (3,) or c.shape != (3,) or om.shape != (3,) or h.shape != (4,):
        raise ValueError("expected a, e, c, om of length 3, B of shape 3x3 and h of length 4")
    for arr in (a, B, e, c, om, h):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite parameter")
    if not math.isfinite(d):
        raise ValueError("non-finite parameter d")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(e <= 0.0) or np.any(c <= 0.0) or d <= 0.0:
        raise ValueError("a_i, B_ii, e_i, c_i and d must be positive")
    if np.any(B < 0.0) or np.any(om < 0.0):
        raise ValueError("B_ij and om_i must be non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, e, c, om, d, h


def _s07_rates(z, P):
    """Per-capita growth rates (f_A, f_B, f_C, f_P) at the state z = (N_A, N_B, N_C, N_P)."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    D = 1.0 + om @ n
    f = np.empty(4)
    f[:3] = a - B @ n - e * z[3] / D
    f[3] = -d + (c @ n) / D
    return f


def _s07_jac(z, P):
    """Jacobian matrix d f_i / d z_j of the per-capita rates."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    w = z[3]
    D = 1.0 + om @ n
    J = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            J[i, j] = -B[i, j] + e[i] * w * om[j] / D ** 2
        J[i, 3] = -e[i] / D
    num = c @ n
    for j in range(3):
        J[3, j] = (c[j] * D - num * om[j]) / D ** 2
    return J


def _s07_rhs_var(t, y, P):
    """Flow, variational (fundamental-matrix) equations and running integrals of the per-capita rates."""
    z = y[:4]
    Phi = y[4:20].reshape(4, 4)
    f = _s07_rates(z, P)
    A = np.diag(f) + z[:, None] * _s07_jac(z, P)
    return np.concatenate([z * f, (A @ Phi).ravel(), f])


def _s07_map_full(z0, tau, P):
    """One-interval map: post-spray state, its Jacobian and the rate integrals over one interval."""
    y0 = np.concatenate([np.asarray(z0, dtype=float), np.eye(4).ravel(), np.zeros(4)])
    sol = solve_ivp(_s07_rhs_var, (0.0, tau), y0, method="DOP853", rtol=1e-13, atol=1e-16, args=(P,))
    if not sol.success:
        raise ValueError("integration failed: %s" % sol.message)
    y = sol.y[:, -1]
    g = 1.0 + P[6]
    return g * y[:4], g[:, None] * y[4:20].reshape(4, 4), y[20:24]


def _s07_face_orbit(present, guess, tau, P, tol=1e-12, maxit=60):
    """Newton shooting for the tau-periodic orbit on the face spanned by `present` (list of indices)."""
    present = list(present)
    k = len(present)
    z = np.zeros(4)
    z[present] = np.asarray(guess, dtype=float)
    if np.any(z[present] <= 0.0):
        raise ValueError("starting guess must be positive on the face")
    for _ in range(maxit):
        pz, Dpi, L = _s07_map_full(z, tau, P)
        F = pz[present] - z[present]
        if np.max(np.abs(F)) <= tol:
            return z, Dpi, L
        step = np.linalg.solve(Dpi[np.ix_(present, present)] - np.eye(k), -F)
        lam = 1.0
        while np.any(z[present] + lam * step <= 0.0):
            lam *= 0.5
            if lam < 1e-8:
                raise ValueError("Newton shooting left the positive face")
        z[present] = z[present] + lam * step
    raise ValueError("Newton shooting did not converge to the required residual")


def _s07_rates_on_orbit(z0, tau, P):
    """Pulse-adjusted long-term per-unit-time growth rates (1/tau)[int f_i dt + ln(1+h_i)] along the orbit through z0."""
    pz, Dpi, L = _s07_map_full(z0, tau, P)
    return (L + np.log1p(P[6])) / tau


def _s07_single_orbit(j, tau, P):
    a, B, e, c, om, d, h = P
    E = math.exp(a[j] * tau)
    return (a[j] / B[j, j]) * ((1.0 + h[j]) * E - 1.0) / (E - 1.0)


def _s07_adjusted(tau, P):
    a, B, e, c, om, d, h = P
    return a + np.log1p(h[:3]) / tau


def _s07_face_eq_with_P(T, P):
    """Unsprayed equilibrium of the face (pests T + parasitoid) from the linear system in (N_T, N_P/D)."""
    a, B, e, c, om, d, h = P
    k = len(T)
    M = np.zeros((k + 1, k + 1))
    rhs = np.zeros(k + 1)
    for i, ti in enumerate(T):
        for j, tj in enumerate(T):
            M[i, j] = B[ti, tj]
        M[i, k] = e[ti]
        rhs[i] = a[ti]
    for j, tj in enumerate(T):
        M[k, j] = c[tj] - d * om[tj]
    rhs[k] = d
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    N = sol[:k]
    w = sol[k]
    D = 1.0 + sum(om[tj] * N[j] for j, tj in enumerate(T))
    return N, w * D


def _s07_census(tau, P):
    """All boundary tau-periodic orbits (post-spray states) as a dict face-tuple -> state, plus their rate vectors."""
    a, B, e, c, om, d, h = P
    at = _s07_adjusted(tau, P)
    orbits = {}
    rates = {}
    # pest-only faces: existence from the averages identity, orbit from Newton shooting
    for k in (1, 2, 3):
        for T in itertools.combinations(range(3), k):
            T = list(T)
            try:
                mean = np.linalg.solve(B[np.ix_(T, T)], at[T])
            except np.linalg.LinAlgError:
                continue
            if np.any(mean <= 0.0):
                continue
            if k == 1:
                z = np.zeros(4)
                z[T[0]] = _s07_single_orbit(T[0], tau, P)
                pz, Dpi, L = _s07_map_full(z, tau, P)
                if np.max(np.abs(pz - z)) > 1e-10:
                    raise ValueError("single-pest orbit is not a fixed point of the one-interval map")
            else:
                z, Dpi, L = _s07_face_orbit(T, mean * (1.0 + h[T]), tau, P)
            orbits[tuple(T)] = z
            rates[tuple(T)] = (L + np.log1p(h)) / tau
    # faces with the parasitoid: pest j + P requires the parasitoid to invade the pest-only orbit
    for j in range(3):
        if (j,) not in orbits or rates[(j,)][3] <= 0.0:
            continue
        eq = _s07_face_eq_with_P([j], P)
        if eq is None or eq[0][0] <= 0.0 or eq[1] <= 0.0:
            guess = [0.5 * orbits[(j,)][j], 0.1]
        else:
            guess = [eq[0][0], eq[1]]
        z, Dpi, L = _s07_face_orbit([j, 3], guess, tau, P)
        if z[3] <= 1e-8:
            continue
        orbits[(j, 3)] = z
        rates[(j, 3)] = (L + np.log1p(h)) / tau
    # two pests + P: candidate only if each pest invades the orbit of the face without it
    for T in itertools.combinations(range(3), 2):
        ok = True
        for j in T:
            other = [i for i in T if i != j]
            key = (other[0], 3)
            if key not in orbits or rates[key][j] <= 0.0:
                ok = False
        if not ok:
            continue
        eq = _s07_face_eq_with_P(list(T), P)
        if eq is None or np.any(eq[0] <= 0.0) or eq[1] <= 0.0:
            continue
        try:
            z, Dpi, L = _s07_face_orbit(list(T) + [3], list(eq[0]) + [eq[1]], tau, P)
        except ValueError:
            continue
        if np.any(z[list(T) + [3]] <= 1e-8):
            continue
        orbits[tuple(T) + (3,)] = z
        rates[tuple(T) + (3,)] = (L + np.log1p(h)) / tau
    return orbits, rates


def boundary_census(params: dict, tau: float) -> float:
    P = _s07_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    orbits, rates = _s07_census(tau, P)
    return float(len(orbits))

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq

def _s08_params(params):
    """Validate the parameter dictionary and return float arrays (a, B, e, c, om, d, h)."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "e", "c", "om", "d", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    e = np.array(params["e"], dtype=float).ravel()
    c = np.array(params["c"], dtype=float).ravel()
    om = np.array(params["om"], dtype=float).ravel()
    d = float(params["d"])
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or e.shape != (3,) or c.shape != (3,) or om.shape != (3,) or h.shape != (4,):
        raise ValueError("expected a, e, c, om of length 3, B of shape 3x3 and h of length 4")
    for arr in (a, B, e, c, om, h):
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite parameter")
    if not math.isfinite(d):
        raise ValueError("non-finite parameter d")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(e <= 0.0) or np.any(c <= 0.0) or d <= 0.0:
        raise ValueError("a_i, B_ii, e_i, c_i and d must be positive")
    if np.any(B < 0.0) or np.any(om < 0.0):
        raise ValueError("B_ij and om_i must be non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, e, c, om, d, h


def _s08_rates(z, P):
    """Per-capita growth rates (f_A, f_B, f_C, f_P) at the state z = (N_A, N_B, N_C, N_P)."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    D = 1.0 + om @ n
    f = np.empty(4)
    f[:3] = a - B @ n - e * z[3] / D
    f[3] = -d + (c @ n) / D
    return f


def _s08_jac(z, P):
    """Jacobian matrix d f_i / d z_j of the per-capita rates."""
    a, B, e, c, om, d, h = P
    n = z[:3]
    w = z[3]
    D = 1.0 + om @ n
    J = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            J[i, j] = -B[i, j] + e[i] * w * om[j] / D ** 2
        J[i, 3] = -e[i] / D
    num = c @ n
    for j in range(3):
        J[3, j] = (c[j] * D - num * om[j]) / D ** 2
    return J


def _s08_rhs_var(t, y, P):
    """Flow, variational (fundamental-matrix) equations and running integrals of the per-capita rates."""
    z = y[:4]
    Phi = y[4:20].reshape(4, 4)
    f = _s08_rates(z, P)
    A = np.diag(f) + z[:, None] * _s08_jac(z, P)
    return np.concatenate([z * f, (A @ Phi).ravel(), f])


def _s08_map_full(z0, tau, P):
    """One-interval map: post-spray state, its Jacobian and the rate integrals over one interval."""
    y0 = np.concatenate([np.asarray(z0, dtype=float), np.eye(4).ravel(), np.zeros(4)])
    sol = solve_ivp(_s08_rhs_var, (0.0, tau), y0, method="DOP853", rtol=1e-13, atol=1e-16, args=(P,))
    if not sol.success:
        raise ValueError("integration failed: %s" % sol.message)
    y = sol.y[:, -1]
    g = 1.0 + P[6]
    return g * y[:4], g[:, None] * y[4:20].reshape(4, 4), y[20:24]


def _s08_face_orbit(present, guess, tau, P, tol=1e-12, maxit=60):
    """Newton shooting for the tau-periodic orbit on the face spanned by `present` (list of indices)."""
    present = list(present)
    k = len(present)
    z = np.zeros(4)
    z[present] = np.asarray(guess, dtype=float)
    if np.any(z[present] <= 0.0):
        raise ValueError("starting guess must be positive on the face")
    for _ in range(maxit):
        pz, Dpi, L = _s08_map_full(z, tau, P)
        F = pz[present] - z[present]
        if np.max(np.abs(F)) <= tol:
            return z, Dpi, L
        step = np.linalg.solve(Dpi[np.ix_(present, present)] - np.eye(k), -F)
        lam = 1.0
        while np.any(z[present] + lam * step <= 0.0):
            lam *= 0.5
            if lam < 1e-8:
                raise ValueError("Newton shooting left the positive face")
        z[present] = z[present] + lam * step
    raise ValueError("Newton shooting did not converge to the required residual")


def _s08_rates_on_orbit(z0, tau, P):
    """Pulse-adjusted long-term per-unit-time growth rates (1/tau)[int f_i dt + ln(1+h_i)] along the orbit through z0."""
    pz, Dpi, L = _s08_map_full(z0, tau, P)
    return (L + np.log1p(P[6])) / tau


def _s08_single_orbit(j, tau, P):
    a, B, e, c, om, d, h = P
    E = math.exp(a[j] * tau)
    return (a[j] / B[j, j]) * ((1.0 + h[j]) * E - 1.0) / (E - 1.0)


def _s08_adjusted(tau, P):
    a, B, e, c, om, d, h = P
    return a + np.log1p(h[:3]) / tau


def _s08_face_eq_with_P(T, P):
    """Unsprayed equilibrium of the face (pests T + parasitoid) from the linear system in (N_T, N_P/D)."""
    a, B, e, c, om, d, h = P
    k = len(T)
    M = np.zeros((k + 1, k + 1))
    rhs = np.zeros(k + 1)
    for i, ti in enumerate(T):
        for j, tj in enumerate(T):
            M[i, j] = B[ti, tj]
        M[i, k] = e[ti]
        rhs[i] = a[ti]
    for j, tj in enumerate(T):
        M[k, j] = c[tj] - d * om[tj]
    rhs[k] = d
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    N = sol[:k]
    w = sol[k]
    D = 1.0 + sum(om[tj] * N[j] for j, tj in enumerate(T))
    return N, w * D


def _s08_census(tau, P):
    """All boundary tau-periodic orbits (post-spray states) as a dict face-tuple -> state, plus their rate vectors."""
    a, B, e, c, om, d, h = P
    at = _s08_adjusted(tau, P)
    orbits = {}
    rates = {}
    # pest-only faces: existence from the averages identity, orbit from Newton shooting
    for k in (1, 2, 3):
        for T in itertools.combinations(range(3), k):
            T = list(T)
            try:
                mean = np.linalg.solve(B[np.ix_(T, T)], at[T])
            except np.linalg.LinAlgError:
                continue
            if np.any(mean <= 0.0):
                continue
            if k == 1:
                z = np.zeros(4)
                z[T[0]] = _s08_single_orbit(T[0], tau, P)
                pz, Dpi, L = _s08_map_full(z, tau, P)
                if np.max(np.abs(pz - z)) > 1e-10:
                    raise ValueError("single-pest orbit is not a fixed point of the one-interval map")
            else:
                z, Dpi, L = _s08_face_orbit(T, mean * (1.0 + h[T]), tau, P)
            orbits[tuple(T)] = z
            rates[tuple(T)] = (L + np.log1p(h)) / tau
    # faces with the parasitoid: pest j + P requires the parasitoid to invade the pest-only orbit
    for j in range(3):
        if (j,) not in orbits or rates[(j,)][3] <= 0.0:
            continue
        eq = _s08_face_eq_with_P([j], P)
        if eq is None or eq[0][0] <= 0.0 or eq[1] <= 0.0:
            guess = [0.5 * orbits[(j,)][j], 0.1]
        else:
            guess = [eq[0][0], eq[1]]
        z, Dpi, L = _s08_face_orbit([j, 3], guess, tau, P)
        if z[3] <= 1e-8:
            continue
        orbits[(j, 3)] = z
        rates[(j, 3)] = (L + np.log1p(h)) / tau
    # two pests + P: candidate only if each pest invades the orbit of the face without it
    for T in itertools.combinations(range(3), 2):
        ok = True
        for j in T:
            other = [i for i in T if i != j]
            key = (other[0], 3)
            if key not in orbits or rates[key][j] <= 0.0:
                ok = False
        if not ok:
            continue
        eq = _s08_face_eq_with_P(list(T), P)
        if eq is None or np.any(eq[0] <= 0.0) or eq[1] <= 0.0:
            continue
        try:
            z, Dpi, L = _s08_face_orbit(list(T) + [3], list(eq[0]) + [eq[1]], tau, P)
        except ValueError:
            continue
        if np.any(z[list(T) + [3]] <= 1e-8):
            continue
        orbits[tuple(T) + (3,)] = z
        rates[tuple(T) + (3,)] = (L + np.log1p(h)) / tau
    return orbits, rates


def _s08_rhs_plain(t, z, P):
    return z * _s08_rates(z, P)


def _s08_follow(z0, tau, P, orbits, nmax=600, rtol=1e-10, atol=1e-13):
    """Follow an invasion: iterate the one-interval map from z0 until a known boundary orbit is reached."""
    z = np.array(z0, dtype=float)
    g = 1.0 + P[6]
    for _ in range(nmax):
        sol = solve_ivp(_s08_rhs_plain, (0.0, tau), z, method="DOP853", rtol=rtol, atol=atol, args=(P,))
        z = g * sol.y[:, -1]
        for key, zs in orbits.items():
            if np.max(np.abs(z - zs)) < 1e-6:
                return key
    return None


def _s08_pieces(tau, P, orbits, rates):
    """Finest Morse decomposition of the boundary attractor: strongly connected components of the invasion graph."""
    keys = list(orbits.keys())
    edges = {k: set() for k in keys}
    for key in keys:
        present = set(key)
        for i in range(4):
            if i in present or rates[key][i] <= 0.0:
                continue
            z0 = orbits[key].copy()
            z0[i] = 1e-4
            target = _s08_follow(z0, tau, P, orbits)
            if target is not None and target != key:
                edges[key].add(target)
    # Tarjan-free SCC: reachability closure
    reach = {k: set([k]) for k in keys}
    changed = True
    while changed:
        changed = False
        for k in keys:
            new = set(reach[k])
            for t in list(reach[k]):
                new |= edges[t]
                new |= reach[t]
            if new != reach[k]:
                reach[k] = new
                changed = True
    pieces = []
    seen = set()
    for k in keys:
        if k in seen:
            continue
        comp = tuple(sorted(t for t in keys if t in reach[k] and k in reach[t]))
        seen |= set(comp)
        pieces.append(comp)
    pieces.append(("origin",))
    return pieces


def _s08_piece_margin(piece, tau, P, rates):
    """max over weights on the species absent from at least one orbit of the piece, of the min weighted rate."""
    a, B, e, c, om, d, h = P
    if piece == ("origin",):
        r0 = np.concatenate([_s08_adjusted(tau, P), [-d + math.log1p(h[3]) / tau]])
        return float(np.max(r0)), r0
    R = np.array([rates[k] for k in piece])
    species = [i for i in range(4) if any(i not in k for k in piece)]
    m = len(piece)
    k = len(species)
    cobj = np.zeros(k + 1)
    cobj[-1] = -1.0
    A_ub = np.hstack([-R[:, species], np.ones((m, 1))])
    res = linprog(cobj, A_ub=A_ub, b_ub=np.zeros(m), A_eq=np.hstack([np.ones((1, k)), [[0.0]]]), b_eq=[1.0],
                  bounds=[(0.0, None)] * k + [(None, None)], method="highs")
    if not res.success:
        raise ValueError("linear programme failed: %s" % res.message)
    val = -res.fun
    # polish: re-solve the active constraints exactly
    p = res.x[:k]
    act_rows = [i for i in range(m) if abs(R[i, species] @ p - val) < 1e-7]
    act_cols = [j for j in range(k) if p[j] > 1e-9]
    if len(act_rows) == len(act_cols):
        M = np.zeros((len(act_cols) + 1, len(act_cols) + 1))
        rhs = np.zeros(len(act_cols) + 1)
        for r_i, i in enumerate(act_rows):
            for c_j, j in enumerate(act_cols):
                M[r_i, c_j] = R[i, species[j]]
            M[r_i, -1] = -1.0
        M[-1, :len(act_cols)] = 1.0
        rhs[-1] = 1.0
        try:
            sol = np.linalg.solve(M, rhs)
            pp = np.zeros(k)
            pp[act_cols] = sol[:-1]
            vv = float(np.min(R[:, species] @ pp))
            if np.all(pp >= -1e-12) and abs(vv - val) < 1e-6:
                val = vv
                p = pp
        except np.linalg.LinAlgError:
            pass
    w = np.zeros(4)
    w[species] = p
    return float(val), w


def permanence_margin(params: dict, tau: float) -> float:
    P = _s08_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    orbits, rates = _s08_census(tau, P)
    pieces = _s08_pieces(tau, P, orbits, rates)
    m = math.inf
    for piece in pieces:
        val, w = _s08_piece_margin(piece, tau, P, rates)
        m = min(m, val)
    return float(m)

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq


def _s09_chain(params, tau, pieces_ref=None):
    """Assemble the boundary orbits, invasion rates, Morse decomposition,
    and permanence margin using the earlier pipeline oracles."""
    P = _s08_params(params)
    a, B, e, c, om, d, h = P

    # Always rebuild the boundary orbit census and invasion rates for the
    # supplied parameter set.  A reference decomposition may only be used
    # for an independent consistency check; it must never supply the orbit
    # census or rates themselves.
    orbits, rates = _s08_census(tau, P)

    # Cross-check every orbit with the individual earlier-step oracles.
    for key, z in orbits.items():
        T = list(key)

        if len(T) == 1 and T[0] < 3:
            if abs(
                z[T[0]]
                - single_pest_orbit(
                    a[T[0]],
                    B[T[0], T[0]],
                    h[T[0]],
                    tau,
                )
            ) > 1e-10:
                raise ValueError(
                    "single-strain orbit disagrees with the closed form"
                )

        if len(T) >= 2:
            for i in T:
                if abs(
                    face_orbit_post_pulse(
                        params,
                        T,
                        z[T],
                        tau,
                        i,
                    )
                    - z[i]
                ) > 1e-10:
                    raise ValueError(
                        "face orbit is not reproduced by the shooting step"
                    )

            if (
                stroboscopic_floquet_radius(
                    params,
                    T,
                    z,
                    tau,
                )
                >= 1.0
                and 3 in T
            ):
                raise ValueError(
                    "a pest-parasitoid orbit does not attract within its face"
                )

        if len(T) == 1:
            if abs(
                stroboscopic_map(
                    params,
                    z,
                    tau,
                    T[0],
                )
                - z[T[0]]
            ) > 1e-10:
                raise ValueError(
                    "single-species orbit is not a fixed point "
                    "of the one-interval map"
                )

        if all(i < 3 for i in T):
            mean = _s09_mean(z, T, tau, P)

            for i in T:
                if abs(
                    pest_face_mean_density(
                        params,
                        T,
                        tau,
                        i,
                    )
                    - mean[T.index(i)]
                ) > 1e-9:
                    raise ValueError(
                        "period average on a pest-only orbit "
                        "disagrees with the averages identity"
                    )

        for i in range(4):
            if abs(
                pulsed_invasion_rate(
                    params,
                    z,
                    i,
                    tau,
                )
                - rates[key][i]
            ) > 1e-10:
                raise ValueError(
                    "rate along an orbit disagrees with "
                    "the invasion-rate step"
                )

            if i in T and abs(rates[key][i]) > 1e-9:
                raise ValueError(
                    "a resident species must have zero "
                    "long-term growth on its own orbit"
                )

    # Recompute the complete boundary census for this parameter set.
    if float(len(orbits)) != boundary_census(params, tau):
        raise ValueError("boundary census mismatch")

    # Build the Morse decomposition for this parameter set unless a
    # reference decomposition was explicitly supplied for validation.
    #
    # IMPORTANT:
    # pieces_ref is never used to calculate the margin.  When supplied,
    # it is only checked against the freshly rebuilt decomposition.
    pieces = _s08_pieces(
        tau,
        P,
        orbits,
        rates,
    )

    if pieces_ref is not None:
        current_set = {
            tuple(piece)
            for piece in pieces
        }

        reference_set = {
            tuple(piece)
            for piece in pieces_ref
        }

        if current_set != reference_set:
            raise ValueError(
                "Morse decomposition changes across the q bracket"
            )

    # Calculate the signed normalized permanence margin from the freshly
    # reconstructed Morse decomposition.
    m = math.inf

    for piece in pieces:
        val, w = _s08_piece_margin(
            piece,
            tau,
            P,
            rates,
        )
        m = min(m, val)

    return float(m), pieces


def _s09_mean(z, T, tau, P):
    """Period averages of present species along a boundary orbit."""
    def _rhs(t, y):
        zz = y[:4]
        return np.concatenate([
            zz * _s08_rates(zz, P),
            zz,
        ])

    sol = solve_ivp(
        _rhs,
        (0.0, tau),
        np.concatenate([z, np.zeros(4)]),
        method="DOP853",
        rtol=1e-13,
        atol=1e-16,
    )

    return sol.y[4:8, -1][T] / tau


def _s09_scaled_params(params, q):
    """Apply h_i(q) = (1 + h_i^0)^q - 1 componentwise."""
    if not isinstance(params, dict) or "h" not in params:
        raise ValueError(
            "params must contain base pulse vector h"
        )

    q = float(q)

    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(
            "q must be finite and positive"
        )

    h0 = np.asarray(
        params["h"],
        dtype=float,
    ).ravel()

    if (
        h0.shape != (4,)
        or not np.all(np.isfinite(h0))
        or np.any(h0 <= -1.0)
    ):
        raise ValueError(
            "base pulse vector must have four finite "
            "entries greater than -1"
        )

    out = dict(params)

    out["h"] = np.expm1(
        q * np.log1p(h0)
    ).tolist()

    return out


def _s09_margin_at_scale(params, tau, q):
    """Recompute the complete permanence certificate at trial q.

    No reference orbit census or Morse decomposition is reused.
    """
    p = _s09_scaled_params(
        params,
        q,
    )

    # _s09_chain independently rebuilds:
    #   1. the boundary orbit census,
    #   2. all invasion rates,
    #   3. the Morse decomposition,
    #   4. the weighted component margins,
    #   5. the global permanence margin.
    m, pieces = _s09_chain(
        p,
        tau,
    )

    return float(m), pieces


def critical_pulse_scale(
    params: dict,
    tau: float,
    q_lo: float,
    q_hi: float,
) -> float:
    """Locate the q value where the fully recomputed permanence
    margin crosses zero.
    """
    tau = float(tau)
    q_lo = float(q_lo)
    q_hi = float(q_hi)

    if not (
        math.isfinite(tau)
        and math.isfinite(q_lo)
        and math.isfinite(q_hi)
    ):
        raise ValueError(
            "tau and bracket endpoints must be finite"
        )

    if tau <= 0.0 or not (0.0 < q_lo < q_hi):
        raise ValueError(
            "require tau > 0 and 0 < q_lo < q_hi"
        )

    # ------------------------------------------------------------------
    # Reference decomposition at q = 1.
    #
    # This decomposition is used only as a validation reference.
    # It is NOT reused to calculate margins at other q values.
    # ------------------------------------------------------------------
    p_ref = _s09_scaled_params(
        params,
        1.0,
    )

    m_ref, pieces_ref = _s09_chain(
        p_ref,
        tau,
    )

    # ------------------------------------------------------------------
    # Independently rebuild the full certificate at both endpoints.
    # ------------------------------------------------------------------
    m_lo, pieces_lo = _s09_margin_at_scale(
        params,
        tau,
        q_lo,
    )

    m_hi, pieces_hi = _s09_margin_at_scale(
        params,
        tau,
        q_hi,
    )

    # Canonicalize the decomposition so ordering of the components
    # does not affect the comparison.
    reference_set = {
        tuple(piece)
        for piece in pieces_ref
    }

    lo_set = {
        tuple(piece)
        for piece in pieces_lo
    }

    hi_set = {
        tuple(piece)
        for piece in pieces_hi
    }

    if lo_set != reference_set:
        raise ValueError(
            "Morse decomposition changes between q=1 and q_lo"
        )

    if hi_set != reference_set:
        raise ValueError(
            "Morse decomposition changes between q=1 and q_hi"
        )

    # The supplied bracket must contain a sign change.
    if not (
        math.isfinite(m_lo)
        and math.isfinite(m_hi)
        and m_lo * m_hi < 0.0
    ):
        raise ValueError(
            "q bracket does not enclose a sign change "
            "of the permanence margin"
        )

    # ------------------------------------------------------------------
    # Root function.
    #
    # Every evaluation independently rebuilds the complete ecological
    # certificate.  No cached orbit census, invasion rates, or Morse
    # decomposition from q=1 is reused.
    # ------------------------------------------------------------------
    def _margin_at_q(q):
        m_q, pieces_q = _s09_margin_at_scale(
            params,
            tau,
            q,
        )

        pieces_set = {
            tuple(piece)
            for piece in pieces_q
        }

        if pieces_set != reference_set:
            raise ValueError(
                "Morse decomposition changes across the q bracket"
            )

        if not math.isfinite(m_q):
            raise ValueError(
                "non-finite permanence margin"
            )

        return float(m_q)

    q_star = brentq(
        _margin_at_q,
        q_lo,
        q_hi,
        xtol=1e-12,
        maxiter=200,
    )

    # Verify the returned root using another complete independent
    # evaluation rather than relying on Brent's internal value.
    m_star, pieces_star = _s09_margin_at_scale(
        params,
        tau,
        q_star,
    )

    pieces_star_set = {
        tuple(piece)
        for piece in pieces_star
    }

    if pieces_star_set != reference_set:
        raise ValueError(
            "Morse decomposition changes at the computed q_star"
        )

    if abs(m_star) > 1e-9:
        raise ValueError(
            "margin at q_star is not zero"
        )

    # Independent reference-scale consistency check.
    m_ref_independent = permanence_margin(
        p_ref,
        tau,
    )

    if abs(
        m_ref - m_ref_independent
    ) > 1e-9:
        raise ValueError(
            "reference margin disagrees with "
            "independent margin evaluation"
        )

    return float(q_star)
SCICODE_GOLD_EOF
