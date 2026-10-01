"""
Locate the positive periodic orbit supported on a specified boundary face by solving the fixed-point equations of the post-spray one-interval map. Use the supplied post-spray state as the starting point for the nonlinear solve and return the requested species coordinate at the post-spray section. The result must correspond to a genuinely positive periodic orbit of the pulsed system and satisfy the fixed-point residual requirement in maximum norm.

A periodic orbit of a periodically impulsed population system becomes a fixed point when the system is sampled immediately after each pulse. Restricting the dynamics to a boundary face removes the absent populations while retaining the ecological interactions among the species present on that face. The resulting fixed-point problem generally requires solving the coupled flow-and-pulse dynamics rather than using the untreated equilibrium. Accurate localization of these boundary periodic states is essential because their growth rates determine whether missing species can invade the boundary.

Returns
-------
the finite positive post-spray density N_which(0+) of the periodic orbit on the specified face, returned as a single float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def face_orbit_post_pulse(params: dict, present: object, guess: object, tau: float, which: int) -> float:
    """Post-spray density of one species on the tau-periodic orbit of a boundary face, by Newton shooting.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    present : sequence of int
        Indices (subset of {0, 1, 2, 3}) of the species present on the face; non-empty, no repeats.
    guess : array_like, shape (len(present),)
        Positive starting densities for the present species (post-spray section).
    tau : float
        Spray interval (weeks), tau > 0.
    which : int
                Species index, one of the values listed in `present`, whose post-spray density on the located orbit is returned.

    Returns
    -------
    float
        Post-spray density N_which(0+) on the tau-periodic orbit of that face.

    Raises
    ------
    ValueError
        On invalid inputs, if `which` is not in `present`, or if the iteration does not converge to a
        positive fixed point of the one-interval map with residual <= 1e-12 (maximum norm).
    """

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_face_orbit_post_pulse(params: dict, present: object, guess: object, tau: float, which: int) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'face_orbit_post_pulse(PZ, [0, 1, 2], [0.2823692486639, 0.4652515063581, 0.2771645763793], 2.0, 0)',
            "gold_call": '_oracle_face_orbit_post_pulse(PZ, [0, 1, 2], [0.2823692486639, 0.4652515063581, 0.2771645763793], 2.0, 0)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'face_orbit_post_pulse(PZ, [0, 3], [0.3, 0.5], 2.0, 3)',
            "gold_call": '_oracle_face_orbit_post_pulse(PZ, [0, 3], [0.3, 0.5], 2.0, 3)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'face_orbit_post_pulse(PZ, [2, 3], [0.2, 0.6], 2.0, 2)',
            "gold_call": '_oracle_face_orbit_post_pulse(PZ, [2, 3], [0.2, 0.6], 2.0, 2)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'face_orbit_post_pulse(PZ, [1], [0.5], 2.0, 1)',
            "gold_call": '_oracle_face_orbit_post_pulse(PZ, [1], [0.5], 2.0, 1)',
        },
    ]
