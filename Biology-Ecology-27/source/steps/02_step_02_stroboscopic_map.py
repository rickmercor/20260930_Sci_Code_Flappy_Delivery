"""
One-interval (stroboscopic) map of the sprayed four-species community.

Given a post-spray state z0 = (NA, NB, NC, NP), this step integrates the between-spray Kolmogorov system over one interval of length tau with DOP853 (rtol 1e-13, atol 1e-16), applies the spray z -> (1+h) z, and returns either one component of the resulting post-spray state (which = 0..3) or the time integral over [0, tau] of one per-capita rate f(i) along the trajectory (which = 4..7), the latter obtained from an augmented state so that no separate quadrature is needed. It validates the parameter dictionary, the non-negativity of z0, tau > 0 and the range of which. Deliberately excluded: locating fixed points and the spray term ln(1+h) of the growth rates (added in step 6).

The sprayed community is a periodically pulsed Kolmogorov system: between sprays dN(i)/dt = N(i) f(i)(z) with f(A) = a(A) - b(AA) NA - b(AB) NB - b(AC) NC - e(A) NP / D, similarly for B and C, f(P) = -d + (c(A) NA + c(B) NB + c(C) NC) / D, D = 1 + om(A) NA + om(B) NB + om(C) NC; at t = k tau every density is multiplied by (1 + h(i)). The map pi(z) = (1+h) Phi(z, tau) from one post-spray instant to the next is the object through which all periodic orbits and all invasion rates of the source are defined; the section is taken immediately after a spray. Accuracy: on this configuration DOP853 with the stated tolerances and classical fixed-step RK4 with 2000 steps per interval agree to about 1e-15 both in the state and in the rate integrals. Check value at tau = 2: the A-only state (0.738278969075, 0, 0, 0) is mapped onto itself, and the integral of f(P) along the C-only orbit (0, 0, 0.7686964714501, 0) equals 3.724067127; divided by tau and increased by ln(1 - 0.2)/tau this is the parasitoid rate 1.750461788039.

Returns
-------
float — a post-spray state component after one interval, or an integrated per-capita rate (which = 0..7)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stroboscopic_map(params: dict, z0: "np.ndarray", tau: float, which: int) -> float:
    """One-interval (stroboscopic) map of the sprayed four-species community, taken post-spray to post-spray.
 
    Parameters
    ----------
    params : dict
        Model parameters: 'a' (3 growth rates), 'B' (3x3 competition matrix, row i = effect on strain i),
        'e' (3 attack rates), 'c' (3 parasitoid gains), 'om' (3 handling coefficients), 'd' (parasitoid
        mortality), 'h' (4 spray factors for A, B, C, P; each > -1).
    z0 : array_like, shape (4,)
        Post-spray state (N_A, N_B, N_C, N_P) at t = 0+; all entries >= 0.
    tau : float
        Spray interval (weeks), tau > 0.
    which : int
        0..3 -> the requested component of the post-spray state after one interval;
        4..7 -> the integral of the per-capita rate f_{which-4} over the interval [0, tau].
 
    Returns
    -------
    float
        The requested component (state after the spray at t = tau, or the rate integral).
 
    Raises
    ------
    ValueError
        On invalid parameters, negative or non-finite state, tau <= 0 or which outside 0..7.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_stroboscopic_map(params: dict, z0, tau: float, which: int) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'normal_single_strain_orbit_is_fixed_point',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'stroboscopic_map(PZ, [0.738278969075, 0.0, 0.0, 0.0], 2.0, 0)',
            "gold_call": '_oracle_stroboscopic_map(PZ, [0.738278969075, 0.0, 0.0, 0.0], 2.0, 0)',
        },
        {
            "name": 'normal_interior_state_parasitoid_component',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'stroboscopic_map(PZ, [0.3, 0.2, 0.25, 0.4], 2.0, 3)',
            "gold_call": '_oracle_stroboscopic_map(PZ, [0.3, 0.2, 0.25, 0.4], 2.0, 3)',
        },
        {
            "name": 'variant_rate_integral_of_parasitoid_on_strain_C',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'stroboscopic_map(PZ, [0.0, 0.0, 0.7686964714501, 0.0], 2.0, 7)',
            "gold_call": '_oracle_stroboscopic_map(PZ, [0.0, 0.0, 0.7686964714501, 0.0], 2.0, 7)',
        },
        {
            "name": 'edge_absent_species_stays_absent',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'stroboscopic_map(PZ, [0.3, 0.0, 0.25, 0.4], 1.2, 1)',
            "gold_call": '_oracle_stroboscopic_map(PZ, [0.3, 0.0, 0.25, 0.4], 1.2, 1)',
        },
    ]
