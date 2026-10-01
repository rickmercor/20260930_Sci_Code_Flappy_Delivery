"""
Compute the long-term per-unit-time growth rate of a specified species along a boundary periodic orbit. Integrate that species' continuous per-capita growth rate over one complete inter-spray interval and incorporate the species-specific logarithmic contribution from its own multiplicative pulse. Normalize the complete period growth by the physical spray interval. The calculation must be valid both for resident species and for species absent from the boundary orbit.

For a population following multiplicative impulsive dynamics, its logarithmic abundance changes continuously between pulses and discontinuously at each intervention. Consequently, its net growth over one period contains two distinct contributions: the integral of the continuous per-capita rate and the logarithm of the multiplicative pulse factor. Dividing their sum by the physical period gives the pulse-adjusted invasion or resident growth rate. For an absent species, its own pulse effect remains part of its invasion rate even though its population is zero along the boundary trajectory.

Returns
-------
the finite pulse-adjusted per-unit-time growth rate r_i of the requested species along the supplied boundary periodic orbit, as a single float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pulsed_invasion_rate(params: dict, z0: object, i: int, tau: float) -> float:
    """Pulse-adjusted long-term per-unit-time growth rate of species i along the tau-periodic orbit through z0.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    z0 : array_like, shape (4,)
        Post-spray state of a tau-periodic boundary orbit (zero for absent species).
    i : int
        Species index in 0..3 (0 = A, 1 = B, 2 = C, 3 = P).
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        r_i = (1/tau) [ int_0^tau f_i(z(t)) dt + ln(1 + h_i) ] along the orbit through z0.

    Raises
    ------
    ValueError
        On invalid inputs, or if z0 is not a fixed point of the one-interval map to 1e-8.
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


def _oracle_pulsed_invasion_rate(params: dict, z0: object, i: int, tau: float) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.738278969075, 0.0, 0.0, 0.0], 3, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.738278969075, 0.0, 0.0, 0.0], 3, 2.0)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 1, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 1, 2.0)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 2, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 2, 2.0)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 0, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 0, 2.0)',
        },
    ]
