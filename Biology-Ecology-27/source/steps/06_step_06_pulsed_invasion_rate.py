"""
Pulse-adjusted long-term growth rate of one species along a boundary orbit.

For the tau-periodic orbit through a post-spray state z0 (checked to be a fixed point of the one-interval map to 1e-8) and a species index i, this step returns r(i) = (1/tau) [ int over [0, tau] of f(i) along the orbit + ln(1 + h(i)) ]: the between-spray integral of the per-capita rate of species i evaluated along the orbit plus the logarithm of its own spray factor, divided by the spray interval. For a species present on the orbit the value is zero to round-off. Deliberately excluded: any combination of the rates across species or orbits (steps 8 and 9).

Along a boundary orbit the source measures the growth of an absent species by the average over one interval of its per-capita rate plus the spray contribution ln(1 + h(i))/tau, which is charged to species i even though it is absent from the orbit; the rate is normalised per unit time of the sprayed system, i.e. by tau and not by tau + 1 (the embedding into R^n x S^1 that spreads the pulse over one extra time unit is a proof device, not a change of clock). Benchmark values at tau = 2: parasitoid on the single-strain orbits 0.589085418532 (A), 0.8158140199195 (B), 1.750461788039 (C) and on the three-strain orbit 1.063347630466; cross rates on the pest-parasitoid orbits r(B; A-P) = 0.2147295235232, r(C; A-P) = -0.4441664561082, r(C; B-P) = 0.3007731594113, r(A; B-P) = -0.1209282717831, r(A; C-P) = 0.1197304022945, r(B; C-P) = -0.1150798828753; residents give |r| < 1e-12. Omitting the spray term, evaluating f(i) at period-averaged densities or dividing by tau + 1 all change these numbers.

Returns
-------
float — pulse-adjusted long-term growth rate of one species along a boundary orbit (0.589085418532 for the parasitoid on the A-only orbit at tau = 2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pulsed_invasion_rate(params: dict, z0: "np.ndarray", i: int, tau: float) -> float:
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


def _oracle_pulsed_invasion_rate(params: dict, z0, i: int, tau: float) -> float:
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
            "name": 'normal_parasitoid_invades_strain_A_orbit',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.738278969075, 0.0, 0.0, 0.0], 3, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.738278969075, 0.0, 0.0, 0.0], 3, 2.0)',
        },
        {
            "name": 'normal_strain_B_invades_A_parasitoid_orbit',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 1, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 1, 2.0)',
        },
        {
            "name": 'variant_strain_C_fails_on_A_parasitoid_orbit',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 2, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 2, 2.0)',
        },
        {
            "name": 'edge_resident_strain_has_zero_rate',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 0, 2.0)',
            "gold_call": '_oracle_pulsed_invasion_rate(PZ, [0.2883307308355, 0.0, 0.0, 0.5804343349675], 0, 2.0)',
        },
    ]
