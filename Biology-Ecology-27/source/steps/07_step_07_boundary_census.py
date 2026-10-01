"""
Census of the boundary tau-periodic orbits.

This step enumerates the fifteen proper non-empty subsets of the four species and decides which of them carry a positive tau-periodic orbit: a pest-only face carries one exactly when the averages identity of step 3 has a positive solution (the orbit is then shot as in step 4 and verified); a face with one strain and the parasitoid carries one when the parasitoid's rate on that strain's orbit is positive (the orbit is shot from the unsprayed face equilibrium); a face with two strains and the parasitoid is examined only when each strain can invade the orbit of the face without it, and the shot fixed point is accepted only if it is positive. It returns the number of orbits found (the origin is not counted) and raises a ValueError on inconsistency. Deliberately excluded: the grouping of the orbits into pieces (step 8).

For the frozen parameters the boundary of the sprayed community carries seven non-trivial tau-periodic orbits for every tau in [1.2, 3.0]: the three single-strain orbits, the three-strain orbit, and the three strain-parasitoid orbits A-P, B-P, C-P. The two-strain faces are empty because the averages identity has a negative component (cyclic dominance), and the two-strain-plus-parasitoid faces are empty because in each of them one strain invades the orbit of the other strain with the parasitoid and the other strain cannot. The unsprayed community also has 7, and with a parasitoid mortality of d = 4 the parasitoid establishes on no face and the count is 4. Unsprayed equilibria, pre-spray sections and orbits that are unstable within their face are not separate entries: an orbit is counted once, at its post-spray state, whether or not it attracts.

Returns
-------
float — number of boundary periodic orbits (7.0 for the frozen set on the whole bracket)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_census(params: dict, tau: float) -> float:
    """Number of non-trivial tau-periodic orbits on the boundary of the sprayed four-species community.
 
    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    tau : float
        Spray interval (weeks), tau > 0.
 
    Returns
    -------
    float
        The number of boundary faces (proper non-empty subsets of the four species) carrying a positive
        tau-periodic orbit; the origin is not counted.
 
    Raises
    ------
    ValueError
        On invalid inputs or on any internal inconsistency of the orbit search.
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


def _oracle_boundary_census(params: dict, tau: float) -> float:
    P = _s07_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    orbits, rates = _s07_census(tau, P)
    return float(len(orbits))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'normal_frozen_reference_interval',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'boundary_census(PZ, 2.0)',
            "gold_call": '_oracle_boundary_census(PZ, 2.0)',
        },
        {
            "name": 'variant_bracket_end',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'boundary_census(PZ, 3.0)',
            "gold_call": '_oracle_boundary_census(PZ, 3.0)',
        },
                {
            "name": 'variant_parasitoid_cannot_establish',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZd = dict(PZ, d=4.0)",
            "call": 'boundary_census(PZd, 2.0)',
            "gold_call": '_oracle_boundary_census(PZd, 2.0)',
        },
        {
            "name": 'edge_unsprayed_community',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZ0 = dict(PZ, h=[0.0, 0.0, 0.0, 0.0])",
            "call": 'boundary_census(PZ0, 2.0)',
            "gold_call": '_oracle_boundary_census(PZ0, 2.0)',
        },
    ]
