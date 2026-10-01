"""
Permanence margin of the sprayed community.

This step builds the census of step 7 with the rates of step 6 along every orbit, constructs the invasion graph (from each orbit, every species with positive rate is introduced at density 1e-4 and the one-interval map is iterated until a known boundary orbit is reached), takes its strongly connected components together with the origin as the pieces of the finest Morse decomposition of the boundary attractor, and evaluates for each piece mu = max over probability vectors p supported on the species absent from at least one orbit of the piece of min over the orbits of the piece of sum p(i) r(i): a small linear programme whose optimal vertex is re-solved exactly. The origin piece uses the largest single-species rate. It returns m(tau) = min over pieces of mu. Deliberately excluded: the root search in tau (step 9).

The certification of the source is a weighted growth-rate condition on the pieces of a Morse decomposition of the boundary attractor. For the frozen parameters the seven orbits form two heteroclinic cycles and one isolated orbit: A -> B -> C -> A among the single-strain orbits, A-P -> B-P -> C-P -> A-P among the strain-parasitoid orbits, and the three-strain orbit on its own, so the pieces are {A-only, B-only, C-only}, {A-B-C}, {A-P, B-P, C-P} and the origin: four pieces for every tau in [1.2, 3.0]. At tau = 2 the piece values are 0.593109813191 (single-strain cycle, weights 0.2729840396 on B and 0.7270159604 on P), 1.063347630466 (three-strain orbit, parasitoid only), 0.00432421629663 (parasitoid cycle, weights 0.4189198137, 0.3982730985, 0.1828070879 on A, B, C) and 1.044587188117 (origin), hence m(2) = 0.00432421629663 attained on the parasitoid cycle; m(1.2) = -0.01445716818084 and m(3.0) = 0.01081570636861. Treating every orbit as its own piece with the best single invader gives 0.1197304023 at tau = 2 and stays positive on the whole bracket; lumping all six one- and two-species orbits into one piece gives 0.004209741818; the unsprayed community gives 0.01505050745307.

Returns
-------
float — permanence margin m(tau) (0.00432421629663 at tau = 2 for the frozen set, attained on the cycle of pest-parasitoid orbits)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def permanence_margin(params: dict, tau: float) -> float:
    """Permanence margin m(tau) of the sprayed four-species community.
 
    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    tau : float
        Spray interval (weeks), tau > 0.
 
    Returns
    -------
    float
        m(tau) = min over the pieces of the finest Morse decomposition of the boundary attractor of
        mu_k(tau), where mu_k is the largest value of min over the orbits of the piece of the weighted
        pulse-adjusted growth rates, the weights ranging over probability vectors supported on the species
        absent from at least one orbit of the piece (the origin piece uses the largest single-species rate).
 
    Raises
    ------
    ValueError
        On invalid inputs or on any internal inconsistency of the census, the decomposition or the
        linear programmes.
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


def _oracle_permanence_margin(params: dict, tau: float) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'normal_frozen_reference_interval',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'permanence_margin(PZ, 2.0)',
            "gold_call": '_oracle_permanence_margin(PZ, 2.0)',
        },
        {
            "name": 'variant_frequent_sprays_negative_margin',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'permanence_margin(PZ, 1.2)',
            "gold_call": '_oracle_permanence_margin(PZ, 1.2)',
        },
        {
            "name": 'variant_rare_sprays_certified',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'permanence_margin(PZ, 3.0)',
            "gold_call": '_oracle_permanence_margin(PZ, 3.0)',
        },
        {
            "name": 'edge_unsprayed_community_margin',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZ0 = dict(PZ, h=[0.0, 0.0, 0.0, 0.0])",
            "call": 'permanence_margin(PZ0, 2.0)',
            "gold_call": '_oracle_permanence_margin(PZ0, 2.0)',
        },
    ]
