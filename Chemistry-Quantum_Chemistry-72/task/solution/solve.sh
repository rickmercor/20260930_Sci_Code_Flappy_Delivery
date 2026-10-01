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
from math import factorial

def _threej_zero(j1, j2, j3):
    from math import factorial
    total = j1 + j2 + j3
    if total % 2 or j3 > j1 + j2 or j3 < abs(j1 - j2):
        return 0.0
    g = total // 2
    return ((-1) ** g) * (factorial(total - 2 * j1) * factorial(total - 2 * j2) * factorial(total - 2 * j3)
                          / factorial(total + 1)) ** 0.5 * factorial(g) / (
        factorial(g - j1) * factorial(g - j2) * factorial(g - j3))


def _triangle_coefficient(a, b, c):
    from math import factorial
    return (factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / factorial(a + b + c + 1)) ** 0.5


def _sixj(j1, j2, j3, j4, j5, j6):
    from math import factorial
    for a, b, c in ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)):
        if c > a + b or c < abs(a - b):
            return 0.0
    lo = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    hi = min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4)
    total = 0.0
    for t in range(lo, hi + 1):
        total += (-1) ** t * factorial(t + 1) / (
            factorial(t - j1 - j2 - j3) * factorial(t - j1 - j5 - j6) * factorial(t - j4 - j2 - j6)
            * factorial(t - j4 - j5 - j3) * factorial(j1 + j2 + j4 + j5 - t) * factorial(j2 + j3 + j5 + j6 - t)
            * factorial(j3 + j1 + j6 + j4 - t))
    return (_triangle_coefficient(j1, j2, j3) * _triangle_coefficient(j1, j5, j6)
            * _triangle_coefficient(j4, j2, j6) * _triangle_coefficient(j4, j5, j3) * total)


def _check_params(params):
    keys = ("mu", "B", "eps", "Rm", "a", "b", "jmax", "J", "parity", "scale")
    if not isinstance(params, dict) or any(k not in params for k in keys):
        raise ValueError("params must contain mu, B, eps, Rm, a, b, jmax, J, parity, scale")
    a, b, jmax, J, parity = params["a"], params["b"], params["jmax"], params["J"], params["parity"]
    if len(a) != 3 or len(b) != 3:
        raise ValueError("a and b need three entries")
    for n in (jmax, J):
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError("jmax and J must be non-negative integers")
    if parity not in (1, -1):
        raise ValueError("parity must be +1 or -1")
    if not (params["mu"] > 0 and params["B"] >= 0 and params["eps"] > 0 and params["Rm"] > 0 and params["scale"] > 0):
        raise ValueError("model parameter out of range")
    tables = _coupling_tables(int(jmax), int(J), int(parity))
    if tables[0].size == 0:
        raise ValueError("no channel for this J, parity and jmax")
    return [float(v) for v in a], [float(v) for v in b], tables


def _coupling_tables(jmax, J, parity):
    """Channel quantum numbers (j, l), j(j+1), l(l+1) and the Percival-Seaton coefficients for L = 0, 1, 2."""
    import numpy as np
    cache = _coupling_tables.__dict__.setdefault("cache", {})
    key = (jmax, J, parity)
    if key not in cache:
        chans = [(j, l) for j in range(jmax + 1) for l in range(abs(J - j), J + j + 1) if (-1) ** (j + l) == parity]
        N = len(chans)
        f = np.zeros((3, N, N))
        for L in range(3):
            for p, (j, l) in enumerate(chans):
                for q, (jp, lp) in enumerate(chans):
                    f[L, p, q] = ((-1) ** (j + jp - J) * ((2 * j + 1) * (2 * jp + 1) * (2 * l + 1) * (2 * lp + 1)) ** 0.5
                                  * _threej_zero(j, L, jp) * _threej_zero(l, L, lp) * _sixj(j, l, J, lp, jp, L))
        jj = np.array([j * (j + 1.0) for j, _ in chans])
        ll = np.array([l * (l + 1.0) for _, l in chans])
        cache[key] = (np.array(chans, dtype=int).reshape(N, 2), jj, ll, f)
    return cache[key]


def channel_matrix(R: float, params: dict) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be finite and positive")
    a, b, (chans, jj, ll, f) = _check_params(params)
    C = 16.8576292
    mu = float(params["mu"])
    x6 = (params["Rm"] / R) ** 6
    lam_eps = params["scale"] * params["eps"]
    M = (lam_eps * (a[0] * x6 * x6 - 2.0 * b[0] * x6)) * f[0]
    M += (lam_eps * (a[1] * x6 * x6 - 2.0 * b[1] * x6)) * f[1]
    M += (lam_eps * (a[2] * x6 * x6 - 2.0 * b[2] * x6)) * f[2]
    M[np.diag_indices(jj.size)] += params["B"] * jj + C * ll / (mu * R * R)
    M *= mu / C
    return M

import numpy as np

def propagate_log_derivative(params: dict, E: float, R_start: float, R_end: float, n_steps: int,
                                     Y_start: float) -> tuple:
    import numpy as np
    if not (np.isfinite(E) and np.isfinite(R_start) and np.isfinite(R_end)) or R_start <= 0.0 or R_end <= 0.0 \
            or R_start == R_end:
        raise ValueError("energy and distances must be finite, positive and distinct")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    N = _check_params(params)[2][0].shape[0]
    if not np.isfinite(Y_start):
        raise ValueError("Y_start must be finite")
    Y = float(Y_start) * np.eye(N)
    n_steps = int(n_steps)
    h = (R_end - R_start) / (2.0 * n_steps)
    ident = np.eye(N)
    shift = params["mu"] / 16.8576292 * E
    Y = Y - (h / 3.0) * (shift * ident - channel_matrix(R_start, params))
    nodes = 0
    for k in range(1, 2 * n_steps + 1):
        Z = ident + h * Y
        nodes += int(np.sum(np.linalg.eigvalsh(0.5 * (Z + Z.T)) < 0.0))
        Y = np.linalg.solve(Z, Y)
        Q = shift * ident - channel_matrix(R_start + k * h, params)
        if k == 2 * n_steps:
            Y = Y - (h / 3.0) * Q
        elif k % 2 == 1:
            Y = Y - (4.0 * h / 3.0) * np.linalg.solve(ident + (h * h / 6.0) * Q, Q)
        else:
            Y = Y - (2.0 * h / 3.0) * Q
        Y = 0.5 * (Y + Y.T)
    return Y, nodes

import numpy as np

def _hashable(value):
    if isinstance(value, str) or not hasattr(value, "__len__"):
        return value
    return tuple(float(v) for v in value)


def _spectrum_key(params, E, grid):
    """Hashable form of one matching-matrix evaluation, so that a repeated evaluation costs no propagation."""
    return (tuple(sorted((k, _hashable(v)) for k, v in params.items())), float(E),
            tuple(sorted((k, float(v)) for k, v in grid.items())))


def _grid_segments(grid):
    import numpy as np
    if not isinstance(grid, dict) or any(k not in grid for k in ("R_min", "R_match", "R_max", "h")):
        raise ValueError("grid must contain R_min, R_match, R_max and h")
    r0, rm, r1, h = (float(grid[k]) for k in ("R_min", "R_match", "R_max", "h"))
    if not all(np.isfinite([r0, rm, r1, h])) or not (0.0 < r0 < rm < r1) or h <= 0.0:
        raise ValueError("grid must satisfy 0 < R_min < R_match < R_max and h > 0")
    return r0, rm, r1, max(1, int(round((rm - r0) / (2.0 * h)))), max(1, int(round((r1 - rm) / (2.0 * h))))


def matching_spectrum(params: dict, E: float, grid: dict) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(E):
        raise ValueError("E must be finite")
    _check_params(params)
    r0, rm, r1, n_in, n_out = _grid_segments(grid)
    cache = matching_spectrum.__dict__.setdefault("cache", {})
    key = _spectrum_key(params, E, grid)
    if key in cache:
        return cache[key].copy()
    Y_out, nodes_out = propagate_log_derivative(params, E, r0, rm, n_in, 1e30)
    Y_in, nodes_in = propagate_log_derivative(params, E, r1, rm, n_out, -1e30)
    Ym = Y_out - Y_in
    ev = np.linalg.eigvalsh(0.5 * (Ym + Ym.T))
    result = np.concatenate(([float(nodes_out + nodes_in), float(np.sum(ev < 0.0))], ev))
    if len(cache) > 8192:
        cache.clear()
    cache[key] = result
    return result.copy()

import numpy as np

def _tracked_from_spectrum(spec, m):
    i = m - int(round(spec[0]))
    if 1 <= i <= spec.size - 2:
        return i, float(spec[1 + i])
    return 0, 0.0


def tracked_eigenvalue(params: dict, m: int, E: float, grid: dict) -> "np.ndarray":
    import numpy as np
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError("m must be an integer >= 1")
    if not np.isfinite(E):
        raise ValueError("E must be finite")
    i, e = _tracked_from_spectrum(matching_spectrum(params, E, grid), int(m))
    return np.array([float(i), e])

import numpy as np

def _problem_at(setting, x):
    """The walled problem at variable value x: params, energy and the grid matched at 4.0 angstrom."""
    params, E, R_min, R_max, h, variable = setting
    if variable == "E":
        E = float(x)
    else:
        params = dict(params, scale=float(x))
    return params, float(E), {"R_min": R_min, "R_match": 4.0, "R_max": R_max, "h": h}


def _node_count(setting, x):
    spec = matching_spectrum(*_problem_at(setting, x))
    return int(round(spec[0] + spec[1]))


def _tracked(setting, x, m):
    params, E, grid = _problem_at(setting, x)
    entry = tracked_eigenvalue(params, m, E, grid)
    return int(round(entry[0])), float(entry[1])


def _converge_levels(setting, labels, x_low, x_high, tol):
    """Bracket bookkeeping of the paper: for each label keep the tightest points with node count below m and at or
    above m, read the eigenvalue of state m from tracked_eigenvalue, bisect until one position brackets a sign change
    of it and then run Brent on that position. A Brent result is accepted only if the node counts it produced bracket
    it within tol."""
    import numpy as np
    from scipy.optimize import brentq
    counts = {float(x_low): _node_count(setting, x_low), float(x_high): _node_count(setting, x_high)}

    def bracket(m):
        return (max(x for x, n in counts.items() if n < m), min(x for x, n in counts.items() if n >= m))

    def signed(m, x):
        x = float(x)
        if x not in counts:
            counts[x] = _node_count(setting, x)
        i, e = _tracked(setting, x, m)
        return e if i else (1.0 if counts[x] < m else -1.0)

    roots = []
    for m in labels:
        brent_allowed = True
        while True:
            lo, hi = bracket(m)
            if hi - lo <= tol:
                roots.append(0.5 * (lo + hi))
                break
            i_lo, e_lo = _tracked(setting, lo, m)
            i_hi, e_hi = _tracked(setting, hi, m)
            if brent_allowed and i_lo and i_lo == i_hi and e_lo > 0.0 > e_hi:
                root = brentq(lambda x: signed(m, x), lo, hi, xtol=0.25 * tol, rtol=4.0 * np.finfo(float).eps)
                lo, hi = bracket(m)
                if hi - lo <= tol and lo - tol <= root <= hi + tol:
                    roots.append(root)
                    break
                brent_allowed = False
                continue
            mid = 0.5 * (lo + hi)
            counts[mid] = _node_count(setting, mid)
    return np.array(roots, dtype=float)


def _extrapolated_roots(setting_of_h, labels, x_low, x_high, tol):
    """Roots on grids h = 0.005 and 0.0025 angstrom, combined by Richardson extrapolation of the h^4 error."""
    import numpy as np
    coarse = setting_of_h(0.005)
    r1 = _converge_levels(coarse, labels, x_low, x_high, 0.05 * tol)
    fine = setting_of_h(0.0025)
    r2 = []
    for m, r in zip(labels, r1):
        width = max(1e-6 * (1.0 + abs(r)), 1e3 * tol)
        while True:
            lo, hi = max(x_low, r - width), min(x_high, r + width)
            if _node_count(fine, lo) < m <= _node_count(fine, hi) or (lo == x_low and hi == x_high):
                break
            width *= 10.0
        r2.append(_converge_levels(fine, [m], lo, hi, 0.05 * tol)[0])
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    return r2 + (r2 - r1) / 15.0


def bound_state_energies(params: dict, E_bottom: float, E_top: float, R_min: float, R_max: float,
                                 tol: float) -> "np.ndarray":
    import numpy as np
    if not all(np.isfinite([E_bottom, E_top, R_min, R_max, tol])) or E_top <= E_bottom:
        raise ValueError("need finite E_bottom < E_top")
    if R_min <= 0.0 or R_max <= R_min or tol < 1e-9:
        raise ValueError("need 0 < R_min < R_max and tol >= 1e-9")
    setting_of_h = lambda h: (dict(params), 0.0, float(R_min), float(R_max), h, "E")
    coarse = setting_of_h(0.005)
    n_bottom = _node_count(coarse, E_bottom)
    labels = list(range(n_bottom + 1, _node_count(coarse, E_top) + 1))
    return _extrapolated_roots(setting_of_h, labels, float(E_bottom), float(E_top), tol)

import numpy as np

def scale_for_level(params: dict, m: int, E: float, lam_low: float, lam_high: float, R_min: float,
                            R_max: float, tol: float) -> float:
    import numpy as np
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError("m must be an integer >= 1")
    if not all(np.isfinite([E, lam_low, lam_high, R_min, R_max, tol])) or lam_low <= 0.0 or lam_high <= lam_low:
        raise ValueError("invalid energy or interval")
    if R_min <= 0.0 or R_max <= R_min or tol < 1e-10:
        raise ValueError("need 0 < R_min < R_max and tol >= 1e-10")
    m = int(m)
    setting_of_h = lambda h: (dict(params), float(E), float(R_min), float(R_max), h, "scale")
    coarse = setting_of_h(0.005)
    if not (_node_count(coarse, lam_low) < m <= _node_count(coarse, lam_high)):
        raise ValueError("the interval does not bracket state m at this energy")
    return float(_extrapolated_roots(setting_of_h, [m], float(lam_low), float(lam_high), tol)[0])

import numpy as np

def predicted_partner_energy(params: dict, m_fit: int, E_obs: float, m_pred: int, lam_low: float,
                                     lam_high: float, R_min: float, R_max: float) -> float:
    import numpy as np
    if isinstance(m_pred, bool) or int(m_pred) != m_pred or m_pred < 1 or m_pred == m_fit:
        raise ValueError("m_pred must be an integer >= 1 different from m_fit")
    m_pred = int(m_pred)
    lam_star = scale_for_level(params, m_fit, E_obs, lam_low, lam_high, R_min, R_max, 1e-10)
    scaled = dict(params, scale=lam_star)
    setting = (scaled, 0.0, float(R_min), float(R_max), 0.005, "E")
    # no state lies below the lowest eigenvalue of the effective potential matrix anywhere on the range
    floor = min(np.linalg.eigvalsh(channel_matrix(R, scaled))[0]
                for R in np.linspace(R_min, R_max, 4001)) * (16.8576292 / params["mu"])
    E_low = floor - 0.05 * abs(floor) - 1.0
    n_low = _node_count(setting, E_low)
    if n_low >= m_pred:
        raise ValueError("nodes appear below the bottom of the potential")
    E_high = float(E_obs) + 1.0
    while _node_count(setting, E_high) < m_pred:
        E_high = E_high + max(10.0, abs(E_high))
    # narrow the window on the node count alone until it holds the predicted state, then hand it to step 05
    lo, hi, n_lo = E_low, E_high, n_low
    for _ in range(200):
        if _node_count(setting, hi) - n_lo == 1:
            break
        mid = 0.5 * (lo + hi)
        n_mid = _node_count(setting, mid)
        if n_mid >= m_pred:
            hi = mid
        else:
            lo, n_lo = mid, n_mid
    energies = bound_state_energies(scaled, lo, hi, R_min, R_max, 1e-9)
    return float(energies[m_pred - n_lo - 1])
SCICODE_GOLD_EOF
