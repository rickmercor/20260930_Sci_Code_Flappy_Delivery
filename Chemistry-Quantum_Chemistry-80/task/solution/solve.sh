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

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def _morse_grid(x, n_max, shift=0.0):
    s = math.sqrt(2.0 * x)
    lam = 1.0 / (2.0 * x)
    e_top = (n_max + 0.5) - x * (n_max + 0.5) ** 2
    e_top = min(e_top, 0.5 * lam - 1e-3)
    xi_turn = -math.log(1.0 - math.sqrt(2.0 * e_top / lam)) / s
    kappa = math.sqrt(max(lam - 2.0 * e_top, 1e-6))
    lo = -1.6 / s - abs(shift)
    hi = xi_turn + 48.0 / kappa + abs(shift)
    h = 0.004
    xi = np.arange(lo, hi + 0.5 * h, h)
    return xi, h


def _morse_states(x, n_max, xi, shift=0.0):
    s = math.sqrt(2.0 * x)
    lam = 1.0 / (2.0 * x)
    z = 2.0 * lam * np.exp(-s * (xi - shift))
    logz = np.log(2.0 * lam) - s * (xi - shift)
    phi = np.empty((n_max + 1, xi.size))
    dphi = np.empty((n_max + 1, xi.size))
    for n in range(n_max + 1):
        k = lam - n - 0.5
        alpha = 2.0 * lam - 2.0 * n - 1.0
        lognorm = 0.5 * (math.log(s) + math.log(alpha) + gammaln(n + 1.0) - gammaln(2.0 * lam - n))
        base = np.exp(lognorm + k * logz - 0.5 * z)
        lag = eval_genlaguerre(n, alpha, z)
        dlag = -eval_genlaguerre(n - 1, alpha + 1.0, z) if n > 0 else np.zeros_like(z)
        phi[n] = base * lag
        dphi[n] = base * ((k / z - 0.5) * lag + dlag) * (-s * z)
    return phi, dphi


def morse_moment_matrices(x: float, n_max: int) -> np.ndarray:
    x = float(x)
    if not x > 0.0:
        raise ValueError("x must be positive")
    if int(n_max) != n_max or n_max < 0:
        raise ValueError("n_max must be a non-negative integer")
    n_max = int(n_max)
    if n_max + 0.5 >= 1.0 / (2.0 * x) - 2.0:
        raise ValueError("n_max must lie well below the dissociation limit")
    xi, h = _morse_grid(x, n_max)
    phi, _ = _morse_states(x, n_max, xi)
    out = np.empty((4, n_max + 1, n_max + 1))
    for p in range(1, 5):
        out[p - 1] = (phi * xi ** p) @ phi.T * h
    return 0.5 * (out + np.transpose(out, (0, 2, 1)))

import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def _morse_grid(x, n_max, shift=0.0):
    s = math.sqrt(2.0 * x)
    lam = 1.0 / (2.0 * x)
    e_top = (n_max + 0.5) - x * (n_max + 0.5) ** 2
    e_top = min(e_top, 0.5 * lam - 1e-3)
    xi_turn = -math.log(1.0 - math.sqrt(2.0 * e_top / lam)) / s
    kappa = math.sqrt(max(lam - 2.0 * e_top, 1e-6))
    lo = -1.6 / s - abs(shift)
    hi = xi_turn + 48.0 / kappa + abs(shift)
    h = 0.004
    xi = np.arange(lo, hi + 0.5 * h, h)
    return xi, h


def _morse_states(x, n_max, xi, shift=0.0):
    s = math.sqrt(2.0 * x)
    lam = 1.0 / (2.0 * x)
    z = 2.0 * lam * np.exp(-s * (xi - shift))
    logz = np.log(2.0 * lam) - s * (xi - shift)
    phi = np.empty((n_max + 1, xi.size))
    dphi = np.empty((n_max + 1, xi.size))
    for n in range(n_max + 1):
        k = lam - n - 0.5
        alpha = 2.0 * lam - 2.0 * n - 1.0
        lognorm = 0.5 * (math.log(s) + math.log(alpha) + gammaln(n + 1.0) - gammaln(2.0 * lam - n))
        base = np.exp(lognorm + k * logz - 0.5 * z)
        lag = eval_genlaguerre(n, alpha, z)
        dlag = -eval_genlaguerre(n - 1, alpha + 1.0, z) if n > 0 else np.zeros_like(z)
        phi[n] = base * lag
        dphi[n] = base * ((k / z - 0.5) * lag + dlag) * (-s * z)
    return phi, dphi


def promoting_integrals(x: float, shift: float, n_max: int) -> np.ndarray:
    x = float(x)
    shift = float(shift)
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if int(n_max) != n_max or n_max < 0:
        raise ValueError("n_max must be a non-negative integer")
    n_max = int(n_max)
    if x == 0.0:
        g = np.empty(n_max + 2)
        for n in range(n_max + 2):
            g[n] = math.exp(-shift * shift / 4.0 + n * math.log(abs(shift) / math.sqrt(2.0)) - 0.5 * gammaln(n + 1.0)) if shift != 0.0 else (1.0 if n == 0 else 0.0)
            if shift < 0.0 and n % 2 == 1:
                g[n] = -g[n]
        t = np.empty(n_max + 1)
        for n in range(n_max + 1):
            lower = math.sqrt(n) * g[n - 1] if n > 0 else 0.0
            t[n] = (lower - math.sqrt(n + 1.0) * g[n + 1]) / math.sqrt(2.0)
        return np.array([g[:n_max + 1], t])
    if n_max + 0.5 >= 1.0 / (2.0 * x) - 2.0:
        raise ValueError("n_max must lie well below the dissociation limit")
    xi, h = _morse_grid(x, n_max, shift)
    phi, dphi = _morse_states(x, n_max, xi)
    phi0, _ = _morse_states(x, 0, xi, shift)
    g = phi @ phi0[0] * h
    t = dphi @ phi0[0] * h
    return np.array([g, t])

import itertools
import math

import numpy as np


def local_force_field(C: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, n_bath: int, thresholds: np.ndarray) -> np.ndarray:
    C = np.array(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a matrix")
    n_x = C.shape[0]
    n_bath = int(n_bath)
    if n_bath < 0:
        raise ValueError("n_bath must be non-negative")
    thr = np.array(thresholds, dtype=float).ravel()
    if thr.size != 6 or np.any(thr <= 0.0):
        raise ValueError("thresholds must hold six positive numbers")
    n_modes = C.shape[1] + n_bath
    n_all = n_x + n_bath
    A = np.linalg.pinv(C)
    T = np.zeros((n_modes, n_all))
    T[:C.shape[1], :n_x] = A
    T[C.shape[1]:, n_x:] = np.eye(n_bath)
    rows = []
    for arr, r in ((cubic, 3), (quartic, 4)):
        arr = np.array(arr, dtype=float).reshape(-1, r + 1)
        phi = np.zeros((n_modes,) * r)
        seen = set()
        for row in arr:
            idx = tuple(int(round(v)) for v in row[:r])
            if any(abs(v - round(v)) > 1e-12 for v in row[:r]):
                raise ValueError("indices must be integers")
            if list(idx) != sorted(idx) or idx[0] < 0 or idx[-1] >= n_modes:
                raise ValueError("index tuples must be canonical and in range")
            if idx in seen:
                raise ValueError("duplicate index tuple")
            seen.add(idx)
            for perm in set(itertools.permutations(idx)):
                phi[perm] = row[r]
        if r == 3:
            psi = np.einsum('ijk,ia,jb,kc->abc', phi, T, T, T, optimize=True)
        else:
            psi = np.einsum('ijkl,ia,jb,kc,ld->abcd', phi, T, T, T, T, optimize=True)
        for tup in itertools.combinations_with_replacement(range(n_all), r):
            n_loc = sum(1 for u in tup if u < n_x)
            if n_loc == r and len(set(tup)) == 1:
                continue
            mult = 1
            for u in set(tup):
                mult *= math.factorial(tup.count(u))
            coef = psi[tup] / mult
            cls = 0 if n_loc == r else (2 if n_loc == 0 else 1)
            if abs(coef) < thr[2 * cls + (r - 3)]:
                continue
            rows.append([r] + list(tup) + [-1] * (4 - r) + [coef, cls])
    if not rows:
        return np.zeros((0, 7))
    return np.array(rows, dtype=float)

import itertools

import numpy as np


def _morse_ladder(omega, x, n):
    n = np.asarray(n, dtype=float)
    return omega * (n - x * (n * n + n))


def window_states(omega_loc: np.ndarray, x_loc: np.ndarray, omega_bath: np.ndarray, e_gap: float, half_width: float, n_max: int) -> np.ndarray:
    om_l = np.array(omega_loc, dtype=float).ravel()
    x_l = np.array(x_loc, dtype=float).ravel()
    om_b = np.array(omega_bath, dtype=float).ravel()
    if om_l.size != x_l.size:
        raise ValueError("omega_loc and x_loc must have the same length")
    if np.any(om_l <= 0.0) or np.any(om_b <= 0.0) or np.any(x_l <= 0.0):
        raise ValueError("frequencies and anharmonicities must be positive")
    if half_width < 0.0 or int(n_max) != n_max or n_max < 0:
        raise ValueError("half_width must be non-negative and n_max a non-negative integer")
    n_max = int(n_max)
    top = e_gap + half_width
    levels = [_morse_ladder(om_l[a], x_l[a], np.arange(n_max + 1)) for a in range(om_l.size)]
    found = []

    def _fill_bath(b, acc, e, loc):
        if b == om_b.size:
            if abs(e_gap - e) <= half_width:
                found.append(loc + acc + [e])
            return
        k = 0
        while e + om_b[b] * k <= top:
            _fill_bath(b + 1, acc + [k], e + om_b[b] * k, loc)
            k += 1

    for nl in itertools.product(range(n_max + 1), repeat=om_l.size):
        e = float(sum(levels[a][nl[a]] for a in range(om_l.size)))
        if e > top:
            continue
        _fill_bath(0, [], e, list(nl))
    if not found:
        return np.zeros((0, om_l.size + om_b.size + 1))
    found.sort(key=lambda r: (round(r[-1], 6), r[:-1]))
    return np.array(found, dtype=float)

import numpy as np


def zero_order_amplitudes(states: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int) -> np.ndarray:
    st = np.array(states, dtype=float)
    x_l = np.array(x_loc, dtype=float).ravel()
    d_l = np.array(delta_loc, dtype=float).ravel()
    d_b = np.array(delta_bath, dtype=float).ravel()
    p_b = np.array(p_bath, dtype=float).ravel()
    n_x, n_b = x_l.size, d_b.size
    if st.ndim != 2 or st.shape[1] != n_x + n_b + 1 or d_l.size != n_x or p_b.size != n_b:
        raise ValueError("inconsistent array shapes")
    if st.shape[0] == 0:
        return np.zeros((0, 2))
    d_loc = np.linalg.pinv(np.array(C, dtype=float)).T @ np.array(p_xh, dtype=float).ravel()
    occ = np.rint(st[:, :n_x + n_b]).astype(int)
    if np.any(occ < 0) or np.any(occ[:, :n_x] > n_max):
        raise ValueError("occupations out of range")
    nb_top = int(occ[:, n_x:].max()) if n_b else 0
    g = np.ones((n_x + n_b, st.shape[0]))
    t = np.zeros((n_x + n_b, st.shape[0]))
    for a in range(n_x):
        gt = promoting_integrals(x_l[a], d_l[a], n_max)
        g[a] = gt[0][occ[:, a]]
        t[a] = gt[1][occ[:, a]]
    for b in range(n_b):
        gt = promoting_integrals(0.0, d_b[b], nb_top)
        g[n_x + b] = gt[0][occ[:, n_x + b]]
        t[n_x + b] = gt[1][occ[:, n_x + b]]
    g_loc = np.prod(g[:n_x], axis=0)
    g_bath = np.prod(g[n_x:], axis=0)
    m_xh = np.zeros(st.shape[0])
    for a in range(n_x):
        others = np.prod(np.delete(g[:n_x], a, axis=0), axis=0)
        m_xh += d_loc[a] * t[a] * others
    m_b = np.zeros(st.shape[0])
    for j in range(n_b):
        others = np.prod(np.delete(g[n_x:], j, axis=0), axis=0)
        m_b += p_b[j] * t[n_x + j] * others
    return np.column_stack([m_xh * g_bath, m_b * g_loc])

import itertools

import numpy as np


def _morse_ladder(omega, x, n):
    n = np.asarray(n, dtype=float)
    return omega * (n - x * (n * n + n))


def first_order_intensities(states: np.ndarray, force_terms: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int, resonance_cutoff: float) -> np.ndarray:
    st = np.array(states, dtype=float)
    om_l = np.array(omega_loc, dtype=float).ravel()
    x_l = np.array(x_loc, dtype=float).ravel()
    n_x = om_l.size
    n_b = np.array(delta_bath, dtype=float).size
    if st.ndim != 2 or st.shape[1] != n_x + n_b + 1:
        raise ValueError("inconsistent array shapes")
    if resonance_cutoff < 0.0:
        raise ValueError("resonance_cutoff must be non-negative")
    if st.shape[0] == 0:
        return np.zeros(0)
    terms = np.array(force_terms, dtype=float).reshape(-1, 7)
    xx = [row for row in terms if int(row[6]) == 0]
    mom = [morse_moment_matrices(x_l[a], n_max) for a in range(n_x)]
    size = n_max + 1
    grid = np.array(list(itertools.product(range(size), repeat=n_x)), dtype=float)
    e_grid = np.zeros(grid.shape[0])
    for a in range(n_x):
        e_grid += _morse_ladder(om_l[a], x_l[a], grid[:, a])
    e_grid = e_grid.reshape((size,) * n_x)
    occ = np.rint(st[:, :n_x + n_b]).astype(int)
    out = np.empty(st.shape[0])
    cache = {}
    for i in range(st.shape[0]):
        nl = tuple(occ[i, :n_x])
        nb = tuple(occ[i, n_x:])
        if nb not in cache:
            rows = np.column_stack([grid, np.tile(np.array(nb, dtype=float), (grid.shape[0], 1)), np.zeros(grid.shape[0])])
            amp = zero_order_amplitudes(rows, C, p_xh, p_bath, x_loc, delta_loc, delta_bath, n_max)
            cache[nb] = (amp[:, 0] + amp[:, 1]).reshape((size,) * n_x)
        m_grid = cache[nb]
        coupling = np.zeros((size,) * n_x)
        for row in xx:
            r = int(row[0])
            idx = [int(v) for v in row[1:1 + r]]
            factors = []
            for a in range(n_x):
                p = idx.count(a)
                if p == 0:
                    e = np.zeros(size)
                    e[nl[a]] = 1.0
                    factors.append(e)
                else:
                    factors.append(mom[a][p - 1][:, nl[a]])
            block = factors[0]
            for f in factors[1:]:
                block = np.multiply.outer(block, f)
            coupling += row[5] * block
        de = e_grid[nl] - e_grid
        keep = np.abs(de) >= resonance_cutoff
        keep[nl] = False
        dm = float(np.sum(coupling[keep] * m_grid[keep] / de[keep]))
        m0 = m_grid[nl]
        out[i] = m0 * m0 + 2.0 * m0 * dm
    return out

import itertools
import math

import numpy as np


def _harmonic_power(p, n_top):
    size = n_top + p + 2
    q = np.zeros((size, size))
    for n in range(size - 1):
        q[n, n + 1] = q[n + 1, n] = math.sqrt((n + 1) / 2.0)
    return np.linalg.matrix_power(q, p)


def _morse_ladder(omega, x, n):
    n = np.asarray(n, dtype=float)
    return omega * (n - x * (n * n + n))


def spreading_widths(states: np.ndarray, force_terms: np.ndarray, omega_loc: np.ndarray, x_loc: np.ndarray, omega_bath: np.ndarray, n_max: int, eta: float, half_width: float) -> np.ndarray:
    st = np.array(states, dtype=float)
    om_l = np.array(omega_loc, dtype=float).ravel()
    x_l = np.array(x_loc, dtype=float).ravel()
    om_b = np.array(omega_bath, dtype=float).ravel()
    n_x, n_b = om_l.size, om_b.size
    if st.ndim != 2 or st.shape[1] != n_x + n_b + 1:
        raise ValueError("inconsistent array shapes")
    if not eta > 0.0 or half_width < 0.0:
        raise ValueError("eta must be positive and half_width non-negative")
    if st.shape[0] == 0:
        return np.zeros(0)
    terms = [row for row in np.array(force_terms, dtype=float).reshape(-1, 7) if int(row[6]) in (1, 2)]
    freq = np.concatenate([om_l, om_b])
    mom = [morse_moment_matrices(x_l[a], n_max) for a in range(n_x)]
    occ = np.rint(st[:, :n_x + n_b]).astype(int)
    top = int(occ[:, n_x:].max()) + 4 if n_b else 4
    hq = {p: _harmonic_power(p, top) for p in range(1, 5)}

    def _energy_of(n):
        e = 0.0
        for a in range(n_x):
            e += _morse_ladder(om_l[a], x_l[a], n[a])
        for b in range(n_b):
            e += om_b[b] * n[n_x + b]
        return float(e)

    out = np.empty(st.shape[0])
    for i in range(st.shape[0]):
        n = tuple(occ[i])
        e_n = _energy_of(n)
        amps = {}
        for row in terms:
            r = int(row[0])
            idx = [int(v) for v in row[1:1 + r]]
            coords = sorted(set(idx))
            choices = []
            for c in coords:
                p = idx.count(c)
                if c < n_x:
                    col = mom[c][p - 1][:, n[c]]
                    choices.append([(k, col[k]) for k in range(n_max + 1)])
                else:
                    lo = max(0, n[c] - p)
                    choices.append([(k, hq[p][k, n[c]]) for k in range(lo, n[c] + p + 1) if hq[p][k, n[c]] != 0.0])
            for combo in itertools.product(*choices):
                val = row[5]
                dest = list(n)
                for c, (k, v) in zip(coords, combo):
                    val *= v
                    dest[c] = k
                dest = tuple(dest)
                amps[dest] = amps.get(dest, 0.0) + val
        width = 0.0
        for dest, v in amps.items():
            if dest == n:
                continue
            de = _energy_of(dest) - e_n
            if abs(de) > half_width:
                continue
            diff = np.array(dest) - np.array(n)
            lose = diff < 0
            gain = diff > 0
            if not lose.any() or not gain.any():
                continue
            if np.any(freq[gain] >= freq[lose].max()):
                continue
            width += 2.0 * eta * v * v / (de * de + eta * eta)
        out[i] = width
    return out

import math

import numpy as np


def internal_conversion_rate(omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, C: np.ndarray, p_xh: np.ndarray, omega_bath: np.ndarray, delta_bath: np.ndarray, p_bath: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, thresholds: np.ndarray, e_gap: float, n_max: int, half_width: float, resonance_cutoff: float, eta: float) -> float:
    if resonance_cutoff < 0.0 or half_width < 0.0 or not eta > 0.0:
        raise ValueError("resonance_cutoff and half_width must be non-negative and eta positive")
    n_b = np.array(omega_bath, dtype=float).size
    states = window_states(omega_loc, x_loc, omega_bath, e_gap, half_width, n_max)
    if states.shape[0] == 0:
        return 0.0
    terms = local_force_field(C, cubic, quartic, n_b, thresholds)
    inten = first_order_intensities(states, terms, C, p_xh, p_bath, omega_loc, x_loc, delta_loc, delta_bath, n_max, resonance_cutoff)
    width = spreading_widths(states, terms, omega_loc, x_loc, omega_bath, n_max, eta, half_width)
    detune = e_gap - states[:, -1]
    shape = np.zeros(states.shape[0])
    live = width > 0.0
    shape[live] = width[live] / (detune[live] ** 2 + 0.25 * width[live] ** 2)
    return float(2.0 * math.pi * 2.99792458e10 * np.sum(inten * shape))
SCICODE_GOLD_EOF
