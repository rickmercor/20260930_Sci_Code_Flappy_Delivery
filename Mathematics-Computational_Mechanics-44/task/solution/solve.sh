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

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_micromodulus(E, delta, mode, z, A):
    """Eq (25): DHBB-PD micro-modulus, half the classical SHBB-PD value."""
    e = _pos(E, "E")
    d = _pos(delta, "delta")
    m = _mode(mode)
    if m == "1d":
        a = _pos(A, "A")
        c, nu = e / (d * d * a), 0.0      # 1D bar imposes no Poisson constraint; spec fixes 0.0
    elif m == "3d":
        c, nu = 6.0 * e / (np.pi * d ** 4), 0.25
    elif m == "plane_stress":
        t = _pos(z, "z")
        c, nu = 9.0 * e / (2.0 * np.pi * d ** 3 * t), 1.0 / 3.0
    else:
        t = _pos(z, "z")
        c, nu = 24.0 * e / (5.0 * np.pi * d ** 3 * t), 0.25
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("non-finite or non-positive micro-modulus")
    return np.array([c, nu, e, d], dtype=np.float64)

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_critical_stretch(Gc, E, delta, mode):
    """Eq (29): critical bond stretch from the critical energy release rate."""
    g = _pos(Gc, "Gc")
    e = _pos(E, "E")
    d = _pos(delta, "delta")
    m = _mode(mode)
    if m == "3d":
        s = np.sqrt(5.0 * g / (6.0 * e * d))
    elif m == "plane_stress":
        s = np.sqrt(4.0 * np.pi * g / (9.0 * e * d))
    elif m == "plane_strain":
        s = np.sqrt(5.0 * np.pi * g / (12.0 * e * d))
    else:
        raise ValueError("critical stretch undefined for mode " + m)
    return np.array([s, g, e, d], dtype=np.float64)

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_dual_horizon_sets(X, deltas):
    """H_i = { j != i : ||X_j - X_i|| <= delta_i } and the dual set H'_i = { j : i in H_j }.
    Returns a (2n, n) array: first n rows the horizon flags, next n rows the dual-horizon flags."""
    P = _points(X, "X")
    n = P.shape[0]
    d = _f64(deltas)
    if d.shape != (n,) or not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("bad horizon array")
    R = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    H = (R <= d[:, None]).astype(np.float64)
    np.fill_diagonal(H, 0.0)
    out = np.zeros((2 * n, n), dtype=np.float64)
    out[0:n, :] = H
    out[n:2 * n, :] = H.T          # j in H'_i  <=>  i in H_j
    return out

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_bond_state(X, x, s_c):
    """Bond stretch s_ij = (||x_ij|| - ||X_ij||) / ||X_ij|| and the intactness flag
    mu_ij = 1 while s_ij < s_c, else 0. Returns (2n, n): stretches then mu."""
    P = _points(X, "X")
    Q = _points(x, "x")
    if Q.shape != P.shape:
        raise ValueError("current and reference configurations differ in shape")
    sc = _pos(s_c, "s_c")
    n = P.shape[0]
    L0 = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    L1 = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    D = np.eye(n)
    S = (L1 - L0) / (L0 + D)       # diagonal guarded, then zeroed
    np.fill_diagonal(S, 0.0)
    MU = (S < sc).astype(np.float64)
    np.fill_diagonal(MU, 0.0)
    out = np.zeros((2 * n, n), dtype=np.float64)
    out[0:n, :] = S
    out[n:2 * n, :] = MU
    return out

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_pairwise_force(X, x, sets, state, c_j, V):
    """Eq (17): f_i = sum over H_i of c(delta_j) mu_ij s_ij (x_ij/||x_ij||) V_j
    PLUS sum over H'_i of c(delta_i) mu_ij s_ij (x_ij/||x_ij||) V_j. The own-horizon term
    carries the NEIGHBOUR's micro-modulus, the dual term the POINT's OWN; the minus sign of
    Eq (17) cancels against x_ji = -x_ij. A pair in both sets contributes twice. Returns (n, 2)."""
    P = _points(X, "X")
    Q = _points(x, "x")
    n = P.shape[0]
    St = _f64(sets)
    Sb = _f64(state)
    if St.shape != (2 * n, n) or Sb.shape != (2 * n, n):
        raise ValueError("bad sets or state array")
    cj = _f64(c_j)
    Vv = _f64(V)
    if cj.shape != (n,) or Vv.shape != (n,):
        raise ValueError("bad micro-modulus or volume array")
    if np.any(Vv <= 0.0) or not np.all(np.isfinite(cj)):
        raise ValueError("non-positive volume or non-finite micro-modulus")
    H = St[0:n, :]                 # j is inside i's own horizon
    DUAL = St[n:2 * n, :]          # i is inside j's horizon
    S = Sb[0:n, :]
    MU = Sb[n:2 * n, :]
    dx = Q[None, :, :] - Q[:, None, :]
    L = np.linalg.norm(dx, axis=2)
    L = np.where(L == 0.0, 1.0, L)
    unit = dx / L[:, :, None]
    # Eq (17): sum_{j in H_i} f_ij V_j - sum_{j in H'_i} f_ji V_j.
    # f_ij carries the NEIGHBOUR's micro-modulus c(delta_j); f_ji carries c(delta_i), and
    # because x_ji = -x_ij the minus sign flips it back to a positive contribution. A pair
    # that lies in BOTH sets therefore contributes twice, once under each micro-modulus -
    # the two sums stay separate rather than collapsing into one neighbour list.
    w = H * MU * S * cj[None, :] * Vv[None, :] + DUAL * MU * S * cj[:, None] * Vv[None, :]
    return np.einsum("ij,ijk->ik", w, unit)

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_point_damage(sets, state, V):
    """Eq (30): D_i = 1 - sum_{j in H_i} mu_ij V_j / sum_{j in H_i} V_j (volume weighted,
    over the point's OWN horizon only). Returns (n,)."""
    St = _f64(sets)
    Sb = _f64(state)
    if St.ndim != 2 or St.shape[0] % 2 != 0:
        raise ValueError("bad sets array")
    n = St.shape[0] // 2
    if St.shape != (2 * n, n) or Sb.shape != (2 * n, n):
        raise ValueError("bad sets or state array")
    Vv = _f64(V)
    if Vv.shape != (n,) or np.any(Vv <= 0.0):
        raise ValueError("bad volume array")
    H = St[0:n, :]
    MU = Sb[n:2 * n, :]
    num = (H * MU * Vv[None, :]).sum(axis=1)
    den = (H * Vv[None, :]).sum(axis=1)
    if np.any(den <= 0.0):
        raise ValueError("a material point has an empty horizon")
    return 1.0 - num / den

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def pd_audit(X, x, deltas, V, E, Gc, mode, z, A):
    """Assembles the dual-horizon bond-based peridynamic state. Returns (6, n)."""
    P = _points(X, "X")
    n = P.shape[0]
    d = _f64(deltas)
    if d.shape != (n,):
        raise ValueError("bad horizon array")
    cj = np.array([pd_micromodulus(E, float(d[j]), mode, z, A)[0] for j in range(n)],
                  dtype=np.float64)
    sc = pd_critical_stretch(Gc, E, float(np.min(d)), mode)[0]
    sets = pd_dual_horizon_sets(X, deltas)
    state = pd_bond_state(X, x, sc)
    F = pd_pairwise_force(X, x, sets, state, cj, V)
    D = pd_point_damage(sets, state, V)
    out = np.zeros((6, n), dtype=np.float64)
    out[0, :] = F[:, 0]
    out[1, :] = F[:, 1]
    out[2, :] = D
    out[3, :] = cj
    out[4, :] = np.linalg.norm(F, axis=1)
    out[5, 0] = float(np.sum(np.linalg.norm(F, axis=1)))
    out[5, 1] = sc
    if n > 2:
        out[5, 2] = float(np.sum(sets[0:n, :]))
    if n > 3:
        out[5, 3] = float(np.sum(D))
    return out
SCICODE_GOLD_EOF
