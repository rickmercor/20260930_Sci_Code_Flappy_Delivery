#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Gold oracle: edge_frames."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def edge_frames(nodes, edges):
    """Per-edge unit tangent, length, and the source's orientation signs.

    Returns (E, 4 + n_nodes): columns 0-2 the unit tangent, column 3 the length,
    columns 4.. the sign nu(e, n) for every node n (0 when n is not an endpoint).
    """
    X = np.asarray(nodes, dtype=np.float64)
    nn = X.shape[0]
    out = np.zeros((len(edges), 4 + nn), dtype=np.float64)
    for k, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        d = X[b] - X[a]; L = float(np.linalg.norm(d))
        if L <= 0.0:
            raise ValueError("edge of zero length")
        out[k, :3] = d / L
        out[k, 3] = L
        out[k, 4 + b] = 1.0      # tangent points TOWARD b
        out[k, 4 + a] = -1.0     # and away from a
    return out

"""Gold oracle: timoshenko_operators."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def _cross(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def timoshenko_operators(i_hat, Cn, Cm, Cu, Cr):
    """Block operators of Eq 2.2a-b: Cq, Cy and the source's i-cross coupling."""
    i_hat = np.asarray(i_hat, dtype=np.float64)
    if i_hat.shape != (3,) or not np.isfinite(i_hat).all():
        raise ValueError("i_hat must be a finite 3-vector")
    if abs(float(np.linalg.norm(i_hat)) - 1.0) > 1e-9:
        raise ValueError("i_hat must be a unit tangent")
    for C in (Cn, Cm, Cu, Cr):
        if np.asarray(C).shape != (3, 3):
            raise ValueError("coefficient blocks must be 3x3")
    Cq = np.zeros((6, 6)); Cq[:3, :3] = Cn; Cq[3:, 3:] = Cm
    Cy = np.zeros((6, 6)); Cy[:3, :3] = Cu; Cy[3:, 3:] = Cr
    IX = np.zeros((6, 6)); IX[3:, :3] = _cross(i_hat)
    return np.stack([Cq, Cy, IX])

"""Gold oracle: hdg_local_matrices."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('timoshenko_operators',):
    if "" + _n not in globals() and _n in globals():
        globals()["" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr):
    """Edge matrices of the source's bilinear forms a, b, c, d and the trace form.

    Returns (5, N, N) with N = 6*(p+1), stacked as [A, B, Mass, D, T].
    """
    if p < 0 or L <= 0:
        raise ValueError("need p >= 0 and L > 0")
    ck = ("loc", p, float(L)) + _key(np.asarray(i_hat, float), np.asarray(Cn, float),
                                     np.asarray(Cm, float), np.asarray(Cu, float), np.asarray(Cr, float))
    if ck in _CACHE:
        return _CACHE[ck]
    M, Dx, e0, eL, _x, _w, _V = _basis(p, L)
    Cq, Cy, IX = timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)
    I6 = np.eye(6)
    A = np.kron(np.linalg.inv(Cq), M)
    B = np.kron(I6, Dx) + np.kron(IX, M)
    Mass = np.kron(I6, M)
    D = np.kron(np.linalg.inv(Cy), M)
    T0 = np.kron(I6, e0.reshape(1, -1)); TL = np.kron(I6, eL.reshape(1, -1))
    T = T0.T @ T0 + TL.T @ TL
    _CACHE[ck] = np.stack([A, B, Mass, D, T])
    return _CACHE[ck]

"""Gold oracle: hdg_numerical_flux."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau):
    """The source's HDG numerical flux at both endpoints (paper Eq 4.2).

    Returns (2, 6): row 0 at the start node, row 1 at the end node, each equal to
    q(n) nu(n) + tau (y(n) - lamN(n)) with nu = -1 at the start and +1 at the end.
    """
    if p < 0 or L <= 0 or tau < 0:
        raise ValueError("need p >= 0, L > 0 and tau >= 0")
    if np.asarray(lam12).size != 12:
        raise ValueError("lam12 must hold twelve nodal values")
    T0, TL = _traces(p, L)
    lam = np.asarray(lam12, dtype=np.float64)
    q = np.asarray(q_edge, dtype=np.float64); y = np.asarray(y_edge, dtype=np.float64)
    f0 = -1.0 * (T0 @ q) + tau * (T0 @ y - lam[:6])
    fL = +1.0 * (TL @ q) + tau * (TL @ y - lam[6:])
    return np.stack([f0, fL])

"""Gold oracle: edge_local_solver."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('timoshenko_operators', 'hdg_local_matrices'):
    if "" + _n not in globals() and _n in globals():
        globals()["" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z):
    """Reconstruct (q, y, z) inside one edge from the hybrid nodal data (paper Sec 5.3).

    Returns (3, N) stacked as [q, y, z] with N = 6*(p+1).
    """
    if dt <= 0 or tau <= 0:
        raise ValueError("need dt > 0 and tau > 0")
    Ms = hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)
    A, B, Mass, D, T = Ms[0], Ms[1], Ms[2], Ms[3], Ms[4]
    T0, TL = _traces(p, L)
    n = A.shape[0]; Zr = np.zeros((n, n))
    ck = ("K", p, float(L), float(tau), float(dt)) + _key(
        np.asarray(i_hat, float), np.asarray(Cn, float), np.asarray(Cm, float),
        np.asarray(Cu, float), np.asarray(Cr, float))
    if ck in _CACHE:
        K = _CACHE[ck]
    else:
        K = np.block([[A, -B.T, Zr],
                      [B, tau * T, (2.0 / dt) * Mass],
                      [Zr, -(2.0 / dt) * Mass, D]])
        _CACHE[ck] = K
    lam = np.asarray(lam12, dtype=np.float64)
    rq = -(nu0 * T0.T @ lam[:6] + nuL * TL.T @ lam[6:])
    ry = np.asarray(rhs_y, dtype=np.float64) + tau * (T0.T @ lam[:6] + TL.T @ lam[6:])
    sol = np.linalg.solve(K, np.concatenate([rq, ry, np.asarray(rhs_z, dtype=np.float64)]))
    return sol.reshape(3, n)

"""Gold oracle: condensed_node_system."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('edge_frames', 'timoshenko_operators', 'hdg_local_matrices', 'hdg_numerical_flux', 'edge_local_solver'):
    if "" + _n not in globals() and _n in globals():
        globals()["" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD):
    """Static condensation onto the free nodes (paper Eqs 5.5-5.7).

    Returns (6*nf, 6*nf + 1): the condensed operator with the load vector appended.
    """
    if p < 0 or tau <= 0 or dt <= 0:
        raise ValueError("need p >= 0, tau > 0 and dt > 0")
    if len(coeffs) != len(edges):
        raise ValueError("one coefficient set per edge is required")
    X = np.asarray(nodes, dtype=np.float64)
    fr = [j for j in range(X.shape[0]) if j not in set(dirichlet)]
    pos = {n: k for k, n in enumerate(fr)}
    nf = len(fr)
    if nf == 0:
        raise ValueError("the network has no free nodes")
    Ahat = np.zeros((6 * nf, 6 * nf)); Fhat = np.zeros(6 * nf)
    FR = edge_frames(nodes, edges)
    for k, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        i_hat, L = FR[k, :3], FR[k, 3]
        nu0, nuL = FR[k, 4 + a], FR[k, 4 + b]
        Cn, Cm, Cu, Cr = coeffs[k]
        cols = []
        for j in range(12):
            e = np.zeros(12); e[j] = 1.0
            cols.append(edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt,
                                          nu0, nuL, e, np.zeros(6 * (p + 1)),
                                          np.zeros(6 * (p + 1))).ravel())
        S_lam = np.stack(cols, axis=1)
        s_par = edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL,
                                  np.zeros(12), rhs_y[k], rhs_z[k]).ravel()
        n = 6 * (p + 1)
        lam_fixed = np.zeros(12)
        for j, nd in enumerate((a, b)):
            if nd in set(dirichlet):
                lam_fixed[6 * j:6 * j + 6] = lamD[nd]
        # the nodal condition is imposed on the numerical flux (paper Eq 4.2), so
        # build the rows by evaluating that flux on each hybrid basis response
        FL = np.zeros((2, 6, 12))
        for j in range(12):
            e = np.zeros(12); e[j] = 1.0
            FL[:, :, j] = hdg_numerical_flux(S_lam[:n, j], S_lam[n:2 * n, j], e, p, L, tau)
        FLp = hdg_numerical_flux(s_par[:n], s_par[n:2 * n], np.zeros(12), p, L, tau)
        for nd, fi, sel in ((a, 0, slice(0, 6)), (b, 1, slice(6, 12))):
            if nd in set(dirichlet):
                continue
            row = -FL[fi]
            rhsv = -FLp[fi] + row @ lam_fixed
            r = pos[nd]
            for other, osel in ((a, slice(0, 6)), (b, slice(6, 12))):
                if other in set(dirichlet):
                    continue
                c = pos[other]
                Ahat[6 * r:6 * r + 6, 6 * c:6 * c + 6] += row[:, osel]
            Fhat[6 * r:6 * r + 6] -= rhsv
    return np.hstack([Ahat, Fhat.reshape(-1, 1)])

"""Gold oracle: network_time_step."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('edge_frames', 'timoshenko_operators', 'hdg_local_matrices', 'edge_local_solver', 'condensed_node_system'):
    if "" + _n not in globals() and _n in globals():
        globals()["" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD):
    """One step of the source's energy-conservative implicit scheme (paper Eqs 5.1-5.2).

    lam_prev: list of per-edge 12-vectors holding the previous hybrid nodal state.
    Returns the (6*nf,) hybrid nodal values on the free nodes at the new time level.
    """
    if p < 0 or tau <= 0 or dt <= 0:
        raise ValueError("need p >= 0, tau > 0 and dt > 0")
    if not (len(Q) == len(Y) == len(Z) == len(lam_prev) == len(edges)):
        raise ValueError("one previous state per edge is required")
    FR = edge_frames(nodes, edges)
    rhs_y, rhs_z = [], []
    for k, (a, b) in enumerate(edges):
        L = FR[k, 3]
        Ms = hdg_local_matrices(p, L, FR[k, :3], *coeffs[k])
        B, Mass, D, T = Ms[1], Ms[2], Ms[3], Ms[4]
        T0, TL = _traces(p, L)
        lp = np.asarray(lam_prev[k], dtype=np.float64)
        Gl = tau * (T0.T @ lp[:6] + TL.T @ lp[6:])
        rhs_y.append(-B @ Q[k] - tau * T @ Y[k] + Gl + (2.0 / dt) * Mass @ Z[k])
        rhs_z.append(-(2.0 / dt) * Mass @ Y[k] - D @ Z[k])
    out = condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt,
                                rhs_y, rhs_z, lamD)
    return np.linalg.solve(out[:, :-1], out[:, -1])

"""Gold oracle: graph_forms."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def graph_forms(nodes, edges, dirichlet):
    """The source's mass-type and weighted graph Laplacian-type forms (Eq 6.1).

    Returns (2, 6*nf, 6*nf) stacked as [M_G, L_G] on the free nodes.
    """
    X = np.asarray(nodes, dtype=np.float64)
    ND = set(dirichlet)
    fr = [j for j in range(X.shape[0]) if j not in ND]
    pos = {n: i for i, n in enumerate(fr)}
    nf = len(fr)
    MG = np.zeros((6 * nf, 6 * nf)); LG = np.zeros((6 * nf, 6 * nf))
    I6 = np.eye(6)
    for (a, b) in edges:
        a = int(a); b = int(b)
        he = float(np.linalg.norm(X[b] - X[a]))
        if he <= 0.0:
            raise ValueError("edge of zero length")
        for nd in (a, b):
            if nd in ND:
                continue
            r = pos[nd]
            MG[6 * r:6 * r + 6, 6 * r:6 * r + 6] += 0.5 * he * I6
        for (u, sgu) in ((a, 1.0), (b, -1.0)):
            if u in ND:
                continue
            ru = pos[u]
            for (v, sgv) in ((a, 1.0), (b, -1.0)):
                if v in ND:
                    continue
                rv = pos[v]
                LG[6 * ru:6 * ru + 6, 6 * rv:6 * rv + 6] += 0.5 / he * sgu * sgv * I6
    return np.stack([MG, LG])

"""Gold oracle: beam_network_audit."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

P_DEG, TAU, NSTEP = 3, 1.0, 24
DT = 0.02
BASE = {1: (7, 3), 2: (6, 5), 3: (8, 2)}

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('edge_frames', 'timoshenko_operators', 'hdg_local_matrices', 'hdg_numerical_flux', 'edge_local_solver', 'condensed_node_system', 'network_time_step', 'graph_forms'):
    if "" + _n not in globals() and _n in globals():
        globals()["" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def _config(v):
    Nn, a = BASE[v]
    X = np.array([[((3 * j + a) % 7) - 3, ((5 * j + 2 * a) % 11) - 5,
                   ((2 * j + 3 * a) % 13) - 6] for j in range(Nn)], dtype=np.float64) / 3.0
    E = [(j, j + 1) for j in range(Nn - 1)] + [(j, j + 3) for j in range(Nn - 3)]
    co = []
    for k in range(len(E)):
        w = np.array([((k + 1) % 3) + 1.0, ((k + 2) % 4) + 1.0, ((k + 3) % 5) + 1.0])
        w = w / np.linalg.norm(w); O = np.outer(w, w); I3 = np.eye(3)
        co.append([I3 + 0.6 * O, I3 + 0.9 * O, I3 + 0.4 * O, I3 + 0.7 * O])
    return X, E, {0, Nn - 1}, co


def _project(p, L, k, which):
    """L2 projection onto V^p of the declared initial fields."""
    M, _Dx, _e0, _eL, x, w, V = _basis(p, L)
    out = np.zeros(6 * (p + 1))
    for c in range(6):
        if which == 0:
            g = np.sin((c + 1) * x / 2.0 + 0.3 * (k + 1)) / (c + 2.0)
        else:
            g = np.cos((c + 2) * x / 3.0 + 0.2 * (k + 1)) / (c + 3.0)
        out[c * (p + 1):(c + 1) * (p + 1)] = np.linalg.solve(M, V @ (w * g))
    return out


def _gather(edges, dirichlet, fr, pos, lam_free, lamD):
    out = []
    for (a, b) in edges:
        v = np.zeros(12)
        for j, nd in enumerate((int(a), int(b))):
            v[6 * j:6 * j + 6] = lamD[nd] if nd in set(dirichlet) else lam_free[6 * pos[nd]:6 * pos[nd] + 6]
        out.append(v)
    return out


def _initial(nodes, edges, dirichlet, coeffs, p, tau, Y0, lamD):
    """Paper Sec 5.1: q^0 and lambda^0 from the constitutive relation and balance."""
    X = np.asarray(nodes, dtype=np.float64)
    fr = [j for j in range(X.shape[0]) if j not in set(dirichlet)]
    pos = {n: i for i, n in enumerate(fr)}; nf = len(fr); n = 6 * (p + 1)
    FR = edge_frames(nodes, edges)
    Ah = np.zeros((6 * nf, 6 * nf)); Fh = np.zeros(6 * nf); per = []
    for k, (a, b) in enumerate(edges):
        a = int(a); b = int(b); L = FR[k, 3]
        nu0, nuL = FR[k, 4 + a], FR[k, 4 + b]
        Ms = hdg_local_matrices(p, L, FR[k, :3], *coeffs[k])
        Ainv = np.linalg.inv(Ms[0]); T0, TL = _traces(p, L)
        Rq = np.zeros((n, 12)); Rq[:, :6] = -nu0 * T0.T; Rq[:, 6:] = -nuL * TL.T
        Sq = Ainv @ Rq; qpar = Ainv @ (Ms[1].T @ Y0[k]); per.append((Sq, qpar))
        # the nodal condition is the source's numerical flux (Eq 4.2), evaluated on
        # each hybrid basis response and on the particular part
        FLs = np.zeros((2, 6, 12))
        for j in range(12):
            e = np.zeros(12); e[j] = 1.0
            FLs[:, :, j] = hdg_numerical_flux(Sq[:, j], np.zeros(n), e, p, L, tau)
        FLp = hdg_numerical_flux(qpar, Y0[k], np.zeros(12), p, L, tau)
        lam_fixed = np.zeros(12)
        for j, nd in enumerate((a, b)):
            if nd in set(dirichlet):
                lam_fixed[6 * j:6 * j + 6] = lamD[nd]
        for nd, fi, sel in ((a, 0, slice(0, 6)), (b, 1, slice(6, 12))):
            if nd in set(dirichlet):
                continue
            row = -FLs[fi]
            rhsv = -FLp[fi] + row @ lam_fixed
            r = pos[nd]
            for other, osel in ((a, slice(0, 6)), (b, slice(6, 12))):
                if other in set(dirichlet):
                    continue
                Ah[6 * r:6 * r + 6, 6 * pos[other]:6 * pos[other] + 6] += row[:, osel]
            Fh[6 * r:6 * r + 6] -= rhsv
    lam_free = np.linalg.solve(Ah, Fh)
    lam_e = _gather(edges, dirichlet, fr, pos, lam_free, lamD)
    Q0 = [per[k][0] @ lam_e[k] + per[k][1] for k in range(len(edges))]
    return Q0, lam_free, lam_e


def _energy(edges, coeffs, p, tau, FR, Q, Y, Z, lam_e):
    E = 0.0
    for k in range(len(edges)):
        L = FR[k, 3]
        Ms = hdg_local_matrices(p, L, FR[k, :3], *coeffs[k])
        T0, TL = _traces(p, L)
        E += 0.5 * Q[k] @ Ms[0] @ Q[k] + 0.5 * Z[k] @ Ms[3] @ Z[k]
        for T, sel in ((T0, slice(0, 6)), (TL, slice(6, 12))):
            j = T @ Y[k] - lam_e[k][sel]
            E += 0.5 * tau * (j @ j)
    return float(E)


def _spectral_constants(Gforms, dt, Ahat):
    """Extremal generalised eigenvalues of Ahat against MG/dt^2 + LG (Eq 6.2)."""
    G = np.asarray(Gforms, dtype=np.float64)
    S = G[0] / (dt * dt) + G[1]
    Ssym = 0.5 * (S + S.T); Asym = 0.5 * (Ahat + Ahat.T)
    w, V = np.linalg.eigh(Ssym)
    if np.min(w) <= 0:
        raise ValueError("graph form is not positive definite")
    Wi = (V * (1.0 / np.sqrt(w))) @ V.T
    ev = np.linalg.eigvalsh(Wi @ Asym @ Wi)
    return float(ev.min()), float(ev.max())


def _step_rhs(E, co, p, tau, dt, FR, Q, Y, Z, lam_prev):
    """The per-edge right-hand sides of the source's time step; the same expressions
    the time-stepping sub-problem builds, kept here so the reconstruction after the
    solve uses exactly what went into it."""
    ry, rz = [], []
    for k in range(len(E)):
        L = FR[k, 3]
        Ms = hdg_local_matrices(p, L, FR[k, :3], *co[k]); T0, TL = _traces(p, L)
        lp = np.asarray(lam_prev[k], dtype=np.float64)
        Gl = tau * (T0.T @ lp[:6] + TL.T @ lp[6:])
        ry.append(-Ms[1] @ Q[k] - tau * Ms[4] @ Y[k] + Gl + (2.0 / dt) * Ms[2] @ Z[k])
        rz.append(-(2.0 / dt) * Ms[2] @ Y[k] - Ms[3] @ Z[k])
    return ry, rz


def beam_network_audit(dt_scale):
    if isinstance(dt_scale, bool) or not np.isfinite(dt_scale) or dt_scale <= 0:
        raise ValueError("dt_scale must be positive and finite")
    rows = []
    for v in (1, 2, 3):
        X, E, ND, co = _config(v)
        dt = DT * float(dt_scale)
        p, tau = P_DEG, TAU
        FR = edge_frames(X, E)
        fr = [j for j in range(X.shape[0]) if j not in ND]
        pos = {n: i for i, n in enumerate(fr)}
        lamD = {nd: np.zeros(6) for nd in ND}
        Y = [_project(p, FR[k, 3], k, 0) for k in range(len(E))]
        Y1 = [_project(p, FR[k, 3], k, 1) for k in range(len(E))]
        Z = [np.kron(timoshenko_operators(FR[k, :3], *co[k])[1], np.eye(p + 1)) @ Y1[k]
             for k in range(len(E))]
        Q, lam_free, lam_e = _initial(X, E, ND, co, p, tau, Y, lamD)
        E0 = _energy(E, co, p, tau, FR, Q, Y, Z, lam_e)
        for _ in range(NSTEP):
            ry, rz = _step_rhs(E, co, p, tau, dt, FR, Q, Y, Z, lam_e)
            lam_free = network_time_step(X, E, ND, co, p, tau, dt, Q, Y, Z, lam_e, lamD)
            lam_e = _gather(E, ND, fr, pos, lam_free, lamD)
            newQ, newY, newZ = [], [], []
            for k in range(len(E)):
                sol = edge_local_solver(p, FR[k, 3], FR[k, :3], *co[k], tau, dt,
                                        FR[k, 4 + int(E[k][0])], FR[k, 4 + int(E[k][1])],
                                        lam_e[k], ry[k], rz[k])
                newQ.append(sol[0]); newY.append(sol[1]); newZ.append(sol[2])
            Q, Y, Z = newQ, newY, newZ
        EK = _energy(E, co, p, tau, FR, Q, Y, Z, lam_e)
        u = sum(float(np.linalg.norm(lam_free[6 * i:6 * i + 3])) for i in range(len(fr)))
        s = sum(float(np.linalg.norm(lam_free[6 * i:6 * i + 6])) for i in range(len(fr)))
        out = condensed_node_system(X, E, ND, co, p, tau, dt,
                                    [np.zeros(6 * (p + 1))] * len(E),
                                    [np.zeros(6 * (p + 1))] * len(E), lamD)
        Gforms = graph_forms(X, E, ND)
        al, be = _spectral_constants(Gforms, dt, out[:, :-1])
        # the source's nodal balance: the fluxes meeting at a free node must cancel
        bal = {nd: np.zeros(6) for nd in fr}
        scale = 0.0
        for k in range(len(E)):
            fl = hdg_numerical_flux(Q[k], Y[k], lam_e[k], p, FR[k, 3], tau)
            scale = max(scale, float(np.max(np.abs(fl))),
                        float(np.max(np.abs(hdg_local_matrices(p, FR[k, 3], FR[k, :3], *co[k])[0]))))
            for j, nd in enumerate((int(E[k][0]), int(E[k][1]))):
                if nd in fr:
                    bal[nd] = bal[nd] + fl[j]
        resid = max(float(np.max(np.abs(bal[nd]))) for nd in fr) if fr else 0.0
        if not np.isfinite(resid) or resid > 1e-6 * max(1.0, scale):
            raise ValueError("the source's nodal balance is not satisfied at the final state")
        rows.append([E0, EK, float(np.linalg.norm(lam_free)), u, s, al, be])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
