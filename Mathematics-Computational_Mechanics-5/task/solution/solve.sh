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


_GP = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])

_CORNER = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)

def hex_stiffness(E, nu, hx, hy, hz):
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError("E must be a positive finite modulus")
    if not np.isfinite(nu) or nu <= -1.0 or nu >= 0.5:
        raise ValueError("nu must lie in (-1, 0.5)")
    if min(hx, hy, hz) <= 0.0:
        raise ValueError("element sizes must be positive")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    Jd = np.array([hx / 2.0, hy / 2.0, hz / 2.0])
    detJ = float(np.prod(Jd))
    K = np.zeros((24, 24))
    s = _CORNER
    for xi in _GP:
        for eta in _GP:
            for zt in _GP:
                dN = np.empty((8, 3))
                dN[:, 0] = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zt)
                dN[:, 1] = 0.125 * s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zt)
                dN[:, 2] = 0.125 * s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta)
                g = dN / Jd[None, :]
                B = np.zeros((6, 24))
                for a in range(8):
                    B[0, 3 * a + 0] = g[a, 0]
                    B[1, 3 * a + 1] = g[a, 1]
                    B[2, 3 * a + 2] = g[a, 2]
                    B[3, 3 * a + 1] = g[a, 2]
                    B[3, 3 * a + 2] = g[a, 1]
                    B[4, 3 * a + 0] = g[a, 2]
                    B[4, 3 * a + 2] = g[a, 0]
                    B[5, 3 * a + 0] = g[a, 1]
                    B[5, 3 * a + 1] = g[a, 0]
                K += (B.T @ C @ B) * detJ
    return K

import numpy as np


def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def block_stiffness(n, Ke):
    if n < 1:
        raise ValueError("n must be at least one element per side")
    Ke = np.asarray(Ke, dtype=float)
    if Ke.shape != (24, 24) or not np.all(np.isfinite(Ke)):
        raise ValueError("Ke must be a finite (24, 24) element stiffness matrix")
    nn = (n + 1) * (n + 1) * 2
    A = np.zeros((3 * nn, 3 * nn))
    for j in range(n):
        for i in range(n):
            nodes = [_node_id(n, 1, i + a, j + b, c)
                     for a, b, c in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
            dofs = np.array([3 * p + d for p in nodes for d in range(3)])
            A[np.ix_(dofs, dofs)] += Ke
    return A

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)

def _hat_integral(a, b, c0, c1, at_lo):
    """Integral over [a,b] of the 1D hat on [c0,c1] equal to one at c0 (at_lo) or at c1."""
    L = c1 - c0
    if at_lo:
        return ((c1 - a) ** 2 - (c1 - b) ** 2) / (2.0 * L)
    return ((b - c0) ** 2 - (a - c0) ** 2) / (2.0 * L)

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _face_id(n, i, j):
    return j * n + i

def mortar_mass(n1):
    if n1 < 1:
        raise ValueError("n1 must be at least one")
    e = _edges(n1)
    nn = (n1 + 1) * (n1 + 1) * 2
    D = np.zeros((3 * n1 * n1, 3 * nn))
    for j in range(n1):
        for i in range(n1):
            f = _face_id(n1, i, j)
            x0, x1 = e[i], e[i + 1]
            y0, y1 = e[j], e[j + 1]
            for di, dj in _QUAD:
                node = _node_id(n1, 1, i + di, j + dj, 1)
                v = _hat_integral(x0, x1, x0, x1, di == 0) * _hat_integral(y0, y1, y0, y1, dj == 0)
                for d in range(3):
                    D[3 * f + d, 3 * node + d] += v
    return D

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)

def _overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo), lo, hi

def _hat_integral(a, b, c0, c1, at_lo):
    """Integral over [a,b] of the 1D hat on [c0,c1] equal to one at c0 (at_lo) or at c1."""
    L = c1 - c0
    if at_lo:
        return ((c1 - a) ** 2 - (c1 - b) ** 2) / (2.0 * L)
    return ((b - c0) ** 2 - (a - c0) ** 2) / (2.0 * L)

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _face_id(n, i, j):
    return j * n + i

def mortar_coupling(n1, n2):
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least one")
    e1, e2 = _edges(n1), _edges(n2)
    nn2 = (n2 + 1) * (n2 + 1) * 2
    M = np.zeros((3 * n1 * n1, 3 * nn2))
    for j in range(n1):
        for i in range(n1):
            f = _face_id(n1, i, j)
            X0, X1, Y0, Y1 = e1[i], e1[i + 1], e1[j], e1[j + 1]
            for j2 in range(n2):
                ly, y0, y1 = _overlap(Y0, Y1, e2[j2], e2[j2 + 1])
                if ly <= 0.0:
                    continue
                for i2 in range(n2):
                    lx, x0, x1 = _overlap(X0, X1, e2[i2], e2[i2 + 1])
                    if lx <= 0.0:
                        continue
                    for di, dj in _QUAD:
                        node = _node_id(n2, 1, i2 + di, j2 + dj, 0)
                        v = (_hat_integral(x0, x1, e2[i2], e2[i2 + 1], di == 0)
                             * _hat_integral(y0, y1, e2[j2], e2[j2 + 1], dj == 0))
                        for d in range(3):
                            M[3 * f + d, 3 * node + d] += v
    return M

import numpy as np


def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)

def _overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo), lo, hi

def _face_id(n, i, j):
    return j * n + i

def macroelement_masks(n1, n2, i2, j2):
    if not (1 <= i2 <= n2 - 1 and 1 <= j2 <= n2 - 1):
        raise ValueError("(i2, j2) must be an internal node of the mortar interface grid")
    mortar = [(a, b) for a in (i2 - 1, i2) for b in (j2 - 1, j2)
              if 0 <= a < n2 and 0 <= b < n2]
    e1, e2 = _edges(n1), _edges(n2)
    X0 = min(e2[a] for a, _ in mortar)
    X1 = max(e2[a + 1] for a, _ in mortar)
    Y0 = min(e2[b] for _, b in mortar)
    Y1 = max(e2[b + 1] for _, b in mortar)
    m = np.zeros(n1 * n1 + n2 * n2)
    for j in range(n1):
        for i in range(n1):
            lx, _, _ = _overlap(e1[i], e1[i + 1], X0, X1)
            ly, _, _ = _overlap(e1[j], e1[j + 1], Y0, Y1)
            if lx > 1e-14 and ly > 1e-14:
                m[_face_id(n1, i, j)] = 1.0
    for a, b in mortar:
        m[n1 * n1 + _face_id(n2, a, b)] = 1.0
    return m

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _face_id(n, i, j):
    return j * n + i

def local_scaling(n1, n2, i2, j2, A1, A2, D, M):
    mask = macroelement_masks(n1, n2, i2, j2)
    ncells = [(i, j) for j in range(n1) for i in range(n1) if mask[_face_id(n1, i, j)] > 0.0]
    mcells = [(a, b) for b in range(n2) for a in range(n2) if mask[n1 * n1 + _face_id(n2, a, b)] > 0.0]
    It = np.array([3 * _face_id(n1, i, j) + d for (i, j) in ncells for d in range(3)])
    n1set = sorted({_node_id(n1, 1, i + di, j + dj, 1) for (i, j) in ncells for di, dj in _QUAD})
    n2set = sorted({_node_id(n2, 1, a + da, b + db, 0) for (a, b) in mcells for da, db in _QUAD})
    Iu1 = np.array([3 * p + d for p in n1set for d in range(3)])
    Iu2 = np.array([3 * p + d for p in n2set for d in range(3)])
    dA = np.concatenate([np.diag(A1[np.ix_(Iu1, Iu1)]), np.diag(A2[np.ix_(Iu2, Iu2)])])
    if np.any(dA <= 0.0):
        raise ValueError("the gathered local stiffness diagonal must be positive")
    Bh = np.hstack([D[np.ix_(It, Iu1)], -M[np.ix_(It, Iu2)]])
    return Bh @ (Bh / dA[None, :]).T

import numpy as np


def _face_id(n, i, j):
    return j * n + i

def stabilization_matrix(n1, n2, A1, A2, D, M):
    if n1 < 2 or n2 < 2:
        raise ValueError("both interface grids need at least two cells per side")
    if np.asarray(D).shape[0] != 3 * n1 * n1 or np.asarray(M).shape[0] != 3 * n1 * n1:
        raise ValueError("both interface operators must have one row block per non-mortar face")
    nt = n1 * n1
    H = np.zeros((3 * nt, 3 * nt))
    for j2 in range(1, n2):
        for i2 in range(1, n2):
            St = local_scaling(n1, n2, i2, j2, A1, A2, D, M)
            mask = macroelement_masks(n1, n2, i2, j2)
            ncells = [(i, j) for j in range(n1) for i in range(n1) if mask[_face_id(n1, i, j)] > 0.0]
            loc = {c: k for k, c in enumerate(ncells)}
            for (i, j) in ncells:
                for di, dj in ((1, 0), (0, 1)):
                    nb = (i + di, j + dj)
                    if nb not in loc:
                        continue
                    K, L = loc[(i, j)], loc[nb]
                    SE = 0.5 * (St[3 * K:3 * K + 3, 3 * K:3 * K + 3]
                                + St[3 * L:3 * L + 3, 3 * L:3 * L + 3])
                    gK = np.array([3 * _face_id(n1, i, j) + d for d in range(3)])
                    gL = np.array([3 * _face_id(n1, nb[0], nb[1]) + d for d in range(3)])
                    H[np.ix_(gK, gK)] += SE
                    H[np.ix_(gL, gL)] += SE
                    H[np.ix_(gK, gL)] -= SE
                    H[np.ix_(gL, gK)] -= SE
    return H

import numpy as np


def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _far_face(n, k):
    out = []
    for j in range(n + 1):
        for i in range(n + 1):
            p = _node_id(n, 1, i, j, k)
            out += [3 * p, 3 * p + 1, 3 * p + 2]
    return np.array(sorted(set(out)))

def infsup_constant(A, B, Q, H, h):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    if A.shape[0] != A.shape[1] or B.shape[0] != A.shape[0]:
        raise ValueError("A must be square and B must have as many rows as A")
    if Q.shape != H.shape or Q.shape[0] != B.shape[1]:
        raise ValueError("Q and H must be square of size equal to the number of tractions")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a positive finite mesh size")
    import scipy.linalg as _sla
    S = B.T @ np.linalg.solve(A, B)
    w = _sla.eigvalsh(S + H, h * Q)
    return float(np.sqrt(max(float(w[0]), 0.0)))

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _far_face(n, k):
    out = []
    for j in range(n + 1):
        for i in range(n + 1):
            p = _node_id(n, 1, i, j, k)
            out += [3 * p, 3 * p + 1, 3 * p + 2]
    return np.array(sorted(set(out)))

def _top_load(n2, varying):
    f = np.zeros(3 * (n2 + 1) * (n2 + 1) * 2)
    h = 1.0 / n2
    a = h * h / 4.0
    for j in range(n2):
        for i in range(n2):
            for di, dj in _QUAD:
                p = _node_id(n2, 1, i + di, j + dj, 1)
                x, y = (i + di) * h, (j + dj) * h
                if varying:
                    f[3 * p + 0] += a * (0.30 * y)
                    f[3 * p + 1] += a * (0.20 * x)
                    f[3 * p + 2] += a * (-(1.0 + 0.5 * x + 0.25 * y))
                else:
                    f[3 * p + 2] += -a
    return f

def interface_tractions(A, B, H, f):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float)
    if f.shape[0] != A.shape[0]:
        raise ValueError("f must have one entry per displacement degree of freedom")
    nt = B.shape[1]
    K = np.block([[A, B], [B.T, -H]])
    rhs = np.concatenate([f, np.zeros(nt)])
    sol = np.linalg.solve(K, rhs)
    return sol[A.shape[0]:]

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

_CONFIGS = ((4, 2), (6, 3), (8, 4))

_PARS = dict(E=1.0, nu=0.0)

def _far_face(n, k):
    out = []
    for j in range(n + 1):
        for i in range(n + 1):
            p = _node_id(n, 1, i, j, k)
            out += [3 * p, 3 * p + 1, 3 * p + 2]
    return np.array(sorted(set(out)))

def _top_load(n2, varying):
    f = np.zeros(3 * (n2 + 1) * (n2 + 1) * 2)
    h = 1.0 / n2
    a = h * h / 4.0
    for j in range(n2):
        for i in range(n2):
            for di, dj in _QUAD:
                p = _node_id(n2, 1, i + di, j + dj, 1)
                x, y = (i + di) * h, (j + dj) * h
                if varying:
                    f[3 * p + 0] += a * (0.30 * y)
                    f[3 * p + 1] += a * (0.20 * x)
                    f[3 * p + 2] += a * (-(1.0 + 0.5 * x + 0.25 * y))
                else:
                    f[3 * p + 2] += -a
    return f

def mortar_audit(depth):
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be a positive finite block thickness")
    import scipy.linalg as _sla
    E, nu = _PARS["E"], _PARS["nu"]
    rows = []
    for (n1, n2) in _CONFIGS:
        Ke1 = hex_stiffness(E, nu, 1.0 / n1, 1.0 / n1, depth)
        Ke2 = hex_stiffness(E, nu, 1.0 / n2, 1.0 / n2, depth)
        A1 = block_stiffness(n1, Ke1)
        A2 = block_stiffness(n2, Ke2)
        D = mortar_mass(n1)
        M = mortar_coupling(n1, n2)
        H = stabilization_matrix(n1, n2, A1, A2, D, M)
        Z = np.zeros_like(H)
        nt = n1 * n1
        Q = np.diag(np.repeat(1.0 / (n1 * n1), 3 * nt))
        h = 1.0 / n1
        # stability setting: both far faces clamped
        S1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))
        S2 = np.setdiff1d(np.arange(A2.shape[0]), _far_face(n2, 1))
        As = _sla.block_diag(A1[np.ix_(S1, S1)], A2[np.ix_(S2, S2)])
        Bs = np.hstack([D[:, S1], -M[:, S2]]).T
        b_un = infsup_constant(As, Bs, Q, Z, h)
        b_st = infsup_constant(As, Bs, Q, H, h)
        # load setting: bottom face clamped, top face loaded
        L1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))
        L2 = np.arange(A2.shape[0])
        Al = _sla.block_diag(A1[np.ix_(L1, L1)], A2[np.ix_(L2, L2)])
        Bl = np.hstack([D[:, L1], -M[:, L2]]).T
        fv = np.concatenate([np.zeros(len(L1)), _top_load(n2, True)[L2]])
        fc = np.concatenate([np.zeros(len(L1)), _top_load(n2, False)[L2]])
        tv = interface_tractions(Al, Bl, H, fv)
        tc = interface_tractions(Al, Bl, H, fc)
        tN = tv[2::3]
        tT = np.sqrt(tv[0::3] ** 2 + tv[1::3] ** 2)
        patch = float(np.max(np.abs(tc[2::3] - np.mean(tc[2::3]))))
        mask = macroelement_masks(n1, n2, 1, 1)
        patch_faces = float(np.sum(mask[:n1 * n1]))
        scal = local_scaling(n1, n2, 1, 1, A1, A2, D, M)
        rows.append([b_un, b_st, float(np.linalg.norm(H)), float(np.mean(tN)),
                     float(np.min(tN)), float(np.max(tN)), float(np.mean(tT)), patch,
                     patch_faces, float(np.linalg.norm(scal))])
    return np.array(rows)
SCICODE_GOLD_EOF
