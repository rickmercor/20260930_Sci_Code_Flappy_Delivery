#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: quadratic B-spline basis and derivative (Cox-de Boor, Eqs. 34-36)."""

import numpy as np


def bspline_basis(knots, p, xi):
    knots = np.asarray(knots, dtype=np.float64)
    if p < 1 or len(knots) < p + 2:
        raise ValueError("need p >= 1 and len(knots) >= p+2")
    m = len(knots) - 1
    n = m - p - 1
    N = np.zeros((n + 1, p + 1), dtype=np.float64)
    for i in range(n + 1):
        hi = knots[i + 1]
        if (knots[i] <= xi < hi) or (xi == knots[-1] and knots[i] <= xi <= hi and hi == knots[-1]):
            N[i, 0] = 1.0
    for q in range(1, p + 1):
        for i in range(n + 1):
            a = 0.0
            d1 = knots[i + q] - knots[i]
            if d1 > 0:
                a = (xi - knots[i]) / d1 * N[i, q - 1]
            b = 0.0
            if i + 1 <= n:
                d2 = knots[i + q + 1] - knots[i + 1]
                if d2 > 0:
                    b = (knots[i + q + 1] - xi) / d2 * N[i + 1, q - 1]
            N[i, q] = a + b
    vals = N[:, p].copy()
    der = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        t1 = 0.0
        d1 = knots[i + p] - knots[i]
        if d1 > 0:
            t1 = p / d1 * N[i, p - 1]
        t2 = 0.0
        if i + 1 <= n:
            d2 = knots[i + p + 1] - knots[i + 1]
            if d2 > 0:
                t2 = p / d2 * N[i + 1, p - 1]
        der[i] = t1 - t2
    return np.vstack([vals, der])

"""Step 2: velocity-dependent DSIF factor A_I and wave speeds (Eqs. 10-14)."""

import numpy as np


def wave_factor(E, nu, rho, V):
    V1 = np.sqrt((1.0 - nu) * E / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho))
    V2 = np.sqrt(E / (2.0 * (1.0 + nu) * rho))
    if not (0.0 < V < V2):
        raise ValueError("crack speed must satisfy 0 < V < shear wave speed")
    b1 = np.sqrt(1.0 - (V / V1) ** 2)
    b2 = np.sqrt(1.0 - (V / V2) ** 2)
    A_I = b1 * (1.0 - b2 ** 2) / (4.0 * b1 * b2 - (1.0 + b2 ** 2) ** 2)
    return np.array([A_I, V1, V2, b1, b2], dtype=np.float64)

"""Step 3: J-integral weight-function nodal values (Eqs. 21-23)."""

import numpy as np


def q_weight(node_xy, h_L):
    xy = np.asarray(node_xy, dtype=np.float64)
    if not np.isfinite(h_L) or h_L <= 0:
        raise ValueError("h_L must be positive and finite")
    R_J = 1.5 * h_L
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    return (r <= R_J).astype(np.float64)

"""Step 4: strain-energy and kinetic-energy densities (Eqs. 18-19)."""

import numpy as np


def energy_densities(sigma, eps, rho, vel):
    sigma = np.asarray(sigma, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if sigma.shape != (3,) or eps.shape != (3,):
        raise ValueError("sigma and eps must be Voigt triples")
    if vel.shape != (2,):
        raise ValueError("vel must hold two components")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(eps)) and np.all(np.isfinite(vel))):
        raise ValueError("non-finite field value")
    W = 0.5 * (sigma[0] * eps[0] + sigma[1] * eps[1] + sigma[2] * eps[2])
    K = 0.5 * rho * (vel[0] ** 2 + vel[1] ** 2)
    return np.array([W, K], dtype=np.float64)

"""Step 5: global(B-spline)-local(Lagrange) coupling stiffness (Eq. 46)."""

import numpy as np


def _psD(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=np.float64)


_GPT = np.array([-(3.0 / 5.0) ** 0.5, 0.0, (3.0 / 5.0) ** 0.5], dtype=np.float64)
_GWT = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)


def _q4b(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], dtype=np.float64)
    dNr = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=np.float64)
    dNs = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=np.float64)
    return N, dNr, dNs

def _bspl(knots, p, xi):
    knots = np.asarray(knots, dtype=np.float64)
    m = len(knots) - 1
    n = m - p - 1
    N = np.zeros((n + 1, p + 1), dtype=np.float64)
    for i in range(n + 1):
        hi = knots[i + 1]
        if (knots[i] <= xi < hi) or (xi == knots[-1] and knots[i] <= xi <= hi and hi == knots[-1]):
            N[i, 0] = 1.0
    for q in range(1, p + 1):
        for i in range(n + 1):
            a = 0.0
            d1 = knots[i + q] - knots[i]
            if d1 > 0:
                a = (xi - knots[i]) / d1 * N[i, q - 1]
            b = 0.0
            if i + 1 <= n:
                d2 = knots[i + q + 1] - knots[i + 1]
                if d2 > 0:
                    b = (knots[i + q + 1] - xi) / d2 * N[i + 1, q - 1]
            N[i, q] = a + b
    vals = N[:, p].copy()
    der = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        t1 = 0.0
        d1 = knots[i + p] - knots[i]
        if d1 > 0:
            t1 = p / d1 * N[i, p - 1]
        t2 = 0.0
        if i + 1 <= n:
            d2 = knots[i + p + 1] - knots[i + 1]
            if d2 > 0:
                t2 = p / d2 * N[i + 1, p - 1]
        der[i] = t1 - t2
    return vals, der


def coupling_matrix(gknx, gkny, p, loc_elems, E, nu):
    if not isinstance(p, (int, np.integer)) or isinstance(p, bool) or p < 1:
        raise ValueError("p must be a positive integer degree")
    if len(gknx) < p + 2 or len(gkny) < p + 2:
        raise ValueError("knot vector too short for the requested degree")
    if len(loc_elems) < 1:
        raise ValueError("no local elements supplied")
    if not np.isfinite(E) or E <= 0 or not (-1.0 < float(nu) < 0.5):
        raise ValueError("invalid plane-strain constants")
    D = _psD(E, nu)
    nGx = len(gknx) - p - 1
    nGy = len(gkny) - p - 1
    nG = nGx * nGy
    node_ids = []
    conn = []
    for xy in loc_elems:
        ids = []
        for x, y in np.asarray(xy, dtype=np.float64):
            key = (round(float(x), 9), round(float(y), 9))
            if key not in node_ids:
                node_ids.append(key)
            ids.append(node_ids.index(key))
        conn.append(ids)
    nL = len(node_ids)
    K_full = np.zeros((2 * nG, 2 * nL), dtype=np.float64)
    for e, xy in enumerate(loc_elems):
        xy = np.asarray(xy, dtype=np.float64)
        Ke = np.zeros((2 * nG, 8), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                xi, eta, w = _GPT[a], _GPT[b], _GWT[a] * _GWT[b]
                N, dNr, dNs = _q4b(xi, eta)
                J00 = dNr @ xy[:, 0]; J01 = dNr @ xy[:, 1]
                J10 = dNs @ xy[:, 0]; J11 = dNs @ xy[:, 1]
                detJ = J00 * J11 - J01 * J10
                dNx = (J11 * dNr - J01 * dNs) / detJ
                dNy = (-J10 * dNr + J00 * dNs) / detJ
                BL = np.zeros((3, 8))
                for k in range(4):
                    BL[0, 2 * k] = dNx[k]; BL[1, 2 * k + 1] = dNy[k]
                    BL[2, 2 * k] = dNy[k]; BL[2, 2 * k + 1] = dNx[k]
                px = N @ xy[:, 0]; py = N @ xy[:, 1]
                Nx, dNxg = _bspl(gknx, p, px)
                Ny, dNyg = _bspl(gkny, p, py)
                BG = np.zeros((3, 2 * nG))
                idx = 0
                for jy in range(nGy):
                    for ix in range(nGx):
                        gx = dNxg[ix] * Ny[jy]
                        gy = Nx[ix] * dNyg[jy]
                        BG[0, 2 * idx] = gx; BG[1, 2 * idx + 1] = gy
                        BG[2, 2 * idx] = gy; BG[2, 2 * idx + 1] = gx
                        idx += 1
                Ke += (BG.T @ D @ BL) * detJ * w
        for k in range(4):
            gk = conn[e][k]
            K_full[:, 2 * gk] += Ke[:, 2 * k]
            K_full[:, 2 * gk + 1] += Ke[:, 2 * k + 1]
    return K_full

"""Step 6: dynamic J-integral (Eq. 17); quasi-static reduction is Eq. 20."""

import numpy as np


def _psD(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=np.float64)


_GPT = np.array([-(3.0 / 5.0) ** 0.5, 0.0, (3.0 / 5.0) ** 0.5], dtype=np.float64)
_GWT = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)


def _q4b(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], dtype=np.float64)
    dNr = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=np.float64)
    dNs = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=np.float64)
    return N, dNr, dNs


def dynamic_j_integral(loc_elems, u, ud, udd, qnod, E, nu, rho, static=False):
    n_el = len(loc_elems)
    if n_el < 1:
        raise ValueError("no local elements supplied")
    if not np.isfinite(E) or E <= 0 or not (-1.0 < float(nu) < 0.5):
        raise ValueError("invalid plane-strain constants")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    for _nm, _a in (("u", u), ("ud", ud), ("udd", udd), ("qnod", qnod)):
        if len(_a) != n_el:
            raise ValueError("field " + _nm + " does not match the element count")
    D = _psD(E, nu)
    J = 0.0
    for e, xy in enumerate(loc_elems):
        xy = np.asarray(xy, dtype=np.float64)
        ue = np.asarray(u[e], dtype=np.float64); ude = np.asarray(ud[e], dtype=np.float64)
        udde = np.asarray(udd[e], dtype=np.float64); qe = np.asarray(qnod[e], dtype=np.float64)
        for a in range(3):
            for b in range(3):
                xi, eta, w = _GPT[a], _GPT[b], _GWT[a] * _GWT[b]
                N, dNr, dNs = _q4b(xi, eta)
                J00 = dNr @ xy[:, 0]; J01 = dNr @ xy[:, 1]
                J10 = dNs @ xy[:, 0]; J11 = dNs @ xy[:, 1]
                detJ = J00 * J11 - J01 * J10
                dNx = (J11 * dNr - J01 * dNs) / detJ
                dNy = (-J10 * dNr + J00 * dNs) / detJ
                du = np.zeros((2, 2))
                for k in range(4):
                    du[0, 0] += dNx[k] * ue[k, 0]; du[0, 1] += dNy[k] * ue[k, 0]
                    du[1, 0] += dNx[k] * ue[k, 1]; du[1, 1] += dNy[k] * ue[k, 1]
                eps = np.array([du[0, 0], du[1, 1], du[0, 1] + du[1, 0]])
                sig = D @ eps
                sigt = np.array([[sig[0], sig[2]], [sig[2], sig[1]]])
                W = 0.5 * (sig @ eps)
                vel = N @ ude
                acc = N @ udde
                Kd = 0.5 * rho * (vel[0] ** 2 + vel[1] ** 2)
                dqdx = dNx @ qe; dqdy = dNy @ qe
                qv = N @ qe
                EW = W + (0.0 if static else Kd)
                term1 = 0.0
                for i in range(2):
                    s = sigt[i, 0] * du[0, 0] + sigt[i, 1] * du[1, 0]
                    s -= EW * (1.0 if i == 0 else 0.0)
                    term1 += s * (dqdx if i == 0 else dqdy)
                integ = term1
                if not static:
                    dvx = dNx @ ude[:, 0]; dvy = dNx @ ude[:, 1]
                    integ += rho * (acc[0] * du[0, 0] + acc[1] * du[1, 0]
                                    - (vel[0] * dvx + vel[1] * dvy)) * qv
                J += integ * detJ * w
    return float(J)

"""Step 7: dynamic and quasi-static stress intensity factors (Eqs. 9, 16)."""

import numpy as np


def dsif(J, A_I, E, nu, dynamic=True):
    if J < 0:
        raise ValueError("J must be non-negative")
    if dynamic:
        return float(np.sqrt(E * J / ((1.0 + nu) * A_I)))
    return float(np.sqrt(E * J / (1.0 - nu ** 2)))

"""Step 8 (final orchestrator): hybrid s-version IGA dynamic-fracture audit."""

import numpy as np

_E, _NU, _RHO, _P, _L, _HL = 1.0, 0.3, 1.0, 2, 1.5, 1.0
_V_REF = 0.4


def _mesh():
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    loc = []
    for j in range(3):
        for i in range(3):
            loc.append(np.array([[xs[i], xs[j]], [xs[i + 1], xs[j]],
                                 [xs[i + 1], xs[j + 1]], [xs[i], xs[j + 1]]], dtype=np.float64))
    return loc


def _disp(x, y):
    return np.array([0.30 * x + 0.12 * x * y + 0.06 * y * y,
                     0.18 * y + 0.15 * x * y + 0.09 * x * x])


def _vfield(x, y):
    return np.array([0.20 * y - 0.10 * x, 0.16 * x + 0.05 * y])


def hsiga_audit(crack_velocity):
    if isinstance(crack_velocity, bool) or not np.isfinite(crack_velocity) or crack_velocity <= 0:
        raise ValueError("crack_velocity must be positive and finite")
    V = float(crack_velocity)
    vs = V / _V_REF
    loc = _mesh()
    ne = len(loc)
    u = np.zeros((ne, 4, 2)); ud = np.zeros((ne, 4, 2)); udd = np.zeros((ne, 4, 2)); qn = np.zeros((ne, 4))
    for e, xy in enumerate(loc):
        for k in range(4):
            x, y = xy[k]
            u[e, k] = _disp(x, y)
            vv = _vfield(x, y)
            ud[e, k] = vs * vv
            udd[e, k] = -0.6 * vs * vv
        qn[e] = q_weight(xy, _HL)
    gknx = np.array([-_L, -_L, -_L, 0.0, _L, _L, _L])
    gkny = gknx.copy()
    wf = wave_factor(_E, _NU, _RHO, V)
    A_I, V1, V2 = float(wf[0]), float(wf[1]), float(wf[2])
    Jd = abs(dynamic_j_integral(loc, u, ud, udd, qn, _E, _NU, _RHO, False))
    Js = abs(dynamic_j_integral(loc, u, ud, udd, qn, _E, _NU, _RHO, True))
    Kd = dsif(Jd, A_I, _E, _NU, True)
    Ks = dsif(Js, A_I, _E, _NU, False)
    Kgl = coupling_matrix(gknx, gkny, _P, loc, _E, _NU)
    coupfro = float(np.linalg.norm(Kgl))
    qcount = float(qn.sum())
    eps0 = np.array([0.36, 0.255, 0.285])
    c0 = _E / ((1.0 + _NU) * (1.0 - 2.0 * _NU))
    D0 = c0 * np.array([[1.0 - _NU, _NU, 0.0], [_NU, 1.0 - _NU, 0.0], [0.0, 0.0, (1.0 - 2.0 * _NU) / 2.0]])
    sig0 = D0 @ eps0
    vel0 = vs * _vfield(0.5, 0.5)
    en = energy_densities(sig0, eps0, _RHO, vel0)
    Nb = bspline_basis(gknx, _P, 0.3)
    Npart = float(Nb[0].sum())
    return np.array([A_I, V1, V2, Jd, Js, Kd, Ks, coupfro, qcount, float(en[0]), float(en[1]), Npart],
                    dtype=np.float64)
SCICODE_GOLD_EOF
