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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def potential_and_derivatives(points: "np.ndarray", a: float, c: float, wy: float) -> "np.ndarray":
    X = np.atleast_2d(np.asarray(points, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    for v in (a, c, wy):
        if not np.isfinite(v):
            raise ValueError("a, c and wy must be finite")
    x, y = X[:, 0], X[:, 1]
    u = y + c * (x * x - 1.0)
    V = 0.125 * (x * x - 1.0) ** 2 + 0.5 * wy * wy * u * u + a * x ** 3
    gx = 0.5 * x * (x * x - 1.0) + wy * wy * u * 2.0 * c * x + 3.0 * a * x * x
    gy = wy * wy * u
    hxx = 1.5 * x * x - 0.5 + wy * wy * (2.0 * c * u + (2.0 * c * x) ** 2) + 6.0 * a * x
    hxy = wy * wy * 2.0 * c * x
    hyy = np.full_like(x, wy * wy)
    return np.stack([V, gx, gy, hxx, hxy, hyy], axis=1)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def well_data(a: float, c: float, wy: float, hbar: float) -> "np.ndarray":
    mass = 1.0
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    rows = []
    for start in ([-1.0, 0.0], [1.0, 0.0]):
        p = np.array(start, dtype=float)
        for _ in range(300):
            row = potential_and_derivatives(p[None, :], a, c, wy)[0]
            g = row[1:3]
            H = np.array([[row[3], row[4]], [row[4], row[5]]])
            if np.max(np.abs(g)) < 1e-15:
                break
            w, U = np.linalg.eigh(H)
            w = np.where(w < 1e-12, 1e-12, w)
            p = p - U @ ((U.T @ g) / w)
        row = potential_and_derivatives(p[None, :], a, c, wy)[0]
        H = np.array([[row[3], row[4]], [row[4], row[5]]])
        om = np.sqrt(np.maximum(np.linalg.eigvalsh(H), 0.0) / mass)
        rows.append([p[0], p[1], om[0], om[1], row[0],
                     row[0] + 0.5 * hbar * float(om.sum())])
    return np.array(rows, dtype=float)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def initial_path(a: float, c: float, wy: float, well: "np.ndarray", n_beads: int, tau_inst: float) -> "np.ndarray":
    n_quad = 200001
    eta = 1e-10
    mass = 1.0
    well = np.asarray(well, dtype=float)
    if well.shape != (2, 6):
        raise ValueError("well must have shape (2, 6)")
    n_beads = int(n_beads)
    if n_beads < 2:
        raise ValueError("n_beads must be at least 2")
    if not np.isfinite(tau_inst) or tau_inst <= 0.0:
        raise ValueError("tau_inst must be positive and finite")
    x_l, x_r, E = well[0, 0], well[1, 0], well[1, 4]
    Veff = lambda z: 0.125 * (z * z - 1.0) ** 2 + a * z ** 3
    dVeff = lambda z: 0.5 * z * (z * z - 1.0) + 3.0 * a * z * z
    arc = lambda z: np.sqrt(1.0 + 4.0 * c * c * z * z)
    om_eff = lambda z: np.sqrt(abs(0.5 * (3.0 * z * z - 1.0) + 6.0 * a * z)) / arc(z)
    x_endR = brentq(lambda z: Veff(z) - E - eta, 0.0, x_r, xtol=1e-16, rtol=8.9e-16)
    tg = np.arange(n_beads + 1) * (tau_inst / n_beads)
    if abs(dVeff(x_l)) < 1e-13 and abs(Veff(x_l) - E) < 1e-18:
        x_endL = brentq(lambda z: Veff(z) - E - eta, x_l, 0.0, xtol=1e-16, rtol=8.9e-16)
        xs = np.linspace(x_endL, x_endR, n_quad)
        integ = arc(xs) / np.sqrt(2.0 * np.maximum(Veff(xs) - E, 1e-300) / mass)
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(xs))])
        lead = 0.5 * (tau_inst - tau[-1])
        xx = np.empty(n_beads + 1)
        m1 = tg < lead
        xx[m1] = x_l + (x_endL - x_l) * np.exp(-om_eff(x_l) * (lead - tg[m1]))
        m2 = (tg >= lead) & (tg <= lead + tau[-1])
        xx[m2] = np.interp(tg[m2] - lead, tau, xs)
        m3 = tg > lead + tau[-1]
        xx[m3] = x_r - (x_r - x_endR) * np.exp(-om_eff(x_r) * (tg[m3] - lead - tau[-1]))
    else:
        x_b = brentq(lambda z: Veff(z) - E, x_l, 0.0, xtol=1e-16, rtol=8.9e-16)
        u = np.linspace(0.0, np.sqrt(x_endR - x_b), n_quad)
        xs = x_b + u * u
        vv = np.maximum(Veff(xs) - E, 0.0)
        integ = np.zeros_like(u)
        nz = vv > 0
        integ[nz] = 2.0 * u[nz] * arc(xs[nz]) / np.sqrt(2.0 * vv[nz] / mass)
        integ[0] = 2.0 * arc(x_b) / np.sqrt(2.0 * abs(dVeff(x_b)) / mass)
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(u))])
        xx = np.empty(n_beads + 1)
        near = tg <= tau[-1]
        xx[near] = np.interp(tg[near], tau, xs)
        xx[~near] = x_r - (x_r - x_endR) * np.exp(-om_eff(x_r) * (tg[~near] - tau[-1]))
    return np.stack([xx, -c * (xx * xx - 1.0)], axis=1)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def path_action(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> float:
    mass = 1.0
    X = np.atleast_2d(np.asarray(path, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 3:
        raise ValueError("path must have shape (n + 1, 2) with n >= 2")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    V = potential_and_derivatives(X, a, c, wy)[:, 0]
    S = (mass / (2.0 * eps)) * float(np.sum(np.diff(X, axis=0) ** 2))
    return float(S + eps * (0.5 * V[0] + float(V[1:-1].sum()) + 0.5 * V[-1]))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def path_gradient(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> "np.ndarray":
    mass = 1.0
    X = np.atleast_2d(np.asarray(path, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 3:
        raise ValueError("path must have shape (n + 1, 2) with n >= 2")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    Gv = potential_and_derivatives(X, a, c, wy)[:, 1:3]
    G = np.zeros_like(X)
    G[1:-1] = (mass / eps) * (2.0 * X[1:-1] - X[:-2] - X[2:]) + eps * Gv[1:-1]
    G[0] = (mass / eps) * (X[0] - X[1]) + 0.5 * eps * Gv[0]
    G[-1] = (mass / eps) * (X[-1] - X[-2]) + 0.5 * eps * Gv[-1]
    return G

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _block_tridiagonal_solve(D, off, R):
    """Block Thomas elimination for a symmetric block-tridiagonal system."""
    n, f, _ = D.shape
    C = np.zeros((n, f, f))
    Y = np.zeros((n, f))
    C[0] = np.linalg.solve(D[0], off)
    Y[0] = np.linalg.solve(D[0], R[0])
    for i in range(1, n):
        B = D[i] - off.T @ C[i - 1]
        rhs = R[i] - off.T @ Y[i - 1]
        if i < n - 1:
            C[i] = np.linalg.solve(B, off)
        Y[i] = np.linalg.solve(B, rhs)
    out = np.zeros((n, f))
    out[-1] = Y[-1]
    for i in range(n - 2, -1, -1):
        out[i] = Y[i] - C[i] @ out[i + 1]
    return out


def optimise_instanton(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> "np.ndarray":
    max_iter = 600
    tol = 1e-13
    mass = 1.0
    X = np.array(np.atleast_2d(np.asarray(path, dtype=float)), dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 3:
        raise ValueError("path must have shape (n + 1, 2) with n >= 2")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    eye = np.eye(2)
    G = path_gradient(X, eps, a, c, wy)
    n0 = np.linalg.norm(G)
    mu = 1e-3
    for _ in range(int(max_iter)):
        if n0 < tol:
            break
        rows = potential_and_derivatives(X, a, c, wy)
        H = np.empty((X.shape[0], 2, 2))
        H[:, 0, 0] = rows[:, 3]
        H[:, 0, 1] = rows[:, 4]
        H[:, 1, 0] = rows[:, 4]
        H[:, 1, 1] = rows[:, 5]
        D = eps * H
        D[0] *= 0.5
        D[-1] *= 0.5
        D[1:-1] += (2.0 * mass / eps) * eye
        D[0] += (mass / eps) * eye
        D[-1] += (mass / eps) * eye
        off = -(mass / eps) * eye
        try:
            step = _block_tridiagonal_solve(D + mu * eye, off, -G)
        except np.linalg.LinAlgError:
            mu *= 4.0
            continue
        Xn = X + step
        Gn = path_gradient(Xn, eps, a, c, wy)
        nn = np.linalg.norm(Gn)
        if nn < n0:
            X, G, n0 = Xn, Gn, nn
            mu = max(mu * 0.4, 1e-15)
        else:
            mu *= 4.0
            if mu > 1e12:
                break
    return X

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def surface_data(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> "np.ndarray":
    X = np.atleast_2d(np.asarray(path, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 3:
        raise ValueError("path must have shape (n + 1, 2) with n >= 2")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    N = X.shape[0] - 1
    ring = np.concatenate([X, X[-2:0:-1]], axis=0)
    n = ring.shape[0]
    V = potential_and_derivatives(X, a, c, wy)[:, 0]
    j = int(np.argmax(V[1:-1])) + 1
    qdot = float(np.linalg.norm(ring[(j + 1) % n] - ring[(j - 1) % n]) / (2.0 * eps))
    return np.array([float(j), float(2 * j), float(2 * N - 2 * j), qdot])

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def collapsed_log_determinant(n_ring: int, eps: float, omegas: "np.ndarray") -> float:
    n_ring = int(n_ring)
    if n_ring < 1:
        raise ValueError("n_ring must be a positive integer")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    om = np.atleast_1d(np.asarray(omegas, dtype=float))
    if om.size == 0 or not np.all(np.isfinite(om)):
        raise ValueError("omegas must be a non-empty array of finite frequencies")
    k = np.arange(n_ring)
    base = 4.0 * np.sin(np.pi * k / float(n_ring)) ** 2
    return float(sum(np.sum(np.log(base + (eps * w) ** 2)) for w in om))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def pinned_log_determinant(path: "np.ndarray", eps: float, a: float, c: float, wy: float) -> float:
    mass = 1.0
    X = np.atleast_2d(np.asarray(path, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 3:
        raise ValueError("path must have shape (n + 1, 2) with n >= 2")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    N = X.shape[0] - 1
    ring = np.concatenate([X, X[-2:0:-1]], axis=0)
    n, f = ring.shape
    sd = surface_data(X, eps, a, c, wy)
    i0 = int(sd[0])
    i1 = 2 * N - i0
    rows = potential_and_derivatives(ring, a, c, wy)
    eye = np.eye(f)
    diag = np.empty((n, f, f))
    for i in range(n):
        H = np.array([[rows[i, 3], rows[i, 4]], [rows[i, 4], rows[i, 5]]])
        diag[i] = (eps / mass) * ((2.0 * mass / eps) * eye + eps * H)
    rot = {}
    for bead in (i0, i1):
        t = ring[(bead + 1) % n] - ring[(bead - 1) % n]
        t = t / np.linalg.norm(t)
        e = np.zeros(f)
        e[0] = 1.0
        v = t - e
        nv = np.linalg.norm(v)
        rot[bead] = eye.copy() if nv < 1e-14 else eye - 2.0 * np.outer(v / nv, v / nv)
        diag[bead] = rot[bead] @ diag[bead] @ rot[bead].T
    ii, jj, vv = [], [], []
    for i in range(n):
        for p in range(f):
            for q in range(f):
                ii.append(i * f + p); jj.append(i * f + q); vv.append(diag[i][p, q])
    for i in range(n):
        k = (i + 1) % n
        B = -eye.copy()
        if i in rot:
            B = rot[i] @ B
        if k in rot:
            B = B @ rot[k].T
        for p in range(f):
            for q in range(f):
                if B[p, q] != 0.0:
                    ii += [i * f + p, k * f + q]
                    jj += [k * f + q, i * f + p]
                    vv += [B[p, q], B[p, q]]
    A = sp.csc_matrix((vv, (ii, jj)), shape=(n * f, n * f))
    keep = np.ones(n * f, dtype=bool)
    keep[i0 * f] = False
    keep[i1 * f] = False
    idx = np.flatnonzero(keep)
    lu = spla.splu(A[idx][:, idx].tocsc(), permc_spec="COLAMD", diag_pivot_thresh=0.1)
    return float(np.sum(np.log(np.abs(lu.U.diagonal()))))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def tunnelling_splitting(a: float, c: float, wy: float, hbar: float, n_beads: int, tau_inst: float) -> "np.ndarray":
    mass = 1.0
    n_beads = int(n_beads)
    if n_beads < 2:
        raise ValueError("n_beads must be at least 2")
    if not np.isfinite(tau_inst) or tau_inst <= 0.0:
        raise ValueError("tau_inst must be positive and finite")
    eps = tau_inst / n_beads
    well = well_data(a, c, wy, hbar)
    X0 = initial_path(a, c, wy, well, n_beads, tau_inst)
    X = optimise_instanton(X0, eps, a, c, wy)
    # keep whichever of the two trajectories is closer to stationary; the refined one
    # always is, but a refinement that failed to move must not be used silently
    if np.linalg.norm(path_gradient(X, eps, a, c, wy)) > \
       np.linalg.norm(path_gradient(X0, eps, a, c, wy)):
        X = X0
    sd = surface_data(X, eps, a, c, wy)
    N_l, N_r, qdot = int(sd[1]), int(sd[2]), sd[3]
    tau_l, tau_r = N_l * eps, N_r * eps
    S_inst = path_action(X, eps, a, c, wy)
    ld_pin = pinned_log_determinant(X, eps, a, c, wy)
    ld_l = collapsed_log_determinant(N_l, eps, well[0, 2:4])
    ld_r = collapsed_log_determinant(N_r, eps, well[1, 2:4])
    log_phi = 0.5 * np.log(eps / mass) + 0.25 * (ld_pin - ld_l - ld_r)
    # the two well depths, read straight from the potential at the located minima
    depths = potential_and_derivatives(well[:, 0:2], a, c, wy)[:, 0]
    expo = -(S_inst - 0.5 * tau_l * depths[0] - 0.5 * tau_r * depths[1]) / hbar
    log_om = -log_phi + np.log(qdot) - 0.5 * np.log(2.0 * np.pi * hbar) + expo
    om = float(np.exp(log_om))
    d = 0.5 * (well[1, 5] - well[0, 5])
    return np.array([d, hbar * om, 2.0 * np.sqrt(d * d + (hbar * om) ** 2)])
SCICODE_GOLD_EOF
