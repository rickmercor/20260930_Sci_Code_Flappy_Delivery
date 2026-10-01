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


def bernoulli(t: "np.ndarray") -> "np.ndarray":
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = np.abs(t) < 1e-8
    out[small] = 1.0 - 0.5 * t[small]
    tb = t[~small]
    res = np.empty_like(tb)
    pos = tb > 0
    # t > 0: t e^{-t} / (1 - e^{-t}) cannot overflow; t < 0: t / expm1(t)
    res[pos] = tb[pos] * np.exp(-tb[pos]) / (-np.expm1(-tb[pos]))
    res[~pos] = tb[~pos] / np.expm1(tb[~pos])
    out[~small] = res
    return out

import numpy as np


def contact_densities(N: "np.ndarray", V_applied: float = 0.0) -> "np.ndarray":
    VT = 0.025852
    nie = 1.087386e10
    N = np.asarray(N, dtype=float)
    maj = 0.5 * (np.abs(N) + np.sqrt(N * N + 4.0 * nie * nie))
    minor = nie * nie / maj
    n = np.where(N >= 0.0, maj, minor)
    p = np.where(N >= 0.0, minor, maj)
    psi = V_applied + VT * np.log(n / nie)
    return np.stack([psi, n, p])

import numpy as np


def build_mesh(M: int) -> tuple:
    M = int(M)
    if M < 2 or M % 2:
        raise ValueError("M must be an even integer >= 2")
    h = 1.0 / M
    j, i = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing="ij")
    shift = (j % 2 == 1) & (i > 0) & (i < M)
    x = i * h + 0.4 * h * shift
    y = j * h
    points = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    tris = []
    for jj in range(M):
        for ii in range(M):
            a = jj * (M + 1) + ii
            b, c, d = a + 1, a + M + 2, a + M + 1
            tris.append((a, b, c))
            tris.append((a, c, d))
    return points, np.array(tris, dtype=int)

import numpy as np


def _cross2d(u: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
    """Scalar cross product of stacks of 2D vectors."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def ddfv_geometry(points: "np.ndarray", triangles: "np.ndarray") -> dict:
    P = np.asarray(points, dtype=float)
    T = np.asarray(triangles, dtype=int)
    nT = len(T)

    tri_area = 0.5 * np.abs(_cross2d(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))
    tri_cent = P[T].mean(axis=1)
    edge_tris = {}
    for t, (a, b, c) in enumerate(T):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault((min(u, v), max(u, v)), []).append(t)
    keys = sorted(edge_tris)
    bnd = [k for k in keys if len(edge_tris[k]) == 1]
    bnd_index = {k: nT + m for m, k in enumerate(bnd)}
    bnd_edges = np.array(bnd, dtype=int).reshape(-1, 2)
    bnd_mid = P[bnd_edges].mean(axis=1)
    primal_xy = np.vstack([tri_cent, bnd_mid])
    dia = np.empty((len(keys), 4), dtype=int)
    for m, k in enumerate(keys):
        ts = edge_tris[k]
        dia[m] = (ts[0], ts[1] if len(ts) == 2 else bnd_index[k], k[0], k[1])
    xK, xL = primal_xy[dia[:, 0]], primal_xy[dia[:, 1]]
    xA, xB = P[dia[:, 2]], P[dia[:, 3]]
    s, ss = xB - xA, xL - xK
    ls, lss = np.linalg.norm(s, axis=1), np.linalg.norm(ss, axis=1)
    nKL = np.column_stack([s[:, 1], -s[:, 0]]) / ls[:, None]
    nKL *= np.sign(np.sum(nKL * ss, axis=1))[:, None]
    nAB = np.column_stack([ss[:, 1], -ss[:, 0]]) / lss[:, None]
    nAB *= np.sign(np.sum(nAB * s, axis=1))[:, None]
    c = np.sum(nKL * nAB, axis=1)
    dA = 0.5 * np.abs(_cross2d(ss, s))
    dual_area = np.zeros(len(P))
    np.add.at(dual_area, dia[:, 2], 0.5 * np.abs(_cross2d(xL - xK, xA - xK)))
    np.add.at(dual_area, dia[:, 3], 0.5 * np.abs(_cross2d(xL - xK, xB - xK)))
    return {"tri_area": tri_area, "tri_centroid": tri_cent,
            "bnd_edges": bnd_edges, "bnd_mid": bnd_mid,
            "dual_area": dual_area, "diamonds": dia,
            "alpha": ls ** 2 / (2 * dA), "beta": ls * lss * c / (2 * dA),
            "gamma": lss ** 2 / (2 * dA), "diamond_area": dA}

import numpy as np


def ddfv_ha_fluxes(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                           lam2: float, Dn: float, Dp: float) -> "np.ndarray":
    dia = geom["diamonds"]
    NP = len(geom["tri_area"]) + len(geom["bnd_edges"])
    K, L, A, B = dia[:, 0], dia[:, 1], NP + dia[:, 2], NP + dia[:, 3]
    al, be, ga = geom["alpha"], geom["beta"], geom["gamma"]
    u, n, p = (np.asarray(v, dtype=float) for v in (u, n, p))
    a, b = u[K] - u[L], u[A] - u[B]
    Ba, Bma = bernoulli(a), bernoulli(-a)
    Bb, Bmb = bernoulli(b), bernoulli(-b)
    sK = Ba * n[K] - Bma * n[L]
    sA = Bb * n[A] - Bmb * n[B]
    tK = Bma * p[K] - Ba * p[L]
    tA = Bmb * p[A] - Bb * p[B]
    return np.stack([lam2 * (al * a + be * b), lam2 * (be * a + ga * b),
                     Dn * (al * sK + be * sA), Dn * (be * sK + ga * sA),
                     Dp * (al * tK + be * tA), Dp * (be * tK + ga * tA)])

import numpy as np
import scipy.sparse as sp


def _bernoulli_derivative(t: "np.ndarray") -> "np.ndarray":
    """Derivative B'(t) = B(t)(1 - B(-t))/t, with the removable limit -1/2 at t = 0."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = np.abs(t) < 1e-5
    out[small] = -0.5 + t[small] / 6.0
    tb = t[~small]
    out[~small] = bernoulli(tb) * (1.0 - bernoulli(-tb)) / tb
    return out


def assemble_system(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                            Nd: "np.ndarray", dirichlet: "np.ndarray",
                            lam2: float, Dn: float, Dp: float) -> tuple:
    u, n, p, Nd = (np.asarray(v, dtype=float) for v in (u, n, p, Nd))
    dia = geom["diamonds"]
    nT, nE = len(geom["tri_area"]), len(geom["bnd_edges"])
    NP = nT + nE
    Nn = len(u)
    area = np.concatenate([geom["tri_area"], np.zeros(nE), geom["dual_area"]])
    K, L, A, B = dia[:, 0], dia[:, 1], NP + dia[:, 2], NP + dia[:, 3]
    al, be, ga = geom["alpha"], geom["beta"], geom["gamma"]
    fl = ddfv_ha_fluxes(geom, u, n, p, lam2, Dn, Dp)
    R = np.zeros(3 * Nn)
    for k in range(3):
        fp, fd = fl[2 * k], fl[2 * k + 1]
        np.add.at(R, 3 * K + k, fp)
        np.add.at(R, 3 * L + k, -fp)
        np.add.at(R, 3 * A + k, fd)
        np.add.at(R, 3 * B + k, -fd)
    R[0::3] -= area * (p - n + Nd)

    a, b = u[K] - u[L], u[A] - u[B]
    Ba, Bma, Bb, Bmb = (bernoulli(x) for x in (a, -a, b, -b))
    dBa, dBma, dBb, dBmb = (_bernoulli_derivative(x) for x in (a, -a, b, -b))
    one = np.ones_like(a)
    sa = dBa * n[K] + dBma * n[L]
    sb = dBb * n[A] + dBmb * n[B]
    ta = -dBma * p[K] - dBa * p[L]
    tb = -dBmb * p[A] - dBb * p[B]
    d_prim = {0: [(K, 0, one), (L, 0, -one)],
              1: [(K, 0, sa), (L, 0, -sa), (K, 1, Ba), (L, 1, -Bma)],
              2: [(K, 0, ta), (L, 0, -ta), (K, 2, Bma), (L, 2, -Ba)]}
    d_dual = {0: [(A, 0, one), (B, 0, -one)],
              1: [(A, 0, sb), (B, 0, -sb), (A, 1, Bb), (B, 1, -Bmb)],
              2: [(A, 0, tb), (B, 0, -tb), (A, 2, Bmb), (B, 2, -Bb)]}
    coef = {0: lam2, 1: Dn, 2: Dp}
    rows, cols, vals = [], [], []
    for k in range(3):
        for plus, minus, cp, cd in ((K, L, al, be), (A, B, be, ga)):
            for parts, c in ((d_prim[k], cp), (d_dual[k], cd)):
                for node, var, v in parts:
                    w = coef[k] * c * v
                    rows += [3 * plus + k, 3 * minus + k]
                    cols += [3 * node + var, 3 * node + var]
                    vals += [w, -w]
    node = np.arange(Nn)
    rows += [3 * node, 3 * node]
    cols += [3 * node + 1, 3 * node + 2]
    vals += [area, -area]
    J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * Nn, 3 * Nn)).tocsr()
    fixed = np.repeat(np.asarray(dirichlet, dtype=bool), 3)
    R[fixed] = 0.0
    J = (sp.diags((~fixed).astype(float)) @ J + sp.diags(fixed.astype(float))).tocsr()
    return R, J

import numpy as np
import scipy.sparse.linalg as spla


def _newton_solve(u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                  ub: "np.ndarray", nb: "np.ndarray", pb: "np.ndarray", args: tuple):
    """Newton iteration at one applied voltage; returns (u, n, p) or None if it fails."""
    geom, Nd, dirichlet, free, lam2, Dn, Dp = args
    u, n, p = u.copy(), n.copy(), p.copy()
    u[dirichlet], n[dirichlet], p[dirichlet] = ub[dirichlet], nb[dirichlet], pb[dirichlet]
    for _ in range(60):
        R, J = assemble_system(geom, u, n, p, Nd, dirichlet, lam2, Dn, Dp)
        d = spla.spsolve(J.tocsc(), -R)
        if not np.all(np.isfinite(d)):
            return None
        du, dn, dp = d[0::3], d[1::3], d[2::3]
        u = u + du
        n = np.maximum(n + dn, 1e-36)
        p = np.maximum(p + dp, 1e-36)
        # re-impose contact values exactly: roundoff in the linear solve must not move them
        u[dirichlet], n[dirichlet], p[dirichlet] = ub[dirichlet], nb[dirichlet], pb[dirichlet]
        if np.max(np.abs(du[free])) < 1e-10 and max(np.max(np.abs(dn[free]) / n[free]),
                                                     np.max(np.abs(dp[free]) / p[free])) < 1e-6:
            return u, n, p
    return None


def solve_drift_diffusion(N0: float, Va: float, M: int) -> "np.ndarray":
    q, eps, VT = 1.602192e-19, 1.035941e-12, 0.025852
    Nstar = 1e16
    lam2 = eps * VT / (q * Nstar * 1e-8)
    Dn, Dp = VT * 1417.0, VT * 470.5
    pts, tri = build_mesh(M)
    geom = ddfv_geometry(pts, tri)
    nT = len(tri)
    xy = np.vstack([geom["tri_centroid"], geom["bnd_mid"], pts])
    y = xy[:, 1]
    Nphys = np.where(np.abs(y - 0.5) < 1e-12, 0.0, np.where(y < 0.5, N0, -N0))
    ey = pts[geom["bnd_edges"]][:, :, 1]
    on_e = np.all(np.abs(ey) < 1e-12, axis=1) | np.all(np.abs(ey - 1.0) < 1e-12, axis=1)
    on_v = (np.abs(pts[:, 1]) < 1e-12) | (np.abs(pts[:, 1] - 1.0) < 1e-12)
    dirichlet = np.concatenate([np.zeros(nT, dtype=bool), on_e, on_v])
    top = dirichlet & (y > 0.5)
    eq = contact_densities(Nphys, 0.0)
    Nd = Nphys / Nstar

    free = ~dirichlet
    nb, pb = eq[1] / Nstar, eq[2] / Nstar
    args = (geom, Nd, dirichlet, free, lam2, Dn, Dp)

    sol = _newton_solve(eq[0] / VT, nb, pb, eq[0] / VT, nb, pb, args)
    if sol is None:
        raise RuntimeError("Newton failed at 0 V")
    V, dV = 0.0, 0.05
    while V < Va - 1e-15:
        Vn = min(V + dV, Va)
        trial = _newton_solve(*sol, eq[0] / VT + (Vn / VT) * top, nb, pb, args)
        if trial is None:
            dV *= 0.5
            if dV < 1e-6:
                raise RuntimeError("voltage ramp failed")
            continue
        sol, V, dV = trial, Vn, 2.0 * dV
    u, n, p = sol
    return np.stack([u * VT, n * Nstar, p * Nstar])

import numpy as np


def terminal_current(psi: "np.ndarray", n: "np.ndarray", p: "np.ndarray", M: int) -> float:
    q, eps, VT, Nstar = 1.602192e-19, 1.035941e-12, 0.025852, 1e16
    lam2 = eps * VT / (q * Nstar * 1e-8)
    pts, tri = build_mesh(M)
    geom = ddfv_geometry(pts, tri)
    fl = ddfv_ha_fluxes(geom, np.asarray(psi) / VT, np.asarray(n) / Nstar,
                                np.asarray(p) / Nstar, lam2, VT * 1417.0, VT * 470.5)
    dia = geom["diamonds"]
    nT = len(tri)
    top = np.zeros(len(dia), dtype=bool)
    b = np.where(dia[:, 1] >= nT)[0]
    ey = pts[geom["bnd_edges"][dia[b, 1] - nT]][:, :, 1]
    top[b] = np.all(np.abs(ey - 1.0) < 1e-12, axis=1)
    return float(q * Nstar * (fl[2][top].sum() - fl[4][top].sum()) / 1e-4)

import numpy as np


def _row_means(field: "np.ndarray", M: int) -> "np.ndarray":
    """Average a vertex field over each horizontal vertex row of the (M+1) x (M+1) grid."""
    return np.asarray(field, dtype=float).reshape(M + 1, M + 1).mean(axis=1)


def junction_electrostatics(psi_v: "np.ndarray", n_v: "np.ndarray", p_v: "np.ndarray",
                                    N0: float, M: int) -> "np.ndarray":
    q, eps = 1.602192e-19, 1.035941e-12
    M = int(M)
    h_um = 1.0 / M
    h_cm = h_um * 1e-4

    ps, nb, pb = (_row_means(f, M) for f in (psi_v, n_v, p_v))
    y = np.arange(M + 1) * h_um
    E = np.max(np.abs(np.diff(ps))) / h_cm + q * N0 * h_cm / (2.0 * eps)
    half = 0.5 * N0
    j = int(np.argmax(nb < half))
    yn = y[j - 1] + (half - nb[j - 1]) / (nb[j] - nb[j - 1]) * h_um
    k = M // 2 + int(np.argmax(pb[M // 2:] > half))
    yp = y[k - 1] + (half - pb[k - 1]) / (pb[k] - pb[k - 1]) * h_um
    return np.array([yp - yn, E])

import numpy as np


def textbook_predictions(N0: float, Va: float) -> "np.ndarray":
    q, eps, VT, nie = 1.602192e-19, 1.035941e-12, 0.025852, 1.087386e10
    Dn, Dp = VT * 1417.0, VT * 470.5
    Vbi = VT * np.log(N0 * N0 / nie ** 2)
    W = np.sqrt(4.0 * eps * Vbi / (q * N0))
    E = q * N0 * W / (2.0 * eps)
    Wn = 0.5e-4 - 0.5 * np.sqrt(4.0 * eps * (Vbi - Va) / (q * N0))
    J = q * nie ** 2 / N0 * np.expm1(Va / VT) * (Dn + Dp) / Wn
    return np.array([Vbi, W * 1e4, E, J])

import numpy as np


def _doping_table(M: int, Va: float) -> "np.ndarray":
    """Simulated and textbook quantities for the three doping levels, one row each."""
    V = (M + 1) ** 2
    rows = []
    for N0 in (1e16, 3e16, 1e17):
        s0 = solve_drift_diffusion(N0, 0.0, M)
        W, E = junction_electrostatics(s0[0, -V:], s0[1, -V:], s0[2, -V:], N0, M)
        s1 = solve_drift_diffusion(N0, Va, M)
        J = terminal_current(s1[0], s1[1], s1[2], M)
        _, Wda, Eda, Jsd = textbook_predictions(N0, Va)
        rows.append([W, E, J, Wda, Eda, Jsd,
                     100.0 * (W / Wda - 1.0), 100.0 * (E / Eda - 1.0), 100.0 * (J / Jsd - 1.0)])
    return np.array(rows)


def simulated_forward_current(M: int = 64, Va: float = 0.5) -> float:
    return float(_doping_table(M, Va)[0, 2])
SCICODE_GOLD_EOF
