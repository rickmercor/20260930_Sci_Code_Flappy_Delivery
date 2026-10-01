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


def signed_normal_distance(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    zc = np.asarray(zc, dtype=float)
    if not (xc.shape == yc.shape == zc.shape):
        raise ValueError("xc, yc and zc must have the same shape")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive")
    r = np.sqrt((xc - float(centre_x)) ** 2 + (yc - float(centre_y)) ** 2
                + (zc - float(centre_z)) ** 2)
    return radius - r

import numpy as np


def boundary_normal(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float) -> "np.ndarray":
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    zc = np.asarray(zc, dtype=float)
    if not (xc.shape == yc.shape == zc.shape):
        raise ValueError("xc, yc and zc must have the same shape")
    ux = xc - float(centre_x)
    uy = yc - float(centre_y)
    uz = zc - float(centre_z)
    r = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    if np.any(r == 0.0):
        raise ValueError("a cell centre coincides with the sphere centre, so the normal is undefined")
    return np.stack([ux / r, uy / r, uz / r], axis=0)

import numpy as np


def normal_cell_thickness(normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    n = np.asarray(normals, dtype=float)
    if n.shape[0] != 3:
        raise ValueError("normals must have shape (3,) + grid shape")
    dx = float(dx); dy = float(dy); dz = float(dz)
    if not (dx > 0.0 and dy > 0.0 and dz > 0.0):
        raise ValueError("dx, dy and dz must be strictly positive")
    q = (n[0] / dx) ** 2 + (n[1] / dy) ** 2 + (n[2] / dz) ** 2
    if np.any(q <= 0.0):
        raise ValueError("a normal vector is degenerate, so the thickness is undefined")
    return q ** (-0.5)

import numpy as np


def _shift(a, axis, step):
    """Translate by one cell along `axis`, repeating the edge value rather than wrapping."""
    out = np.roll(a, step, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = 0 if step > 0 else -1
    out[tuple(idx)] = a[tuple(idx)]
    return out


def hybrid_ghost_tags(psi: "np.ndarray", thickness: "np.ndarray") -> "np.ndarray":
    psi = np.asarray(psi, dtype=float)
    th = np.asarray(thickness, dtype=float)
    if psi.shape != th.shape:
        raise ValueError("psi and thickness must have the same shape")
    if np.any(th <= 0.0):
        raise ValueError("thickness must be strictly positive everywhere")
    solid = psi > 0.0
    half = th / 2.0
    # Eq 7, with the Sec 7.1 deeper threshold (1.5 Delta n = 3 * half) on the solid side so
    # that a second layer of ghost cells is retained, and the shallower half-thickness on
    # the fluid side.  The solid test is non-strict and the fluid test is strict.
    cand_solid = solid & (psi <= 3.0 * half)
    cand_fluid = (~solid) & (-psi < half)
    tags = np.where(solid, 1.0, 0.0)
    tags = np.where(cand_solid, 2.0, tags)
    tags = np.where(cand_fluid, -2.0, tags)
    # Sec 3.1 reclassification, FLUID side only.  Sec 7.1 states that the qualifying solid
    # cells are included in the ghost-cell set, and the solid-side clause exists to drop
    # solid ghosts the fluid stencil does not need; a configuration that asks for a second
    # layer needs them by construction.  The source does not settle the interaction, so the
    # prompt states which reading this configuration uses.  Neighbours are judged against
    # the ORIGINAL solid/fluid split, never against a tag this same call has written.
    has_solid = np.zeros_like(psi, dtype=bool)
    for ax in range(psi.ndim):
        for k in (1, -1):
            has_solid |= _shift(solid, ax, k)
    tags = np.where((tags == -2.0) & (~has_solid), 0.0, tags)
    return tags

import numpy as np


def boundary_point(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    n = boundary_normal(xc, yc, zc, centre_x, centre_y, centre_z)
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be strictly positive")
    return np.stack([float(centre_x) + radius * n[0],
                     float(centre_y) + radius * n[1],
                     float(centre_z) + radius * n[2]], axis=0)

import numpy as np


def stencil_multipliers(psi: "np.ndarray", normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    ps = np.asarray(psi, dtype=float)
    n = np.asarray(normals, dtype=float)
    if n.shape[0] != 3 or n.shape[1:] != ps.shape:
        raise ValueError("normals must have shape (3,) + psi.shape")
    h = (float(dx), float(dy), float(dz))
    if not all(v > 0.0 for v in h):
        raise ValueError("dx, dy and dz must be strictly positive")
    # Eq 18, per direction, floored at one cell.
    return np.stack([np.maximum(1.0, np.ceil(2.0 * np.abs(n[a] * ps) / h[a])) for a in range(3)],
                    axis=0)

import numpy as np


def trilinear_value_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    c = [np.asarray(v, dtype=float) for v in (xc, yc, zc)]
    b = np.asarray(boundary, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if b.shape[0] != 3 or n.shape[0] != 3 or r.shape[0] != 3:
        raise ValueError("boundary, normals and mult must have leading dimension 3")
    h = (float(dx), float(dy), float(dz))
    if not all(v > 0.0 for v in h):
        raise ValueError("dx, dy and dz must be strictly positive")
    out = []
    for a in range(3):
        # the diagonally opposite stencil point: sign of the normal component, +1 when it
        # vanishes, times that direction's multiplier in whole cells
        far = c[a] + np.where(n[a] < 0.0, -1.0, 1.0) * r[a] * h[a]
        den = far - b[a]
        if np.any(den == 0.0):
            raise ValueError("degenerate stencil: the boundary point coincides with a stencil coordinate")
        out.append((c[a] - b[a]) / den)
    return np.stack(out, axis=0)

import numpy as np


def trilinear_derivative_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    c = [np.asarray(v, dtype=float) for v in (xc, yc, zc)]
    b = np.asarray(boundary, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if b.shape[0] != 3 or n.shape[0] != 3 or r.shape[0] != 3:
        raise ValueError("boundary, normals and mult must have leading dimension 3")
    h = (float(dx), float(dy), float(dz))
    if not all(v > 0.0 for v in h):
        raise ValueError("dx, dy and dz must be strictly positive")
    far = [c[a] + np.where(n[a] < 0.0, -1.0, 1.0) * r[a] * h[a] for a in range(3)]
    # X[1], Y[1], Z[1] are the ghost cell's own coordinates; X[2], Y[2], Z[2] the opposite
    # stencil point's.  Offsets are measured from the boundary point.
    Xo = {1: c[0] - b[0], 2: far[0] - b[0]}
    Yo = {1: c[1] - b[1], 2: far[1] - b[1]}
    Zo = {1: c[2] - b[2], 2: far[2] - b[2]}
    a_ = {(u, v): n[0] * Yo[u] * Zo[v] for u in (1, 2) for v in (1, 2)}
    b_ = {(u, v): n[1] * Xo[u] * Zo[v] for u in (1, 2) for v in (1, 2)}
    c_ = {(u, v): n[2] * Xo[u] * Yo[v] for u in (1, 2) for v in (1, 2)}
    den = a_[2, 2] + b_[2, 2] + c_[2, 2]
    if np.any(den == 0.0):
        raise ValueError("degenerate stencil: the derivative-type denominator vanishes")
    w1 = (a_[2, 2] + b_[1, 2] + c_[1, 2]) / den
    w2 = (a_[1, 2] + b_[2, 2] + c_[2, 1]) / den
    w3 = (a_[2, 1] + b_[2, 1] + c_[2, 2]) / den
    w4 = -(a_[1, 2] + b_[1, 2] + c_[1, 1]) / den
    w5 = -(a_[1, 1] + b_[2, 1] + c_[2, 1]) / den
    w6 = -(a_[2, 1] + b_[1, 1] + c_[1, 2]) / den
    w7 = (a_[1, 1] + b_[1, 1] + c_[1, 1]) / den
    ws = -((far[0] - c[0]) * (far[1] - c[1]) * (far[2] - c[2])) / den
    return np.stack([w1, w2, w3, w4, w5, w6, w7, ws], axis=0)

import numpy as np


def dependency_levels(tags: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    t = np.asarray(tags, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if n.shape[0] != 3 or r.shape[0] != 3:
        raise ValueError("normals and mult must have leading dimension 3")
    if n.shape[1:] != t.shape or r.shape[1:] != t.shape:
        raise ValueError("normals and mult must match the shape of tags")
    nz, ny, nx = t.shape
    ghost = (t == 2.0) | (t == -2.0)
    off = [(np.where(n[a] < 0.0, -1, 1) * r[a]).astype(int) for a in range(3)]
    lev = np.full(t.shape, -1.0)
    memo = {}

    def _neighbours(k, j, i):
        i2 = min(max(i + off[0][k, j, i], 0), nx - 1)
        j2 = min(max(j + off[1][k, j, i], 0), ny - 1)
        k2 = min(max(k + off[2][k, j, i], 0), nz - 1)
        points = [(k, j, i2), (k, j2, i), (k2, j, i),
                  (k, j2, i2), (k2, j2, i), (k2, j, i2), (k2, j2, i2)]
        directions = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
        return [q for q, axes in zip(points, directions)
                if all(n[a, k, j, i] != 0.0 for a in axes)]

    def _level_of(p, active):
        if p in memo:
            return memo[p]
        if p in active:
            raise ValueError("the ghost-cell dependencies contain a cycle")
        active = active | {p}
        deps = [q for q in _neighbours(*p) if ghost[q]]
        value = 0 if not deps else 1 + max(_level_of(q, active) for q in deps)
        memo[p] = value
        return value

    for p in (tuple(v) for v in np.argwhere(ghost)):
        lev[p] = float(_level_of(p, frozenset()))
    return lev

import numpy as np


def sweep_reconstruct(field: "np.ndarray", tags: "np.ndarray", levels: "np.ndarray", weights: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    f = np.array(field, dtype=float, copy=True)
    t = np.asarray(tags, dtype=float)
    lv = np.asarray(levels, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = np.asarray(normals, dtype=float)
    r = np.asarray(mult, dtype=float)
    if w.shape[0] != 8:
        raise ValueError("weights must have shape (8,) + grid shape: seven neighbours and the boundary term")
    if not (f.shape == t.shape == lv.shape == w.shape[1:] == n.shape[1:] == r.shape[1:]):
        raise ValueError("every argument must agree on the grid shape")
    nz, ny, nx = f.shape
    ghost = (t == 2.0) | (t == -2.0)
    off = [(np.where(n[a] < 0.0, -1, 1) * r[a]).astype(int) for a in range(3)]
    nlev = int(lv[ghost].max()) + 1 if ghost.any() else 0
    for L in range(nlev):
        upd = {}
        for k, j, i in np.argwhere(ghost & (lv == float(L))):
            i2 = min(max(i + off[0][k, j, i], 0), nx - 1)
            j2 = min(max(j + off[1][k, j, i], 0), ny - 1)
            k2 = min(max(k + off[2][k, j, i], 0), nz - 1)
            p = [f[k, j, i2], f[k, j2, i], f[k2, j, i],
                 f[k, j2, i2], f[k2, j2, i], f[k2, j, i2], f[k2, j2, i2]]
            upd[(k, j, i)] = sum(w[s, k, j, i] * p[s] for s in range(7)) + w[7, k, j, i]
        for key, value in upd.items():
            f[key] = value
    return f

import numpy as np


def embedded_boundary_audit(refine: int) -> "np.ndarray":
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    # Evaluation configuration: GIVEN in the prompt.
    nx = ny = nz = 30 * refine + 1
    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1.2, 1.2, ny)
    z = np.linspace(-0.9, 0.9, nz)
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dz = float(z[1] - z[0])
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    cx, cy, cz, R = 0.07, -0.11, 0.05, 0.61

    psi = signed_normal_distance(X, Y, Z, cx, cy, cz, R)
    nrm = boundary_normal(X, Y, Z, cx, cy, cz)
    th = normal_cell_thickness(nrm, dx, dy, dz)
    tags = hybrid_ghost_tags(psi, th)
    bpt = boundary_point(X, Y, Z, cx, cy, cz, R)
    mult = stencil_multipliers(psi, nrm, dx, dy, dz)
    wv = trilinear_value_weights(X, Y, Z, bpt, nrm, mult, dx, dy, dz)
    wdv = trilinear_derivative_weights(X, Y, Z, bpt, nrm, mult, dx, dy, dz)
    lev = dependency_levels(tags, nrm, mult)

    # Manufactured field, exact everywhere.  The boundary data is MIXED: the NORMAL
    # DERIVATIVE is prescribed on the part of the sphere whose outward normal has a
    # non-negative x component, and the field value is prescribed on the rest.  The arcs
    # are that way round so the largest error lands on a value-type cell, which is what
    # makes the reported answer depend on the mixed data at all.
    exact = lambda a, b, c: np.sin(1.3 * a) * np.cos(0.9 * b) * np.sin(0.7 * c) + 0.45 * a * b * c
    d_dx = lambda a, b, c: 1.3 * np.cos(1.3 * a) * np.cos(0.9 * b) * np.sin(0.7 * c) + 0.45 * b * c
    d_dy = lambda a, b, c: -0.9 * np.sin(1.3 * a) * np.sin(0.9 * b) * np.sin(0.7 * c) + 0.45 * a * c
    d_dz = lambda a, b, c: 0.7 * np.sin(1.3 * a) * np.cos(0.9 * b) * np.cos(0.7 * c) + 0.45 * a * b

    xs, ys, zs = bpt[0], bpt[1], bpt[2]
    phi_s = exact(xs, ys, zs)
    dphi_dn = d_dx(xs, ys, zs) * nrm[0] + d_dy(xs, ys, zs) * nrm[1] + d_dz(xs, ys, zs) * nrm[2]
    ghost = (tags == 2.0) | (tags == -2.0)
    deriv = nrm[0] >= 0.0

    # The two boundary-condition types do not share a stencil.  The value-type form carries
    # the three single weights, minus their three pairwise products, plus their triple
    # product; the derivative-type form carries seven independent weights built from the
    # normal components and the coordinate offsets.  One weight set drives the same sweep.
    v1, v2, v3 = wv[0], wv[1], wv[2]
    W = np.stack([
        np.where(deriv, wdv[0], v1),
        np.where(deriv, wdv[1], v2),
        np.where(deriv, wdv[2], v3),
        np.where(deriv, wdv[3], -v1 * v2),
        np.where(deriv, wdv[4], -v2 * v3),
        np.where(deriv, wdv[5], -v1 * v3),
        np.where(deriv, wdv[6], v1 * v2 * v3),
        np.where(deriv, wdv[7] * dphi_dn, (1.0 - v1) * (1.0 - v2) * (1.0 - v3) * phi_s),
    ], axis=0)

    seeded = np.where(ghost, 0.0, exact(X, Y, Z))
    rec = sweep_reconstruct(seeded, tags, lev, W, nrm, mult)
    err = np.abs(rec - exact(X, Y, Z))[ghost]

    n_ghost = float(ghost.sum())
    n_levels = float(lev[ghost].max() + 1) if ghost.any() else 0.0
    n_deriv = float((ghost & deriv).sum())
    n_wide = float((ghost & (mult > 1.0).any(axis=0)).sum())
    n_fluid = float((tags == -2.0).sum())
    return np.array([float(err.max()), float(err.mean()), n_ghost, n_levels,
                     n_deriv, float(th[ghost].min()), n_wide, n_fluid])
SCICODE_GOLD_EOF
