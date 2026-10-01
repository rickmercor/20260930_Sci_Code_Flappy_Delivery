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
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def clc_shape_offsets(px: int, py: int, pz: int, spherical: bool) -> "np.ndarray":
    a = _int(px, "px", 1); b = _int(py, "py", 1); c = _int(pz, "pz", 1)
    if a < 1 or b < 1 or c < 1:
        raise ValueError("every half width must be at least one")
    g = _offsets(a, b, c, bool(spherical))
    if g.shape[0] == 0:
        raise ValueError("the finite crystal contains no cells besides the central one")
    return np.asarray(g, dtype=np.float64)

import numpy as np
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def clc_direct_potential(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool) -> "np.ndarray":
    L = _edges(cell_edges)
    p, q = _cell(basis_pos, basis_q)
    ref = _int(ref_index, "ref_index", 0)
    if ref >= p.shape[0]:
        raise ValueError("ref_index must be smaller than the number of basis ions")
    a = _int(px, "px", 1); b = _int(py, "py", 1); c = _int(pz, "pz", 1)
    off = _offsets(a, b, c, bool(spherical))
    if off.shape[0] == 0:
        raise ValueError("the finite crystal contains no cells besides the central one")
    return np.array([_direct(L, p, q, ref, off)], dtype=np.float64)

import numpy as np
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def clc_ewald_potential(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    L = _edges(cell_edges)
    p, q = _cell(basis_pos, basis_q)
    ref = _int(ref_index, "ref_index", 0)
    if ref >= p.shape[0]:
        raise ValueError("ref_index must be smaller than the number of basis ions")
    al = _pos(alpha, "alpha")
    nr = _int(n_real, "n_real", 1); nk = _int(n_rec, "n_rec", 1)
    if al <= 0.0:
        raise ValueError("alpha must be positive")
    return np.array([_ewald(L, p, q, ref, al, nr, nk)], dtype=np.float64)

import numpy as np
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def clc_boundary_term(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool, alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    if _pos(alpha, "alpha") <= 0.0:
        raise ValueError("alpha must be positive")
    d = clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, px, py, pz, spherical)
    e = clc_ewald_potential(cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec)
    v = float(d[0] - e[0])
    if not np.isfinite(v):
        raise ValueError("non-finite boundary term")
    return np.array([v], dtype=np.float64)

import numpy as np
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def clc_nearest_neighbour(cell_edges: "np.ndarray", basis_pos: "np.ndarray", ref_index: int, span: int) -> "np.ndarray":
    L = _edges(cell_edges)
    p = np.asarray(basis_pos, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    ref = _int(ref_index, "ref_index", 0)
    if ref >= p.shape[0]:
        raise ValueError("ref_index must be smaller than the number of basis ions")
    sp = _int(span, "span", 1)
    return np.array([_nearest(L, p, ref, sp)], dtype=np.float64)

import numpy as np
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def clc_shape_sequence(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, ratios_z: "np.ndarray", alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    pp = _int(p, "p", 1)
    rr = np.asarray(ratios_z, dtype=np.float64)
    if rr.ndim != 1 or rr.size == 0:
        raise ValueError("ratios_z must be a non-empty one dimensional array")
    if not np.all(np.isfinite(rr)) or np.any(rr <= 0.0):
        raise ValueError("every thickness ratio must be finite and positive")
    e = float(clc_ewald_potential(cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec)[0])
    L = _edges(cell_edges)
    positions, charges = _cell(basis_pos, basis_q)
    if isinstance(ref_index, bool) or not np.isscalar(ref_index):
        raise ValueError("ref_index must be an integer scalar")
    ref_value = float(ref_index)
    if not np.isfinite(ref_value) or ref_value != int(ref_value) or not 0 <= int(ref_value) < len(charges):
        raise ValueError("ref_index is outside the basis")
    displacement = (positions - positions[int(ref_value)]) * L
    volume = float(np.prod(L))
    out = np.empty((rr.size, 2), dtype=np.float64)
    for i, r in enumerate(rr):
        pz = max(1, int(np.floor(pp * float(r) + 0.5)))
        out[i, 0] = float(clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, pz, False)[0]) - e
        sides = np.array([L[0], L[1], float(r) * L[2]], dtype=np.float64)
        diagonal = float(np.linalg.norm(sides))
        a, b, c = sides
        angular = np.arctan(np.array([b*c/(a*diagonal), a*c/(b*diagonal), a*b/(c*diagonal)]))
        pair_boundary = -4.0 / volume * (displacement * displacement) @ angular
        out[i, 1] = float(charges @ pair_boundary)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite shape sequence")
    return out

import numpy as np


def clc_boundary_decomposition(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, fit_sizes: "np.ndarray", alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    if isinstance(p, bool) or isinstance(pz, bool) or not np.isscalar(p) or not np.isscalar(pz):
        raise ValueError("p and pz must be integer scalars")
    pp = float(p)
    zz = float(pz)
    if not np.isfinite(pp) or not np.isfinite(zz) or pp != int(pp) or zz != int(zz):
        raise ValueError("p and pz must be finite integers")
    pp, zz = int(pp), int(zz)
    if pp < 4 or zz < 1 or zz > pp:
        raise ValueError("require p at least four and 1 <= pz <= p")

    s = np.asarray(fit_sizes, dtype=np.float64)
    if s.ndim != 1 or s.size < 6 or not np.all(np.isfinite(s)):
        raise ValueError("fit_sizes must be a finite one-dimensional array with at least six values")
    if np.any(s != np.floor(s)) or np.any(s < 4) or np.any(np.diff(s) <= 0) or np.unique(s).size != s.size:
        raise ValueError("fit_sizes must be distinct increasing integers at least four")

    rho = float(zz) / float(pp)
    y = []
    for n in s.astype(int):
        nz = max(1, int(np.floor(float(n) * rho + 0.5)))
        y.append(float(clc_boundary_term(cell_edges, basis_pos, basis_q, ref_index,
                                                  n, n, nz, False, alpha, n_real, n_rec)[0]))
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack((np.ones(s.size), 1.0 / s, 1.0 / s**2, 1.0 / s**3))
    sw = np.sqrt(s)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    residual = y - X @ beta
    rmse = float(np.sqrt(np.mean(residual**2)))

    loo = []
    for k in range(s.size):
        keep = np.arange(s.size) != k
        b = np.linalg.lstsq(X[keep] * sw[keep, None], y[keep] * sw[keep], rcond=None)[0]
        loo.append(float(b[0]))
    jack = float(np.max(np.abs(np.asarray(loo) - beta[0])))

    target = float(clc_boundary_term(cell_edges, basis_pos, basis_q, ref_index,
                                              pp, pp, zz, False, alpha, n_real, n_rec)[0])
    cube = float(clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index,
                                               pp, pp, pp, False)[0])
    dnn = float(clc_nearest_neighbour(cell_edges, basis_pos, ref_index, span)[0])
    mad = cube * dnn
    out = np.array([beta[0], target - beta[0], target, rmse, jack, mad], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite boundary decomposition")
    return out

import numpy as np


def clc_audit(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    if isinstance(p, bool) or isinstance(pz, bool) or not np.isscalar(p) or not np.isscalar(pz):
        raise ValueError("p and pz must be integer scalars")
    pp, zz = float(p), float(pz)
    if not np.isfinite(pp) or not np.isfinite(zz) or pp != int(pp) or zz != int(zz):
        raise ValueError("p and pz must be finite integers")
    pp, zz = int(pp), int(zz)
    if pp < 6 or zz < 1 or zz > pp - 2:
        raise ValueError("require p at least six and 1 <= pz <= p-2")
    midz = (pp + zz + 1) // 2

    off = clc_shape_offsets(pp, pp, pp, False)
    if off.shape[0] != (2 * pp + 1) ** 3 - 1:
        raise ValueError("the cell-cubic summation region has the wrong number of cells")

    cube = float(clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, pp, False)[0])
    flat = float(clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, zz, False)[0])
    middle = float(clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, midz, False)[0])
    bulk = float(clc_ewald_potential(cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec)[0])
    dnn = float(clc_nearest_neighbour(cell_edges, basis_pos, ref_index, span)[0])
    sph = float(clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, pp, pp, pp, True)[0])

    zs = (zz, midz, pp)
    seq = clc_shape_sequence(cell_edges, basis_pos, basis_q, ref_index, pp,
                                     np.asarray(zs, dtype=np.float64) / float(pp), alpha, n_real, n_rec)
    fit_sizes = np.array([4, 8, 12, 16, 20, 24], dtype=np.float64)
    decs = [clc_boundary_decomposition(cell_edges, basis_pos, basis_q, ref_index,
                                                pp, z, fit_sizes, alpha, n_real, n_rec, span)
            for z in zs]
    direct_residuals = np.array([flat - bulk, middle - bulk, cube - bulk], dtype=np.float64)
    q = []
    for k, dec in enumerate(decs):
        B, correction, residual, rmse, jackknife = map(float, dec[:5])
        scale = max(1.0, abs(residual))
        if abs(float(seq[k, 0]) - residual) > 1e-9 * scale or abs(direct_residuals[k] - residual) > 1e-9 * scale:
            raise ValueError("a reconstructed residual disagrees with an independent path")
        if abs((B + correction) - residual) > 1e-9 * scale:
            raise ValueError("boundary plus correction does not reconstruct a residual")
        dc = abs(B) + abs(correction)
        di = jackknife + rmse
        if dc <= 0.0 or di <= 0.0:
            raise ValueError("an aspect score denominator is zero")
        q.append(abs(correction) / dc + rmse / di)

    mad = float(decs[0][5])
    if abs(mad - cube * dnn) > 1e-9 * max(1.0, abs(mad)):
        raise ValueError("the Madelung constant disagrees with the scaled lattice sum")
    Bf, Cf, Rf, Ef, Kf = map(float, decs[0][:5])
    Bm = float(decs[1][0])
    Bc = float(decs[2][0])
    den_curve = abs(Bf) + abs(Bm) + abs(Bc)
    if den_curve <= 0.0:
        raise ValueError("the curvature denominator is zero")
    curvature = abs(Bf - 2.0 * Bm + Bc) / den_curve
    q = np.asarray(q, dtype=np.float64)
    J = float(np.median(q) + np.std(q, ddof=0) + curvature)
    out = np.array([cube, flat, bulk, flat - cube, dnn, mad, sph, Bf, Cf, Rf, Ef, Kf,
                    Bm, Bc, q[0], q[1], q[2], curvature,
                    seq[0, 1], seq[1, 1], seq[2, 1], J], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite audit record")
    return out
SCICODE_GOLD_EOF
