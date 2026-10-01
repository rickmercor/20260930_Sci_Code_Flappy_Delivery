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


import math


def _finite(value, label):
    """Return an argument as a float once it is known to be finite."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    return out


def _contrast(value, label):
    """Return a contrast as a float once it is known to lie strictly between zero and one."""
    out = _finite(value, label)
    if not 0.0 < out < 1.0:
        raise ValueError("%s must lie strictly between zero and one" % label)
    return out


def resolve_contrast_scaling(
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
) -> dict:
    """Reference implementation."""
    lam = _finite(lam, "lam")
    mu = _finite(mu, "mu")
    rho = _finite(rho, "rho")
    delta = _contrast(delta, "delta")
    eps = _contrast(eps, "eps")
    if mu <= 0.0:
        raise ValueError("mu must be above zero")
    if rho <= 0.0:
        raise ValueError("rho must be above zero")
    if 3.0 * lam + 2.0 * mu <= 0.0:
        raise ValueError("three lam plus two mu must be above zero")
    return {
        "lam_in": lam / delta,
        "mu_in": mu / delta,
        "rho_in": rho / eps,
        "tau": math.sqrt(delta / eps),
        "c_s": math.sqrt(mu / rho),
        "c_p": math.sqrt((lam + 2.0 * mu) / rho),
    }

import numpy as np


def _flat(idx, n_side):
    """Flat index of a node triple taken modulo the periodic cell."""
    return ((idx[..., 0] % n_side) * n_side * n_side
            + (idx[..., 1] % n_side) * n_side
            + (idx[..., 2] % n_side))


def partition_unit_cell(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
) -> dict:
    """Reference implementation."""
    L = float(lattice_constant)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("lattice_constant must be finite and above zero")
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer)) or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    n_side = int(n_side)
    span = tuple(spans)
    if len(span) != 3:
        raise ValueError("spans must hold exactly three integers")
    for s in span:
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
            raise ValueError("every span must be an integer")
        if int(s) < 1 or int(s) > n_side - 2:
            raise ValueError("spans must lie in [1, n_side - 2]")
    span = np.array([int(s) for s in span], dtype=np.int64)

    lo = (n_side - span) // 2
    hi = lo + span
    grid = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(grid, grid, grid, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    elements = cells[~inside]

    rngs = [np.arange(lo[d], hi[d] + 1, dtype=np.int64) for d in range(3)]
    aa, bb, cc = np.meshgrid(rngs[0], rngs[1], rngs[2], indexing="ij")
    pts = np.stack([aa.ravel(), bb.ravel(), cc.ravel()], axis=1)
    on_face = np.any((pts == lo) | (pts == hi), axis=1)
    surface = np.unique(_flat(pts[on_face], n_side))

    offs = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int64)
    touched_idx = elements[:, None, :] + offs[None, :, :]
    touched = np.zeros(n_side ** 3, dtype=bool)
    touched[_flat(touched_idx, n_side).ravel()] = True
    every = np.arange(n_side ** 3, dtype=np.int64)
    free = every[touched & ~np.isin(every, surface)]

    h = L / n_side
    return {
        "h": h,
        "volume_D": float(np.prod(span)) * h ** 3,
        "elements": elements,
        "surface_nodes": surface,
        "free_nodes": free,
        "n_surface": int(surface.size),
        "n_free": int(free.size),
        "sides": span.astype(np.float64) * h,
    }

import numpy as np


def _corner_offsets():
    """The eight corner offsets, first axis varying slowest."""
    return np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int64)


def _voigt_isotropic(lam, mu):
    """Six by six isotropic constitutive matrix in the Voigt ordering used here."""
    C = np.zeros((6, 6), dtype=np.float64)
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _strain_displacement(signs, xi, eta, zeta, h):
    """Six by twenty-four strain-displacement matrix at one local point."""
    s = signs.astype(np.float64) * 2.0 - 1.0          # zero or one becomes minus or plus one
    dN = np.empty((8, 3), dtype=np.float64)
    dN[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
    dN[:, 1] = 0.125 * (1.0 + s[:, 0] * xi) * s[:, 1] * (1.0 + s[:, 2] * zeta)
    dN[:, 2] = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta) * s[:, 2]
    g = dN * (2.0 / h)
    B = np.zeros((6, 24), dtype=np.float64)
    B[0, 0::3] = g[:, 0]
    B[1, 1::3] = g[:, 1]
    B[2, 2::3] = g[:, 2]
    B[3, 1::3] = g[:, 2]
    B[3, 2::3] = g[:, 1]
    B[4, 0::3] = g[:, 2]
    B[4, 2::3] = g[:, 0]
    B[5, 0::3] = g[:, 1]
    B[5, 1::3] = g[:, 0]
    return B


def hex_element_stiffness(
    lam: float,
    mu: float,
    h: float,
) -> dict:
    """Reference implementation."""
    lam = float(lam)
    mu = float(mu)
    h = float(h)
    if not np.isfinite(lam):
        raise ValueError("lam must be finite")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and above zero")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and above zero")

    signs = _corner_offsets()
    C = _voigt_isotropic(lam, mu)
    g = 1.0 / np.sqrt(3.0)
    det = (h / 2.0) ** 3
    K = np.zeros((24, 24), dtype=np.float64)
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _strain_displacement(signs, xi, eta, zeta, h)
                K += B.T @ C @ B * det
    K = 0.5 * (K + K.T)
    return {"element_stiffness": K, "corner_signs": signs, "volume": h ** 3}

import numpy as np
import scipy.sparse as sp


def _checked_alpha(alpha):
    """Return the quasi-momentum as a float array once it is known to be admissible."""
    a = np.asarray(alpha, dtype=np.float64).ravel()
    if a.size != 3:
        raise ValueError("alpha must hold exactly three components")
    if not np.all(np.isfinite(a)):
        raise ValueError("every component of alpha must be finite")
    if not np.any(a != 0.0):
        raise ValueError("alpha must not be the zero vector")
    return a


def assemble_bloch_exterior(
    elements: np.ndarray,
    surface_nodes: np.ndarray,
    free_nodes: np.ndarray,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    n_side: int,
    alpha: tuple,
) -> dict:
    """Reference implementation."""
    el = np.asarray(elements)
    if el.ndim != 2 or el.shape[1] != 3 or not np.issubdtype(el.dtype, np.integer):
        raise ValueError("elements must be an integer array of three columns")
    Ke = np.asarray(element_stiffness, dtype=np.float64)
    if Ke.shape != (24, 24):
        raise ValueError("element_stiffness must be twenty-four by twenty-four")
    signs = np.asarray(corner_signs)
    if signs.shape != (8, 3):
        raise ValueError("corner_signs must be eight by three")
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer)) or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    n_side = int(n_side)
    a = _checked_alpha(alpha)

    offs = signs.astype(np.int64)
    reach = el[:, None, :] + offs[None, :, :]                 # (ne, 8, 3)
    wrap = reach // n_side
    node = ((reach[..., 0] % n_side) * n_side * n_side
            + (reach[..., 1] % n_side) * n_side
            + (reach[..., 2] % n_side))
    phase = np.exp(1j * (wrap @ a))                           # (ne, 8)

    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(el), 24)
    ph24 = np.repeat(phase, 3, axis=1)
    factor = np.conj(ph24)[:, :, None] * ph24[:, None, :]
    vals = (Ke[None, :, :] * factor).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel()
    cols = np.tile(dof, (1, 24)).ravel()
    ndof = 3 * n_side ** 3
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    con = np.asarray(surface_nodes)
    fre = np.asarray(free_nodes)
    for name, arr in (("surface_nodes", con), ("free_nodes", fre)):
        if arr.ndim != 1 or arr.size == 0 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("%s must be a non-empty one-dimensional integer array" % name)
        if arr.min() < 0 or arr.max() >= n_side ** 3:
            raise ValueError("%s holds an index outside the cell" % name)
    con = con.astype(np.int64)
    fre = fre.astype(np.int64)
    cdof = np.sort(np.concatenate([3 * con + d for d in range(3)]))
    fdof = np.sort(np.concatenate([3 * fre + d for d in range(3)]))
    return {
        "K_ff": K[fdof][:, fdof].tocsr(),
        "K_fc": K[fdof][:, cdof].tocsr(),
        "K_cc": K[cdof][:, cdof].tocsr(),
        "free_dofs": fdof,
        "surface_dofs": cdof,
    }

import numpy as np
import scipy.sparse as sp


def _inner(a, b):
    """Hermitian inner product by plain summation, so the value does not depend on BLAS threading."""
    return np.sum(np.conj(a) * b)


def _pcg(A, B, rel_tol, max_iter=200000):
    """Jacobi-preconditioned conjugate gradient for a Hermitian positive definite A, column by column."""
    tol = rel_tol
    diag = A.diagonal().real
    if not np.all(np.isfinite(diag)):
        raise ValueError("K_ff has a diagonal entry that is not finite")
    if np.any(diag <= 0.0):
        raise ValueError("K_ff has a non-positive diagonal entry")
    prec = 1.0 / diag
    out = np.zeros_like(B)
    counts = []
    for col in range(B.shape[1]):
        rhs = B[:, col]
        x = np.zeros_like(rhs)
        r = rhs - A @ x
        z = prec * r
        p = z.copy()
        rz = _inner(r, z)
        target = rel_tol * np.sqrt(_inner(rhs, rhs).real)
        steps = 0
        while steps < max_iter:
            if np.sqrt(_inner(r, r).real) <= target:
                break
            Ap = A @ p
            curvature = _inner(p, Ap)
            if curvature == 0.0 or rz == 0.0:
                break
            step = rz / curvature
            x = x + step * p
            r = r - step * Ap
            z = prec * r
            rz_next = _inner(r, z)
            p = z + (rz_next / rz) * p
            rz = rz_next
            steps += 1
        true_residual = np.sqrt(_inner(rhs - A @ x, rhs - A @ x).real)
        if true_residual > 10.0 * target:
            raise ValueError(
                "conjugate gradient did not reach rel_tol: residual %.3e against %.3e" % (true_residual / max(np.sqrt(_inner(rhs, rhs).real), 1e-300), tol))
        out[:, col] = x
        counts.append(steps)
    return out, counts


def _rigid_boundary_fields(cdof, n_side, h, reference_point):
    """The six rigid motions of the resonator sampled at the surface degrees of freedom.

    Columns 0 to 2 are the unit Cartesian translations; column 3 + a is the rotation about the
    axis a through the centroid, holding the relevant component of e_a cross r.
    """
    node = cdof // 3
    comp = cdof % 3
    i1 = node // (n_side * n_side)
    i2 = (node // n_side) % n_side
    i3 = node % n_side
    r = (np.stack([i1, i2, i3], 1).astype(np.float64) * h
         - np.asarray(reference_point, dtype=np.float64)[None, :])
    E = np.zeros((cdof.size, 6), dtype=np.complex128)
    for c in range(3):
        E[comp == c, c] = 1.0
    for a in range(3):
        for c in range(3):
            if a == c:
                continue
            b = 3 - a - c
            sign = 1.0 if (c, a, b) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
            sel = comp == c
            E[sel, 3 + a] = sign * r[sel, b]
    return E


def capacity_schur_matrix(
    K_ff: "scipy.sparse.spmatrix",
    K_fc: "scipy.sparse.spmatrix",
    K_cc: "scipy.sparse.spmatrix",
    surface_dofs: np.ndarray,
    n_side: int,
    h: float,
    reference_point: np.ndarray,
    rel_tol: float,
) -> dict:
    """Reference implementation."""
    tol = float(rel_tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("rel_tol must be finite and above zero")
    A = sp.csr_matrix(K_ff)
    Bfc = sp.csr_matrix(K_fc)
    Ccc = sp.csr_matrix(K_cc)
    if A.shape[0] != A.shape[1]:
        raise ValueError("K_ff must be square")
    if Bfc.shape[0] != A.shape[0]:
        raise ValueError("K_fc must have one row per free degree of freedom")
    if Ccc.shape[0] != Ccc.shape[1] or Ccc.shape[0] != Bfc.shape[1]:
        raise ValueError("K_cc must be square and match K_fc")
    cdof = np.asarray(surface_dofs)
    if cdof.ndim != 1 or not np.issubdtype(cdof.dtype, np.integer):
        raise ValueError("surface_dofs must be a one-dimensional integer array")
    cdof = cdof.astype(np.int64)
    if cdof.size != Ccc.shape[0]:
        raise ValueError("surface_dofs must match the surface block")

    if int(n_side) != n_side or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    hh = float(h)
    if not np.isfinite(hh) or hh <= 0.0:
        raise ValueError("h must be finite and above zero")
    ctr = np.asarray(reference_point, dtype=np.float64).ravel()
    if ctr.size != 3 or not np.all(np.isfinite(ctr)):
        raise ValueError("reference_point must hold exactly three finite values")

    E = _rigid_boundary_fields(cdof, int(n_side), hh, ctr)

    lifted, counts = _pcg(A, -(Bfc @ E), tol)
    raw = E.conj().T @ (Ccc @ E) + E.conj().T @ (Bfc.conj().T @ lifted)
    dg = np.sqrt(np.abs(np.diag(raw)))
    dg = np.where(dg > 0.0, dg, 1.0)
    defect = float((np.abs(raw - raw.conj().T) / np.outer(dg, dg)).max())
    if defect > 1.0e-6:
        raise ValueError(
            "condensed matrix not Hermitian: entrywise defect %.3e against 1.0e-06" % (defect,))
    Q = 0.5 * (raw + raw.conj().T)
    w = np.linalg.eigvalsh(Q)
    if not np.all(w > 0.0):
        raise ValueError("the capacity matrix must be positive definite")
    return {
        "capacity": Q,
        "eigenvalues": w,
        "hermitian_defect": defect,
        "iterations": counts,
    }

import numpy as np


def _pencil(Qh, Mmat):
    """Eigenvalues of the pencil Qh v = w Mmat v through the Cholesky factor of Mmat, ascending."""
    L = np.linalg.cholesky(Mmat)
    Li = np.linalg.inv(L)
    return np.sort(np.linalg.eigvalsh(Li @ Qh @ Li.conj().T))


def subwavelength_frequencies(
    capacity: np.ndarray,
    volume_D: float,
    inertia: np.ndarray,
    rho: float,
    eps: float,
) -> dict:
    """Reference implementation."""
    Q = np.asarray(capacity)
    if Q.ndim != 2 or Q.shape != (6, 6):
        raise ValueError("capacity must be a six by six array")
    if not np.all(np.isfinite(Q)):
        raise ValueError("capacity must be finite")
    dgq = np.sqrt(np.abs(np.diag(Q)))
    dgq = np.where(dgq > 0.0, dgq, 1.0)
    if float((np.abs(Q - Q.conj().T) / np.outer(dgq, dgq)).max()) > 1.0e-6:
        raise ValueError("capacity must be Hermitian to working accuracy")
    Mi = np.asarray(inertia)
    if Mi.ndim != 2 or Mi.shape != (6, 6) or not np.all(np.isfinite(Mi)):
        raise ValueError("inertia must be a finite six by six array")
    if np.abs(Mi - Mi.conj().T).max() > 1.0e-9 * max(float(np.abs(Mi).max()), 1e-300):
        raise ValueError("inertia must be Hermitian")
    Mh = 0.5 * (Mi + Mi.conj().T)
    if not np.all(np.linalg.eigvalsh(Mh) > 0.0):
        raise ValueError("inertia must be positive definite")
    vol = float(volume_D)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("volume_D must be finite and above zero")
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be finite and above zero")
    eps = float(eps)
    if not np.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between zero and one")

    mass = (rho / eps) * vol
    Qh = 0.5 * (Q + Q.conj().T)
    w = _pencil(Qh, Mh)
    if not np.all(w > 0.0):
        raise ValueError("the reduced pencil must be positive definite")
    wb = _pencil(Qh, Mh * eps) * eps
    first = np.sqrt(wb)
    second = np.sqrt(w)
    defect = float(np.abs(first - second).max() / (2.0 * np.pi))
    return {
        "angular": second,
        "hertz": second / (2.0 * np.pi),
        "mass": mass,
        "cross_check_defect": defect,
    }

import numpy as np


import math


def _positive(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def dilute_ball_reference(
    lam: float,
    mu: float,
    rho: float,
    eps: float,
    volume_D: float,
) -> dict:
    """Reference implementation."""
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("lam must be finite")
    mu = _positive(mu, "mu")
    rho = _positive(rho, "rho")
    vol = _positive(volume_D, "volume_D")
    eps = float(eps)
    if not math.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between zero and one")
    denom = 5.0 * mu + 2.0 * lam
    if denom <= 0.0:
        raise ValueError("five mu plus two lam must be above zero")
    if 2.0 * mu + lam <= 0.0:
        raise ValueError("two mu plus lam must be above zero")

    radius = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    beta = 12.0 * mu * math.pi * radius * (2.0 * mu + lam) / denom
    omega_min = math.sqrt(9.0 * mu * (2.0 * mu + lam) / (denom * rho * radius ** 2)) * math.sqrt(eps)
    omega_max = math.sqrt(15.0 * mu / (rho * radius ** 2)) * math.sqrt(eps)
    return {
        "radius": radius,
        "beta_ball": beta,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "hertz_min": omega_min / (2.0 * math.pi),
        "hertz_max": omega_max / (2.0 * math.pi),
        "width_ratio": omega_max / omega_min,
    }

import numpy as np
import scipy.sparse as sp


def _hex_consistent_mass(h, offs):
    """Element mass for unit density on a cube of edge h, two-point Gauss in each direction."""
    q = 1.0 / np.sqrt(3.0)
    s = offs.astype(np.float64) * 2.0 - 1.0
    Me = np.zeros((24, 24))
    for xi in (-q, q):
        for et in (-q, q):
            for ze in (-q, q):
                Nn = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * et) * (1.0 + s[:, 2] * ze)
                Nm = np.zeros((3, 24))
                for c in range(3):
                    Nm[c, c::3] = Nn
                Me += (Nm.T @ Nm) * (h ** 3 / 8.0)
    return Me


def _bloch_assemble(cells, scale, E24, offs, n_side, alpha, real):
    """Quasi-periodic assembly of one 24 by 24 element matrix over the listed elements."""
    a = np.asarray(alpha, dtype=np.float64)
    reach = cells[:, None, :] + offs[None, :, :]
    wrap = reach // n_side
    node = ((reach[..., 0] % n_side) * n_side * n_side
            + (reach[..., 1] % n_side) * n_side
            + (reach[..., 2] % n_side))
    phase = np.exp(1j * (wrap @ a))
    if real:
        phase = phase.real
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    ph24 = np.repeat(phase, 3, axis=1)
    factor = np.conj(ph24)[:, :, None] * ph24[:, None, :] * scale[:, None, None]
    vals = (E24[None, :, :] * factor).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel()
    cols = np.tile(dof, (1, 24)).ravel()
    n = 3 * n_side ** 3
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return 0.5 * (A + A.conj().T)


def assemble_full_bloch_pencil(
    n_side: int,
    spans: tuple,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    h: float,
    rho: float,
    delta: float,
    eps: float,
    alpha: tuple,
) -> dict:
    """Reference implementation."""
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer)) or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    n_side = int(n_side)
    try:
        span = tuple(spans)
    except TypeError:
        raise ValueError("spans must hold exactly three integers")
    if len(span) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in span):
        raise ValueError("spans must hold exactly three integers")
    span = np.array([int(v) for v in span], dtype=np.int64)
    if np.any(span < 1) or np.any(span > n_side - 2):
        raise ValueError("spans must lie in [1, n_side - 2]")
    Ke = np.asarray(element_stiffness, dtype=np.float64)
    if Ke.shape != (24, 24):
        raise ValueError("element_stiffness must be twenty-four by twenty-four")
    offs = np.asarray(corner_signs)
    if offs.shape != (8, 3):
        raise ValueError("corner_signs must be eight by three")
    offs = offs.astype(np.int64)
    for name, v in (("h", h), ("rho", rho), ("delta", delta), ("eps", eps)):
        v = float(v)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError("%s must be finite and above zero" % name)
    h = float(h); rho = float(rho); delta = float(delta); eps = float(eps)
    try:
        a = np.asarray(alpha, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise ValueError("alpha must hold exactly three finite values")
    if a.size != 3 or not np.all(np.isfinite(a)):
        raise ValueError("alpha must hold exactly three finite values")

    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    lo = (n_side - span) // 2
    hi = lo + span
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    real = bool(np.allclose(np.sin(a), 0.0, atol=1e-12))
    kscale = np.where(inside, 1.0 / delta, 1.0)
    mscale = np.where(inside, rho / eps, rho)
    K = _bloch_assemble(cells, kscale, Ke, offs, n_side, a, real)
    M = _bloch_assemble(cells, mscale, _hex_consistent_mass(h, offs), offs, n_side, a, real)
    n_in = int(inside.sum())
    total_mass = rho * (len(cells) - n_in) * h ** 3 + (rho / eps) * n_in * h ** 3
    return {"K_full": K, "M_full": M, "n_inside": n_in, "total_mass": float(total_mass)}

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def full_pencil_spectrum(
    K_full: "scipy.sparse.spmatrix",
    M_full: "scipy.sparse.spmatrix",
    count: int,
    rel_tol: float,
) -> dict:
    """Reference implementation."""
    if not sp.issparse(K_full) or not sp.issparse(M_full):
        raise ValueError("K_full and M_full must be sparse matrices")
    K = K_full.tocsc()
    M = M_full.tocsc()
    if K.shape[0] != K.shape[1] or M.shape[0] != M.shape[1]:
        raise ValueError("K_full and M_full must be square")
    if K.shape != M.shape:
        raise ValueError("K_full and M_full must have the same shape")
    n = K.shape[0]
    for name, A in (("K_full", K), ("M_full", M)):
        top = float(abs(A).max()) if A.nnz else 0.0
        if top == 0.0 or float(abs(A - A.conj().T).max()) > 1.0e-9 * top:
            raise ValueError("%s must be Hermitian" % name)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= int(count) < n:
        raise ValueError("count must lie in [1, order - 1]")
    count = int(count)
    tol = float(rel_tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("rel_tol must be finite and above zero")

    real = not (np.iscomplexobj(K.data) and np.abs(K.data.imag).max() > 0.0) \
        and not (np.iscomplexobj(M.data) and np.abs(M.data.imag).max() > 0.0)
    if real:
        K = K.real.tocsc(); M = M.real.tocsc()
    sigma = -1.0
    v0 = np.ones(n, dtype=np.float64 if real else np.complex128)
    # a few pairs beyond the ones wanted, then keep the lowest: at a quasi-momentum whose phases
    # are all real the cell is symmetric about the resonator centroid and the all-ones start
    # vector lies in one parity sector, so a block of exactly count pairs can converge without
    # ever reaching a mode of the opposite symmetry and silently report a higher eigenvalue
    k = min(count + 4, n - 1)
    w, V = spla.eigsh(K, k=k, M=M, sigma=sigma, which="LM", v0=v0, tol=1e-12)
    order = np.argsort(w.real)[:count]
    w = w.real[order]; V = V[:, order]
    if not np.all(w > 0.0):
        raise ValueError("every eigenvalue of the pencil must be positive")
    res = np.empty(count)
    for j in range(count):
        v = V[:, j]
        kv = K @ v
        r = kv - w[j] * (M @ v)
        res[j] = float(np.sqrt(np.sum(np.abs(r) ** 2).real) / np.sqrt(np.sum(np.abs(kv) ** 2).real))
    if np.any(res > tol):
        raise ValueError("an eigenpair did not reach rel_tol: worst residual %.3e" % float(res.max()))
    ang = np.sqrt(w)
    return {"hertz": ang / (2.0 * np.pi), "angular": ang, "residuals": res, "count": count}

import numpy as np


def report_finite_contrast_correction(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
    alphas: tuple,
    rel_tol: float,
) -> dict:
    """Reference implementation chaining every earlier stage."""
    grid = tuple(alphas)
    if len(grid) == 0:
        raise ValueError("alphas must hold at least one quasi-momentum")

    scaling = resolve_contrast_scaling(lam, mu, rho, delta, eps)  # noqa: F821
    part = partition_unit_cell(lattice_constant, n_side, spans)   # noqa: F821
    elem = hex_element_stiffness(lam, mu, part["h"])              # noqa: F821

    span = np.asarray(spans, dtype=np.float64)
    lo_corner = ((np.asarray([n_side] * 3, dtype=np.float64) - span) // 2)
    centre = (lo_corner + span / 2.0) * part["h"]
    mass = (rho / eps) * part["volume_D"]
    sd = np.asarray(part["sides"], dtype=np.float64)
    moments = np.array([mass * (sd[1] ** 2 + sd[2] ** 2) / 12.0,
                        mass * (sd[0] ** 2 + sd[2] ** 2) / 12.0,
                        mass * (sd[0] ** 2 + sd[1] ** 2) / 12.0])
    inertia = np.diag(np.concatenate([np.full(3, mass), moments]))

    rows = []
    for alpha in grid:
        blocks = assemble_bloch_exterior(                          # noqa: F821
            part["elements"], part["surface_nodes"], part["free_nodes"],
            elem["element_stiffness"], elem["corner_signs"], n_side, alpha,
        )
        cap = capacity_schur_matrix(                               # noqa: F821
            blocks["K_ff"], blocks["K_fc"], blocks["K_cc"], blocks["surface_dofs"],
            n_side, part["h"], centre, rel_tol,
        )
        freq = subwavelength_frequencies(                          # noqa: F821
            cap["capacity"], part["volume_D"], inertia, rho, eps,
        )
        rows.append(freq["hertz"])

    table = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(table)):
        raise ValueError("the scan produced a frequency that is not finite")

    lower = table[:, :3]
    upper = table[:, 3:]
    lower_flat = int(np.argmax(lower))
    lower_idx, lower_branch = divmod(lower_flat, 3)
    gap_lower = float(lower[lower_idx, lower_branch])
    upper_flat = int(np.argmin(upper))
    upper_idx, upper_branch = divmod(upper_flat, 3)
    gap_upper = float(upper[upper_idx, upper_branch])
    if not gap_upper > gap_lower:
        raise ValueError("the sampled branches overlap: no first gap")
    edge = gap_upper

    ball = dilute_ball_reference(lam, mu, rho, eps, part["volume_D"])  # noqa: F821
    lo, hi = ball["hertz_min"], ball["hertz_max"]

    pencil = assemble_full_bloch_pencil(                          # noqa: F821
        n_side, spans, elem["element_stiffness"], elem["corner_signs"], part["h"],
        rho, delta, eps, grid[upper_idx],
    )
    spec = full_pencil_spectrum(pencil["K_full"], pencil["M_full"], 8, 1.0e-8)  # noqa: F821
    full = np.asarray(spec["hertz"], dtype=np.float64)
    corrected = float(full[3])

    tops = table.max(axis=0)
    return {
        "correction_hertz": corrected - edge,
        "finite_contrast_edge_hertz": corrected,
        "full_pencil_hertz": full,
        "ordinary_branch_hertz": float(full[6]),
        "bandwidth_sum_hertz": float(tops.sum()),
        "gap_upper_hertz": gap_upper,
        "gap_lower_hertz": gap_lower,
        "gap_width_hertz": gap_upper - gap_lower,
        "bandgap_edge_hertz": edge,
        "bandgap_edge_angular": edge * 2.0 * np.pi,
        "argmax_index": int(upper_idx),
        "argmax_branch": int(upper_branch) + 3,
        "lower_argmax_index": int(lower_idx),
        "lower_argmax_branch": int(lower_branch),
        "band_tops": tops,
        "frequencies": table,
        "ball_hertz_min": float(lo),
        "ball_hertz_max": float(hi),
        "position_in_interval": float((gap_lower - lo) / (hi - lo)),
        "tau": float(scaling["tau"]),
        "wavelength_ratio": float(scaling["c_s"] / edge / float(lattice_constant)),
    }
SCICODE_GOLD_EOF
