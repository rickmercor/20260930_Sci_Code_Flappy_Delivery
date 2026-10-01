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

# ---------------------------------------------------------------------------
# shared fixtures (step 1 block)
# ---------------------------------------------------------------------------

def _check_params(params):
    """Default model parameters, kept inside the function so that field
    extraction cannot separate them from their only user."""
    p = dict(M=10.0, eps=0.4, alpha=0.25, C_sav=400.0)
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dict of overrides, or None")
    if params is not None:
        for k in params:
            if k not in p:
                raise ValueError("unknown model parameter %r" % (k,))
        p.update({k: float(v) for k, v in params.items()})
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("model parameters must be finite")
    if p["M"] <= 0.0:
        raise ValueError("the mobility M must be positive")
    if p["alpha"] < 0.0:
        raise ValueError("alpha must be non-negative")
    return p


def _check_coef(coef):
    """Prescribed member of the source's coefficient family, kept inside the
    function for the same reason."""
    c = dict(a11t=0.5, a32t=-0.25, a33t=1.5, a31=1.0 / 3.0, a43=2.0 / 3.0)
    if coef is not None and not isinstance(coef, dict):
        raise ValueError("coef must be a dict of overrides, or None")
    if coef is not None:
        for k in coef:
            if k not in c:
                raise ValueError("unknown Runge-Kutta coefficient %r" % (k,))
        c.update({k: float(v) for k, v in coef.items()})
    if not all(np.isfinite(v) for v in c.values()):
        raise ValueError("Runge-Kutta coefficients must be finite")
    if c["a11t"] == 0.0 or c["a33t"] == 0.0 or c["a43"] == 0.0:
        raise ValueError("a11t, a33t and a43 must be nonzero")
    return c


def _check_cell(cell, shape):
    L = np.atleast_1d(np.asarray(cell, float)).ravel()
    if L.size == 1:
        L = np.full(len(shape), float(L[0]))
    if L.size != len(shape):
        raise ValueError("cell must be a scalar or match the field rank")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("cell edge lengths must be finite and positive")
    return L


def _check_field(u, name="u"):
    if np.asarray(u).dtype.kind == "c":
        raise ValueError("%s must be a real field, not a complex one" % name)
    u = np.asarray(u, float)
    if u.ndim < 1:
        raise ValueError("%s must be a periodic field of at least one dimension" % name)
    if min(u.shape) < 2:
        raise ValueError("%s must have at least two points along every axis" % name)
    if not np.all(np.isfinite(u)):
        raise ValueError("%s must be finite" % name)
    return u


def _wavenumbers(shape, L):
    return [2.0 * np.pi * np.fft.fftfreq(n, d=Ln / n) for n, Ln in zip(shape, L)]


def _kgrids(shape, L):
    """The d wavenumber axes, each shaped so that it broadcasts over the field."""
    ks = _wavenumbers(shape, L)
    d = len(shape)
    out = []
    for i, k in enumerate(ks):
        sh = [1] * d
        sh[i] = len(k)
        out.append(k.reshape(sh))
    return out


def _k2(shape, L):
    return sum(k ** 2 for k in _kgrids(shape, L))


def _ip(f, g, L):
    """Discrete L2 inner product: plain grid sum times the cell area."""
    dv = float(np.prod(L)) / float(np.prod(f.shape))
    return float(np.sum(f * g) * dv)


def mpfc_spectral_operators(u: "np.ndarray", cell: "float | tuple") -> "np.ndarray":
    u = _check_field(u)
    L = _check_cell(cell, u.shape)
    ks = _kgrids(u.shape, L)
    k2 = sum(k ** 2 for k in ks)
    uh = np.fft.fftn(u)
    planes = [np.real(np.fft.ifftn(1j * k * uh)) for k in ks]
    planes.append(np.real(np.fft.ifftn(-k2 * uh)))
    planes.append(np.real(np.fft.ifftn(k2 ** 2 * uh)))
    planes.append(np.real(np.fft.ifftn((1.0 - k2) ** 2 * uh)))
    invh = np.zeros_like(uh)
    nz = np.broadcast_to(k2, u.shape) > 0.0
    invh[nz] = uh[nz] / np.broadcast_to(k2, u.shape)[nz]
    planes.append(np.real(np.fft.ifftn(invh)))
    return np.stack(planes)

import numpy as np

def mpfc_free_energy(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    ops = mpfc_spectral_operators(phi, L)
    lphi = ops[-2]
    chi = ops[-1]
    mean = float(np.sum(phi)) / float(phi.size)
    e_quad = 0.5 * _ip(phi, lphi, L)
    bulk = _ip(0.25 * phi ** 4 - 0.5 * p["eps"] * phi ** 2, np.ones_like(phi), L)
    e_nl = 0.5 * p["alpha"] * _ip(chi, phi - mean, L)
    e1 = bulk + e_nl
    return np.array([e_quad + e1, e_quad, e1, mean])

import numpy as np

def mpfc_sav_terms(phi: "np.ndarray", cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    ops = mpfc_spectral_operators(phi, L)
    lphi, chi = ops[-2], ops[-1]
    e1 = mpfc_free_energy(phi, L, params)[2]
    rad = e1 + p["C_sav"]
    if not (rad > 0.0):
        raise ValueError("E1(phi) + C_sav must be strictly positive")
    nl = phi ** 3 - p["eps"] * phi + p["alpha"] * chi
    return np.stack([nl / np.sqrt(rad), nl + lphi])

import numpy as np

def mpfc_butcher(a11t: "float | None" = None, a32t: "float | None" = None,
        a33t: "float | None" = None, a31: "float | None" = None,
        a43: "float | None" = None) -> "np.ndarray":
    given = dict(a11t=a11t, a32t=a32t, a33t=a33t, a31=a31, a43=a43)
    c = _check_coef({k: v for k, v in given.items() if v is not None})
    At = np.zeros((4, 4))
    A = np.zeros((4, 4))
    b = np.array([0.0, 0.75, 0.0, 0.25])
    At[0, 0] = c["a11t"]
    At[1, 1] = 2.0 / 3.0
    At[2, 0] = (2.0 - 6.0 * c["a11t"]) / (3.0 * c["a43"]) + c["a11t"] - c["a32t"] - c["a33t"]
    At[2, 1] = c["a32t"]
    At[2, 2] = c["a33t"]
    At[3, 1] = -1.0
    At[3, 3] = 1.0
    A[1, 0] = 2.0 / 3.0
    A[2, 0] = c["a31"]
    A[2, 1] = 2.0 / (3.0 * c["a43"]) - c["a31"]
    A[3, 0] = -c["a43"]
    A[3, 2] = c["a43"]
    st = [sum(At[l, m] for m in range(l + 1)) for l in range(4)]
    s = [sum(A[l, m] for m in range(l)) for l in range(4)]
    ct = [sum(b[m] * At[m, l] for m in range(l, 4)) for l in range(4)]
    ca = [sum(b[m] * A[m, l] for m in range(l + 1, 4)) for l in range(4)]
    res = np.array([
        sum(b) - 1.0,
        sum(b[l] * st[l] for l in range(4)) - 0.5,
        sum(ct[l] * st[l] for l in range(4)) - 1.0 / 6.0,
        sum(b[l] * s[l] for l in range(1, 4)) - 0.5,
        sum(ct[l] * s[l] for l in range(1, 4)) - 1.0 / 6.0,
        sum(b[l] * st[l] * s[l] for l in range(1, 4)) - 1.0 / 3.0,
        sum(b[l] * s[l] ** 2 for l in range(1, 4)) - 1.0 / 3.0,
        sum(ca[l] * st[l] for l in range(0, 3)) - 1.0 / 6.0,
        sum(ca[l] * s[l] for l in range(1, 3)) - 1.0 / 6.0,
    ])
    P = np.diag(b) @ At + At.T @ np.diag(b) - np.outer(b, b)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (P + P.T)))
    return np.concatenate([At.ravel(), A.ravel(), b, res, eig])

import numpy as np

def mpfc_stage_solve(acc: "np.ndarray", Hl: "np.ndarray", rho: float, a_ll: float,
        dt: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    acc = _check_field(acc, "acc")
    Hl = _check_field(Hl, "Hl")
    if Hl.shape != acc.shape:
        raise ValueError("acc and Hl must have the same shape")
    L = _check_cell(cell, acc.shape)
    p = _check_params(params)
    rho = float(rho)
    a_ll = float(a_ll)
    dt = float(dt)
    if not np.isfinite(rho) or not np.isfinite(a_ll):
        raise ValueError("rho and a_ll must be finite scalars")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    k2 = _k2(acc.shape, L)
    Lt = (k2 - 1.0) ** 2
    Gt = -p["M"] * k2
    den = 1.0 - dt * a_ll * Gt * Lt
    if np.any(den == 0.0):
        raise ValueError("the stage operator is singular for these data")
    Hh = np.fft.fftn(Hl)
    Al = np.real(np.fft.ifftn((Gt * Lt * np.fft.fftn(acc) + rho * Gt * Hh) / den))
    Bl = np.real(np.fft.ifftn((Gt * Hh) / den))
    dn = 2.0 - dt * a_ll * _ip(Hl, Bl, L)
    if dn == 0.0:
        raise ValueError("the stage equation for the auxiliary variable is singular")
    rkl = _ip(Hl, Al, L) / dn
    pkl = Al + dt * a_ll * rkl * Bl
    return np.stack([Al, Bl, pkl, np.full_like(Al, rkl)])

import numpy as np

def mpfc_time_step(phi: "np.ndarray", r: float, dt: float, cell: "float | tuple",
        params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    c = _check_coef(coef)
    r = float(r)
    dt = float(dt)
    if not np.isfinite(r):
        raise ValueError("r must be a finite scalar")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    tab = mpfc_butcher(**c)
    At = tab[:16].reshape(4, 4)
    A = tab[16:32].reshape(4, 4)
    b = tab[32:36]
    phik = []
    rk = []
    for l in range(4):
        psi = phi.copy()
        for m in range(l):
            if A[l, m] != 0.0:
                psi = psi + dt * A[l, m] * phik[m]
        Hl = mpfc_sav_terms(psi, L, params)[0]
        acc = phi.copy()
        rho = r
        for m in range(l):
            if At[l, m] != 0.0:
                acc = acc + dt * At[l, m] * phik[m]
                rho = rho + dt * At[l, m] * rk[m]
        st = mpfc_stage_solve(acc, Hl, rho, At[l, l], dt, L, params)
        phik.append(st[2])
        rk.append(float(st[3].flat[0]))
    new = phi.copy()
    rn = r
    for l in range(4):
        if b[l] != 0.0:
            new = new + dt * b[l] * phik[l]
            rn = rn + dt * b[l] * rk[l]
    return np.stack([new, np.full_like(new, rn)])

import numpy as np

def mpfc_modified_energy(phi: "np.ndarray", r: float, cell: "float | tuple",
        params: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    r = float(r)
    if not np.isfinite(r):
        raise ValueError("r must be a finite scalar")
    e = mpfc_free_energy(phi, L, params)
    emod = e[1] + r * r - p["C_sav"]
    rad = e[2] + p["C_sav"]
    if not (rad > 0.0):
        raise ValueError("E1(phi) + C_sav must be strictly positive")
    return np.array([emod, e[0], emod - e[0], r - np.sqrt(rad)])

import numpy as np

def mpfc_energy_identity(phi: "np.ndarray", r: float, dt: float,
        cell: "float | tuple", params: "dict | None" = None,
        coef: "dict | None" = None) -> "np.ndarray":
    phi = _check_field(phi, "phi")
    L = _check_cell(cell, phi.shape)
    p = _check_params(params)
    c = _check_coef(coef)
    r = float(r)
    dt = float(dt)
    if not np.isfinite(r):
        raise ValueError("r must be a finite scalar")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    tab = mpfc_butcher(**c)
    At = tab[:16].reshape(4, 4)
    A = tab[16:32].reshape(4, 4)
    b = tab[32:36]

    phik = []
    rk = []
    for l in range(4):
        psi = phi.copy()
        for m in range(l):
            if A[l, m] != 0.0:
                psi = psi + dt * A[l, m] * phik[m]
        Hl = mpfc_sav_terms(psi, L, params)[0]
        acc = phi.copy()
        rho = r
        for m in range(l):
            if At[l, m] != 0.0:
                acc = acc + dt * At[l, m] * phik[m]
                rho = rho + dt * At[l, m] * rk[m]
        st = mpfc_stage_solve(acc, Hl, rho, At[l, l], dt, L, params)
        phik.append(st[2])
        rk.append(float(st[3].flat[0]))
    new = phi.copy()
    for l in range(4):
        if b[l] != 0.0:
            new = new + dt * b[l] * phik[l]
    # the accumulation of the identity INCLUDES the diagonal term, unlike the
    # three accumulations that feed a stage
    phin = [phi + dt * sum(At[l, m] * phik[m] for m in range(l + 1)) for l in range(4)]
    P = np.diag(b) @ At + At.T @ np.diag(b) - np.outer(b, b)
    # the shifted bi-Laplacian is the last but one plane of step 1's return
    lk = [mpfc_spectral_operators(f, L)[-2] for f in phik]
    q_next = _ip(new, mpfc_spectral_operators(new, L)[-2], L)
    q_now = _ip(phi, mpfc_spectral_operators(phi, L)[-2], L)
    work = 2.0 * dt * sum(b[l] * _ip(phin[l], lk[l], L) for l in range(4))
    pform = dt ** 2 * sum(P[l, m] * _ip(phik[l], lk[m], L)
                          for l in range(4) for m in range(4))
    return np.array([q_next, q_now, work, pform])

import numpy as np

def _mpfc_initial_field(grid, cell):
    L = _check_cell(cell, (0, 0))
    x = np.arange(grid[0]) * (L[0] / grid[0])
    y = np.arange(grid[1]) * (L[1] / grid[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 2.0 * np.pi / L[0]
    ky = 2.0 * np.pi / L[1]
    return (0.15 + 0.30 * np.cos(6.0 * kx * X) * np.cos(4.0 * ky * Y)
            + 0.20 * np.sin(2.0 * kx * X) * np.sin(6.0 * ky * Y))


def mpfc_rksav_run(n_steps: int = 16, dt: float = 0.125, grid: tuple = (48, 64),
        cell: "float | tuple" = (32.0, 48.0), params: "dict | None" = None,
        coef: "dict | None" = None, quantity: str = "energy") -> float:
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    gr = np.atleast_1d(np.asarray(grid)).ravel()
    if gr.dtype.kind not in "buif" or not all(float(n).is_integer() for n in gr):
        raise ValueError("grid must be two integers, each at least four")
    g = tuple(int(n) for n in gr)
    if len(g) != 2 or min(g) < 4:
        raise ValueError("grid must be two integers, each at least four")
    if quantity not in ("energy", "energy0", "emod", "emod0", "quad", "r",
                        "mass", "drift", "work", "qform"):
        raise ValueError("unknown quantity %r" % (quantity,))
    L = _check_cell(cell, g)
    p = _check_params(params)
    c = _check_coef(coef)
    phi = _mpfc_initial_field(g, L)
    e0 = mpfc_free_energy(phi, L, params)
    r = float(np.sqrt(e0[2] + p["C_sav"]))
    area = float(L[0] * L[1])
    m0 = _ip(phi, np.ones_like(phi), L)
    emod0 = mpfc_modified_energy(phi, r, L, params)[0]
    nst = int(n_steps)
    ident = None
    for i in range(nst):
        # the last step is also taken through the energy identity of step 8,
        # whose first entry IS the quadratic block of the final layer, paired
        # with L, and whose remaining entries are the two terms the energy
        # theorem splits the change in that block into
        if i == nst - 1:
            ident = mpfc_energy_identity(phi, r, dt, L, params, c)
        out = mpfc_time_step(phi, r, dt, L, params, c)
        phi = out[0]
        r = float(out[1].flat[0])
    ef = mpfc_free_energy(phi, L, params)
    # the quadratic half of the reported energy comes from the identity, the
    # nonlinear half from the energy functional; with no step taken there is no
    # identity to read and the quadratic half is the initial one
    e_quad = 0.5 * float(ident[0]) if ident is not None else ef[1]
    work = float(ident[2]) if ident is not None else 0.0
    qform = float(ident[3]) if ident is not None else 0.0
    md = mpfc_modified_energy(phi, r, L, params)
    m1 = _ip(phi, np.ones_like(phi), L)
    table = dict(energy=(e_quad + ef[2]) / area, energy0=e0[0] / area,
                 emod=md[0] / area, emod0=emod0 / area, quad=e_quad / area,
                 r=r, mass=abs(m1 - m0) / abs(m0), drift=abs(md[3]),
                 work=work / area, qform=qform / area)
    return float(table[quantity])
SCICODE_GOLD_EOF
