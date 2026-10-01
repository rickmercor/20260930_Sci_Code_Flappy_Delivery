#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Gold oracle: ice_strength."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def ice_strength(H, A, Pstar, C):
    """Eq 8."""
    H = np.asarray(H, dtype=np.float64); A = np.asarray(A, dtype=np.float64)
    if np.any(H < 0) or np.any(A < 0) or np.any(A > 1):
        raise ValueError("need H >= 0 and A in [0, 1]")
    if Pstar <= 0 or C <= 0:
        raise ValueError("need Pstar > 0 and C > 0")
    return Pstar * H * np.exp(-C * (1.0 - A))

"""Gold oracle: effective_deformation."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def effective_deformation(eps, dmin, e):
    """Eq 7, with the two-dimensional trace-free part."""
    eps = np.asarray(eps, dtype=np.float64)
    if eps.shape[-2:] != (2, 2):
        raise ValueError("eps must be a stack of 2x2 tensors")
    if dmin <= 0 or e <= 0:
        raise ValueError("need dmin > 0 and e > 0")
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    d = eps.copy()
    d[..., 0, 0] = d[..., 0, 0] - 0.5 * tr
    d[..., 1, 1] = d[..., 1, 1] - 0.5 * tr
    return np.sqrt(dmin ** 2 + 2.0 * e ** -2 * np.einsum('...ij,...ij->...', d, d) + tr ** 2)

"""Gold oracle: vp_viscosities."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def vp_viscosities(P, Delta, e):
    """Eq 6."""
    P = np.asarray(P, dtype=np.float64); Delta = np.asarray(Delta, dtype=np.float64)
    if np.any(Delta <= 0) or e <= 0:
        raise ValueError("need Delta > 0 and e > 0")
    zeta = P / (2.0 * Delta)
    return np.stack([zeta, zeta / e ** 2])

"""Gold oracle: vp_stress."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def vp_stress(eps, P, dmin, e):
    """Eq 5."""
    eps = np.asarray(eps, dtype=np.float64)
    if eps.shape[-2:] != (2, 2):
        raise ValueError("eps must be a stack of 2x2 tensors")
    if np.any(np.asarray(P, dtype=np.float64) < 0):
        raise ValueError("the ice strength cannot be negative")
    D = effective_deformation(eps, dmin, e)
    zeta, eta = vp_viscosities(P, D, e)
    tr = eps[..., 0, 0] + eps[..., 1, 1]
    return (2.0 * eta[..., None, None] * eps
            + ((zeta - eta) * tr - 0.5 * np.asarray(P, dtype=np.float64))[..., None, None] * np.eye(2))

"""Gold oracle: ldg_numerical_fluxes."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, on_boundary):
    """Eqs 42-45. Returns (2, 2): row 0 the velocity flux, row 1 the traction flux."""
    if not (0.0 <= a < 0.5) or b <= 0:
        raise ValueError("need a in [0, 0.5) and b > 0")
    uL = np.asarray(uL, dtype=np.float64); n = np.asarray(n, dtype=np.float64)
    snL = np.asarray(sL, dtype=np.float64) @ n
    if on_boundary:
        return np.stack([np.zeros(2), snL - (b / (0.5 - a)) * uL])
    uR = np.asarray(uR, dtype=np.float64); snR = np.asarray(sR, dtype=np.float64) @ n
    return np.stack([0.5 * (uL + uR) + a * (uL - uR),
                     0.5 * (snL + snR) - a * (snL - snR) - b * (uL - uR)])

"""Gold oracle: ldg_strain."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GW = np.array([1.0, 1.0])

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



def _q1(xi, et):
    N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                         (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
    dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                          [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
    return N, dN


def _mirror(xi, et, f):
    return {0: (xi, 1.0), 1: (-1.0, et), 2: (xi, -1.0), 3: (1.0, et)}[f]


def _nb(i, j, f, nx):
    d = [(0, -1), (1, 0), (0, 1), (-1, 0)][f]
    ii, jj = i + d[0], j + d[1]
    return (ii, jj) if 0 <= ii < nx and 0 <= jj < nx else None


def _tables():
    """Face normals, face quadrature points and the reference basis values there."""
    fn = [np.array([0.0, -1.0]), np.array([1.0, 0.0]),
          np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    fp = [[(x, -1.0) for x in GP], [(1.0, x) for x in GP],
          [(x, 1.0) for x in GP], [(-1.0, x) for x in GP]]
    fb = [[_q1(xi, et)[0] for (xi, et) in pts] for pts in fp]
    fbo = [[_q1(*_mirror(xi, et, f))[0] for (xi, et) in fp[f]] for f in range(4)]
    vol = [_q1(xi, et) for xi in GP for et in GP]
    return fn, fp, fb, fbo, vol


_FACE_N, _FACE_PTS, _FB, _FBO, _VOL = _tables()


def ldg_strain(U, S, nx, h, a, b):
    """Eq 41 with element-constant stress: eps_K = (1/|K|) sum_F int_F sym(u_hat (x) n)."""
    if nx < 1 or h <= 0:
        raise ValueError("need nx >= 1 and h > 0")
    U = np.asarray(U, dtype=np.float64); S = np.asarray(S, dtype=np.float64)
    E = np.zeros((nx, nx, 2, 2))
    for j in range(nx):
        for i in range(nx):
            acc = np.zeros((2, 2))
            for f in range(4):
                n = _FACE_N[f]; nb = _nb(i, j, f, nx)
                for q, w in enumerate(GW):
                    uK = _FB[f][q] @ U[i, j]
                    if nb is None:
                        fl = ldg_numerical_fluxes(uK, None, S[i, j], None, n, a, b, True)
                    else:
                        ii, jj = nb
                        fl = ldg_numerical_fluxes(uK, _FBO[f][q] @ U[ii, jj], S[i, j], S[ii, jj], n, a, b, False)
                    uh = fl[0]
                    acc += 0.5 * (np.outer(uh, n) + np.outer(n, uh)) * w * (h / 2.0)
            E[i, j] = acc / (h * h)
    return E

"""Gold oracle: ldg_divergence."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GW = np.array([1.0, 1.0])

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



def _q1(xi, et):
    N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                         (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
    dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                          [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
    return N, dN


def _mirror(xi, et, f):
    return {0: (xi, 1.0), 1: (-1.0, et), 2: (xi, -1.0), 3: (1.0, et)}[f]


def _nb(i, j, f, nx):
    d = [(0, -1), (1, 0), (0, 1), (-1, 0)][f]
    ii, jj = i + d[0], j + d[1]
    return (ii, jj) if 0 <= ii < nx and 0 <= jj < nx else None


def _tables():
    """Face normals, face quadrature points and the reference basis values there."""
    fn = [np.array([0.0, -1.0]), np.array([1.0, 0.0]),
          np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    fp = [[(x, -1.0) for x in GP], [(1.0, x) for x in GP],
          [(x, 1.0) for x in GP], [(-1.0, x) for x in GP]]
    fb = [[_q1(xi, et)[0] for (xi, et) in pts] for pts in fp]
    fbo = [[_q1(*_mirror(xi, et, f))[0] for (xi, et) in fp[f]] for f in range(4)]
    vol = [_q1(xi, et) for xi in GP for et in GP]
    return fn, fp, fb, fbo, vol


_FACE_N, _FACE_PTS, _FB, _FBO, _VOL = _tables()


def ldg_divergence(U, S, nx, h, a, b):
    """The weak (div sigma, v) of Eq 46a: -(sigma, grad v)_K plus the traction fluxes."""
    if nx < 1 or h <= 0:
        raise ValueError("need nx >= 1 and h > 0")
    U = np.asarray(U, dtype=np.float64); S = np.asarray(S, dtype=np.float64)
    R = np.zeros((nx, nx, 4, 2)); J = h / 2.0
    for j in range(nx):
        for i in range(nx):
            r = np.zeros((4, 2))
            for (_N, dN) in _VOL:
                r -= (dN / J) @ S[i, j].T * J * J
            for f in range(4):
                n = _FACE_N[f]; nb = _nb(i, j, f, nx)
                for q, w in enumerate(GW):
                    N = _FB[f][q]; uK = N @ U[i, j]
                    if nb is None:
                        fl = ldg_numerical_fluxes(uK, None, S[i, j], None, n, a, b, True)
                    else:
                        ii, jj = nb
                        fl = ldg_numerical_fluxes(uK, _FBO[f][q] @ U[ii, jj], S[i, j], S[ii, jj], n, a, b, False)
                    r += np.outer(N, fl[1]) * w * J
            R[i, j] = r
    return R

"""Gold oracle: mevp_subcycle."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

RHO_ICE, RHO_A, RHO_O = 900.0, 1.3, 1026.0
CA, CO, FC = 1.2e-3, 5.5e-3, 1.46e-4
PSTAR, CCONC, ECC, DMIN = 27.5e3, 20.0, 2.0, 2e-9
LDOM, ALPHA, BETA, DTP, NSUB, AFLX, BFLX = 512e3, 500.0, 500.0, 600.0, 800, 0.25, 3.0
BASE = {1: (3, 1, 8.0), 2: (3, 2, 11.0), 3: (4, 1, 9.5)}
GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GW = np.array([1.0, 1.0])

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



def _q1(xi, et):
    N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                         (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
    dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                          [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
    return N, dN


def _mirror(xi, et, f):
    return {0: (xi, 1.0), 1: (-1.0, et), 2: (xi, -1.0), 3: (1.0, et)}[f]


def _nb(i, j, f, nx):
    d = [(0, -1), (1, 0), (0, 1), (-1, 0)][f]
    ii, jj = i + d[0], j + d[1]
    return (ii, jj) if 0 <= ii < nx and 0 <= jj < nx else None


def _tables():
    """Face normals, face quadrature points and the reference basis values there."""
    fn = [np.array([0.0, -1.0]), np.array([1.0, 0.0]),
          np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    fp = [[(x, -1.0) for x in GP], [(1.0, x) for x in GP],
          [(x, 1.0) for x in GP], [(-1.0, x) for x in GP]]
    fb = [[_q1(xi, et)[0] for (xi, et) in pts] for pts in fp]
    fbo = [[_q1(*_mirror(xi, et, f))[0] for (xi, et) in fp[f]] for f in range(4)]
    vol = [_q1(xi, et) for xi in GP for et in GP]
    return fn, fp, fb, fbo, vol


_FACE_N, _FACE_PTS, _FB, _FBO, _VOL = _tables()


def mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b):
    """One mEVP sub-iteration, Eqs 9-10.

    Returns a flat float64 array holding, in order, the new velocity
    (nx*nx*4*2 entries), the new stress (nx*nx*2*2) and the VP stress target
    (nx*nx*2*2) that the stress relaxes towards."""
    if alpha <= 0 or beta <= 0 or dt <= 0:
        raise ValueError("need alpha > 0, beta > 0 and dt > 0")
    E = ldg_strain(U, S, nx, h, a, b)
    tgt = np.zeros_like(np.asarray(S, dtype=np.float64))
    for j in range(nx):
        for i in range(nx):
            tgt[i, j] = vp_stress(E[i, j], P[i, j], DMIN, ECC)
    Snew = (alpha * np.asarray(S, dtype=np.float64) + tgt) / (alpha + 1.0)
    R = ldg_divergence(U, Snew, nx, h, a, b)
    M = np.zeros((4, 4)); J = h / 2.0
    for xi in GP:
        for et in GP:
            N, _ = _q1(xi, et)
            M += np.outer(N, N) * J * J
    Minv = np.linalg.inv(M)
    Unew = np.zeros_like(np.asarray(U, dtype=np.float64))
    for j in range(nx):
        for i in range(nx):
            m = RHO_ICE * H[i, j]
            rhs = ((beta * m / dt) * (M @ U[i, j]) + (m / dt) * (M @ Un[i, j])
                   + R[i, j] + M @ Fv[i, j])
            Unew[i, j] = Minv @ rhs / (beta * m / dt + m / dt)
    return np.concatenate([Unew.ravel(), Snew.ravel(), tgt.ravel()])

"""Gold oracle: sea_ice_audit."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

RHO_ICE, RHO_A, RHO_O = 900.0, 1.3, 1026.0
CA, CO, FC = 1.2e-3, 5.5e-3, 1.46e-4
PSTAR, CCONC, ECC, DMIN = 27.5e3, 20.0, 2.0, 2e-9
LDOM, ALPHA, BETA, DTP, NSUB, AFLX, BFLX = 512e3, 500.0, 500.0, 600.0, 800, 0.25, 3.0
BASE = {1: (3, 1, 8.0), 2: (3, 2, 11.0), 3: (4, 1, 9.5)}
GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GW = np.array([1.0, 1.0])

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



def _q1(xi, et):
    N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                         (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
    dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                          [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
    return N, dN


def _mirror(xi, et, f):
    return {0: (xi, 1.0), 1: (-1.0, et), 2: (xi, -1.0), 3: (1.0, et)}[f]


def _nb(i, j, f, nx):
    d = [(0, -1), (1, 0), (0, 1), (-1, 0)][f]
    ii, jj = i + d[0], j + d[1]
    return (ii, jj) if 0 <= ii < nx and 0 <= jj < nx else None


def _tables():
    """Face normals, face quadrature points and the reference basis values there."""
    fn = [np.array([0.0, -1.0]), np.array([1.0, 0.0]),
          np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    fp = [[(x, -1.0) for x in GP], [(1.0, x) for x in GP],
          [(x, 1.0) for x in GP], [(-1.0, x) for x in GP]]
    fb = [[_q1(xi, et)[0] for (xi, et) in pts] for pts in fp]
    fbo = [[_q1(*_mirror(xi, et, f))[0] for (xi, et) in fp[f]] for f in range(4)]
    vol = [_q1(xi, et) for xi in GP for et in GP]
    return fn, fp, fb, fbo, vol


def _fields(nx, v):
    h = LDOM / nx
    H = np.zeros((nx, nx)); A = np.zeros((nx, nx))
    for j in range(nx):
        for i in range(nx):
            xc, yc = (i + 0.5) / nx, (j + 0.5) / nx
            H[i, j] = 0.3 + 0.05 * np.sin((3 + v) * np.pi * xc) * np.cos((2 + v) * np.pi * yc)
            A[i, j] = 0.90 + 0.05 * np.sin((2 + v) * np.pi * (xc + yc))
    return H, A


def _forcing(U, A, H, nx, wind):
    """Eqs 2-3: F = A (tau_a + tau_o) + rho_ice H f k x (u_o - u), both drags quadratic
    in the velocity RELATIVE to the ice and both weighted by the concentration; the
    Coriolis term acts on the ocean-minus-ice velocity, which carries the -k x u sign."""
    F = np.zeros((nx, nx, 4, 2))
    for j in range(nx):
        for i in range(nx):
            for c, (dx, dy) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
                x, y = (i + dx) / nx, (j + dy) / nx
                ua = wind * np.array([np.sin(np.pi * x) * np.cos(np.pi * y) + 0.8 * (x - 0.5),
                                      -np.cos(np.pi * x) * np.sin(np.pi * y) + 0.8 * (y - 0.5)])
                uo = 0.01 * np.array([2.0 * y - 1.0, -2.0 * x + 1.0])
                u = U[i, j, c]
                rel_a = ua - u; rel_o = uo - u
                drag = (RHO_A * CA * np.linalg.norm(rel_a) * rel_a
                        + RHO_O * CO * np.linalg.norm(rel_o) * rel_o)
                F[i, j, c] = (A[i, j] * drag
                              + RHO_ICE * H[i, j] * FC * np.array([-rel_o[1], rel_o[0]]))
    return F


def _shear_deformation(E):
    E = np.asarray(E, dtype=np.float64)
    return np.sqrt((E[..., 0, 0] - E[..., 1, 1]) ** 2 + 4.0 * E[..., 0, 1] ** 2)


_FACE_N, _FACE_PTS, _FB, _FBO, _VOL = _tables()


def sea_ice_audit(wind_scale):
    if isinstance(wind_scale, bool) or not np.isfinite(wind_scale) or wind_scale <= 0:
        raise ValueError("wind_scale must be positive and finite")
    rows = []
    for v in (1, 2, 3):
        nx, fv, wind = BASE[v]
        h = LDOM / nx
        H, A = _fields(nx, fv)
        P = ice_strength(H, A, PSTAR, CCONC)
        U = np.zeros((nx, nx, 4, 2)); Sg = np.zeros((nx, nx, 2, 2)); Un = U.copy()
        r0 = rlast = rprev = 0.0
        for k in range(NSUB):
            Fv = _forcing(U, A, H, nx, wind * wind_scale)
            out = mevp_subcycle(U, Sg, Un, P, H, Fv, nx, h, ALPHA, BETA, DTP, AFLX, BFLX)
            nu = nx * nx * 8; ns = nx * nx * 4
            U = out[:nu].reshape(nx, nx, 4, 2)
            Sg = out[nu:nu + ns].reshape(nx, nx, 2, 2)
            tgt = out[nu + ns:].reshape(nx, nx, 2, 2)
            r = float(np.max(np.abs(Sg - tgt)))
            if k == 0:
                r0 = r
            rprev, rlast = rlast, r
        E = ldg_strain(U, Sg, nx, h, AFLX, BFLX)
        # diagnostics at the converged state, each an invariant the source implies
        D = effective_deformation(E, DMIN, ECC)
        vis = vp_viscosities(P, D, ECC)
        sig_vp = vp_stress(E, P, DMIN, ECC)
        Rmom = ldg_divergence(U, Sg, nx, h, AFLX, BFLX)
        fl_b = ldg_numerical_fluxes(U[0, 0, 0], None, Sg[0, 0], None,
                                    np.array([-1.0, 0.0]), AFLX, BFLX, True)
        if not (np.all(D >= DMIN) and np.all(vis[0] >= vis[1])
                and np.all(np.isfinite(sig_vp)) and np.all(np.isfinite(Rmom))
                and np.all(np.isfinite(fl_b)) and np.allclose(fl_b[0], 0.0)):
            raise ValueError("the converged state violates the source's invariants")
        eII = _shear_deformation(E)
        rows.append([float(np.max(eII) * 1e6), float(np.mean(eII) * 1e6),
                     float(np.max(np.abs(U))), float(rlast / rprev),
                     float(np.sum(eII) * 1e6)])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
