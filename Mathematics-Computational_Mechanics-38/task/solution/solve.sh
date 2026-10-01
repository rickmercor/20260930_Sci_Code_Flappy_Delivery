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


def _s01_kernel(r, delta):
    q = np.asarray(r, dtype=float) / delta
    f = np.where(q <= 0.5, 1.0 - 6.0*q**2 + 6.0*q**3, 2.0*(1.0 - q)**3)
    f = np.where(q <= 1.0, np.maximum(f, 0.0), 0.0)
    return (8.0/np.pi) / delta**3 * f


def build_families(nx, ny, nz, dx, delta):
    if not all(isinstance(v, (int, np.integer)) and v >= 1 for v in (nx, ny, nz)):
        raise ValueError("grid dimensions must be positive integers")
    if not (dx > 0.0 and delta > dx):
        raise ValueError("require dx > 0 and delta > dx")
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    X = (ijk + 0.5) * dx
    N = nx * ny * nz
    V = dx**3
    m = int(np.floor(delta/dx)) + 1
    offs = []
    for a in range(-m, m+1):
        for b in range(-m, m+1):
            for c in range(-m, m+1):
                if (a, b, c) == (0, 0, 0):
                    continue
                if dx*np.sqrt(a*a + b*b + c*c) <= delta:
                    offs.append((a, b, c))
    offs = np.array(offs, dtype=int).reshape(-1, 3)
    pos = offs[(offs[:, 0] > 0) | ((offs[:, 0] == 0) & (offs[:, 1] > 0)) |
               ((offs[:, 0] == 0) & (offs[:, 1] == 0) & (offs[:, 2] > 0))]
    lut = -np.ones((nx, ny, nz), dtype=int)
    lut[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(N)
    K, L = [], []
    for (a, b, c) in pos:
        i2 = ijk[:, 0] + a; j2 = ijk[:, 1] + b; k2 = ijk[:, 2] + c
        ok = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny) & (k2 >= 0) & (k2 < nz)
        K.append(np.arange(N)[ok]); L.append(lut[i2[ok], j2[ok], k2[ok]])
    pk = np.concatenate(K) if K else np.zeros(0, dtype=int)
    pn = np.concatenate(L) if L else np.zeros(0, dtype=int)
    if pk.size == 0:
        raise ValueError("no bond lies within the horizon: every point is isolated")
    dX = X[pn] - X[pk]
    r = np.linalg.norm(dX, axis=1)
    om = _s01_kernel(r, delta)
    om0 = np.zeros(N)
    np.add.at(om0, pk, om*V)
    np.add.at(om0, pn, om*V)
    if np.any(om0 <= 0.0):
        raise ValueError("a point has an empty family (zero discrete kernel integral)")
    omb = om*0.5*(1.0/om0[pk] + 1.0/om0[pn])
    jk = ijk[pk, 1]; jn = ijk[pn, 1]
    ik = ijk[pk, 0]; iN = ijk[pn, 0]
    half = ny // 2
    cross = ((jk <= half-1) & (jn >= half)) | ((jn <= half-1) & (jk >= half))
    notch = cross & (ik <= nx//2 - 1) & (iN <= nx//2 - 1)
    nofail = (jk == 0) | (jn == 0) | (jk == ny-1) | (jn == ny-1)
    return {'nx': int(nx), 'ny': int(ny), 'nz': int(nz), 'dx': float(dx),
            'delta': float(delta), 'V': float(V), 'ijk': ijk, 'X': X,
            'pk': pk, 'pn': pn, 'dX': dX, 'r': r, 'omega': om, 'omega0': om0,
            'omega_b': omb, 'notch': notch, 'nofail': nofail}

import numpy as np


def _s02_kernel_shape(q, kernel):
    q = np.asarray(q, dtype=float)
    if kernel == 'cubic':
        f = np.where(q <= 0.5, 1.0 - 6.0*q**2 + 6.0*q**3, 2.0*(1.0 - q)**3)
        return np.where(q <= 1.0, np.maximum(f, 0.0), 0.0)
    if kernel == 'linear':
        return np.where(q <= 1.0, 1.0 - q, 0.0)
    if kernel == 'constant':
        return np.where(q <= 1.0, 1.0, 0.0)
    raise ValueError("kernel must be 'cubic', 'linear' or 'constant'")


def pfpd_normalization_constant(kernel):
    if kernel not in ('cubic', 'linear', 'constant'):
        raise ValueError("kernel must be 'cubic', 'linear' or 'constant'")
    x, w = np.polynomial.legendre.leggauss(200)
    num = den = 0.0
    for a, b in ((0.0, 0.5), (0.5, 1.0)):
        q = 0.5*(b - a)*(x + 1.0) + a
        wq = 0.5*(b - a)*w
        f = _s02_kernel_shape(q, kernel)
        num += float(np.sum(wq * f * q**3))
        den += float(np.sum(wq * f * q**2))
    return float(num / (2.0 * den))

import numpy as np


def kinematic_operators(fam, s, s_c):
    if not (0.0 < s_c < 1.0):
        raise ValueError("require 0 < s_c < 1")
    pk = fam['pk']; pn = fam['pn']; dX = fam['dX']; om = fam['omega']; V = fam['V']
    N = fam['X'].shape[0]
    s = np.asarray(s, dtype=float)
    if s.shape != pk.shape:
        raise ValueError("s must hold one phase-field value per bond pair")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("bond phase-field values must lie in [0, 1]")
    h = np.where(s <= s_c, 1.0, ((1.0 - s)/(1.0 - s_c))**2)
    w_h = om * h
    outer = dX[:, :, None]*dX[:, None, :]
    M = np.zeros((N, 3, 3))
    np.add.at(M, pk, (w_h*V)[:, None, None]*outer)
    np.add.at(M, pn, (w_h*V)[:, None, None]*outer)
    Minv = np.linalg.inv(M)
    dphi_kn = (w_h*V)[:, None]*np.einsum('pij,pj->pi', Minv[pk], dX)
    dphi_nk = (w_h*V)[:, None]*np.einsum('pij,pj->pi', Minv[pn], -dX)
    return {'h': h, 'M': M, 'dphi_kn': dphi_kn, 'dphi_nk': dphi_nk}

import numpy as np


def bond_deformation_gradients(fam, ops, u):
    pk = fam['pk']; pn = fam['pn']; dX = fam['dX']; r = fam['r']
    N = fam['X'].shape[0]
    u = np.asarray(u, dtype=float)
    if u.shape != (N, 3):
        raise ValueError("u must be an (N, 3) displacement array")
    dphi_kn = ops['dphi_kn']; dphi_nk = ops['dphi_nk']
    dU = u[pn] - u[pk]
    Fb = np.tile(np.eye(3), (N, 1, 1))
    np.add.at(Fb, pk, dU[:, :, None]*dphi_kn[:, None, :])
    np.add.at(Fb, pn, (-dU)[:, :, None]*dphi_nk[:, None, :])
    Fav = 0.5*(Fb[pk] + Fb[pn])
    corr = (dX + dU) - np.einsum('pij,pj->pi', Fav, dX)
    Ft = Fav + corr[:, :, None]*dX[:, None, :]/(r**2)[:, None, None]
    return Ft

import numpy as np


def bond_stress_and_driving_force(F, E, nu):
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (3, 3):
        raise ValueError("F must have shape (3, 3) or (P, 3, 3)")
    if not (E > 0.0 and -1.0 < nu < 0.5):
        raise ValueError("require E > 0 and -1 < nu < 0.5")
    single = (F.ndim == 2)
    Ft = F.reshape(-1, 3, 3)
    J = np.linalg.det(Ft)
    if np.any(J <= 0.0):
        raise ValueError("deformation gradient must have positive determinant")
    lam = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    mu = E/(2.0*(1.0 + nu))
    Ftt = np.swapaxes(Ft, -1, -2)
    Egr = 0.5*(Ftt @ Ft - np.eye(3))
    trE = np.trace(Egr, axis1=-2, axis2=-1)[..., None, None]
    S = lam*trE*np.eye(3) + 2.0*mu*Egr
    P0 = Ft @ S
    sig = (P0 @ Ftt)/J[..., None, None]
    sig = 0.5*(sig + np.swapaxes(sig, -1, -2))
    s1 = np.linalg.eigvalsh(sig)[..., -1]
    Y = np.maximum(s1, 0.0)**2/(2.0*E)
    if single:
        return {'P0': P0[0], 'Y': float(Y[0])}
    return {'P0': P0, 'Y': Y}

import numpy as np


def bond_phase_field(calY, Gc, delta, c0):
    calY = np.asarray(calY, dtype=float)
    if np.any(calY < 0.0):
        raise ValueError("history values must be non-negative")
    if not (Gc > 0.0 and delta > 0.0):
        raise ValueError("require Gc > 0 and delta > 0")
    if not (0.0 < c0 < 1.0):
        raise ValueError("normalization constant must lie in (0, 1)")
    Yc = Gc/(2.0*c0*delta)
    s = np.minimum(1.0, calY/(calY + Yc))
    if s.ndim == 0:
        return float(s)
    return s

import numpy as np


def internal_force_evaluation(fam, u, calY, s_prev, E, nu, Gc, s_c, c0):
    if not (E > 0.0 and -1.0 < nu < 0.5 and Gc > 0.0 and 0.0 < s_c < 1.0 and 0.0 < c0 < 1.0):
        raise ValueError("invalid material, fracture or normalization parameters")
    pk = fam['pk']; pn = fam['pn']; dX = fam['dX']; r = fam['r']
    omb = fam['omega_b']; V = fam['V']; nofail = fam['nofail']
    N = fam['X'].shape[0]
    u = np.asarray(u, dtype=float)
    calY = np.asarray(calY, dtype=float)
    s_prev = np.asarray(s_prev, dtype=float)
    if u.shape != (N, 3) or calY.shape != pk.shape or s_prev.shape != pk.shape:
        raise ValueError("u must be (N, 3); calY and s_prev must hold one value per bond pair")
    # (1) kinematic operators from the phase-field of the PREVIOUS evaluation (step 3)
    ops = kinematic_operators(fam, s_prev, s_c)
    # (2) bond-associated deformation gradients (step 4)
    Ft = bond_deformation_gradients(fam, ops, u)
    # (3) undamaged bond stress and stress-based crack driving force (step 5)
    resp = bond_stress_and_driving_force(Ft, E, nu)
    P0 = resp['P0']
    Y = np.where(nofail, 0.0, resp['Y'])
    # (4) history update and closed-form phase-field (step 6)
    calY_new = np.maximum(calY, Y)
    s = bond_phase_field(calY_new, Gc, fam['delta'], c0)
    # (5) degraded stress enters the two-term force state
    Pt = ((1.0 - s)**2)[:, None, None]*P0
    nhat = dX/r[:, None]
    proj = np.eye(3) - nhat[:, :, None]*nhat[:, None, :]
    ZC = (omb*V)[:, None, None]*(Pt @ proj)
    Z = np.zeros((N, 3, 3))
    np.add.at(Z, pk, ZC)
    np.add.at(Z, pn, ZC)
    dphi_kn = ops['dphi_kn']; dphi_nk = ops['dphi_nk']
    t_kn = omb[:, None]*np.einsum('pij,pj->pi', Pt, nhat) + np.einsum('pij,pj->pi', Z[pk], dphi_kn)/V
    t_nk = omb[:, None]*np.einsum('pij,pj->pi', Pt, -nhat) + np.einsum('pij,pj->pi', Z[pn], dphi_nk)/V
    B = np.zeros((N, 3))
    np.add.at(B, pk, (t_kn - t_nk)*V)
    np.add.at(B, pn, (t_nk - t_kn)*V)
    return {'B': B, 'calY': calY_new, 's': s}

import numpy as np


def velocity_verlet_run(fam, E, nu, rho0, Gc, s_c, v0, dt, n_steps, c0):
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 0):
        raise ValueError("n_steps must be a non-negative integer")
    if not (E > 0.0 and -1.0 < nu < 0.5 and rho0 > 0.0 and Gc > 0.0 and 0.0 < s_c < 1.0
            and v0 >= 0.0 and dt > 0.0 and 0.0 < c0 < 1.0):
        raise ValueError("invalid material, loading, integration or normalization parameters")
    ijk = fam['ijk']; ny = fam['ny']; notch = fam['notch']
    N = fam['X'].shape[0]
    if ny < 4:
        raise ValueError("need at least 4 rows for boundary layers and interior")
    jt = ijk[:, 1] == ny - 1
    jb = ijk[:, 1] == 0
    bc = jt | jb
    vbc = np.zeros((N, 3)); vbc[jt, 1] = v0; vbc[jb, 1] = -v0
    u = np.zeros((N, 3)); v = vbc.copy()
    # pre-notch: fully damaged initial bond state through the history variable (step 6)
    calY = np.where(notch, 1.0e30, 0.0)
    s = bond_phase_field(calY, Gc, fam['delta'], c0)
    # one force evaluation of the initial state initializes the acceleration (step 7)
    out = internal_force_evaluation(fam, u, calY, s, E, nu, Gc, s_c, c0)
    calY, s = out['calY'], out['s']
    a = out['B']/rho0; a[bc] = 0.0
    for step in range(n_steps):
        u = u + dt*v + 0.5*dt*dt*a
        u[bc] = vbc[bc]*(dt*(step + 1))
        out = internal_force_evaluation(fam, u, calY, s, E, nu, Gc, s_c, c0)
        calY, s = out['calY'], out['s']
        an = out['B']/rho0; an[bc] = 0.0
        v = v + 0.5*dt*(a + an); v[bc] = vbc[bc]
        a = an
    return {'u': u, 'v': v, 'a': a, 'calY': calY, 's': s}

import numpy as np


def pfpd_crack_dissipation(nx, ny, nz, dx, delta, E, nu, rho0, Gc, s_c,
                                   v0, dt, n_steps):
    if not all(isinstance(v, (int, np.integer)) and v >= 2 for v in (nx, ny, nz)):
        raise ValueError("each grid dimension must be an integer >= 2 (the moment matrix "
                         "is singular for a single-layer grid)")
    if not (isinstance(n_steps, (int, np.integer)) and n_steps >= 0):
        raise ValueError("n_steps must be a non-negative integer")
    if not (dx > 0.0 and delta > dx and dt > 0.0):
        raise ValueError("require dx > 0, delta > dx and dt > 0")
    if not (E > 0.0 and -1.0 < nu < 0.5 and rho0 > 0.0 and Gc > 0.0 and 0.0 < s_c < 1.0
            and v0 >= 0.0):
        raise ValueError("invalid material or loading parameters")
    if ny < 4:
        raise ValueError("need at least 4 rows for boundary layers and interior")
    # step 1: point cloud, families, kernel, truncation-corrected bond weights, bond masks
    fam = build_families(nx, ny, nz, dx, delta)
    # step 2: Griffith-consistency normalization constant of the cubic B-spline kernel
    c0 = pfpd_normalization_constant('cubic')
    # step 6: bond phase-field of the pre-notched initial state (history 1e30 on notch bonds)
    calY0 = np.where(fam['notch'], 1.0e30, 0.0)
    s0 = bond_phase_field(calY0, Gc, delta, c0)
    # step 8 (calling step 7, which calls steps 3-6 once per force evaluation): the dynamics
    state = velocity_verlet_run(fam, E, nu, rho0, Gc, s_c, v0, dt, n_steps, c0)
    s_T = state['s']
    # crack dissipation functional of the method: double sum over body and family of
    # omega_b * s, i.e. both directions of every unordered pair, scaled by Gc/(c0 delta)
    omb = fam['omega_b']; V = fam['V']
    eG_T = (Gc/(c0*delta))*2.0*float(np.sum(omb*s_T))*V*V
    eG_0 = (Gc/(c0*delta))*2.0*float(np.sum(omb*s0))*V*V
    if eG_T < eG_0 - 1.0e-12*max(abs(eG_0), 1.0):
        raise ValueError("crack dissipation decreased: irreversibility violated")
    return float(eG_T - eG_0)
SCICODE_GOLD_EOF
