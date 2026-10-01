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
import itertools


def plane_wave_shell_at_X(eps_shell: int, n_max: int) -> np.ndarray:
    if int(eps_shell) < 1:
        raise ValueError("eps_shell must be a positive integer")
    if int(n_max) < 1:
        raise ValueError("n_max must be a positive integer")
    out = []
    for n1, n2, n3 in itertools.product(range(-int(n_max), int(n_max) + 1), repeat=3):
        if (n1 % 2 != n2 % 2) or (n2 % 2 != n3 % 2):
            continue
        p = (n1, n2, n3 - 1)
        if p[0] ** 2 + p[1] ** 2 + p[2] ** 2 == int(eps_shell):
            out.append(p)
    return np.array(sorted(out), dtype=float).ravel()

import numpy as np


def wave_vector_group_at_X(axis: int) -> np.ndarray:
    import numpy as np
    if axis not in (0, 1, 2) or isinstance(axis, bool):
        raise ValueError("axis must be 0, 1 or 2")
    d2d = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],        # E
           [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],      # C2 about z
           [[1, 0, 0], [0, -1, 0], [0, 0, -1]],      # C2 about x
           [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],      # C2 about y
           [[0, -1, 0], [1, 0, 0], [0, 0, -1]],      # S4z
           [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],      # S4z inverse
           [[0, 1, 0], [1, 0, 0], [0, 0, 1]],        # diagonal mirror m1
           [[0, -1, 0], [-1, 0, 0], [0, 0, 1]]]      # diagonal mirror m2
    mz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=int)
    T = np.array([0.5, 0.5, 0.5])                    # a(1,1,1)/4 in units of pi
    Q = np.array([0.0, -1.0, -1.0])                  # a(0,-1,-1)/2 in units of pi
    rot, tra = [], []
    for qt in (np.zeros(3), Q):
        for gR, gt in ((np.eye(3, dtype=int), np.zeros(3)), (mz, T)):
            for d in d2d:
                rot.append((gR @ np.array(d, dtype=int)).astype(int))
                tra.append(gt + qt)
    rot = np.array(rot, dtype=int)
    tra = np.array(tra, dtype=float)
    # threefold rotation about [111] taking the z axis onto the requested axis
    C = np.linalg.matrix_power(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int), (axis + 1) % 3)
    rot = np.array([C @ r @ C.T for r in rot], dtype=int)
    tra = np.array([C @ t for t in tra], dtype=float)
    if len({(tuple(r.ravel().tolist()), tuple(np.round(t, 6).tolist()))
            for r, t in zip(rot, tra)}) != 32:
        raise ValueError("the assembled complex does not close into thirty-two distinct operations")
    order = sorted(range(32), key=lambda i: (tuple(rot[i].ravel().tolist()),
                                             tuple(np.round(tra[i], 6).tolist())))
    return np.concatenate([rot[order].ravel().astype(float), tra[order].ravel()])

import numpy as np


def _shell_from_flat(v):
    return np.rint(np.asarray(v, dtype=float)).astype(int).reshape(-1, 3)


def _group_from_flat(v):
    v = np.asarray(v, dtype=float).ravel()
    return np.rint(v[:288]).astype(int).reshape(32, 3, 3), v[288:384].reshape(32, 3) * np.pi


def _op_matrix(R, t, P, idx):
    n = len(P)
    M = np.zeros((n, n), dtype=complex)
    for j, p in enumerate(P):
        M[idx[tuple((np.asarray(R).T @ p).astype(int))], j] = np.exp(-1j * float(np.dot(p, t)))
    return M


def valleyor_basis(shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    P = _shell_from_flat(shell)
    n = len(P)
    if n != 8:
        raise ValueError("the X1 construction expects the eight-fold shell of step 01")
    rot, tra = _group_from_flat(group)
    idx = {tuple(p): i for i, p in enumerate(P)}
    keep = {((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            ((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (-1, 0, 0), (0, 0, 1))}
    S = np.zeros((n, n), dtype=complex)
    Qm = None
    for R, t in zip(rot, tra):
        zero_t = bool(np.allclose(t, 0.0))
        if tuple(map(tuple, R.tolist())) in keep and zero_t:
            S = S + _op_matrix(R, t, P, idx)
        if np.array_equal(R, np.eye(3, dtype=int)) and not zero_t:
            Qm = _op_matrix(R, t, P, idx)
    Pi = S @ (np.eye(n) - Qm)
    U, sv, _ = np.linalg.svd(Pi)
    space = U[:, : int((sv > 1e-9).sum())]
    rows = []
    for p3 in (-1, +1):
        ind = (P[:, 2] == p3).astype(float)
        v = space @ (space.conj().T @ ind)
        v = v / np.linalg.norm(v)
        first = int(np.argmax(np.abs(v) > 1e-9))
        rows.append(v * np.exp(-1j * np.angle(v[first])))
    z = np.array(rows).ravel()
    return np.concatenate([z.real, z.imag]).astype(float)

import numpy as np


def _shell_from_flat(v):
    return np.rint(np.asarray(v, dtype=float)).astype(int).reshape(-1, 3)


def _group_from_flat(v):
    v = np.asarray(v, dtype=float).ravel()
    return np.rint(v[:288]).astype(int).reshape(32, 3, 3), v[288:384].reshape(32, 3) * np.pi


def _unpack_complex(v, shape):
    v = np.asarray(v, dtype=float).ravel()
    h = v.size // 2
    return (v[:h] + 1j * v[h:]).reshape(shape)


def _op_matrix(R, t, P, idx):
    n = len(P)
    M = np.zeros((n, n), dtype=complex)
    for j, p in enumerate(P):
        M[idx[tuple((np.asarray(R).T @ p).astype(int))], j] = np.exp(-1j * float(np.dot(p, t)))
    return M


def valley_representation(basis: np.ndarray, shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    shell_arr = np.asarray(shell, dtype=float).ravel()
    group_arr = np.asarray(group, dtype=float).ravel()
    basis_arr = np.asarray(basis, dtype=float).ravel()
    if shell_arr.size % 3 or group_arr.size != 384 or basis_arr.size != 4 * (shell_arr.size // 3):
        raise ValueError("shell, group and basis must have matching, well-formed shapes")
    P = _shell_from_flat(shell)
    n = len(P)
    rot, tra = _group_from_flat(group)
    B = _unpack_complex(basis, (2, n))
    idx = {tuple(p): i for i, p in enumerate(P)}
    D = np.empty((32, 2, 2), dtype=complex)
    for g in range(32):
        D[g] = B.conj() @ (_op_matrix(rot[g], tra[g], P, idx) @ B.T)
    z = D.ravel()
    return np.concatenate([z.real, z.imag]).astype(float)

import numpy as np



def _shell_from_flat(v):
    return np.rint(np.asarray(v, dtype=float)).astype(int).reshape(-1, 3)


def _unpack_complex(v, shape):
    v = np.asarray(v, dtype=float).ravel()
    h = v.size // 2
    return (v[:h] + 1j * v[h:]).reshape(shape)


def tau_symmetry_data(rep: np.ndarray, basis: np.ndarray, shell: np.ndarray) -> np.ndarray:
    import numpy as np
    _TAU3 = [np.array([[0, 1], [1, 0]], dtype=complex),
             np.array([[0, -1j], [1j, 0]], dtype=complex),
             np.array([[1, 0], [0, -1]], dtype=complex)]
    D = _unpack_complex(rep, (32, 2, 2))
    P = _shell_from_flat(shell)
    n = len(P)
    B = _unpack_complex(basis, (2, n))
    signs = np.zeros((3, 32))
    for g in range(32):
        for j in range(3):
            Tj = D[g] @ _TAU3[j] @ D[g].conj().T
            s = float(np.trace(_TAU3[j].conj().T @ Tj).real) / 2.0
            if abs(abs(s) - 1.0) > 1e-6:
                raise ValueError("a valley Pauli matrix is not mapped onto plus or minus itself")
            signs[j, g] = round(s)
    idx = {tuple(p): i for i, p in enumerate(P)}
    Pi = np.zeros((n, n))
    for i, p in enumerate(P):
        Pi[idx[tuple(-p)], i] = 1.0
    U = B.conj() @ (Pi @ B.conj().T)
    tr = [round(float(np.trace(_TAU3[j].conj().T @ (U @ _TAU3[j].conj() @ U.conj().T)).real) / 2.0)
          for j in range(3)]
    return np.concatenate([signs.ravel(), np.array(tr, dtype=float)])

import numpy as np



def _group_from_flat(v):
    v = np.asarray(v, dtype=float).ravel()
    return np.rint(v[:288]).astype(int).reshape(32, 3, 3), v[288:384].reshape(32, 3) * np.pi


def _term_vector(P, B, E, Q, kxky, Fz):
    import numpy as np
    Px, Py, Pz = P
    Bx, By, Bz = B
    exx, eyy, ezz = E[0, 0], E[1, 1], E[2, 2]
    exy, exz, eyz = E[0, 1], E[0, 2], E[1, 2]
    Qxy, Qxx, Qyy = Q[0, 1], Q[0, 0], Q[1, 1]
    return np.array([
        exy, exx - eyy, ezz, exx + eyy, kxky,
        Px * By - Py * Bx, Px * Bx - Py * By, Px * By + Py * Bx, Px * Bx + Py * By,
        (Px * By - Py * Bx) * ezz, (Px * By - Py * Bx) * (exx + eyy),
        (Px * Bx - Py * By) * exy, (Px * By + Py * Bx) * (exx - eyy),
        Bz * (Px * By + Py * Bx), Bz * (Px * Bx + Py * By), Bz * (exx - eyy),
        Bx * exz + By * eyz, Bx * eyz + By * exz, Bz * exy,
        Qxy, (Px * Bx - Py * By) * Qxy, (Px * By + Py * Bx) * (Qxx - Qyy),
        Px * Py, Bx * By, Fz, Fz * exy, Px * eyz + Py * exz, Px * exz + Py * eyz,
        Bz * (Bx * Bx - By * By)], dtype=float)


def allowed_valley_channels(group: np.ndarray, tau_data: np.ndarray) -> np.ndarray:
    import numpy as np
    # time-reversal parity of each candidate combination, in index order
    _TERM_TR = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                         1, 1, -1, -1, -1, -1, 1, -1, -1,
                         1, 1, 1, 1, 1, 1, -1])
    if np.asarray(group, dtype=float).ravel().size != 384:
        raise ValueError("the group array must carry thirty-two operations")
    if np.asarray(tau_data, dtype=float).ravel().size != 99:
        raise ValueError("the tau data must carry 96 crystal signs and 3 parities")
    rot, _ = _group_from_flat(group)
    td = np.asarray(tau_data, dtype=float).ravel()
    signs = np.rint(td[:96]).astype(int).reshape(3, 32)
    trp = np.rint(td[96:99]).astype(int)
    sgn = np.vstack([np.ones(32, dtype=int), signs])
    trv = np.concatenate([[1], trp])
    nt = _TERM_TR.size
    ok = np.ones((nt, 4), dtype=int)
    rng = np.random.default_rng(20260831)
    for _ in range(8):
        Pv = rng.normal(size=3)
        Bv = rng.normal(size=3)
        kvec = rng.normal(size=3)
        Fv = np.array([0.0, 0.0, rng.normal()])
        E = rng.normal(size=(3, 3))
        E = E + E.T
        q = rng.normal(size=(2, 2))
        q = q + q.T
        q = q - np.trace(q) / 2.0 * np.eye(2)
        Q = np.zeros((3, 3))
        Q[:2, :2] = q
        base = _term_vector(Pv, Bv, E, Q, kvec[0] * kvec[1], Fv[2])
        for g in range(32):
            R = rot[g]
            d = int(round(float(np.linalg.det(R))))
            kg = R @ kvec
            Fg = R @ Fv
            got = _term_vector(R @ Pv, d * (R @ Bv), R @ E @ R.T, R @ Q @ R.T, kg[0] * kg[1], Fg[2])
            for j in range(4):
                ok[np.abs(got - sgn[j, g] * base) > 1e-9, j] = 0
    for j in range(4):
        ok[_TERM_TR * trv[j] != 1, j] = 0
    return ok.astype(float).ravel()

import numpy as np


def dot_geometry(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float, m_t_rel: float, a_ang: float, Bz_T: float) -> np.ndarray:
    import numpy as np
    if hw_maj_meV <= 0.0 or hw_min_meV <= 0.0:
        raise ValueError("confinement energies must be positive")
    if hw_maj_meV > hw_min_meV:
        raise ValueError("the major axis is the weakly confined one, so hw_maj cannot exceed hw_min")
    if m_t_rel <= 0.0 or a_ang <= 0.0:
        raise ValueError("effective mass and lattice constant must be positive")
    hbar2_2m0 = 3.8099821161548593      # eV A^2
    hbar_e_over_m0 = 0.11576763         # meV per tesla
    hm = 2.0e3 * hbar2_2m0 / m_t_rel    # hbar^2/m in meV A^2
    w0 = np.sqrt(hw_maj_meV * hw_min_meV)
    ell2 = hm / w0                       # squared oscillator length in A^2
    al = np.deg2rad(alpha_deg)
    R = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
    # dimensionless quadratic form: lengths in ell, momenta in 1/ell, energies in w0
    K = R @ np.diag([hw_maj_meV ** 2, hw_min_meV ** 2]) @ R.T / w0 ** 2
    b = hbar_e_over_m0 * Bz_T / m_t_rel / w0           # eB ell^2 / hbar
    # (x, y, pi_x, pi_y) = T (x, y, p_x, p_y) in the symmetric gauge
    T = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                  [0.0, -0.5 * b, 1.0, 0.0], [0.5 * b, 0.0, 0.0, 1.0]])
    Mpi = np.zeros((4, 4))
    Mpi[:2, :2] = K
    Mpi[2:, 2:] = np.eye(2)
    M = T.T @ Mpi @ T
    J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    w, V = np.linalg.eigh(M)
    Mh = V @ np.diag(np.sqrt(w)) @ V.T
    Mih = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    N = Mh @ J @ Mh
    wn, Vn = np.linalg.eigh(N.T @ N)
    absN = Vn @ np.diag(np.sqrt(np.clip(wn, 0.0, None))) @ Vn.T
    G = 0.5 * Mih @ absN @ Mih          # ground-state covariance, symmetrised
    Gpi = T @ G @ T.T
    kxky = Gpi[2, 3] / ell2 * a_ang ** 2
    nvec = np.array([np.cos(al), np.sin(al)])
    Q = np.outer(nvec, nvec) - 0.5 * np.eye(2)
    return np.array([kxky, Q[0, 1], Q[0, 0], Q[1, 1]], dtype=float)

import numpy as np


def _term_vector(P, B, E, Q, kxky, Fz):
    import numpy as np
    Px, Py, Pz = P
    Bx, By, Bz = B
    exx, eyy, ezz = E[0, 0], E[1, 1], E[2, 2]
    exy, exz, eyz = E[0, 1], E[0, 2], E[1, 2]
    Qxy, Qxx, Qyy = Q[0, 1], Q[0, 0], Q[1, 1]
    return np.array([
        exy, exx - eyy, ezz, exx + eyy, kxky,
        Px * By - Py * Bx, Px * Bx - Py * By, Px * By + Py * Bx, Px * Bx + Py * By,
        (Px * By - Py * Bx) * ezz, (Px * By - Py * Bx) * (exx + eyy),
        (Px * Bx - Py * By) * exy, (Px * By + Py * Bx) * (exx - eyy),
        Bz * (Px * By + Py * Bx), Bz * (Px * Bx + Py * By), Bz * (exx - eyy),
        Bx * exz + By * eyz, Bx * eyz + By * exz, Bz * exy,
        Qxy, (Px * Bx - Py * By) * Qxy, (Px * By + Py * Bx) * (Qxx - Qyy),
        Px * Py, Bx * By, Fz, Fz * exy, Px * eyz + Py * exz, Px * exz + Py * eyz,
        Bz * (Bx * Bx - By * By)], dtype=float)


def valley_magnetic_components(coeffs: np.ndarray, allowed: np.ndarray, P_vec: np.ndarray, B_vec: np.ndarray, Fz_MV_per_m: float, strain: np.ndarray, dot_geom: np.ndarray) -> np.ndarray:
    import numpy as np
    c = np.asarray(coeffs, dtype=float).ravel()
    if c.size != 29:
        raise ValueError("coeffs must hold exactly one constant per candidate term")
    A = np.rint(np.asarray(allowed, dtype=float)).astype(int).reshape(29, 4)
    E = np.asarray(strain, dtype=float).reshape(3, 3)
    g = np.asarray(dot_geom, dtype=float).ravel()
    Q = np.array([[g[2], g[1], 0.0], [g[1], g[3], 0.0], [0.0, 0.0, 0.0]])
    X = _term_vector(np.asarray(P_vec, dtype=float), np.asarray(B_vec, dtype=float), E, Q, g[0],
                     float(Fz_MV_per_m))
    return np.array([float(c @ (A[:, j] * X)) for j in (1, 2, 3)], dtype=float)

import numpy as np


def _ncx2_cdf_two_dof(x, lam):
    import numpy as np
    if x <= 0.0:
        return 0.0
    half_x, half_lam = 0.5 * float(x), 0.5 * float(lam)
    total, poisson, inner, term = 0.0, np.exp(-half_lam), 0.0, 1.0
    for j in range(4000):
        inner += term
        total += poisson * (1.0 - np.exp(-half_x) * inner)
        if j > 8 and poisson < 1e-18 and j > 4.0 * half_lam:
            break
        term *= half_x / (j + 1.0)
        poisson *= half_lam / (j + 1.0)
    return float(total)


def wafer_valley_statistics(coeffs: np.ndarray, allowed: np.ndarray, tau_parities: np.ndarray, P_vec: np.ndarray, B_vec: np.ndarray, Fz_MV_per_m: float, strain: np.ndarray, dot_geom: np.ndarray, sigma_ge_ueV: float, sigma_P: float, e_threshold_ueV: float) -> np.ndarray:
    import numpy as np
    from scipy import integrate
    sigma = float(sigma_ge_ueV)
    sP = float(sigma_P)
    e_th = float(e_threshold_ueV)
    if sigma <= 0.0:
        raise ValueError("the germanium standard deviation must be positive")
    if sP < 0.0:
        raise ValueError("the polar-vector standard deviation cannot be negative")
    if e_th <= 0.0:
        raise ValueError("the splitting threshold must be positive")
    parity = np.rint(np.asarray(tau_parities, dtype=float).ravel()).astype(int)
    odd = [j for j in range(3) if parity[j] == -1]
    if len(odd) != 1:
        raise ValueError("expected exactly one time-reversal-odd valley operator")
    k = odd[0]
    pl = [j for j in range(3) if j != k]
    P0 = np.asarray(P_vec, dtype=float).ravel().copy()

    def _comps(dx, dy):
        P = P0.copy()
        P[0] += dx
        P[1] += dy
        return valley_magnetic_components(coeffs, allowed, P, B_vec, Fz_MV_per_m,
                                                  strain, dot_geom)

    v0 = _comps(0.0, 0.0)
    vxp, vxm = _comps(1.0, 0.0), _comps(-1.0, 0.0)
    vyp, vym = _comps(0.0, 1.0), _comps(0.0, -1.0)
    vpp, vpm, vmp, vmm = _comps(1.0, 1.0), _comps(1.0, -1.0), _comps(-1.0, 1.0), _comps(-1.0, -1.0)
    Jx, Jy = 0.5 * (vxp - vxm), 0.5 * (vyp - vym)
    Hxx, Hyy = vxp + vxm - 2.0 * v0, vyp + vym - 2.0 * v0
    Hxy = 0.25 * (vpp - vpm - vmp + vmm)
    scale = max(1.0, float(np.max(np.abs(np.concatenate([v0, Jx, Jy])))))
    if max(abs(Hxx[k]), abs(Hyy[k]), abs(Hxy[k])) > 1e-9 * scale:
        raise ValueError("the protected component must be affine in the polar vector")
    var_k = sP ** 2 * (Jx[k] ** 2 + Jy[k] ** 2)
    std_k = float(np.sqrt(var_k))

    R2 = 0.25 * e_th ** 2

    def _cond(v):
        r2 = R2 - v[k] ** 2
        if r2 <= 0.0:
            return 0.0
        return _ncx2_cdf_two_dof(r2 / sigma ** 2, (v[pl[0]] ** 2 + v[pl[1]] ** 2) / sigma ** 2)

    fixed = _cond(v0)
    if sP == 0.0:
        return np.array([std_k, 100.0 * fixed, 100.0 * fixed], dtype=float)

    gn = float(np.hypot(Jx[k], Jy[k]))
    L = 12.0
    if gn > 0.0:
        ex = np.array([Jx[k], Jy[k]]) / gn
    else:
        ex = np.array([1.0, 0.0])
    ey = np.array([-ex[1], ex[0]])
    if gn > 0.0:
        u_lo = max(-L, (-np.sqrt(R2) - v0[k]) / (gn * sP))
        u_hi = min(L, (np.sqrt(R2) - v0[k]) / (gn * sP))
    else:
        u_lo, u_hi = (-L, L) if v0[k] ** 2 < R2 else (0.0, 0.0)
    if u_hi <= u_lo:
        return np.array([std_k, 100.0 * fixed, 0.0], dtype=float)
    norm = 1.0 / (2.0 * np.pi)

    def _inner(w, u):
        d = sP * (u * ex + w * ey)
        return norm * np.exp(-0.5 * (u * u + w * w)) * _cond(_comps(d[0], d[1]))

    val, _ = integrate.dblquad(_inner, u_lo, u_hi, -L, L, epsabs=1e-13, epsrel=1e-11)
    return np.array([std_k, 100.0 * fixed, 100.0 * val], dtype=float)

import numpy as np


def valley_low_tail_percent(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float, m_t_rel: float, a_ang: float, B_inplane_T: float, theta_B_deg: float, Bz_T: float, P0: float, theta_P_deg: float, Fz_MV_per_m: float, strain: np.ndarray, coeffs: np.ndarray, sigma_ge_ueV: float, sigma_P: float, e_threshold_ueV: float) -> float:
    import numpy as np
    shell = plane_wave_shell_at_X(5, 4)
    if shell.size != 24:
        raise ValueError("the conduction-band shell at X must carry eight plane waves")
    group = wave_vector_group_at_X(2)
    basis = valleyor_basis(shell, group)
    rep = valley_representation(basis, shell, group)
    tau = tau_symmetry_data(rep, basis, shell)
    allowed = allowed_valley_channels(group, tau)
    geom = dot_geometry(hw_maj_meV, hw_min_meV, alpha_deg, m_t_rel, a_ang, Bz_T)
    tb = np.deg2rad(theta_B_deg)
    tp = np.deg2rad(theta_P_deg)
    B_vec = np.array([B_inplane_T * np.cos(tb), B_inplane_T * np.sin(tb), Bz_T])
    P_vec = np.array([P0 * np.cos(tp), P0 * np.sin(tp), 0.0])
    parity = np.rint(np.asarray(tau, dtype=float).ravel()[96:99]).astype(int)
    stats = wafer_valley_statistics(coeffs, allowed, parity, P_vec, B_vec, Fz_MV_per_m,
                                            strain, geom, sigma_ge_ueV, sigma_P, e_threshold_ueV)
    return float(stats[2])
SCICODE_GOLD_EOF
