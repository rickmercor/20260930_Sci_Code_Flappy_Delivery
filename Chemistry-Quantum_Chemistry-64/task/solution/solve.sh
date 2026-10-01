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
from scipy.special import erf


def _boys0(t):
    """Zeroth-order Boys function, elementwise, with the t -> 0 series."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = t < 1e-10
    out[small] = 1.0 - t[small] / 3.0
    big = ~small
    out[big] = 0.5 * np.sqrt(np.pi / t[big]) * erf(np.sqrt(t[big]))
    return out


def _contracted_shells(coords, exponents, coefficients):
    """List of (centre, exponents, normalised coefficients) per basis function."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 1:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite")
    if len(exponents) == 0 or len(exponents) != len(coefficients):
        raise ValueError("exponents and coefficients must be non-empty lists of equal length")
    shells = []
    for exps, coefs in zip(exponents, coefficients):
        a = np.asarray(exps, dtype=float).ravel()
        c = np.asarray(coefs, dtype=float).ravel()
        if a.size == 0 or a.size != c.size:
            raise ValueError("each shell needs matching, non-empty exponent and coefficient arrays")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))) or np.any(a <= 0.0):
            raise ValueError("exponents must be positive and all values finite")
        d = c * (2.0 * a / np.pi) ** 0.75
        pab = a[:, None] + a[None, :]
        self_overlap = float(np.sum(d[:, None] * d[None, :] * (np.pi / pab) ** 1.5))
        if self_overlap <= 0.0:
            raise ValueError("a contracted shell has non-positive self-overlap")
        shells.append((a, d / np.sqrt(self_overlap)))
    basis = []
    for centre in coords:
        for a, d in shells:
            basis.append((centre, a, d))
    return basis


def s_type_one_electron_integrals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    basis = _contracted_shells(coords, exponents, coefficients)
    coords = np.asarray(coords, dtype=float)
    charges = np.asarray(charges, dtype=float)
    if charges.shape != (coords.shape[0],) or not np.all(np.isfinite(charges)) or np.any(charges < 0.0):
        raise ValueError("charges must be a finite, non-negative array of shape (n_atoms,)")
    n = len(basis)
    out = np.zeros((4, n, n))
    for i, (A, a, da) in enumerate(basis):
        for j, (B, b, db) in enumerate(basis):
            p = a[:, None] + b[None, :]
            mu = a[:, None] * b[None, :] / p
            ab2 = float(np.sum((A - B) ** 2))
            prim_s = (np.pi / p) ** 1.5 * np.exp(-mu * ab2)
            dd = da[:, None] * db[None, :]
            P = (a[:, None, None] * A + b[None, :, None] * B) / p[:, :, None]
            out[0, i, j] = np.sum(dd * prim_s)
            out[1, i, j] = np.sum(dd * mu * (3.0 - 2.0 * mu * ab2) * prim_s)
            v = 0.0
            for C, Z in zip(coords, charges):
                pc2 = np.sum((P - C) ** 2, axis=2)
                v -= Z * np.sum(dd * (2.0 * np.pi / p) * np.exp(-mu * ab2) * _boys0(p * pc2))
            out[2, i, j] = v
            out[3, i, j] = np.sum(dd * prim_s * P[:, :, 2])
    return out

import numpy as np


def s_type_electron_repulsion(coords: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    basis = _contracted_shells(coords, exponents, coefficients)
    n = len(basis)
    pairs = []
    for i in range(n):
        A, a, da = basis[i]
        for j in range(i + 1):
            B, b, db = basis[j]
            p = a[:, None] + b[None, :]
            k_ab = np.exp(-a[:, None] * b[None, :] / p * float(np.sum((A - B) ** 2)))
            P = (a[:, None, None] * A + b[None, :, None] * B) / p[:, :, None]
            pairs.append((i, j, p.ravel(), P.reshape(-1, 3), (da[:, None] * db[None, :] * k_ab).ravel()))
    eri = np.zeros((n, n, n, n))
    for x, (i, j, p, P, w1) in enumerate(pairs):
        for (k, l, q, Q, w2) in pairs[: x + 1]:
            pq = p[:, None] * q[None, :]
            ps = p[:, None] + q[None, :]
            pq2 = np.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=2)
            val = float(np.sum(w1[:, None] * w2[None, :] * 2.0 * np.pi ** 2.5
                               / (pq * np.sqrt(ps)) * _boys0(pq / ps * pq2)))
            for (r, s, t, u) in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                 (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                eri[r, s, t, u] = val
    return eri

import numpy as np


def _rhf_solve(S, H, eri, n_occ):
    """Converged closed-shell RHF: (electronic energy, orbital energies, C)."""
    n = S.shape[0]
    s_val, s_vec = np.linalg.eigh(S)
    if s_val[0] <= 1e-10:
        raise ValueError("overlap matrix is not positive definite")
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T

    def _fock(D):
        J = np.einsum('mnls,ls->mn', eri, D)
        K = np.einsum('mlns,ls->mn', eri, D)
        return H + J - 0.5 * K

    _, cp = np.linalg.eigh(X.T @ H @ X)
    C = X @ cp
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    e_old = None
    focks, errs = [], []
    converged = False
    for _ in range(1000):
        F = _fock(D)
        focks.append(F)
        errs.append(F @ D @ S - S @ D @ F)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        B = -np.ones((m + 1, m + 1))
        B[m, m] = 0.0
        for x in range(m):
            for y in range(m):
                B[x, y] = np.sum(errs[x] * errs[y])
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            w = np.linalg.solve(B, rhs)
            F_use = sum(w[k] * focks[k] for k in range(m))
        except np.linalg.LinAlgError:
            F_use = F
        _, cp = np.linalg.eigh(X.T @ F_use @ X)
        C = X @ cp
        D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        e_new = 0.5 * float(np.sum(D_new * (H + _fock(D_new))))
        if e_old is not None and np.max(np.abs(D_new - D)) < 1e-10 and abs(e_new - e_old) < 1e-12:
            D = D_new
            converged = True
            break
        D = D_new
        e_old = e_new
    if not converged:
        raise ValueError("RHF did not converge in 1000 iterations")
    F = _fock(D)
    eps, cp = np.linalg.eigh(X.T @ F @ X)
    C = X @ cp
    for k in range(n):
        lead = C[np.abs(C[:, k]) > 1e-8, k]
        if lead.size and lead[0] < 0.0:
            C[:, k] = -C[:, k]
    e_elec = 0.5 * float(np.sum(D * (H + F)))
    return e_elec, eps, C


def rhf_canonical_orbitals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list, n_occ: int) -> np.ndarray:
    ints = s_type_one_electron_integrals(coords, charges, exponents, coefficients)
    eri = s_type_electron_repulsion(coords, exponents, coefficients)
    n_bf = ints.shape[1]
    if isinstance(n_occ, bool) or not isinstance(n_occ, (int, np.integer)) or not 1 <= int(n_occ) <= n_bf:
        raise ValueError("n_occ must be an integer between 1 and n_bf")
    _, eps, C = _rhf_solve(ints[0], ints[1] + ints[2], eri, int(n_occ))
    return np.vstack([eps[None, :], C])

import numpy as np


def _cc_blocks(f, g, o):
    O, V = slice(0, o), slice(o, f.shape[0])
    return dict(foo=f[O, O], fov=f[O, V], fvv=f[V, V],
                oooo=g[O, O, O, O], ooov=g[O, O, O, V], oovo=g[O, O, V, O], oovv=g[O, O, V, V],
                ovov=g[O, V, O, V], ovvo=g[O, V, V, O], ovoo=g[O, V, O, O], ovvv=g[O, V, V, V],
                vovv=g[V, O, V, V], vvvo=g[V, V, V, O], vvvv=g[V, V, V, V])


def _cc_ein(s, *a):
    return np.einsum(s, *a, optimize=True)


def _cc_taus(t1, t2):
    tt = _cc_ein('ia,jb->ijab', t1, t1)
    tt = tt - tt.transpose(0, 1, 3, 2)
    return t2 + tt, t2 + 0.5 * tt


def _cc_pij(x):
    return x - x.transpose(1, 0, 2, 3)


def _cc_pab(x):
    return x - x.transpose(0, 1, 3, 2)


def _cc_t_residuals(f, g, o, t1, t2):
    """<Phi_mu| exp(-T) H exp(T) |Phi0> for singles and doubles (general, non-canonical f)."""
    B = _cc_blocks(f, g, o)
    e = _cc_ein
    tau, taut = _cc_taus(t1, t2)
    Fae = B['fvv'] - 0.5 * e('me,ma->ae', B['fov'], t1) + e('mf,mafe->ae', t1, B['ovvv']) - 0.5 * e('mnaf,mnef->ae', taut, B['oovv'])
    Fmi = B['foo'] + 0.5 * e('ie,me->mi', t1, B['fov']) + e('ne,mnie->mi', t1, B['ooov']) + 0.5 * e('inef,mnef->mi', taut, B['oovv'])
    Fme = B['fov'] + e('nf,mnef->me', t1, B['oovv'])
    x = e('je,mnie->mnij', t1, B['ooov'])
    Wmnij = B['oooo'] + x - x.transpose(0, 1, 3, 2) + 0.25 * e('ijef,mnef->mnij', tau, B['oovv'])
    x = e('mb,amef->abef', t1, B['vovv'])
    Wabef = B['vvvv'] - x + x.transpose(1, 0, 2, 3) + 0.25 * e('mnab,mnef->abef', tau, B['oovv'])
    Wmbej = (B['ovvo'] + e('jf,mbef->mbej', t1, B['ovvv']) - e('nb,mnej->mbej', t1, B['oovo'])
             - e('jnfb,mnef->mbej', 0.5 * t2 + e('jf,nb->jnfb', t1, t1), B['oovv']))
    R1 = (B['fov'] + e('ie,ae->ia', t1, Fae) - e('ma,mi->ia', t1, Fmi) + e('imae,me->ia', t2, Fme)
          - e('nf,naif->ia', t1, B['ovov']) - 0.5 * e('imef,maef->ia', t2, B['ovvv'])
          - 0.5 * e('mnae,nmei->ia', t2, B['oovo']))
    R2 = B['oovv'] + _cc_pab(e('ijae,be->ijab', t2, Fae - 0.5 * e('mb,me->be', t1, Fme)))
    R2 = R2 - _cc_pij(e('imab,mj->ijab', t2, Fmi + 0.5 * e('je,me->mj', t1, Fme)))
    R2 = R2 + 0.5 * e('mnab,mnij->ijab', tau, Wmnij) + 0.5 * e('ijef,abef->ijab', tau, Wabef)
    R2 = R2 + _cc_pij(_cc_pab(e('imae,mbej->ijab', t2, Wmbej) - e('ie,ma,mbej->ijab', t1, t1, B['ovvo'])))
    R2 = R2 + _cc_pij(e('ie,abej->ijab', t1, B['vvvo'])) - _cc_pab(e('ma,mbij->ijab', t1, B['ovoo']))
    return R1, R2


def _cc_l_residuals(f, g, o, t1, t2, l1, l2):
    """<Phi0|(1 + Lambda)[exp(-T) H exp(T), X_mu]|Phi0> for singles and doubles, exact for any T."""
    B = _cc_blocks(f, g, o)
    e = _cc_ein
    tau, taut = _cc_taus(t1, t2)
    Fme = B['fov'] + e('nf,mnef->me', t1, B['oovv'])
    Fae = B['fvv'] - 0.5 * e('me,ma->ae', B['fov'], t1) + e('mf,mafe->ae', t1, B['ovvv']) - 0.5 * e('mnaf,mnef->ae', taut, B['oovv'])
    Fmi = B['foo'] + 0.5 * e('ie,me->mi', t1, B['fov']) + e('ne,mnie->mi', t1, B['ooov']) + 0.5 * e('inef,mnef->mi', taut, B['oovv'])
    Hvv = Fae - 0.5 * e('ma,me->ae', t1, Fme)
    Hoo = Fmi + 0.5 * e('ie,me->mi', t1, Fme)
    x = e('je,mnie->mnij', t1, B['ooov'])
    Hoooo = B['oooo'] + x - x.transpose(0, 1, 3, 2) + 0.5 * e('ijef,mnef->mnij', tau, B['oovv'])
    x = e('mb,amef->abef', t1, B['vovv'])
    Hvvvv = B['vvvv'] - x + x.transpose(1, 0, 2, 3) + 0.5 * e('mnab,mnef->abef', tau, B['oovv'])
    Hooov = B['ooov'] + e('if,mnfe->mnie', t1, B['oovv'])
    Hvovv = B['vovv'] - e('na,nmef->amef', t1, B['oovv'])
    Hovvo = (B['ovvo'] + e('jf,mbef->mbej', t1, B['ovvv']) - e('nb,mnej->mbej', t1, B['oovo'])
             - e('jnfb,mnef->mbej', t2 + e('jf,nb->jnfb', t1, t1), B['oovv']))
    Z = B['ovvo'] - e('njbf,mnef->mbej', t2, B['oovv'])
    x = e('mnie,jnbe->mbij', B['ooov'], t2) + e('ie,mbej->mbij', t1, Z)
    Hovoo = (B['ovoo'] - e('me,ijbe->mbij', Fme, t2) - e('nb,mnij->mbij', t1, Hoooo)
             + 0.5 * e('mbef,ijef->mbij', B['ovvv'], tau) + x - x.transpose(0, 1, 3, 2))
    Z = B['ovvo'] - e('nibf,mnef->mbei', t2, B['oovv'])
    x = e('mbef,miaf->abei', B['ovvv'], t2) + e('ma,mbei->abei', t1, Z)
    Hvvvo = (B['vvvo'] - e('me,miab->abei', Fme, t2) + e('if,abef->abei', t1, Hvvvv)
             + 0.5 * e('mnei,mnab->abei', B['oovo'], tau) - x + x.transpose(1, 0, 2, 3))
    Gvv = -0.5 * e('mnef,mnaf->ae', t2, l2)
    Goo = 0.5 * e('mnef,inef->mi', t2, l2)
    G1 = (Fme + e('ie,ea->ia', l1, Hvv) - e('ma,im->ia', l1, Hoo) + e('me,ieam->ia', l1, Hovvo)
          + 0.5 * e('imef,efam->ia', l2, Hvvvo) - 0.5 * e('mnae,iemn->ia', l2, Hovoo)
          - e('ef,eifa->ia', Gvv, Hvovv) - e('mn,mina->ia', Goo, Hooov))
    G2 = B['oovv'] + _cc_pab(e('ijae,eb->ijab', l2, Hvv)) - _cc_pij(e('imab,jm->ijab', l2, Hoo))
    G2 = G2 + 0.5 * e('mnab,ijmn->ijab', l2, Hoooo) + 0.5 * e('ijef,efab->ijab', l2, Hvvvv)
    G2 = G2 + _cc_pij(e('ie,ejab->ijab', l1, Hvovv)) - _cc_pab(e('ma,ijmb->ijab', l1, Hooov))
    G2 = G2 + _cc_pij(_cc_pab(e('ia,jb->ijab', l1, Fme))) + _cc_pij(_cc_pab(e('imae,jebm->ijab', l2, Hovvo)))
    G2 = G2 + _cc_pab(e('ijae,be->ijab', B['oovv'], Gvv)) - _cc_pij(e('imab,mj->ijab', B['oovv'], Goo))
    return G1, G2


def _cc_check_hamiltonian(fock, eri_as, n_electrons):
    """Validate the spin-orbital Hamiltonian; return float arrays and (o, v)."""
    f = np.asarray(fock, dtype=float)
    g = np.asarray(eri_as, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or f.shape[0] < 2:
        raise ValueError("fock must be a square matrix of size at least 2")
    n = f.shape[0]
    if not np.all(np.isfinite(f)) or not np.allclose(f, f.T, atol=1e-10):
        raise ValueError("fock must be finite and symmetric")
    if g.shape != (n, n, n, n) or not np.all(np.isfinite(g)):
        raise ValueError("eri_as must be a finite (n, n, n, n) array")
    if (not np.allclose(g, -g.transpose(1, 0, 2, 3), atol=1e-10) or not np.allclose(g, -g.transpose(0, 1, 3, 2), atol=1e-10)
            or not np.allclose(g, g.transpose(2, 3, 0, 1), atol=1e-10)):
        raise ValueError("eri_as must be antisymmetric in each index pair and symmetric under pair exchange")
    if isinstance(n_electrons, bool) or not isinstance(n_electrons, (int, np.integer)) or not 1 <= int(n_electrons) <= n - 1:
        raise ValueError("n_electrons must be an integer between 1 and n - 1")
    o = int(n_electrons)
    return f, g, o, n - o


def _cc_denominators(f, o):
    d = np.diag(f)
    D1 = d[:o, None] - d[None, o:]
    D2 = d[:o, None, None, None] + d[None, :o, None, None] - d[None, None, o:, None] - d[None, None, None, o:]
    return D1, D2


def _cc_pack(t1, t2, l1, l2):
    return np.concatenate([t1.ravel(), t2.ravel(), l1.ravel(), l2.ravel()])


def _cc_unpack(y, o, v):
    n1, n2 = o * v, o * o * v * v
    y = np.asarray(y)
    if y.ndim != 1 or y.size != 2 * (n1 + n2):
        raise ValueError("amplitude vector has the wrong length for this spin-orbital space")
    return (y[:n1].reshape(o, v), y[n1:n1 + n2].reshape(o, o, v, v),
            y[n1 + n2:2 * n1 + n2].reshape(o, v), y[2 * n1 + n2:].reshape(o, o, v, v))


def ccsd_lambda_ground_state(fock: np.ndarray, eri_as: np.ndarray, n_electrons: int) -> np.ndarray:
    f, g, o, v = _cc_check_hamiltonian(fock, eri_as, n_electrons)
    D1, D2 = _cc_denominators(f, o)
    pairs = (~np.eye(o, dtype=bool))[:, :, None, None] & (~np.eye(v, dtype=bool))[None, None, :, :]
    if np.any(np.abs(D1) < 1e-8) or np.any(np.abs(D2[pairs]) < 1e-8):
        raise ValueError("a Jacobi denominator vanishes")
    D2s = np.where(np.abs(D2) < 1e-8, 1.0, D2)
    t1 = np.zeros((o, v))
    t2 = _cc_blocks(f, g, o)['oovv'] / D2s
    for _ in range(500):
        R1, R2 = _cc_t_residuals(f, g, o, t1, t2)
        if max(np.max(np.abs(R1)), np.max(np.abs(R2))) < 1e-11:
            break
        t1 = t1 + R1 / D1
        t2 = t2 + R2 / D2s
    else:
        raise ValueError("the cluster amplitude equations did not converge")
    l1, l2 = t1.copy(), t2.copy()
    for _ in range(500):
        G1, G2 = _cc_l_residuals(f, g, o, t1, t2, l1, l2)
        if max(np.max(np.abs(G1)), np.max(np.abs(G2))) < 1e-11:
            break
        l1 = l1 + G1 / D1
        l2 = l2 + G2 / D2s
    else:
        raise ValueError("the de-excitation amplitude equations did not converge")
    return _cc_pack(t1, t2, l1, l2)

import numpy as np


def _td_check_inputs(fock, dipole, eri_as, n_electrons, amplitudes):
    f, g, o, v = _cc_check_hamiltonian(fock, eri_as, n_electrons)
    z = np.asarray(dipole, dtype=float)
    if z.shape != f.shape or not np.all(np.isfinite(z)) or not np.allclose(z, z.T, atol=1e-10):
        raise ValueError("dipole must be a finite symmetric matrix of the Fock shape")
    y = np.asarray(amplitudes, dtype=complex)
    if y.ndim != 1 or y.size != 2 * (o * v + o * o * v * v) or not np.all(np.isfinite(y)):
        raise ValueError("amplitudes must be a finite 1D vector of length 2 (o v + o^2 v^2)")
    return f, z, g, o, v, y


def _td_derivative(f, z, g, o, v, y, field):
    t1, t2, l1, l2 = _cc_unpack(y, o, v)
    ft = f + field * z
    R1, R2 = _cc_t_residuals(ft, g, o, t1, t2)
    G1, G2 = _cc_l_residuals(ft, g, o, t1, t2, l1, l2)
    return _cc_pack(-1j * R1, -1j * R2, 1j * G1, 1j * G2)


def tdccsd_time_derivative(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, field: float) -> np.ndarray:
    f, z, g, o, v, y = _td_check_inputs(fock, dipole, eri_as, n_electrons, amplitudes)
    if isinstance(field, bool) or not isinstance(field, (int, float, np.integer, np.floating)) or not np.isfinite(field):
        raise ValueError("field must be a finite real number")
    dy = _td_derivative(f, z, g, o, v, y, float(field))
    return np.concatenate([dy.real, dy.imag])

import numpy as np


def _pulse_field(t, e0, omega, t_center, t_foot, n_env):
    x = t - t_center
    if abs(x) > 0.5 * t_foot:
        return 0.0
    return e0 * np.cos(omega * x) * np.cos(np.pi * x / t_foot) ** n_env


def tdccsd_propagate(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, e0: float, omega: float, t_center: float, t_foot: float, n_env: int, t_start: float, dt: float, n_steps: int) -> np.ndarray:
    f, z, g, o, v, y = _td_check_inputs(fock, dipole, eri_as, n_electrons, amplitudes)
    for name, val in (('e0', e0), ('omega', omega), ('t_center', t_center), ('t_start', t_start), ('t_foot', t_foot), ('dt', dt)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError("%s must be a finite real number" % name)
    if omega < 0.0:
        raise ValueError("omega must be non-negative")
    if t_foot <= 0.0 or dt <= 0.0:
        raise ValueError("t_foot and dt must be positive")
    if isinstance(n_env, bool) or not isinstance(n_env, (int, np.integer)) or n_env < 1:
        raise ValueError("n_env must be a positive integer")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    E = lambda t: _pulse_field(t, float(e0), float(omega), float(t_center), float(t_foot), int(n_env))
    F = lambda t, yy: _td_derivative(f, z, g, o, v, yy, E(t))
    t_start = float(t_start)
    dt = float(dt)
    for k in range(int(n_steps)):
        tk = t_start + k * dt
        k1 = F(tk, y)
        k2 = F(tk + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = F(tk + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = F(tk + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.concatenate([y.real, y.imag])

import numpy as np


def _cw_coefficients(t1, t2, l1, l2):
    """Ket coefficients (c1, c2) and bra coefficients (~c0, ~c1, ~c2); the ket reference coefficient is one."""
    tt = np.einsum('ia,jb->ijab', t1, t1)
    c1 = t1
    c2 = t2 + tt - tt.transpose(0, 1, 3, 2)
    ct0 = (1.0 - np.einsum('ia,ia->', l1, t1) - 0.25 * np.einsum('ijab,ijab->', l2, t2)
           + 0.5 * np.einsum('ijab,ia,jb->', l2, t1, t1))
    ct1 = l1 - np.einsum('ijab,jb->ia', l2, t1)
    return c1, c2, ct0, ct1, l2


def _cw_weights(t1, t2, l1, l2):
    o, v = t1.shape
    c1, c2, ct0, ct1, ct2 = _cw_coefficients(t1, t2, l1, l2)
    occ_pairs = [(i, j) for i in range(o) for j in range(i + 1, o)]
    vir_pairs = [(a, b) for a in range(v) for b in range(a + 1, v)]
    w2 = [np.real(ct2[i, j, a, b] * c2[i, j, a, b]) for (i, j) in occ_pairs for (a, b) in vir_pairs]
    return np.concatenate([[np.real(ct0)], np.real(ct1 * c1).ravel(), np.asarray(w2, dtype=float)])


def configuration_weights(amplitudes: np.ndarray, n_electrons: int, n_spin_orbitals: int) -> np.ndarray:
    if isinstance(n_spin_orbitals, bool) or not isinstance(n_spin_orbitals, (int, np.integer)) or n_spin_orbitals < 2:
        raise ValueError("n_spin_orbitals must be an integer of at least 2")
    if isinstance(n_electrons, bool) or not isinstance(n_electrons, (int, np.integer)) or not 1 <= n_electrons <= n_spin_orbitals - 1:
        raise ValueError("n_electrons must be an integer between 1 and n_spin_orbitals - 1")
    o, v = int(n_electrons), int(n_spin_orbitals) - int(n_electrons)
    y = np.asarray(amplitudes, dtype=complex)
    if y.ndim != 1 or y.size != 2 * (o * v + o * o * v * v) or not np.all(np.isfinite(y)):
        raise ValueError("amplitudes must be a finite 1D vector of length 2 (o v + o^2 v^2)")
    t1, t2, l1, l2 = _cc_unpack(y, o, v)
    for x in (t2, l2):
        if np.max(np.abs(x + x.transpose(1, 0, 2, 3)), initial=0.0) > 1e-10 or np.max(np.abs(x + x.transpose(0, 1, 3, 2)), initial=0.0) > 1e-10:
            raise ValueError("doubles amplitudes must be antisymmetric in (i, j) and in (a, b)")
    return _cw_weights(t1, t2, l1, l2)

import numpy as np


def _spin_orbital_operators(C, hcore, zmat, eri, n_occ):
    """Spin-orbital Fock matrix, z matrix and <pq||rs> in the canonical orbitals (interleaved spins)."""
    n = C.shape[1]
    h_mo = C.T @ hcore @ C
    z_mo = C.T @ zmat @ C
    e_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, C, C, C, C, optimize=True)
    nso = 2 * n
    sp = np.arange(nso) // 2
    same = (np.arange(nso)[:, None] % 2 == np.arange(nso)[None, :] % 2).astype(float)
    h = h_mo[np.ix_(sp, sp)] * same
    z = z_mo[np.ix_(sp, sp)] * same
    phys = e_mo[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3) * same[:, None, :, None] * same[None, :, None, :]
    g = phys - phys.transpose(0, 1, 3, 2)
    ne = 2 * n_occ
    f = h + np.einsum('piqi->pq', g[:, :ne, :, :ne])
    return f, z, g


def dark_weight_change(n_atoms: int, spacing: float, exponents: list, coefficients: list, e0: float, omega: float, t_foot: float, n_env: int, dt: float, occ: int, vir: int) -> float:
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, (int, np.integer)) or n_atoms < 2 or n_atoms % 2:
        raise ValueError("n_atoms must be an even integer of at least 2")
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float, np.integer, np.floating)) or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be a finite positive number")
    for name, val in (('t_foot', t_foot), ('dt', dt)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val) or val <= 0.0:
            raise ValueError("%s must be a finite positive number" % name)
    ratio = float(t_foot) / float(dt)
    n_steps = int(round(ratio))
    if n_steps < 1 or abs(ratio - n_steps) > 1e-9 * max(1.0, ratio):
        raise ValueError("t_foot / dt must be a positive integer")
    n_occ = int(n_atoms) // 2
    zk = (np.arange(n_atoms) - 0.5 * (n_atoms - 1)) * float(spacing)
    coords = np.stack([np.zeros(n_atoms), np.zeros(n_atoms), zk], axis=1)
    charges = np.ones(n_atoms)
    ints = s_type_one_electron_integrals(coords, charges, exponents, coefficients)
    eri = s_type_electron_repulsion(coords, exponents, coefficients)
    orbitals = rhf_canonical_orbitals(coords, charges, exponents, coefficients, n_occ)
    n_bf = orbitals.shape[1]
    for name, val in (('occ', occ), ('vir', vir)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError("%s must be an integer" % name)
    if not 0 <= occ < n_occ or not n_occ <= vir < n_bf:
        raise ValueError("occ must be occupied and vir virtual in the reference")
    f, z, g = _spin_orbital_operators(orbitals[1:], ints[1] + ints[2], ints[3], eri, n_occ)
    ne, nso = 2 * n_occ, 2 * n_bf
    y0 = ccsd_lambda_ground_state(f, g, ne)
    yf = tdccsd_propagate(f, z, g, ne, y0, e0, omega, 0.0, t_foot, n_env, -0.5 * float(t_foot), float(dt), n_steps)
    L = yf.size // 2
    w_start = configuration_weights(y0.astype(complex), ne, nso)
    w_end = configuration_weights(yf[:L] + 1j * yf[L:], ne, nso)
    v = nso - ne
    cols = [1 + (2 * occ) * v + (2 * vir - ne), 1 + (2 * occ + 1) * v + (2 * vir + 1 - ne)]
    return float(np.sum(w_end[cols]) - np.sum(w_start[cols]))
SCICODE_GOLD_EOF
