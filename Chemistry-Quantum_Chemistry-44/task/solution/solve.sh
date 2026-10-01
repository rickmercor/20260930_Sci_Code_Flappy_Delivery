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
import numpy.typing as npt


def _boys_zero(t):
    """Boys function of order zero, F0(t) = (1/2) sqrt(pi / t) erf(sqrt(t))."""
    import numpy as np
    from scipy.special import erf
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = t < 1.0e-12
    out[small] = 1.0 - t[small] / 3.0
    big = ~small
    out[big] = 0.5 * np.sqrt(np.pi / t[big]) * erf(np.sqrt(t[big]))
    return out


def _s_basis_functions(coords, exponents, coefficients):
    """Validate the geometry and basis; return the normalised contracted s functions, atom-major."""
    import numpy as np
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 1 or not np.all(np.isfinite(xyz)):
        raise ValueError("coords must be a finite array of shape (n_atoms, 3)")
    if xyz.shape[0] > 1:
        dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        if np.min(dist[np.triu_indices(xyz.shape[0], 1)]) < 1.0e-6:
            raise ValueError("two nuclei coincide")
    if len(exponents) < 1 or len(exponents) != len(coefficients):
        raise ValueError("exponents and coefficients must be non-empty sequences of equal length")
    shells = []
    for e, c in zip(exponents, coefficients):
        e = np.atleast_1d(np.asarray(e, dtype=float))
        c = np.atleast_1d(np.asarray(c, dtype=float))
        if e.ndim != 1 or e.shape != c.shape:
            raise ValueError("each shell needs one-dimensional exponents and coefficients of equal length")
        if not (np.all(np.isfinite(e)) and np.all(e > 0.0) and np.all(np.isfinite(c)) and np.any(c != 0.0)):
            raise ValueError("exponents must be positive and finite, coefficients finite and not all zero")
        shells.append((e, c))
    functions = []
    for centre in xyz:
        for e, c in shells:
            d = c * (2.0 * e / np.pi) ** 0.75
            norm2 = np.sum(d[:, None] * d[None, :] * (np.pi / (e[:, None] + e[None, :])) ** 1.5)
            functions.append((centre.copy(), e, d / np.sqrt(norm2)))
    return xyz, functions


def one_electron_integrals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                   coefficients: list) -> np.ndarray:
    import numpy as np
    xyz, functions = _s_basis_functions(coords, exponents, coefficients)
    Z = np.atleast_1d(np.asarray(charges, dtype=float))
    if Z.shape != (xyz.shape[0],) or not np.all(np.isfinite(Z)) or np.any(Z <= 0.0):
        raise ValueError("charges must hold one positive finite nuclear charge per atom")
    n = len(functions)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    for p in range(n):
        A, ea, da = functions[p]
        for q in range(n):
            B, eb, db = functions[q]
            gam = ea[:, None] + eb[None, :]
            mu = ea[:, None] * eb[None, :] / gam
            AB2 = float(np.dot(A - B, A - B))
            K = np.exp(-mu * AB2)
            P = (ea[:, None, None] * A + eb[None, :, None] * B) / gam[..., None]
            dd = da[:, None] * db[None, :]
            s_prim = (np.pi / gam) ** 1.5 * K
            S[p, q] = np.sum(dd * s_prim)
            T[p, q] = np.sum(dd * mu * (3.0 - 2.0 * mu * AB2) * s_prim)
            for C, zc in zip(xyz, Z):
                PC2 = np.sum((P - C) ** 2, axis=-1)
                V[p, q] -= zc * np.sum(dd * 2.0 * np.pi / gam * K * _boys_zero(gam * PC2))
    return np.stack([S, T + V])

import numpy as np
import numpy.typing as npt


def electron_repulsion_integrals(coords: npt.ArrayLike, exponents: list,
                                         coefficients: list) -> np.ndarray:
    import numpy as np
    xyz, functions = _s_basis_functions(coords, exponents, coefficients)
    n = len(functions)
    pair = {}
    for p in range(n):
        A, ea, da = functions[p]
        for q in range(n):
            B, eb, db = functions[q]
            gam = ea[:, None] + eb[None, :]
            K = np.exp(-ea[:, None] * eb[None, :] / gam * float(np.dot(A - B, A - B)))
            P = (ea[:, None, None] * A + eb[None, :, None] * B) / gam[..., None]
            pair[(p, q)] = (gam, K * da[:, None] * db[None, :], P)
    eri = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(p + 1):
            g1, w1, P1 = pair[(p, q)]
            for r in range(n):
                for s in range(r + 1):
                    if r * (r + 1) // 2 + s > p * (p + 1) // 2 + q:
                        continue
                    g2, w2, P2 = pair[(r, s)]
                    G1 = g1[:, :, None, None]
                    G2 = g2[None, None, :, :]
                    PQ2 = np.sum((P1[:, :, None, None, :] - P2[None, None, :, :, :]) ** 2, axis=-1)
                    val = np.sum(w1[:, :, None, None] * w2[None, None, :, :]
                                 * 2.0 * np.pi ** 2.5 / (G1 * G2 * np.sqrt(G1 + G2))
                                 * _boys_zero(G1 * G2 / (G1 + G2) * PQ2))
                    for idx in ((p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                                (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)):
                        eri[idx] = val
    return eri

import numpy as np
import numpy.typing as npt


def _positive_phase_columns(M):
    """Flip each column so that its largest-magnitude entry (first one among near-ties) is positive."""
    import numpy as np
    M = np.array(M, dtype=float)
    for j in range(M.shape[1]):
        col = np.abs(M[:, j])
        k = int(np.flatnonzero(col >= col.max() * (1.0 - 1.0e-8))[0])
        if M[k, j] < 0.0:
            M[:, j] = -M[:, j]
    return M


def _closed_shell_count(n_electrons, n_basis):
    import numpy as np
    ne = float(n_electrons)
    if not np.isfinite(ne) or ne != int(ne) or int(ne) < 2 or int(ne) % 2 != 0:
        raise ValueError("n_electrons must be a positive even integer")
    nocc = int(ne) // 2
    if nocc >= n_basis:
        raise ValueError("the basis must leave at least one virtual orbital")
    return nocc


def _closed_shell_fock(H, eri, D):
    """Closed-shell Fock matrix H + 2J - K for the density D = C_occ C_occ^T."""
    import numpy as np
    return H + 2.0 * np.einsum('pqrs,rs->pq', eri, D) - np.einsum('prqs,rs->pq', eri, D)


def rhf_canonical_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                   coefficients: list, n_electrons: int) -> np.ndarray:
    import numpy as np
    SH = one_electron_integrals(coords, charges, exponents, coefficients)
    S, H = SH[0], SH[1]
    eri = electron_repulsion_integrals(coords, exponents, coefficients)
    n = S.shape[0]
    nocc = _closed_shell_count(n_electrons, n)
    s, U = np.linalg.eigh(S)
    if s[0] < 1.0e-10:
        raise ValueError("the basis is numerically linearly dependent")
    X = U @ np.diag(s ** -0.5) @ U.T
    e, Cp = np.linalg.eigh(X.T @ H @ X)
    C = X @ Cp
    D = C[:, :nocc] @ C[:, :nocc].T
    F_hist, R_hist = [], []
    for it in range(1000):
        F = _closed_shell_fock(H, eri, D)
        F_hist.append(F)
        R_hist.append(F @ D @ S - S @ D @ F)
        F_hist, R_hist = F_hist[-8:], R_hist[-8:]
        m = len(F_hist)
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    B[a, b] = np.sum(R_hist[a] * R_hist[b])
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            w = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            F = sum(wi * Fi for wi, Fi in zip(w, F_hist))
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        D_new = C[:, :nocc] @ C[:, :nocc].T
        change = np.max(np.abs(D_new - D))
        D = D_new
        if change < 1.0e-12 and it > 1:
            break
    else:
        raise ValueError("the SCF iterations did not converge")
    e, Cp = np.linalg.eigh(X.T @ _closed_shell_fock(H, eri, D) @ X)
    C = _positive_phase_columns(X @ Cp)
    return np.vstack([e[None, :], C])

import numpy as np
import numpy.typing as npt


def _spin_orbital_antisymmetrized(eri_mo):
    """<pq||rs> for spin orbitals p = 2k + s (s = 0 alpha, 1 beta) from chemist integrals (kl|mn)."""
    import numpy as np
    n = eri_mo.shape[0]
    k = np.arange(2 * n) // 2
    s = np.arange(2 * n) % 2
    g = eri_mo[np.ix_(k, k, k, k)].transpose(0, 2, 1, 3)
    same = (s[:, None, None, None] == s[None, None, :, None]) & (s[None, :, None, None] == s[None, None, None, :])
    g = np.where(same, g, 0.0)
    return g - g.transpose(0, 1, 3, 2)


def _pair_denominators(eo, ev):
    """e_a + e_b - e_i - e_j as an (o, o, v, v) array."""
    import numpy as np
    return ev[None, None, :, None] + ev[None, None, None, :] - eo[:, None, None, None] - eo[None, :, None, None]


def dressed_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                             coefficients: list, n_electrons: int, A0: float, B0: float) -> np.ndarray:
    import numpy as np
    A0 = float(A0)
    B0 = float(B0)
    if not (np.isfinite(A0) and np.isfinite(B0)):
        raise ValueError("A0 and B0 must be finite")
    ref = rhf_canonical_orbitals(coords, charges, exponents, coefficients, n_electrons)
    e, C = ref[0], ref[1:]
    eri = electron_repulsion_integrals(coords, exponents, coefficients)
    eri_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, C, C, C, C, optimize=True)
    g = _spin_orbital_antisymmetrized(eri_mo)
    n = e.size
    nocc = int(n_electrons) // 2
    no = 2 * nocc
    eps = np.repeat(e, 2)
    Ro = np.eye(nocc)
    Rv = np.eye(n - nocc)
    eo = eps[:no].copy()
    ev = eps[no:].copy()
    E_old = None
    for it in range(2000):
        Uo = np.kron(Ro, np.eye(2))
        Uv = np.kron(Rv, np.eye(2))
        goovv = np.einsum('pqrs,pi,qj,ra,sb->ijab', g[:no, :no, no:, no:], Uo, Uo, Uv, Uv, optimize=True)
        t = goovv / _pair_denominators(eo, ev)
        E2 = -0.25 * np.sum(goovv * t)
        X = np.einsum('ikab,jkab->ij', goovv, t)
        Y = np.einsum('ijac,ijbc->ab', goovv, t)
        Woo = -(A0 / 8.0) * (X + X.T)
        Wvv = -(B0 / 8.0) * (Y + Y.T)
        Ho = (Uo.T @ np.diag(eps[:no]) @ Uo + Woo)[0::2, 0::2]
        Hv = (Uv.T @ np.diag(eps[no:]) @ Uv + Wvv)[0::2, 0::2]
        off = max(np.max(np.abs(Ho - np.diag(np.diag(Ho)))), np.max(np.abs(Hv - np.diag(np.diag(Hv)))))
        converged = E_old is not None and abs(E2 - E_old) < 1.0e-13 and off < 1.0e-12
        wo, Qo = np.linalg.eigh(Ho)
        wv, Qv = np.linalg.eigh(Hv)
        if not (np.all(np.isfinite(wo)) and np.all(np.isfinite(wv))) or wv[0] <= wo[-1]:
            raise ValueError("the dressed occupied and virtual levels overlap; the dressing has no solution here")
        Ro = Ro @ Qo
        Rv = Rv @ Qv
        eo = np.repeat(wo, 2)
        ev = np.repeat(wv, 2)
        E_old = E2
        if converged:
            break
    else:
        raise ValueError("the self-consistent dressing did not converge")
    R = np.zeros((n, n))
    R[:nocc, :nocc] = Ro
    R[nocc:, nocc:] = Rv
    Cd = _positive_phase_columns(C @ R)
    return np.vstack([np.concatenate([wo, wv])[None, :], Cd])

import numpy as np
import numpy.typing as npt


def _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ):
    """Validate zeroth-order energies, Fock matrix and antisymmetrised integrals; return arrays."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    f = np.asarray(fock, dtype=float)
    g = np.asarray(eri, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("orbital_energies must be one-dimensional with at least two entries")
    m = e.size
    if f.shape != (m, m) or g.shape != (m, m, m, m):
        raise ValueError("fock must be (m, m) and eri (m, m, m, m) for m orbital energies")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        raise ValueError("inputs must be finite")
    no = float(n_occ)
    if no != int(no) or not (0 < int(no) < m):
        raise ValueError("n_occ must be an integer strictly between 0 and the number of orbitals")
    no = int(no)
    if np.max(np.abs(f - f.T)) > 1.0e-10 or np.max(np.abs(f[:no, no:])) > 1.0e-10:
        raise ValueError("fock must be symmetric with a vanishing occupied-virtual block")
    if (np.max(np.abs(g + g.transpose(1, 0, 2, 3))) > 1.0e-10
            or np.max(np.abs(g + g.transpose(0, 1, 3, 2))) > 1.0e-10
            or np.max(np.abs(g - g.transpose(2, 3, 0, 1))) > 1.0e-10):
        raise ValueError("eri must be antisymmetric in each index pair and symmetric under pair exchange")
    if np.min(e[no:]) <= np.max(e[:no]):
        raise ValueError("every virtual orbital energy must lie above every occupied one")
    return e, f, g, no


def second_order_ground_state(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike,
                                      eri: npt.ArrayLike, n_occ: int) -> np.ndarray:
    import numpy as np
    e, f, g, no = _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ)
    nv = e.size - no
    o = slice(0, no)
    v = slice(no, no + nv)
    eo, ev = e[:no], e[no:]
    D = _pair_denominators(eo, ev)
    t = g[o, o, v, v] / D
    X = np.einsum('ikac,kbjc->ijab', t, g[o, v, o, v])
    X = X - X.transpose(1, 0, 2, 3) - X.transpose(0, 1, 3, 2) + X.transpose(1, 0, 3, 2)
    X = X - 0.5 * np.einsum('klab,ijkl->ijab', t, g[o, o, o, o]) - 0.5 * np.einsum('ijcd,abcd->ijab', t, g[v, v, v, v])
    F = (np.einsum('ac,ijcb->ijab', f[v, v], t) + np.einsum('bc,ijac->ijab', f[v, v], t)
         - np.einsum('ik,kjab->ijab', f[o, o], t) - np.einsum('jk,ikab->ijab', f[o, o], t))
    t2 = (X - F) / D + t
    t1 = -(0.5 * np.einsum('ijbc,jabc->ia', t, g[o, v, v, v])
           + 0.5 * np.einsum('jkab,jkib->ia', t, g[o, o, o, v])) / (ev[None, :] - eo[:, None])
    E2 = -0.25 * np.sum(g[o, o, v, v] * t)
    E3 = -0.25 * np.sum(g[o, o, v, v] * t2)
    return np.concatenate([np.array([E2, E3]), t2.ravel(), t1.ravel()])

import numpy as np
import numpy.typing as npt


def _occupied_delta(A, n_vir):
    """A_ij delta_ab laid out as an (i, a, j, b) array."""
    import numpy as np
    return A[:, None, :, None] * np.eye(n_vir)[None, :, None, :]


def _virtual_delta(A, n_occ):
    """A_ab delta_ij laid out as an (i, a, j, b) array."""
    import numpy as np
    return A[None, :, None, :] * np.eye(n_occ)[:, None, :, None]


def adc3_singles_block(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike,
                               n_occ: int) -> np.ndarray:
    import numpy as np
    e, f, g, no = _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ)
    nv = e.size - no
    o = slice(0, no)
    v = slice(no, no + nv)
    gs = second_order_ground_state(e, f, g, no)
    t2 = gs[2:2 + no * no * nv * nv].reshape(no, no, nv, nv)
    ts = gs[2 + no * no * nv * nv:].reshape(no, nv)
    t = g[o, o, v, v] / _pair_denominators(e[:no], e[no:])
    roo = -0.5 * np.einsum('ikab,jkab->ij', t, t)
    rvv = 0.5 * np.einsum('ijac,ijbc->ab', t, t)
    goooo, goovv, govov, gvvvv = g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v]
    gooov, govvv = g[o, o, o, v], g[o, v, v, v]
    M = (_virtual_delta(f[v, v], no) - _occupied_delta(f[o, o], nv) - np.einsum('ibja->iajb', govov))
    X2 = (-0.5 * np.einsum('ikac,jkbc->iajb', t, goovv)
          + _virtual_delta(0.25 * np.einsum('klac,klbc->ab', t, goovv), no)
          + _occupied_delta(0.25 * np.einsum('ikcd,jkcd->ij', t, goovv), nv))
    M = M + X2 + X2.transpose(2, 3, 0, 1)
    Z4 = np.einsum('jkac,kbic->ijab', t, govov)
    Z3 = np.einsum('klab,ijkl->ijab', t, goooo)
    Z5 = np.einsum('ijcd,abcd->ijab', t, gvvvv)
    Zsq = np.einsum('ikac,jkbc->iajb', t, t)
    Y = (np.einsum('ka,jkib->iajb', ts, gooov)
         + np.einsum('ic,jabc->iajb', ts, govvv)
         + 0.5 * np.einsum('ikac,jkbc->iajb', t, Z4)
         + 0.5 * np.einsum('ikac,kjcb->iajb', t, Z4)
         + 0.5 * np.einsum('ibjc,ac->iajb', govov, rvv)
         - np.einsum('ibkc,jcka->iajb', govov, Zsq)
         - 0.5 * np.einsum('ikac,jkbc->iajb', t2, goovv)
         - 0.5 * np.einsum('ibka,jk->iajb', govov, roo)
         + 0.25 * np.einsum('ikac,jkbc->iajb', t, Z3)
         + 0.25 * np.einsum('ikac,jkbc->iajb', t, Z5)
         + _virtual_delta(0.5 * np.einsum('klac,klcb->ab', t, Z4), no)
         + _occupied_delta(0.5 * np.einsum('ikcd,jkdc->ij', t, Z4), nv)
         - _occupied_delta(np.einsum('kc,ikjc->ij', ts, gooov), nv)
         - _virtual_delta(np.einsum('kc,kabc->ab', ts, govvv), no)
         - _virtual_delta(0.125 * np.einsum('klac,klbc->ab', t, Z5), no)
         - _occupied_delta(0.125 * np.einsum('ikcd,jkcd->ij', t, Z3), nv)
         + _virtual_delta(0.25 * np.einsum('klac,klbc->ab', t2, goovv), no)
         + _occupied_delta(0.25 * np.einsum('ikcd,jkcd->ij', t2, goovv), nv))
    M = M + Y + Y.transpose(2, 3, 0, 1)
    M = M + (-np.einsum('adbc,icjd->iajb', gvvvv, Zsq)
             - np.einsum('iljk,kalb->iajb', goooo, Zsq)
             + _virtual_delta(np.einsum('adbc,cd->ab', gvvvv, rvv), no)
             + _virtual_delta(np.einsum('kalb,kl->ab', govov, roo), no)
             + 0.5 * np.einsum('klac,klbd,icjd->iajb', t, t, govov, optimize=True)
             + 0.5 * np.einsum('ikcd,jlcd,kalb->iajb', t, t, govov, optimize=True)
             - _occupied_delta(np.einsum('icjd,cd->ij', govov, rvv), nv)
             - _occupied_delta(np.einsum('iljk,kl->ij', goooo, roo), nv))
    return M.reshape(no * nv, no * nv)

import numpy as np
import numpy.typing as npt


def _check_spin_adapted(e, f, g, no):
    """Raise unless the spin-orbital arrays come from restricted spatial orbitals, p = 2k + s."""
    import numpy as np
    m = e.size
    if m % 2 or no % 2:
        raise ValueError("spin-adapted input needs an even number of spin orbitals and of occupied ones")
    s = np.arange(m) % 2
    flip = np.arange(m) ^ 1
    if np.max(np.abs(e - e[flip])) > 1.0e-10:
        raise ValueError("alpha and beta orbital energies differ")
    if (np.max(np.abs(np.where(s[:, None] != s[None, :], f, 0.0))) > 1.0e-10
            or np.max(np.abs(f - f[np.ix_(flip, flip)])) > 1.0e-10):
        raise ValueError("fock must be spin-diagonal with identical alpha and beta blocks")
    sp, sq, sr, ss = s[:, None, None, None], s[None, :, None, None], s[None, None, :, None], s[None, None, None, :]
    conserving = ((sp == sr) & (sq == ss)) | ((sp == ss) & (sq == sr))
    if (np.max(np.abs(np.where(conserving, 0.0, g))) > 1.0e-10
            or np.max(np.abs(g - g[np.ix_(flip, flip, flip, flip)])) > 1.0e-10):
        raise ValueError("eri must conserve spin and be invariant under exchanging alpha and beta")


def adc3_spin_sector_spectrum(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike,
                                      n_occ: int, flip_parity: int) -> np.ndarray:
    import numpy as np
    from itertools import combinations
    e, f, g, no = _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ)
    if flip_parity not in (1, -1):
        raise ValueError("flip_parity must be +1 or -1")
    _check_spin_adapted(e, f, g, no)
    nv = e.size - no
    o = slice(0, no)
    v = slice(no, no + nv)
    ph = [(i, a) for i in range(no) for a in range(nv) if i % 2 == a % 2]
    dd = [(i, j, a, b) for i, j in combinations(range(no), 2) for a, b in combinations(range(nv), 2)
          if (i % 2) + (j % 2) == (a % 2) + (b % 2)]
    n1, n2 = len(ph), len(dd)
    iph = np.array([x[0] for x in ph])
    aph = np.array([x[1] for x in ph])
    M11 = adc3_singles_block(e, f, g, no)
    rows = iph * nv + aph
    M = np.zeros((n1 + n2, n1 + n2))
    M[:n1, :n1] = M11[np.ix_(rows, rows)]
    if n2 > 0:
        di, dj, da, db = (np.array([x[k] for x in dd]) for k in range(4))
        t = g[o, o, v, v] / _pair_denominators(e[:no], e[no:])
        gooov, govvv, govov = g[o, o, o, v], g[o, v, v, v], g[o, v, o, v]
        Yu = np.zeros((n2, no, no, nv, nv))
        z = np.arange(n2)
        Yu[z, di, dj, da, db] = 1.0
        Yu[z, dj, di, da, db] = -1.0
        Yu[z, di, dj, db, da] = -1.0
        Yu[z, dj, di, db, da] = 1.0
        ZI = np.einsum('ijbc,kabc->ijka', t, govvv)
        ZII = np.einsum('ilab,lkjb->ijka', t, gooov)
        ZA = 0.5 * ZI + ZII - ZII.transpose(1, 0, 2, 3)
        ZVI = np.einsum('jkbc,jkia->iabc', t, gooov)
        ZVII = np.einsum('ijbd,jcad->iabc', t, govvv)
        ZB = -0.5 * ZVI + ZVII - ZVII.transpose(0, 1, 3, 2)
        w = (np.einsum('zjkab,jkib->zia', Yu, gooov) + np.einsum('zijbc,jabc->zia', Yu, govvv)
             + np.einsum('zijbc,jabc->zia', Yu, ZB) - np.einsum('zjkab,jkib->zia', Yu, ZA)
             + np.einsum('zjkbc,jlbc,ilka->zia', Yu, t, gooov, optimize=True)
             + np.einsum('zjkbc,jkbd,icad->zia', Yu, t, govvv, optimize=True))
        M12 = -0.5 * w[:, iph, aph].T
        X1 = np.einsum('zjkab,ik->zijab', Yu, f[o, o]) + np.einsum('zijac,bc->zijab', Yu, f[v, v])
        X2 = np.einsum('zjkac,ickb->zijab', Yu, govov) - np.einsum('zikac,jckb->zijab', Yu, govov)
        W1 = (X1 + X1.transpose(0, 2, 1, 4, 3) + X2 + X2.transpose(0, 2, 1, 4, 3)
              + 0.5 * np.einsum('zklab,ijkl->zijab', Yu, g[o, o, o, o])
              + 0.5 * np.einsum('zijcd,abcd->zijab', Yu, g[v, v, v, v]))
        M22 = W1[:, di, dj, da, db].T
        M[:n1, n1:] = M12
        M[n1:, :n1] = M12.T
        M[n1:, n1:] = 0.5 * (M22 + M22.T)
    P = np.zeros((n1 + n2, n1 + n2))
    index1 = {x: k for k, x in enumerate(ph)}
    index2 = {x: k for k, x in enumerate(dd)}
    for k, (i, a) in enumerate(ph):
        P[index1[(i ^ 1, a ^ 1)], k] = 1.0
    for k, (i, j, a, b) in enumerate(dd):
        sign = 1.0
        i2, j2, a2, b2 = i ^ 1, j ^ 1, a ^ 1, b ^ 1
        if i2 > j2:
            i2, j2, sign = j2, i2, -sign
        if a2 > b2:
            a2, b2, sign = b2, a2, -sign
        P[n1 + index2[(i2, j2, a2, b2)], n1 + k] = sign
    lam, Q = np.linalg.eigh(P)
    Qs = Q[:, np.abs(lam - flip_parity) < 0.5]
    energies, V = np.linalg.eigh(Qs.T @ M @ Qs)
    X = Qs @ V
    weights = np.sum(X[:n1] ** 2, axis=0)
    return np.vstack([energies, weights])

import numpy as np
import numpy.typing as npt


def _dressed_spin_orbital_hamiltonian(coords, charges, exponents, coefficients, n_electrons, A0, B0):
    """Zeroth-order energies, true Fock matrix and <pq||rs> in the converged dressed spin-orbital basis."""
    import numpy as np
    SH = one_electron_integrals(coords, charges, exponents, coefficients)
    ref = rhf_canonical_orbitals(coords, charges, exponents, coefficients, n_electrons)
    dressed = dressed_orbitals(coords, charges, exponents, coefficients, n_electrons, A0, B0)
    eri = electron_repulsion_integrals(coords, exponents, coefficients)
    Cd = dressed[1:]
    R = ref[1:].T @ SH[0] @ Cd
    f_spatial = R.T @ np.diag(ref[0]) @ R
    eri_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, Cd, Cd, Cd, Cd, optimize=True)
    g = _spin_orbital_antisymmetrized(eri_mo)
    return np.repeat(dressed[0], 2), np.kron(f_spatial, np.eye(2)), g, 2 * (int(n_electrons) // 2)


def spin_resolved_excitations(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                      coefficients: list, n_electrons: int, A0: float, B0: float,
                                      n_states: int) -> np.ndarray:
    import numpy as np
    ns = float(n_states)
    if ns != int(ns) or int(ns) < 1:
        raise ValueError("n_states must be a positive integer")
    ns = int(ns)
    e, f, g, no = _dressed_spin_orbital_hamiltonian(coords, charges, exponents, coefficients, n_electrons, A0, B0)
    out = np.zeros((2, ns))
    for row, parity in enumerate((1, -1)):
        spectrum = adc3_spin_sector_spectrum(e, f, g, no, parity)
        picked = spectrum[0][spectrum[1] > 0.5]
        if picked.size < ns:
            raise ValueError("fewer predominantly single-excitation states than requested")
        out[row] = picked[:ns]
    return out

import numpy as np
import numpy.typing as npt


def lowest_singlet_excitation(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                      coefficients: list, n_electrons: int, A0: float, B0: float) -> float:
    import numpy as np
    states = spin_resolved_excitations(coords, charges, exponents, coefficients, n_electrons, A0, B0, 1)
    return float(states[0, 0] * 27.211386245988)
SCICODE_GOLD_EOF
