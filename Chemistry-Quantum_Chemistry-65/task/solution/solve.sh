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


def ppp_polyene_hamiltonian(n_sites: int, scale: float, params: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = int(n_sites)
    if n < 2 or n % 2 != 0:
        raise ValueError("n_sites must be an even integer of at least 2")
    if not scale > 0.0:
        raise ValueError("scale must be positive")
    w, u, t_short, t_long, r_short, r_long = [float(x) for x in np.asarray(params, dtype=float)]
    # planar zigzag: bond directions alternate between +30 and -30 degrees, which gives 120 degree angles
    pos = np.zeros((n, 2))
    for k in range(n - 1):
        length = r_short if k % 2 == 0 else r_long
        angle = np.deg2rad(30.0 if k % 2 == 0 else -30.0)
        pos[k + 1] = pos[k] + length * np.array([np.cos(angle), np.sin(angle)])
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    gamma = float(scale) * u / np.sqrt(1.0 + (u * dist / 14.397) ** 2)
    h = np.zeros((n, n))
    for k in range(n - 1):
        h[k, k + 1] = h[k + 1, k] = t_short if k % 2 == 0 else t_long
    h[np.diag_indices(n)] = -w - (gamma.sum(axis=1) - np.diag(gamma))
    return np.stack([h, gamma])

import numpy as np


def restricted_hartree_fock(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    h = np.asarray(h, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or g.shape != h.shape:
        raise ValueError("h and gamma must be square arrays of the same shape")
    n = h.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")

    def _fock(dens):
        return h + np.diag(g @ np.diag(dens)) - 0.5 * dens * g

    e, c = np.linalg.eigh(h)
    dens = 2.0 * c[:, :o] @ c[:, :o].T
    for _ in range(2000):
        e, c = np.linalg.eigh(_fock(dens))
        new = 2.0 * c[:, :o] @ c[:, :o].T
        converged = np.abs(new - dens).max() < 1e-12
        dens = 0.5 * (dens + new)   # damping keeps strongly scaled repulsions from oscillating
        if converged:
            dens = new
            break
    e, c = np.linalg.eigh(_fock(dens))
    lead = np.argmax(np.abs(c) > 1e-6, axis=0)     # first site with a non-negligible coefficient
    c = c * np.sign(c[lead, np.arange(n)])
    return np.vstack([e[None, :], c])

import numpy as np


def ring_ecc_amplitudes(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    v4 = np.asarray(eri, dtype=float)
    n = e.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    if v4.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    v = n - o
    gap = (e[o:][None, :] - e[:o][:, None]).reshape(-1)
    k = v4[:o, o:, :o, o:].reshape(o * v, o * v)
    a_blk = np.diag(gap) + 2.0 * k
    b_blk = 2.0 * k
    # positive-frequency eigenvectors of the direct RPA through the symmetric form (A-B)^1/2 (A+B) (A-B)^1/2
    root = np.sqrt(gap)
    omega2, vec = np.linalg.eigh(root[:, None] * (a_blk + b_blk) * root[None, :])
    omega = np.sqrt(omega2)
    x_plus_y = (root[:, None] * vec) / np.sqrt(omega)[None, :]
    x_minus_y = (vec / root[:, None]) * np.sqrt(omega)[None, :]
    x = 0.5 * (x_plus_y + x_minus_y)
    y = 0.5 * (x_plus_y - x_minus_y)
    t = y @ np.linalg.inv(x)
    t = 0.5 * (t + t.T)
    z = t @ np.linalg.inv(np.eye(o * v) - t @ t)
    return np.stack([t, z])

import numpy as np


def ecc_quasiparticle_ionization(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int,
                                         fock: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    v4 = np.asarray(eri, dtype=float)
    f = np.asarray(fock, dtype=float)
    n = e.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    if v4.shape != (n, n, n, n) or f.shape != (n, n):
        raise ValueError("eri must have shape (n, n, n, n) and fock shape (n, n)")
    v = n - o
    m = o * v
    amps = ring_ecc_amplitudes(e, v4, o)
    t, z = amps[0], amps[1]
    gap = (e[o:][None, :] - e[:o][:, None]).reshape(-1)
    k = v4[:o, o:, :o, o:].reshape(m, m)
    a_blk = np.diag(gap) + 2.0 * k
    b_blk = 2.0 * k
    eye = np.eye(m)
    # spin-adapted direct electron-boson coupling V_{pq,nu} = sqrt(2) (pq|ia)
    coup = np.sqrt(2.0) * v4[:, :, :o, o:].reshape(n, n, m)
    right = coup @ (eye + t)                  # N blocks
    left = coup @ (eye + z + t @ z)           # N-tilde blocks, (eye + z + t z) = (eye - t)^-1
    d_hole = np.kron(np.diag(e[:o]), eye) - np.kron(np.eye(o), a_blk + t @ b_blk)
    d_part = np.kron(np.diag(e[o:]), eye) + np.kron(np.eye(v), a_blk + b_blk @ t)
    zero = np.zeros((o * m, v * m))
    heff = np.block([
        [f, left[:, :o, :].reshape(n, o * m), right[:, o:, :].reshape(n, v * m)],
        [right[:, :o, :].reshape(n, o * m).T, d_hole, zero],
        [left[:, o:, :].reshape(n, v * m).T, zero.T, d_part],
    ])
    w, vr = np.linalg.eig(heff)
    vl = np.linalg.inv(vr)                    # rows are left eigenvectors, biorthonormal to the columns of vr
    weight = (vl[:, o - 1] * vr[o - 1, :]).real
    best = int(np.argmax(weight))
    return np.array([-w[best].real, weight[best]])

import numpy as np


def linearized_gw_density_matrix(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    v4 = np.asarray(eri, dtype=float)
    n = e.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    if v4.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    v = n - o
    gap = (e[o:][None, :] - e[:o][:, None]).reshape(-1)
    k = v4[:o, o:, :o, o:].reshape(o * v, o * v)
    root = np.sqrt(gap)
    omega2, vec = np.linalg.eigh(root[:, None] * (np.diag(gap) + 4.0 * k) * root[None, :])
    omega = np.sqrt(omega2)
    x_plus_y = (root[:, None] * vec) / np.sqrt(omega)[None, :]
    mcoup = np.sqrt(2.0) * np.einsum('pqk,kv->pqv', v4[:, :, :o, o:].reshape(n, n, o * v), x_plus_y)
    ei, ea = e[:o], e[o:]
    den = ei[:, None, None] - ea[None, :, None] - omega[None, None, :]          # (i, a, nu)
    m_ov = mcoup[:o, o:, :] / den
    g_oo = np.eye(o) - np.einsum('iav,jav->ij', m_ov, m_ov)
    g_vv = np.einsum('iav,ibv->ab', m_ov, m_ov)
    term_b = np.einsum('ibv,abv->ia', m_ov, mcoup[o:, o:, :])
    term_j = np.einsum('ijv,ajv->ia', mcoup[:o, :o, :], (mcoup[o:, :o, :] / den.transpose(1, 0, 2)))
    g_ov = (term_b - term_j) / (ei[:, None] - ea[None, :])   # hole term enters with a minus sign
    dens = np.zeros((n, n))
    dens[:o, :o] = g_oo
    dens[o:, o:] = g_vv
    dens[:o, o:] = g_ov
    dens[o:, :o] = g_ov.T
    return 0.5 * (dens + dens.T)

import numpy as np


def static_self_energy_correction(eri: "np.ndarray", density: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    v4 = np.asarray(eri, dtype=float)
    dens = np.asarray(density, dtype=float)
    n = dens.shape[0]
    o = int(n_occ)
    if dens.shape != (n, n) or v4.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n) and density shape (n, n)")
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    delta = dens.copy()
    delta[np.arange(o), np.arange(o)] -= 1.0
    return 2.0 * np.einsum('pqrs,rs->pq', v4, delta) - np.einsum('prqs,rs->pq', v4, delta)

import itertools
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def _fci_lowest_energy(h, g, n_up, n_down):
    """Lowest eigenvalue of the zero-differential-overlap Hamiltonian with n_up and n_down electrons."""
    import itertools
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg
    n = h.shape[0]

    def _strings(k):
        return [sum(1 << p for p in comb) for comb in itertools.combinations(range(n), k)]

    def _hop_matrix(strs):
        index = {s: j for j, s in enumerate(strs)}
        rows, cols, vals = [], [], []
        for j, s in enumerate(strs):
            for p in range(n):
                for q in range(n):
                    if p == q or h[p, q] == 0.0 or not (s >> q) & 1 or (s >> p) & 1:
                        continue
                    lo, hi = min(p, q), max(p, q)
                    passed = s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
                    rows.append(index[s ^ (1 << q) ^ (1 << p)])
                    cols.append(j)
                    vals.append((-1.0) ** bin(passed).count('1') * h[p, q])
        return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(strs), len(strs)))

    up, down = _strings(n_up), _strings(n_down)
    occ_up = np.array([[(s >> p) & 1 for p in range(n)] for s in up], dtype=float).reshape(len(up), n)
    occ_dn = np.array([[(s >> p) & 1 for p in range(n)] for s in down], dtype=float).reshape(len(down), n)
    off = g - np.diag(np.diag(g))
    e_up = occ_up @ np.diag(h) + 0.5 * np.einsum('ap,pq,aq->a', occ_up, off, occ_up)
    e_dn = occ_dn @ np.diag(h) + 0.5 * np.einsum('bp,pq,bq->b', occ_dn, off, occ_dn)
    diag = e_up[:, None] + e_dn[None, :] + occ_up @ g @ occ_dn.T
    hmat = (scipy.sparse.kron(_hop_matrix(up), scipy.sparse.identity(len(down)))
            + scipy.sparse.kron(scipy.sparse.identity(len(up)), _hop_matrix(down))
            + scipy.sparse.diags(diag.reshape(-1)))
    if hmat.shape[0] <= 3000:
        return float(np.linalg.eigvalsh(hmat.toarray())[0])
    vals = scipy.sparse.linalg.eigsh(hmat.tocsr(), k=1, which='SA', tol=1e-12)[0]
    return float(vals[0])


def fci_ionization_energy(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> float:
    """Reference implementation."""
    import numpy as np
    h = np.asarray(h, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or g.shape != h.shape:
        raise ValueError("h and gamma must be square arrays of the same shape")
    n = h.shape[0]
    o = int(n_occ)
    if not 1 <= o <= n:
        raise ValueError("n_occ must satisfy 1 <= n_occ <= n")
    return _fci_lowest_energy(h, g, o, o - 1) - _fci_lowest_energy(h, g, o, o)

import numpy as np
from scipy.optimize import brentq


def break_even_interaction_scale(n_sites: int, params: "np.ndarray", scale_low: float,
                                         scale_high: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not 0.0 < scale_low < scale_high:
        raise ValueError("the bracket must satisfy 0 < scale_low < scale_high")
    occ = int(n_sites) // 2

    def _ionization_energies(lam):
        hg = ppp_polyene_hamiltonian(n_sites, lam, params)
        h, g = hg[0], hg[1]
        scf = restricted_hartree_fock(h, g, occ)
        e, c = scf[0], scf[1:]
        pair = np.einsum('mp,mq->mpq', c, c)
        eri = np.einsum('mpq,mn,nrs->pqrs', pair, g, pair, optimize=True)
        ip_plain = ecc_quasiparticle_ionization(e, eri, occ, np.diag(e))[0]
        dens = linearized_gw_density_matrix(e, eri, occ)
        sigma = static_self_energy_correction(eri, dens, occ)
        ip_corr = ecc_quasiparticle_ionization(e, eri, occ, np.diag(e) + sigma)[0]
        ip_exact = fci_ionization_energy(h, g, occ)
        return ip_plain, ip_corr, ip_exact

    def _balance(lam):
        a, b, c_ = _ionization_energies(lam)
        return a + b - 2.0 * c_

    f_low, f_high = _balance(scale_low), _balance(scale_high)
    if f_low * f_high > 0.0:
        raise ValueError("the break-even function does not change sign inside the bracket")
    lam_be = brentq(_balance, scale_low, scale_high, xtol=1e-9, rtol=1e-12)
    ips = _ionization_energies(lam_be)
    return np.array([lam_be, ips[0], ips[1], ips[2]])
SCICODE_GOLD_EOF
