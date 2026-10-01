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


def _boys0(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x > 1e-12
    out[m] = 0.5 * np.sqrt(np.pi / x[m]) * erf(np.sqrt(x[m]))
    return out


def _check_model(centres, exponent):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must be a 1D array of an even number of positions")
    if np.any(np.abs(R[:, None] - R[None, :]) + np.eye(R.size) < 1e-8):
        raise ValueError("centres must be distinct")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    return R, float(exponent)


def ao_overlap_and_core(centres, exponent, charges):
    R, a = _check_model(centres, exponent)
    Z = np.asarray(charges, dtype=float)
    if Z.shape != R.shape:
        raise ValueError("charges must have one entry per centre")
    N = R.size
    d2 = (R[:, None] - R[None, :]) ** 2
    p = 2.0 * a
    nrm = (2.0 * a / np.pi) ** 0.75
    E = np.exp(-0.5 * a * d2)
    S = nrm * nrm * (np.pi / p) ** 1.5 * E
    T = 0.5 * a * (3.0 - a * d2) * S
    P = 0.5 * (R[:, None] + R[None, :])
    Vne = np.zeros((N, N))
    for c in range(N):
        Vne -= Z[c] * (2.0 * np.pi / p) * nrm * nrm * E * _boys0(p * (P - R[c]) ** 2)
    return np.stack([S, T + Vne])

import numpy as np
from scipy.special import erf


def _boys0(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x > 1e-12
    out[m] = 0.5 * np.sqrt(np.pi / x[m]) * erf(np.sqrt(x[m]))
    return out


def _check_model(centres, exponent):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must be a 1D array of an even number of positions")
    if np.any(np.abs(R[:, None] - R[None, :]) + np.eye(R.size) < 1e-8):
        raise ValueError("centres must be distinct")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    return R, float(exponent)


def ao_two_electron_integrals(centres, exponent):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must be a 1D array of an even number of positions")
    if np.any(np.abs(R[:, None] - R[None, :]) + np.eye(R.size) < 1e-8):
        raise ValueError("centres must be distinct")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    a = float(exponent)
    p = 2.0 * a
    nrm = (2.0 * a / np.pi) ** 0.75
    d2 = (R[:, None] - R[None, :]) ** 2
    E = np.exp(-0.5 * a * d2)
    P = 0.5 * (R[:, None] + R[None, :])
    pre = 2.0 * np.pi ** 2.5 / (p * p * np.sqrt(2.0 * p)) * nrm ** 4
    Q = (P[:, :, None, None] - P[None, None, :, :]) ** 2
    return pre * E[:, :, None, None] * E[None, None, :, :] * _boys0(0.5 * p * Q)

import numpy as np


def vbs_orbital_coefficients(overlap, n_units):
    S = np.asarray(overlap, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("overlap must be square")
    if S.shape[0] != 2 * int(n_units):
        raise ValueError("overlap must have two functions per valence bond space")
    ev, U = np.linalg.eigh(0.5 * (S + S.T))
    if np.min(ev) <= 1e-10:
        raise ValueError("overlap matrix is not positive definite")
    X = U @ np.diag(ev ** -0.5) @ U.T
    N = S.shape[0]
    B = np.zeros((N, N))
    s = 1.0 / np.sqrt(2.0)
    for u in range(int(n_units)):
        B[2 * u, 2 * u] = s
        B[2 * u + 1, 2 * u] = s
        B[2 * u, 2 * u + 1] = s
        B[2 * u + 1, 2 * u + 1] = -s
    return X @ B

import numpy as np


def _mo_transform(coefficients, core, eri):
    C = np.asarray(coefficients, dtype=float)
    Hc = np.asarray(core, dtype=float)
    W = np.asarray(eri, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if Hc.shape != C.shape or W.shape != C.shape * 2:
        raise ValueError("core and eri must match the basis size")
    h = C.T @ Hc @ C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, W, C, C, optimize=True)
    return h, V


def _families(coefficients, core, eri):
    h, V = _mo_transform(coefficients, core, eri)
    J = np.einsum('ppqq->pq', V).copy()
    K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0 * J - K]), V


def pair_integral_families(coefficients, core, eri):
    C = np.asarray(coefficients, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if np.shape(core) != C.shape or np.shape(eri) != C.shape + C.shape:
        raise ValueError("core and eri must match the basis size")
    h, V = _mo_transform(coefficients, core, eri)
    J = np.einsum('ppqq->pq', V).copy()
    K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0 * J - K])

import numpy as np


def _unpack(families):
    F = np.asarray(families, dtype=float)
    if F.ndim != 3 or F.shape[0] != 5 or F.shape[1] != F.shape[2]:
        raise ValueError("families must have shape (5, N, N)")
    return F[0], F[1], F[2], F[3], F[4]


def _occupations(gaps):
    w = np.asarray(gaps, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("gaps must be a 1D array with one entry per bond space")
    eta = np.sqrt(w * w + 1.0)
    n = np.empty(2 * w.size)
    n[0::2] = 1.0 + w / eta
    n[1::2] = 1.0 - w / eta
    return n, eta


def _orbital_energies(families, gaps):
    h, J, K, L, G = _unpack(families)
    n, eta = _occupations(gaps)
    N = h.shape[0]
    eps = np.empty(N)
    for p in range(N):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(N):
            if q // 2 == p // 2:
                continue
            s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps


def optimal_pair_gaps(families, n_units, tol=1e-13, max_iter=500):
    h, J, K, L, G = _unpack(families)
    M = int(n_units)
    if 2 * M != h.shape[0]:
        raise ValueError("families must hold two orbitals per valence bond space")
    for a in range(M):
        if L[2 * a, 2 * a + 1] <= 0.0:
            raise ValueError("pair-transfer integral of a bond space must be positive")
    w = np.ones(M)
    for _ in range(int(max_iter)):
        eps = _orbital_energies(families, w)
        new = np.array([(eps[2*a+1] - eps[2*a]) / L[2*a, 2*a+1] for a in range(M)])
        if np.max(np.abs(new - w)) < tol:
            return new
        w = 0.5 * (w + new)
    raise ValueError("pair-amplitude stationarity did not converge")

import numpy as np


def _unpack(families):
    F = np.asarray(families, dtype=float)
    if F.ndim != 3 or F.shape[0] != 5 or F.shape[1] != F.shape[2]:
        raise ValueError("families must have shape (5, N, N)")
    return F[0], F[1], F[2], F[3], F[4]


def _occupations(gaps):
    w = np.asarray(gaps, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("gaps must be a 1D array with one entry per bond space")
    eta = np.sqrt(w * w + 1.0)
    n = np.empty(2 * w.size)
    n[0::2] = 1.0 + w / eta
    n[1::2] = 1.0 - w / eta
    return n, eta


def _orbital_energies(families, gaps):
    h, J, K, L, G = _unpack(families)
    n, eta = _occupations(gaps)
    N = h.shape[0]
    eps = np.empty(N)
    for p in range(N):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(N):
            if q // 2 == p // 2:
                continue
            s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps


def pp_reference_energy(families, gaps):
    h, J, K, L, G = _unpack(families)
    w = np.asarray(gaps, dtype=float)
    if 2 * w.size != h.shape[0]:
        raise ValueError("gaps must hold one entry per valence bond space")
    n, eta = _occupations(w)
    eps = _orbital_energies(families, w)
    N = h.shape[0]
    E = float(np.dot(eps, n))
    for a in range(w.size):
        E -= L[2*a, 2*a+1] * np.sqrt(n[2*a] * n[2*a+1])
    for p in range(N):
        for q in range(p + 1, N):
            if p // 2 == q // 2:
                continue
            E -= 0.5 * G[p, q] * n[p] * n[q]
    return float(E)

import numpy as np


def _mo_transform(coefficients, core, eri):
    C = np.asarray(coefficients, dtype=float)
    Hc = np.asarray(core, dtype=float)
    W = np.asarray(eri, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if Hc.shape != C.shape or W.shape != C.shape * 2:
        raise ValueError("core and eri must match the basis size")
    h = C.T @ Hc @ C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, W, C, C, optimize=True)
    return h, V


def _families(coefficients, core, eri):
    h, V = _mo_transform(coefficients, core, eri)
    J = np.einsum('ppqq->pq', V).copy()
    K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0 * J - K]), V


def _unpack(families):
    F = np.asarray(families, dtype=float)
    if F.ndim != 3 or F.shape[0] != 5 or F.shape[1] != F.shape[2]:
        raise ValueError("families must have shape (5, N, N)")
    return F[0], F[1], F[2], F[3], F[4]


def _occupations(gaps):
    w = np.asarray(gaps, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("gaps must be a 1D array with one entry per bond space")
    eta = np.sqrt(w * w + 1.0)
    n = np.empty(2 * w.size)
    n[0::2] = 1.0 + w / eta
    n[1::2] = 1.0 - w / eta
    return n, eta


def _orbital_energies(families, gaps):
    h, J, K, L, G = _unpack(families)
    n, eta = _occupations(gaps)
    N = h.shape[0]
    eps = np.empty(N)
    for p in range(N):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(N):
            if q // 2 == p // 2:
                continue
            s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps


def generalized_fock_matrix(coefficients, core, eri, gaps):
    h, V = _mo_transform(coefficients, core, eri)
    w = np.asarray(gaps, dtype=float)
    if h.shape[0] != 2 * w.size:
        raise ValueError("gaps must hold one amplitude per valence bond space")
    n, eta = _occupations(w)
    N = h.shape[0]
    D = np.zeros((N, N))
    P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q:
                P[p, q] = n[p]
            elif p // 2 == q // 2:
                P[p, q] = -1.0 / eta[p // 2]
            else:
                D[p, q] = n[p] * n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            s = h[p, q] * n[p]
            for r in range(N):
                s += 0.5 * (2 * V[r, r, p, q] - V[r, q, p, r]) * D[r, p]
                s += V[r, p, r, q] * P[r, p]
            f[p, q] = s
    return f

import numpy as np


def _mo_transform(coefficients, core, eri):
    C = np.asarray(coefficients, dtype=float)
    Hc = np.asarray(core, dtype=float)
    W = np.asarray(eri, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if Hc.shape != C.shape or W.shape != C.shape * 2:
        raise ValueError("core and eri must match the basis size")
    h = C.T @ Hc @ C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, W, C, C, optimize=True)
    return h, V


def _families(coefficients, core, eri):
    h, V = _mo_transform(coefficients, core, eri)
    J = np.einsum('ppqq->pq', V).copy()
    K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0 * J - K]), V


def _unpack(families):
    F = np.asarray(families, dtype=float)
    if F.ndim != 3 or F.shape[0] != 5 or F.shape[1] != F.shape[2]:
        raise ValueError("families must have shape (5, N, N)")
    return F[0], F[1], F[2], F[3], F[4]


def _occupations(gaps):
    w = np.asarray(gaps, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("gaps must be a 1D array with one entry per bond space")
    eta = np.sqrt(w * w + 1.0)
    n = np.empty(2 * w.size)
    n[0::2] = 1.0 + w / eta
    n[1::2] = 1.0 - w / eta
    return n, eta


def _orbital_energies(families, gaps):
    h, J, K, L, G = _unpack(families)
    n, eta = _occupations(gaps)
    N = h.shape[0]
    eps = np.empty(N)
    for p in range(N):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(N):
            if q // 2 == p // 2:
                continue
            s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps


def _t_intermediates(families):
    h, J, K, L, G = _unpack(families)
    N = h.shape[0]
    t = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            base = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
            t[p, q] = base if p // 2 == q // 2 else base - 0.5 * G[p, q]
    return t


def _reduced_g(families, gaps):
    h, J, K, L, G = _unpack(families)
    w = np.asarray(gaps, dtype=float)
    n, eta = _occupations(w)
    M, N = w.size, h.shape[0]
    g = np.zeros((M, M))
    gm = np.zeros((M, N))
    gmm = np.zeros((M, M))
    for a in range(M):
        for p in range(N):
            gm[a, p] = 0.5 * (w[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])
        for b in range(M):
            g[a, b] = 0.5 * (G[2*a, 2*b] + G[2*a, 2*b+1] + G[2*a+1, 2*b] + G[2*a+1, 2*b+1])
            gmm[a, b] = 0.5 * (w[a] * w[b] / (eta[a] * eta[b])) * (
                G[2*a, 2*b] - G[2*a, 2*b+1] - G[2*a+1, 2*b] + G[2*a+1, 2*b+1])
    return g, gm, gmm


def _accumulate(pairs):
    total = 0.0
    for dE, c in pairs:
        if dE <= 1e-6:
            raise ValueError("an excitation energy is not positive; the reference is not stable")
        total -= c * c / dE
    return float(total)


def _context(coefficients, core, eri, gaps):
    fam, V = _families(coefficients, core, eri)
    w = np.asarray(gaps, dtype=float)
    if fam.shape[1] != 2 * w.size:
        raise ValueError("gaps must hold one amplitude per valence bond space")
    n, eta = _occupations(w)
    g, gm, gmm = _reduced_g(fam, w)
    return dict(fam=fam, V=V, w=w, n=n, eta=eta, eps=_orbital_energies(fam, w),
                t=_t_intermediates(fam), J=fam[1], K=fam[2], L=fam[3], G=fam[4],
                g=g, gm=gm, gmm=gmm, M=w.size)


def _transfer_energy(cx, a, mu, b, nu):
    n, eta, eps, t, L, G = cx['n'], cx['eta'], cx['eps'], cx['t'], cx['L'], cx['G']
    gm, gmm = cx['gm'], cx['gmm']
    o = lambda x, y: 2 * x + y
    return (eta[a] * L[o(a,0), o(a,1)] + eta[b] * L[o(b,0), o(b,1)] + t[o(a,mu), o(b,nu)]
            + eps[o(b,1-nu)] - eps[o(a,1-mu)] + G[o(b,0), o(b,1)]
            + gmm[a, b] - gm[a, o(b,1-nu)] + gm[b, o(a,1-mu)] - 0.5 * G[o(a,1-mu), o(b,1-nu)])


def single_excitation_en2(coefficients, core, eri, gaps, fock):
    cx = _context(coefficients, core, eri, gaps)
    f = np.asarray(fock, dtype=float)
    if f.shape != (2 * cx['M'], 2 * cx['M']):
        raise ValueError("fock must be square with two orbitals per bond space")
    n, eta, eps, t, L, V = cx['n'], cx['eta'], cx['eps'], cx['t'], cx['L'], cx['V']
    M = cx['M']
    SQ = np.sqrt
    o = lambda x, y: 2 * x + y
    out = []
    for a in range(M):
        out.append((2 * eta[a] * L[o(a,0), o(a,1)],
                    (eps[o(a,0)] - eps[o(a,1)] + cx['w'][a] * L[o(a,0), o(a,1)]) / eta[a]))
        out.append((eta[a] * L[o(a,0), o(a,1)] + t[o(a,0), o(a,1)],
                    (f[o(a,0), o(a,1)] - f[o(a,1), o(a,0)]) / (SQ(n[o(a,0)]) + SQ(n[o(a,1)]))))
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            for mu in (0, 1):
                for nu in (0, 1):
                    rhs = (f[o(a,mu), o(b,nu)]
                           + n[o(a,mu)] * SQ(n[o(b,nu)] / n[o(b,1-nu)])
                           * V[o(b,nu), o(b,1-nu), o(a,mu), o(b,1-nu)]
                           + 0.5 * (2 * V[o(b,1-nu), o(b,1-nu), o(a,mu), o(b,nu)]
                                    - V[o(b,1-nu), o(b,nu), o(a,mu), o(b,1-nu)]
                                    - V[o(b,nu), o(b,nu), o(a,mu), o(b,nu)])
                           * n[o(a,mu)] * n[o(b,nu)])
                    c = rhs / (((-1) ** (mu + nu + 1)) * SQ(2 * n[o(a,mu)] / n[o(b,1-nu)]))
                    out.append((_transfer_energy(cx, a, mu, b, nu), c))
    return _accumulate(out)

import numpy as np


def _mo_transform(coefficients, core, eri):
    C = np.asarray(coefficients, dtype=float)
    Hc = np.asarray(core, dtype=float)
    W = np.asarray(eri, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if Hc.shape != C.shape or W.shape != C.shape * 2:
        raise ValueError("core and eri must match the basis size")
    h = C.T @ Hc @ C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, W, C, C, optimize=True)
    return h, V


def _families(coefficients, core, eri):
    h, V = _mo_transform(coefficients, core, eri)
    J = np.einsum('ppqq->pq', V).copy()
    K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0 * J - K]), V


def _unpack(families):
    F = np.asarray(families, dtype=float)
    if F.ndim != 3 or F.shape[0] != 5 or F.shape[1] != F.shape[2]:
        raise ValueError("families must have shape (5, N, N)")
    return F[0], F[1], F[2], F[3], F[4]


def _occupations(gaps):
    w = np.asarray(gaps, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("gaps must be a 1D array with one entry per bond space")
    eta = np.sqrt(w * w + 1.0)
    n = np.empty(2 * w.size)
    n[0::2] = 1.0 + w / eta
    n[1::2] = 1.0 - w / eta
    return n, eta


def _orbital_energies(families, gaps):
    h, J, K, L, G = _unpack(families)
    n, eta = _occupations(gaps)
    N = h.shape[0]
    eps = np.empty(N)
    for p in range(N):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(N):
            if q // 2 == p // 2:
                continue
            s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps


def _t_intermediates(families):
    h, J, K, L, G = _unpack(families)
    N = h.shape[0]
    t = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            base = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
            t[p, q] = base if p // 2 == q // 2 else base - 0.5 * G[p, q]
    return t


def _reduced_g(families, gaps):
    h, J, K, L, G = _unpack(families)
    w = np.asarray(gaps, dtype=float)
    n, eta = _occupations(w)
    M, N = w.size, h.shape[0]
    g = np.zeros((M, M))
    gm = np.zeros((M, N))
    gmm = np.zeros((M, M))
    for a in range(M):
        for p in range(N):
            gm[a, p] = 0.5 * (w[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])
        for b in range(M):
            g[a, b] = 0.5 * (G[2*a, 2*b] + G[2*a, 2*b+1] + G[2*a+1, 2*b] + G[2*a+1, 2*b+1])
            gmm[a, b] = 0.5 * (w[a] * w[b] / (eta[a] * eta[b])) * (
                G[2*a, 2*b] - G[2*a, 2*b+1] - G[2*a+1, 2*b] + G[2*a+1, 2*b+1])
    return g, gm, gmm


def _accumulate(pairs):
    total = 0.0
    for dE, c in pairs:
        if dE <= 1e-6:
            raise ValueError("an excitation energy is not positive; the reference is not stable")
        total -= c * c / dE
    return float(total)


def _context(coefficients, core, eri, gaps):
    fam, V = _families(coefficients, core, eri)
    w = np.asarray(gaps, dtype=float)
    if fam.shape[1] != 2 * w.size:
        raise ValueError("gaps must hold one amplitude per valence bond space")
    n, eta = _occupations(w)
    g, gm, gmm = _reduced_g(fam, w)
    return dict(fam=fam, V=V, w=w, n=n, eta=eta, eps=_orbital_energies(fam, w),
                t=_t_intermediates(fam), J=fam[1], K=fam[2], L=fam[3], G=fam[4],
                g=g, gm=gm, gmm=gmm, M=w.size)


def _transfer_energy(cx, a, mu, b, nu):
    n, eta, eps, t, L, G = cx['n'], cx['eta'], cx['eps'], cx['t'], cx['L'], cx['G']
    gm, gmm = cx['gm'], cx['gmm']
    o = lambda x, y: 2 * x + y
    return (eta[a] * L[o(a,0), o(a,1)] + eta[b] * L[o(b,0), o(b,1)] + t[o(a,mu), o(b,nu)]
            + eps[o(b,1-nu)] - eps[o(a,1-mu)] + G[o(b,0), o(b,1)]
            + gmm[a, b] - gm[a, o(b,1-nu)] + gm[b, o(a,1-mu)] - 0.5 * G[o(a,1-mu), o(b,1-nu)])


def double_excitation_en2(coefficients, core, eri, gaps):
    C = np.asarray(coefficients, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if 2 * np.size(gaps) != C.shape[0]:
        raise ValueError("gaps must hold one amplitude per valence bond space")
    cx = _context(coefficients, core, eri, gaps)
    n, eta, t, L, K, V = cx['n'], cx['eta'], cx['t'], cx['L'], cx['K'], cx['V']
    G = cx['G']
    gm, gmm = cx['gm'], cx['gmm']
    M = cx['M']
    SQ = np.sqrt
    o = lambda x, y: 2 * x + y
    swapE = [2 * eta[a] * L[o(a,0), o(a,1)] for a in range(M)]
    splitE = [eta[a] * L[o(a,0), o(a,1)] + t[o(a,0), o(a,1)] for a in range(M)]
    out = []
    for a in range(M):
        for b in range(a + 1, M):
            out.append((swapE[a] + swapE[b] + 4 * gmm[a, b],
                        0.5 / (eta[a] * eta[b]) * sum(((-1) ** (mu + nu)) * G[o(a,mu), o(b,nu)]
                                                      for mu in (0, 1) for nu in (0, 1))))
            E4 = splitE[a] + splitE[b] + gmm[a, b]
            c4 = ((SQ(n[o(a,0)]) - SQ(n[o(a,1)])) * (SQ(n[o(b,0)]) - SQ(n[o(b,1)]))
                  * V[o(a,0), o(a,1), o(b,0), o(b,1)]
                  - 0.5 * (SQ(n[o(a,0)] * n[o(b,0)]) + SQ(n[o(a,1)] * n[o(b,1)]))
                  * V[o(a,0), o(b,1), o(b,0), o(a,1)]
                  + 0.5 * (SQ(n[o(a,1)] * n[o(b,0)]) + SQ(n[o(a,0)] * n[o(b,1)]))
                  * V[o(a,1), o(b,1), o(b,0), o(a,0)])
            out.append((E4, c4))
            E4b = (E4 + K[o(a,0), o(b,0)] + K[o(a,0), o(b,1)] + K[o(a,1), o(b,0)]
                   + K[o(a,1), o(b,1)] - 2 * K[o(a,0), o(a,1)] - 2 * K[o(b,0), o(b,1)])
            c4b = (-SQ(3.0) / 2 * (SQ(n[o(a,0)] * n[o(b,0)]) + SQ(n[o(a,1)] * n[o(b,1)]))
                   * V[o(a,0), o(b,1), o(b,0), o(a,1)]
                   - SQ(3.0) / 2 * (SQ(n[o(a,1)] * n[o(b,0)]) + SQ(n[o(a,0)] * n[o(b,1)]))
                   * V[o(a,1), o(b,1), o(b,0), o(a,0)])
            out.append((E4b, c4b))
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            c = (0.5 / eta[a]) * (SQ(n[o(b,0)]) - SQ(n[o(b,1)])) * sum(
                ((-1) ** mu) * (2 * V[o(a,mu), o(a,mu), o(b,0), o(b,1)]
                                - V[o(a,mu), o(b,1), o(b,0), o(a,mu)]) for mu in (0, 1))
            out.append((swapE[a] + splitE[b] + 2 * gmm[a, b], c))
    for a in range(M):
        for b in range(M):
            for d in range(M):
                if len({a, b, d}) < 3:
                    continue
                for nu in (0, 1):
                    for lam in (0, 1):
                        tE = _transfer_energy(cx, b, nu, d, lam)
                        shift = gmm[a, b] + gmm[a, d] + gm[a, o(b,1-nu)] - gm[a, o(d,1-lam)]
                        sgn = (-1) ** (nu + lam + 1)
                        pre = sgn / (2 * SQ(2.0)) * SQ(n[o(b,nu)] * n[o(d,1-lam)])
                        c_swap = (pre / eta[a]) * sum(
                            ((-1) ** mu) * (2 * V[o(b,nu), o(d,lam), o(a,mu), o(a,mu)]
                                            - V[o(a,mu), o(d,lam), o(b,nu), o(a,mu)])
                            for mu in (0, 1))
                        out.append((swapE[a] + tE + 2 * shift, c_swap))
                        c_split = pre * sum(
                            ((-1) ** mu) * SQ(n[o(a,mu)])
                            * (2 * V[o(b,nu), o(d,lam), o(a,mu), o(a,1-mu)]
                               - V[o(a,mu), o(d,lam), o(b,nu), o(a,1-mu)]) for mu in (0, 1))
                        E4 = splitE[a] + tE + shift
                        out.append((E4, c_split))
                        E4b = (E4 + K[o(a,0), o(b,nu)] + K[o(a,0), o(d,lam)]
                               + K[o(a,1), o(b,nu)] + K[o(a,1), o(d,lam)]
                               - 2 * K[o(a,0), o(a,1)] - 2 * K[o(b,nu), o(d,lam)])
                        c4b = (((-1) ** (nu + lam)) / 2) * SQ(1.5) * SQ(n[o(b,nu)] * n[o(d,1-lam)]) * sum(
                            SQ(n[o(a,mu)]) * V[o(a,mu), o(d,lam), o(b,nu), o(a,1-mu)] for mu in (0, 1))
                        out.append((E4b, c4b))
    return _accumulate(out)

import numpy as np


def _mo_transform(coefficients, core, eri):
    C = np.asarray(coefficients, dtype=float)
    Hc = np.asarray(core, dtype=float)
    W = np.asarray(eri, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if Hc.shape != C.shape or W.shape != C.shape * 2:
        raise ValueError("core and eri must match the basis size")
    h = C.T @ Hc @ C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, W, C, C, optimize=True)
    return h, V


def _families(coefficients, core, eri):
    h, V = _mo_transform(coefficients, core, eri)
    J = np.einsum('ppqq->pq', V).copy()
    K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0 * J - K]), V


def _unpack(families):
    F = np.asarray(families, dtype=float)
    if F.ndim != 3 or F.shape[0] != 5 or F.shape[1] != F.shape[2]:
        raise ValueError("families must have shape (5, N, N)")
    return F[0], F[1], F[2], F[3], F[4]


def _occupations(gaps):
    w = np.asarray(gaps, dtype=float)
    if w.ndim != 1 or w.size < 1:
        raise ValueError("gaps must be a 1D array with one entry per bond space")
    eta = np.sqrt(w * w + 1.0)
    n = np.empty(2 * w.size)
    n[0::2] = 1.0 + w / eta
    n[1::2] = 1.0 - w / eta
    return n, eta


def _orbital_energies(families, gaps):
    h, J, K, L, G = _unpack(families)
    n, eta = _occupations(gaps)
    N = h.shape[0]
    eps = np.empty(N)
    for p in range(N):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(N):
            if q // 2 == p // 2:
                continue
            s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps


def _t_intermediates(families):
    h, J, K, L, G = _unpack(families)
    N = h.shape[0]
    t = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            base = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
            t[p, q] = base if p // 2 == q // 2 else base - 0.5 * G[p, q]
    return t


def _reduced_g(families, gaps):
    h, J, K, L, G = _unpack(families)
    w = np.asarray(gaps, dtype=float)
    n, eta = _occupations(w)
    M, N = w.size, h.shape[0]
    g = np.zeros((M, M))
    gm = np.zeros((M, N))
    gmm = np.zeros((M, M))
    for a in range(M):
        for p in range(N):
            gm[a, p] = 0.5 * (w[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])
        for b in range(M):
            g[a, b] = 0.5 * (G[2*a, 2*b] + G[2*a, 2*b+1] + G[2*a+1, 2*b] + G[2*a+1, 2*b+1])
            gmm[a, b] = 0.5 * (w[a] * w[b] / (eta[a] * eta[b])) * (
                G[2*a, 2*b] - G[2*a, 2*b+1] - G[2*a+1, 2*b] + G[2*a+1, 2*b+1])
    return g, gm, gmm


def _accumulate(pairs):
    total = 0.0
    for dE, c in pairs:
        if dE <= 1e-6:
            raise ValueError("an excitation energy is not positive; the reference is not stable")
        total -= c * c / dE
    return float(total)


def _context(coefficients, core, eri, gaps):
    fam, V = _families(coefficients, core, eri)
    w = np.asarray(gaps, dtype=float)
    if fam.shape[1] != 2 * w.size:
        raise ValueError("gaps must hold one amplitude per valence bond space")
    n, eta = _occupations(w)
    g, gm, gmm = _reduced_g(fam, w)
    return dict(fam=fam, V=V, w=w, n=n, eta=eta, eps=_orbital_energies(fam, w),
                t=_t_intermediates(fam), J=fam[1], K=fam[2], L=fam[3], G=fam[4],
                g=g, gm=gm, gmm=gmm, M=w.size)


def _transfer_energy(cx, a, mu, b, nu):
    n, eta, eps, t, L, G = cx['n'], cx['eta'], cx['eps'], cx['t'], cx['L'], cx['G']
    gm, gmm = cx['gm'], cx['gmm']
    o = lambda x, y: 2 * x + y
    return (eta[a] * L[o(a,0), o(a,1)] + eta[b] * L[o(b,0), o(b,1)] + t[o(a,mu), o(b,nu)]
            + eps[o(b,1-nu)] - eps[o(a,1-mu)] + G[o(b,0), o(b,1)]
            + gmm[a, b] - gm[a, o(b,1-nu)] + gm[b, o(a,1-mu)] - 0.5 * G[o(a,1-mu), o(b,1-nu)])


def pair_transfer_en2(coefficients, core, eri, gaps):
    C = np.asarray(coefficients, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coefficients must be square")
    if 2 * np.size(gaps) != C.shape[0]:
        raise ValueError("gaps must hold one amplitude per valence bond space")
    cx = _context(coefficients, core, eri, gaps)
    n, eta, eps, t, L, V = cx['n'], cx['eta'], cx['eps'], cx['t'], cx['L'], cx['V']
    G, K = cx['G'], cx['K']
    g, gm, gmm = cx['g'], cx['gm'], cx['gmm']
    M = cx['M']
    SQ = np.sqrt
    o = lambda x, y: 2 * x + y
    out = []
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            E = (eta[a] * L[o(a,0), o(a,1)] + eta[b] * L[o(b,0), o(b,1)]
                 + eps[o(b,0)] + eps[o(b,1)] - eps[o(a,0)] - eps[o(a,1)]
                 + 2 * G[o(b,0), o(b,1)] - g[a, b] + gmm[a, b]
                 - gm[a, o(b,0)] - gm[a, o(b,1)] + gm[b, o(a,0)] + gm[b, o(a,1)])
            c = 0.5 * sum(((-1) ** (mu + nu + 1)) * L[o(a,mu), o(b,nu)]
                          * SQ(n[o(a,mu)] * n[o(b,1-nu)]) for mu in (0, 1) for nu in (0, 1))
            out.append((E, c))
    for a in range(M):
        for b in range(a + 1, M):
            for d in range(M):
                if d in (a, b):
                    continue
                for mu in (0, 1):
                    for nu in (0, 1):
                        B = (eta[a] * L[o(a,0), o(a,1)] + eta[b] * L[o(b,0), o(b,1)]
                             + eta[d] * L[o(d,0), o(d,1)] + t[o(a,mu), o(b,nu)]
                             + gmm[a, b] + gmm[a, d] + gmm[b, d]
                             + 0.5 * G[o(a,1-mu), o(b,1-nu)]
                             - 0.5 * G[o(a,1-mu), o(d,0)] - 0.5 * G[o(a,1-mu), o(d,1)]
                             - 0.5 * G[o(b,1-nu), o(d,0)] - 0.5 * G[o(b,1-nu), o(d,1)])
                        Cc = (eps[o(d,0)] + eps[o(d,1)] - eps[o(a,1-mu)] - eps[o(b,1-nu)]
                              - gm[a, o(d,0)] - gm[a, o(d,1)] - gm[b, o(d,0)] - gm[b, o(d,1)]
                              + gm[b, o(a,1-mu)] + gm[d, o(a,1-mu)]
                              + gm[a, o(b,1-nu)] + gm[d, o(b,1-nu)])
                        c1 = 0.5 * ((-1) ** (mu + nu + 1)) * SQ(n[o(a,mu)] * n[o(b,nu)]) * (
                            V[o(a,mu), o(d,1), o(b,nu), o(d,1)] * SQ(n[o(d,0)])
                            - V[o(a,mu), o(d,0), o(b,nu), o(d,0)] * SQ(n[o(d,1)]))
                        out.append((B + Cc + 2 * G[o(d,0), o(d,1)], c1))
                        c2 = 0.5 * ((-1) ** (mu + nu)) * SQ(n[o(a,1-mu)] * n[o(b,1-nu)]) * (
                            V[o(d,0), o(a,mu), o(d,0), o(b,nu)] * SQ(n[o(d,0)])
                            - V[o(d,1), o(a,mu), o(d,1), o(b,nu)] * SQ(n[o(d,1)]))
                        out.append((B - Cc + G[o(a,0), o(a,1)] + G[o(b,0), o(b,1)], c2))

    for a in range(M):
        for b in range(a + 1, M):
            for c in range(M):
                if c in (a, b):
                    continue
                for d in range(c + 1, M):
                    if d in (a, b):
                        continue
                    base = (eta[a] * L[o(a,0), o(a,1)] + eta[b] * L[o(b,0), o(b,1)]
                            + eta[c] * L[o(c,0), o(c,1)] + eta[d] * L[o(d,0), o(d,1)]
                            + gmm[a, b] + gmm[a, c] + gmm[a, d]
                            + gmm[b, c] + gmm[b, d] + gmm[c, d]
                            + G[o(c,0), o(c,1)] + G[o(d,0), o(d,1)])
                    for mu in (0, 1):
                        for nu in (0, 1):
                            for lam in (0, 1):
                                for kap in (0, 1):
                                    am, bn = o(a,mu), o(b,nu)
                                    cl, dk = o(c,lam), o(d,kap)
                                    ap, bp = o(a,1-mu), o(b,1-nu)
                                    cp, dp = o(c,1-lam), o(d,1-kap)
                                    E4 = (base + t[am, bn] + t[cl, dk]
                                          + eps[cp] + eps[dp] - eps[ap] - eps[bp]
                                          + 0.5 * G[ap, bp] + 0.5 * G[cp, dp]
                                          - gm[a, cp] - gm[b, cp] - gm[d, cp]
                                          - gm[a, dp] - gm[b, dp] - gm[c, dp]
                                          + gm[b, ap] + gm[c, ap] + gm[d, ap]
                                          + gm[a, bp] + gm[c, bp] + gm[d, bp]
                                          - 0.5 * G[ap, cp] - 0.5 * G[ap, dp]
                                          - 0.5 * G[bp, cp] - 0.5 * G[bp, dp])
                                    root = SQ(n[am] * n[bn] * n[cp] * n[dp])
                                    sgn = (-1) ** (mu + nu + lam + kap)
                                    direct, cross = V[am, cl, bn, dk], V[am, dk, bn, cl]
                                    out.append((E4, -0.25 * sgn * root * (direct + cross)))
                                    shift = (K[am, cl] + K[am, dk] + K[bn, cl] + K[bn, dk]
                                             - 2.0 * K[am, bn] - 2.0 * K[cl, dk])
                                    out.append((E4 + shift,
                                                0.25 * SQ(3.0) * sgn * root * (direct - cross)))
    return _accumulate(out)

import numpy as np


def total_pp_en2_energy(centres=(0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6),
                                exponent=0.30, charges=(1.0,) * 8, n_units=4):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must hold an even number of positions")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    if np.size(charges) != R.size:
        raise ValueError("charges must hold one entry per centre")
    if 2 * int(n_units) != R.size:
        raise ValueError("n_units must be half the number of centres")
    stack = ao_overlap_and_core(centres, exponent, charges)
    overlap, core = stack[0], stack[1]
    eri = ao_two_electron_integrals(centres, exponent)
    C = vbs_orbital_coefficients(overlap, n_units)
    fam = pair_integral_families(C, core, eri)
    gaps = optimal_pair_gaps(fam, n_units)
    fock = generalized_fock_matrix(C, core, eri, gaps)
    reference = pp_reference_energy(fam, gaps)
    singles = single_excitation_en2(C, core, eri, gaps, fock)
    doubles = double_excitation_en2(C, core, eri, gaps)
    pairs = pair_transfer_en2(C, core, eri, gaps)
    correction = singles + doubles + pairs
    R = np.asarray(centres, dtype=float)
    Z = np.asarray(charges, dtype=float)
    repulsion = 0.0
    for i in range(R.size):
        for j in range(i + 1, R.size):
            repulsion += Z[i] * Z[j] / abs(R[i] - R[j])
    return float(reference + correction + repulsion)
SCICODE_GOLD_EOF
