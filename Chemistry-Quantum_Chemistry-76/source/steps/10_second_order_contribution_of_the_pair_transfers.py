"""
Return the second order Epstein-Nesbet contribution of the seniority zero, seniority two and seniority four pair transfers between bond spaces.

Moving an entire electron pair from one bond space to another is not the same

object as a double substitution between determinants, and it needs its own

treatment. In the seniority zero case the donor space is emptied and the

acceptor doubly filled, and the excitation energy collects the pairing penalty

of both spaces, the sum of the two acceptor orbital energies minus the two

donor orbital energies, twice the acceptor mean field integral, and the reduced

contractions that repair the double counting. Its coupling to the reference is

built entirely from pair-transfer integrals between the two spaces weighted by

square roots of occupation numbers. In the seniority two case a third space

supplies or receives the pair while the first two carry an open shell singlet,

which gives two states that share a common part and differ by the sign of a

second part, so both are obtained from one pair of intermediates. Four bond

spaces admit one further case that fewer cannot: a pair transfer accompanied by

a second electron transfer, leaving two spaces holding a single electron and two

holding three, with the four singly occupied orbitals coupled into a singlet in

the two independent ways. Its excitation energy collects the pairing penalty of

all four spaces, the open shell penalty of the donor pair and of the acceptor

pair, the difference of the partner orbital energies, the mean field integral of

both acceptor spaces, and the reduced contractions over all six pairs of spaces,

while the complementary state is once again a fixed exchange update. The

coupling of the first is one quarter of the sum of the two ways of pairing the

four orbitals into an interbond two electron integral, weighted by the square

root of the product of four occupation numbers, and the coupling of the second

is that same prefactor times the square root of three, acting on the difference

of those two integrals rather than their sum.

Returns
-------
float, the second order contribution of the pair transfer family in hartree as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_transfer_en2(coefficients: np.ndarray, core: np.ndarray,
                      eri: np.ndarray, gaps: np.ndarray) -> float:
    '''Second order energy from transferring correlated pairs between spaces.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).

    Returns
    -------
    energy : float
        Contribution of the pair transfer family in hartree, negative.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, if `core` and `eri` do not
        match its dimension, if `gaps` does not hold one amplitude per bond
        space, or if any excitation energy is not strictly positive, meaning at
        or below 1e-6 hartree.
'''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_pair_transfer_en2(coefficients, core, eri, gaps):
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6], 0.30)
w = np.array([1.342660461, 1.143111761, 1.471527444, 0.950077190])
""",
            "call": "round(pair_transfer_en2(C, Hc, eri, w), 14)",
            "gold_call": "round(_oracle_pair_transfer_en2(C, Hc, eri, w), 14)",
        },
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 3.0, 6.6, 9.8, 13.5, 16.4], 0.30)
w = np.array([1.34266686, 1.14315028, 1.45899920])
""",
            "call": "round(pair_transfer_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_pair_transfer_en2(C, Hc, eri, w), 12)",
        },
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 2.0, 4.0, 6.2], 0.30)
fam = _fam(C, Hc, eri)
w = _oracle_optimal_pair_gaps(fam, 2)
""",
            "call": "round(pair_transfer_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_pair_transfer_en2(C, Hc, eri, w), 12)",
        },
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 2.9, 6.4, 9.5, 13.1, 15.9, 19.6, 22.3], 0.22)
w = np.array([1.2, 1.4, 1.1, 1.6])
""",
            "call": "round(pair_transfer_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_pair_transfer_en2(C, Hc, eri, w), 12)",
        },
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 2.8, 4.8, 7.8, 9.8, 13.0], 0.30)
fam = _fam(C, Hc, eri)
w = _oracle_optimal_pair_gaps(fam, 3)
""",
            "call": "round(pair_transfer_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_pair_transfer_en2(C, Hc, eri, w), 12)",
        },
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 3.0, 6.6, 9.8, 13.5, 16.4], 0.30)
w = np.ones(1)

def run_model():
    try:
        pair_transfer_en2(C, Hc, eri, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pair_transfer_en2(C, Hc, eri, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
from scipy.special import erf
def _F0(x):
    x = np.asarray(x, dtype=float); o = np.ones_like(x); m = x > 1e-12
    o[m] = 0.5*np.sqrt(np.pi/x[m])*erf(np.sqrt(x[m])); return o
def _model(X, a):
    R = np.asarray(X, dtype=float); p = 2.0*a; nm = (2.0*a/np.pi)**0.75
    d2 = (R[:, None]-R[None, :])**2; E = np.exp(-0.5*a*d2)
    S = nm*nm*(np.pi/p)**1.5*E; T = 0.5*a*(3.0-a*d2)*S
    P = 0.5*(R[:, None]+R[None, :]); Vne = np.zeros_like(S)
    for c in range(R.size):
        Vne -= (2.0*np.pi/p)*nm*nm*E*_F0(p*(P-R[c])**2)
    pre = 2.0*np.pi**2.5/(p*p*np.sqrt(2.0*p))*nm**4
    Q = (P[:, :, None, None]-P[None, None, :, :])**2
    eri = pre*E[:, :, None, None]*E[None, None, :, :]*_F0(0.5*p*Q)
    ev, U = np.linalg.eigh(S); Xm = U@np.diag(ev**-0.5)@U.T
    B = np.zeros_like(S); s = 1.0/np.sqrt(2.0)
    for u in range(R.size//2):
        B[2*u, 2*u] = s; B[2*u+1, 2*u] = s; B[2*u, 2*u+1] = s; B[2*u+1, 2*u+1] = -s
    return S, T+Vne, eri, Xm@B
def _fam(C, Hc, eri):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    J = np.einsum('ppqq->pq', V).copy(); K = np.einsum('pqqp->pq', V).copy()
    L = np.einsum('pqpq->pq', V).copy()
    return np.stack([h, J, K, L, 2.0*J-K])
def _fockm(C, Hc, eri, w):
    h = C.T@Hc@C
    V = np.einsum('mp,nq,mnab,ar,bs->pqrs', C, C, eri, C, C, optimize=True)
    eta = np.sqrt(np.asarray(w, dtype=float)**2+1.0)
    n = np.empty(2*len(w)); n[0::2] = 1.0+w/eta; n[1::2] = 1.0-w/eta
    N = h.shape[0]; D = np.zeros((N, N)); P = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q: P[p, q] = n[p]
            elif p//2 == q//2: P[p, q] = -1.0/eta[p//2]
            else: D[p, q] = n[p]*n[q]
    f = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            t = h[p, q]*n[p]
            for r in range(N):
                t += 0.5*(2*V[r, r, p, q]-V[r, q, p, r])*D[r, p]
                t += V[r, p, r, q]*P[r, p]
            f[p, q] = t
    return f
S, Hc, eri, C = _model([0.0, 3.0, 6.6, 9.8, 13.5, 16.4], 0.30)
w = np.ones(3)

def run_model():
    try:
        pair_transfer_en2(C, Hc, eri, np.ones((3, 2)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pair_transfer_en2(C, Hc, eri, np.ones((3, 2)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
