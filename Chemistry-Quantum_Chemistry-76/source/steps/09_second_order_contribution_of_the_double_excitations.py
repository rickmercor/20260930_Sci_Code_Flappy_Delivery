"""
Return the second order Epstein-Nesbet contribution of every double excitation, including both seniority four singlets of a double split.

Doubles combine two of the single excitation moves across different bond

spaces. Two swaps, a swap with a split, a split with a split, and a swap or a

split accompanied by a single electron transfer all appear, and their energies

are sums of the corresponding single excitation energies plus a short

correction built from the reduced contractions of the mean field combination

over the gaps and pair amplitudes. Splitting two spaces at once leaves four

singly occupied orbitals, which support two independent singlets: the first

pairs the orbitals as the excitation groups them, and the second is its

orthogonal complement. The two differ in their exchange elements only, so the

second excitation energy is the first plus the four interbond exchange

integrals minus twice each intrabond exchange integral, and their couplings to

the reference are different combinations of the same two exchange type

integrals.

Returns
-------
float, the second order contribution of the double excitation family in hartree as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def double_excitation_en2(coefficients: np.ndarray, core: np.ndarray,
                          eri: np.ndarray, gaps: np.ndarray) -> float:
    '''Second order energy from the double excitation family.

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
        Contribution of the double excitation family in hartree, negative.
    
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


def _oracle_double_excitation_en2(coefficients, core, eri, gaps):
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
            "call": "round(double_excitation_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_double_excitation_en2(C, Hc, eri, w), 12)",
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
            "call": "round(double_excitation_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_double_excitation_en2(C, Hc, eri, w), 12)",
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
S, Hc, eri, C = _model([0.0, 2.8, 6.2, 8.9], 0.85)
w = np.array([1.6, 1.9])
""",
            "call": "round(double_excitation_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_double_excitation_en2(C, Hc, eri, w), 12)",
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
            "call": "round(double_excitation_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_double_excitation_en2(C, Hc, eri, w), 12)",
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
w = np.array([4.0, 4.0, 4.0])
""",
            "call": "round(double_excitation_en2(C, Hc, eri, w), 12)",
            "gold_call": "round(_oracle_double_excitation_en2(C, Hc, eri, w), 12)",
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
w = np.ones(4)

def run_model():
    try:
        double_excitation_en2(C, Hc, eri, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_double_excitation_en2(C, Hc, eri, w)
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
        double_excitation_en2(C, Hc, eri[:4], w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_double_excitation_en2(C, Hc, eri[:4], w)
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
