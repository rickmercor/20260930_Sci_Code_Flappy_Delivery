"""
Return the electronic energy of the pair reference for a given set of gaps, nuclear repulsion excluded.

Because the reference is a product of independent pair states, its energy takes

a Hartree-Fock-like form: a sum of orbital energies weighted by occupation

numbers, minus a pairing term for each bond space that is the pair-transfer

integral times the geometric mean of the two occupations of that space, minus

the usual double counting correction over the mean field combination 2J - K.

The double counting sum runs over pairs of orbitals belonging to different bond

spaces only, since the interaction inside a bond space is already carried by

the pairing term. The orbital energy of a valence orbital is its diagonal one

electron integral, plus half its diagonal pair-transfer integral, plus a half

weighted sum of 2J - K over the occupations of every other bond space.

Returns
-------
float, the electronic energy of the pair reference in hartree as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pp_reference_energy(families: np.ndarray, gaps: np.ndarray) -> float:
    '''Electronic energy of the pair reference.

    Parameters
    ----------
    families : np.ndarray
        Integral families of shape (5, N, N) in the bond space basis.
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).

    Returns
    -------
    energy : float
        Electronic energy of the reference in hartree, nuclear repulsion
        excluded.
    
    Raises
    ------
    ValueError
        If `families` does not have shape (5, N, N), or if `gaps` is not a one
        dimensional array holding one entry per bond space.
'''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_pp_reference_energy(families, gaps):
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
fam = _fam(C, Hc, eri)
w = np.array([1.342660461, 1.143111761, 1.471527444, 0.950077190])
""",
            "call": "round(pp_reference_energy(fam, w), 9)",
            "gold_call": "round(_oracle_pp_reference_energy(fam, w), 9)",
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
fam = _fam(C, Hc, eri)
w = np.array([1.34266686, 1.14315028, 1.45899920])
""",
            "call": "round(pp_reference_energy(fam, w), 9)",
            "gold_call": "round(_oracle_pp_reference_energy(fam, w), 9)",
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
fam = _fam(C, Hc, eri)
w = np.array([0.4, 2.6, 1.0])
""",
            "call": "round(pp_reference_energy(fam, w), 9)",
            "gold_call": "round(_oracle_pp_reference_energy(fam, w), 9)",
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
fam = _fam(C, Hc, eri)
w = np.array([1.0, 1.0])
""",
            "call": "round(pp_reference_energy(fam, w), 9)",
            "gold_call": "round(_oracle_pp_reference_energy(fam, w), 9)",
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
fam = _fam(C, Hc, eri)
w = np.zeros(3)
""",
            "call": "round(pp_reference_energy(fam, w), 9)",
            "gold_call": "round(_oracle_pp_reference_energy(fam, w), 9)",
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
fam = _fam(C, Hc, eri)
w = np.array([60.0, 60.0, 60.0])
""",
            "call": "round(pp_reference_energy(fam, w), 9)",
            "gold_call": "round(_oracle_pp_reference_energy(fam, w), 9)",
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
fam = _fam(C, Hc, eri)
w = np.ones(2)

def run_model():
    try:
        pp_reference_energy(fam, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pp_reference_energy(fam, w)
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
fam = _fam(C, Hc, eri)
w = np.ones((3, 1))

def run_model():
    try:
        pp_reference_energy(fam, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pp_reference_energy(fam, w)
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
