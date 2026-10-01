"""
Return the generalised Fock matrix of the pair reference, which is not symmetric because the orbitals are frozen rather than optimised.

The gradient of the energy with respect to an orbital rotation is twice the

antisymmetric part of a generalised Fock matrix, and for a state of zero

seniority that matrix reduces to the one electron matrix weighted by the

occupation of its row index, plus a mean field term contracted with the direct

two particle density, plus a pairing term contracted with the pair density. For

the reference here the direct elements factorise into products of occupation

numbers except inside a bond space where they vanish, the pair elements are

minus the reciprocal of the pair amplitude inside a bond space and zero

between spaces, and the diagonal conventions are that the direct element is

zero while the pair element is the occupation number. Because the orbitals are

frozen rather than optimised this matrix is not symmetric, and its

antisymmetric part is what keeps single electron transfers coupled to the

reference.

Returns
-------
np.ndarray of shape (N, N): the generalised Fock matrix of the reference, not symmetric in general
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generalized_fock_matrix(coefficients: np.ndarray, core: np.ndarray,
                            eri: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    '''Generalised Fock matrix of the pair reference.

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
    fock : np.ndarray
        Array of shape (N, N). It is not symmetric in general.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, if `core` and `eri` do not
        match its dimension, or if `gaps` does not hold one amplitude per bond
        space.
'''
    return fock

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


def _oracle_generalized_fock_matrix(coefficients, core, eri, gaps):
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
            "call": "np.round(generalized_fock_matrix(C, Hc, eri, w), 10)",
            "gold_call": "np.round(_oracle_generalized_fock_matrix(C, Hc, eri, w), 10)",
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
            "call": "np.round(generalized_fock_matrix(C, Hc, eri, w), 10)",
            "gold_call": "np.round(_oracle_generalized_fock_matrix(C, Hc, eri, w), 10)",
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
w = np.array([0.7, 1.9])
""",
            "call": "np.round(generalized_fock_matrix(C, Hc, eri, w), 10)",
            "gold_call": "np.round(_oracle_generalized_fock_matrix(C, Hc, eri, w), 10)",
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
w = np.zeros(3)
""",
            "call": "np.round(generalized_fock_matrix(C, Hc, eri, w), 10)",
            "gold_call": "np.round(_oracle_generalized_fock_matrix(C, Hc, eri, w), 10)",
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
S, Hc, eri, C = _model([0.0, 6.0, 14.0, 20.0], 0.30)
w = np.array([40.0, 40.0])
""",
            "call": "np.round(generalized_fock_matrix(C, Hc, eri, w), 10)",
            "gold_call": "np.round(_oracle_generalized_fock_matrix(C, Hc, eri, w), 10)",
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
w = np.ones(2)

def run_model():
    try:
        generalized_fock_matrix(C, Hc, eri, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generalized_fock_matrix(C, Hc, eri, w)
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
        generalized_fock_matrix(C, Hc[:4, :4], eri, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generalized_fock_matrix(C, Hc[:4, :4], eri, w)
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
