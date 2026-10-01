"""
Transform the one and two electron integrals into the bond space orbitals and return the one electron matrix together with the direct, exchange, pair-transfer and mean field families.

Transforming the one and two electron integrals into the bond space orbitals

exposes the only combinations the theory ever uses. Writing the two electron

integrals in chemists' notation, the direct family is J = (pp|qq), the exchange

family is K = (pq|qp) and the pair-transfer family is L = (pq|pq); for real

orbitals the last two coincide. The combination G = 2J - K carries the mean

field interaction between orbitals, while the diagonal element L of a bond

space is what binds its electron pair and sets the scale of every excitation

energy in the problem. The step returns the transformed one electron matrix

together with those four families stacked in the order h, J, K, L, G.

Returns
-------
np.ndarray of shape (5, N, N): the transformed one electron matrix followed by the direct, exchange, pair-transfer and 2J-K families
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_integral_families(coefficients: np.ndarray, core: np.ndarray,
                           eri: np.ndarray) -> np.ndarray:
    '''One electron matrix and the direct, exchange, pair-transfer families.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).

    Returns
    -------
    families : np.ndarray
        Array of shape (5, N, N) holding, in order, the transformed one
        electron matrix, the direct family, the exchange family, the
        pair-transfer family and the mean field combination 2J - K.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, or if `core` and `eri` do not
        match its dimension.
'''
    return families

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


def _oracle_pair_integral_families(coefficients, core, eri):
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
""",
            "call": "pair_integral_families(C, Hc, eri)",
            "gold_call": "_oracle_pair_integral_families(C, Hc, eri)",
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
""",
            "call": "pair_integral_families(C, Hc, eri)",
            "gold_call": "_oracle_pair_integral_families(C, Hc, eri)",
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
""",
            "call": "pair_integral_families(C, Hc, eri)",
            "gold_call": "_oracle_pair_integral_families(C, Hc, eri)",
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
""",
            "call": "pair_integral_families(C, Hc, eri)",
            "gold_call": "_oracle_pair_integral_families(C, Hc, eri)",
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
C = np.eye(6)
""",
            "call": "pair_integral_families(C, Hc, eri)",
            "gold_call": "_oracle_pair_integral_families(C, Hc, eri)",
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

def run_model():
    try:
        pair_integral_families(C[:, :4], Hc, eri)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pair_integral_families(C[:, :4], Hc, eri)
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

def run_model():
    try:
        pair_integral_families(C, Hc[:4, :4], eri)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pair_integral_families(C, Hc[:4, :4], eri)
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
