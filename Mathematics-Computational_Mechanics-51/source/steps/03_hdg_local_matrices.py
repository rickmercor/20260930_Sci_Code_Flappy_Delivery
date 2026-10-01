"""
Build the five edge matrices of the source's hybrid dual mixed formulation on a single edge, using the Legendre polynomial basis of degree at most p in the arclength parameter, taken component by component so that the degree-of-freedom index is component*(p+1) + mode. Return a stack of five N x N matrices in this order: the matrix of the source's dual bilinear form, the matrix of the source's mixed form coupling the dual and primal variables, the plain mass matrix, the matrix of the source's auxiliary bilinear form used by the first-order-in-time formulation, and the matrix that collects the two endpoint traces. The mixed form contains a derivative contribution together with the tangent coupling of the previous sub-problem; its exact composition is the source's. Integrate exactly with Gauss-Legendre quadrature on 2p+6 points. Raise ValueError unless p >= 0 and the length is positive.

Hybridizable discontinuous Galerkin methods keep the dual and primal variables local to each element and couple elements only through a separate unknown living on the element boundaries. All element-level algebra therefore reduces to a handful of small dense matrices.

Returns
-------
return (5, N, N) float64 with N = 6*(p+1): the edge matrices of the source's bilinear forms
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr):
    """p: polynomial degree; L: edge length; i_hat: (3,) unit tangent;
    Cn, Cm, Cu, Cr: (3, 3) coefficient blocks. Returns (5, N, N) float64 with
    N = 6*(p+1), stacked in the order described above."""
    return np.zeros((5, 6 * (p + 1), 6 * (p + 1)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: hdg_local_matrices."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('timoshenko_operators',):
    if "_oracle_" + _n not in globals() and _n in globals():
        globals()["_oracle_" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _oracle_hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr):
    """Edge matrices of the source's bilinear forms a, b, c, d and the trace form.

    Returns (5, N, N) with N = 6*(p+1), stacked as [A, B, Mass, D, T].
    """
    if p < 0 or L <= 0:
        raise ValueError("need p >= 0 and L > 0")
    ck = ("loc", p, float(L)) + _key(np.asarray(i_hat, float), np.asarray(Cn, float),
                                     np.asarray(Cm, float), np.asarray(Cu, float), np.asarray(Cr, float))
    if ck in _CACHE:
        return _CACHE[ck]
    M, Dx, e0, eL, _x, _w, _V = _basis(p, L)
    Cq, Cy, IX = _oracle_timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)
    I6 = np.eye(6)
    A = np.kron(np.linalg.inv(Cq), M)
    B = np.kron(I6, Dx) + np.kron(IX, M)
    Mass = np.kron(I6, M)
    D = np.kron(np.linalg.inv(Cy), M)
    T0 = np.kron(I6, e0.reshape(1, -1)); TL = np.kron(I6, eL.reshape(1, -1))
    T = T0.T @ T0 + TL.T @ TL
    _CACHE[ck] = np.stack([A, B, Mass, D, T])
    return _CACHE[ck]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\np=2;L=1.3;i_hat=np.array([0.,0.,1.]);Cn=np.eye(3);Cm=np.eye(3);Cu=np.eye(3);Cr=np.eye(3)', "call": 'hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-10},
        {"setup": 'import numpy as np\np=3;L=2.0548046676563256;i_hat=np.array([1.,1.,1.])/np.sqrt(3)\nw=np.array([2.,3.,4.]);w=w/np.linalg.norm(w);O=np.outer(w,w)\nCn=np.eye(3)+0.6*O;Cm=np.eye(3)+0.9*O;Cu=np.eye(3)+0.4*O;Cr=np.eye(3)+0.7*O', "call": 'hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-10},
        {"setup": 'import numpy as np\np=1;L=0.75;i_hat=np.array([0.6,-0.8,0.])\nCn=np.diag([1.,2.,3.]);Cm=np.diag([2.,1.,4.]);Cu=np.diag([1.,1.,2.]);Cr=np.diag([3.,1.,1.])', "call": 'hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-10},
        {"setup": 'import numpy as np\np=4;L=3.1;i_hat=np.array([0.,1.,0.]);Cn=np.eye(3)*2;Cm=np.eye(3)*0.5;Cu=np.eye(3)*1.5;Cr=np.eye(3)*0.8', "call": 'hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-10},
    ]
