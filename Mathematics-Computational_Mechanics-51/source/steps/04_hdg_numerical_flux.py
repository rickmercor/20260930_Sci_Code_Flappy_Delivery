"""
Evaluate the source's HDG numerical flux at both endpoints of one edge. Row 0 is the value at the start node of the edge and row 1 the value at the end node. The flux combines the endpoint trace of the dual variable, carried with that endpoint's orientation sign, and a stabilisation contribution proportional to the mismatch between the endpoint trace of the primal variable and the hybrid nodal value there. The exact combination, including which variable the stabilisation penalises, is the source's; recover it from the paper. The 12-vector of hybrid values holds the start-node value in its first six entries and the end-node value in the last six.

The numerical flux is what makes a discontinuous method consistent and stable: it is the single quantity through which neighbouring elements communicate, and the nodal balance of the continuous problem is imposed on it rather than on the raw dual variable.

Returns
-------
return (2, 6) float64: the source's numerical flux at the two endpoints of the edge
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau):
    """q_edge, y_edge: (6*(p+1),) coefficient vectors of the dual and primal
    variables on the edge; lam12: (12,) hybrid nodal values; p, L: degree and
    length; tau: stabilisation parameter. Returns (2, 6) float64."""
    return np.zeros((2, 6))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: hdg_numerical_flux."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



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


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def _oracle_hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau):
    """The source's HDG numerical flux at both endpoints (paper Eq 4.2).

    Returns (2, 6): row 0 at the start node, row 1 at the end node, each equal to
    q(n) nu(n) + tau (y(n) - lamN(n)) with nu = -1 at the start and +1 at the end.
    """
    if p < 0 or L <= 0 or tau < 0:
        raise ValueError("need p >= 0, L > 0 and tau >= 0")
    if np.asarray(lam12).size != 12:
        raise ValueError("lam12 must hold twelve nodal values")
    T0, TL = _traces(p, L)
    lam = np.asarray(lam12, dtype=np.float64)
    q = np.asarray(q_edge, dtype=np.float64); y = np.asarray(y_edge, dtype=np.float64)
    f0 = -1.0 * (T0 @ q) + tau * (T0 @ y - lam[:6])
    fL = +1.0 * (TL @ q) + tau * (TL @ y - lam[6:])
    return np.stack([f0, fL])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\np=2;L=1.3;tau=1.0\nrng=np.random.default_rng(1)\nq_edge=rng.standard_normal(6*(p+1));y_edge=rng.standard_normal(6*(p+1));lam12=rng.standard_normal(12)', "call": 'hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau)', "gold_call": '_oracle_hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau)', "tol": 1e-10},
        {"setup": 'import numpy as np\np=3;L=2.05;tau=2.5\nrng=np.random.default_rng(4)\nq_edge=rng.standard_normal(6*(p+1))*0.3;y_edge=rng.standard_normal(6*(p+1))*2.0;lam12=np.zeros(12)', "call": 'hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau)', "gold_call": '_oracle_hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau)', "tol": 1e-10},
        {"setup": 'import numpy as np\np=1;L=0.8;tau=0.25\nq_edge=np.zeros(6*(p+1));y_edge=np.zeros(6*(p+1));lam12=np.arange(12.0)', "call": 'hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau)', "gold_call": '_oracle_hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau)', "tol": 1e-10},
    ]
