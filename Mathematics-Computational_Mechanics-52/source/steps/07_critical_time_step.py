"""
Return the source's estimate of the critical time step for explicit integration of this point cloud (the paper's Eq. 54), taken as the smallest per-point value over all points. The per-point value is built from the point's density and a sum over its family involving the bond micro-moduli, the neighbour volumes and the reference bond lengths, with the horizon convention of Eq. 24. Use E = 1.0 and A = 1.0. Recover the exact expression from the paper.

Explicit peridynamic time integration is only stable below this bound; with a spatially varying horizon it differs markedly between the coarse and the refined region, which is what motivates asynchronous time stepping.

Returns
-------
return float: the source's critical time step
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def critical_time_step(X, V, delta, H, rho):
    """X: (N,) reference coordinates; V: (N,) point volumes; delta: (N,)
    horizons; H: (N, N) family indicator; rho: mass density. Returns float:
    the source's critical time step for the whole point cloud (paper Eq. 54),
    using E = 1.0 and A = 1.0. Raises ValueError if a point has no family."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: critical time step (Eq. 54)."""

import numpy as np

_E = 1.0
_A = 1.0


def _oracle_critical_time_step(X, V, delta, H, rho):
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    c = _oracle_micro_modulus(delta, _E, _A)
    Hb = (np.asarray(H, dtype=np.float64) > 0)
    R = np.abs(X[None, :] - X[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(Hb, c[None, :] * V[None, :] / np.where(R > 0, R, 1.0), 0.0)
    den = term.sum(axis=1)
    if np.any(den <= 0):
        raise ValueError("point with no family members")
    return float(np.min(np.sqrt(2.0 * rho / den)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nrho=1.0', "call": 'critical_time_step(X, V, delta, H, rho)', "gold_call": '_oracle_critical_time_step(X, V, delta, H, rho)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nrho=1.0', "call": 'critical_time_step(X, V, delta, H, rho)', "gold_call": '_oracle_critical_time_step(X, V, delta, H, rho)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nrho=2.0', "call": 'critical_time_step(X, V, delta, H, rho)', "gold_call": '_oracle_critical_time_step(X, V, delta, H, rho)', "tol": 1e-12},
    ]
