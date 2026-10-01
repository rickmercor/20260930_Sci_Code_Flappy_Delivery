"""
Compute the bond stretch for every ordered pair of points from the reference coordinates and the displacement field (the paper's Eq. 22). The deformed position of a point is its reference coordinate plus its displacement. The diagonal is zero.

The stretch is the relative change in length of the bond joining two points and is the single kinematic quantity the bond-based constitutive model depends on.

Returns
-------
return (N, N) float64: bond stretch for every point pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bond_stretch(X, u):
    """X: (N,) reference coordinates; u: (N,) displacements. Returns (N, N)
    float64 with the bond stretch of every ordered pair (paper Eq. 22), zero on
    the diagonal."""
    n = np.asarray(X).size
    return np.zeros((n, n))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: bond stretch (Eq. 22)."""

import numpy as np


def _oracle_bond_stretch(X, u):
    X = np.asarray(X, dtype=np.float64)
    x = X + np.asarray(u, dtype=np.float64)
    Xij = X[None, :] - X[:, None]
    xij = x[None, :] - x[:, None]
    d0 = np.abs(Xij)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (np.abs(xij) - d0) / d0
    np.fill_diagonal(s, 0.0)
    return s

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p', "call": 'bond_stretch(X, u)', "gold_call": '_oracle_bond_stretch(X, u)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p', "call": 'bond_stretch(X, u)', "gold_call": '_oracle_bond_stretch(X, u)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p', "call": 'bond_stretch(X, u)', "gold_call": '_oracle_bond_stretch(X, u)', "tol": 1e-10},
    ]
