"""
Build the family (neighbourhood) indicator of the point cloud (the paper's Eq. 1): entry (i, j) is 1.0 when point j belongs to the family of point i under that point's own horizon, and 0.0 otherwise. A point is never its own family member. With spatially varying horizons this relation is NOT symmetric, and the source relies on that asymmetry.

Each point interacts with the neighbours inside its own horizon; when the horizon varies in space the membership relation loses the symmetry it has in the uniform-horizon case.

Returns
-------
return (N, N) float64: family membership indicator, 1.0 if in family
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def families(X, delta):
    """X: (N,) reference coordinates; delta: (N,) per-point horizons.
    Returns (N, N) float64: 1.0 where point j lies in the family of point i
    under point i's own horizon, else 0.0; the diagonal is 0.0 (paper Eq. 1).
    The result is in general not symmetric."""
    n = np.asarray(X).size
    return np.zeros((n, n))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: family membership under spatially varying horizons (Eq. 1)."""

import numpy as np


def _oracle_families(X, delta):
    X = np.asarray(X, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    R = np.abs(X[None, :] - X[:, None])
    H = (R <= d[:, None]).astype(np.float64)
    np.fill_diagonal(H, 0.0)
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p', "call": 'families(X, delta)', "gold_call": '_oracle_families(X, delta)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p', "call": 'families(X, delta)', "gold_call": '_oracle_families(X, delta)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p', "call": 'families(X, delta)', "gold_call": '_oracle_families(X, delta)', "tol": 1e-12},
    ]
