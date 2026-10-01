"""
Return the source's local damage at each material point from the bond damage indicator, the family indicator and the point volumes (the paper's Eq. 30). The damage is zero when every bond of the point is intact and one when all of them are broken, and the source weights the bonds rather than merely counting them. Recover the exact weighting from the paper.

The local damage summarises how much of a point's neighbourhood has been lost and is the field normally plotted to visualise a crack.

Returns
-------
return (N,) float64: the source's local damage at each point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def point_damage(mu, H, V):
    """mu: (N, N) bond damage indicator; H: (N, N) family indicator; V: (N,)
    point volumes. Returns (N,) float64 with the source's local damage at each
    point (paper Eq. 30), between 0.0 and 1.0. The weighting used in the ratio
    is the source's convention."""
    return np.zeros(np.asarray(V).size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: local damage at a point (Eq. 30)."""

import numpy as np


def _oracle_point_damage(mu, H, V):
    mu = np.asarray(mu, dtype=np.float64)
    Hb = (np.asarray(H, dtype=np.float64) > 0).astype(np.float64)
    V = np.asarray(V, dtype=np.float64)
    num = (mu * Hb * V[None, :]).sum(axis=1)
    den = (Hb * V[None, :]).sum(axis=1)
    out = np.ones_like(num)
    nz = den > 0
    out[nz] = 1.0 - num[nz] / den[nz]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nmu=damage_state(bond_stretch(X, u), H, 0.16125)', "call": 'point_damage(mu, H, V)', "gold_call": '_oracle_point_damage(mu, H, V)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nmu=damage_state(bond_stretch(X, u), H, 0.1370625)', "call": 'point_damage(mu, H, V)', "gold_call": '_oracle_point_damage(mu, H, V)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nmu=damage_state(bond_stretch(X, u), H, 0.209625)', "call": 'point_damage(mu, H, V)', "gold_call": '_oracle_point_damage(mu, H, V)', "tol": 1e-10},
    ]
