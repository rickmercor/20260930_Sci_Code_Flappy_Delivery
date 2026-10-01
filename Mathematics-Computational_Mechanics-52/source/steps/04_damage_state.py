"""
Return the source's bond damage indicator for the given stretch state, family indicator and critical stretch (the paper's Eq. 27): a bond that belongs to a family is intact while its stretch stays below the critical stretch, and broken otherwise. Pairs that are not family members carry no bond. The comparison is strict.

Fracture enters bond-based peridynamics by irreversibly removing bonds whose stretch exceeds a critical value; on a single prescribed load state this reduces to the stretch criterion.

Returns
-------
return (N, N) float64: bond damage indicator, 1.0 if intact
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def damage_state(s, H, sc):
    """s: (N, N) bond stretch; H: (N, N) family indicator; sc: critical stretch.
    Returns (N, N) float64: 1.0 for an intact family bond and 0.0 for a broken
    bond or a non-bond (paper Eq. 27)."""
    return np.zeros_like(np.asarray(s, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: bond damage indicator (Eq. 27)."""

import numpy as np


def _oracle_damage_state(s, H, sc):
    s = np.asarray(s, dtype=np.float64)
    Hb = np.asarray(H, dtype=np.float64) > 0
    mu = (s < sc).astype(np.float64)
    mu[~Hb] = 0.0
    return mu

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\ns=bond_stretch(X, u)\nsc=0.16125', "call": 'damage_state(s, H, sc)', "gold_call": '_oracle_damage_state(s, H, sc)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\ns=bond_stretch(X, u)\nsc=0.1370625', "call": 'damage_state(s, H, sc)', "gold_call": '_oracle_damage_state(s, H, sc)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\ns=bond_stretch(X, u)\nsc=0.209625', "call": 'damage_state(s, H, sc)', "gold_call": '_oracle_damage_state(s, H, sc)', "tol": 1e-12},
    ]
