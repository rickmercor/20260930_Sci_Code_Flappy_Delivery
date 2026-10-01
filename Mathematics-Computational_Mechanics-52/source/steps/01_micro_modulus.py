"""
Return the source's one-dimensional bond micro-modulus c for each given horizon, for Young's modulus E and cross-sectional area A (the paper's Eq. 25, one-dimensional case). Note the paper states that the micro-modulus of the dual-horizon formulation differs by a constant factor from the classical single-horizon one; use the source's dual-horizon value. Recover the exact expression from the paper.

The micro-modulus calibrates the pairwise bond stiffness so that the non-local model reproduces the correct macroscopic elastic response; its exact form for the dual-horizon formulation is stated only in the source.

Returns
-------
return array like delta (float64): the source's bond micro-modulus
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def micro_modulus(delta, E, A):
    """delta: horizon value(s) > 0; E: Young's modulus; A: cross-sectional area.
    Returns a float64 array shaped like delta with the source's 1D dual-horizon
    bond micro-modulus (paper Eq. 25). Raises ValueError on a non-positive or
    nonfinite horizon."""
    return np.zeros_like(np.asarray(delta, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: 1D dual-horizon bond micro-modulus (Eq. 25)."""

import numpy as np


def _oracle_micro_modulus(delta, E, A):
    d = np.asarray(delta, dtype=np.float64)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise ValueError("horizon must be positive and finite")
    return E / (d ** 2 * A)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nE=1.0;A=1.0', "call": 'micro_modulus(delta, E, A)', "gold_call": '_oracle_micro_modulus(delta, E, A)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nE=1.0;A=1.0', "call": 'micro_modulus(delta, E, A)', "gold_call": '_oracle_micro_modulus(delta, E, A)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nE=2.5;A=0.5', "call": 'micro_modulus(delta, E, A)', "gold_call": '_oracle_micro_modulus(delta, E, A)', "tol": 1e-10},
    ]
