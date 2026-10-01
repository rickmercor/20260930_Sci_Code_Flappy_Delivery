"""
Build the normalized one-mass-triangle Baikov polynomial.

The source constructs the one-loop Baikov polynomial from a Cayley-Menger determinant.

Returns
-------
np.ndarray shape-(10,) normalized Baikov coefficients in the fixed degree-two basis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triangle_normalized_baikov(s: float) -> "np.ndarray":
    '''Return normalized one-mass-triangle Baikov coefficients.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.

    Returns
    -------
    np.ndarray
        Shape-(10,) coefficients of Bhat = 4 B / s in the fixed degree-two
        graded-lex basis (1,z0,z1,z2,z0^2,z0z1,z0z2,z1^2,z1z2,z2^2).

    Raises
    ------
    ValueError
        If s is non-finite or non-positive.
    '''
    return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np

def _padd(p,q,scale=1.0):
    r=dict(p)
    for m,v in q.items(): r[m]=r.get(m,0.0)+scale*v
    return {m:v for m,v in r.items() if abs(v)>1e-15}

def _pmul(p,q):
    r={}
    for m,v in p.items():
        for n,w in q.items():
            e=tuple(m[i]+n[i] for i in range(3)); r[e]=r.get(e,0.0)+v*w
    return {m:v for m,v in r.items() if abs(v)>1e-15}

def _oracle_triangle_normalized_baikov(s: float) -> "np.ndarray":
    s=float(s)
    if not np.isfinite(s) or s<=0: raise ValueError('s must be positive and finite')
    basis=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    C=np.array([[0.,0.,s],[0.,0.,0.],[s,0.,0.]])
    z=[{(1,0,0):1.},{(0,1,0):1.},{(0,0,1):1.}]
    A=[[{} for _ in range(5)] for _ in range(5)]
    A[0]=[{},z[0],z[1],z[2],{(0,0,0):1.}]
    for i in range(3):
        A[i+1][0]=z[i]
        for j in range(3): A[i+1][j+1]={(0,0,0):float(C[i,j])} if C[i,j]!=0 else {}
        A[i+1][4]={(0,0,0):1.}
    A[4]=[{(0,0,0):1.},{(0,0,0):1.},{(0,0,0):1.},{(0,0,0):1.},{}]
    det={}
    for perm in itertools.permutations(range(5)):
        inv=sum(perm[i]>perm[j] for i in range(5) for j in range(i+1,5))
        term={(0,0,0):-1. if inv%2 else 1.}
        for i,j in enumerate(perm):
            term=_pmul(term,A[i][j])
            if not term: break
        det=_padd(det,term)
    Bhat={m:(4.0/s)*(v/8.0) for m,v in det.items()}
    return np.array([Bhat.get(m,0.0) for m in basis])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'s=5.0','call':'triangle_normalized_baikov(s)','gold_call':'_oracle_triangle_normalized_baikov(s)','tol':1e-13},
        {'setup':'s=2.0','call':'triangle_normalized_baikov(s)','gold_call':'_oracle_triangle_normalized_baikov(s)','tol':1e-13},
        {'setup':'s=0.75','call':'triangle_normalized_baikov(s)','gold_call':'_oracle_triangle_normalized_baikov(s)','tol':1e-13},
    ]
