"""
Map the reconstructed critical-syzygy lift to the source surface term.

This step implements the source syzygy-to-surface-term map specialized to the no-ISP unit-propagator triangle.

Returns
-------
np.ndarray shape-(10,) surface-polynomial coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triangle_surface_polynomial(s: float, gamma: float) -> "np.ndarray":
    '''Map the canonical syzygy lift to the unit-propagator surface polynomial.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.
    gamma : float
        Finite nonzero Baikov exponent parameter in the source surface-term map.

    Returns
    -------
    np.ndarray
        Shape-(10,) coefficients of the quadratic surface polynomial S in the
        fixed degree-two graded-lex basis.

    Raises
    ------
    ValueError
        If gamma is non-finite or zero, or if an upstream s precondition fails.
    '''
    return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _poly(c,basis): return {m:float(v) for m,v in zip(basis,c) if abs(v)>1e-15}
def _d(p,j):
    r={}
    for m,v in p.items():
        if m[j]:
            e=list(m); e[j]-=1; e=tuple(e); r[e]=r.get(e,0.0)+v*m[j]
    return r

def _zmul(p,j):
    r={}
    for m,v in p.items():
        e=list(m); e[j]+=1; r[tuple(e)]=r.get(tuple(e),0.0)+v
    return r

def _acc(dst,src,scale=1.0):
    for m,v in src.items(): dst[m]=dst.get(m,0.0)+scale*v

def _oracle_triangle_surface_polynomial(s: float, gamma: float) -> "np.ndarray":
    gamma=float(gamma)
    if not np.isfinite(gamma) or gamma==0: raise ValueError('gamma must be finite and nonzero')
    m1=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    m2=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    c=_oracle_triangle_minimum_norm_lift(float(s)); S={(0,0,0):float(c[0])}
    for e in range(3):
        bar=_poly(c[1+10*e:1+10*(e+1)],m2); til=_poly(c[31+4*e:31+4*(e+1)],m1)
        _acc(S,_zmul(til,e),1.0); _acc(S,_zmul(_d(bar,e),e),-1.0/gamma)
    return np.array([S.get(m,0.0) for m in m2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'s=5.0; gamma=-2.5','call':'triangle_surface_polynomial(s,gamma)','gold_call':'_oracle_triangle_surface_polynomial(s,gamma)','tol':5e-10},
        {'setup':'s=2.0; gamma=-1.5','call':'triangle_surface_polynomial(s,gamma)','gold_call':'_oracle_triangle_surface_polynomial(s,gamma)','tol':5e-10},
        {'setup':'s=7.5; gamma=-4.0','call':'triangle_surface_polynomial(s,gamma)','gold_call':'_oracle_triangle_surface_polynomial(s,gamma)','tol':8e-10},
    ]
