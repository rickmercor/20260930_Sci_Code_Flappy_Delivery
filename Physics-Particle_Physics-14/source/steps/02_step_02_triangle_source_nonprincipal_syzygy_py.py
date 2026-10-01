"""
Construct the paper's explicit non-principal one-mass-triangle syzygy.

This step implements the source analytic exceptional-triangle critical syzygy and encodes it in the benchmark 43-coefficient layout.

Returns
-------
np.ndarray shape-(43,) source syzygy coefficient vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triangle_source_nonprincipal_syzygy(s: float) -> "np.ndarray":
    '''Return the paper's explicit non-principal triangle syzygy.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.

    Returns
    -------
    np.ndarray
        Shape-(43,) coefficient vector in the benchmark layout
        [a0; bar_a0^(2); bar_a1^(2); bar_a2^(2); tilde_a0^(1);
        tilde_a1^(1); tilde_a2^(1)].

    Raises
    ------
    ValueError
        If s is non-finite or non-positive.
    '''
    return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_triangle_source_nonprincipal_syzygy(s: float) -> "np.ndarray":
    s=float(s)
    if not np.isfinite(s) or s<=0: raise ValueError('s must be positive and finite')
    m1=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    m2=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    i1={m:i for i,m in enumerate(m1)}; i2={m:i for i,m in enumerate(m2)}
    c=np.zeros(43,dtype=float); c[0]=-2.0*s*s
    bars=[{}, {}, {}]
    bars[0]={(0,0,0):s*s,(1,0,0):-s,(0,0,1):s,(0,1,0):8*s,(1,1,0):-4,(0,2,0):4,(0,1,1):-8}
    bars[1]={(0,0,0):2*s*s,(1,0,0):s,(0,1,0):6*s,(0,0,1):s,(1,1,0):-4,(1,0,1):-4,(0,2,0):4,(0,1,1):-4}
    bars[2]={(0,0,0):s*s,(1,0,0):s,(0,1,0):8*s,(0,0,1):-s,(1,1,0):-8,(0,2,0):4,(0,1,1):-4}
    for e,p in enumerate(bars):
        for m,v in p.items(): c[1+10*e+i2[m]]=v
    # source tilde term: only the middle propagator block is nonzero
    tilde1={(0,0,0):-8*s,(1,0,0):8,(0,1,0):-8,(0,0,1):8}
    for m,v in tilde1.items(): c[35+i1[m]]=v
    return c

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'s=5.0','call':'triangle_source_nonprincipal_syzygy(s)','gold_call':'_oracle_triangle_source_nonprincipal_syzygy(s)','tol':1e-13},
        {'setup':'s=2.0','call':'triangle_source_nonprincipal_syzygy(s)','gold_call':'_oracle_triangle_source_nonprincipal_syzygy(s)','tol':1e-13},
        {'setup':'s=0.75','call':'triangle_source_nonprincipal_syzygy(s)','gold_call':'_oracle_triangle_source_nonprincipal_syzygy(s)','tol':1e-13},
    ]
