"""
Convert the degree-bounded triangle syzygy ansatz to coefficient matching.

This step implements the source computational monomial ansatz and coefficient-matching linearization.

Returns
-------
np.ndarray shape-(35,43) coefficient matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triangle_syzygy_matrix(b: "np.ndarray") -> "np.ndarray":
    '''Assemble the degree-bounded syzygy coefficient-matching matrix.

    Parameters
    ----------
    b : np.ndarray
        Finite shape-(10,) coefficient vector of the normalized quadratic
        Baikov polynomial in the fixed degree-two graded-lex basis.

    Returns
    -------
    np.ndarray
        Shape-(35,43) coefficient matrix M for monomial matching through
        total degree four in the benchmark unknown layout.

    Raises
    ------
    ValueError
        If b does not have shape (10,) or contains non-finite values.
    '''
    return matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _dict(c,basis): return {m:float(v) for m,v in zip(basis,c) if abs(v)>1e-15}
def _mul(p,q):
    r={}
    for m,v in p.items():
        for n,w in q.items():
            e=tuple(m[i]+n[i] for i in range(3)); r[e]=r.get(e,0.0)+v*w
    return r

def _deriv(p,j):
    r={}
    for m,v in p.items():
        if m[j]:
            e=list(m); e[j]-=1; e=tuple(e); r[e]=r.get(e,0.0)+v*m[j]
    return r

def _oracle_triangle_syzygy_matrix(b: "np.ndarray") -> "np.ndarray":
    b=np.asarray(b,dtype=float)
    if b.shape!=(10,) or not np.all(np.isfinite(b)): raise ValueError('b must be finite shape-(10,)')
    m1=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    m2=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    m4=[]
    for d in range(5):
        for a in range(d,-1,-1):
            for bb in range(d-a,-1,-1): m4.append((a,bb,d-a-bb))
    B=_dict(b,m2); dB=[_deriv(B,e) for e in range(3)]; cols=[B]
    for e in range(3):
        ze={tuple(1 if i==e else 0 for i in range(3)):1.}; base=_mul(ze,dB[e])
        for m in m2: cols.append(_mul({m:1.},base))
    for e in range(3):
        ze={tuple(1 if i==e else 0 for i in range(3)):1.}; base=_mul(ze,B)
        for m in m1: cols.append(_mul({m:1.},base))
    ridx={m:i for i,m in enumerate(m4)}; M=np.zeros((35,43))
    for j,p in enumerate(cols):
        for m,v in p.items(): M[ridx[m],j]=v
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nb=np.array([0,0,-5,0,0,1,-1,-1,1,0.])',
            'call': 'triangle_syzygy_matrix(b.copy())',
            'gold_call': '_oracle_triangle_syzygy_matrix(b.copy())',
            'tol': 1e-13,
        },
        {
            'setup': 'import numpy as np\nb=np.array([0,0,-2,0,0,1,-1,-1,1,0.])',
            'call': 'triangle_syzygy_matrix(b.copy())',
            'gold_call': '_oracle_triangle_syzygy_matrix(b.copy())',
            'tol': 1e-13,
        },
        {
            'setup': 'import numpy as np\nb=np.array([1,.2,-.7,.1,.3,-.4,.2,.6,-.1,.5])',
            'call': 'triangle_syzygy_matrix(b.copy())',
            'gold_call': '_oracle_triangle_syzygy_matrix(b.copy())',
            'tol': 1e-13,
        },
    ]
