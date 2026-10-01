"""
Eq. (2) and the signed skew-factorization foundation: preserve the Pfaffian sign/phase. This supporting operation is established mathematics, not a new contribution claimed for this paper.

Eq. (2) and the signed skew-factorization foundation: preserve the Pfaffian sign/phase. This supporting operation is established mathematics, not a new contribution claimed for this paper.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def signed_pfaffian(matrix):
    """Evaluate a signed, possibly complex Pfaffian.

    Parameters
    ----------
    matrix : real or complex array, (2*m,2*m)
        Finite skew-symmetric matrix A=-A.T, including singular and empty cases.

    Returns
    -------
    complex scalar
        Pf(A), with Pf([[0,a],[-a,0]])=a and Pf(empty)=1. This is a bilinear
        Pfaffian, not a Hermitian determinant. Preserve its sign/complex phase.

    Raises
    ------
    ValueError
        For a nonfinite matrix, nonsquare/odd shape, or failure of A=-A.T
        at absolute tolerance 1e-12 (rtol=0).
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def _oracle_signed_pfaffian(matrix):
    """Evaluate a signed, possibly complex Pfaffian.

    Parameters
    ----------
    matrix : real or complex array, (2*m,2*m)
        Finite skew-symmetric matrix A=-A.T, including singular and empty cases.

    Returns
    -------
    complex scalar
        Pf(A), with Pf([[0,a],[-a,0]])=a and Pf(empty)=1. This is a bilinear
        Pfaffian, not a Hermitian determinant. Preserve its sign/complex phase.

    Raises
    ------
    ValueError
        For a nonfinite matrix, nonsquare/odd shape, or failure of A=-A.T
        at absolute tolerance 1e-12 (rtol=0).
    """
    a=np.array(matrix,dtype=complex,copy=True)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)%2 or not np.isfinite(a).all() or not np.allclose(a,-a.T,atol=1e-12,rtol=0):
        raise ValueError('Invalid skew matrix')
    value=1.+0j
    for i in range(0,len(a),2):
        pidx=i+1+int(np.argmax(abs(a[i,i+1:])))
        if a[i,pidx]==0:return 0j
        if pidx!=i+1:
            a[[i+1,pidx],:]=a[[pidx,i+1],:];a[:,[i+1,pidx]]=a[:,[pidx,i+1]];value=-value
        pivot=a[i,i+1];value*=pivot
        u=a[i,i+2:].copy();v=a[i+1,i+2:].copy()
        a[i+2:,i+2:]+=(np.outer(v,u)-np.outer(u,v))/pivot
    return complex(value)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'empty Pfaffian identity',
      'setup': 'import numpy as np\narg0=np.array([],dtype=float).reshape((0, 0))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'negative two-dimensional pivot',
      'setup': 'import numpy as np\narg0=np.array([[0.0, -3.0], [3.0, 0.0]],dtype=float).reshape((2, 2))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'four-dimensional three-term cancellation',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0j, (2+0j), (3+0j), (5+0j)], [(-2+0j), 0j, (7+0j), (11+0j)], [(-3+0j), (-7+0j), '
               '0j, (13+0j)], [(-5+0j), (-11+0j), (-13+0j), 0j]],dtype=complex).reshape((4, 4))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'zero leading pivot requires swap',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0, 0, 2, 1], [0, 0, 3, 5], [-2, -3, 0, 4], [-1, -5, -4, '
               '0]],dtype=int).reshape((4, 4))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'odd simultaneous index permutation',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0j, (-2+0j), (7+0j), (11+0j)], [(2+0j), 0j, (3+0j), (5+0j)], [(-7+0j), (-3+0j), '
               '0j, (13+0j)], [(-11+0j), (-5+0j), (-13+0j), 0j]],dtype=complex).reshape((4, 4))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'complex bilinear not Hermitian',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0j, (2+4j), (3+6j), (5+10j)], [(-2-4j), 0j, (7+14j), (11+22j)], [(-3-6j), '
               '(-7-14j), 0j, (13+26j)], [(-5-10j), (-11-22j), (-13-26j), 0j]],dtype=complex).reshape((4, 4))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'singular skew matrix',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, '
               '0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, '
               '0.0, 0.0, 0.0]],dtype=float).reshape((6, 6))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'},
     {'name': 'signed independent blocks',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [-2.0, 0.0, -0.0, 0.0, -0.0, 0.0], [0.0, 0.0, '
               '-0.0, -3.0, 0.0, 0.0], [-0.0, 0.0, 3.0, -0.0, -0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.4], [-0.0, '
               '0.0, -0.0, 0.0, -0.4, 0.0]],dtype=float).reshape((6, 6))',
      'call': 'signed_pfaffian(arg0)',
      'gold_call': '_oracle_signed_pfaffian(arg0)'}]
