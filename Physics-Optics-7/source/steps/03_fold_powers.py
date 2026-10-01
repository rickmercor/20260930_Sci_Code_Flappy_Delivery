"""
Construct the graded perturbation polynomials for the full two-dimensional fold.

Section 3.3.2, equations (74)–(83) and the multidimensional extension (86). The normalized canonical phase is t^3/3+zeta*t+sign(lambda)*s^2/2. Define epsilon=omega^(-1/6). The local displacement is c*epsilon^2*t along the null direction and epsilon^3*s/sqrt(abs(lambda)) transversely. Expand the exponential of the higher Taylor terms in epsilon; the output is its coefficient tensor.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fold_powers(jet: "np.ndarray", scale: float, transverse: float, order: int) -> "np.ndarray":
    """Construct the graded perturbation polynomials for the full two-dimensional fold.

    jet : ndarray, shape (D+1,D+1), float
        Phase Taylor coefficients J[a,b] including factorial division, from a
        simple fold. The cubic canonical coefficient satisfies J[3,0]*scale^3=1/3.
        D>=floor(order/2)+3; entries outside total degree D are zero.
    scale : float
        Positive null-coordinate cubic scale c.
    transverse : float
        Nonzero transverse Hessian eigenvalue lambda.
    order : int
        Highest epsilon power K, 0 through 12.
    Returns
    -------
    ndarray, shape (order+1,2*order+1,order+1), complex
        [k,a,b] is the coefficient of epsilon^k t^a s^b in the exponential
        perturbation multiplying the canonical fold/transverse integrand.
        Unused coefficients are zero. All entries are dimensionless.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def _oracle_fold_powers(jet: "np.ndarray", scale: float, transverse: float, order: int) -> "np.ndarray":
    C = np.asarray(jet, dtype=float)
    R = [{} for _ in range(order + 1)]
    for a in range(C.shape[0]):
        for b in range(C.shape[1]):
            k = 2*a + 3*b - 6
            if a+b >= 3 and 1 <= k <= order and C[a, b] != 0:
                R[k][a, b] = C[a, b] * scale**a * abs(transverse)**(-b/2)
    P = [{(0, 0): 1.0}] + [{} for _ in range(order)]
    for k in range(1, order + 1):
        for j in range(1, k + 1):
            for (a, b), v in R[j].items():
                for (aa, bb), w in P[k-j].items():
                    key = a+aa, b+bb
                    P[k][key] = P[k].get(key, 0) + 1j*j/k * v*w
    out = np.zeros((order+1, 2*order+1, order+1), dtype=complex)
    for k, p in enumerate(P):
        for (a, b), v in p.items():
            out[k, a, b] = v
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\nJ=np.zeros((4,4))\nJ[3,0]=0.3333333333333333',
      'call': 'fold_powers(J.copy(),1,1,0)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,1,0)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((5,5))\nJ[3,0]=0.3333333333333333\nJ[2,1]=0.2',
      'call': 'fold_powers(J.copy(),1,0.7,2)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,0.7,2)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((6,6))\nJ[3,0]=0.3333333333333333\nJ[4,0]=-0.04',
      'call': 'fold_powers(J.copy(),1,1.3,4)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,1.3,4)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((6,6))\nJ[3,0]=0.3333333333333333\nJ[1,2]=0.15',
      'call': 'fold_powers(J.copy(),1,-0.8,4)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,-0.8,4)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((7,7))\nJ[3,0]=0.3333333333333333\nJ[0,3]=0.03\nJ[2,1]=-0.12',
      'call': 'fold_powers(J.copy(),1,0.6,6)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,0.6,6)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'J=np.zeros((8,8))\n'
               'J[3,0]=0.041666666666666664\n'
               'J[4,0]=0.01\n'
               'J[2,2]=-0.02\n'
               'J[5,0]=-0.005',
      'call': 'fold_powers(J.copy(),2,1.2,8)',
      'gold_call': '_oracle_fold_powers(J.copy(),2,1.2,8)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'J=np.zeros((9,9))\n'
               'J[3,0]=0.3333333333333333\n'
               'J[2,1]=0.1\n'
               'J[4,0]=0.04\n'
               'J[1,2]=-0.06\n'
               'J[3,1]=0.02\n'
               'J[0,4]=0.008',
      'call': 'fold_powers(J.copy(),1,1.7,10)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,1.7,10)',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'J=np.zeros((10,10))\n'
               'J[3,0]=0.3333333333333333\n'
               'J[2,1]=-0.08\n'
               'J[4,0]=-0.03\n'
               'J[5,0]=0.004\n'
               'J[6,0]=0.002\n'
               'J[2,2]=0.01\n'
               'J[1,3]=-0.007',
      'call': 'fold_powers(J.copy(),1,0.9,12)',
      'gold_call': '_oracle_fold_powers(J.copy(),1,0.9,12)',
      'tol': 2e-08}]
