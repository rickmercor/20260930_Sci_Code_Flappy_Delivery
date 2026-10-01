"""
Evaluate the nondegenerate two-dimensional refractive correction series.

Algorithm 1, equations (53)–(65). The input Taylor coefficients are already in the Hessian eigenframe. The normalized saddle series multiplies exp(i*omega*T_s-i*pi*Morse_index/2)/sqrt(abs(det(H))) and is ordered in omega^(-m).

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def saddle_series(jet: "np.ndarray", eigenvalues: "np.ndarray", order: int) -> "np.ndarray":
    """Evaluate the nondegenerate two-dimensional refractive correction series.

    jet : ndarray, shape (D+1,D+1), float
        Phase Taylor coefficients in the nondegenerate Hessian eigenframe,
        including factorial division. D>=2*order+2. Constant, linear and quadratic
        entries can be present; the eigenvalues separately describe total T.
    eigenvalues : ndarray, shape (2,), float
        Nonzero real eigenvalues of the full arrival-time Hessian, in jet order.
    order : int
        Highest inverse-frequency order M, 0 through 3.
    Returns
    -------
    ndarray, shape (order+1,), complex
        Coefficients t[m] of sum_m t[m]/omega^m after removing the full leading
        saddle prefactor, with t[0]=1. All entries are dimensionless.
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

def _oracle_saddle_series(jet: "np.ndarray", eigenvalues: "np.ndarray", order: int) -> "np.ndarray":
    C = np.asarray(jet, dtype=float)
    lam = np.asarray(eigenvalues, dtype=float)
    R = {(a,b): C[a,b] for a in range(len(C)) for b in range(len(C))
         if 3 <= a+b <= 2*order+2 and C[a,b] != 0}
    P = {(0,0): 1.0}
    out = np.zeros(order+1, dtype=complex)
    for r in range(2*order+1):
        for (a,b), v in P.items():
            if a % 2 or b % 2:
                continue
            m = (a+b)//2 - r
            if 0 <= m <= order:
                out[m] += (1j)**(r+(a+b)//2) * v / factorial(r) * prod(range(1,a,2)) * prod(range(1,b,2)) / lam[0]**(a//2) / lam[1]**(b//2)
        nxt = {}
        for (a,b), v in P.items():
            for (aa,bb), w in R.items():
                if a+b+aa+bb <= 2*(order+r+1):
                    key = a+aa,b+bb
                    nxt[key] = nxt.get(key,0) + v*w
        P = nxt
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\nJ=np.zeros((3,3))\n',
      'call': 'saddle_series(J.copy(),np.array([0.8, 1.3]),0)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([0.8, 1.3]),0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((5,5))\nJ[4,0]=0.02',
      'call': 'saddle_series(J.copy(),np.array([1, 2]),1)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([1, 2]),1)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((5,5))\nJ[3,0]=0.1',
      'call': 'saddle_series(J.copy(),np.array([0.7, 1.2]),1)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([0.7, 1.2]),1)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((7,7))\nJ[2,1]=0.08\nJ[1,2]=-0.04',
      'call': 'saddle_series(J.copy(),np.array([1, -0.6]),2)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([1, -0.6]),2)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((7,7))\nJ[2,2]=0.05\nJ[0,4]=-0.02',
      'call': 'saddle_series(J.copy(),np.array([-1, -1.5]),2)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([-1, -1.5]),2)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((7,7))\nJ[5,0]=0.03\nJ[3,0]=0.06\nJ[0,6]=0.002',
      'call': 'saddle_series(J.copy(),np.array([1.2, 0.9]),2)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([1.2, 0.9]),2)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((9,9))\nJ[3,0]=0.06\nJ[2,1]=0.08\nJ[0,3]=-0.02\nJ[4,0]=0.005',
      'call': 'saddle_series(J.copy(),np.array([1.1, 0.8]),3)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([1.1, 0.8]),3)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nJ=np.zeros((9,9))\nJ[8,0]=0.0002\nJ[4,2]=0.003\nJ[2,4]=-0.002\nJ[0,8]=0.0001',
      'call': 'saddle_series(J.copy(),np.array([0.9, 1.7]),3)',
      'gold_call': '_oracle_saddle_series(J.copy(),np.array([0.9, 1.7]),3)',
      'tol': 1e-08}]
