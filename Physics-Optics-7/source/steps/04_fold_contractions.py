"""
Evaluate the Airy and transverse Gaussian contractions of each graded polynomial.

Equations (60), (80)–(83), and (86). Contract powers of the fold coordinate with the outgoing Airy integral, and powers of the transverse coordinate with its signed Gaussian integral. The result is normalized by the canonical transverse Gaussian integral and by 2*pi for the fold integral. Ai is the real Airy function with Ai\'\'(z)=z*Ai(z).

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fold_contractions(powers: "np.ndarray", unfolding: float, transverse: float) -> "np.ndarray":
    """Evaluate the Airy and transverse Gaussian contractions of each graded polynomial.

    powers : ndarray, shape (K+1,2*K+1,K+1), complex
        Graded perturbation coefficient tensor P[k,a,b], 0<=K<=12.
    unfolding : float
        Real normalized Airy argument zeta.
    transverse : float
        Nonzero transverse Hessian eigenvalue lambda.
    Returns
    -------
    ndarray, shape (K+1,), complex
        The coefficient u[k] after normalized oscillatory integration of P[k].
        The normalization is by 2*pi times sqrt(2*pi)*exp(i*pi*sign(lambda)/4).
        Thus the constant polynomial contracts to Ai(zeta). All entries are
        dimensionless and ordered by increasing epsilon power.
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

def _oracle_fold_contractions(powers: "np.ndarray", unfolding: float, transverse: float) -> "np.ndarray":
    P = np.asarray(powers, dtype=complex)
    D = np.zeros(P.shape[1] + 1)
    D[:2] = airy(unfolding)[:2]
    for n in range(P.shape[1]-1):
        D[n+2] = unfolding * D[n] + (n * D[n-1] if n else 0)
    out = np.zeros(P.shape[0], dtype=complex)
    for a in range(P.shape[1]):
        for b in range(0, P.shape[2], 2):
            factor = 1j**(-a) * D[a] * (1j*np.sign(transverse))**(b//2) * prod(range(1,b,2))
            out += P[:, a, b] * factor
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\nP=np.zeros((1,1,1),complex)\nP[0,0,0]=1',
      'call': 'fold_contractions(P.copy(),0,1)',
      'gold_call': '_oracle_fold_contractions(P.copy(),0,1)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\nP=np.zeros((3,5,3),complex)\nP[0,0,0]=1\nP[2,4,0]=0.1j',
      'call': 'fold_contractions(P.copy(),-1,1)',
      'gold_call': '_oracle_fold_contractions(P.copy(),-1,1)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\nP=np.zeros((3,5,3),complex)\nP[1,2,1]=1',
      'call': 'fold_contractions(P.copy(),0.6,1)',
      'gold_call': '_oracle_fold_contractions(P.copy(),0.6,1)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\nP=np.zeros((5,9,5),complex)\nP[4,0,4]=1',
      'call': 'fold_contractions(P.copy(),1.2,-1)',
      'gold_call': '_oracle_fold_contractions(P.copy(),1.2,-1)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\nP=np.zeros((5,9,5),complex)\nP[2,1,2]=(-0-0.4j)\nP[4,4,2]=0.07',
      'call': 'fold_contractions(P.copy(),-0.4,0.5)',
      'gold_call': '_oracle_fold_contractions(P.copy(),-0.4,0.5)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\nP=np.zeros((7,13,7),complex)\nP[6,12,0]=0.01\nP[0,0,0]=1',
      'call': 'fold_contractions(P.copy(),2.0,1)',
      'gold_call': '_oracle_fold_contractions(P.copy(),2.0,1)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\nP=np.zeros((9,17,9),complex)\nP[8,6,4]=(-0-0.04j)\nP[6,4,6]=0.01',
      'call': 'fold_contractions(P.copy(),-2.0,-2)',
      'gold_call': '_oracle_fold_contractions(P.copy(),-2.0,-2)',
      'tol': 5e-08},
     {'setup': 'import numpy as np\n'
               'P=np.zeros((13,25,13),complex)\n'
               'P[12,24,0]=1e-07\n'
               'P[12,4,8]=0.03\n'
               'P[10,8,6]=0.004j',
      'call': 'fold_contractions(P.copy(),0.3,1.3)',
      'gold_call': '_oracle_fold_contractions(P.copy(),0.3,1.3)',
      'tol': 5e-08}]
