"""
Compute the mixed Taylor coefficients of the Gaussian screen in an orthonormal frame.

The local Taylor data in Algorithm 1 and equations (53), (63), (74), and (86) are coefficients of the phase, expressed in the specified frame. This step uses the Gaussian screen of Section 2; the polynomial is the phase Taylor polynomial.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gaussian_jet(precision: "np.ndarray", alpha: float, point: "np.ndarray", basis: "np.ndarray", degree: int) -> "np.ndarray":
    """Compute the mixed Taylor coefficients of the Gaussian screen in an orthonormal frame.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A.
    alpha : float
        Real reduced screen strength, including zero for the empty-screen limit.
    point : ndarray, shape (2,), float
        Expansion location.
    basis : ndarray, shape (2,2), float
        Orthonormal matrix whose columns define local coordinates (u,v).
    degree : int
        Total Taylor degree, 0 through 12.
    Returns
    -------
    ndarray, shape (degree+1,degree+1), float
        Entry [a,b] is the coefficient of u^a v^b in
        alpha*exp(-(point+basis@(u,v))^T A (point+basis@(u,v))/2).
        Entries with a+b>degree are zero. Coefficients include factorial division.
        All entries are dimensionless.
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

def _oracle_gaussian_jet(precision: "np.ndarray", alpha: float, point: "np.ndarray", basis: "np.ndarray", degree: int) -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    x = np.asarray(point, dtype=float)
    V = np.asarray(basis, dtype=float)
    linear = -V.T @ A @ x
    quadratic = -0.5 * V.T @ A @ V
    C = np.zeros((degree + 1, degree + 1))
    C[0, 0] = alpha * np.exp(-x @ A @ x / 2)
    def _at(a, b):
        return C[a, b] if a >= 0 and b >= 0 else 0.0
    for b in range(1, degree + 1):
        C[0, b] = (linear[1] * _at(0, b-1) + 2 * quadratic[1, 1] * _at(0, b-2)) / b
    for a in range(1, degree + 1):
        for b in range(degree - a + 1):
            C[a, b] = (linear[0] * _at(a-1, b) + 2 * quadratic[0, 0] * _at(a-2, b)
                       + 2 * quadratic[0, 1] * _at(a-1, b-1)) / a
    return C

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.2], [0.2, 0.6]]);x=np.array([0.2, '
               '-0.3]);t=0.2;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),1.3,x.copy(),V.copy(),6)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),1.3,x.copy(),V.copy(),6)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0], [0, 0.5]]);x=np.array([1, '
               '2]);t=0;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),0,x.copy(),V.copy(),0)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),0,x.copy(),V.copy(),0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0], [0, 1]]);x=np.array([0, '
               '0]);t=0.7;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),1,x.copy(),V.copy(),8)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),1,x.copy(),V.copy(),8)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.3], [0.3, 0.8]]);x=np.array([0, '
               '0]);t=-0.4;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),2,x.copy(),V.copy(),7)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),2,x.copy(),V.copy(),7)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[0.7, 0], [0, 1.1]]);x=np.array([-0.7, '
               '0.1]);t=1.5707963267948966;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),1.2,x.copy(),V.copy(),5)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),1.2,x.copy(),V.copy(),5)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, -0.2], [-0.2, 0.5]]);x=np.array([0.1, '
               '0.8]);t=1.0;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),1.7,x.copy(),V.copy(),9)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),1.7,x.copy(),V.copy(),9)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.1], [0.1, 0.4]]);x=np.array([0.6, '
               '-0.4]);t=-0.8;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),0.8,x.copy(),V.copy(),3)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),0.8,x.copy(),V.copy(),3)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'A=np.array([[0.6, 0.1], [0.1, 0.9]]);x=np.array([-0.3, '
               '-0.2]);t=2.2;V=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])',
      'call': 'gaussian_jet(A.copy(),2.1,x.copy(),V.copy(),10)',
      'gold_call': '_oracle_gaussian_jet(A.copy(),2.1,x.copy(),V.copy(),10)',
      'tol': 1e-08}]
