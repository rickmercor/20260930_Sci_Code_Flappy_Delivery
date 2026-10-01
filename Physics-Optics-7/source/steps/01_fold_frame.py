"""
Recover the oriented Gaussian fold and its cubic scale.

Section 3.1, equations (26)–(32), and Section 3.3.2, equations (72)–(78). Construct the smallest positive lens strength that creates a null Hessian direction at the supplied point, while the other direction remains positive.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fold_frame(precision: "np.ndarray", point: "np.ndarray") -> "np.ndarray":
    """Recover the oriented Gaussian fold and its cubic scale.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A.
    point : ndarray, shape (2,), float
        Prescribed simple fold location x_c. The smallest eigenvalue of
        (A x_c)(A x_c)^T-A is strictly negative and simple, its chosen
        strength gives a positive transverse eigenvalue, and the cubic is nonzero.
    Returns
    -------
    ndarray, shape (9,), float
        [alpha, y_c[0], y_c[1], V[0,0], V[0,1], V[1,0], V[1,1], lambda, c].
        V=[e,f] with D_e^3 phi>0 and f=(-e[1],e[0]); c is positive.
        All quantities are dimensionless.
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

def _oracle_fold_frame(precision: "np.ndarray", point: "np.ndarray") -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    x = np.asarray(point, dtype=float)
    v = A @ x
    eig, vec = np.linalg.eigh(np.outer(v, v) - A)
    g = -1.0 / eig[0]
    alpha = g * np.exp(x @ A @ x / 2)
    H = np.eye(2) + g * (np.outer(v, v) - A)
    e = vec[:, 0]
    cubic = g * (3 * (e @ v) * (e @ A @ e) - (e @ v)**3)
    if cubic < 0:
        e = -e
        cubic = -cubic
    V = np.column_stack((e, [-e[1], e[0]]))
    lam = V[:, 1] @ H @ V[:, 1]
    return np.r_[alpha, x - g * v, V.ravel(), lam, (2 / cubic)**(1 / 3)]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\nA=np.array([[1, 0], [0, 0.35]]);x=np.array([0.45, 0.1])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0.12], [0.12, 0.5]]);x=np.array([0.5, 0.32])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[0.6, 0.2], [0.2, 1.1]]);x=np.array([0.25, 0.55])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0], [0, 0.45]]);x=np.array([-0.4, 0.2])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0.1], [0.1, 0.4]]);x=np.array([0.5, -0.3])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0], [0, 0.9]]);x=np.array([0.4, 0.5])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0.05], [0.05, 0.6]]);x=np.array([0.65, 0.1])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1.2, -0.1], [-0.1, 0.45]]);x=np.array([0.35, 0.3])',
      'call': 'fold_frame(A.copy(),x.copy())',
      'gold_call': '_oracle_fold_frame(A.copy(),x.copy())',
      'tol': 1e-08}]
