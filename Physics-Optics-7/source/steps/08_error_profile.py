"""
Measure the weighted complex-field error of every truncation in each trial.

This task-specific accuracy functional compares the coherent approximation to the main paper\'s full diffraction field. Source intensities provide the normalization separately at each frequency. Input trials and retained orders remain in their supplied order.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def error_profile(approximations: "np.ndarray", references: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Measure the weighted complex-field error of every truncation in each trial.

    approximations : ndarray, shape (F,Z,M+1), complex
        Approximate fields indexed by frequency, unfolding, retained order.
    references : ndarray, shape (F,Z), complex
        Full diffraction fields on the same grid.
    weights : ndarray, shape (Z,), float
        Nonnegative unfolding weights, with positive sum. Every reference
        frequency row has strictly positive weighted squared norm.
    Returns
    -------
    ndarray, shape (F,M+1), float
        Entry [i,m] is the square root of weighted squared complex field error
        divided by weighted squared reference norm for frequency i and order m.
        Values are dimensionless fractions, before multiplication by 100.
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

def _oracle_error_profile(approximations: "np.ndarray", references: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    fields = np.asarray(approximations, dtype=complex)
    exact = np.asarray(references, dtype=complex)
    w = np.asarray(weights, dtype=float)
    denominator = np.sum(w[None,:] * abs(exact)**2, axis=1)
    errors = np.sqrt(np.sum(w[None,:,None] * abs(fields-exact[:,:,None])**2,axis=1)/denominator[:,None])
    return errors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\n'
               'R=np.array([[1+1j,2-.5j]]);F=np.repeat(R[:,:,None],3,axis=2);w=np.array([1.,2.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[1,1j]],complex);F=np.stack([R*np.exp(.1j),R*np.exp(.4j)],axis=2);w=np.array([1.,1.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[1,2],[10,20]],complex);F=np.stack([R+.1,R+.3j],axis=2);w=np.array([2.,1.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[1,2,3]],complex);F=np.stack([R+np.array([.1,100,.2]),R+.4],axis=2);w=np.array([1.,0.,2.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[1+1j,3],[2,-1j]],complex);F=np.stack([R+.3j,R*.9],axis=2);w=np.array([7.,21.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[1,2]],complex);F=np.stack([R+.3,R+.1j,R-.2,R+.5j],axis=2);w=np.array([1.,3.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[.01+.02j,.04-.03j],[2,3]],complex);F=np.stack([R+.01,R+.02j],axis=2);w=np.array([2.,1.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'R=np.array([[1+2j,3],[2,4j]],complex)[::-1];F=np.stack([R+.1j,R+.4],axis=2);w=np.array([1.,2.])',
      'call': 'error_profile(F.copy(),R.copy(),w.copy())',
      'gold_call': '_oracle_error_profile(F.copy(),R.copy(),w.copy())',
      'tol': 1e-08}]
