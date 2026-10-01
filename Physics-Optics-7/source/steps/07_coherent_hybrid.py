"""
Match the uniform fold pair to the regular image at each retained order.

Equations (43), (61), (82)–(86). The fold coefficient u[k] multiplies omega^(-k/6) after the leading fold prefactor; the regular coefficient t[m] multiplies omega^(-m). The local fold contribution replaces the coalescing pair in the field. Construct one field for every even cutoff k=0,2,...,K.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coherent_hybrid(fold_coefficients: "np.ndarray", regular_coefficients: "np.ndarray", fold_delay: float, regular_delay: float, scale: float, transverse: float, regular_eigenvalues: "np.ndarray", frequency: float) -> "np.ndarray":
    """Match the uniform fold pair to the regular image at each retained order.

    fold_coefficients : ndarray, shape (2*M+1,), complex
        Normalized coefficients u[k] from fold_contractions, 0<=M<=6.
    regular_coefficients : ndarray, shape (L+1,), complex
        Normalized nondegenerate coefficients t[m], 0<=L<=3.
    fold_delay : float
        T(x_c,y), using the actual source position y.
    regular_delay : float
        T(x_r,y), in the same phase convention.
    scale : float
        Positive cubic scale c of the fold.
    transverse : float
        Nonzero transverse eigenvalue lambda at the fold.
    regular_eigenvalues : ndarray, shape (2,), float
        Nonzero Hessian eigenvalues of the regular image.
    frequency : float
        Positive dimensionless frequency.
    Returns
    -------
    ndarray, shape (M+1,), complex
        Entry m is the coherent total field from the fold truncated through
        relative omega^(-m/3) and the supplied regular series. All entries are
        dimensionless; leading fold and Morse phases follow the outgoing Abel
        convention in the problem.
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

def _oracle_coherent_hybrid(fold_coefficients: "np.ndarray", regular_coefficients: "np.ndarray", fold_delay: float, regular_delay: float, scale: float, transverse: float, regular_eigenvalues: "np.ndarray", frequency: float) -> "np.ndarray":
    u = np.asarray(fold_coefficients, dtype=complex)
    t = np.asarray(regular_coefficients, dtype=complex)
    ev = np.asarray(regular_eigenvalues, dtype=float)
    fold_prefactor = scale * frequency**(1/6) * np.sqrt(2*np.pi) / np.sqrt(abs(transverse))
    fold_prefactor *= np.exp(1j*frequency*fold_delay + 1j*np.pi*np.sign(transverse)/4) / 1j
    fold = fold_prefactor * np.cumsum(u[::2] / frequency**(np.arange(len(u[::2]))/3))
    regular = np.exp(1j*frequency*regular_delay - 1j*np.pi*np.sum(ev < 0)/2) / np.sqrt(abs(np.prod(ev)))
    regular *= np.sum(t / frequency**np.arange(len(t)))
    return fold + regular

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\nu=np.array([0.35],complex);t=np.array([1],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),1.2,0.7,1.0,1.0,np.array([1, 2]),20)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),1.2,0.7,1.0,1.0,np.array([1, 2]),20)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nu=np.array([0.35, 0, 0.02j],complex);t=np.array([1, 0.04j],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),0.9,1.1,0.8,0.6,np.array([0.7, 1.3]),30)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),0.9,1.1,0.8,0.6,np.array([0.7, 1.3]),30)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'u=np.array([0.2, 0, (-0-0.03j), 0, 0.01],complex);t=np.array([1, 0, 0.005],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),1.0,1.0,1.2,-0.8,np.array([-1, 2]),40)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),1.0,1.0,1.2,-0.8,np.array([-1, 2]),40)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'u=np.array([0.4, 0, 0.01j, 0, -0.02, 0, 0.003j],complex);t=np.array([1, 0.1j, -0.02],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),0.4,0.7,0.7,1.1,np.array([-1, -2]),55)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),0.4,0.7,0.7,1.1,np.array([-1, -2]),55)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nu=np.array([0, 0, 0.02j, 0, 0.01],complex);t=np.array([0],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),1.3,0.0,1.0,0.9,np.array([1, 1]),25)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),1.3,0.0,1.0,0.9,np.array([1, 1]),25)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nu=np.array([0.3, 0, 0, 0, 0],complex);t=np.array([1],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),2.1,0.2,0.9,1.4,np.array([0.8, 1.2]),75)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),2.1,0.2,0.9,1.4,np.array([0.8, 1.2]),75)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'u=np.array([0.2, 0, 0.03j, 0, -0.01, 0, 0.005j, 0, 0.002],complex);t=np.array([1, 0.05j, '
               '0.01],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),0.7,0.9,1.1,0.4,np.array([1.5, 0.6]),90)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),0.7,0.9,1.1,0.4,np.array([1.5, 0.6]),90)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'u=np.array([0.4, 0, 0.02j, 0, -0.01],complex);t=np.array([1, (-0-0.08j), 0.006],complex)',
      'call': 'coherent_hybrid(u.copy(),t.copy(),1.4,1.4001,0.8,-1.1,np.array([-0.7, 1.3]),120)',
      'gold_call': '_oracle_coherent_hybrid(u.copy(),t.copy(),1.4,1.4001,0.8,-1.1,np.array([-0.7, 1.3]),120)',
      'tol': 1e-08}]
