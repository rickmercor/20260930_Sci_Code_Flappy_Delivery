"""
Obtain the symbol transition matrix for the stated optical receiver.

Sections IV.1–IV.2 supply the pure-loss BPSK and QPSK optical channel. A fixed rotation of the constellation is a constructed extension; the receiver axes remain fixed.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def psk_channel(N: int, amplitude: float, eta: float, phase: float) -> np.ndarray:
    """Obtain the symbol transition matrix for the stated optical receiver.

    N : int
        Alphabet size, 2 (BPSK) or 4 (QPSK).
    amplitude : float
        Nonnegative coherent launch amplitude; its square is mean photons per signal.
    eta : float
        Channel power transmittance in [0,1].
    phase : float
        Finite constellation phase offset in radians.
    Returns
    -------
    ndarray, shape (N,N), float
        Dimensionless P[y,x]=Pr(Y=y|X=x). BPSK: y=0 means q>0.
        QPSK: counterclockwise quadrants starting with Re>0, Im>0;
        launch angles are pi/4+phase+2*pi*x/N. Use the global receiver variances.
    Raises
    ------
    ValueError
        If N, amplitude, eta or phase is outside its domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _oracle_psk_channel(N: int, amplitude: float, eta: float, phase: float) -> np.ndarray:
    if N not in (2,4) or not np.isfinite([amplitude,eta,phase]).all() or amplitude<0 or not 0<=eta<=1:
        raise ValueError('N must be 2 or 4; amplitude >= 0; eta in [0,1]; phase finite')
    if N==2:
        p=.5*(1+erf(np.sqrt(2*eta)*amplitude*np.cos(phase)))
        return np.array([[p,1-p],[1-p,p]])
    z=np.sqrt(eta)*amplitude*np.exp(1j*(np.pi/4+phase+np.arange(4)*np.pi/2))
    a=.5*(1+erf(z.real));b=.5*(1+erf(z.imag))
    return np.array([a*b,(1-a)*b,(1-a)*(1-b),a*(1-b)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np',
      'call': 'psk_channel(2,0.0,0.65,0.0)',
      'gold_call': '_oracle_psk_channel(2,0.0,0.65,0.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(2,1.3,0.0,0.4)',
      'gold_call': '_oracle_psk_channel(2,1.3,0.0,0.4)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(2,0.93,0.78,1.5707963267948966)',
      'gold_call': '_oracle_psk_channel(2,0.93,0.78,1.5707963267948966)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(2,1.07,0.64,3.141592653589793)',
      'gold_call': '_oracle_psk_channel(2,1.07,0.64,3.141592653589793)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(4,1.2,0.8,0.0)',
      'gold_call': '_oracle_psk_channel(4,1.2,0.8,0.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(4,1.4,0.73,0.7853981633974483)',
      'gold_call': '_oracle_psk_channel(4,1.4,0.73,0.7853981633974483)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(4,0.8,0.94,1.6907963267948967)',
      'gold_call': '_oracle_psk_channel(4,0.8,0.94,1.6907963267948967)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'psk_channel(4,1.1,0.87,-0.31)',
      'gold_call': '_oracle_psk_channel(4,1.1,0.87,-0.31)',
      'tol': 1e-08}]
