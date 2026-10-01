"""
Construct Eve's states conditional on the reverse-reconciled symbol.

Equations 21–31 and 46–54 construct the uniform classical-quantum state from the optical transition matrix and the cyclic coherent-state support. Indices of the optical channel are outcome first, preparation second.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reverse_states(channel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Construct Eve's states conditional on the reverse-reconciled symbol.

    channel : ndarray, shape (N,N), float
        Nonnegative doubly stochastic P[y,x], N=2 or 4.
    weights : ndarray, shape (N,), float
        Nonnegative normalized cyclic weights in the stated basis convention.
    Returns
    -------
    ndarray, shape (N,N,N), complex
        result[y,s,t]=<psi_s|rho_(E|Y=y)|psi_t>; every outcome state has trace one.
        The first axis is the classical outcome y; the other two axes are cyclic
        basis row and column. All entries are dimensionless.
    Raises
    ------
    ValueError
        If dimensions, stochastic normalization or weights are invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _oracle_reverse_states(channel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    P=np.asarray(channel,float);w=np.asarray(weights,float)
    if w.ndim!=1:
        raise ValueError('weights must be a one-dimensional normalized vector')
    N=len(w)
    if N not in (2,4) or P.shape!=(N,N) or not np.isfinite(P).all() or not np.isfinite(w).all() or np.min(P)<0 or np.min(w)<0 or not np.allclose(P.sum(axis=0),1,atol=1e-10) or not np.allclose(P.sum(axis=1),1,atol=1e-10) or abs(w.sum()-1)>1e-10:
        raise ValueError('channel must be doubly stochastic; weights must be normalized')
    v=np.sqrt(w)[:,None]*np.exp(2j*np.pi*np.arange(N)[:,None]*np.arange(N)/N)
    return np.array([(v*row)@v.conj().T for row in P])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np\nw=np.array([1.,0.]);P=np.array([[.81,.19],[.19,.81]],float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nw=np.array([.5,.5]);P=np.array([[1.,0.],[0.,1.]],float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nw=np.array([.4,.3,.2,.1]);P=np.array(np.ones((4,4))/4,float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nw=np.array([.55,.25,.15,.05]);P=np.array(np.roll(np.eye(4),1,axis=0),float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'w=np.array([.41,.27,.19,.13]);P=np.array(np.array([np.roll([.61,.23,.05,.11],y) for y in '
               'range(4)]),float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'w=np.array([.6,.23,.12,.05]);P=np.array(np.array([np.roll([.64,.16,.04,.16],y) for y in '
               'range(4)]),float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nw=np.array([.77,.23]);P=np.array([[.12,.88],[.88,.12]],float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'w=np.array([.11,.39,.17,.33]);P=np.array(np.array([np.roll([.49,.27,.09,.15],y) for y in '
               'range(4)]),float)',
      'call': 'reverse_states(P.copy(),w.copy())',
      'gold_call': '_oracle_reverse_states(P.copy(),w.copy())',
      'tol': 1e-08}]
