"""
Evaluate entropy statistics for the cyclic classical-quantum ensemble.

Equations 6–8 and 14 together with Appendix A define H(Y|E), the conditional entropy variance, and the order-two Petz conditional entropy. For the ensemble of cyclic rotations of state, the outcome alphabet has the same size as the matrix dimension.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def entropy_statistics(state: np.ndarray) -> np.ndarray:
    """Evaluate entropy statistics for the cyclic classical-quantum ensemble.

    state : ndarray, shape (N,N), complex
        Normalized Hermitian positive semidefinite rho_(E|Y=0), N=2 or 4,
        in the cyclic basis; its unitary orbit has equal outcome probabilities.
    Returns
    -------
    ndarray, shape (3,), float
        [H(Y|E), V(Y|E), H_2_down(Y|E)], in [bits, bits**2, bits].
        The marginal is diag(diag(state)). Spectral support weights below 1e-15
        may be removed and the retained state normalized.
    Raises
    ------
    ValueError
        If state has invalid dimensions, entries, normalization or positivity.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _checked_state(state):
    r=np.array(state,dtype=complex,copy=True)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or len(r) not in (2,4) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T,atol=1e-10) or abs(np.trace(r)-1)>1e-9 or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('state must be a normalized positive Hermitian 2x2 or 4x4 matrix')
    return (r+r.conj().T)/2

def _oracle_entropy_statistics(state: np.ndarray) -> np.ndarray:
    r=_checked_state(state);N=len(r);w=r.diagonal().real
    active=w>1e-15;r=r[np.ix_(active,active)];w=w[active];r/=w.sum();w=w/w.sum()
    e,u=np.linalg.eigh(r);e=np.maximum(e,0)
    loge=np.log2(np.maximum(e,1e-300));logw=np.log2(w)
    D=np.sum(e*loge)-np.sum(w*logw)
    # Spectral-overlap formula avoids forming a logarithm on a null eigenvector.
    overlaps=np.abs(u)**2
    second=np.sum((overlaps*e[None,:])*(loge[None,:]-logw[:,None])**2)
    H2=np.log2(N)-np.log2(np.sum(np.abs(r)**2/w[None,:]))
    return np.array([np.log2(N)-D,max(0,second-D*D),H2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np\nr=np.diag([.74,.26]).astype(complex)',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nr=np.diag([.41,.29,.19,.11]).astype(complex)',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nr=np.ones((4,4),complex)/4',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nr=np.array([[.6,0,.17j,0],[0,0,0,0],[-.17j,0,.4,0],[0,0,0,0]],complex)',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nr=np.array([[.63,.21],[.21,.37]],complex)',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'v=np.array([[1,.2j,.1,.3],[.1,.8,.2j,-.1],[.2,.1j,.7,.2],[-.1j,.1,.2,.6]],complex); '
               'r=v@v.conj().T; r/=np.trace(r)',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nv=np.sqrt([.72,.28]);r=.999*np.outer(v,v)+.001*np.eye(2)/2',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'r=np.array([[.25,.08,0,0],[.08,.25,0,0],[0,0,.25,.08j],[0,0,-.08j,.25]],complex)',
      'call': 'entropy_statistics(r.copy())',
      'gold_call': '_oracle_entropy_statistics(r.copy())',
      'tol': 1e-08}]
