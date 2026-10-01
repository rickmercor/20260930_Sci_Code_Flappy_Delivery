"""
Evaluate the optimized infinite-order endpoint and its auxiliary state.

Appendix A, Eqs. 75–77, and the short-block discussion of Section V include the infinite-order endpoint of the optimized sandwiched entropy. The cyclic-invariant restriction is the same as at finite order.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def min_entropy_optimum(state: np.ndarray) -> np.ndarray:
    """Evaluate the optimized infinite-order endpoint and its auxiliary state.

    state : ndarray, shape (N,N), complex
        Normalized Hermitian positive semidefinite conditional state in cyclic basis,
        N=2 or 4, defining a uniform cyclic classical-quantum ensemble.
    Returns
    -------
    ndarray, shape (N+1,), float
        [h_inf, sigma_0,...,sigma_(N-1)], entropy in bits and unit-trace auxiliary
        diagonal probabilities at the infinite-order optimum, in cyclic order.
        Entries on support weights below 1e-15 are zero. Entropy tolerance 5e-7
        bits and auxiliary probabilities 2e-6 absolute. Tests use unique optima.
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

def _oracle_min_entropy_optimum(state: np.ndarray) -> np.ndarray:
    r=_checked_state(state);N=len(r);w=r.diagonal().real;active=w>1e-15
    rr=r[np.ix_(active,active)];rr/=w[active].sum();out=np.zeros(N+1)
    if len(rr)==1:out[0]=np.log2(N);out[1:][active]=1;return out
    def _eig(z):return np.linalg.eigvalsh(np.diag(z)-rr)
    def _jac(z):return (np.abs(np.linalg.eigh(np.diag(z)-rr)[1])**2).T
    initial=np.sum(np.abs(rr),axis=1)+1e-10
    fit=minimize(lambda z:z.sum(),initial,jac=lambda z:np.ones_like(z),method='SLSQP',
      constraints=[{'type':'ineq','fun':_eig,'jac':_jac}],bounds=[(1e-14,None)]*len(rr),options={'ftol':3e-13,'maxiter':400})
    z=fit.x.copy();z+=max(0.,-_eig(z).min())+2e-14
    out[0]=np.log2(N)-np.log2(z.sum());out[1:][active]=z/z.sum()
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np\nr=np.diag([.41,.29,.19,.11]).astype(complex)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nv=np.sqrt([.67,.33]); r=np.outer(v,v)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'v=np.array([[1,.2j,.1,.3],[.1,.8,.2j,-.1],[.2,.1j,.7,.2],[-.1j,.1,.2,.6]],complex); '
               'r=v@v.conj().T; r/=np.trace(r)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.ones((4,4),complex)/4',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.diag([1.,0.,0.,0.]).astype(complex)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.array([[.6,0,.17j,0],[0,0,0,0],[-.17j,0,.4,0],[0,0,0,0]],complex)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.array([[1-1e-7,1e-5j],[-1e-5j,1e-7]],complex)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'r=np.array([[.25,.08,0,0],[.08,.25,0,0],[0,0,.25,.08j],[0,0,-.08j,.25]],complex)',
      'call': '(min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_min_entropy_optimum(r.copy())) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07}]
