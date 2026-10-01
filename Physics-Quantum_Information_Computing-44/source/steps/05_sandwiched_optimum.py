"""
Optimize the finite-order quantum reference state for a cyclic ensemble.

Equations 17 and 98–100 restrict the auxiliary state to the invariant set. The task uses t=1/a, and all outcomes are cyclic rotations of the supplied state with uniform probabilities.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sandwiched_optimum(state: np.ndarray, reciprocal_order: float) -> np.ndarray:
    """Optimize the finite-order quantum reference state for a cyclic ensemble.

    state : ndarray, shape (N,N), complex
        Normalized Hermitian positive semidefinite conditional state in cyclic basis,
        N=2 or 4; its cyclic orbit defines the classical-quantum ensemble.
    reciprocal_order : float
        t=1/a in (0,1), dimensionless.
    Returns
    -------
    ndarray, shape (N+1,), float
        [h_a, sigma_0,...,sigma_(N-1)]: invariant optimized sandwiched conditional
        entropy in bits followed by the optimizing auxiliary state's cyclic
        diagonal probabilities. Retain the input's cyclic order. Set entries
        on marginal support weights below 1e-15 to zero. Entropy tolerance 5e-7
        bits; auxiliary probabilities 2e-6 absolute. Inputs with a unique active
        optimum are used. The input array remains unchanged.
    Raises
    ------
    ValueError
        If state is invalid or t is outside (0,1).
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

def _sand_loss(state, sigma, reciprocal_order, gradient=False):
    t=reciprocal_order;d=np.exp((t-1)*np.log(sigma)/2)
    B=d[:,None]*state*d[None,:]
    e,u=np.linalg.eigh(B);e=np.maximum(e,1e-300)
    logs=np.log(e)/t;L=logsumexp(logs)
    value=t/(1-t)*L/np.log(2)
    if not gradient:return value
    diagonal=np.abs(u)**2@np.exp(logs-L)
    return value,-diagonal/sigma/np.log(2)

def _oracle_sandwiched_optimum(state: np.ndarray, reciprocal_order: float) -> np.ndarray:
    r=_checked_state(state);N=len(r);t=float(reciprocal_order)
    if not np.isfinite(t) or not 0<t<1:raise ValueError('reciprocal_order must be in (0,1)')
    w=r.diagonal().real;active=w>1e-15;rr=r[np.ix_(active,active)];ww=w[active];rr=rr/ww.sum();ww=ww/ww.sum()
    out=np.zeros(N+1)
    if len(ww)==1:out[0]=np.log2(N);out[1:][active]=1;return out
    # Optimize on the symmetry-invariant simplex; the derivative is analytic.
    fit=minimize(lambda s:_sand_loss(rr,s,t,True),ww,jac=True,method='SLSQP',
        bounds=[(1e-14,1)]*len(ww),constraints=[{'type':'eq','fun':lambda s:s.sum()-1,'jac':lambda s:np.ones_like(s)}],
        options={'ftol':3e-13,'maxiter':300})
    sigma=np.maximum(fit.x,1e-14);sigma/=sigma.sum()
    residual=_sand_loss(rr,sigma,t,True)[1]
    if np.ptp(residual)>3e-5:
        # A second, unconstrained parametrization repairs poorly scaled endpoints.
        def _fun(z):
            logits=np.r_[z,0.];s=np.exp(logits-logsumexp(logits));val,g=_sand_loss(rr,s,t,True)
            return val,(s*(g-np.dot(s,g)))[:-1]
        alt=minimize(_fun,np.log(sigma[:-1]/sigma[-1]),jac=True,method='BFGS',options={'gtol':2e-10,'maxiter':400})
        logits=np.r_[alt.x,0.];sigma=np.exp(logits-logsumexp(logits))
    out[0]=np.log2(N)-_sand_loss(rr,sigma,t);out[1:][active]=sigma
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np\nr=np.diag([.74,.26]).astype(complex)',
      'call': '(sandwiched_optimum(r.copy(),0.33)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.33)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.array([[.63,.21],[.21,.37]],complex)',
      'call': '(sandwiched_optimum(r.copy(),0.5)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.5)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'v=np.array([[1,.2j,.1,.3],[.1,.8,.2j,-.1],[.2,.1j,.7,.2],[-.1j,.1,.2,.6]],complex); '
               'r=v@v.conj().T; r/=np.trace(r)',
      'call': '(sandwiched_optimum(r.copy(),0.37)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.37)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'r=np.array([[.25,.08,0,0],[.08,.25,0,0],[0,0,.25,.08j],[0,0,-.08j,.25]],complex)',
      'call': '(sandwiched_optimum(r.copy(),0.0625)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.0625)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nv=np.sqrt([.72,.28]);r=.999*np.outer(v,v)+.001*np.eye(2)/2',
      'call': '(sandwiched_optimum(r.copy(),0.97)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.97)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.diag([1.,0.,0.,0.]).astype(complex)',
      'call': '(sandwiched_optimum(r.copy(),0.23)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.23)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nr=np.array([[.6,0,.17j,0],[0,0,0,0],[-.17j,0,.4,0],[0,0,0,0]],complex)',
      'call': '(sandwiched_optimum(r.copy(),0.71)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.71)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'v=np.sqrt([.46,.28,.17,.09])*np.exp(1j*np.array([.1,-.3,.7,1.2])); r=np.outer(v,v.conj())',
      'call': '(sandwiched_optimum(r.copy(),0.44)) * np.r_[1.,np.full(len(r),.25)]',
      'gold_call': '(_oracle_sandwiched_optimum(r.copy(),0.44)) * np.r_[1.,np.full(len(r),.25)]',
      'tol': 5e-07}]
