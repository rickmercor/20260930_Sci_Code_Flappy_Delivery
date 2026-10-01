"""
Ancestry-aware state probabilities

Infer smoothed joint probabilities over ancestry and composite reference state conditional on the supplied ancestry parameters.

Inputs are a finite, strictly increasing cM vector g; aligned mismatch and panels arrays with shape (M,H); a counts vector of positive integers; a nonnegative row stochastic copying matrix p with shape (K,J); a finite rate vector rho with K positive entries in units per Morgan; a finite nonnegative ancestry distribution mu with K entries that sum to one; and a finite time greater than 0 in generations. mismatch must be binary. panels must contain integer indices valid for counts. The entries of counts must sum to at least 2. The represented joint state set must have positive model support at every marker.

Return a binary64 array with shape (M,K,H). Its joint ancestry state probabilities must sum to one at every marker.

Raise ValueError if any input is outside the stated domain.

Returns
-------
Binary64 array (M,K,H); the joint state probabilities sum to one at each marker.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ancestry_posterior(g: np.ndarray, mismatch: np.ndarray, panels: np.ndarray, counts \
    : np.ndarray, p: np.ndarray, rho: np.ndarray, mu: np.ndarray, time: float) -> np. \
    ndarray:
    (
        'Binary64 array (M,K,H); the joint state probabilities sum '
        'to one at each marker. Raises ValueError for inputs outsid'
        'e the stated domain.'
    )
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def _oracle_ancestry_posterior(g: np.ndarray, mismatch: np.ndarray, panels: np.ndarray \
    , counts: np.ndarray, p: np.ndarray, rho: np.ndarray, mu: np.ndarray, time: float) \
    -> np.ndarray:
    import math
    import numpy as np
    from scipy.special import logsumexp
    g = _array(g, 1); z0 = _array(mismatch, 2); j0 = _array(panels, 2)
    c0 = _array(counts, 1); p = _array(p, 2); rho = _array(rho, 1); mu = _array(mu, 1)
    time = float(_array(time, 0))
    if not (z0.shape == j0.shape and len(g) == len(z0)):
        raise ValueError('input outside the declared domain')
    if not (np.all(np.diff(g) > 0) and np.all((z0 == 0) | (z0 == 1))):
        raise ValueError('input outside the declared domain')
    if not (np.all(j0 == np.floor(j0)) and np.all((j0 >= 0) & (j0 < len(c0)))):
        raise ValueError('input outside the declared domain')
    if not (np.all(c0 >= 1) and np.all(c0 == np.floor(c0)) and c0.sum() >= 2):
        raise ValueError('input outside the declared domain')
    if not (p.shape == (len(rho), len(c0)) and len(mu) == len(rho)):
        raise ValueError('input outside the declared domain')
    if not (np.all(p >= 0) and np.allclose(p.sum(1), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    if not (np.all(rho > 0) and np.all(mu >= 0) and abs(mu.sum()-1) <= 1e-9):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(time) and time > 0):
        raise ValueError('input outside the declared domain')

    g = np.asarray(g,float); z = np.asarray(mismatch,int)
    panels = np.asarray(panels,int); p = np.asarray(p,float)
    rho, mu = np.asarray(rho,float), np.asarray(mu,float)
    mcount,hcount = z.shape; k = len(p)
    nref = sum(counts)
    lam = 1./(math.log(nref)+.5); theta = lam/(2*(lam+nref))
    em = np.where(z,theta,1-theta)
    dg = np.r_[0.,np.diff(g)]*.01
    t = -np.expm1(-time*dg)
    r = -np.expm1(-rho[:,None]*dg[None,:])
    stay = (1.-t)[None,:]*(1.-r)
    within = (1.-t)[None,:]*r
    q = p/np.asarray(counts)[None,:]
    f = np.empty((mcount,k,hcount)); b = np.ones_like(f)
    f[0] = mu[:,None]*q[:,panels[0]]*em[0][None,:]
    if not (f[0].sum() > 0):
        raise ValueError('input outside the declared domain')
    f[0] /= f[0].sum()
    for m in range(1,mcount):
        shift = (t[m]*mu + within[:,m]*f[m-1].sum(axis=1))[:,None]*q[:,panels[m]]
        f[m] = (stay[:,m,None]*f[m-1]+shift)*em[m][None,:]
        if not (f[m].sum() > 0):
            raise ValueError('input outside the declared domain')
        f[m] /= f[m].sum()
    for m in range(mcount-2,-1,-1):
        v = b[m+1]*em[m+1][None,:]
        a = (v*q[:,panels[m+1]]).sum(axis=1)
        shift = t[m+1]*np.dot(mu,a)+within[:,m+1]*a
        b[m] = stay[:,m+1,None]*v+shift[:,None]
        b[m] /= b[m].sum()
    gamma = f*b
    gamma /= gamma.sum(axis=(1,2))[:,None,None]
    return gamma

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def _pack(value):
    values = value if isinstance(value, tuple) else (value,)
    pieces = []
    for v in values:
        a = np.asarray(v, dtype=float)
        pieces.append(np.r_[float(a.ndim), a.shape, a.ravel()])
    return np.concatenate(pieces)

def _error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

def test_cases():
    return [{'setup': (
        'import numpy as np\ng=[0.,.2,.3]; z=[[0,1,0],[1,0,0],[0,0,1'
        ']]; j=[[0,1,1],[1,0,1],[1,1,0]]; co=[30,70]; p=[[.7,.3],[.'
        '2,.8]]; rh=[20.,80.]; mu=[.4,.6]\ng,z,j,co,p,rh,mu=map(np.array,(g,z,j,co,p,rh,mu))'
    ), 'call': '_pack(ancestry_posterior(g,z,j,co,p,rh,mu,10.))', 'gold_call': \
        '_pack(_oracle_ancestry_posterior(g,z,j,co,p,rh,mu,10.))'}, {'setup': (
        'import numpy as np\ng=[0.]; z=[[0,1]]; j=[[0,1]]; co=[10,90'
        ']; p=[[.7,.3],[.2,.8]]; rh=[1.,100.]; mu=[.5,.5]\ng,z,j,co,p,rh,mu=map(np.array,(g,z,j,co,p,rh,mu))'
    ), 'call': '_pack(ancestry_posterior(g,z,j,co,p,rh,mu,10.))', 'gold_call': \
        '_pack(_oracle_ancestry_posterior(g,z,j,co,p,rh,mu,10.))'}, {'setup': (
        'import numpy as np\ng=[0.,1e-7,.05,.7]; z=[[0,1],[1,0],[1,1'
        '],[0,0]]; j=[[1,0],[0,1],[1,0],[0,1]]; co=[7,93]; p=[[.999'
        ',.001],[.001,.999],[.5,.5]]; rh=[.001,20.,2000.]; mu=[.01,'
        '.29,.7]\ng,z,j,co,p,rh,mu=map(np.array,(g,z,j,co,p,rh,mu))'
    ), 'call': '_pack(ancestry_posterior(g,z,j,co,p,rh,mu,3.))', 'gold_call': \
        '_pack(_oracle_ancestry_posterior(g,z,j,co,p,rh,mu,3.))'}, {'setup': (
        'import numpy as np\ng=[0.]; z=[[0,1]]; j=[[0,1]]; co=[0,10]'
        '; p=[[.4,.6]]; rh=[10.]; mu=[1.]\ng,z,j,co,p,rh,mu=map(np.array,(g,z,j,co,p,rh,mu))'
    ), 'call': '_error_code(ancestry_posterior,g,z,j,co,p,rh,mu,10.)', 'gold_call': \
        '_error_code(_oracle_ancestry_posterior,g,z,j,co,p,rh,mu,10.)'}]
