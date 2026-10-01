"""
Compute a valid scenario-based subgradient of the finite-scenario upper-tail risk at one portfolio allocation.

At finite-support risk thresholds, several scenarios can share the same loss; the task fixes a deterministic convention for allocating any fractional tail mass.

Returns
-------
The returned vector has one component for each asset.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_cvar_subgradient(returns: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    """Return an asset-space subgradient of finite-scenario CVaR.

    Parameters
    ----------
    returns : np.ndarray
        Scenario-return matrix with shape (S, n).
    probabilities : np.ndarray
        Scenario probabilities of length S.
    weights : np.ndarray
        Portfolio weights of length n.
    beta : float
        Confidence level in (0, 1).

    Returns
    -------
    gradient : np.ndarray
        Subgradient vector in asset coordinates. When several scenarios share
        the threshold loss, the residual tail mass is assigned to them in
        increasing scenario-index order, each up to its full probability.

    Raises
    ------
    ValueError
        If dimensions or probability inputs are invalid.
    """
    return gradient

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_cvar_subgradient(returns: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    R=np.asarray(returns,dtype=float); p=np.asarray(probabilities,dtype=float); w=np.asarray(weights,dtype=float); b=float(beta)
    if R.ndim!=2 or w.ndim!=1 or p.ndim!=1 or R.shape[0]!=p.size or R.shape[1]!=w.size or p.size==0:
        raise ValueError("incompatible dimensions")
    if np.any(~np.isfinite(R)) or np.any(~np.isfinite(p)) or np.any(~np.isfinite(w)) or np.any(p<=0) or not np.isclose(p.sum(),1.0,atol=1e-12,rtol=0.0):
        raise ValueError("invalid finite scenario data")
    if not (0.0<b<1.0):
        raise ValueError("beta must lie in (0,1)")
    L=-R@w
    order=np.argsort(L,kind='stable')
    cum=0.0; u=None
    for j in order:
        cum += p[j]
        if cum >= b-1e-13:
            u=float(L[j]); break
    theta=np.zeros(p.size,dtype=float)
    equal=np.isclose(L,u,atol=1e-10,rtol=0.0)
    tail=L>u+1e-10
    theta[tail]=1.0
    rem=(1.0-b)-float(p[tail].sum())
    for j in np.where(equal)[0]:
        if rem <= 1e-12: break
        take=min(float(p[j]),rem)
        theta[j]=take/float(p[j])
        rem-=take
    if rem>1e-8: raise ValueError("tail mass could not be allocated")
    return -(p*theta)@R/(1.0-b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return tie and non-tie cases."""
    return [
        {"setup": "import numpy as np\nR=np.array([[-.15,.05,.03],[.04,-.13,.02],[.03,.02,-.12],[-.05,-.04,.06],[.08,.07,.09],[-.02,.03,-.06]]); p=np.array([.05,.1,.15,.2,.22,.28]); w=np.array([.45,.3,.25]); b=.65", "call": "tuple(compute_cvar_subgradient(R,p,w,b))", "gold_call": "tuple(_oracle_compute_cvar_subgradient(R,p,w,b))", "tol": 1e-12},
        {"setup": "import numpy as np\nR=np.array([[-.12,.04,.02],[.05,-.11,.03],[.02,.03,-.10],[-.04,-.03,.05],[.09,.08,.10],[-.03,.02,-.05]]); p=np.array([.08,.09,.12,.16,.24,.31]); w=np.array([.30,.45,.25]); b=.65", "call": "tuple(compute_cvar_subgradient(R,p,w,b))", "gold_call": "tuple(_oracle_compute_cvar_subgradient(R,p,w,b))", "tol": 1e-12},
        {"setup": "import numpy as np\nR=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]]); p=np.array([.2,.3,.1,.4]); w=np.array([.2,.3,.5]); b=.5", "call": "tuple(compute_cvar_subgradient(R,p,w,b))", "gold_call": "tuple(_oracle_compute_cvar_subgradient(R,p,w,b))", "tol": 1e-12},
        {"setup": "import numpy as np\nR=np.zeros((6,3)); p=np.array([.08,.09,.12,.16,.24,.31]); w=np.array([1/3,1/3,1/3]); b=.65", "call": "tuple(compute_cvar_subgradient(R,p,w,b))", "gold_call": "tuple(_oracle_compute_cvar_subgradient(R,p,w,b))", "tol": 1e-12},
    ]
