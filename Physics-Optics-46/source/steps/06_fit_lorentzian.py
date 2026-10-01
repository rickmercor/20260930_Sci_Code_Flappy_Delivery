"""
Fit a normalized Lorentzian half-width to a measured finite-record correlation profile.

For channel lag $$k$$ the model is $$h_k(a)=a^2/(k^2+a^2)$$, where $$a=\Gamma_{\mathrm{corr}}/\Delta\omega$$ and $$\Gamma_{\mathrm{corr}}$$ denotes half-width at half maximum. The benchmark uses the global minimum, on a supplied closed positive interval with endpoints included, of the unweighted residual $$E(a)=\sum_{k=1}^L[h_k(a)-c_k]^2$$, where $$c_k$$ is the supplied correlation at lag $$k$$. The fixed zero-lag value is excluded from the residual, and throughput is not a free fit amplitude. This fit is a task-defined estimator for a single finite cavity.

Returns
-------
np.ndarray: Real vector [a_fit, residual_sum_of_squares], shape (2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_lorentzian(profile: "np.ndarray", width_interval: "np.ndarray") -> "np.ndarray":
    r"""Fit a half-width in units of frequency-channel spacing.
    
    Parameters
    ----------
    profile : np.ndarray
        Real vector [T0, rho_0, ..., rho_L], L >= 1, finite T0 > 0,
        rho_0 = 1 and all rho values in [-1,1]. The loss described in the
        background has at most one interior stationary point on the interval.
    width_interval : np.ndarray
        Real vector [a_lower,a_upper], with 0 < a_lower < a_upper <= 30.
    
    Returns
    -------
    result : np.ndarray
        Real vector [a_fit, residual_sum_of_squares], shape (2,).
        The fit includes both endpoints and uses unweighted positive lags.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Return an
    endpoint exactly when it minimizes the loss. The unique minimum is supported.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_fit_lorentzian(profile: "np.ndarray", width_interval: "np.ndarray") -> "np.ndarray":
    rho = profile[2:]
    k2 = np.arange(1,len(rho)+1,dtype=float)**2
    lo, hi = np.log(width_interval)
    def _loss_derivative(u):
        a2 = np.exp(2*u)
        h = a2/(k2+a2)
        hp = 2*h*(1-h)
        return float(2*np.dot(h-rho,hp))
    points = [lo,hi]
    if _loss_derivative(lo)*_loss_derivative(hi)<0:
        points.append(brentq(_loss_derivative,lo,hi,xtol=1e-13))
    values = [np.sum((np.exp(2*u)/(k2+np.exp(2*u))-rho)**2) for u in points]
    index = int(np.argmin(values))
    a = width_interval[index] if index<2 else np.exp(points[index])
    return np.array([a,values[index]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit differential cases for Studio contract validation."""
    return [
        {
            "setup": "import numpy as np\nk=np.arange(1,9,dtype=float)\nprofile=np.r_[.7,1.,3.2**2/(k*k+3.2**2)]\nbounds=np.array([.05,20.])\ndef measure(fn):\n    p=profile.copy(); b=bounds.copy(); out=np.asarray(fn(p,b))\n    assert out.shape==(2,) and np.array_equal(p,profile) and np.array_equal(b,bounds)\n    return out\n",
            "call": "measure(fit_lorentzian)",
            "gold_call": "measure(_oracle_fit_lorentzian)",
            "tol": 2e-06
        },
        {
            "setup": "import numpy as np\nk=np.arange(1,9,dtype=float)\nprofile=np.r_[.7,1.,3.2**2/(k*k+3.2**2)]\nbounds=np.array([.05,20.])\ndef measure(fn):\n    p=profile.copy(); b=bounds.copy(); out=np.asarray(fn(p,b))\n    assert out.shape==(2,) and np.array_equal(p,profile) and np.array_equal(b,bounds)\n    return out\nprofile[2:]=0\n",
            "call": "measure(fit_lorentzian)",
            "gold_call": "measure(_oracle_fit_lorentzian)",
            "tol": 2e-07
        },
        {
            "setup": "import numpy as np\nk=np.arange(1,9,dtype=float)\nprofile=np.r_[.7,1.,3.2**2/(k*k+3.2**2)]\nbounds=np.array([.05,20.])\ndef measure(fn):\n    p=profile.copy(); b=bounds.copy(); out=np.asarray(fn(p,b))\n    assert out.shape==(2,) and np.array_equal(p,profile) and np.array_equal(b,bounds)\n    return out\nprofile[2:]=1\n",
            "call": "measure(fit_lorentzian)",
            "gold_call": "measure(_oracle_fit_lorentzian)",
            "tol": 2e-07
        },
        {
            "setup": "import numpy as np\nk=np.arange(1,9,dtype=float)\nprofile=np.r_[.7,1.,3.2**2/(k*k+3.2**2)]\nbounds=np.array([.05,20.])\ndef measure(fn):\n    p=profile.copy(); b=bounds.copy(); out=np.asarray(fn(p,b))\n    assert out.shape==(2,) and np.array_equal(p,profile) and np.array_equal(b,bounds)\n    return out\nprofile=np.array([.7,1.,.83,.57,.31,.14,.04,-.03,-.1,-.15])\n",
            "call": "measure(fit_lorentzian)",
            "gold_call": "measure(_oracle_fit_lorentzian)",
            "tol": 2e-07
        },
        {
            "setup": "import numpy as np\nk=np.arange(1,9,dtype=float)\nprofile=np.r_[.7,1.,3.2**2/(k*k+3.2**2)]\nbounds=np.array([.05,20.])\ndef measure(fn):\n    p=profile.copy(); b=bounds.copy(); out=np.asarray(fn(p,b))\n    assert out.shape==(2,) and np.array_equal(p,profile) and np.array_equal(b,bounds)\n    return out\nprofile=np.array([.4,1.,.2]); bounds=np.array([.1,5.])\n",
            "call": "measure(fit_lorentzian)",
            "gold_call": "measure(_oracle_fit_lorentzian)",
            "tol": 2e-06
        }
    ]
