"""
Evaluate interacting thermal means, impurity charge variance and the analytic response to two chemical potentials.

The derivative of a noncommuting Gibbs state is a matrix-exponential Frechet derivative, not an ordinary product covariance. Degenerate energy levels require its continuous divided-difference limit. The impurity second moment is the expectation of the squared operator.

Returns
-------
tuple[np.ndarray, float, np.ndarray], means in (total,impurity) order, native-float impurity variance, and the 2x2 analytic susceptibility.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thermal_response(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, mu: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Evaluate the full thermal state and analytic two-field susceptibility.

    Parameters
    ----------
    H, NA, NT : np.ndarray
        Real finite symmetric (d,d) operators of equal nonzero size. They
        need not commute. Symmetry tolerance is 1e-9; symmetrize accepted
        inputs. Operators may be general observables for testing.
    beta : float
        Real finite scalar, 0 < beta <= 100.
    mu : np.ndarray
        Real finite shape (2,), ordered (mu_gc,mu_imp).

    Returns
    -------
    (means, variance, J) : tuple[np.ndarray, float, np.ndarray]
        For K=H-mu[0]*NT-mu[1]*NA and rho=exp(-beta*K)/Z,
        means=(Tr(rho NT),Tr(rho NA)), variance=Tr(rho NA@NA)-means[1]**2,
        and J[i,j]=d means[i]/d mu[j], at fixed H,NA,NT,beta.
        J must be analytic, including noncommuting eigenvector response
        and continuous limits at degeneracies; finite differences are
        excluded. Do not clip the returned variance or response.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real float arrays, operator
        shapes/symmetry or mu shape violate the above, or beta is not a
        real finite scalar in (0,100].
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_thermal_response(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, mu: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if any(np.iscomplexobj(x) for x in [H,NA,NT,mu,beta]) or np.ndim(beta)!=0: raise ValueError('real inputs')
        H=np.asarray(H,dtype=float); NA=np.asarray(NA,dtype=float); NT=np.asarray(NT,dtype=float); mu=np.asarray(mu,dtype=float); beta=float(beta)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if H.ndim!=2 or H.shape[0]!=H.shape[1] or len(H)<1 or NA.shape!=H.shape or NT.shape!=H.shape or mu.shape!=(2,): raise ValueError('shape')
    if not all(np.isfinite(x).all() for x in [H,NA,NT,mu]) or not np.isfinite(beta) or not 0<beta<=100: raise ValueError('finite inputs')
    if any(np.max(np.abs(x-x.T))>1e-9 for x in [H,NA,NT]): raise ValueError('symmetry')
    H=(H+H.T)/2; NA=(NA+NA.T)/2; NT=(NT+NT.T)/2
    e,V=np.linalg.eigh(H-mu[0]*NT-mu[1]*NA)
    w=np.exp(-beta*(e-e.min())); w/=w.sum()
    ops=[V.T@NT@V,V.T@NA@V]
    means=np.array([w@np.diag(O) for O in ops])
    variance=float(w@np.sum(ops[1]**2,axis=1)-means[1]**2)
    gap=np.abs(e[:,None]-e[None,:]); x=beta*gap
    ratio=np.ones_like(x)
    np.divide(-np.expm1(-x),x,out=ratio,where=x!=0)
    L=beta*np.maximum(w[:,None],w[None,:])*ratio
    J=np.empty((2,2))
    for i in range(2):
        for j in range(2): J[i,j]=np.sum(L*ops[i]*ops[j])-beta*means[i]*means[j]
    return means,variance,(J+J.T)/2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return six differential cases for thermal moments and response."""

    capture = """import numpy as np

def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
"""

    return [
        {
            "setup": """import numpy as np
H = np.array([
    [0.0, 0.3, 0.1],
    [0.3, 0.7, -0.2],
    [0.1, -0.2, 1.2],
])
NA = np.diag([0.0, 1.0, 2.0])
NT = np.diag([0.0, 2.0, 3.0])
beta = 1.7
mu = np.array([0.2, -0.1])
""",
            "call": "thermal_response(H, NA, NT, beta, mu)",
            "gold_call": "_oracle_thermal_response(H, NA, NT, beta, mu)",
        },
        {
            "setup": """import numpy as np
H = np.zeros((3, 3))
NA = np.diag([0.0, 1.0, 2.0])
NT = np.diag([1.0, 0.0, 2.0])
beta = 2.0
mu = np.zeros(2)
""",
            "call": "thermal_response(H, NA, NT, beta, mu)",
            "gold_call": "_oracle_thermal_response(H, NA, NT, beta, mu)",
        },
        {
            "setup": """import numpy as np
H = np.array([[0.0, 0.3], [0.3, 1e-12]])
NA = np.diag([0.0, 2.0])
NT = np.array([[0.2, 0.5], [0.5, 1.0]])
beta = 60.0
mu = np.array([0.1, 0.2])
""",
            "call": "thermal_response(H, NA, NT, beta, mu)",
            "gold_call": "_oracle_thermal_response(H, NA, NT, beta, mu)",
        },
        {
            "setup": capture + """
H = np.eye(2)
NA = np.eye(2)
NT = np.eye(2)
beta = 0.0
mu = np.zeros(2)
""",
            "call": "_exception_code(thermal_response, H, NA, NT, beta, mu)",
            "gold_call": "_exception_code(_oracle_thermal_response, H, NA, NT, beta, mu)",
        },
        {
            "setup": capture + """
H = np.eye(2)
NA = np.eye(2)
NT = np.eye(2)
beta = 1.0
mu = np.ones(3)
""",
            "call": "_exception_code(thermal_response, H, NA, NT, beta, mu)",
            "gold_call": "_exception_code(_oracle_thermal_response, H, NA, NT, beta, mu)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(118)
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
NA0 = np.array([
    [0.0, 0.4, 0.2, 0.0],
    [0.4, 1.0, -0.3, 0.1],
    [0.2, -0.3, 2.0, 0.5],
    [0.0, 0.1, 0.5, 3.0],
])
NT0 = np.array([
    [1.0, 0.2, 0.0, 0.3],
    [0.2, 2.0, 0.4, 0.0],
    [0.0, 0.4, 1.0, -0.2],
    [0.3, 0.0, -0.2, 3.0],
])
NA = Q @ NA0 @ Q.T
NT = Q @ NT0 @ Q.T
mu = np.array([0.35, -0.45])
K = Q @ np.diag([0.0, 0.0, 0.7, 0.7]) @ Q.T
H = K + mu[0] * NT + mu[1] * NA
beta = 3.2
""",
            "call": "thermal_response(H, NA, NT, beta, mu)",
            "gold_call": "_oracle_thermal_response(H, NA, NT, beta, mu)",
        },
    ]
