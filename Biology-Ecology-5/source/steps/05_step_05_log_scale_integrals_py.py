"""
Evaluate logarithms of diffusion scale integrals and their mixed parameter derivatives.

The backward diffusion equation determines a scale density. Logarithmic scale integrals retain the probability normalization and allow environmental differentiation without overflowing large exponentials.

Returns
-------
Shape (4,K): value and epsilon, zeta, epsilon-zeta derivatives of the natural log of each scale integral for the absorbing diffusion df=f*(1-f)(s-etaf)dt+sqrt(2Df*(1-f)/N0)dW. Use scale density normalized to one at frequency zero and the stated quadrature order. All derivatives are ordinary, not factorial-scaled. Large finite exponents must be handled by a shift before exponentiation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def log_scale_integrals(selection: 'np.ndarray', N0: float, D: float, intervals: 'np.ndarray', order: int = 64) -> 'np.ndarray':
    """Evaluate logarithms of diffusion scale integrals and their mixed parameter derivatives.

    Parameters
    ----------
    selection : np.ndarray
        Finite (4,2) jet of (s,eta), rows value, epsilon, zeta, epsilon-zeta.
    N0 : float
        Positive fixed combined abundance.
    D : float
        Positive fixed noise coefficient, lineage variance rate 2*D*N.
    intervals : np.ndarray
        Finite (K,2), K >= 1, with 0 <= lower < upper <= 1 in each row.
    order : int
        Gauss-Legendre nodes per interval, integer in [16,256], not boolean.

    Returns
    -------
    result : np.ndarray
        Shape (4,K): value and epsilon, zeta, epsilon-zeta derivatives of
        the natural log of each scale integral for the absorbing diffusion
        df=f*(1-f)*(s-eta*f)dt+sqrt(2*D*f*(1-f)/N0)dW.
        Use scale density normalized to one at frequency zero and the stated
        quadrature order. All derivatives are ordinary, not factorial-scaled.
        Large finite exponents must be handled by a shift before exponentiation.

    Raises
    ------
    ValueError
        If selection or intervals are nonreal, nonfinite or have invalid shapes,
        interval bounds fail 0 <= lower < upper <= 1, N0 or D is nonpositive
        or nonfinite, order is not an integer in [16,256] or is boolean,
        or effective size, arithmetic or final results are nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _diffusion_inputs(selection,N0,D):
    selection=_arr(selection,2,(4,2));N0=float(_arr(N0,0));D=float(_arr(D,0))
    if N0<=0 or D<=0:raise ValueError('positive abundance and noise required')
    n=N0/(2*D)
    if not np.isfinite(n):raise ValueError('nonfinite effective size')
    return selection,n

def _order(order):
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 16<=order<=256:raise ValueError('order must be an integer in [16,256]')
    return np.polynomial.legendre.leggauss(order)

def _potential(sel,n,x):
    return n*(sel[:,1,None]*x*x-2*sel[:,0,None]*x)

def _exp(a):
    e=np.exp(a[0]);return np.stack([e,e*a[1],e*a[2],e*(a[3]+a[1]*a[2])])

def _log(a):
    if np.any(a[0]<=0): raise ValueError('positive baseline needed')
    return np.stack([np.log(a[0]),a[1]/a[0],a[2]/a[0],a[3]/a[0]-a[1]*a[2]/a[0]**2])

def _oracle_log_scale_integrals(selection: 'np.ndarray', N0: float, D: float, intervals: 'np.ndarray', order: int = 64) -> 'np.ndarray':
    sel,n=_diffusion_inputs(selection,N0,D);iv=_arr(intervals,2)
    if iv.shape[1]!=2 or iv.shape[0]<1 or np.any(iv[:,0]<0) or np.any(iv[:,1]>1) or np.any(iv[:,0]>=iv[:,1]):raise ValueError('intervals must obey 0<=lo<hi<=1')
    nodes,weights=_order(order);out=[]
    for lo,hi in iv:
        x=lo+(hi-lo)*(nodes+1)/2;V=_potential(sel,n,x);_arr(V)
        shift=V[0].max();V[0]-=shift
        integral=(_exp(V)*weights).sum(axis=1)*(hi-lo)/2
        lg=_log(integral);lg[0]+=shift;out.append(lg)
    return _arr(np.stack(out,axis=1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Four original cases plus overflow/underflow and cancellation coverage."""
    return [
        {
            "setup": """import numpy as np
import copy
# normal
a0=np.array([[1.2011099483335636e-06, 2.4022125203113796e-06], [0.00030386556673751256, 1.590454928850751e-07], [0.0004714218741368572, 2.3138128145951028e-07], [3.692447978065164e-05, 3.976566960660779e-09]], dtype=float)
a1=0.1
a2=2e-08
a3=np.array([[0.0, 0.07], [0.07, 1.0], [0.0, 1.0]], dtype=float)
""",
            "call": 'log_scale_integrals(a0.copy(), a1, a2, a3.copy())',
            "gold_call": '_oracle_log_scale_integrals(a0.copy(), a1, a2, a3.copy())',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# boundary
a0=np.array([[0.0, 0.0], [0.01, 0.02], [-0.02, 0.01], [0.003, -0.004]], dtype=float)
a1=1.0
a2=0.1
a3=np.array([[0.0, 1.0], [0.2, 0.8]], dtype=float)
""",
            "call": 'log_scale_integrals(a0.copy(), a1, a2, a3.copy())',
            "gold_call": '_oracle_log_scale_integrals(a0.copy(), a1, a2, a3.copy())',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# edge
a0=np.array([[-0.004, 0.0], [0.001, 0.0002], [0.0003, -0.0001], [1e-05, 2e-05]], dtype=float)
a1=1.0
a2=5e-06
a3=np.array([[0.0, 0.4], [0.4, 1.0]], dtype=float)
a4=128
""",
            "call": 'log_scale_integrals(a0.copy(), a1, a2, a3.copy(), a4)',
            "gold_call": '_oracle_log_scale_integrals(a0.copy(), a1, a2, a3.copy(), a4)',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# invalid_declared_condition
a0=np.array([[1.2011099483335636e-06, 2.4022125203113796e-06], [0.00030386556673751256, 1.590454928850751e-07], [0.0004714218741368572, 2.3138128145951028e-07], [3.692447978065164e-05, 3.976566960660779e-09]], dtype=float)
a1=0.1
a2=2e-08
a3=np.array([[0.4, 0.4]], dtype=float)

def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": 'expect_value_error(log_scale_integrals, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2), copy.deepcopy(a3))',
            "gold_call": 'expect_value_error(_oracle_log_scale_integrals, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2), copy.deepcopy(a3))',
            "tol": 0,
        },
        {
            "setup": """import numpy as np
a0 = np.array([[-0.008, 0.0], [0.0001, 0.0002], [-0.0002, 0.0001], [3e-05, -4e-05]], dtype=float)
a1 = 1.0
a2 = 1e-5
a3 = np.array([[0.0, 0.25], [0.25, 1.0], [0.9, 1.0]], dtype=float)
a4 = 128
""",
            "call": 'log_scale_integrals(a0.copy(), a1, a2, a3.copy(), a4)',
            "gold_call": '_oracle_log_scale_integrals(a0.copy(), a1, a2, a3.copy(), a4)',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
a0 = np.array([[0.008, 0.0], [0.0001, 0.0002], [-0.0002, 0.0001], [3e-05, -4e-05]], dtype=float)
a1 = 1.0
a2 = 1e-5
a3 = np.array([[0.0, 0.25], [0.25, 1.0], [0.9, 1.0]], dtype=float)
a4 = 128
""",
            "call": 'log_scale_integrals(a0.copy(), a1, a2, a3.copy(), a4)',
            "gold_call": '_oracle_log_scale_integrals(a0.copy(), a1, a2, a3.copy(), a4)',
            "tol": 1e-08,
        },
    ]
