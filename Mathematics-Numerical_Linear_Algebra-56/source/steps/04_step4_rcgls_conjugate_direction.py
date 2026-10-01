"""
Construct the RCGLS conjugate transition.

Several conjugate-gradient-style coefficients are compatible with the available state variables, but RCGLS uses a method-specific randomized transition. This step implements the transition selected by the RCGLS definition rather than substituting another familiar conjugate-gradient rule.

Returns
-------
tuple[np.ndarray,np.ndarray,float,float,np.ndarray,float,float,float], selected RCGLS direction, its image, conjugacy coefficient, RCGLS step coefficient, updated residual, squared residual, conjugacy defect, and residual/search-image defect.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rcgls_conjugate_direction(
    A: np.ndarray,
    r_next: np.ndarray,
    S_next: np.ndarray,
    p_prev: np.ndarray,
    v_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float, float, float]:
    """Construct the RCGLS conjugate transition.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    r_next : np.ndarray
        Finite residual after the preceding RCGLS update, shape ``(n,)``.
    S_next : np.ndarray
        Finite next sketch matrix with shape ``(d, q)`` or sketch vector with
        shape ``(d,)``.
    p_prev : np.ndarray
        Finite preceding RCGLS search direction with shape ``(d,)``.
    v_prev : np.ndarray
        Finite preceding search-direction image ``A @ p_prev``, shape ``(n,)``.

    Returns
    -------
    p_next : np.ndarray
        Corrected next RCGLS search direction with shape ``(d,)``.
    v_next : np.ndarray
        Image ``A @ p_next`` with shape ``(n,)``.
    tau : float
        Signed RCGLS conjugacy coefficient in the convention
        ``p_next = seed + tau * p_prev``.
    mu_next : float
        RCGLS step coefficient for the selected next direction. For a valid
        RCGLS trajectory state, it coincides with the exact line-search
        coefficient along that direction.
    r_after : np.ndarray
        Residual after applying the RCGLS step.
    residual_sq : float
        Squared Euclidean norm of ``r_after``.
    conjugacy_defect : float
        Inner product ``v_next @ v_prev``; it should vanish up to roundoff.
    line_search_defect : float
        Residual/search-image inner product after the update; it vanishes up
        to roundoff for a valid RCGLS trajectory state.

    Raises
    ------
    ValueError
        If inputs are invalid, ``v_prev`` is inconsistent with ``A @ p_prev``,
        a required denominator vanishes, or the coupled RCGLS transition is
        nonfinite or degenerate.
    """
    return p_next, v_next, tau, mu_next, r_after, residual_sq, conjugacy_defect, line_search_defect

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rcgls_conjugate_direction(A,r_next,S_next,p_prev,v_prev):
    """Reference implementation of a coupled RCGLS conjugate/update transition."""
    try:
        A64=np.asarray(A,dtype=np.float64); r64=np.asarray(r_next,dtype=np.float64)
        p64=np.asarray(p_prev,dtype=np.float64); v64=np.asarray(v_prev,dtype=np.float64)
        S64=np.asarray(S_next,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("inputs must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1: raise ValueError("A must be nonempty 2D")
    n,d=A64.shape
    if r64.ndim!=1 or r64.shape!=(n,) or p64.ndim!=1 or p64.shape!=(d,) or v64.ndim!=1 or v64.shape!=(n,): raise ValueError("shape mismatch")
    if S64.ndim==1:
        if S64.shape!=(d,): raise ValueError("S shape mismatch")
        Smat=S64.reshape(d,1)
    elif S64.ndim==2:
        if S64.shape[0]!=d or S64.shape[1]<1: raise ValueError("S shape mismatch")
        Smat=S64
    else: raise ValueError("S must be vector or matrix")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(r64)) or not np.all(np.isfinite(p64)) or not np.all(np.isfinite(v64)) or not np.all(np.isfinite(Smat)): raise ValueError("nonfinite input")
    Ap=A64@p64
    if not np.allclose(Ap,v64,rtol=1e-12,atol=1e-12): raise ValueError("v_prev inconsistent with A @ p_prev")
    prev_den=float(v64@v64)
    if not np.isfinite(prev_den) or prev_den<=0.0: raise ValueError("zero previous image")
    g=A64.T@r64
    base=Smat@(Smat.T@g)
    Abase=A64@base
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(Abase)) or float(Abase@Abase)<=0.0: raise ValueError("invalid sketch seed")
    tau=-float(Abase@v64)/prev_den
    p_next=base+tau*p64
    v_next=Abase+tau*v64
    next_den=float(v_next@v_next)
    if not np.isfinite(tau) or not np.all(np.isfinite(p_next)) or not np.all(np.isfinite(v_next)) or next_den<=0.0: raise ValueError("invalid corrected direction")
    dnext=Smat.T@g
    mu_next=float(dnext@dnext)/next_den
    if not np.isfinite(mu_next): raise ValueError("nonfinite mu")
    r_after=r64-mu_next*v_next
    residual_sq=float(r_after@r_after)
    conjugacy_defect=float(v_next@v64)
    line_search_defect=float(r_after@v_next)
    if not np.all(np.isfinite(r_after)) or not np.isfinite(residual_sq) or not np.isfinite(conjugacy_defect) or not np.isfinite(line_search_defect): raise ValueError("nonfinite updated state")
    return (np.asarray(p_next,dtype=np.float64),np.asarray(v_next,dtype=np.float64),float(tau),
            float(mu_next),np.asarray(r_after,dtype=np.float64),float(residual_sq),
            float(conjugacy_defect),float(line_search_defect))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[2.,0.],[1.,3.],[0.,1.]])
r=np.array([.5,-1.,2.]); S=np.array([0.,1./np.sqrt(10.)]); p=np.array([.4,0.]); v=A@p""",'call':'rcgls_conjugate_direction(A,r,S,p,v)','gold_call':'_oracle_rcgls_conjugate_direction(A,r,S,p,v)'},
        {'setup':"""import numpy as np
A=np.array([[1.,2.,0.],[0.,1.,3.],[2.,-1.,1.],[1.,0.,1.]])
p=np.array([.2,-.1,.3]); v=A@p
r=np.array([1.,-1.2058823529411766,1.2941176470588234,-.14117647058823535])
S=np.array([0.,0.,1./np.sqrt(11.)])""",'call':'rcgls_conjugate_direction(A,r,S,p,v)','gold_call':'_oracle_rcgls_conjugate_direction(A,r,S,p,v)'},
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[0.,2.],[1.,1.]])
p=np.array([.5,.25]); v=A@p
r=np.array([.6470588235294117,1.6470588235294117,-1.5294117647058822])
S=np.eye(2)""",'call':'rcgls_conjugate_direction(A,r,S,p,v)','gold_call':'_oracle_rcgls_conjugate_direction(A,r,S,p,v)'},
        {'setup':"""import numpy as np
A=np.eye(2); r=np.ones(2); S=np.array([1.,0.]); p=np.array([1.,0.]); v=np.array([2.,0.])
def run_model():
    try: rcgls_conjugate_direction(A,r,S,p,v); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_conjugate_direction(A,r,S,p,v); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); r=np.ones(2); S=np.array([1.,0.]); p=np.zeros(2); v=np.zeros(2)
def run_model():
    try: rcgls_conjugate_direction(A,r,S,p,v); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_conjugate_direction(A,r,S,p,v); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); r=np.array([1.,np.nan]); S=np.array([1.,0.]); p=np.array([1.,0.]); v=A@p
def run_model():
    try: rcgls_conjugate_direction(A,r,S,p,v); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_conjugate_direction(A,r,S,p,v); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
