"""
Form the RCGLS information associated with the current sketch.

A sketch contributes localized information to an RCGLS iteration. This information links the sampled coordinate choice to the least-squares residual geometry that governs how the iterative state evolves.

Returns
-------
tuple[np.ndarray,np.ndarray], float64 arrays seed of shape (d,) and image of shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rcgls_sketch_seed(A: np.ndarray, r: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Form the RCGLS information associated with the current sketch.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    r : np.ndarray
        Finite residual vector with shape ``(n,)``.
    S : np.ndarray
        Finite sketch matrix with shape ``(d, q)`` or a finite sketch vector
        with shape ``(d,)``.

    Returns
    -------
    seed : np.ndarray
        Float64 coefficient-space RCGLS sketched-gradient seed, shape ``(d,)``.
    image : np.ndarray
        Float64 vector ``A @ seed`` with shape ``(n,)``.

    Raises
    ------
    ValueError
        If conversion fails, dimensions are inconsistent, any value is
        nonfinite, the sketch has zero columns, or the resulting seed/image is
        zero or nonfinite.
    """
    return seed, image

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rcgls_sketch_seed(A,r,S):
    """Reference implementation of the RCGLS sketched-gradient seed."""
    try:
        A64=np.asarray(A,dtype=np.float64); r64=np.asarray(r,dtype=np.float64); S64=np.asarray(S,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("inputs must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1: raise ValueError("A must be nonempty 2D")
    n,d=A64.shape
    if r64.ndim!=1 or r64.shape!=(n,): raise ValueError("r shape mismatch")
    if S64.ndim==1:
        if S64.shape!=(d,): raise ValueError("S shape mismatch")
        Smat=S64.reshape(d,1)
    elif S64.ndim==2:
        if S64.shape[0]!=d or S64.shape[1]<1: raise ValueError("S shape mismatch")
        Smat=S64
    else: raise ValueError("S must be vector or matrix")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(r64)) or not np.all(np.isfinite(Smat)): raise ValueError("nonfinite input")
    g=A64.T@r64
    seed=Smat@(Smat.T@g)
    image=A64@seed
    if not np.all(np.isfinite(seed)) or not np.all(np.isfinite(image)): raise ValueError("nonfinite seed")
    if float(image@image)<=0.0: raise ValueError("zero search image")
    return np.asarray(seed,dtype=np.float64),np.asarray(image,dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[2.,0.,1.],[1.,3.,0.],[0.,1.,2.]])
r=np.array([1.,-2.,.5]); S=np.array([0.,.2,0.])""",'call':'rcgls_sketch_seed(A,r,S)','gold_call':'_oracle_rcgls_sketch_seed(A,r,S)'},
        {'setup':"""import numpy as np
A=np.array([[1.,2.],[3.,-1.],[2.,1.]])
r=np.array([.5,1.,-2.]); S=np.array([[1.,0.],[0.,.5]])""",'call':'rcgls_sketch_seed(A,r,S)','gold_call':'_oracle_rcgls_sketch_seed(A,r,S)'},
        {'setup':"""import numpy as np
A=np.array([[2.,1.],[0.,3.],[1.,-1.]])
r=np.array([1.,-2.,.5]); S=np.array([0.,1./np.sqrt(11.)])""",'call':'rcgls_sketch_seed(A,r,S)','gold_call':'_oracle_rcgls_sketch_seed(A,r,S)'},
        {'setup':"""import numpy as np
A=np.eye(2); r=np.ones(3); S=np.array([1.,0.])
def run_model():
    try: rcgls_sketch_seed(A,r,S); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_sketch_seed(A,r,S); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); r=np.ones(2); S=np.zeros(2)
def run_model():
    try: rcgls_sketch_seed(A,r,S); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_sketch_seed(A,r,S); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); r=np.array([1.,np.nan]); S=np.array([1.,0.])
def run_model():
    try: rcgls_sketch_seed(A,r,S); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_sketch_seed(A,r,S); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
