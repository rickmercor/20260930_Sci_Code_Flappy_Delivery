"""
Establish the least-squares reference and orthonormal-sketch benchmark.

Random orthonormal sketching gives a finite-dimensional accuracy reference relative to the underlying least-squares problem. The applicable benchmark depends on the problem geometry, sketch dimension, and field, so all quantities must refer to the same instance.

Returns
-------
tuple[float,int,float], the optimal squared residual, numerical rank, and sharp orthonormal inflation rho_orth.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def least_squares_orthonormal_reference(
    A: np.ndarray, b: np.ndarray, ell: int, field: str = "real"
) -> tuple[float, int, float]:
    """Establish the least-squares reference and orthonormal-sketch benchmark.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    b : np.ndarray
        Finite real vector with shape ``(n,)``.
    ell : int
        Random-orthonormal sketch-and-solve embedding dimension.
    field : str, default="real"
        Either ``"real"`` or ``"complex"``.

    Returns
    -------
    optimal_residual_sq : float
        Minimum unsketched squared Euclidean residual.
    rank : int
        Numerical rank returned by ``numpy.linalg.lstsq`` with ``rcond=None``.
    rho_orth : float
        Exact finite-dimensional expected squared-residual inflation for the
        random-orthonormal sketch-and-solve benchmark.

    Raises
    ------
    ValueError
        If inputs are invalid, the least-squares solve fails, the optimum is
        zero/nonfinite, or ``n``, ``rank``, ``ell``, and ``field`` violate the
        paper's benchmark domain.
    """
    return optimal_residual_sq, rank, rho_orth

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_least_squares_orthonormal_reference(A,b,ell,field="real"):
    """Reference implementation of the coupled LS/orthonormal benchmark."""
    try: A64=np.asarray(A,dtype=np.float64); b64=np.asarray(b,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("A and b must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1: raise ValueError("A must be nonempty 2D")
    n=A64.shape[0]
    if b64.ndim!=1 or b64.shape!=(n,): raise ValueError("b shape mismatch")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(b64)): raise ValueError("nonfinite data")
    try: x,_,rank,_=np.linalg.lstsq(A64,b64,rcond=None)
    except np.linalg.LinAlgError as exc: raise ValueError("least-squares solve failed") from exc
    r=b64-A64@x; opt=float(r@r); rank=int(rank)
    if not np.isfinite(opt) or opt<=0.0: raise ValueError("optimal residual must be positive finite")
    if isinstance(ell,(bool,np.bool_)) or not isinstance(ell,(int,np.integer)): raise ValueError("ell must be integer")
    ell=int(ell)
    if field not in {"real","complex"}: raise ValueError("field must be real or complex")
    alpha=1 if field=="real" else 0
    if ell<1 or ell>n or not (rank < ell-alpha): raise ValueError("dimensions outside theorem domain")
    rho=1.0+((n-ell)/(n-rank))*(rank/(ell-rank-alpha))
    if not np.isfinite(rho): raise ValueError("nonfinite benchmark")
    return float(opt),rank,float(rho)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,-1.],[1.,2.]])
b=np.array([1.,-1.,.2,2.,.5])""",'call':"least_squares_orthonormal_reference(A,b,4,'real')",'gold_call':"_oracle_least_squares_orthonormal_reference(A,b,4,'real')"},
        {'setup':"""import numpy as np
A=np.array([[1.,2.],[2.,4.],[3.,6.],[1.,0.],[0.,1.]])
b=np.array([1.,0.,1.,.5,-.25])""",'call':"least_squares_orthonormal_reference(A,b,4,'real')",'gold_call':"_oracle_least_squares_orthonormal_reference(A,b,4,'real')"},
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,-1.]])
b=np.array([1.,2.,0.,1.])""",'call':"least_squares_orthonormal_reference(A,b,3,'complex')",'gold_call':"_oracle_least_squares_orthonormal_reference(A,b,3,'complex')"},
        {'setup':"""import numpy as np
A=np.eye(2); b=np.array([1.,2.])
def run_model():
    try: least_squares_orthonormal_reference(A,b,2,'real'); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_least_squares_orthonormal_reference(A,b,2,'real'); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[0.,1.],[1.,1.]]); b=np.array([1.,2.,0.])
def run_model():
    try: least_squares_orthonormal_reference(A,b,2,'real'); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_least_squares_orthonormal_reference(A,b,2,'real'); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[0.,1.],[1.,1.]]); b=np.array([1.,2.,0.])
def run_model():
    try: least_squares_orthonormal_reference(A,b,3.0,'complex'); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_least_squares_orthonormal_reference(A,b,3.0,'complex'); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
