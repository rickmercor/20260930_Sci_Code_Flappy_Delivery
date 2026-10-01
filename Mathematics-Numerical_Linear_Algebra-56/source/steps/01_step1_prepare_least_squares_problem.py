"""
Prepare the common least-squares inputs.

A single least-squares instance provides the common numerical setting for both the iterative method and its external benchmark. Keeping that problem definition consistent allows later comparisons to reflect method behavior rather than differences in the underlying data.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], (A64, b64, x064).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def prepare_least_squares_problem(
    A: np.ndarray, b: np.ndarray, x0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the common least-squares inputs.

    Parameters
    ----------
    A : np.ndarray
        Two-dimensional real matrix with shape ``(n, d)``.
    b : np.ndarray
        Real right-hand side with shape ``(n,)``.
    x0 : np.ndarray
        Real initial iterate with shape ``(d,)``.

    Returns
    -------
    A64 : np.ndarray
        Finite float64 matrix with shape ``(n, d)``.
    b64 : np.ndarray
        Finite float64 vector with shape ``(n,)``.
    x064 : np.ndarray
        Finite float64 vector with shape ``(d,)``.

    Raises
    ------
    ValueError
        If conversion fails, dimensions are inconsistent, an array is empty,
        or any value is nonfinite.
    """
    return A64, b64, x064

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_prepare_least_squares_problem(A, b, x0):
    """Reference implementation for least-squares input preparation."""
    try:
        A64=np.asarray(A,dtype=np.float64)
        b64=np.asarray(b,dtype=np.float64)
        x064=np.asarray(x0,dtype=np.float64)
    except (TypeError,ValueError) as exc:
        raise ValueError("A, b, and x0 must be real numeric arrays") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1:
        raise ValueError("A must be a nonempty 2D array")
    n,d=A64.shape
    if b64.ndim!=1 or b64.shape!=(n,):
        raise ValueError("b must have shape (n,)")
    if x064.ndim!=1 or x064.shape!=(d,):
        raise ValueError("x0 must have shape (d,)")
    if not np.all(np.isfinite(A64)) or not np.all(np.isfinite(b64)) or not np.all(np.isfinite(x064)):
        raise ValueError("all entries must be finite")
    return A64.copy(),b64.copy(),x064.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[1,2],[3,4]],float); b=np.array([1,2]); x0=np.zeros(2)""",'call':'prepare_least_squares_problem(A,b,x0)','gold_call':'_oracle_prepare_least_squares_problem(A,b,x0)'},
        {'setup':"""import numpy as np
A=np.array([[1.],[2.],[3.]]); b=np.array([1.,0.,-1.]); x0=np.array([.5])""",'call':'prepare_least_squares_problem(A,b,x0)','gold_call':'_oracle_prepare_least_squares_problem(A,b,x0)'},
        {'setup':"""import numpy as np
A=np.array([[2,-1,0]],int); b=np.array([3]); x0=np.array([1,2,3])""",'call':'prepare_least_squares_problem(A,b,x0)','gold_call':'_oracle_prepare_least_squares_problem(A,b,x0)'},
        {'setup':"""import numpy as np
A=np.array([1.,2.]); b=np.array([1.,2.]); x0=np.array([0.])
def run_model():
    try: prepare_least_squares_problem(A,b,x0); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_prepare_least_squares_problem(A,b,x0); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); b=np.ones(3); x0=np.zeros(2)
def run_model():
    try: prepare_least_squares_problem(A,b,x0); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_prepare_least_squares_problem(A,b,x0); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.array([[1.,np.nan]]); b=np.array([1.]); x0=np.zeros(2)
def run_model():
    try: prepare_least_squares_problem(A,b,x0); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_prepare_least_squares_problem(A,b,x0); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
