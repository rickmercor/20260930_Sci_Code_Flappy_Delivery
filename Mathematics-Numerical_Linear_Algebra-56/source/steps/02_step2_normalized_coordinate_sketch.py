"""
Form the sketch associated with a selected coordinate.

Coordinate sketches provide low-dimensional views tied to individual coordinates of a least-squares problem. Within RCGLS, the selected sketch determines which local information is exposed at a given iteration while remaining connected to the geometry of the original system.

Returns
-------
np.ndarray, S.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def normalized_coordinate_sketch(A: np.ndarray, index: int) -> np.ndarray:
    """Construct a one-column normalized coordinate sketch vector.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    index : int
        One-based coordinate index in ``{1, ..., d}``.

    Returns
    -------
    S : np.ndarray
        Float64 vector of shape ``(d,)`` representing the normalized selected
        coordinate.

    Raises
    ------
    ValueError
        If ``A`` is invalid, ``index`` is not a valid one-based integer, or
        the selected column of ``A`` has zero norm.
    """
    return S

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_normalized_coordinate_sketch(A, index):
    """Reference implementation for the normalized coordinate sketch."""
    try: A64=np.asarray(A,dtype=np.float64)
    except (TypeError,ValueError) as exc: raise ValueError("A must be real numeric") from exc
    if A64.ndim!=2 or A64.shape[0]<1 or A64.shape[1]<1 or not np.all(np.isfinite(A64)):
        raise ValueError("A must be a nonempty finite 2D array")
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)):
        raise ValueError("index must be an integer")
    j=int(index)-1
    if j<0 or j>=A64.shape[1]: raise ValueError("index out of range")
    norm=float(np.linalg.norm(A64[:,j]))
    if not np.isfinite(norm) or norm<=0.0: raise ValueError("selected column must have positive norm")
    S=np.zeros(A64.shape[1],dtype=np.float64); S[j]=1.0/norm
    return S

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[3.,0.],[4.,2.]])""",'call':'normalized_coordinate_sketch(A,1)','gold_call':'_oracle_normalized_coordinate_sketch(A,1)'},
        {'setup':"""import numpy as np
A=np.eye(3)*2""",'call':'normalized_coordinate_sketch(A,3)','gold_call':'_oracle_normalized_coordinate_sketch(A,3)'},
        {'setup':"""import numpy as np
A=np.array([[1.,-2.],[2.,1.],[-1.,3.]])""",'call':'normalized_coordinate_sketch(A,2)','gold_call':'_oracle_normalized_coordinate_sketch(A,2)'},
        {'setup':"""import numpy as np
A=np.eye(2); index=0
def run_model():
    try: normalized_coordinate_sketch(A,index); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_normalized_coordinate_sketch(A,index); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[2.,0.]]); index=2
def run_model():
    try: normalized_coordinate_sketch(A,index); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_normalized_coordinate_sketch(A,index); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); index=1.0
def run_model():
    try: normalized_coordinate_sketch(A,index); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_normalized_coordinate_sketch(A,index); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
