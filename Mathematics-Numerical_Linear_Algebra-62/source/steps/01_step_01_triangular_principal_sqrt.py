"""
Compute the deterministic binary32 principal square root of an upper-triangular matrix.

The first stage evaluates the matrix square root in the supplied structured basis at lower precision. The triangular factor selects the principal branch used throughout refinement.

Returns
-------
np.ndarray, deterministic binary32 upper-triangular principal square root
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def triangular_principal_sqrt(T: np.ndarray) -> np.ndarray:
    """Compute the deterministic binary32 principal square root of an upper-triangular matrix.

    Parameters
    ----------
    T : np.ndarray
        Finite nonempty upper-triangular matrix with strictly positive diagonal.

    Returns
    -------
    S : np.ndarray
        Binary32 upper-triangular principal square root under the task-wide deterministic arithmetic convention.

    Raises
    ------
    ValueError
        If `T` is empty, non-square, non-finite, non-triangular, or has a non-positive diagonal."""
    return S

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _dot32_left(A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("incompatible matrix product")
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = np.float32(0.0)
            for k in range(A.shape[1]):
                acc = np.float32(acc + np.float32(A[i, k] * B[k, j]))
            C[i, j] = acc
    return C

def _dot64_left(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("incompatible matrix product")
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = np.float64(0.0)
            for k in range(A.shape[1]):
                acc = np.float64(acc + np.float64(A[i, k] * B[k, j]))
            C[i, j] = acc
    return C

def _fro64_row_major(A):
    A = np.asarray(A, dtype=np.float64)
    acc = np.float64(0.0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            acc = np.float64(acc + np.float64(A[i, j] * A[i, j]))
    return float(np.sqrt(acc))

def _oracle_triangular_principal_sqrt(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] == 0:
        raise ValueError("T must be a nonempty square matrix")
    if not np.all(np.isfinite(T)):
        raise ValueError("T must be finite")
    T = np.asarray(T, dtype=np.float32)
    if np.any(np.tril(T, -1) != np.float32(0.0)):
        raise ValueError("T must be upper triangular")
    if np.any(np.diag(T) <= np.float32(0.0)):
        raise ValueError("T must have strictly positive diagonal entries")

    n = T.shape[0]
    S = np.zeros_like(T, dtype=np.float32)
    for i in range(n):
        S[i, i] = np.float32(np.sqrt(np.float32(T[i, i])))

    for gap in range(1, n):
        for i in range(n - gap):
            j = i + gap
            rhs = np.float32(T[i, j])
            for k in range(i + 1, j):
                rhs = np.float32(rhs - np.float32(S[i, k] * S[k, j]))
            denom = np.float32(S[i, i] + S[j, j])
            if denom == np.float32(0.0):
                raise ValueError("zero triangular square-root denominator")
            S[i, j] = np.float32(rhs / denom)
    return S

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; T=np.array([[1024.0001220703125, -225.39634704589844, 123.67469787597656, -1950.7784423828125], [0.0, 64.0001449584961, -45.988956451416016, 683.4092407226562], [0.0, 0.0, 3.9994139671325684, -56.81679916381836], [0.0, 0.0, 0.0, 0.25045859813690186]],dtype=np.float32)""",
            "call": """triangular_principal_sqrt(T).tolist()""",
            "gold_call": """_oracle_triangular_principal_sqrt(T).tolist()""",
        },
        {
            "setup": """import numpy as np; T=np.array([[4.,6.],[0.,9.]],dtype=np.float32)""",
            "call": """triangular_principal_sqrt(T).tolist()""",
            "gold_call": """_oracle_triangular_principal_sqrt(T).tolist()""",
        },
        {
            "setup": """import numpy as np; T=np.array([[0.25]],dtype=np.float32)""",
            "call": """triangular_principal_sqrt(T).tolist()""",
            "gold_call": """_oracle_triangular_principal_sqrt(T).tolist()""",
        },
        {
            "setup": """import numpy as np; T=np.array([[9.,3.,-2.],[0.,4.,5.],[0.,0.,1.]],dtype=np.float32)""",
            "call": """triangular_principal_sqrt(T).tolist()""",
            "gold_call": """_oracle_triangular_principal_sqrt(T).tolist()""",
        },
        {
            "setup": """import numpy as np; T=np.empty((0,0),dtype=np.float32)
def run_model():
    try:
        triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; T=np.ones((2,3),dtype=np.float32)
def run_model():
    try:
        triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; T=np.array([[1.0,0.0],[0.0,np.nan]],dtype=np.float32)
def run_model():
    try:
        triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; T=np.array([[1.0,0.25],[0.1,2.0]],dtype=np.float32)
def run_model():
    try:
        triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; T=np.array([[1.0,0.0],[0.0,0.0]],dtype=np.float32)
def run_model():
    try:
        triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_triangular_principal_sqrt(T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
