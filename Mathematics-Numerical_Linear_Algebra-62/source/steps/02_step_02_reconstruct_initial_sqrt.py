"""
Reconstruct the lower-precision initial square-root approximation.

Computes the structured square-root approximation to the original basis. This produces the lower-precision matrix approximation whose defect drives the refinement process.

Returns
-------
np.ndarray, the deterministic binary32 initial square-root approximation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconstruct_initial_sqrt(Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Reconstruct the lower-precision initial square-root approximation.

    Parameters
    ----------
    Q : np.ndarray
        Supplied binary32 Schur-vector factor.
    S : np.ndarray
        Compatible binary32 principal square-root factor.

    Returns
    -------
    X0 : np.ndarray
        Binary32 initial square-root approximation in the original coordinates under the task-wide deterministic arithmetic convention.

    Raises
    ------
    ValueError
        If the factors are empty, non-square, shape-incompatible, or non-finite."""
    return X0

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

def _oracle_reconstruct_initial_sqrt(Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q)
    S = np.asarray(S)
    if Q.ndim != 2 or S.ndim != 2 or Q.shape[0] != Q.shape[1] or S.shape[0] != S.shape[1]:
        raise ValueError("Q and S must be square")
    if Q.shape != S.shape or Q.shape[0] == 0:
        raise ValueError("Q and S must have the same nonempty shape")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(S)):
        raise ValueError("Q and S must be finite")
    Q32 = np.asarray(Q, dtype=np.float32)
    S32 = np.asarray(S, dtype=np.float32)
    QS = _dot32_left(Q32, S32)
    return _dot32_left(QS, Q32.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; Q=np.array([[0.6966455578804016, 0.4486113488674164, 0.5052264332771301, -0.24120382964611053], [0.21158075332641602, -0.4446275234222412, 0.4567680060863495, 0.7408797144889832], [-0.6356939673423767, 0.5600026845932007, 0.48396551609039307, 0.21924364566802979], [-0.2565384805202484, -0.5361445546150208, 0.5494421124458313, -0.5872394442558289]],dtype=np.float32); S=np.array([[32.0, -5.6349077224731445, 2.8753058910369873, -46.20516586303711], [0.0, 8.000008583068848, -4.598959445953369, 68.10249328613281], [0.0, 0.0, 1.999853491783142, -22.72388458251953], [0.0, 0.0, 0.0, 0.5004583597183228]],dtype=np.float32)""",
            "call": """reconstruct_initial_sqrt(Q,S).tolist()""",
            "gold_call": """_oracle_reconstruct_initial_sqrt(Q,S).tolist()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); S=np.array([[2.,1.],[0.,3.]],dtype=np.float32)""",
            "call": """reconstruct_initial_sqrt(Q,S).tolist()""",
            "gold_call": """_oracle_reconstruct_initial_sqrt(Q,S).tolist()""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.,-1.],[1.,0.]],dtype=np.float32); S=np.diag(np.array([2.,4.],dtype=np.float32))""",
            "call": """reconstruct_initial_sqrt(Q,S).tolist()""",
            "gold_call": """_oracle_reconstruct_initial_sqrt(Q,S).tolist()""",
        },
        {
            "setup": """import numpy as np; Q=np.empty((0,0),dtype=np.float32); S=np.empty((0,0),dtype=np.float32)
def run_model():
    try:
        reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.ones((2,3),dtype=np.float32); S=np.eye(2,dtype=np.float32)
def run_model():
    try:
        reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); S=np.eye(3,dtype=np.float32)
def run_model():
    try:
        reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); S=np.array([[1.0,np.inf],[0.0,1.0]],dtype=np.float32)
def run_model():
    try:
        reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_initial_sqrt(Q, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
