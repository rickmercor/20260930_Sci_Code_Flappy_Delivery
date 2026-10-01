"""
Transform the working residual to the supplied Schur coordinates.

Computes the higher-precision defect in the same structured coordinates used by the square-root factor. This provides the coordinate-space residual needed to formulate the first refinement correction.

Returns
-------
tuple[np.ndarray, float], binary32 transformed residual and binary64 Frobenius norm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def schur_residual_transform(Q: np.ndarray, R0: np.ndarray) -> tuple[np.ndarray, float]:
    """Transform the working residual to the supplied Schur coordinates.

    Parameters
    ----------
    Q : np.ndarray
        Supplied Schur-vector factor.
    R0 : np.ndarray
        Compatible binary64 working residual.

    Returns
    -------
    Rhat : np.ndarray
        Binary32 Schur-coordinate residual under the task-wide deterministic mixed-precision convention.
    rhat_norm : float
        Binary64 Frobenius norm of `Rhat`.

    Raises
    ------
    ValueError
        If the inputs are empty, non-square, shape-incompatible, or non-finite."""
    return Rhat, rhat_norm

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

def _oracle_schur_residual_transform(Q: np.ndarray, R0: np.ndarray) -> tuple[np.ndarray, float]:
    Q = np.asarray(Q)
    R0 = np.asarray(R0)
    if Q.ndim != 2 or R0.ndim != 2 or Q.shape[0] != Q.shape[1] or R0.shape[0] != R0.shape[1]:
        raise ValueError("Q and R0 must be square")
    if Q.shape != R0.shape or Q.shape[0] == 0:
        raise ValueError("Q and R0 must have the same nonempty shape")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(R0)):
        raise ValueError("Q and R0 must be finite")
    Q64 = np.asarray(Q, dtype=np.float64)
    R64 = np.asarray(R0, dtype=np.float64)
    left = _dot64_left(Q64.T, R64)
    full = _dot64_left(left, Q64)
    Rhat = np.asarray(full, dtype=np.float32)
    return Rhat, _fro64_row_major(Rhat)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; Q=np.array([[0.6966455578804016, 0.4486113488674164, 0.5052264332771301, -0.24120382964611053], [0.21158075332641602, -0.4446275234222412, 0.4567680060863495, 0.7408797144889832], [-0.6356939673423767, 0.5600026845932007, 0.48396551609039307, 0.21924364566802979], [-0.2565384805202484, -0.5361445546150208, 0.5494421124458313, -0.5872394442558289]],dtype=np.float32); R0=np.array([[-0.0006150363078631926, 0.0007059133731672773, 0.0005957527537248097, -0.0007854900577513035], [-0.0002648560985107906, 1.4817182091064751e-05, 0.00015251154400175437, -0.00015609611000400037], [0.0006939346785657108, -0.0005672639272233937, -0.000541794863238465, 0.0008202870012610219], [8.15210078144446e-05, -0.00018981749599333853, -8.943807915784419e-05, 0.00016918159963097423]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(schur_residual_transform(Q,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_schur_residual_transform(Q,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); R0=np.array([[1.,2.],[3.,4.]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(schur_residual_transform(Q,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_schur_residual_transform(Q,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.,-1.],[1.,0.]],dtype=np.float32); R0=np.array([[1.,2.],[3.,4.]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(schur_residual_transform(Q,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_schur_residual_transform(Q,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[-0.7633581757545471, 0.6285871267318726, 0.14887088537216187], [-0.3773997128009796, -0.24693620204925537, -0.892520010471344], [-0.5242649912834167, -0.7374962568283081, 0.425729364156723]],dtype=np.float32); R0=np.array([[1040.896884021924, -293.82333175982217, 1754.7522823710347], [-227.12202229291185, -1123.4477961256648, 698.9024755110498], [-412.45741277712807, -1523.0907965794247, 225.7960668998636]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(schur_residual_transform(Q,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_schur_residual_transform(Q,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.empty((0,0),dtype=np.float32); R0=np.empty((0,0),dtype=np.float64)
def run_model():
    try:
        schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.ones((2,3),dtype=np.float32); R0=np.ones((2,3),dtype=np.float64)
def run_model():
    try:
        schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); R0=np.eye(3,dtype=np.float64)
def run_model():
    try:
        schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[1.0,np.nan],[0.0,1.0]],dtype=np.float32); R0=np.eye(2,dtype=np.float64)
def run_model():
    try:
        schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_schur_residual_transform(Q, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
