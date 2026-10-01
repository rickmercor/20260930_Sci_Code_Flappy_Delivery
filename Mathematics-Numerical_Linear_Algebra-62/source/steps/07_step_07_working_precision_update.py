"""
Apply a correction in binary64 and evaluate the next residual.

Mixed-precision refinement separates the precision used to obtain a correction from the precision used to update the iterate and recompute the nonlinear residual. This separation makes cancellation, accumulation order, and very small corrections scientifically relevant to the next refinement state.

Returns
-------
tuple[np.ndarray, np.ndarray, float], binary64 updated iterate, residual, and residual norm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def working_precision_update(A: np.ndarray, Xk: np.ndarray, DeltaXk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a correction in binary64 and evaluate the next residual.

    Parameters
    ----------
    A : np.ndarray
        Original finite square matrix.
    Xk : np.ndarray
        Current square-root approximation.
    DeltaXk : np.ndarray
        Compatible correction.

    Returns
    -------
    Xnext : np.ndarray
        Binary64 updated approximation.
    Rnext : np.ndarray
        Binary64 residual of the updated approximation.
    residual_norm : float
        Binary64 Frobenius norm of `Rnext`.

    Raises
    ------
    ValueError
        If inputs are empty, non-square, shape-incompatible, or non-finite."""
    return Xnext, Rnext, residual_norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _dot64_left(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError('incompatible matrix product')
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

def _oracle_working_precision_update(A: np.ndarray, Xk: np.ndarray, DeltaXk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    A = np.asarray(A)
    Xk = np.asarray(Xk)
    DeltaXk = np.asarray(DeltaXk)
    for M in (A, Xk, DeltaXk):
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
            raise ValueError('inputs must be nonempty square matrices')
        if not np.all(np.isfinite(M)):
            raise ValueError('inputs must be finite')
    if not A.shape == Xk.shape == DeltaXk.shape:
        raise ValueError('shape mismatch')
    A64 = np.asarray(A, dtype=np.float64)
    X64 = np.asarray(Xk, dtype=np.float64)
    D64 = np.asarray(DeltaXk, dtype=np.float64)
    Xnext = np.empty_like(X64)
    for i in range(X64.shape[0]):
        for j in range(X64.shape[1]):
            Xnext[i, j] = np.float64(np.float64(X64[i, j]) + np.float64(D64[i, j]))
    P = _dot64_left(Xnext, Xnext)
    R = np.empty_like(A64)
    for i in range(A64.shape[0]):
        for j in range(A64.shape[1]):
            R[i, j] = np.float64(np.float64(A64[i, j]) - np.float64(P[i, j]))
    return (Xnext, R, _fro64_row_major(R))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; A=np.array([[5.]],dtype=np.float64); X=np.array([[2.]],dtype=np.float32); D=np.array([[0.25]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[4.,1.],[0.,9.]],dtype=np.float64); X=np.array([[2.,0.2],[0.,3.]],dtype=np.float32); D=np.array([[1e-7,-2e-7],[0.,3e-7]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[1e16]],dtype=np.float64); X=np.array([[1e8]],dtype=np.float64); D=np.array([[1.]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[-20402.46477000264, -56644.49454583192, -7029.395807881041], [-93112.10513241563, 76390.03017134029, 94103.22538401232], [-21629.926117275478, -1079.3677922462518, -52867.88860840608]],dtype=np.float64); X=np.array([[81.36517283755903, -93.07958832331812, -324.0546994929779], [-297.6930933191524, 223.4098848848529, -14.104408647251542], [168.89578199331584, 87.25810478439202, -55.6239322453368]],dtype=np.float64); D=np.array([[-1.3531133845390286e-05, 3.5879077131539816e-06, -5.216237241256749e-06], [-1.3576455785369035e-05, 1.263542435481213e-05, 1.360668920824537e-05], [-6.81030314808595e-06, 5.076884917798452e-06, 2.2488560716737993e-05]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z: (z[0].tolist(), z[1].tolist(), float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[1.0000000000005,2e-7,-3e-7],[-4e-7,0.999999999999,5e-7],[6e-7,-7e-7,1.0000000000002]],dtype=np.float64); X=np.array([[1.0,1e-7,-2e-7],[-2e-7,1.0,3e-7],[4e-7,-5e-7,1.0]],dtype=np.float64); D=np.array([[2e-8,-3e-8,1e-8],[4e-8,-2e-8,5e-8],[-1e-8,2e-8,-4e-8]],dtype=np.float32)""",
            "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[9.0,-8.0,3.0],[7.0,5.0,-6.0],[-4.0,2.0,8.0]],dtype=np.float64); X=np.array([[2.7,-1.1,0.6],[0.9,1.8,-0.7],[-0.5,0.4,2.2]],dtype=np.float64); D=np.array([[0.03125,-0.015625,0.0078125],[-0.0234375,0.01953125,-0.01171875],[0.005859375,-0.009765625,0.02734375]],dtype=np.float32)""",
            "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[4.0,0.0],[0.0,9.0]],dtype=np.float64); X=np.array([[2.0,-0.0],[0.0,3.0]],dtype=np.float64); D=np.array([[0.0,1e-30],[-1e-30,0.0]],dtype=np.float32)""",
            "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""",
            "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))""",
        },
        {"setup": """import numpy as np; A=np.array([[1e16,1.,-1.],[1.,1e-16,2.],[-1.,2.,3.]],dtype=np.float64); X=np.array([[1e8,1.,-1.],[0.,1e-8,2.],[0.,0.,1.7320508075688772]],dtype=np.float64); D=np.array([[1.,-2**-52,2**-51],[2**-50,-2**-52,-2**-49],[2**-48,-2**-47,2**-52]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""", "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))"""},
        {"setup": """import numpy as np; A=np.arange(1,26,dtype=np.float64).reshape(5,5); X=np.array([[1e-8,1e8,-3.,4.,-5.],[2.,-1e-8,6.,-7.,8.],[-9.,10.,1e4,11.,-12.],[13.,-14.,15.,-1e4,16.],[-17.,18.,-19.,20.,0.125]],dtype=np.float64); D=np.array([[2**-52,-2**-20,2**-30,-2**-40,2**-50],[-2**-49,2**-48,-2**-47,2**-46,-2**-45],[2**-44,-2**-43,2**-42,-2**-41,2**-40],[-2**-39,2**-38,-2**-37,2**-36,-2**-35],[2**-34,-2**-33,2**-32,-2**-31,2**-30]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""", "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))"""},
        {"setup": """import numpy as np; X=np.array([[2.,3.,-4.],[5.,-6.,7.],[-8.,9.,10.]],dtype=np.float64); D=np.array([[2**-51,-2**-50,2**-49],[-2**-48,2**-47,-2**-46],[2**-45,-2**-44,2**-43]],dtype=np.float64); A=X@X""", "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""", "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))"""},
        {"setup": """import numpy as np; A=np.array([[0.,0.],[0.,0.]],dtype=np.float64); X=np.array([[2**150,2**-150],[-2**149,2**-149]],dtype=np.float64); D=np.array([[-2**98,2**-98],[2**97,-2**-97]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(working_precision_update(A,X,D))""", "gold_call": """(lambda z:(z[0].tolist(),z[1].tolist(),float(z[2])))(_oracle_working_precision_update(A,X,D))"""},
        {
            "setup": """import numpy as np; A=np.empty((0,0),dtype=np.float64); Xk=np.empty((0,0),dtype=np.float64); DeltaXk=np.empty((0,0),dtype=np.float32)
def run_model():
    try:
        working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.ones((2,3),dtype=np.float64); Xk=np.ones((2,3),dtype=np.float64); DeltaXk=np.ones((2,3),dtype=np.float32)
def run_model():
    try:
        working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); Xk=np.eye(3,dtype=np.float64); DeltaXk=np.eye(2,dtype=np.float32)
def run_model():
    try:
        working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); Xk=np.array([[1.0,0.0],[0.0,np.nan]],dtype=np.float64); DeltaXk=np.zeros((2,2),dtype=np.float32)
def run_model():
    try:
        working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_precision_update(A, Xk, DeltaXk)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
