"""
Evaluate the square-root residual and normalized backward error in working precision.

Measures the defect of the initial square-root approximation at higher precision. The resulting residual and backward-error measures quantify how far the approximation is from satisfying the defining matrix equation.

Returns
-------
tuple[np.ndarray, float], the binary64 residual and normalized Frobenius backward error
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def working_residual_backward_error(A: np.ndarray, X0: np.ndarray) -> tuple[np.ndarray, float]:
    """Evaluate the square-root residual and normalized backward error in working precision.

    Parameters
    ----------
    A : np.ndarray
        Original finite square matrix.
    X0 : np.ndarray
        Compatible initial square-root approximation.

    Returns
    -------
    R0 : np.ndarray
        Binary64 square-root residual.
    eta0 : float
        Binary64 normalized Frobenius backward error.

    Raises
    ------
    ValueError
        If the matrices are empty, non-square, shape-incompatible, non-finite, or `A` has zero Frobenius norm."""
    return R0, eta0

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

def _oracle_working_residual_backward_error(A: np.ndarray, X0: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.asarray(A)
    X0 = np.asarray(X0)
    if A.ndim != 2 or X0.ndim != 2 or A.shape[0] != A.shape[1] or X0.shape[0] != X0.shape[1]:
        raise ValueError("A and X0 must be square")
    if A.shape != X0.shape or A.shape[0] == 0:
        raise ValueError("A and X0 must have the same nonempty shape")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(X0)):
        raise ValueError("A and X0 must be finite")
    A64 = np.asarray(A, dtype=np.float64)
    X64 = np.asarray(X0, dtype=np.float64)
    XX = _dot64_left(X64, X64)
    R0 = np.empty_like(A64)
    for i in range(A64.shape[0]):
        for j in range(A64.shape[1]):
            R0[i, j] = np.float64(A64[i, j] - XX[i, j])
    anorm = _fro64_row_major(A64)
    if anorm == 0.0:
        raise ValueError("A must have nonzero Frobenius norm")
    eta0 = _fro64_row_major(R0) / anorm
    return R0, float(eta0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; A=np.array([[734.3125, -562.1875, -729.6875, 557.8125], [320.3125, -448.1875, -319.6875, 447.8125], [-809.6875, 917.8125, 814.3125, -922.1875], [-199.6875, 7.8125, 200.3125, -8.1875]],dtype=np.float64); X0=np.array([[19.052352905273438, -4.507687568664551, -16.802370071411133, 2.757863998413086], [16.463993072509766, -31.918928146362305, -16.213943481445312, 32.16926193237305], [-25.94766616821289, 32.49231719970703, 28.19763946533203, -34.24215316772461], [3.4639892578125, -26.91893768310547, -3.213947296142578, 27.169265747070312]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(working_residual_backward_error(A,X0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_working_residual_backward_error(A,X0))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[4.]],dtype=float); X0=np.array([[2.]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(working_residual_backward_error(A,X0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_working_residual_backward_error(A,X0))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[4.,1.],[0.,9.]],dtype=float); X0=np.array([[2.,0.2],[0.,3.]],dtype=np.float32)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(working_residual_backward_error(A,X0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_working_residual_backward_error(A,X0))""",
        },
        {
            "setup": """import numpy as np; A=np.empty((0,0),dtype=np.float64); X0=np.empty((0,0),dtype=np.float32)
def run_model():
    try:
        working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.ones((2,3),dtype=np.float64); X0=np.ones((2,3),dtype=np.float32)
def run_model():
    try:
        working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); X0=np.eye(3,dtype=np.float32)
def run_model():
    try:
        working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[1.0,np.nan],[0.0,1.0]],dtype=np.float64); X0=np.eye(2,dtype=np.float32)
def run_model():
    try:
        working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.zeros((2,2),dtype=np.float64); X0=np.zeros((2,2),dtype=np.float32)
def run_model():
    try:
        working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_working_residual_backward_error(A, X0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
