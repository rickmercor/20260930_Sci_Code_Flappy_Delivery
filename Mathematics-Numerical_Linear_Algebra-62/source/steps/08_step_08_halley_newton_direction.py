"""
Return the binary64 Newton direction for a matrix-power residual.

Newton corrections for matrix power equations are governed by Fréchet derivatives, not scalar derivatives applied entrywise. Requiring both square and cube residual maps tests whether the implementation can construct the correct noncommutative derivative operator while retaining the task-wide vectorization and deterministic solve conventions.

Returns
-------
np.ndarray, deterministic binary64 Newton direction for the selected matrix-power residual
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def halley_newton_direction(A: np.ndarray, X: np.ndarray, power: int = 2) -> np.ndarray:
    """Return the binary64 Newton direction for a matrix-power residual.

    Parameters
    ----------
    A : np.ndarray
        Finite nonempty square target matrix.
    X : np.ndarray
        Finite compatible iterate.
    power : int, default=2
        Supported residual-map power, 2 or 3.

    Returns
    -------
    H : np.ndarray
        Binary64 Newton direction obtained from the Fréchet derivative of the selected noncommutative matrix-power map.

    Raises
    ------
    ValueError
        If the inputs violate the matrix domain, `power` is unsupported, or the required Fréchet derivative is singular."""
    return H

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _halley_gauss_solve64(A, b):
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    vec = b.ndim == 1
    if vec:
        b = b[:, None]
    n = A.shape[0]
    for k in range(n):
        p = k
        best = np.float64(abs(float(A[k, k])))
        for i in range(k + 1, n):
            cand = np.float64(abs(float(A[i, k])))
            if cand > best:
                best = cand
                p = i
        if best == 0.0:
            raise ValueError('singular Frechet derivative')
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            f = np.float64(A[i, k] / A[k, k])
            A[i, k] = 0.0
            for j in range(k + 1, n):
                A[i, j] = np.float64(A[i, j] - np.float64(f * A[k, j]))
            for j in range(b.shape[1]):
                b[i, j] = np.float64(b[i, j] - np.float64(f * b[k, j]))
    z = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0.0:
            raise ValueError('singular Frechet derivative')
        for c in range(b.shape[1]):
            rhs = np.float64(b[i, c])
            for j in range(i + 1, n):
                rhs = np.float64(rhs - np.float64(A[i, j] * z[j, c]))
            z[i, c] = np.float64(rhs / A[i, i])
    return z[:, 0] if vec else z

def _halley_dot64(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            s = np.float64(0.0)
            for k in range(A.shape[1]):
                s = np.float64(s + np.float64(A[i, k] * B[k, j]))
            C[i, j] = s
    return C

def _halley_power64(X, power):
    X = np.asarray(X, dtype=np.float64)
    if power == 2:
        return _halley_dot64(X, X)
    return _halley_dot64(_halley_dot64(X, X), X)

def _halley_frechet64(X, V, power):
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if power == 2:
        return np.asarray(_halley_dot64(X, V) + _halley_dot64(V, X), dtype=np.float64)
    X2 = _halley_dot64(X, X)
    return np.asarray(_halley_dot64(X2, V) + _halley_dot64(_halley_dot64(X, V), X) + _halley_dot64(V, X2), dtype=np.float64)

def _halley_operator_matrix64(X, power):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    N = n * n
    B = np.zeros((N, N), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            E = np.zeros((n, n), dtype=np.float64)
            E[i, j] = 1.0
            C = _halley_frechet64(X, E, power)
            B[:, i + j * n] = np.array([C[a, b] for b in range(n) for a in range(n)], dtype=np.float64)
    return B

def _oracle_halley_newton_direction(A: np.ndarray, X: np.ndarray, power: int=2) -> np.ndarray:
    A = np.asarray(A)
    X = np.asarray(X)
    for M in (A, X):
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError('square inputs required')
    if A.shape != X.shape or A.shape[0] == 0:
        raise ValueError('same nonempty shape required')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(X)):
        raise ValueError('finite inputs required')
    if not isinstance(power, (int, np.integer)) or isinstance(power, (bool, np.bool_)) or int(power) not in (2, 3):
        raise ValueError('power must be 2 or 3')
    power = int(power)
    X64 = np.asarray(X, dtype=np.float64)
    A64 = np.asarray(A, dtype=np.float64)
    R = np.asarray(A64 - _halley_power64(X64, power), dtype=np.float64)
    B = _halley_operator_matrix64(X64, power)
    r = np.array([R[i, j] for j in range(X64.shape[0]) for i in range(X64.shape[0])], dtype=np.float64)
    h = _halley_gauss_solve64(B, r)
    return h.reshape(X64.shape, order='F')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three normal and three explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; A=np.array([[4.,1.],[0.,9.]],dtype=np.float64); X=np.array([[2.,0.2],[0.,3.]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[2.25]],dtype=np.float64); X=np.array([[1.4]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[5.,-1.,0.5],[0.,3.,0.2],[0.,0.,2.]],dtype=np.float64); X=np.array([[2.,0.1,-0.2],[0.,1.7,0.3],[0.,0.,1.3]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[6.0,-1.2,0.7],[2.1,4.5,-0.8],[-1.4,0.9,3.2]],dtype=np.float64); X=np.array([[2.1,-0.35,0.18],[0.42,1.75,-0.27],[-0.31,0.22,1.4]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[8.0,1.5,-2.0,0.3],[-0.7,5.0,1.1,-1.4],[1.2,-0.9,3.5,0.8],[-1.1,0.4,1.7,6.2]],dtype=np.float64); X=np.array([[2.4,0.3,-0.2,0.1],[-0.25,1.9,0.35,-0.15],[0.2,-0.3,1.55,0.28],[-0.18,0.12,-0.22,2.05]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[3.2,-4.1,1.7],[2.6,0.9,-3.3],[-1.8,2.2,4.7]],dtype=np.float64); X=np.array([[0.02,7.0,-1.0],[-0.003,0.021,4.5],[0.002,-0.004,0.022]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[2.0,-1.0,0.5,0.25],[1.5,3.0,-2.0,0.75],[-0.5,1.25,4.0,-1.5],[0.2,-0.8,1.1,2.5]],dtype=np.float64); X=np.array([[1.2,2.0,-1.5,0.7],[-0.4,0.9,1.8,-1.1],[0.3,-0.6,1.4,2.2],[-0.2,0.5,-0.9,1.1]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {
            "setup": """import numpy as np; A=np.array([[1.0,0.2],[0.3,1.4]],dtype=np.float64); X=np.array([[1e-4,1.0],[-0.999999,2e-4]],dtype=np.float64)""",
            "call": """halley_newton_direction(A,X).tolist()""",
            "gold_call": """_oracle_halley_newton_direction(A,X).tolist()""",
        },
        {"setup": """import numpy as np; A=np.array([[2.0,-3.0,4.0],[5.0,-6.0,7.0],[-8.0,9.0,10.0]],dtype=np.float64); X=np.array([[1.0,20.0,-10.0],[0.0,1.1,15.0],[0.0,0.0,0.9]],dtype=np.float64)""", "call": """halley_newton_direction(A,X).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X).tolist()"""},
        {"setup": """import numpy as np; A=np.arange(1,26,dtype=np.float64).reshape(5,5); X=np.array([[2.,3.,-1.,4.,-2.],[-5.,1.5,6.,-3.,2.],[4.,-2.,0.75,5.,-6.],[1.,7.,-4.,2.5,3.],[-3.,2.,8.,-1.,1.25]],dtype=np.float64)""", "call": """halley_newton_direction(A,X).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X).tolist()"""},
        {"setup": """import numpy as np; A=np.array([[1e-12,2e6,-3.],[-4e-6,5e12,6.],[7.,-8e-9,9e-3]],dtype=np.float64); X=np.array([[1e-6,2e3,-3e-2],[-4e-3,5e6,6e-1],[7e-2,-8e-4,9e-2]],dtype=np.float64)""", "call": """halley_newton_direction(A,X).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X).tolist()"""},
        {"setup": """import numpy as np; A=np.array([[3.,1.,4.,1.],[5.,9.,2.,6.],[5.,3.,5.,8.],[9.,7.,9.,3.]],dtype=np.float64); X=np.array([[1.,2.,3.,4.],[-2.,1.,-4.,3.],[3.,-4.,1.,-2.],[-4.,3.,-2.,1.]],dtype=np.float64)""", "call": """halley_newton_direction(A,X).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X).tolist()"""},
        {
            "setup": """import numpy as np; A=np.empty((0,0),dtype=np.float64); X=np.empty((0,0),dtype=np.float64)
def run_model():
    try:
        halley_newton_direction(A,X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_halley_newton_direction(A,X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); X=np.eye(3,dtype=np.float64)
def run_model():
    try:
        halley_newton_direction(A,X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_halley_newton_direction(A,X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); X=np.zeros((2,2),dtype=np.float64)
def run_model():
    try:
        halley_newton_direction(A,X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_halley_newton_direction(A,X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },

        {"setup": """import numpy as np; X=np.array([[1.4,.3],[-.2,1.1]],dtype=np.float64); A=np.array([[3.1,-.4],[.7,1.9]],dtype=np.float64); power=3""", "call": """halley_newton_direction(A,X,power).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X,power).tolist()"""},
        {"setup": """import numpy as np; X=np.array([[1.3,.4,-.2],[-.1,1.1,.5],[.2,-.3,.9]],dtype=np.float64); A=np.array([[2.7,-.6,.8],[.3,1.8,-.4],[-.5,.2,1.4]],dtype=np.float64); power=3""", "call": """halley_newton_direction(A,X,power).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X,power).tolist()"""},
        {"setup": """import numpy as np; X=np.array([[1.2,.7,-.4,.2],[-.3,1.0,.5,-.1],[.2,-.6,.9,.4],[-.1,.3,-.2,1.1]],dtype=np.float64); A=np.arange(1,17,dtype=np.float64).reshape(4,4)/7.; power=3""", "call": """halley_newton_direction(A,X,power).tolist()""", "gold_call": """_oracle_halley_newton_direction(A,X,power).tolist()"""},
        {"setup": """import numpy as np; A=np.eye(2); X=np.eye(2); power=4
def run_model():
    try: halley_newton_direction(A,X,power); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_halley_newton_direction(A,X,power); return 0
    except ValueError: return 1
    except Exception: return 2""", "call": """run_model()""", "gold_call": """run_gold()"""},
    ]
