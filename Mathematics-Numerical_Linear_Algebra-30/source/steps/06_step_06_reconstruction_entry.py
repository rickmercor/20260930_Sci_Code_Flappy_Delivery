"""
From the column subset C and the middle matrix M, return the 0-based (row, col) entry of the reconstruction C M C^T. Require C of shape (n, r) with 1 <= r < n, M of shape (r, r), and integer row and col in 0, ..., n-1.

Once the middle matrix is fixed, the reconstruction C M C^T is formed and read at one location. Indices are 0-based, and the location may lie off the selected column set. This step returns that single entry, a float, not the (r, r) core and not a residual-bound scalar.

Returns
-------
float: the (row, col) entry of C M C^T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconstruction_entry(C: np.ndarray, M: np.ndarray, row: int, col: int) -> float:
    """Return (C M C^T)[row, col], 0-based.

    Parameters
    ----------
    C : np.ndarray
        Column subset of shape (n, r).
    M : np.ndarray
        Middle matrix of shape (r, r).
    row, col : int
        0-based indices in 0, ..., n-1.

    Returns
    -------
    value : float
        One entry of C M C^T.

    Raises
    ------
    ValueError
        If C is not 2D, if n <= r or r < 1, if M does not have shape
        (r, r), if row or col is not an integer, or if row or col lies
        outside 0, ..., n-1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconstruction_entry(C, M, row, col):
    C = np.asarray(C, dtype=float)
    M = np.asarray(M, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be 2D")
    n, r = C.shape
    if r < 1 or n <= r:
        raise ValueError("require n > r >= 1")
    if M.shape != (r, r):
        raise ValueError("M must have shape (r, r)")
    if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
        raise ValueError("row and col must be integers")
    row, col = int(row), int(col)
    if row < 0 or col < 0 or row >= n or col >= n:
        raise ValueError("row and col must lie in 0, ..., n-1")
    Ahat = C @ M @ C.T
    return float(Ahat[row, col])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, seed, t = 12, 7, 6
s = np.array([6.5, 5.2, 4.1, -3.8, -2.9, -1.6, 0.9, 0.55, 0.35, -0.25, 0.18, 0.12])
rng = np.random.default_rng(seed)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = Q @ np.diag(s) @ Q.T
A = 0.5 * (A + A.T)
C = A[:, [5, 8, 9]]
X = np.random.default_rng(11).standard_normal((t, n))
Qxc, Rxc = np.linalg.qr(X @ C, mode='reduced')
core = Qxc.T @ (X @ A @ X.T) @ Qxc
M = np.linalg.solve(Rxc, np.linalg.solve(Rxc, core.T).T)
row, col = 2, 0
""",
            "call": "reconstruction_entry(C, M, row, col)",
            "gold_call": "_oracle_reconstruction_entry(C, M, row, col)",
        },
        {
            "setup": """import numpy as np
C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
M = np.array([[2.0, 0.5], [0.5, -1.0]])
row, col = 2, 1
""",
            "call": "reconstruction_entry(C, M, row, col)",
            "gold_call": "_oracle_reconstruction_entry(C, M, row, col)",
        },
        {
            "setup": """import numpy as np
C = np.array([[3.0], [-1.0]])
M = np.array([[0.5]])
row, col = 0, 0
""",
            "call": "reconstruction_entry(C, M, row, col)",
            "gold_call": "_oracle_reconstruction_entry(C, M, row, col)",
        },
        {
            "setup": """import numpy as np
C = np.eye(3)[:, :1]
M = np.array([[1.0]])
def run_model():
    try:
        reconstruction_entry(C, M, 3, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruction_entry(C, M, 3, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
C = np.ones((3, 2))
M = np.eye(3)
def run_model():
    try:
        reconstruction_entry(C, M, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruction_entry(C, M, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
