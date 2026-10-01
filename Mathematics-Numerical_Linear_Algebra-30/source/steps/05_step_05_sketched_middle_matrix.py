"""
From A, the column subset C and the sketch X, return the r-by-r matrix M of the cited source's reconstruction C M C^T (Source 1, Algorithm 3.1). Require t > r and t < n. Return M symmetric to rounding.

The cited source determines a middle matrix from the column subset and one sketch. Which sketched matrices enter that fit, and how they are reduced to an r-by-r core, is the source's design choice.

Returns
-------
2D ndarray (r, r): sketched middle matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sketched_middle_matrix(A: np.ndarray, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return the r-by-r matrix M of the cited source's reconstruction C M C^T.

    The cited source (Source 1, Algorithm 3.1) determines a square middle
    matrix M from the column subset C and one sketch X. Return that
    matrix, of shape (r, r), symmetric to rounding.

    Parameters
    ----------
    A : np.ndarray
        Real symmetric matrix of shape (n, n).
    C : np.ndarray
        Column subset of shape (n, r) with 1 <= r < n.
    X : np.ndarray
        Sketch of shape (t, n) with r < t < n.

    Returns
    -------
    M : np.ndarray
        Array of shape (r, r).

    Raises
    ------
    ValueError
        If A is not a square 2D array, if C does not have shape (n, r) with
        1 <= r < n, if X does not have shape (t, n) with r < t < n, or if
        the sketched column block is rank deficient.
    """
    return np.zeros((1, 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sketched_middle_matrix(A, C, X):
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    n = A.shape[0]
    if C.ndim != 2 or C.shape[0] != n:
        raise ValueError("C must have shape (n, r)")
    r = C.shape[1]
    if r < 1 or r >= n:
        raise ValueError("require 1 <= r < n")
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError("X must have shape (t, n)")
    t = X.shape[0]
    if t <= r:
        raise ValueError("require t > r")
    if t >= n:
        raise ValueError("require t < n")
    XC = X @ C
    Msk = X @ A @ X.T
    Q, R = np.linalg.qr(XC, mode="reduced")
    if np.min(np.abs(np.diag(R))) < 1e-14:
        raise ValueError("XC is rank deficient")
    core = Q.T @ Msk @ Q
    Y = np.linalg.solve(R, core)
    M = np.linalg.solve(R, Y.T).T
    return 0.5 * (M + M.T)

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
""",
            "call": "sketched_middle_matrix(A, C, X)",
            "gold_call": "_oracle_sketched_middle_matrix(A, C, X)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, 0.0], [0.5, -1.0, 0.2], [0.0, 0.2, 0.8]])
C = A[:, :1]
X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
""",
            "call": "sketched_middle_matrix(A, C, X)",
            "gold_call": "_oracle_sketched_middle_matrix(A, C, X)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
A = rng.standard_normal((5, 5))
A = 0.5 * (A + A.T)
C = A[:, [0, 2]]
X = rng.standard_normal((3, 5))
""",
            "call": "sketched_middle_matrix(A, C, X)",
            "gold_call": "_oracle_sketched_middle_matrix(A, C, X)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(2)
A = rng.standard_normal((5, 5))
A = 0.5 * (A + A.T)
C = A[:, [0, 3]]
X = rng.standard_normal((3, 5))
""",
            "call": "sketched_middle_matrix(A, C, X)",
            "gold_call": "_oracle_sketched_middle_matrix(A, C, X)",
        },
        {
            "setup": """import numpy as np
A = np.eye(4)
C = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
X = np.ones((3, 4))
def run_model():
    try:
        sketched_middle_matrix(A, C, X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sketched_middle_matrix(A, C, X)
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
A = np.eye(4)
C = A[:, :2]
X = np.ones((2, 4))
def run_model():
    try:
        sketched_middle_matrix(A, C, X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sketched_middle_matrix(A, C, X)
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
