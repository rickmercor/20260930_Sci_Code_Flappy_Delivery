"""
Return C = A[:, I] for a 0-based index set I of distinct columns. Require 1 <= r < n and 0 <= I[j] < n for every index. Invalid shapes, repeated indices, and out-of-range indices raise ValueError.

The column subset is treated as given. Selecting it once, with 0-based indices, makes the later intersection and the sketched core deterministic.

Returns
-------
ndarray of shape (n, r): selected columns of A
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def extract_column_subset(A: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return the columns of A listed in indices.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric matrix of shape (n, n).
    indices : np.ndarray
        Distinct 0-based column indices, length r with 1 <= r < n.

    Returns
    -------
    C : np.ndarray
        Array of shape (n, r).

    Raises
    ------
    ValueError
        If A is not a square 2D array, if n < 2, if r is not in
        1 <= r < n, if indices are not distinct, or if any index lies
        outside 0, ..., n-1.
    """
    return np.zeros((1, 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_extract_column_subset(A, indices):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    n = A.shape[0]
    if n < 2:
        raise ValueError("require n >= 2")
    indices = np.asarray(indices, dtype=int).reshape(-1)
    r = indices.size
    if r < 1 or r >= n:
        raise ValueError("require 1 <= r < n")
    if np.unique(indices).size != r:
        raise ValueError("indices must be distinct")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("indices must lie in 0, ..., n-1")
    return A[:, indices]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, seed = 12, 7
s = np.array([6.5, 5.2, 4.1, -3.8, -2.9, -1.6, 0.9, 0.55, 0.35, -0.25, 0.18, 0.12])
rng = np.random.default_rng(seed)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = Q @ np.diag(s) @ Q.T
A = 0.5 * (A + A.T)
indices = np.array([5, 8, 9])
""",
            "call": "extract_column_subset(A, indices)",
            "gold_call": "_oracle_extract_column_subset(A, indices)",
        },
        {
            "setup": """import numpy as np
A = np.arange(16, dtype=float).reshape(4, 4)
A = 0.5 * (A + A.T)
indices = np.array([0, 2])
""",
            "call": "extract_column_subset(A, indices)",
            "gold_call": "_oracle_extract_column_subset(A, indices)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
indices = np.array([1])
""",
            "call": "extract_column_subset(A, indices)",
            "gold_call": "_oracle_extract_column_subset(A, indices)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
indices = np.array([0, 0])
def run_model():
    try:
        extract_column_subset(A, indices)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_column_subset(A, indices)
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
A = np.eye(3)
indices = np.array([0, 3])
def run_model():
    try:
        extract_column_subset(A, indices)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_column_subset(A, indices)
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
