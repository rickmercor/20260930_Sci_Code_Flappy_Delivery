"""
Build a clustered-spectrum least-squares instance [A | b]. Draw thin QR factors of Gaussians from numpy Generator default_rng(seed), set A = U diag(s) V^T with prescribed positive singular values s, then draw b from the same generator. Return an array of shape (m, n+1) whose first n columns are A and whose last column is b.

A controlled singular-value spectrum isolates the effect of dominant spectral components on an iterative least-squares solver. Fixed random generators make the matrix and right-hand side reproducible.

Returns
-------
ndarray of shape (m, n+1): A in the first n columns and b in the last column
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_clustered_ls_data(
    m: int,
    n: int,
    s: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Build [A | b] with prescribed singular values.

    Parameters
    ----------
    m : int
        Row dimension, m >= n >= 1.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape (n,).
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    data : np.ndarray
        Array of shape (m, n+1) whose first n columns are A and whose last
        column is b.

    Raises
    ------
    ValueError
        If m or n is not an integer, if m >= n >= 1 does not hold, if s does not
        have shape (n,), or if s is not positive and finite.
    """
    return np.zeros((m, n + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_clustered_ls_data(
    m: int,
    n: int,
    s: np.ndarray,
    seed: int,
) -> np.ndarray:
    np = __import__("numpy")
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    if m < 1 or n < 1 or m < n:
        raise ValueError("require m >= n >= 1")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(seed))
    U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = U @ np.diag(s) @ V.T
    b = rng.standard_normal(m)
    return np.hstack([A, b[:, None]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, seed = 12, 8, 7
s = np.array([10.0, 9.0, 8.0, 0.45, 0.3, 0.22, 0.15, 0.1])
""",
            "call": "construct_clustered_ls_data(m, n, s, seed)",
            "gold_call": "_oracle_construct_clustered_ls_data(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 2, 2, 0
s = np.array([1.0, 0.1])
""",
            "call": "construct_clustered_ls_data(m, n, s, seed)",
            "gold_call": "_oracle_construct_clustered_ls_data(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 4, 1, 3
s = np.array([2.5])
""",
            "call": "construct_clustered_ls_data(m, n, s, seed)",
            "gold_call": "_oracle_construct_clustered_ls_data(m, n, s, seed)",
        },
        {
            "setup": """import numpy as np
s = np.array([1.0, 0.5])
def run_model():
    try:
        construct_clustered_ls_data(3, 5, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_clustered_ls_data(3, 5, s, 0)
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
s = np.array([1.0, 0.0])
def run_model():
    try:
        construct_clustered_ls_data(3, 2, s, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_clustered_ls_data(3, 2, s, 1)
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
