"""
Draw one Gaussian sketch S of shape (n_sketch, m) from default_rng(seed) and return Y = S A. The sketch is computed once and reused for CUR index selection; no second embedding is drawn.

A small Gaussian embedding compresses the row dimension while retaining information used for later CUR index selection. The same sketch is reused throughout the phase.

Returns
-------
ndarray of shape (n_sketch, n): the sketched matrix Y
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def form_sketched_range(A: np.ndarray, n_sketch: int, seed: int) -> np.ndarray:
    """Compute Y = S A with S ~ randn(n_sketch, m).

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    n_sketch : int
        Sketch dimension, 1 <= n_sketch.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    Y : np.ndarray
        Sketched matrix of shape (n_sketch, n).

    Raises
    ------
    ValueError
        If A is not a 2D array with at least one row and one column, or if
        n_sketch is not an integer >= 1.
    """
    return np.zeros((n_sketch, A.shape[1]))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_form_sketched_range(A: np.ndarray, n_sketch: int, seed: int) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must be a 2D array with at least one row and column")
    if not isinstance(n_sketch, (int, np.integer)) or int(n_sketch) < 1:
        raise ValueError("n_sketch must be an integer >= 1")
    rng = np.random.default_rng(int(seed))
    S = rng.standard_normal((int(n_sketch), A.shape[0]))
    return S @ A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
U, _ = np.linalg.qr(rng.standard_normal((12, 8)), mode='reduced')
V, _ = np.linalg.qr(rng.standard_normal((8, 8)), mode='reduced')
s = np.array([10.0, 9.0, 8.0, 0.45, 0.3, 0.22, 0.15, 0.1])
A = U @ np.diag(s) @ V.T
n_sketch, seed = 3, 11
""",
            "call": "form_sketched_range(A, n_sketch, seed)",
            "gold_call": "_oracle_form_sketched_range(A, n_sketch, seed)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
n_sketch, seed = 1, 0
""",
            "call": "form_sketched_range(A, n_sketch, seed)",
            "gold_call": "_oracle_form_sketched_range(A, n_sketch, seed)",
        },
        {
            "setup": """import numpy as np
A = np.eye(4)
n_sketch, seed = 4, 2
""",
            "call": "form_sketched_range(A, n_sketch, seed)",
            "gold_call": "_oracle_form_sketched_range(A, n_sketch, seed)",
        },
        {
            "setup": """import numpy as np
A = np.ones((3, 2))
def run_model():
    try:
        form_sketched_range(A, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_form_sketched_range(A, 0, 1)
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
A = np.array([1.0, 2.0, 3.0])
def run_model():
    try:
        form_sketched_range(A, 2, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_form_sketched_range(A, 2, 1)
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
