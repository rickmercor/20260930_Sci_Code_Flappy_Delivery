"""
Build A = Q diag(s) Q^T from a thin QR factor of a Gaussian drawn by default_rng(seed). The eigenvalues s must contain both a positive and a negative entry. Require n >= 2.

A mixed-sign spectrum is the regime in which an interpolatory intersection core can become nearly singular. This step manufactures a reproducible symmetric indefinite instance so later sketches see a known deterministic matrix.

Returns
-------
ndarray of shape (n, n): symmetric matrix A with eigenvalues s
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_indefinite_symmetric_matrix(n: int, s: np.ndarray, seed: int) -> np.ndarray:
    """Build A of shape (n, n) equal to Q diag(s) Q^T.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    s : np.ndarray
        Eigenvalues of length n, with at least one positive and one negative entry.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    A : np.ndarray
        Symmetric array of shape (n, n).

    Raises
    ------
    ValueError
        If n is not an integer, if n < 2, if s does not have shape (n,),
        if s has a non-finite entry, or if s does not contain both a
        positive and a negative entry.
    """
    return np.zeros((n, n))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_indefinite_symmetric_matrix(n, s, seed):
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 2:
        raise ValueError("require n >= 2")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if not np.all(np.isfinite(s)):
        raise ValueError("s must be finite")
    if not (np.any(s > 0.0) and np.any(s < 0.0)):
        raise ValueError("s must contain both a positive and a negative entry")
    rng = np.random.default_rng(int(seed))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = Q @ np.diag(s) @ Q.T
    return 0.5 * (A + A.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, seed = 12, 7
s = np.array([6.5, 5.2, 4.1, -3.8, -2.9, -1.6, 0.9, 0.55, 0.35, -0.25, 0.18, 0.12])
""",
            "call": "construct_indefinite_symmetric_matrix(n, s, seed)",
            "gold_call": "_oracle_construct_indefinite_symmetric_matrix(n, s, seed)",
        },
        {
            "setup": """import numpy as np
n, seed = 4, 1
s = np.array([3.0, 1.0, -2.0, -0.4])
""",
            "call": "construct_indefinite_symmetric_matrix(n, s, seed)",
            "gold_call": "_oracle_construct_indefinite_symmetric_matrix(n, s, seed)",
        },
        {
            "setup": """import numpy as np
n, seed = 2, 3
s = np.array([2.5, -1.25])
""",
            "call": "construct_indefinite_symmetric_matrix(n, s, seed)",
            "gold_call": "_oracle_construct_indefinite_symmetric_matrix(n, s, seed)",
        },
        {
            "setup": """import numpy as np
s = np.array([4.0, 1.0, 0.2])
def run_model():
    try:
        construct_indefinite_symmetric_matrix(3, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_indefinite_symmetric_matrix(3, s, 0)
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
s = np.array([1.0, -1.0])
def run_model():
    try:
        construct_indefinite_symmetric_matrix(1, s, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_indefinite_symmetric_matrix(1, s, 0)
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
