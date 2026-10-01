"""
Return the packed sketched action of w against the columns of U: the coefficient vector, the leftover, and the leftover continuation scale measured by Omega. Pack (k, h, leftover, scale). Require d >= k >= 1.

Each inner step of randomized Gram-Schmidt measures coefficients and residual size through Omega. A Euclidean projection onto range(U), or a continuation scale taken from the Euclidean leftover length, is a different recurrence.

Returns
-------
1d ndarray: packed (k, h, leftover, scale)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sketched_action_pack(
    Omega: np.ndarray, U: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Pack sketched coefficients, leftover, and continuation scale.

    Parameters
    ----------
    Omega : np.ndarray
        Sketch of shape (d, n) with d >= k.
    U : np.ndarray
        Accepted basis, shape (n, k), k >= 1.
    w : np.ndarray
        New action, length n.

    Returns
    -------
    pack : np.ndarray
        Concatenation of (k, h, leftover, scale).

    Raises
    ------
    ValueError
        If Omega or U is not 2-dimensional, if U has no columns, if
        Omega does not have shape (d, n) with d >= k, if w does not
        have length n, if an input is nonfinite, or if the sketched
        leftover has zero scale (Arnoldi breakdown).
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sketched_action_pack(Omega, U, w):
    Omega = np.asarray(Omega, dtype=float)
    U = np.asarray(U, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if Omega.ndim != 2:
        raise ValueError("Omega must be 2-dimensional")
    if U.ndim != 2:
        raise ValueError("U must be 2-dimensional")
    n = U.shape[0]
    k = U.shape[1]
    if k < 1:
        raise ValueError("U must have at least one column")
    if Omega.shape[1] != n:
        raise ValueError("Omega must have shape (d, n)")
    d = Omega.shape[0]
    if d < k:
        raise ValueError("require d >= k")
    if w.shape != (n,):
        raise ValueError("w must have length n")
    if not np.all(np.isfinite(Omega)) or not np.all(np.isfinite(U)) or not np.all(np.isfinite(w)):
        raise ValueError("inputs must be finite")
    hk, *_ = np.linalg.lstsq(Omega @ U, Omega @ w, rcond=None)
    leftover = w - U @ hk
    scale = float(np.linalg.norm(Omega @ leftover))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("Arnoldi breakdown")
    return np.concatenate(([float(k)], hk.ravel(), leftover, [scale]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
Omega = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
U = np.array([[1.0], [0.0], [0.0]])
w = np.array([1.0, 2.0, 3.0])
""",
            "call": "sketched_action_pack(Omega, U, w)",
            "gold_call": "_oracle_sketched_action_pack(Omega, U, w)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(4)
Omega = rng.standard_normal((4, 6))
U = rng.standard_normal((6, 2))
w = rng.standard_normal(6)
""",
            "call": "sketched_action_pack(Omega, U, w)",
            "gold_call": "_oracle_sketched_action_pack(Omega, U, w)",
        },
        {
            "setup": """import numpy as np
Omega = np.ones((2, 3))
U = np.eye(3, 3)
w = np.ones(3)
def run_model():
    try:
        sketched_action_pack(Omega, U, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sketched_action_pack(Omega, U, w)
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
Omega = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
U = np.array([[1.0], [0.0], [0.0]])
w = np.array([0.0, 0.0, 1.0])
def run_model():
    try:
        sketched_action_pack(Omega, U, w)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sketched_action_pack(Omega, U, w)
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
Omega = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
w = np.array([0.5, -0.25, 2.0])
""",
            "call": "sketched_action_pack(Omega, U, w)",
            "gold_call": "_oracle_sketched_action_pack(Omega, U, w)",
        },
    ]
