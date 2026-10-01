"""
Return a new search direction obtained by orthogonalizing d against a window of previous directions. An empty window of shape (n, 0) returns d. Previous columns must be nonzero.

A short window of previous directions is a different algorithm from a full history and from leaving the new direction unorthogonalized.

Returns
-------
ndarray of shape (n,): orthogonalized search direction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def window_orthogonalize(d: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Return the direction after orthogonalizing $d$ against the window.

    Parameters
    ----------
    d : np.ndarray
        New projected direction, length $n \ge 1$.
    window : np.ndarray
        Previous directions, shape $(n, w)$ with $w \ge 0$. An empty window
        ($w = 0$) returns $d$ unchanged.

    Returns
    -------
    p : np.ndarray
        Orthogonalized direction of length $n$.

    Raises
    ------
    ValueError
        If $d$ is not a nonempty finite vector, if window does not have
        shape $(n, w)$, or if a window direction is zero.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_window_orthogonalize(d, window):
    d = np.asarray(d, dtype=float).reshape(-1)
    n = d.size
    if n < 1 or not np.all(np.isfinite(d)):
        raise ValueError("d must be a nonempty finite vector")
    W = np.asarray(window, dtype=float)
    if W.size == 0:
        W = np.zeros((n, 0))
    if W.ndim != 2 or W.shape[0] != n:
        raise ValueError("window must have shape (n, w)")
    p = d.copy()
    for j in range(W.shape[1]):
        pj = W[:, j]
        nrm2 = float(np.dot(pj, pj))
        if nrm2 <= 0.0:
            raise ValueError("window directions must be nonzero")
        p = p - (np.dot(d, pj) / nrm2) * pj
    return p

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
d = np.array([1.0, 2.0, 3.0])
window = np.zeros((3, 0))
""",
            "call": "window_orthogonalize(d, window)",
            "gold_call": "_oracle_window_orthogonalize(d, window)",
        },
        {
            "setup": """import numpy as np
d = np.array([3.0, 4.0])
window = np.array([[1.0], [0.0]])
""",
            "call": "window_orthogonalize(d, window)",
            "gold_call": "_oracle_window_orthogonalize(d, window)",
        },
        {
            "setup": """import numpy as np
d = np.array([1.0, 1.0, 1.0, 1.0])
window = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
""",
            "call": "window_orthogonalize(d, window)",
            "gold_call": "_oracle_window_orthogonalize(d, window)",
        },
        {
            "setup": """import numpy as np
d = np.array([1.0, 0.0])
window = np.array([[1.0], [2.0], [3.0]])
def run_model():
    try:
        window_orthogonalize(d, window)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_window_orthogonalize(d, window)
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
d = np.array([1.0, 2.0])
window = np.array([[0.0], [0.0]])
def run_model():
    try:
        window_orthogonalize(d, window)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_window_orthogonalize(d, window)
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
