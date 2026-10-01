"""
Form the unoccupied projector of the mapped initial-state density.

P_t is shape (n, n), nonempty, finite, and symmetric. Otherwise raise
ValueError.

Return the unoccupied projector Q = I − P_t of shape (n, n).

Returns
-------
return Q
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def frame_matrix(
    P_t: np.ndarray,
) -> np.ndarray:
    """
    Return the unoccupied projector of the mapped initial density.

    Q is I minus P_t.

    Parameters
    ----------
    P_t : np.ndarray
        Mapped initial-state density from step 4, shape (n, n).

    Returns
    -------
    np.ndarray
        Unoccupied projector, shape (n, n).

    Raises
    ------
    ValueError
        If P_t is not a nonempty finite square matrix.
    """
    return Q

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_frame_matrix(P_t: np.ndarray) -> np.ndarray:
    P_t = np.asarray(P_t, dtype=float)
    if P_t.ndim != 2 or P_t.shape[0] != P_t.shape[1] or P_t.shape[0] == 0:
        raise ValueError("P_t must be a nonempty square matrix")
    if not np.all(np.isfinite(P_t)):
        raise ValueError("P_t must be finite")
    if np.max(np.abs(P_t - P_t.T)) > 1e-10:
        raise ValueError("P_t must be symmetric")
    P_t = 0.5 * (P_t + P_t.T)
    return np.eye(P_t.shape[0]) - P_t

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
P_t = np.diag([1.0, 1.0, 0.0, 0.0])

def run_model():
    return frame_matrix(P_t.copy())

def run_gold():
    return _oracle_frame_matrix(P_t.copy())
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_t = np.diag([1.0, 0.0])

def run_model():
    pred = np.asarray(frame_matrix(P_t.copy()), dtype=float)
    return np.concatenate([
        pred.ravel(),
        [np.max(np.abs(pred - P_t)), np.max(np.abs(pred - np.eye(2)))],
    ])

def run_gold():
    Q = np.asarray(_oracle_frame_matrix(P_t.copy()), dtype=float)
    return np.concatenate([
        Q.ravel(),
        [np.max(np.abs(Q - P_t)), np.max(np.abs(Q - np.eye(2)))],
    ])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_t = np.array([[0.8, 0.1], [0.1, 0.3]], dtype=float)

def run_model():
    pred = np.asarray(frame_matrix(P_t.copy()), dtype=float)
    return np.array([pred[0, 0], pred[1, 1], pred[0, 1], np.trace(pred + P_t)])

def run_gold():
    Q = _oracle_frame_matrix(P_t.copy())
    return np.array([Q[0, 0], Q[1, 1], Q[0, 1], np.trace(Q + P_t)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_t = np.array([[1.0, np.nan], [np.nan, 0.0]])

def run_model():
    try:
        frame_matrix(P_t.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_frame_matrix(P_t.copy())
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
