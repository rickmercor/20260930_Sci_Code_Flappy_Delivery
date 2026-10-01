"""
Map the initial-state AO density with the supplied overlap factors.

Inputs are the initial-state density P0 and the overlap factors G and H.
All three are shape (n, n) and finite. Misaligned, non-square, or
non-finite input raises ValueError.

Map P0 with G, not H. Return the symmetrized product G @ P0 @ G as one
symmetric mapped density P_t of shape (n, n).

Returns
-------
return P_t
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def image_matrix(
    P0: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Return the mapped initial-state AO density.

    Map P0 with G, not H.

    Parameters
    ----------
    P0 : np.ndarray
        Initial-state AO density, shape (n, n).
    G : np.ndarray
        First overlap factor from step 2, shape (n, n).
    H : np.ndarray
        Second overlap factor from step 2, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric mapped density, shape (n, n).

    Raises
    ------
    ValueError
        If the matrices are misaligned, not square, or not finite.
    """
    return P_t

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_image_matrix(P0: np.ndarray, G: np.ndarray, H: np.ndarray) -> np.ndarray:
    P0 = np.asarray(P0, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if P0.ndim != 2 or P0.shape[0] != P0.shape[1] or P0.shape[0] == 0:
        raise ValueError("P0 must be a nonempty square matrix")
    n = P0.shape[0]
    if G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("P0, G, and H must have the same shape")
    if not np.all(np.isfinite(P0)) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("P0, G, and H must be finite")
    if np.max(np.abs(P0 - P0.T)) > 1e-10:
        raise ValueError("P0 must be symmetric")
    if np.max(np.abs(G - G.T)) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.max(np.abs(H - H.T)) > 1e-10:
        raise ValueError("H must be symmetric")
    P0 = 0.5 * (P0 + P0.T)
    G = 0.5 * (G + G.T)
    P_t = G @ P0 @ G
    return 0.5 * (P_t + P_t.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
P0 = np.diag([1.0, 1.0, 0.0, 0.0])
G = np.eye(4)
H = np.eye(4)

def run_model():
    return image_matrix(P0.copy(), G.copy(), H.copy())

def run_gold():
    return _oracle_image_matrix(P0.copy(), G.copy(), H.copy())
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P0 = np.diag([1.0, 0.0])
S = np.array([[1.0, 0.2], [0.2, 1.0]])
w, u = np.linalg.eigh(0.5 * (S + S.T))
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T
def run_model():
    pred = np.asarray(image_matrix(P0.copy(), G.copy(), H.copy()), dtype=float)
    wrong = H @ P0 @ H
    wrong = 0.5 * (wrong + wrong.T)
    return np.concatenate([
        pred.ravel(),
        [np.max(np.abs(pred - wrong)), np.max(np.abs(pred - P0))],
    ])

def run_gold():
    right = np.asarray(_oracle_image_matrix(P0.copy(), G.copy(), H.copy()), dtype=float)
    wrong = H @ P0 @ H
    wrong = 0.5 * (wrong + wrong.T)
    return np.concatenate([
        right.ravel(),
        [np.max(np.abs(right - wrong)), np.max(np.abs(right - P0))],
    ])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P0 = np.diag([1.0, 0.0])
G = np.eye(2)
H = np.eye(3)

def run_model():
    try:
        image_matrix(P0.copy(), G.copy(), H.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_image_matrix(P0.copy(), G.copy(), H.copy())
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
P0 = np.array([[0.8, 0.1], [0.1, 0.3]], dtype=float)
S = np.array([[1.0, 0.3], [0.3, 1.0]])
w, u = np.linalg.eigh(0.5 * (S + S.T))
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T

def run_model():
    pred = np.asarray(image_matrix(P0.copy(), G.copy(), H.copy()), dtype=float)
    return np.array([np.max(np.abs(pred - pred.T)), pred[0, 0], pred[1, 1]])

def run_gold():
    gold = _oracle_image_matrix(P0.copy(), G.copy(), H.copy())
    return np.array([np.max(np.abs(gold - gold.T)), gold[0, 0], gold[1, 1]])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
