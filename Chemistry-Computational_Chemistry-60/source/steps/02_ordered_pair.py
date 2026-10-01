"""
Build the ordered overlap-factor pair from the AO overlap S.

S is shape (n, n). It must be finite, square, symmetric, and positive
definite. Otherwise raise ValueError.

Return (G, H) in that order. Both factors are symmetric and have shape
(n, n). G @ G reconstructs S. G @ H is the identity.

Returns
-------
return G, H
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ordered_pair(
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the ordered overlap-factor pair (G, H).

    G @ G reconstructs S. G @ H is the identity.

    Parameters
    ----------
    S : np.ndarray
        AO overlap, shape (n, n).

    Returns
    -------
    G : np.ndarray
        First overlap factor, shape (n, n).
    H : np.ndarray
        Second overlap factor, shape (n, n).

    Raises
    ------
    ValueError
        If S is not square, not symmetric, not positive definite, or
        not finite.
    """
    return G, H

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ordered_pair(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] == 0:
        raise ValueError("S must be a nonempty square matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    S = 0.5 * (S + S.T)
    w, u = np.linalg.eigh(S)
    if np.any(w <= 1e-12):
        raise ValueError("S must be positive definite")
    sqrt_w = np.sqrt(w)
    G = (u * sqrt_w) @ u.T
    H = (u * (1.0 / sqrt_w)) @ u.T
    return G.astype(float), H.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
S = np.eye(4)

def run_model():
    G, H = ordered_pair(S.copy())
    return np.array([
        np.max(np.abs(np.asarray(G) - np.eye(4))),
        np.max(np.abs(np.asarray(H) - np.eye(4))),
    ])

def run_gold():
    G_g, H_g = _oracle_ordered_pair(S.copy())
    return np.array([
        np.max(np.abs(G_g - np.eye(4))),
        np.max(np.abs(H_g - np.eye(4))),
    ])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.2], [0.2, 1.0]])

def run_model():
    G, H = ordered_pair(S.copy())
    prod = np.asarray(G) @ np.asarray(H)
    return np.max(np.abs(prod - np.eye(2)))

def run_gold():
    G_g, H_g = _oracle_ordered_pair(S.copy())
    return np.max(np.abs(G_g @ H_g - np.eye(2)))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.25], [0.25, 1.0]])

def run_model():
    G, H = ordered_pair(S.copy())
    recon = np.asarray(G) @ np.asarray(G)
    return np.max(np.abs(recon - S))

def run_gold():
    G_g, _ = _oracle_ordered_pair(S.copy())
    return np.max(np.abs(G_g @ G_g - S))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.0], [0.0, -0.2]])

def run_model():
    try:
        ordered_pair(S.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_ordered_pair(S.copy())
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
S = np.array([[1.0, 0.35], [0.35, 1.0]])
chol = np.linalg.cholesky(0.5 * (S + S.T))

def run_model():
    G, H = ordered_pair(S.copy())
    G = np.asarray(G, dtype=float)
    return np.array([
        np.max(np.abs(G - G.T)),
        np.max(np.abs(G - chol)),
    ])

def run_gold():
    G_g, H_g = _oracle_ordered_pair(S.copy())
    return np.array([
        np.max(np.abs(G_g - G_g.T)),
        np.max(np.abs(G_g - chol)),
    ])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
