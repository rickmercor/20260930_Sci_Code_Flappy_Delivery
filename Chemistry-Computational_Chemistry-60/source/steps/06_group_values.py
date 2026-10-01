"""
Return the two unoccupied-block populations of this instance.

From (delta, V, eps) form two reconstructing AO densities, using the
source paper's first group and second group. Map each with G, not H.
Q is a symmetric projector in the G-mapped basis. Do not map Q.
Against Q, return each mapped density's block population, first-group
then second-group.

Arrays must be aligned and finite, and eps must lie in (0, 1).
Otherwise raise ValueError.

Returns
-------
return s1, s2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def group_values(
    delta: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    """
    Return the two unoccupied-block populations.

    Form two reconstructing AO densities from (delta, V, eps) using
    the source first group and second group. Map each with G, not H.
    Score each mapped density against the projector Q.

    Parameters
    ----------
    delta : np.ndarray
        Occupation-change numbers, shape (n,).
    V : np.ndarray
        AO coefficients, shape (n, n).
    Q : np.ndarray
        Unoccupied projector from step 5, shape (n, n).
    G : np.ndarray
        First overlap factor from step 2, shape (n, n).
    H : np.ndarray
        Second overlap factor from step 2, shape (n, n).
    eps : float
        Printed fixture tolerance.

    Returns
    -------
    tuple[float, float]
        First-group block scalar, then the second-group block scalar.

    Raises
    ------
    ValueError
        If the arrays are misaligned, a value is non-finite, or eps
        is not in (0, 1).
    """
    return s1, s2

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_group_values(
    delta: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] == 0:
        raise ValueError("V must be a nonempty square matrix")
    n = V.shape[0]
    if delta.size != n:
        raise ValueError("delta must have one entry per column of V")
    if Q.shape != (n, n) or G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("Q, G, H, and V must have the same shape")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(V)):
        raise ValueError("delta and V must be finite")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("Q, G, and H must be finite")
    if np.max(np.abs(Q - Q.T)) > 1e-10:
        raise ValueError("Q must be symmetric")
    if np.max(np.abs(G - G.T)) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.max(np.abs(H - H.T)) > 1e-10:
        raise ValueError("H must be symmetric")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0 or eps >= 1.0:
        raise ValueError("eps must be a positive number strictly less than 1")
    x = np.zeros_like(delta)
    r = np.zeros_like(delta)
    unit = 1.0 - eps
    for i, d in enumerate(delta):
        ad = abs(d)
        if ad >= unit:
            x[i] = d
        elif ad > 0.0:
            r[i] = d
    D1 = V @ np.diag(x) @ V.T
    D2 = V @ np.diag(r) @ V.T
    D1 = 0.5 * (D1 + D1.T)
    D2 = 0.5 * (D2 + D2.T)
    D1_t = _oracle_image_matrix(D1, G, H)
    D2_t = _oracle_image_matrix(D2, G, H)
    Q = 0.5 * (Q + Q.T)
    s1 = float(np.trace(Q @ D1_t @ Q))
    s2 = float(np.trace(Q @ D2_t @ Q))
    if not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("block scalars must be finite")
    return s1, s2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
delta = np.array([-0.2, -1.0, 1.0, 0.2])
V = np.eye(4)
P_t = np.diag([1.0, 1.0, 0.0, 0.0])
Q = np.eye(4) - P_t
G = np.eye(4)
H = np.eye(4)
eps = 1.0e-3

def run_model():
    return np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)

def run_gold():
    return np.array(_oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
delta = np.array([-0.2, -1.0, 1.0, 0.2])
V = np.eye(4)
P_t = np.diag([1.0, 1.0, 0.0, 0.0])
Q = np.eye(4) - P_t
G = np.eye(4)
H = np.eye(4)
eps = 1.0e-3
pos = V @ np.diag(np.where(delta > 0.0, delta, 0.0)) @ V.T
neg = V @ np.diag(np.where(delta < 0.0, delta, 0.0)) @ V.T
sign_s1 = float(np.trace(Q @ pos @ Q))
sign_s2 = float(np.trace(Q @ neg @ Q))

def run_model():
    pred = np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
    return np.array([pred[0], pred[1], sign_s1, sign_s2], dtype=float)

def run_gold():
    s1, s2 = _oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps)
    return np.array([s1, s2, sign_s1, sign_s2], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.3], [0.3, 1.0]])
w, u = np.linalg.eigh(0.5 * (S + S.T))
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T
U = np.eye(2)
V = H @ U
P_t = np.diag([1.0, 0.0])
Q = np.eye(2) - P_t
delta = np.array([-0.2, 1.0])
eps = 1.0e-3

def run_model():
    return np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)

def run_gold():
    return np.array(_oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_t = np.diag([1.0, 0.0])
Q = np.eye(2) - P_t
V = np.eye(2)
G = np.eye(2)
H = np.eye(2)
delta = np.array([1.0, 0.5])
eps = 1.0e-3

def run_model():
    return np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)

def run_gold():
    return np.array(_oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
delta = np.array([-0.25, -1.0, 1.0, 0.25])
V = np.eye(4)
P_t = np.diag([1.0, 1.0, 0.0, 0.0])
Q = np.eye(4) - P_t
G = np.eye(4)
H = np.eye(4)
eps = 1.0e-3
guess = float(np.sum(delta[delta > 0.0]))

def run_model():
    pred = np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
    return np.array([pred[0], pred[1], guess], dtype=float)

def run_gold():
    s1, s2 = _oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps)
    return np.array([s1, s2, guess], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
eps = 1.0e-3
P_t = np.diag([1.0, 0.0, 0.0, 0.0])
Q = np.eye(4) - P_t
V = np.eye(4)
G = np.eye(4)
H = np.eye(4)
delta = np.array([1.0, -0.5, 1.0 - 0.5 * eps, 1.0 - 2.0 * eps])

def run_model():
    return np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)

def run_gold():
    return np.array(_oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
delta = np.array([-1.0, 1.0])
V = np.eye(2)
P_t = np.diag([1.0, 0.0])
Q = np.eye(2) - P_t
G = np.eye(2)
H = np.eye(2)
eps = 0.0

def run_model():
    try:
        group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps)
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
delta = np.array([-0.2, -1.0, 1.0, 0.2])
V = np.eye(4)
P_t = np.diag([1.0, 1.0, 0.0, 0.0])
Q = np.eye(4) - P_t
G = np.eye(4)
H = np.eye(4)
eps = 1.0e-3
dumped = float(np.trace(Q @ (V @ np.diag(delta) @ V.T) @ Q))

def run_model():
    pred = np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
    return np.array([pred[0], pred[1], dumped], dtype=float)

def run_gold():
    s1, s2 = _oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps)
    return np.array([s1, s2, dumped], dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([
    [1.00, 0.20, 0.00, 0.00],
    [0.20, 1.00, 0.10, 0.00],
    [0.00, 0.10, 1.00, 0.15],
    [0.00, 0.00, 0.15, 1.00],
], dtype=float)
w, u = np.linalg.eigh(0.5 * (S + S.T))
H = (u * (1.0 / np.sqrt(w))) @ u.T
G = (u * np.sqrt(w)) @ u.T
theta = 0.4
U = np.eye(4)
c, s = np.cos(theta), np.sin(theta)
U[0, 0], U[0, 1], U[1, 0], U[1, 1] = c, -s, s, c
V = H @ U
P_t = U[:, :2] @ U[:, :2].T
P_t = 0.5 * (P_t + P_t.T)
Q = np.eye(4) - P_t
delta = np.array([1.0, -1.0, 0.2, -0.2])
eps = 1.0e-3

def run_model():
    return np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)

def run_gold():
    return np.array(_oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([
    [1.00, 0.20, 0.00, 0.00],
    [0.20, 1.00, 0.10, 0.00],
    [0.00, 0.10, 1.00, 0.15],
    [0.00, 0.00, 0.15, 1.00],
], dtype=float)
w, u = np.linalg.eigh(0.5 * (S + S.T))
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T
theta = 0.3
Pt_i = np.diag([1.0, 1.0, 0.0, 0.0])
Pt_f = np.zeros((4, 4))
Pt_f[2, 2] = 1.0
c, s = np.cos(theta), np.sin(theta)
Pt_f[1, 1], Pt_f[1, 3], Pt_f[3, 1], Pt_f[3, 3] = c * c, c * s, c * s, s * s
delta, U = np.linalg.eigh(Pt_f - Pt_i)
V = H @ U
Q = np.eye(4) - Pt_i
eps = 1.0e-3

def run_model():
    return np.array(group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)

def run_gold():
    return np.array(_oracle_group_values(delta.copy(), V.copy(), Q.copy(), G.copy(), H.copy(), eps), dtype=float)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
