"""
Evaluate the supplied AO densities and overlap.

Return the second of the two unoccupied-block populations produced
by the upstream stages. That scalar is not the first-group population
and not the unoccupied-block population of the collapsed difference
density.

Returns
-------
return value
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_instance(
    P_i: np.ndarray,
    P_f: np.ndarray,
    S: np.ndarray,
    eps: float,
) -> float:
    """
    Return the second-group unoccupied-block population.

    Parameters
    ----------
    P_i : np.ndarray
        Initial-state AO density, shape (n, n).
    P_f : np.ndarray
        Final-state AO density, shape (n, n).
    S : np.ndarray
        AO overlap, shape (n, n).
    eps : float
        Printed fixture tolerance, 0 < eps < 1.

    Returns
    -------
    float
        Second-group unoccupied-block population.

    Raises
    ------
    ValueError
        If any input array is invalid, eps is not in (0, 1), a pipeline
        stage returns a non-finite value, or the returned scalar is negative.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_instance(P_i: np.ndarray, P_f: np.ndarray, S: np.ndarray, eps: float) -> float:
    dP = _oracle_density_pair(P_i, P_f)
    dP = np.asarray(dP, dtype=float)
    P_i = np.asarray(P_i, dtype=float)
    S = np.asarray(S, dtype=float)
    if dP.shape != P_i.shape:
        raise ValueError("difference density must match P_i")
    if S.shape != P_i.shape:
        raise ValueError("S must match P_i")

    G, H = _oracle_ordered_pair(S)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    n = dP.shape[0]
    if G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("overlap factors must have shape (n, n)")

    delta, V = _oracle_coefficient_pair(dP, G, H)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    if delta.size != n or V.shape != (n, n):
        raise ValueError("DDNO factors must cover the full AO basis")

    P_t = _oracle_image_matrix(P_i, G, H)
    P_t = np.asarray(P_t, dtype=float)
    if P_t.shape != (n, n):
        raise ValueError("mapped initial density must have shape (n, n)")

    Q = _oracle_frame_matrix(P_t)
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (n, n):
        raise ValueError("unoccupied projector must have shape (n, n)")

    s1, s2 = _oracle_group_values(delta, V, Q, G, H, eps)
    s1 = float(s1)
    s2 = float(s2)
    if not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("block scalars must be finite")
    if s2 < -1e-12:
        raise ValueError("second-group scalar must be nonnegative")
    return s2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
S = np.array([
    [1.00, 0.16, 0.05, 0.02, 0.01, 0.00],
    [0.16, 1.00, 0.11, 0.04, 0.03, 0.01],
    [0.05, 0.11, 1.00, 0.08, 0.05, 0.02],
    [0.02, 0.04, 0.08, 1.00, 0.14, 0.06],
    [0.01, 0.03, 0.05, 0.14, 1.00, 0.12],
    [0.00, 0.01, 0.02, 0.06, 0.12, 1.00],
], dtype=float)
w, u = np.linalg.eigh(0.5 * (S + S.T))
Sih = (u * (1.0 / np.sqrt(w))) @ u.T
C_i = np.zeros((6, 3))
C_i[0, 0] = 1.0
C_i[1, 1] = 1.0
C_i[2, 2] = 1.0
C_f = np.zeros((6, 3))
C_f[0, 0] = np.cos(0.31)
C_f[3, 0] = np.sin(0.31)
C_f[1, 1] = np.cos(0.5 * np.pi)
C_f[4, 1] = np.sin(0.5 * np.pi)
C_f[2, 2] = np.cos(0.05)
C_f[5, 2] = np.sin(0.05)
P_i = Sih @ (C_i @ C_i.T) @ Sih
P_f = Sih @ (C_f @ C_f.T) @ Sih
P_i = 0.5 * (P_i + P_i.T)
P_f = 0.5 * (P_f + P_f.T)
eps = 1.0e-3

def run_model():
    return evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)

def run_gold():
    return _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([
    [1.00, 0.16, 0.05, 0.02, 0.01, 0.00],
    [0.16, 1.00, 0.11, 0.04, 0.03, 0.01],
    [0.05, 0.11, 1.00, 0.08, 0.05, 0.02],
    [0.02, 0.04, 0.08, 1.00, 0.14, 0.06],
    [0.01, 0.03, 0.05, 0.14, 1.00, 0.12],
    [0.00, 0.01, 0.02, 0.06, 0.12, 1.00],
], dtype=float)
w, u = np.linalg.eigh(0.5 * (S + S.T))
Sih = (u * (1.0 / np.sqrt(w))) @ u.T
C_i = np.zeros((6, 3))
C_i[0, 0] = 1.0
C_i[1, 1] = 1.0
C_i[2, 2] = 1.0
C_f = np.zeros((6, 3))
C_f[0, 0] = np.cos(0.31)
C_f[3, 0] = np.sin(0.31)
C_f[1, 1] = np.cos(0.5 * np.pi)
C_f[4, 1] = np.sin(0.5 * np.pi)
C_f[2, 2] = np.cos(0.05)
C_f[5, 2] = np.sin(0.05)
P_i = 0.5 * (Sih @ (C_i @ C_i.T) @ Sih + (Sih @ (C_i @ C_i.T) @ Sih).T)
P_f = 0.5 * (Sih @ (C_f @ C_f.T) @ Sih + (Sih @ (C_f @ C_f.T) @ Sih).T)
eps = 1.0e-3

def run_model():
    return evaluate_instance(P_f.copy(), P_i.copy(), S.copy(), eps)

def run_gold():
    return _oracle_evaluate_instance(P_f.copy(), P_i.copy(), S.copy(), eps)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_i = np.diag([1.0, 1.0, 0.0, 0.0])
X = np.diag([0.0, -1.0, 1.0, 0.0])
R = np.diag([-0.2, 0.0, 0.0, 0.2])
P_f = P_i + X + R
S = np.eye(4)
eps = 1.0e-3

def run_model():
    return evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)

def run_gold():
    return _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_i = np.eye(3)
P_f = np.eye(3)
P_f = P_f.astype(float)
P_f[0, 0] = np.nan
S = np.eye(3)
eps = 1.0e-3

def run_model():
    try:
        evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
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
P_i = np.diag([1.0, 1.0, 0.0])
P_f = np.diag([1.0, 0.0, 1.0])
S = np.eye(3)
eps = 0.0

def run_model():
    try:
        evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
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
P_i = np.diag([1.0, 1.0, 0.0, 0.0])
X = np.diag([0.0, -1.0, 1.0, 0.0])
R = np.diag([-0.05, 0.0, 0.0, 0.05])
P_f = P_i + X + R
S = np.eye(4)
eps = 1.0e-3

def run_model():
    return evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)

def run_gold():
    return _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([
    [1.00, 0.22, 0.08, 0.03],
    [0.22, 1.00, 0.15, 0.07],
    [0.08, 0.15, 1.00, 0.20],
    [0.03, 0.07, 0.20, 1.00],
], dtype=float)
w, u = np.linalg.eigh(0.5 * (S + S.T))
Sih = (u * (1.0 / np.sqrt(w))) @ u.T
C_i = np.zeros((4, 2))
C_i[0, 0] = 1.0
C_i[1, 1] = 1.0
C_f = np.zeros((4, 2))
C_f[0, 0] = np.cos(0.22)
C_f[2, 0] = np.sin(0.22)
C_f[1, 1] = np.cos(0.5 * np.pi)
C_f[3, 1] = np.sin(0.5 * np.pi)
P_i = 0.5 * (Sih @ (C_i @ C_i.T) @ Sih + (Sih @ (C_i @ C_i.T) @ Sih).T)
P_f = 0.5 * (Sih @ (C_f @ C_f.T) @ Sih + (Sih @ (C_f @ C_f.T) @ Sih).T)
eps = 1.0e-3

def run_model():
    return evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)

def run_gold():
    return _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_i = np.diag([1.0, 0.0])
P_f = np.diag([0.75, 0.25])
S = np.eye(2)
eps = 1.0e-3

def run_model():
    return evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)

def run_gold():
    return _oracle_evaluate_instance(P_i.copy(), P_f.copy(), S.copy(), eps)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
