"""
Form the projection-state density that precedes the frozen intermediate.

The source paper builds this density from the ground-state spatial
density and the two final-state open-shell orbitals. Recover that
construction from the paper. Do not replace it by deleting a canonical
HOMO from the ground-state occupied set.

Returns
-------
return P_PRJ
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def frozen_projection_density(
    P_GS: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Return the projection-state AO density.

    Parameters
    ----------
    P_GS : np.ndarray
        Closed-shell ground-state AO spatial density, shape (n, n).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).
    S : np.ndarray
        AO overlap, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric projection-state AO density, shape (n, n).

    Raises
    ------
    ValueError
        If the arrays are misaligned or numerical values are invalid.
    """
    return P_PRJ

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_frozen_projection_density(P_GS: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray, S: np.ndarray) -> np.ndarray:
    P_GS = np.asarray(P_GS, dtype=float)
    S = np.asarray(S, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    if P_GS.ndim != 2 or P_GS.shape[0] != P_GS.shape[1] or P_GS.shape[0] == 0:
        raise ValueError("P_GS must be a nonempty square matrix")
    n = P_GS.shape[0]
    if S.shape != (n, n):
        raise ValueError("S must match P_GS")
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO vectors must match P_GS")
    if not np.all(np.isfinite(P_GS)) or not np.all(np.isfinite(S)):
        raise ValueError("P_GS and S must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO vectors must be finite")
    if np.max(np.abs(P_GS - P_GS.T)) > 1e-8:
        raise ValueError("P_GS must be symmetric")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    P_GS = 0.5 * (P_GS + P_GS.T)
    S = 0.5 * (S + S.T)
    C_hl = np.column_stack([h_ES, l_ES])
    ident = np.eye(n)
    left = ident - C_hl @ C_hl.T @ S
    right = ident - S @ C_hl @ C_hl.T
    P_PRJ = left @ P_GS @ right
    return 0.5 * (P_PRJ + P_PRJ.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
S = np.eye(4)
C_GS = np.eye(4)[:, :2]
P_GS = 2.0 * (C_GS @ C_GS.T)
h_ES = np.eye(4)[:, 0]
l_ES = np.eye(4)[:, 3]

def run_model():
    return frozen_projection_density(P_GS, h_ES, l_ES, S)

def run_gold():
    return _oracle_frozen_projection_density(P_GS, h_ES, l_ES, S)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(4)
C_GS = np.eye(4)[:, :3]
P_GS = 2.0 * (C_GS @ C_GS.T)
h_ES = np.array([0.6, 0.8, 0.0, 0.0])
h_ES = h_ES / np.linalg.norm(h_ES)
l_ES = np.eye(4)[:, 3]
P_g = _oracle_frozen_projection_density(P_GS, h_ES, l_ES, S)

def run_model():
    P = np.asarray(frozen_projection_density(P_GS, h_ES, l_ES, S))
    return np.array([np.trace(P), np.max(np.abs(P - P.T))])

def run_gold():
    return np.array([np.trace(P_g), np.max(np.abs(P_g - P_g.T))])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.2], [0.2, 1.0]])
C_GS = np.array([[1.0], [0.0]])
# not used as a full ROKS set; only tests the projector algebra on a 2x2
P_GS = 2.0 * (C_GS @ C_GS.T)
h_ES = np.array([0.0, 1.0])
l_ES = np.array([1.0, 0.0])
w, u = np.linalg.eigh(S)
sih = (u * (1.0 / np.sqrt(w))) @ u.T
h_ES = sih @ np.array([0.0, 1.0])
l_ES = sih @ np.array([1.0, 0.0])

def run_model():
    P = np.asarray(frozen_projection_density(P_GS, h_ES, l_ES, S))
    return np.array([np.trace(P @ S), np.max(np.abs(P - P.T))])

def run_gold():
    P = _oracle_frozen_projection_density(P_GS, h_ES, l_ES, S)
    return np.array([np.trace(P @ S), np.max(np.abs(P - P.T))])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_GS = np.eye(3)
h_ES = np.eye(3)[:, 0]
l_ES = np.eye(3)[:, 1]
S = np.eye(4)

def run_model():
    try:
        frozen_projection_density(P_GS, h_ES, l_ES, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_frozen_projection_density(P_GS, h_ES, l_ES, S)
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
P_GS = np.eye(3).astype(float)
P_GS[0, 0] = np.nan
h_ES = np.eye(3)[:, 0]
l_ES = np.eye(3)[:, 1]
S = np.eye(3)

def run_model():
    try:
        frozen_projection_density(P_GS, h_ES, l_ES, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_frozen_projection_density(P_GS, h_ES, l_ES, S)
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
