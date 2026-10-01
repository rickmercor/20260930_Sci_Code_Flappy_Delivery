"""
Build a valid intermediate frozen-state density.

The projection-state density is not a valid ROKS density. Recover from
the source paper how the doubly occupied intermediate orbitals are
selected from it and how they are combined with the two final-state
open-shell orbitals. n_occ is the ground-state occupied count from
the first stage.

Returns
-------
return P_INT
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def purify_intermediate_state(
    P_PRJ: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
    S: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    """
    Return the intermediate frozen-state AO spatial density.

    Parameters
    ----------
    P_PRJ : np.ndarray
        Projection-state AO density, shape (n, n).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).
    S : np.ndarray
        AO overlap, shape (n, n).
    n_occ : int
        Number of occupied spatial orbitals in the ground state.

    Returns
    -------
    np.ndarray
        Symmetric intermediate AO spatial density, shape (n, n).

    Raises
    ------
    ValueError
        If the arrays are misaligned, n_occ is invalid, or values are
        non-finite.
    """
    return P_INT

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_purify_intermediate_state(P_PRJ: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray, S: np.ndarray, n_occ: int) -> np.ndarray:
    P_PRJ = np.asarray(P_PRJ, dtype=float)
    S = np.asarray(S, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    n_occ = int(n_occ)
    if P_PRJ.ndim != 2 or P_PRJ.shape[0] != P_PRJ.shape[1] or P_PRJ.shape[0] == 0:
        raise ValueError("P_PRJ must be a nonempty square matrix")
    n = P_PRJ.shape[0]
    if S.shape != (n, n):
        raise ValueError("S must match P_PRJ")
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO vectors must match P_PRJ")
    if n_occ < 2 or n_occ >= n:
        raise ValueError("n_occ must satisfy 2 <= n_occ < n")
    if not np.all(np.isfinite(P_PRJ)) or not np.all(np.isfinite(S)):
        raise ValueError("P_PRJ and S must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO vectors must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    P_PRJ = 0.5 * (P_PRJ + P_PRJ.T)
    S = 0.5 * (S + S.T)
    w_s, u_s = np.linalg.eigh(S)
    if np.any(w_s <= 1e-12):
        raise ValueError("S must be positive definite")
    sh = (u_s * np.sqrt(w_s)) @ u_s.T
    sih = (u_s * (1.0 / np.sqrt(w_s))) @ u_s.T
    a = 0.5 * ((sh @ P_PRJ @ sh) + (sh @ P_PRJ @ sh).T)
    w, y = np.linalg.eigh(a)
    C = sih @ y[:, np.argsort(w)[::-1]]
    n_d = n_occ - 1
    C_d = C[:, :n_d]
    C_d = C_d - np.outer(h_ES, h_ES @ S @ C_d) - np.outer(l_ES, l_ES @ S @ C_d)
    g = 0.5 * ((C_d.T @ S @ C_d) + (C_d.T @ S @ C_d).T)
    gw, gu = np.linalg.eigh(g)
    if np.any(gw <= 1e-12):
        raise ValueError("selected intermediate orbitals are linearly dependent")
    C_d = C_d @ (gu * (1.0 / np.sqrt(gw))) @ gu.T
    P_INT = 2.0 * (C_d @ C_d.T) + np.outer(h_ES, h_ES) + np.outer(l_ES, l_ES)
    return 0.5 * (P_INT + P_INT.T)

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
h_ES = np.array([0.6, 0.0, 0.8, 0.0])
h_ES = h_ES / np.linalg.norm(h_ES)
l_ES = np.eye(4)[:, 3]
I = np.eye(4)
C_hl = np.column_stack([h_ES, l_ES])
P_PRJ = (I - C_hl @ C_hl.T @ S) @ P_GS @ (I - S @ C_hl @ C_hl.T)
n_occ = 2

def run_model():
    return purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)

def run_gold():
    return _oracle_purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(5)
C_GS = np.eye(5)[:, :3]
P_GS = 2.0 * (C_GS @ C_GS.T)
h_ES = np.array([0.5, 0.5, 0.0, 0.7071067811865476, 0.0])
h_ES = h_ES / np.linalg.norm(h_ES)
l_ES = np.eye(5)[:, 4]
I = np.eye(5)
C_hl = np.column_stack([h_ES, l_ES])
P_PRJ = (I - C_hl @ C_hl.T) @ P_GS @ (I - C_hl @ C_hl.T)
n_occ = 3
P_g = _oracle_purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)
w = np.sort(np.linalg.eigvalsh(P_g))[::-1]

def run_model():
    P = np.asarray(purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ))
    return np.concatenate([np.sort(np.linalg.eigvalsh(P))[::-1], [np.trace(P)]])

def run_gold():
    return np.concatenate([w, [np.trace(P_g)]])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(4)
P_PRJ = np.diag([2.0, 1.7, 0.4, 0.0])
h_ES = np.eye(4)[:, 2]
l_ES = np.eye(4)[:, 3]
n_occ = 2
P_g = _oracle_purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)

def run_model():
    P = np.asarray(purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ))
    return np.sort(np.linalg.eigvalsh(P))[::-1]

def run_gold():
    return np.sort(np.linalg.eigvalsh(P_g))[::-1]
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(3)
P_PRJ = np.eye(3)
h_ES = np.eye(3)[:, 0]
l_ES = np.eye(3)[:, 1]
n_occ = 1

def run_model():
    try:
        purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)
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
P_PRJ = np.eye(4).astype(float)
P_PRJ[0, 1] = np.inf
h_ES = np.eye(4)[:, 0]
l_ES = np.eye(4)[:, 1]
S = np.eye(4)
n_occ = 2

def run_model():
    try:
        purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ)
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
