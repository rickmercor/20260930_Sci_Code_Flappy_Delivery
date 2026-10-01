"""
Validate the overlap and the ROKS orbital sets of the source paper.

The closed-shell ground-state occupied orbitals, the final-state
doubly occupied orbitals, and the two final-state open-shell orbitals
must be S-orthonormal with the printed overlap, and the open shells
must be orthogonal to the final-state doubly occupied set. The occupied
count returned here is consumed when the frozen intermediate is built.

Returns
-------
return n_occ
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def validate_roks_orbitals(
    S: np.ndarray,
    C_GS: np.ndarray,
    C_d_ES: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
) -> int:
    """
    Return the number of occupied spatial orbitals in the ground state.

    Parameters
    ----------
    S : np.ndarray
        AO overlap, shape (n, n).
    C_GS : np.ndarray
        Closed-shell ground-state MO coefficients, shape (n, n_occ).
    C_d_ES : np.ndarray
        Final-state doubly MO coefficients, shape (n, n_d).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).

    Returns
    -------
    int
        Number of occupied spatial orbitals in the ground state.

    Raises
    ------
    ValueError
        If the overlap is invalid, the orbital sets are misaligned, or
        the S-orthonormality conditions fail.
    """
    return n_occ

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_validate_roks_orbitals(S: np.ndarray, C_GS: np.ndarray, C_d_ES: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray) -> int:
    S = np.asarray(S, dtype=float)
    C_GS = np.asarray(C_GS, dtype=float)
    C_d_ES = np.asarray(C_d_ES, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] == 0:
        raise ValueError("S must be a nonempty square matrix")
    n = S.shape[0]
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    S = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(S)
    if np.any(w <= 1e-12):
        raise ValueError("S must be positive definite")
    if C_GS.ndim != 2 or C_GS.shape[0] != n or C_GS.shape[1] < 2:
        raise ValueError("C_GS must have shape (n, n_occ) with n_occ >= 2")
    n_occ = int(C_GS.shape[1])
    if C_d_ES.ndim != 2 or C_d_ES.shape[0] != n:
        raise ValueError("C_d_ES must have shape (n, n_d)")
    if C_d_ES.shape[1] != n_occ - 1:
        raise ValueError("the final-state doubly occupied set must have n_occ - 1 columns")
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO coefficient vectors must have length n")
    if not np.all(np.isfinite(C_GS)) or not np.all(np.isfinite(C_d_ES)):
        raise ValueError("MO coefficients must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO coefficients must be finite")

    def _metric_ok(C, ident, name):
        G = C.T @ S @ C
        if np.max(np.abs(G - ident)) > 1e-8:
            raise ValueError(f"{name} must be S-orthonormal")

    _metric_ok(C_GS, np.eye(n_occ), "C_GS")
    _metric_ok(C_d_ES, np.eye(n_occ - 1), "C_d_ES")
    C_hl = np.column_stack([h_ES, l_ES])
    _metric_ok(C_hl, np.eye(2), "open-shell pair")
    if np.max(np.abs(C_d_ES.T @ S @ C_hl)) > 1e-8:
        raise ValueError("final-state doubly occupied orbitals must be orthogonal to both SOMOs")
    return n_occ

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
C_d_ES = np.eye(4)[:, :1]
h_ES = np.eye(4)[:, 1]
l_ES = np.eye(4)[:, 2]

def run_model():
    return validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)

def run_gold():
    return _oracle_validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(4)
C_GS = np.eye(4)[:, :3]
C_d_ES = np.eye(4)[:, :2]
h_ES = np.eye(4)[:, 2]
l_ES = np.eye(4)[:, 3]

def run_model():
    return validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)

def run_gold():
    return _oracle_validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.eye(4)
C_GS = np.eye(4)[:, :2]
C_d_ES = np.eye(4)[:, 1:2]
h_ES = np.eye(4)[:, 1]
l_ES = np.eye(4)[:, 2]

def run_model():
    try:
        validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
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
S = np.eye(3)
C_GS = np.eye(3)[:, :2]
C_d_ES = np.eye(3)[:, :1]
h_ES = np.array([np.nan, 0.0, 0.0])
l_ES = np.eye(3)[:, 2]

def run_model():
    try:
        validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
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
S = np.ones((3, 2))
C_GS = np.eye(3)[:, :2]
C_d_ES = np.eye(3)[:, :1]
h_ES = np.eye(3)[:, 1]
l_ES = np.eye(3)[:, 2]

def run_model():
    try:
        validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
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
