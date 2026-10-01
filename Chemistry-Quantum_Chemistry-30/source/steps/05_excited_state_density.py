"""
Form the final ROKS spatial density.

The final state of the source paper is built from a doubly occupied
spatial set and two singly occupied orbitals that share the same
spatial functions in the mixed and triplet determinants. Recover the
occupancies from the paper.

Returns
-------
return P_ES
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def excited_state_density(
    C_d_ES: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
) -> np.ndarray:
    """
    Return the final-state AO spatial density.

    Parameters
    ----------
    C_d_ES : np.ndarray
        Final-state occupied MO coefficients, shape (n, n_d).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).

    Returns
    -------
    np.ndarray
        Symmetric AO spatial density, shape (n, n).

    Raises
    ------
    ValueError
        If the coefficient arrays are misaligned or non-finite.
    """
    return P_ES

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_excited_state_density(C_d_ES: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray) -> np.ndarray:
    C_d_ES = np.asarray(C_d_ES, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    if C_d_ES.ndim != 2 or C_d_ES.shape[0] == 0 or C_d_ES.shape[1] == 0:
        raise ValueError("C_d_ES must be a nonempty coefficient matrix")
    n = C_d_ES.shape[0]
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO vectors must match C_d_ES")
    if not np.all(np.isfinite(C_d_ES)):
        raise ValueError("C_d_ES must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO vectors must be finite")
    P_ES = 2.0 * (C_d_ES @ C_d_ES.T) + np.outer(h_ES, h_ES) + np.outer(l_ES, l_ES)
    return 0.5 * (P_ES + P_ES.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
C_d_ES = np.eye(4)[:, :2]
h_ES = np.eye(4)[:, 2]
l_ES = np.eye(4)[:, 3]

def run_model():
    return excited_state_density(C_d_ES, h_ES, l_ES)

def run_gold():
    return _oracle_excited_state_density(C_d_ES, h_ES, l_ES)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
C_d_ES = np.eye(4)[:, :1]
h_ES = np.eye(4)[:, 1]
l_ES = np.eye(4)[:, 2]
P_g = _oracle_excited_state_density(C_d_ES, h_ES, l_ES)

def run_model():
    P = np.asarray(excited_state_density(C_d_ES, h_ES, l_ES))
    return np.array([np.trace(P), np.max(np.abs(P - P.T))])

def run_gold():
    return np.array([np.trace(P_g), np.max(np.abs(P_g - P_g.T))])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
C_d_ES = np.eye(3)[:, :1]
h_ES = np.eye(4)[:, 1]
l_ES = np.eye(4)[:, 2]

def run_model():
    try:
        excited_state_density(C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_excited_state_density(C_d_ES, h_ES, l_ES)
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
C_d_ES = np.eye(3)[:, :1].astype(float)
C_d_ES[0, 0] = np.nan
h_ES = np.eye(3)[:, 1]
l_ES = np.eye(3)[:, 2]

def run_model():
    try:
        excited_state_density(C_d_ES, h_ES, l_ES)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_excited_state_density(C_d_ES, h_ES, l_ES)
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
