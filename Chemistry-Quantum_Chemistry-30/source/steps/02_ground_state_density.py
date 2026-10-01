"""
Form the closed-shell ground-state spatial density.

The ground state from the source paper is a restricted closed-shell
determinant. For this code step, return the full spin-summed AO density
P_GS = 2 C_GS C_GS^T: each occupied spatial orbital contributes
two electrons. The paper's unit-normalized selection projector
has the same occupied subspace but half these occupations.
This matrix is the starting point for the frozen intermediate.

Returns
-------
return P_GS
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ground_state_density(
    C_GS: np.ndarray,
) -> np.ndarray:
    """
    Return the closed-shell ground-state AO spatial density.

    Parameters
    ----------
    C_GS : np.ndarray
        Closed-shell occupied MO coefficients, shape (n, n_occ).

    Returns
    -------
    np.ndarray
        Symmetric spin-summed AO spatial density
        P_GS = 2 C_GS C_GS^T, shape (n, n).

    Raises
    ------
    ValueError
        If C_GS is not a finite, nonempty coefficient matrix.
    """
    return P_GS

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ground_state_density(C_GS: np.ndarray) -> np.ndarray:
    C_GS = np.asarray(C_GS, dtype=float)
    if C_GS.ndim != 2 or C_GS.shape[0] == 0 or C_GS.shape[1] == 0:
        raise ValueError("C_GS must be a nonempty coefficient matrix")
    if not np.all(np.isfinite(C_GS)):
        raise ValueError("C_GS must be finite")
    P_GS = 2.0 * (C_GS @ C_GS.T)
    return 0.5 * (P_GS + P_GS.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
C_GS = np.eye(3)[:, :2]

def run_model():
    return ground_state_density(C_GS)

def run_gold():
    return _oracle_ground_state_density(C_GS)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
C_GS = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
P_g = _oracle_ground_state_density(C_GS)

def run_model():
    P = np.asarray(ground_state_density(C_GS))
    return np.array([np.trace(P), np.max(np.abs(P - P.T))])

def run_gold():
    return np.array([np.trace(P_g), np.max(np.abs(P_g - P_g.T))])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
C_GS = np.array([[1.0, np.inf], [0.0, 0.0]])

def run_model():
    try:
        ground_state_density(C_GS)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_ground_state_density(C_GS)
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
C_GS = np.array([1.0, 0.0, 0.0])

def run_model():
    try:
        ground_state_density(C_GS)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_ground_state_density(C_GS)
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
