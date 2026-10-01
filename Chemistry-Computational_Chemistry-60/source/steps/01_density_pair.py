"""
Form the AO-basis difference density.

Inputs are the initial-state density P_i and the final-state density P_f.
Both are shape (n, n), with the same n, and every entry finite. A density
that is not square, a pair that is not aligned, or a non-finite entry
raises ValueError.

Return one symmetric matrix dP of shape (n, n).

Returns
-------
return dP
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def density_pair(
    P_i: np.ndarray,
    P_f: np.ndarray,
) -> np.ndarray:
    """
    Return the AO-basis difference density for the analysis.

    Parameters
    ----------
    P_i : np.ndarray
        Initial-state AO density, shape (n, n).
    P_f : np.ndarray
        Final-state AO density, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric difference density, shape (n, n).

    Raises
    ------
    ValueError
        If the densities are not square, not aligned, or not finite.
    """
    return dP

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_density_pair(P_i: np.ndarray, P_f: np.ndarray) -> np.ndarray:
    P_i = np.asarray(P_i, dtype=float)
    P_f = np.asarray(P_f, dtype=float)
    if P_i.ndim != 2 or P_i.shape[0] != P_i.shape[1] or P_i.shape[0] == 0:
        raise ValueError("P_i must be a nonempty square matrix")
    if P_f.shape != P_i.shape:
        raise ValueError("P_i and P_f must have the same shape")
    if not np.all(np.isfinite(P_i)) or not np.all(np.isfinite(P_f)):
        raise ValueError("P_i and P_f must be finite")
    if np.max(np.abs(P_i - P_i.T)) > 1e-10:
        raise ValueError("P_i must be symmetric")
    if np.max(np.abs(P_f - P_f.T)) > 1e-10:
        raise ValueError("P_f must be symmetric")
    dP = P_f - P_i
    return 0.5 * (dP + dP.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
P_i = np.diag([1.0, 1.0, 0.0])
P_f = np.diag([1.0, 0.0, 1.0])

def run_model():
    return density_pair(P_i.copy(), P_f.copy())

def run_gold():
    return _oracle_density_pair(P_i.copy(), P_f.copy())
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_i = np.eye(3)
P_f = np.diag([1.0, 0.0, 1.0])

def run_model():
    return density_pair(P_i.copy(), P_f.copy())

def run_gold():
    return _oracle_density_pair(P_i.copy(), P_f.copy())
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_i = np.array([[1.0, 0.1], [0.1, 0.0]])
P_f = np.array([[0.0, 0.1], [0.1, 1.0]])

def run_model():
    out = density_pair(P_i.copy(), P_f.copy())
    return np.array([np.max(np.abs(out - out.T)), np.trace(out)])

def run_gold():
    dP = _oracle_density_pair(P_i.copy(), P_f.copy())
    return np.array([np.max(np.abs(dP - dP.T)), np.trace(dP)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_i = np.eye(2)
P_f = np.ones((3, 3))

def run_model():
    try:
        density_pair(P_i.copy(), P_f.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_density_pair(P_i.copy(), P_f.copy())
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
P_i = np.eye(2)
P_f = np.array([[1.0, np.inf], [np.inf, 0.0]])

def run_model():
    try:
        density_pair(P_i.copy(), P_f.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_density_pair(P_i.copy(), P_f.copy())
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
