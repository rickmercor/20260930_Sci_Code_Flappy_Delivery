"""
Return the reactivity of a square matrix. Refuse entries larger than 1e150 in absolute value.

The eigenvalues of a community matrix describe the long run. The first instant after a shock is a different number: the steepest initial slope of a perturbation. That slope is read from the symmetric part of the matrix. Huge entries are refused so the eigensolver is not asked to finish an overflow.

Returns
-------
float, reactivity of M
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reactivity(M: "np.ndarray") -> float:
    '''Return reactivity of a square matrix M.

    Parameters
    ----------
    M : np.ndarray
        Square real matrix, usually a Jacobian.

    Returns
    -------
    r : float
        Reactivity of M.

    Raises
    ------
    ValueError
        If M is not a square 2-d array with N >= 1, if any entry is not finite,
        if any absolute entry exceeds 1e150, or if the symmetric part is not finite.
    '''
    return r

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reactivity(M: "np.ndarray") -> float:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < 1:
        raise ValueError("M must be a square 2D array with shape (N,N), N>=1")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must be finite")
    if np.any(np.abs(M) > 1e150):
        raise ValueError("M entries must satisfy abs(M) <= 1e150")
    S = 0.5 * M + 0.5 * M.T
    if not np.all(np.isfinite(S)):
        raise ValueError("symmetric part is not finite")
    return float(np.linalg.eigvalsh(S)[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
M = np.array([[-1.0, 2.0],[-0.5, -0.2]], dtype=float)
""",
            "call": "reactivity(M)",
            "gold_call": "_oracle_reactivity(M)",
        },
        {
            "setup": """import numpy as np
M = np.array([[-0.3]], dtype=float)
""",
            "call": "reactivity(M)",
            "gold_call": "_oracle_reactivity(M)",
        },
        {
            "setup": """import numpy as np
M = np.diag([-2.0, -1.0, -0.5])
""",
            "call": "reactivity(M)",
            "gold_call": "_oracle_reactivity(M)",
        },
        {
            "setup": """import numpy as np
M = np.zeros((2, 3))
def run_model():
    try:
        reactivity(M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reactivity(M)
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
M = np.array([[1e308]], dtype=float)
def run_model():
    try:
        reactivity(M)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reactivity(M)
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
