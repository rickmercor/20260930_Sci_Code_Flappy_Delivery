"""
From a feeding matrix, compute Levine trophic levels and return row timescales α_i=R^(-(t_i-1)/4).

Higher species turn over more slowly. That fact enters the community matrix as a row scale. The coordinate used here is a Levine trophic level measured from 1, then mapped by a metabolic base. If the level iteration does not settle, the scale is not defined.

Returns
-------
np.ndarray, shape (N,), row timescales alpha as float
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def row_timescales(A: "np.ndarray", R: float = 42.0) -> "np.ndarray":
    '''Return row timescales from Levine trophic levels.

    Parameters
    ----------
    A : np.ndarray
        Square feeding matrix, A[i, j] = 1 if i eats j.
    R : float
        Positive metabolic base. Default 42.

    Returns
    -------
    alpha : np.ndarray
        Shape (N,), alpha_i = R**(-(t_i-1)/4).

    Raises
    ------
    ValueError
        If A is not a square 2-d 0-1 array with N >= 1, if R is not finite
        and > 0, or if Levine iteration does not converge within 10000 steps.
    '''
    return alpha

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _trophic_levels(A: "np.ndarray") -> "np.ndarray":
    N = A.shape[0]
    t = np.ones(N, dtype=float)
    for _ in range(10000):
        new = np.ones(N, dtype=float)
        for i in range(N):
            prey = np.flatnonzero(A[i] == 1.0)
            if prey.size:
                new[i] += float(t[prey].mean())
        if np.max(np.abs(new - t)) < 1e-15:
            return new
        t = new
    raise ValueError("Levine trophic levels did not converge")


def _oracle_row_timescales(A: "np.ndarray", R: float = 42.0) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    if not (isinstance(R, (int, float)) and np.isfinite(R) and float(R) > 0.0):
        raise ValueError("R must be finite and > 0")
    t = _trophic_levels(A)
    return np.power(float(R), -0.25 * (t - 1.0)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,0,0],[0,1,0,0,0,0],[1,0,1,0,0,0],[0,1,0,1,1,0]], dtype=float)
R = 42.0
""",
            "call": "row_timescales(A, R)",
            "gold_call": "_oracle_row_timescales(A, R)",
        },
        {
            "setup": """import numpy as np
A = np.zeros((3, 3), dtype=float)
R = 42.0
""",
            "call": "row_timescales(A, R)",
            "gold_call": "_oracle_row_timescales(A, R)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0],[1,0]], dtype=float)
R = 10.0
""",
            "call": "row_timescales(A, R)",
            "gold_call": "_oracle_row_timescales(A, R)",
        },
        {
            "setup": """import numpy as np
A = np.eye(2, dtype=float)
R = 0.0
def run_model():
    try:
        row_timescales(A, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_row_timescales(A, R)
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
A = np.array([[0,1],[1,0]], dtype=float)
R = 42.0
def run_model():
    try:
        row_timescales(A, R)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_row_timescales(A, R)
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
