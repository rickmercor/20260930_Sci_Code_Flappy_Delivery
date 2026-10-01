"""
From a binary feeding matrix, return consumer flags, prey flags, row-normalized diet shares, and column-normalized predator shares.

A feeding link has two readings at once. The row share says how much of a consumer’s intake comes from one prey. The column share says how much of a prey’s loss goes to one predator. Species with no prey are basal. Species with no predator sit at the top. Later matrix entries are written from these four arrays, so a mistake here moves every rate.

Returns
-------
tuple, (rho, sigma, chi, beta) with two length-N float vectors and two (N, N) float arrays
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def feeding_branching(A: "np.ndarray") -> tuple:
    '''Return rho, sigma, chi and beta from a binary feeding matrix.

    Parameters
    ----------
    A : np.ndarray
        Square array with A[i, j] = 1 if i consumes j, else 0.

    Returns
    -------
    rho : np.ndarray
        Shape (N,), 1.0 if row i has at least one prey.
    sigma : np.ndarray
        Shape (N,), 1.0 if column i has at least one predator.
    chi : np.ndarray
        Shape (N, N), row-normalized diet shares.
    beta : np.ndarray
        Shape (N, N), column-normalized predator shares.

    Raises
    ------
    ValueError
        If A is not a square 2-d array with N >= 1, or if any entry is not 0 or 1.
    '''
    return rho, sigma, chi, beta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_feeding_branching(A: "np.ndarray") -> tuple:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    N = A.shape[0]
    rho = (A.sum(axis=1) > 0).astype(float)
    sigma = (A.sum(axis=0) > 0).astype(float)
    row = A.sum(axis=1, keepdims=True)
    col = A.sum(axis=0)
    chi = np.divide(A, row, out=np.zeros((N, N)), where=row != 0)
    beta = np.divide(A, col, out=np.zeros((N, N)), where=col != 0)
    return rho, sigma, chi, beta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,0,0],[0,0,0],[1,1,0]], dtype=float)
""",
            "call": "feeding_branching(A)",
            "gold_call": "_oracle_feeding_branching(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0]], dtype=float)
""",
            "call": "feeding_branching(A)",
            "gold_call": "_oracle_feeding_branching(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,0,1,0]], dtype=float)
""",
            "call": "feeding_branching(A)",
            "gold_call": "_oracle_feeding_branching(A)",
        },
        {
            "setup": """import numpy as np
A = np.zeros((2, 3), dtype=float)
def run_model():
    try:
        feeding_branching(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_feeding_branching(A)
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
