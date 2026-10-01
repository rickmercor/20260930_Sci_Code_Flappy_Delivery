"""
List every apparent-competition triad of a feeding matrix as a row [p, a, b] with a < b, rows sorted by (p, a, b).

Two resources can affect each other through a shared consumer even when they never eat one another. That three-species pattern is not exploitative competition and it is not omnivory. The next step scores those patterns one row at a time, so the list has to be complete and ordered.

Returns
-------
np.ndarray, shape (n_motifs, 3), integer rows [predator, prey_lo, prey_hi] in lexicographic order
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apparent_competition_motifs(A: "np.ndarray") -> "np.ndarray":
    '''Return apparent-competition triads of a feeding matrix.

    Parameters
    ----------
    A : np.ndarray
        Square 0-1 feeding matrix, A[i, j] = 1 if i eats j.

    Returns
    -------
    motifs : np.ndarray
        Shape (n_motifs, 3), rows [p, a, b] with a < b, 0-based indices,
        rows sorted lexicographically by (p, a, b).
        Shape (0, 3) if none exist.

    Raises
    ------
    ValueError
        If A is not a square 2-d 0-1 array with N >= 1.
    '''
    return motifs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apparent_competition_motifs(A: "np.ndarray") -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    rows = []
    for p in range(A.shape[0]):
        prey = np.flatnonzero(A[p] == 1.0)
        for ii in range(prey.size):
            for jj in range(ii + 1, prey.size):
                a = int(prey[ii])
                b = int(prey[jj])
                if A[a, b] == 0.0 and A[b, a] == 0.0:
                    if a > b:
                        a, b = b, a
                    rows.append([p, a, b])
    if not rows:
        return np.zeros((0, 3), dtype=int)
    rows.sort()
    return np.asarray(rows, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,0,0],[0,1,0,0,0,0],[1,0,1,0,0,0],[0,1,0,1,1,0]], dtype=float)
""",
            "call": "apparent_competition_motifs(A)",
            "gold_call": "_oracle_apparent_competition_motifs(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0],[1,0]], dtype=float)
""",
            "call": "apparent_competition_motifs(A)",
            "gold_call": "_oracle_apparent_competition_motifs(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0,0],[0,0,0],[1,1,0]], dtype=float)
""",
            "call": "apparent_competition_motifs(A)",
            "gold_call": "_oracle_apparent_competition_motifs(A)",
        },
    ]
