"""
Given a square matrix and a nonempty motif index array, return the largest block reactivity among those rows. Indices in a row must be distinct.

A motif score is the reactivity of the principal block on those nodes. Duplicate indices do not name a block. Empty lists do not name a comparison.

Returns
-------
float, largest motif reactivity
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def max_motif_reactivity(S: "np.ndarray", motifs: "np.ndarray") -> float:
    '''Return the largest block reactivity among listed motifs.

    Parameters
    ----------
    S : np.ndarray
        Square matrix whose blocks are scored.
    motifs : np.ndarray
        Shape (n_motifs, k) integer node indices, 0-based.
        Each row must contain distinct indices.

    Returns
    -------
    r_motif : float
        Largest block reactivity over motif rows.

    Raises
    ------
    ValueError
        If S is not square and finite, if motifs is empty, if any index is
        out of range, or if a motif row contains duplicate indices.
    '''
    return r_motif

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_max_motif_reactivity(S: "np.ndarray", motifs: "np.ndarray") -> float:
    S = np.asarray(S, dtype=float)
    motifs = np.asarray(motifs, dtype=int)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 1:
        raise ValueError("S must be a square 2D array with shape (N,N), N>=1")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if motifs.ndim != 2 or motifs.shape[0] < 1 or motifs.shape[1] < 1:
        raise ValueError("motifs must be a nonempty 2D index array")
    N = S.shape[0]
    best = None
    for row in motifs:
        if row.size != len(set(int(x) for x in row)):
            raise ValueError("motif row must contain distinct indices")
        if np.any(row < 0) or np.any(row >= N):
            raise ValueError("motif index out of range")
        idx = np.asarray(row, dtype=int)
        val = float(np.linalg.eigvalsh(S[np.ix_(idx, idx)])[-1])
        best = val if best is None else max(best, val)
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
S = np.array([[0.2, 0.1, 0.0],[0.1, 0.3, 0.05],[0.0, 0.05, -0.4]], dtype=float)
motifs = np.array([[0, 1, 2], [1, 0, 2]], dtype=int)
""",
            "call": "max_motif_reactivity(S, motifs)",
            "gold_call": "_oracle_max_motif_reactivity(S, motifs)",
        },
        {
            "setup": """import numpy as np
S = np.array([[-1.0]], dtype=float)
motifs = np.array([[0]], dtype=int)
""",
            "call": "max_motif_reactivity(S, motifs)",
            "gold_call": "_oracle_max_motif_reactivity(S, motifs)",
        },
        {
            "setup": """import numpy as np
S = np.diag([1.0, 0.2, -3.0])
motifs = np.array([[0, 2], [1, 2]], dtype=int)
""",
            "call": "max_motif_reactivity(S, motifs)",
            "gold_call": "_oracle_max_motif_reactivity(S, motifs)",
        },
        {
            "setup": """import numpy as np
S = np.array([[0.2, 0.1, 0.0],[0.1, 0.3, 0.05],[0.0, 0.05, -0.4]], dtype=float)
motifs = np.array([[0, 1, 1]], dtype=int)
def run_model():
    try:
        max_motif_reactivity(S, motifs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_max_motif_reactivity(S, motifs)
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
S = np.eye(2)
motifs = np.zeros((0, 3), dtype=int)
def run_model():
    try:
        max_motif_reactivity(S, motifs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_max_motif_reactivity(S, motifs)
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
