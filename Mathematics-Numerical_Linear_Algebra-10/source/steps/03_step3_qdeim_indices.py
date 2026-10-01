"""
Determine the Q-DEIM row indices for a basis matrix.

Q-DEIM is a deterministic interpolation strategy that selects a rank-sized set of coordinates from a basis. Its role is to provide an initial nonsingular sampling pattern whose quality reflects the geometry of the current basis.

Returns
-------
np.ndarray, one-based integer Q-DEIM row indices of length $m$ (the number of columns of $V$).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def qdeim_indices(V: np.ndarray, tie_tol: float = 1e-12) -> np.ndarray:
    r"""Determine the Q-DEIM row indices for a basis matrix.

    Parameters
    ----------
    V : np.ndarray
        Finite two-dimensional full-column-rank basis matrix with
        shape $n\times m$, where $n\ge m$.
    tie_tol : float, optional
        Finite nonnegative tolerance used to resolve numerically tied
        pivot scores. The default is 1e-12.

    Returns
    -------
    indices : np.ndarray
        One-dimensional integer array of length $m$ containing the
        selected one-based row indices in pivot order (pivoted QR of
        $V^{\top}$; ties within `tie_tol` resolve to the smallest index).

    Raises
    ------
    ValueError
        If V is not a finite nonempty two-dimensional matrix, if
        $n<m$, if V is not full column rank, if tie_tol is invalid,
        or if a rank-deficient pivot is encountered.
    """
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_qdeim_indices(V, tie_tol=1e-12):
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2 or min(V.shape) < 1 or not np.all(np.isfinite(V)):
        raise ValueError("V must be finite nonempty 2D")
    n, m = V.shape
    if n < m or np.linalg.matrix_rank(V) != m:
        raise ValueError("V must have full column rank with n>=m")
    if not isinstance(tie_tol, (int, float, np.integer, np.floating)) or not np.isfinite(tie_tol) or float(tie_tol) < 0:
        raise ValueError("invalid tie_tol")
    tol = float(tie_tol)
    B = V.T.copy()
    selected = []
    Q = []
    for _ in range(m):
        residuals = {}
        norms = {}
        for j in range(n):
            if j in selected:
                continue
            r = B[:, j].copy()
            for q in Q:
                r -= q * float(q @ r)
            for q in Q:
                r -= q * float(q @ r)
            residuals[j] = r
            norms[j] = float(np.linalg.norm(r))
        mx = max(norms.values())
        cand = [j for j, v in norms.items() if abs(v - mx) <= tol]
        j = min(cand)
        r = residuals[j]
        nr = float(np.linalg.norm(r))
        if nr <= np.finfo(np.float64).eps:
            raise ValueError("rank-deficient pivot")
        selected.append(j)
        Q.append(r / nr)
    return np.asarray(selected, dtype=np.int64) + 1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
V=np.array([[1.,0.,.1],[.2,1.,0.],[.7,.1,1.],[.1,.5,.4],[.4,.3,.2]])""",'call':'qdeim_indices(V)','gold_call':'_oracle_qdeim_indices(V)'},
        {'setup':"""import numpy as np
V=np.array([[1.,0.],[1.,0.],[0.,1.]])
tie_tol=1e-12""",'call':'qdeim_indices(V,tie_tol)','gold_call':'_oracle_qdeim_indices(V,tie_tol)'},
        {'setup':"""import numpy as np
V=np.array([[2.],[1.],[-3.]])""",'call':'qdeim_indices(V)','gold_call':'_oracle_qdeim_indices(V)'},
        {'setup':"""import numpy as np
V=np.array([[1.,0.],[2.,0.],[3.,0.]])
def run_model():
    try: qdeim_indices(V); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_qdeim_indices(V); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
V=np.eye(2); tie_tol=-1.
def run_model():
    try: qdeim_indices(V,tie_tol); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_qdeim_indices(V,tie_tol); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
V=np.array([[1.,np.nan],[0.,1.]])
def run_model():
    try: qdeim_indices(V); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_qdeim_indices(V); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
