"""
Project sampled initial data onto the fixed-rank manifold and return the resulting factors packed as a single array. Retain the leading rank terms of the truncated singular value decomposition, keep the coefficient matrix as the diagonal matrix of retained singular values, and fix the sign freedom by requiring that, in each left factor column, the earliest entry whose magnitude is within a relative 1e-9 of that column's largest magnitude be positive, applying the same flip to the matching right factor column. Return the left factors stacked above the coefficient matrix, stacked above the right factors. Raise ValueError if f_grid is not a non-empty two-dimensional array, if it contains non-finite entries, if rank is not an integer lying between one and the smaller grid dimension, or if the retained singular value at the truncation is no larger than 1e-14 times the leading one.

The low-rank ansatz confines the quasi-distribution to the manifold of matrices of exactly the prescribed rank, factored as a left basis, a small square coefficient matrix, and a right basis, with both bases carrying orthonormal columns. Starting the evolution therefore requires projecting the initial data onto that manifold.

Returns
-------
np.ndarray of shape (Nx + rank + Nk, rank), the left factors stacked above the coefficient matrix stacked above the right factors, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def initial_low_rank_state(f_grid: "np.ndarray", rank: int) -> "np.ndarray":
    '''Project sampled initial data onto the fixed-rank manifold.

    Parameters
    ----------
    f_grid : np.ndarray
        (Nx, Nk) array of the initial data sampled on the phase-space grid.
    rank : int
        Retained rank, satisfying 1 <= rank <= min(Nx, Nk).

    Returns
    -------
    state : np.ndarray
        (Nx + rank + Nk, rank) array holding the left factors in its first Nx
        rows, the coefficient matrix in the next rank rows, and the right
        factors in the remaining Nk rows, float64.
    '''
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_initial_low_rank_state(f_grid: "np.ndarray", rank: int) -> "np.ndarray":
    A = np.asarray(f_grid, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        raise ValueError("f_grid must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("f_grid must contain only finite values")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")
    r = int(rank)
    if r < 1 or r > min(A.shape):
        raise ValueError("rank must satisfy 1 <= rank <= min(f_grid.shape)")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s[0] <= 0.0 or s[r - 1] <= 1e-14 * s[0]:
        raise ValueError("f_grid does not carry the requested rank; S would be singular")
    Ur = U[:, :r].copy()
    Vr = Vt[:r, :].T.copy()
    for j in range(r):
        col = Ur[:, j]
        peak = np.abs(col).max()
        anchor = int(np.flatnonzero(np.abs(col) >= (1.0 - 1e-9) * peak)[0])
        if col[anchor] < 0.0:
            Ur[:, j] = -Ur[:, j]
            Vr[:, j] = -Vr[:, j]
    return np.vstack([Ur, np.diag(s[:r]), Vr]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the production initial datum at the production rank ---
        {
            "setup": """import numpy as np
g = np.linspace(-10.0, 10.0, 128, endpoint=False)
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
f_grid = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
""",
            "call": "initial_low_rank_state(f_grid, 12)",
            "gold_call": "_oracle_initial_low_rank_state(f_grid, 12)",
        },
        # --- boundary: minimal retained rank ---
        {
            "setup": """import numpy as np
g = np.linspace(-10.0, 10.0, 128, endpoint=False)
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
f_grid = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
""",
            "call": "initial_low_rank_state(f_grid, 1)",
            "gold_call": "_oracle_initial_low_rank_state(f_grid, 1)",
        },
        # --- edge: globally negated data, exercising the sign convention ---
        {
            "setup": """import numpy as np
g = np.linspace(-10.0, 10.0, 128, endpoint=False)
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
f_grid = -np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
""",
            "call": "initial_low_rank_state(f_grid, 6)",
            "gold_call": "_oracle_initial_low_rank_state(f_grid, 6)",
        },
        # --- edge: rectangular grid, slowly decaying spectrum ---
        {
            "setup": """import numpy as np
xa = np.linspace(-2.0, 2.0, 20)
yb = np.linspace(-3.0, 3.0, 13)
f_grid = 1.0/(1.0 + (xa[:, None] - yb[None, :])**2)
""",
            "call": "initial_low_rank_state(f_grid, 5)",
            "gold_call": "_oracle_initial_low_rank_state(f_grid, 5)",
        },
        # --- boundary: retained rank equal to the full dimension ---
        {
            "setup": """import numpy as np
f_grid = np.diag([4.0, 3.0, 2.0, 1.0]) + 0.1*np.tri(4)
""",
            "call": "initial_low_rank_state(f_grid, 4)",
            "gold_call": "_oracle_initial_low_rank_state(f_grid, 4)",
        },
        # --- boundary: separable data, whose effective rank is one, accepted
        #     at rank one and rejected above it ---
        {
            "setup": """import numpy as np
g = np.linspace(-10.0, 10.0, 64)
f_grid = np.exp(-(g[:, None] - 1.0)**2/2.0)*np.exp(-2.0*g[None, :]**2)
def probe(fn):
    out = []
    for r in [1, 2, 12]:
        try:
            out.append(int(fn(f_grid, r).shape[1]))
        except ValueError:
            out.append(-1)
        except Exception:
            out.append(-2)
    return out
def run_model():
    return probe(initial_low_rank_state)
def run_gold():
    return probe(_oracle_initial_low_rank_state)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: rank below one ---
        {
            "setup": """import numpy as np
f_grid = np.diag([4.0, 3.0, 2.0, 1.0]) + 0.1*np.tri(4)
def run_model():
    try:
        initial_low_rank_state(f_grid, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_initial_low_rank_state(f_grid, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: rank exceeding the smaller grid dimension ---
        {
            "setup": """import numpy as np
f_grid = np.diag([4.0, 3.0, 2.0, 1.0]) + 0.1*np.tri(4)
def run_model():
    try:
        initial_low_rank_state(f_grid, 9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_initial_low_rank_state(f_grid, 9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-integer rank ---
        {
            "setup": """import numpy as np
f_grid = np.diag([4.0, 3.0, 2.0, 1.0]) + 0.1*np.tri(4)
def run_model():
    try:
        initial_low_rank_state(f_grid, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_initial_low_rank_state(f_grid, 3.0)
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
