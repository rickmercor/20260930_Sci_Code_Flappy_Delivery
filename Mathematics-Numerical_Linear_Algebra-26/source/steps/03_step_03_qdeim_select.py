"""
Implement qdeim_select for a real or complex orthonormal basis.

For an m-by-r matrix U with r <= m, return the r selected row indices in the exact QDEIM pivot order so that U[indices, :] defines the interpolation submatrix. The selection procedure must be recovered from the QDEIM literature; reject nonmatrix or incompatible inputs with ValueError.

QDEIM chooses interpolation rows from an orthonormal basis so the selected square submatrix supports stable interpolation. The returned pivot order matters because every downstream oblique projector uses these indices without reordering.

Returns
-------
np.ndarray as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def qdeim_select(U: np.ndarray) -> np.ndarray:
    """Select r row indices from an orthonormal matrix following the QDEIM
    procedure from the literature.

    Parameters
    ----------
    U : np.ndarray, shape (m, r)
        Matrix with orthonormal columns.

    Returns
    -------
    indices : np.ndarray, shape (r,), dtype int
        Selected row indices in pivoting order.
    """
    indices = np.array([], dtype=int)
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_qdeim_select(U: np.ndarray) -> np.ndarray:
    """Reference implementation of QDEIM (Algorithm 1)."""
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError('U must be a 2D array')
    m, r = U.shape
    if r > m:
        raise ValueError('U must have at most as many columns as rows')
    if r == 0:
        return np.array([], dtype=int)
    U_work = U.astype(complex).copy()
    indices = np.empty(r, dtype=int)
    for k in range(r):
        row_norms = np.linalg.norm(U_work, axis=1).real
        pk = int(np.argmax(row_norms))
        indices[k] = pk
        u = U_work[pk, :].conj()
        u = u / np.linalg.norm(u)
        proj = U_work @ u
        U_work = U_work - np.outer(proj, u.conj())
    return indices

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
U = np.eye(3, dtype=float)
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 2
        {
            "setup": """import numpy as np
U = np.array([[0.2], [0.8], [0.4], [0.3], [0.2]], dtype=float)
U = U / np.linalg.norm(U)
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 3
        {
            "setup": """import numpy as np
np.random.seed(77)
G = np.random.randn(6, 3) + 1j * np.random.randn(6, 3)
U, _ = np.linalg.qr(G, mode='reduced')
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 4
        {
            "setup": """import numpy as np
np.random.seed(123)
G = np.random.randn(16, 5) + 1j * np.random.randn(16, 5)
U, _ = np.linalg.qr(G, mode='reduced')
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 5
        {
            "setup": """import numpy as np
np.random.seed(303)
G = np.random.randn(12, 4) + 1j * np.random.randn(12, 4)
U, _ = np.linalg.qr(G, mode='reduced')
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 6
        {
            "setup": """import numpy as np
np.random.seed(55)
G = np.random.randn(8, 3) + 1j * np.random.randn(8, 3)
U, _ = np.linalg.qr(G, mode='reduced')
def is_well_conditioned(U, idx):
    sub = U[idx, :]
    cond = np.linalg.cond(sub)
    return float(cond) < 1e8
""",
            "call": "is_well_conditioned(U, qdeim_select(U))",
            "gold_call": "is_well_conditioned(U, _oracle_qdeim_select(U))",
        },
        # Case 7
        {
            "setup": """import numpy as np
perm = np.array([2, 0, 3, 1])
U = np.eye(4, dtype=float)[perm]
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 8
        {
            "setup": """import numpy as np
np.random.seed(999)
G = np.random.randn(20, 6) + 1j * np.random.randn(20, 6)
U, _ = np.linalg.qr(G, mode='reduced')
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 9
        {
            "setup": """import numpy as np
U_bad = np.ones(5)
def run_model():
    try:
        qdeim_select(U_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_qdeim_select(U_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 10
        {
            "setup": """import numpy as np
U_bad = np.ones((2, 5))
def run_model():
    try:
        qdeim_select(U_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_qdeim_select(U_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 11
        {
            "setup": """import numpy as np
U = np.zeros((5, 0))
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
        # Case 12
        {
            "setup": """import numpy as np
np.random.seed(42)
G = 1j * np.random.randn(10, 4)
U, _ = np.linalg.qr(G, mode='reduced')
""",
            "call": "qdeim_select(U).tolist()",
            "gold_call": "_oracle_qdeim_select(U).tolist()",
        },
    ]
