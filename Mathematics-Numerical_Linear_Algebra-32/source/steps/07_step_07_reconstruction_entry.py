"""
Form Q times the best rank-r approximation of B and return the entry in the given 0-based row and column. Require 1 <= rank <= min(B.shape) and indices inside the reconstruction.

The one-pass method returns a truncated QB product, not the full sketched coefficient matrix and not the optimal truncated SVD of A. The prompt scalar is one entry of that truncated product.

Returns
-------
native Python float: one entry of Q [[B]]_rank
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconstruction_entry(
    Q: np.ndarray, B: np.ndarray, rank: int, row: int, col: int
) -> float:
    """Return (Q [[B]]_rank)[row, col] with 0-based indices.

    Raises
    ------
    ValueError
        If `Q` or `B` is not 2D, if `Q` and `B` have incompatible inner
        dimensions, if `rank` is not an integer, if `rank` does not lie
        between 1 and `min(B.shape)`, if `row` or `col` is not an
        integer, or if `row` or `col` is out of bounds.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconstruction_entry(Q, B, rank, row, col):
    Q = np.asarray(Q, dtype=float)
    B = np.asarray(B, dtype=float)
    if Q.ndim != 2 or B.ndim != 2:
        raise ValueError("Q and B must be 2D")
    if Q.shape[1] != B.shape[0]:
        raise ValueError("Q and B have incompatible inner dimensions")
    if not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")
    rank = int(rank)
    max_rank = min(B.shape)
    if rank < 1 or rank > max_rank:
        raise ValueError("rank must lie between 1 and min(B.shape)")
    if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
        raise ValueError("row and col must be integers")
    row, col = int(row), int(col)
    if row < 0 or row >= Q.shape[0] or col < 0 or col >= B.shape[1]:
        raise ValueError("row and col are out of bounds")
    Ub, sb, Vbh = np.linalg.svd(B, full_matrices=False)
    Ahat = (Q @ Ub[:, :rank]) * sb[:rank] @ Vbh[:rank]
    return float(Ahat[row, col])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, s_width, d, l, rank, row, col = 12, 10, 4, 7, 8, 3, 2, 0
rng = np.random.default_rng(7)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode='reduced')
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
s = np.array([4.0, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7])
A = U @ np.diag(s) @ V.T
rng = np.random.default_rng(11)
Omega = rng.standard_normal((n, s_width))
Psi = rng.standard_normal((d, m))
Phi = rng.standard_normal((n, l))
Y = A @ Omega
W = Psi @ A
Z = A @ Phi
X, _ = np.linalg.qr(Z.T @ Y, mode='reduced')
Yhat = Z @ X
Q, _ = np.linalg.qr(Yhat, mode='reduced')
B = np.linalg.lstsq(Psi @ Q, W, rcond=None)[0]
""",
            "call": "reconstruction_entry(Q, B, rank, row, col)",
            "gold_call": "_oracle_reconstruction_entry(Q, B, rank, row, col)",
        },
        {
            "setup": """import numpy as np
Q = np.eye(3)[:, :2]
B = np.array([[3.0, 0.0, 1.0], [0.0, 2.0, 0.0]], dtype=float)
rank, row, col = 2, 0, 0
""",
            "call": "reconstruction_entry(Q, B, rank, row, col)",
            "gold_call": "_oracle_reconstruction_entry(Q, B, rank, row, col)",
        },
        {
            "setup": """import numpy as np
Q = np.array([[1.0], [0.0]], dtype=float)
B = np.array([[-2.0, 4.0]], dtype=float)
rank, row, col = 1, 1, 1
""",
            "call": "reconstruction_entry(Q, B, rank, row, col)",
            "gold_call": "_oracle_reconstruction_entry(Q, B, rank, row, col)",
        },
        {
            "setup": """import numpy as np
Q = np.eye(2)
B = np.eye(2)
def run_model():
    try:
        reconstruction_entry(Q, B, 0, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruction_entry(Q, B, 0, 0, 0)
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
Q = np.eye(2)
B = np.eye(2)
def run_model():
    try:
        reconstruction_entry(Q, B, 1, 2, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruction_entry(Q, B, 1, 2, 0)
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
