"""
Recover the coefficient matrix B by solving (Psi Q) B = W in the least-squares sense. Require d >= s_width and matching inner dimensions. This is a sketched corange fit, not Q^T A.

After the rangefinder is orthonormalized, the corange sketch W = Psi A determines B without a second pass over A. Replacing this solve by the two-pass identity B = Q^T A produces a different reconstruction on the prompt instance.

Returns
-------
ndarray of shape (s_width, n): coefficient matrix B
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sketched_coefficient_matrix(Q: np.ndarray, Psi: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Solve (Psi Q) B = W in the least-squares sense.

    Raises
    ------
    ValueError
        If `Q`, `Psi`, or `W` is not 2D; if `Q` is empty; if `Psi` does
        not have as many columns as `Q` has rows; if `W` does not have
        as many rows as `Psi`; if `d >= s_width` does not hold; or if
        `W` has no columns.
    """
    return np.zeros((1, 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sketched_coefficient_matrix(Q, Psi, W):
    Q = np.asarray(Q, dtype=float)
    Psi = np.asarray(Psi, dtype=float)
    W = np.asarray(W, dtype=float)
    if Q.ndim != 2 or Psi.ndim != 2 or W.ndim != 2:
        raise ValueError("Q, Psi, and W must be 2D")
    m, s_width = Q.shape
    d, m_psi = Psi.shape
    d_w, n = W.shape
    if m < 1 or s_width < 1:
        raise ValueError("Q must be nonempty")
    if m_psi != m:
        raise ValueError("Psi must have as many columns as Q has rows")
    if d_w != d:
        raise ValueError("W must have as many rows as Psi")
    if d < s_width:
        raise ValueError("require d >= s_width")
    if n < 1:
        raise ValueError("W must have at least one column")
    B, *_ = np.linalg.lstsq(Psi @ Q, W, rcond=None)
    return np.asarray(B, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, s_width, d, l = 12, 10, 4, 7, 8
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
""",
            "call": "sketched_coefficient_matrix(Q, Psi, W)",
            "gold_call": "_oracle_sketched_coefficient_matrix(Q, Psi, W)",
        },
        {
            "setup": """import numpy as np
Q = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
Psi = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
W = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]], dtype=float)
""",
            "call": "sketched_coefficient_matrix(Q, Psi, W)",
            "gold_call": "_oracle_sketched_coefficient_matrix(Q, Psi, W)",
        },
        {
            "setup": """import numpy as np
Q = np.array([[1.0]], dtype=float)
Psi = np.array([[2.0]], dtype=float)
W = np.array([[4.0, -2.0]], dtype=float)
""",
            "call": "sketched_coefficient_matrix(Q, Psi, W)",
            "gold_call": "_oracle_sketched_coefficient_matrix(Q, Psi, W)",
        },
        {
            "setup": """import numpy as np
Q = np.eye(2)
Psi = np.ones((1, 2))
W = np.ones((1, 3))
def run_model():
    try:
        sketched_coefficient_matrix(Q, Psi, W)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sketched_coefficient_matrix(Q, Psi, W)
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
Psi = np.eye(3)
W = np.ones((2, 2))
def run_model():
    try:
        sketched_coefficient_matrix(Q, Psi, W)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sketched_coefficient_matrix(Q, Psi, W)
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
