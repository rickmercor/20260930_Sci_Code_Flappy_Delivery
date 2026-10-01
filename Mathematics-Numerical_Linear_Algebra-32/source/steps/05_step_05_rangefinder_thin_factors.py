"""
Compute thin QR factors of the amplified rangefinder Yhat and return them stacked as [Q; R], with Q of shape (m, s_width) on top of the s_width-by-s_width triangular factor. Require m >= s_width >= 1 and a full-column-rank rangefinder.

The one-pass reconstruction uses an orthonormal basis for the amplified rangefinder, not the raw sketch. Stacking the triangular factor with that basis keeps both pieces of the factorization in one array for the next step.

Returns
-------
ndarray of shape (m + s_width, s_width): stacked thin factors [Q; R]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rangefinder_thin_factors(Yhat: np.ndarray) -> np.ndarray:
    """Return vstack(Q, R) for the reduced factorization of Yhat.

    Raises
    ------
    ValueError
        If `Yhat` is not 2D, if `m >= s_width >= 1` does not hold, or
        if the rangefinder is rank deficient.
    """
    return np.zeros((1, 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rangefinder_thin_factors(Yhat):
    Yhat = np.asarray(Yhat, dtype=float)
    if Yhat.ndim != 2:
        raise ValueError("Yhat must be 2D")
    m, s_width = Yhat.shape
    if m < s_width or s_width < 1:
        raise ValueError("require m >= s_width >= 1")
    Q, R = np.linalg.qr(Yhat, mode="reduced")
    if np.any(np.abs(np.diag(R)) < 1e-14):
        raise ValueError("rangefinder is rank deficient")
    return np.vstack([Q, R])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
Yhat = np.array([[2.0, 1.0], [0.0, 3.0], [1.0, 1.0]], dtype=float)
""",
            "call": "rangefinder_thin_factors(Yhat)",
            "gold_call": "_oracle_rangefinder_thin_factors(Yhat)",
        },
        {
            "setup": """import numpy as np
Yhat = np.eye(3)
""",
            "call": "rangefinder_thin_factors(Yhat)",
            "gold_call": "_oracle_rangefinder_thin_factors(Yhat)",
        },
        {
            "setup": """import numpy as np
Yhat = np.array([[4.0]], dtype=float)
""",
            "call": "rangefinder_thin_factors(Yhat)",
            "gold_call": "_oracle_rangefinder_thin_factors(Yhat)",
        },
        {
            "setup": """import numpy as np
m, n, s_width, l = 12, 10, 4, 8
rng = np.random.default_rng(7)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode='reduced')
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
s = np.array([4.0, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7])
A = U @ np.diag(s) @ V.T
rng = np.random.default_rng(11)
Omega = rng.standard_normal((n, s_width))
_ = rng.standard_normal((7, m))
Phi = rng.standard_normal((n, l))
Y = A @ Omega
Z = A @ Phi
X, _ = np.linalg.qr(Z.T @ Y, mode='reduced')
Yhat = Z @ X
""",
            "call": "rangefinder_thin_factors(Yhat)",
            "gold_call": "_oracle_rangefinder_thin_factors(Yhat)",
        },
        {
            "setup": """import numpy as np
Yhat = np.ones((2, 3))
def run_model():
    try:
        rangefinder_thin_factors(Yhat)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rangefinder_thin_factors(Yhat)
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
Yhat = np.zeros((3, 2))
def run_model():
    try:
        rangefinder_thin_factors(Yhat)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rangefinder_thin_factors(Yhat)
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
