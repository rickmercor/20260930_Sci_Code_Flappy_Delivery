"""
Extract the orthonormal basis required by the subsequent randomized preconditioning subproblems from the sample matrix.

The randomized construction uses an orthonormal representation of the sampled matrix to form the subsequent approximation state. For this benchmark, the basis is obtained deterministically from the previously computed sample.

Returns
-------
np.ndarray, shape $(m,k)$ — orthonormal basis $V$ with $V^T V = I_k$, using a thin QR factorization whose $R$ has nonnegative diagonal entries; leave zero-diagonal columns unflipped.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thin_qr_orthonormal_basis(W: np.ndarray) -> np.ndarray:
    r"""
    Construct the orthonormal basis required by the subsequent
    randomized preconditioning subproblems.

    Raises
    ------
    ValueError
        If $W$ is not a 2D array or its dimensions satisfy $m<k$.
    """
    return V

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_thin_qr_orthonormal_basis(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array")

    m, k = W.shape

    if m < k:
        raise ValueError("W must satisfy m >= k")

    V, R = np.linalg.qr(W, mode="reduced")

    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0

    V = V * signs

    return V.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: tall W ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(31)
W = rng.standard_normal((20, 5))
""",
            "call": "thin_qr_orthonormal_basis(W)",
            "gold_call": "_oracle_thin_qr_orthonormal_basis(W)",
        },

        # --- Valid: square W ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(32)
W = rng.standard_normal((6, 6))
""",
            "call": "thin_qr_orthonormal_basis(W)",
            "gold_call": "_oracle_thin_qr_orthonormal_basis(W)",
        },

        # --- Valid: rank-deficient W ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(33)
W = rng.standard_normal((10, 4))
W[:, 2] = 0.0

def run_model():
    V = thin_qr_orthonormal_basis(W)
    return float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))

def run_gold():
    V = _oracle_thin_qr_orthonormal_basis(W)
    return float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },

        # --- Invalid: m < k ---
        {
            "setup": """import numpy as np
W = np.random.default_rng(34).standard_normal((3, 7))

def run_model():
    try:
        thin_qr_orthonormal_basis(W)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_thin_qr_orthonormal_basis(W)
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
