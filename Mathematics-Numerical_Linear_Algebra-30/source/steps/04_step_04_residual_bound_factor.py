"""
Return the specialised Frobenius-norm residual factor from the cited source's general bound (Source 1, Theorem 4.1), for this C and this sketch, using the sketch as given. Do not return the unitarily invariant factor. Do not orthonormalise the rows of X. Do not drop the square root from the Frobenius form. Require full column rank of C and t > r.

Theorem 4.1 states a factor for every unitarily invariant norm and a specialised Frobenius-norm factor. This step asks for the Frobenius factor on the sketch as drawn. The unitarily invariant factor, the same factor after orthonormalising the rows of X, and the Frobenius form without the square root are different numbers and fail.

Returns
-------
float: controlling scalar of the source's general accuracy bound
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def residual_bound_factor(C: np.ndarray, X: np.ndarray) -> float:
    """Return the source's specialised Frobenius residual factor.

    The cited source (Source 1, Theorem 4.1) states a factor for every
    unitarily invariant norm and a specialised factor for the Frobenius
    norm. Return the Frobenius factor for this C and this sketch, using
    the sketch as given. Do not orthonormalise the rows of X. Do not
    return the unitarily invariant factor. Do not drop the square root
    from the Frobenius form. The factor is finite and strictly greater
    than one when the source's rank assumption holds.

    Parameters
    ----------
    C : np.ndarray
        Column subset of shape (n, r) with full column rank and 1 <= r < n.
    X : np.ndarray
        Sketch of shape (t, n) with t > r.

    Returns
    -------
    value : float
        The multiplicative factor.

    Raises
    ------
    ValueError
        If C is not a 2D array with 1 <= r < n, if X is not a 2D array with
        n columns, if t <= r, or if C is rank deficient.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_residual_bound_factor(C, X):
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a 2D array")
    n, r = C.shape
    if r < 1 or r >= n:
        raise ValueError("require 1 <= r < n")
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError("X must have shape (t, n)")
    t = X.shape[0]
    if t <= r:
        raise ValueError("require t > r")
    Q, R = np.linalg.qr(C, mode="reduced")
    if np.min(np.abs(np.diag(R))) < 1e-14:
        raise ValueError("C is rank deficient")
    sv = np.linalg.svd(X @ Q, compute_uv=False)
    smin = float(sv[-1])
    if smin <= 0.0:
        raise ValueError("sketch does not embed the column span")
    return float(np.sqrt(1.0 + 1.0 / (smin ** 4)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, seed, t = 12, 7, 6
s = np.array([6.5, 5.2, 4.1, -3.8, -2.9, -1.6, 0.9, 0.55, 0.35, -0.25, 0.18, 0.12])
rng = np.random.default_rng(seed)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = Q @ np.diag(s) @ Q.T
A = 0.5 * (A + A.T)
C = A[:, [5, 8, 9]]
X = np.random.default_rng(11).standard_normal((t, n))
""",
            "call": "residual_bound_factor(C, X)",
            "gold_call": "_oracle_residual_bound_factor(C, X)",
        },
        {
            "setup": """import numpy as np
C = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
X = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
""",
            "call": "residual_bound_factor(C, X)",
            "gold_call": "_oracle_residual_bound_factor(C, X)",
        },
        {
            "setup": """import numpy as np
C = np.array([[3.0], [0.0], [0.0], [0.0]])
X = np.array([[0.0, 1.0, 0.0, 0.0], [0.6, 0.0, 0.8, 0.0]])
""",
            "call": "residual_bound_factor(C, X)",
            "gold_call": "_oracle_residual_bound_factor(C, X)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
C = rng.standard_normal((7, 2))
X = rng.standard_normal((4, 7))
""",
            "call": "residual_bound_factor(C, X)",
            "gold_call": "_oracle_residual_bound_factor(C, X)",
        },
        {
            "setup": """import numpy as np
C = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 0.0]])
X = np.eye(3)
def run_model():
    try:
        residual_bound_factor(C, X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_residual_bound_factor(C, X)
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
C = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
def run_model():
    try:
        residual_bound_factor(C, X)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_residual_bound_factor(C, X)
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
