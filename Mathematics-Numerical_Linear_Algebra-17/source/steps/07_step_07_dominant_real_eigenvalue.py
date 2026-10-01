"""
Return the largest real eigenvalue of a square matrix as a native Python float. Reject a matrix whose eigenvalues are not all real.

After last-vector restoration the Hessenberg is similar to a Hermitian Rayleigh-Ritz matrix, so its eigenvalues are real Ritz values. A complex pair means the restoration was skipped. The prompt asks for the largest of those values, not a residual norm and not lambda_max(A).

Returns
-------
native Python float: largest real eigenvalue
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dominant_real_eigenvalue(Hbar: np.ndarray) -> float:
    """Return the largest real eigenvalue as a Python float.

    Parameters
    ----------
    Hbar : np.ndarray
        Square matrix whose spectrum must be real.

    Returns
    -------
    value : float
        Largest real eigenvalue.

    Raises
    ------
    ValueError
        If Hbar is not a nonempty square matrix, or if any eigenvalue
        is non-real.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_dominant_real_eigenvalue(Hbar):
    Hbar = np.asarray(Hbar, dtype=float)
    if Hbar.ndim != 2 or Hbar.shape[0] != Hbar.shape[1]:
        raise ValueError("Hbar must be square")
    if Hbar.shape[0] < 1:
        raise ValueError("Hbar must be nonempty")
    vals = np.linalg.eigvals(Hbar)
    if np.any(np.abs(vals.imag) > 1e-8):
        raise ValueError("corrected Hessenberg must have real eigenvalues")
    return float(np.max(vals.real))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
Hbar = np.array([[2.0, 0.1], [0.1, 1.0]])
""",
            "call": "dominant_real_eigenvalue(Hbar)",
            "gold_call": "_oracle_dominant_real_eigenvalue(Hbar)",
        },
        {
            "setup": """import numpy as np
Hbar = np.diag([4.0, 3.0, 0.5])
""",
            "call": "dominant_real_eigenvalue(Hbar)",
            "gold_call": "_oracle_dominant_real_eigenvalue(Hbar)",
        },
        {
            "setup": """import numpy as np
Hbar = np.array([[0.0, -1.0], [1.0, 0.0]])
def run_model():
    try:
        dominant_real_eigenvalue(Hbar)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_dominant_real_eigenvalue(Hbar)
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
def run_model():
    try:
        dominant_real_eigenvalue(np.ones(3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_dominant_real_eigenvalue(np.ones(3))
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
Hbar = np.array([[-0.5]])
""",
            "call": "dominant_real_eigenvalue(Hbar)",
            "gold_call": "_oracle_dominant_real_eigenvalue(Hbar)",
        },
    ]
