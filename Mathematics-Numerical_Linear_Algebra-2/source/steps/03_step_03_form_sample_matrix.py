"""
Form the sample matrix required by the subsequent randomized preconditioning subproblems from the outputs of the preceding step.

The randomized construction uses sampled matrix information from the coupled system to build the subsequent approximation state. For this benchmark, the sample is formed deterministically from the previously computed state and the constraint data.

Returns
-------
np.ndarray, shape $(m,k)$ — the sample matrix $W = C^T(Y - Y_D)$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def form_sample_matrix(
    C: np.ndarray,
    Y: np.ndarray,
    Y_D: np.ndarray,
) -> np.ndarray:
    r"""
    Form the matrix state required by the subsequent randomized
    preconditioning subproblems.

    Raises
    ------
    ValueError
        If $C$, $Y$, or $Y_D$ is not a 2D array, if $Y$ and $Y_D$ have
        different shapes, or if $C$ and $Y$ have incompatible row dimensions.
    """
    return W

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_form_sample_matrix(
    C: np.ndarray,
    Y: np.ndarray,
    Y_D: np.ndarray,
) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Y_D = np.asarray(Y_D, dtype=float)

    if C.ndim != 2 or Y.ndim != 2 or Y_D.ndim != 2:
        raise ValueError("C, Y, and Y_D must be 2D arrays")

    if Y.shape != Y_D.shape:
        raise ValueError("Y and Y_D must have the same shape")

    if C.shape[0] != Y.shape[0]:
        raise ValueError("C and Y have incompatible row dimensions")

    W = C.T @ (Y - Y_D)
    return W.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: ordinary case ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(21)
C = rng.standard_normal((4, 7))
Y = rng.standard_normal((4, 3))
Y_D = rng.standard_normal((4, 3))
""",
            "call": "form_sample_matrix(C, Y, Y_D)",
            "gold_call": "_oracle_form_sample_matrix(C, Y, Y_D)",
        },

        # --- Valid: zero residual ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(22)
C = rng.standard_normal((4, 7))
Y = rng.standard_normal((4, 3))
Y_D = Y.copy()
""",
            "call": "form_sample_matrix(C, Y, Y_D)",
            "gold_call": "_oracle_form_sample_matrix(C, Y, Y_D)",
        },

        # --- Valid: single sketch column ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(23)
C = rng.standard_normal((5, 9))
Y = rng.standard_normal((5, 1))
Y_D = rng.standard_normal((5, 1))
""",
            "call": "form_sample_matrix(C, Y, Y_D)",
            "gold_call": "_oracle_form_sample_matrix(C, Y, Y_D)",
        },

        # --- Invalid: Y and Y_D mismatch ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(24)
C = rng.standard_normal((4, 7))
Y = rng.standard_normal((4, 3))
Y_D = rng.standard_normal((4, 2))

def run_model():
    try:
        form_sample_matrix(C, Y, Y_D)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_form_sample_matrix(C, Y, Y_D)
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
