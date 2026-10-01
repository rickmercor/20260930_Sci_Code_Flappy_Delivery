"""
Compute the four auxiliary scalars that let the three-multiplication evaluation be carried out inside a workspace of only three matrices.

After the first product and its shift, the second workspace slot represents X**2 + (c1/2)X; squaring that slot supplies the second product. The four requested scalars are the unique coefficients that rebuild, using only scaled additions, the two factors d0*I + d1*X + d2*X**2 + c1*X**3 + X**4 and e1*X + e2*X**2 + c1*X**3 + X**4, while leaving the first slot equal to f0*I + f1*X + f2*X**2 before the final product. Determine them by matching powers of X rather than by introducing another matrix multiplication.

Returns
-------
np.ndarray, float, shape (4,): the rearrangement scalars [r1, r2, r3, r4].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def workspace_rearrangement_scalars(evaluation: np.ndarray) -> np.ndarray:
    """Compute the workspace rearrangement scalars of the evaluation scheme.

    Parameters
    ----------
    evaluation : np.ndarray
        Shape (10,) array [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of
        evaluation coefficients, in that order.

    Returns
    -------
    scalars : np.ndarray
        Shape (4,) float array [r1, r2, r3, r4] of rearrangement scalars.

    Raises
    ------
    ValueError
        If ``evaluation`` is not a one-dimensional array of length 10, or if
        any of its entries is not finite. The function must raise rather
        than return a placeholder or pad a short input.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(4, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_workspace_rearrangement_scalars(evaluation: np.ndarray) -> np.ndarray:
    import numpy as np

    k = np.asarray(evaluation, dtype=float)
    if k.ndim != 1 or k.shape[0] != 10:
        raise ValueError("evaluation must be a one-dimensional array of length 10")
    if not np.all(np.isfinite(k)):
        raise ValueError("evaluation must be finite")

    c1, d1, d2 = k[0], k[2], k[3]
    e1, f1, f2 = k[4], k[7], k[8]

    quarter = 0.25 * c1 ** 2
    r2 = d2 - quarter
    r1 = d1 - 0.5 * c1 * r2
    r3 = e1 - d1 - 0.5 * c1
    r4 = f1 - f2 * (e1 - d1)
    return np.array([r1, r2, r3, r4], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle by substituting
        # a coefficient set with c1 = 2 straight into the four definitions:
        # r2 = d2 - 1, r1 = d1 - (d2 - 1), r3 = e1 - d1 - 1, r4 = f1 - f2*(e1 - d1).
        # The alternating probe separates all four entries, so a dropped
        # quarter-square term or a swapped sign cannot pass.
        {
            "setup": """import numpy as np
k = np.array([2.0, 0.3, 1.5, 4.0, -2.5, 5.0, 0.0, 7.0, -1.25, 3.0])
exact = np.array([1.5 - (4.0 - 1.0), 4.0 - 1.0, -2.5 - 1.5 - 1.0,
                  7.0 + 1.25 * (-2.5 - 1.5)])
w = np.array([1.0, -10.0, 100.0, -1000.0])
EXPECTED = float(np.dot(w, exact))
""",
            "call": "float(np.dot(w, workspace_rearrangement_scalars(k)))",
            "gold_call": "EXPECTED",
        },
        # --- Valid: the coefficient set of a tabulated family member ---
        {
            "setup": """import numpy as np
k = np.array([-1.714285714285714, 0.3442940441, 0.5835068721, 0.0306122449,
              -0.3642648896, 1.030612245, 0.0, -4.38949812, 4.979890698, -35.0])
w = np.array([1.0, 2.0, 3.0, 4.0])
""",
            "call": "float(np.dot(w, workspace_rearrangement_scalars(k)))",
            "gold_call": "float(np.dot(w, _oracle_workspace_rearrangement_scalars(k)))",
        },
        # --- Valid: a coefficient set with a large first entry, where the
        # quarter-square correction dominates ---
        {
            "setup": """import numpy as np
k = np.array([-6.5, 1.0, -2.0, 3.0, 0.5, 4.0, 1.0, -0.75, 2.5, 8.0])
w = np.array([1.0, -1.0, 1.0, -1.0])
""",
            "call": "float(np.dot(w, workspace_rearrangement_scalars(k)))",
            "gold_call": "float(np.dot(w, _oracle_workspace_rearrangement_scalars(k)))",
        },
        # --- Boundary: a symmetric coefficient set in which c1 vanishes, so the
        # rearrangement reduces to plain differences ---
        {
            "setup": """import numpy as np
k = np.array([0.0, 0.25, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, -0.125, 1.0])
w = np.array([1.0, 8.0, 64.0, 512.0])
""",
            "call": "float(np.dot(w, workspace_rearrangement_scalars(k)))",
            "gold_call": "float(np.dot(w, _oracle_workspace_rearrangement_scalars(k)))",
        },
        # --- Edge: entries that make r3 and r4 vanish simultaneously ---
        {
            "setup": """import numpy as np
k = np.array([1.0, 0.0, 2.0, 0.0, 2.5, 0.0, 0.0, 1.0, 2.0, 1.0])
w = np.array([1.0, 3.0, 9.0, 27.0])
""",
            "call": "float(np.dot(w, workspace_rearrangement_scalars(k)))",
            "gold_call": "float(np.dot(w, _oracle_workspace_rearrangement_scalars(k)))",
        },
        # --- Invalid: wrong number of evaluation coefficients ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        workspace_rearrangement_scalars(np.ones(9))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_workspace_rearrangement_scalars(np.ones(9))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite evaluation coefficient ---
        {
            "setup": """import numpy as np
bad = np.ones(10); bad[3] = np.inf
def run_model():
    try:
        workspace_rearrangement_scalars(bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_workspace_rearrangement_scalars(bad)
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
