"""
Assemble the global Newton Hessian from local contributions.



The finite-element relation is ``H = M + sum_e S_e.T @ H_e @ S_e``. This

unprojected assembly is the first matrix whose factorization is attempted, before any

local curvature is modified.

Returns
-------
np.ndarray of shape (n_dof, n_dof), the assembled symmetric Hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_global_hessian(
    regularizer: np.ndarray,
    element_hessians: np.ndarray,
    selection_matrices: np.ndarray,
) -> np.ndarray:
    """Assemble ``M + sum(S_e.T @ H_e @ S_e)``.

    Raises ``ValueError`` unless every one of the following holds:
    ``regularizer`` is a nonempty square two-dimensional array;
    ``element_hessians`` and ``selection_matrices`` are both rank three with
    equal and nonempty leading axes; ``element_hessians`` is square along its
    last two axes with side equal to the middle axis of
    ``selection_matrices``; the last axis of ``selection_matrices`` equals the
    side of ``regularizer``; every entry of all three arguments is finite;
    ``regularizer`` equals its transpose to within ``1e-12`` absolute; and
    each element Hessian equals its own transpose to the same tolerance. The
    two symmetry conditions are verified rather than assumed.

    Parameters
    ----------
    regularizer : np.ndarray
        Finite symmetric matrix of shape ``(n_dof, n_dof)``.
    element_hessians : np.ndarray
        Symmetric local matrices of shape ``(E, m, m)``.
    selection_matrices : np.ndarray
        Scatter/extract matrices of shape ``(E, m, n_dof)``.

    Returns
    -------
    np.ndarray
        Symmetric global Hessian of shape ``(n_dof, n_dof)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_global_hessian(regularizer, element_hessians, selection_matrices):
    """Reference finite-element assembly."""
    
    regularizer = np.asarray(regularizer, dtype=float)
    local = np.asarray(element_hessians, dtype=float)
    selections = np.asarray(selection_matrices, dtype=float)
    if (
        regularizer.ndim != 2
        or regularizer.shape[0] < 1
        or regularizer.shape[0] != regularizer.shape[1]
    ):
        raise ValueError("regularizer must be a nonempty square matrix")
    if local.ndim != 3 or selections.ndim != 3:
        raise ValueError("element_hessians and selections must be rank three")
    if local.shape[0] < 1 or local.shape[0] != selections.shape[0]:
        raise ValueError("element counts must agree")
    if local.shape[1] != local.shape[2] or local.shape[1] != selections.shape[1]:
        raise ValueError("local dimensions must agree")
    if selections.shape[2] != regularizer.shape[0]:
        raise ValueError("global dimensions must agree")
    if (
        not np.all(np.isfinite(regularizer))
        or not np.all(np.isfinite(local))
        or not np.all(np.isfinite(selections))
    ):
        raise ValueError("all inputs must be finite")
    if not np.allclose(regularizer, regularizer.T, rtol=0.0, atol=1e-12):
        raise ValueError("regularizer must be symmetric")
    if not np.allclose(local, np.swapaxes(local, 1, 2), rtol=0.0, atol=1e-12):
        raise ValueError("each element Hessian must be symmetric")
    result = regularizer.copy()
    for hessian, selection in zip(local, selections):
        result += selection.T @ hessian @ selection
    return 0.5 * (result + result.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return full, one-element, and mismatched-shape cases."""
    return [
        {
            "setup": """
import numpy as np
M = 0.4 * np.eye(10)
rows = np.array([
    [-4.0, 0.5, 1.5], [-0.5, 0.75, 1.25], [-4.0, 0.25, 2.0], [-0.75, 0.5, 1.25],
    [2.0, 0.25, 2.0], [-0.5, 0.5, 1.0], [1.75, 0.25, 1.75], [2.0, 0.5, 2.0],
    [-0.25, 0.25, 1.0], [1.75, 0.5, 1.75], [2.0, 0.25, 2.0], [2.25, 0.5, 1.5],
])
E = np.stack([np.array([[p, q], [q, r]]) for p, q, r in rows])
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
S = np.zeros((12, 2, 10))
for e, pair in enumerate(pairs):
    S[e, np.arange(2), pair] = 1.0
""",
            "call": "assemble_global_hessian(M, E, S)",
            "gold_call": "_oracle_assemble_global_hessian(M, E, S)",
        },
        {
            "setup": """
import numpy as np
M = np.eye(2)
E = np.zeros((1, 2, 2))
S = np.eye(2)[None, :, :]
""",
            "call": "assemble_global_hessian(M, E, S)",
            "gold_call": "_oracle_assemble_global_hessian(M, E, S)",
        },
        {
            "setup": """
import numpy as np
M = np.eye(3)
E = np.zeros((1, 2, 2))
S = np.eye(2)[None, :, :]
def run_model():
    try:
        assemble_global_hessian(M, E, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_global_hessian(M, E, S)
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
