"""
Compute the element priority scores.



Each score summarizes the assembled residual restricted to the two degrees of

freedom of one element, `$S_e g$`, as a single nonnegative number. The ordered

score vector sets the order in which elements are considered by the

factorization stage, and the summary chosen here fixes which elements tie

against a given threshold. Which summary of `$S_e g$` is meant is fixed by the

established convention and is not restated here.

Returns
-------
np.ndarray of shape (n_elements,), one nonnegative score per element, in input order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_local_residual_scores(
    selection_matrices: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Score every element from the restricted assembled residual.

    Raises ``ValueError`` unless every one of the following holds:
    ``selection_matrices`` is rank three with a nonempty leading axis and at
    least one row per element; ``residual`` is one-dimensional with length
    equal to the last axis of ``selection_matrices``; every entry of both
    arguments is finite; every entry of ``selection_matrices`` equals ``0`` or
    ``1``; and each of its rows sums to exactly one.

    Parameters
    ----------
    selection_matrices : np.ndarray
        Float array of shape ``(n_elements, 2, n_dof)`` with entries ``0.0``
        and ``1.0`` and exactly one unit entry per row. It is float-typed, so
        do not reject it for not being ``dtype=bool``.
    residual : np.ndarray
        Finite assembled residual of shape ``(n_dof,)``.

    Returns
    -------
    np.ndarray
        Element scores of shape ``(n_elements,)`` in input order.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_local_residual_scores(selection_matrices, residual):
    """Reference implementation of the local residual proxy."""
    
    selections = np.asarray(selection_matrices, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if selections.ndim != 3 or selections.shape[0] < 1 or selections.shape[1] < 1:
        raise ValueError("selection_matrices must have shape (E, m, n)")
    if residual.ndim != 1 or residual.size != selections.shape[2]:
        raise ValueError("residual shape is incompatible with selections")
    if not np.all(np.isfinite(selections)) or not np.all(np.isfinite(residual)):
        raise ValueError("inputs must be finite")
    if not np.all((selections == 0.0) | (selections == 1.0)):
        raise ValueError("selection matrices must be Boolean")
    if not np.all(np.sum(selections, axis=2) == 1.0):
        raise ValueError("each selection row must contain one unit entry")
    restricted = np.einsum("emn,n->em", selections, residual)
    return np.max(np.abs(restricted), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, zero-residual, and invalid-shape cases."""
    return [
        {
            "setup": """
import numpy as np
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
S = np.zeros((12, 2, 10))
for e, pair in enumerate(pairs):
    S[e, np.arange(2), pair] = 1.0
g = np.array([24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875])
""",
            "call": "compute_local_residual_scores(S, g)",
            "gold_call": "_oracle_compute_local_residual_scores(S, g)",
        },
        {
            "setup": """
import numpy as np
S = np.eye(2)[None, :, :]
g = np.zeros(2)
""",
            "call": "compute_local_residual_scores(S, g)",
            "gold_call": "_oracle_compute_local_residual_scores(S, g)",
        },
        {
            "setup": """
import numpy as np
S = np.eye(2)[None, :, :]
g = np.zeros(3)
def run_model():
    try:
        compute_local_residual_scores(S, g)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_local_residual_scores(S, g)
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
