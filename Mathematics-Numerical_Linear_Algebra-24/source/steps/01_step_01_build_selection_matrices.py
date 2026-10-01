"""
Build the local-to-global selectors used by element assembly.



For ordered element indices `$I_e = (i, j)$$, the 0/1 matrix$$S_e$`

extracts `$(x_i, x_j)$` from the global vector. Its transpose scatters local

Hessian contributions through ``S_e.T @ H_e @ S_e``.

Returns
-------
float np.ndarray of shape (n_elements, 2, n_dof), entries 0.0/1.0, with exactly one unit entry per row
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_selection_matrices(index_pairs: np.ndarray, n_dof: int) -> np.ndarray:
    """Construct one two-row 0/1 selection matrix per element.

    Raises ``ValueError`` unless every one of the following holds: ``n_dof`` is
    an integer of at least 2; ``index_pairs`` is an integer array of shape
    ``(n_elements, 2)`` with at least one element; every index lies in
    ``[0, n_dof)``; and the two indices of each element are distinct.

    Parameters
    ----------
    index_pairs : np.ndarray
        Integer array of shape ``(n_elements, 2)`` in local row order. The two
        entries of a row address different degrees of freedom, so an element
        whose two indices are equal is rejected rather than assembled.
    n_dof : int
        Number of global degrees of freedom.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_elements, 2, n_dof)`` whose entries are
        ``0.0`` and ``1.0``, with exactly one unit entry per row. The array is
        float-typed; a ``dtype=bool`` array is not the expected return value.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_selection_matrices(index_pairs, n_dof):
    """Reference construction with strict index validation."""
    
    pairs = np.asarray(index_pairs)
    if not isinstance(n_dof, (int, np.integer)) or int(n_dof) < 2:
        raise ValueError("n_dof must be an integer at least 2")
    if pairs.ndim != 2 or pairs.shape[0] < 1 or pairs.shape[1] != 2:
        raise ValueError("index_pairs must have shape (n_elements, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("index_pairs must contain integers")
    pairs = pairs.astype(int, copy=False)
    if np.any(pairs < 0) or np.any(pairs >= int(n_dof)):
        raise ValueError("index_pairs contain an out-of-range index")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("the two indices of an element must be distinct")
    result = np.zeros((pairs.shape[0], 2, int(n_dof)), dtype=float)
    rows = np.arange(2)
    for e, pair in enumerate(pairs):
        result[e, rows, pair] = 1.0
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and invalid cases."""
    return [
        {
            "setup": """
import numpy as np
index_pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]], dtype=int)
n_dof = 10
""",
            "call": "build_selection_matrices(index_pairs, n_dof)",
            "gold_call": "_oracle_build_selection_matrices(index_pairs, n_dof)",
        },
        {
            "setup": """
import numpy as np
index_pairs = np.array([[1, 0]], dtype=int)
n_dof = 2
""",
            "call": "build_selection_matrices(index_pairs, n_dof)",
            "gold_call": "_oracle_build_selection_matrices(index_pairs, n_dof)",
        },
        {
            "setup": """
import numpy as np
index_pairs = np.array([[0, 0]], dtype=int)
n_dof = 2
def run_model():
    try:
        build_selection_matrices(index_pairs, n_dof)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_selection_matrices(index_pairs, n_dof)
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
