"""
Encode graph-node reaches with adaptive sparse or dense storage.

A genotype representation graph node represents the union of the sample

leaves below its children. The multitree property makes sibling reaches

disjoint. A reach with density d / N at least dense_threshold is stored

densely; otherwise it remains sparse. Integer bitmasks retain the exact reach

while a flag records the adaptive choice.

Returns
-------
np.ndarray of shape (n_nodes, 3), containing reach mask, reach size, and dense flag as int64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def encode_adaptive_reaches(
    children: list[list[int]], n_samples: int, dense_threshold: float
) -> "np.ndarray":
    '''Encode exact descendant reaches and their adaptive storage class.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered child identifiers for every graph node.
    n_samples : int
        Number of leading sample leaves.
    dense_threshold : float
        Density threshold in the interval (0, 1].

    Raises
    ------
    ValueError
        If n_samples is invalid or above 62, dense_threshold is outside (0, 1],
        the graph is empty or not topologically ordered, a sample has children,
        an internal node has no children, a child is repeated, or sibling
        descendant reaches overlap.

    Returns
    -------
    reach_encoding : np.ndarray
        Integer columns for reach bitmask, reach size, and dense-storage flag.
    '''
    return reach_encoding  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math  # noqa: E402

import numpy as np  # noqa: E402, F811


def _oracle_encode_adaptive_reaches(
    children: list[list[int]], n_samples: int, dense_threshold: float
) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples <= 62):
        raise ValueError("n_samples must be an integer from 1 through 62")
    if not isinstance(dense_threshold, (int, float)) or isinstance(dense_threshold, bool):
        raise ValueError("dense_threshold must be real")
    dense_threshold = float(dense_threshold)
    if not math.isfinite(dense_threshold) or not (0.0 < dense_threshold <= 1.0):
        raise ValueError("dense_threshold must lie in (0, 1]")
    if not isinstance(children, (list, tuple)) or len(children) <= n_samples:
        raise ValueError("children must contain sample and internal nodes")

    reach_masks = np.zeros(len(children), dtype=np.int64)
    reach_sizes = np.zeros(len(children), dtype=np.int64)
    dense_flags = np.zeros(len(children), dtype=np.int64)
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples:
            if len(raw_children) != 0:
                raise ValueError("sample leaves cannot have children")
            reach_masks[node] = np.int64(1 << node)
        else:
            if len(raw_children) == 0:
                raise ValueError("internal nodes must have children")
            if any(not isinstance(child, int) or isinstance(child, bool) for child in raw_children):
                raise ValueError("child identifiers must be integers")
            if len(set(raw_children)) != len(raw_children):
                raise ValueError("child identifiers cannot repeat")
            if any(child < 0 or child >= node for child in raw_children):
                raise ValueError("children must precede their parent")
            mask = 0
            for child in raw_children:
                child_mask = int(reach_masks[child])
                if mask & child_mask:
                    raise ValueError("sibling descendant reaches must be disjoint")
                mask |= child_mask
            reach_masks[node] = np.int64(mask)
        reach_sizes[node] = int(int(reach_masks[node]).bit_count())
        dense_flags[node] = int(float(reach_sizes[node]) / n_samples >= dense_threshold)
    return np.column_stack((reach_masks, reach_sizes, dense_flags)).astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """children = [[], [], [], [], [0,1], [2,3], [4,5]]
n_samples = 4
dense_threshold = 0.5
""",
            "call": "encode_adaptive_reaches(children, n_samples, dense_threshold).tolist()",
            "gold_call": "_oracle_encode_adaptive_reaches(children, n_samples, dense_threshold).tolist()",
        },
        {
            "setup": """children = [[], [], [0,1]]
n_samples = 2
dense_threshold = 1.0
""",
            "call": "encode_adaptive_reaches(children, n_samples, dense_threshold).tolist()",
            "gold_call": "_oracle_encode_adaptive_reaches(children, n_samples, dense_threshold).tolist()",
        },
        {
            "setup": """children = [[], [], [0], [0,1], [2,3]]
n_samples = 2
dense_threshold = 0.5
def run_model():
    try:
        encode_adaptive_reaches(children, n_samples, dense_threshold)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_encode_adaptive_reaches(children, n_samples, dense_threshold)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
