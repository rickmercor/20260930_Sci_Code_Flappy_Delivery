"""
Compute all pairwise candidate-to-reference Jaccard similarities over aligned arrow-environment features.

mechanism_arrow_sets and reference_arrow_sets are non-empty aligned binary feature arrays. For every mechanism-reference pair, compute the size of the feature intersection divided by the size of the feature union. Return a mechanism-by-reference float matrix and preserve both row orders. Every row must contain at least one feature. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array containing pairwise Jaccard scores for mechanism rows and reference columns.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_mechanism_reference_jaccard(
    mechanism_arrow_sets: np.ndarray,
    reference_arrow_sets: np.ndarray,
) -> np.ndarray:
    """Compute all mechanism-reference Jaccard similarities."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_mechanism_reference_jaccard(mechanism_arrow_sets, reference_arrow_sets):
    import numpy as np

    mechanism = np.asarray(mechanism_arrow_sets, dtype=float)
    reference = np.asarray(reference_arrow_sets, dtype=float)

    if mechanism.ndim != 2 or mechanism.shape[0] < 1 or mechanism.shape[1] < 1:
        raise ValueError('mechanism_arrow_sets must be a non-empty two-dimensional array')
    if reference.ndim != 2 or reference.shape[0] < 1 or reference.shape[1] != mechanism.shape[1]:
        raise ValueError('reference_arrow_sets must align with arrow columns')
    if (
        not np.all(np.isfinite(mechanism))
        or not np.all((mechanism == 0.0) | (mechanism == 1.0))
        or not np.all(np.isfinite(reference))
        or not np.all((reference == 0.0) | (reference == 1.0))
    ):
        raise ValueError('arrow-set arrays must contain only 0.0 and 1.0')
    if np.any(np.sum(mechanism, axis=1) < 1.0) or np.any(np.sum(reference, axis=1) < 1.0):
        raise ValueError('every mechanism and reference must contain at least one arrow feature')

    result = np.empty(
        (mechanism.shape[0], reference.shape[0]),
        dtype=float,
    )

    for mechanism_index in range(mechanism.shape[0]):
        mechanism_mask = mechanism[mechanism_index] == 1.0
        for reference_index in range(reference.shape[0]):
            reference_mask = reference[reference_index] == 1.0
            intersection = int(np.count_nonzero(mechanism_mask & reference_mask))
            union = int(np.count_nonzero(mechanism_mask | reference_mask))
            result[mechanism_index, reference_index] = intersection / union

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
mechanism_arrow_sets = np.array([[1,1,0,0],[1,0,1,0]], dtype=float)
reference_arrow_sets = np.array([[1,1,1,0],[0,0,1,1],[1,0,0,0]], dtype=float)
""",
            "call": 'compute_mechanism_reference_jaccard(mechanism_arrow_sets.copy(), reference_arrow_sets.copy())',
            "gold_call": '_oracle_compute_mechanism_reference_jaccard(mechanism_arrow_sets.copy(), reference_arrow_sets.copy())',
        },
        {
            "setup": """import numpy as np
mechanism_arrow_sets = np.array([[1,0,1],[0,1,0]], dtype=float)
reference_arrow_sets = np.array([[1,0,1],[1,0,0],[0,0,1]], dtype=float)
""",
            "call": 'compute_mechanism_reference_jaccard(mechanism_arrow_sets.copy(), reference_arrow_sets.copy())',
            "gold_call": '_oracle_compute_mechanism_reference_jaccard(mechanism_arrow_sets.copy(), reference_arrow_sets.copy())',
        },
        {
            "setup": """import numpy as np
mechanism_arrow_sets = np.array([[1,1,0,0],[1,0,1,0],[0,1,0,1]], dtype=float)
reference_arrow_sets = np.array([[1,1,1,0],[0,0,1,1]], dtype=float)
feature_order = np.array([3, 1, 0, 2], dtype=int)
mechanism_order = np.array([2, 0, 1], dtype=int)
reference_order = np.array([1, 0], dtype=int)
mechanism_arrow_sets = mechanism_arrow_sets[mechanism_order][:, feature_order]
reference_arrow_sets = reference_arrow_sets[reference_order][:, feature_order]
""",
            "call": 'compute_mechanism_reference_jaccard(mechanism_arrow_sets.copy(), reference_arrow_sets.copy())',
            "gold_call": '_oracle_compute_mechanism_reference_jaccard(mechanism_arrow_sets.copy(), reference_arrow_sets.copy())',
        },
        {
            "setup": """import numpy as np
mechanism_arrow_sets = np.array([[1.0, 0.0]], dtype=float)
reference_arrow_sets = np.array([[0.0, 0.0]], dtype=float)

def candidate_wrapper():
    try:
        compute_mechanism_reference_jaccard(mechanism_arrow_sets, reference_arrow_sets)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_compute_mechanism_reference_jaccard(mechanism_arrow_sets, reference_arrow_sets)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
