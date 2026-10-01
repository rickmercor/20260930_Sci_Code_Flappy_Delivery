"""
Assign each mechanism its closest reference analog and corresponding maximum similarity score.

jaccard_matrix contains pairwise scores aligned to mechanism_ids and reference_ids. For each mechanism, find its exact largest score. Treat scores within score_tolerance of that largest value as tied and choose the lower numerical reference ID. Return three columns: mechanism ID, selected reference ID, and the exact largest score. Preserve mechanism-row order. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array with columns mechanism ID, selected closest reference ID, and maximum similarity score
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def summarize_mechanism_similarity(
    jaccard_matrix: np.ndarray,
    mechanism_ids: np.ndarray,
    reference_ids: np.ndarray,
    score_tolerance: float,
) -> np.ndarray:
    """Assign each mechanism its closest reference and maximum score."""
    return np.empty((0, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_summarize_mechanism_similarity(
    jaccard_matrix,
    mechanism_ids,
    reference_ids,
    score_tolerance,
):
    import numpy as np

    scores = np.asarray(jaccard_matrix, dtype=float)
    raw_mechanism_ids = np.asarray(mechanism_ids)
    raw_reference_ids = np.asarray(reference_ids)

    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError('jaccard_matrix must be a non-empty two-dimensional array')
    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError('jaccard_matrix must contain finite values in [0, 1]')

    if (
        raw_mechanism_ids.ndim != 1
        or raw_mechanism_ids.size != scores.shape[0]
        or not np.issubdtype(raw_mechanism_ids.dtype, np.integer)
    ):
        raise ValueError('mechanism_ids must be an integer array aligned with mechanism rows')
    mechanism_id_values = raw_mechanism_ids.astype(int, copy=False)
    if np.any(mechanism_id_values <= 0) or np.unique(mechanism_id_values).size != mechanism_id_values.size:
        raise ValueError('mechanism_ids must contain unique positive integers')

    if (
        raw_reference_ids.ndim != 1
        or raw_reference_ids.size != scores.shape[1]
        or not np.issubdtype(raw_reference_ids.dtype, np.integer)
    ):
        raise ValueError('reference_ids must be an integer array aligned with reference columns')
    reference_id_values = raw_reference_ids.astype(int, copy=False)
    if np.any(reference_id_values <= 0) or np.unique(reference_id_values).size != reference_id_values.size:
        raise ValueError('reference_ids must contain unique positive integers')

    try:
        tolerance = float(score_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError('score_tolerance must be a real scalar') from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('score_tolerance must be finite and non-negative')

    result = np.empty((scores.shape[0], 3), dtype=float)

    for mechanism_index in range(scores.shape[0]):
        best_score = float(np.max(scores[mechanism_index]))
        tied_positions = np.flatnonzero(
            np.abs(scores[mechanism_index] - best_score) <= tolerance
        )
        chosen_position = tied_positions[
            np.argmin(reference_id_values[tied_positions])
        ]
        result[mechanism_index] = (
            float(mechanism_id_values[mechanism_index]),
            float(reference_id_values[chosen_position]),
            best_score,
        )

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
jaccard_matrix = np.array([[0.7, 0.7, 0.2], [0.1, 0.8, 0.8]], dtype=float)
mechanism_ids = np.array([4, 2], dtype=int)
reference_ids = np.array([9, 3, 7], dtype=int)
score_tolerance = 1e-12
""",
            "call": 'summarize_mechanism_similarity(jaccard_matrix.copy(), mechanism_ids.copy(), reference_ids.copy(), score_tolerance)',
            "gold_call": '_oracle_summarize_mechanism_similarity(jaccard_matrix.copy(), mechanism_ids.copy(), reference_ids.copy(), score_tolerance)',
        },
        {
            "setup": """import numpy as np
jaccard_matrix = np.array([[0.8, 0.8000000000005, 0.3]], dtype=float)
mechanism_ids = np.array([11], dtype=int)
reference_ids = np.array([2, 9, 5], dtype=int)
score_tolerance = 1e-12
""",
            "call": 'summarize_mechanism_similarity(jaccard_matrix.copy(), mechanism_ids.copy(), reference_ids.copy(), score_tolerance)',
            "gold_call": '_oracle_summarize_mechanism_similarity(jaccard_matrix.copy(), mechanism_ids.copy(), reference_ids.copy(), score_tolerance)',
        },
        {
            "setup": """import numpy as np
jaccard_matrix = np.array([[0.2,0.9,0.4],[0.6,0.3,0.6],[0.1,0.2,0.5]], dtype=float)
mechanism_ids = np.array([8, 3, 12], dtype=int)
reference_ids = np.array([20, 4, 9], dtype=int)
mechanism_order = np.array([2,0,1], dtype=int)
reference_order = np.array([1,2,0], dtype=int)
jaccard_matrix = jaccard_matrix[mechanism_order][:, reference_order]
mechanism_ids = mechanism_ids[mechanism_order]
reference_ids = reference_ids[reference_order]
score_tolerance = 1e-12
""",
            "call": 'summarize_mechanism_similarity(jaccard_matrix.copy(), mechanism_ids.copy(), reference_ids.copy(), score_tolerance)',
            "gold_call": '_oracle_summarize_mechanism_similarity(jaccard_matrix.copy(), mechanism_ids.copy(), reference_ids.copy(), score_tolerance)',
        },
        {
            "setup": """import numpy as np
jaccard_matrix = np.array([[0.5, 0.5]], dtype=float)
mechanism_ids = np.array([1], dtype=int)
reference_ids = np.array([3, 3], dtype=int)
score_tolerance = 1e-12

def candidate_wrapper():
    try:
        summarize_mechanism_similarity(jaccard_matrix, mechanism_ids, reference_ids, score_tolerance)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_summarize_mechanism_similarity(jaccard_matrix, mechanism_ids, reference_ids, score_tolerance)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
