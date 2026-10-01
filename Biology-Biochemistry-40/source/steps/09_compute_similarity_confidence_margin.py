"""
Return the difference between the first- and second-ranked similarity scores.

reranked_summary contains one-based consecutive ranks, unique positive mechanism IDs, positive reference IDs, and non-increasing similarity scores in [0, 1]. Return rank 1 score minus rank 2 score as one finite float. Raise ValueError if fewer than two rows exist or the public contract is violated.

Returns
-------
One finite float equal to the rank-1 score minus the rank-2 score.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_similarity_confidence_margin(
    reranked_summary: np.ndarray,
) -> float:
    """Return the top-minus-runner-up similarity margin."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_similarity_confidence_margin(reranked_summary):
    import numpy as np

    ranking = np.asarray(reranked_summary, dtype=float)

    if ranking.ndim != 2 or ranking.shape[0] < 2 or ranking.shape[1] != 4:
        raise ValueError('reranked_summary must have shape (n_mechanisms, 4) with at least two rows')
    if not np.all(np.isfinite(ranking)):
        raise ValueError('reranked_summary must contain only finite values')

    expected_ranks = np.arange(1, ranking.shape[0] + 1, dtype=float)
    if not np.array_equal(ranking[:, 0], expected_ranks):
        raise ValueError('the rank column must be consecutive and one-based')
    if (
        not np.all(ranking[:, 1] == np.floor(ranking[:, 1]))
        or np.any(ranking[:, 1] <= 0.0)
        or np.unique(ranking[:, 1]).size != ranking.shape[0]
    ):
        raise ValueError('mechanism IDs must be unique positive integers')
    if not np.all(ranking[:, 2] == np.floor(ranking[:, 2])) or np.any(ranking[:, 2] <= 0.0):
        raise ValueError('reference IDs must be positive integers')
    if np.any(ranking[:, 3] < 0.0) or np.any(ranking[:, 3] > 1.0):
        raise ValueError('similarity scores must lie in [0, 1]')
    if np.any(ranking[:-1, 3] < ranking[1:, 3]):
        raise ValueError('reranked_summary must be ordered by non-increasing score')

    return float(ranking[0, 3] - ranking[1, 3])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
reranked_summary = np.array([[1,4,2,0.9],[2,3,7,0.75],[3,1,5,0.6]], dtype=float)
""",
            "call": 'compute_similarity_confidence_margin(reranked_summary.copy())',
            "gold_call": '_oracle_compute_similarity_confidence_margin(reranked_summary.copy())',
        },
        {
            "setup": """import numpy as np
reranked_summary = np.array([[1,8,2,0.8],[2,2,7,0.8]], dtype=float)
""",
            "call": 'compute_similarity_confidence_margin(reranked_summary.copy())',
            "gold_call": '_oracle_compute_similarity_confidence_margin(reranked_summary.copy())',
        },
        {
            "setup": """import numpy as np
reranked_summary = np.array([[1,9,1,1.0],[2,5,3,0.9],[3,2,4,0.4],[4,7,8,0.0]], dtype=float)
""",
            "call": 'compute_similarity_confidence_margin(reranked_summary.copy())',
            "gold_call": '_oracle_compute_similarity_confidence_margin(reranked_summary.copy())',
        },
        {
            "setup": """import numpy as np
reranked_summary = np.array([[1,1,2,0.4],[3,2,3,0.5]], dtype=float)

def candidate_wrapper():
    try:
        compute_similarity_confidence_margin(reranked_summary)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_compute_similarity_confidence_margin(reranked_summary)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
