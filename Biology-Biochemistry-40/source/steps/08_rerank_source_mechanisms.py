"""
Rerank the source-feasible mechanisms by assigned similarity score under the deterministic tie convention.

similarity_summary contains mechanism ID, selected reference ID, and assigned score. Rank mechanisms by non-increasing assigned score. Scores within score_tolerance are tied; tied mechanisms are ordered by lower numerical mechanism ID, which encodes source-generation order. Return four columns: one-based rank, mechanism ID, reference ID, and assigned score. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array with columns rank, mechanism ID, selected reference ID, and assigned similarity score.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rerank_source_mechanisms(
    similarity_summary: np.ndarray,
    score_tolerance: float,
) -> np.ndarray:
    """Rerank source-feasible mechanisms by assigned similarity."""
    return np.empty((0, 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rerank_source_mechanisms(similarity_summary, score_tolerance):
    import numpy as np

    summary = np.asarray(similarity_summary, dtype=float)

    if summary.ndim != 2 or summary.shape[0] < 2 or summary.shape[1] != 3:
        raise ValueError('similarity_summary must have shape (n_mechanisms, 3) with at least two rows')
    if not np.all(np.isfinite(summary)):
        raise ValueError('similarity_summary must contain only finite values')

    mechanism_ids = summary[:, 0]
    reference_ids = summary[:, 1]
    scores = summary[:, 2]

    if (
        not np.all(mechanism_ids == np.floor(mechanism_ids))
        or np.any(mechanism_ids <= 0.0)
        or np.unique(mechanism_ids).size != mechanism_ids.size
    ):
        raise ValueError('mechanism IDs must be unique positive integers')
    if not np.all(reference_ids == np.floor(reference_ids)) or np.any(reference_ids <= 0.0):
        raise ValueError('reference IDs must be positive integers')
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError('similarity scores must lie in [0, 1]')

    try:
        tolerance = float(score_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError('score_tolerance must be a real scalar') from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('score_tolerance must be finite and non-negative')

    remaining = list(range(summary.shape[0]))
    ordered_positions = []

    while remaining:
        best_score = max(float(scores[position]) for position in remaining)
        tied_positions = [
            position
            for position in remaining
            if abs(float(scores[position]) - best_score) <= tolerance
        ]
        tied_positions.sort(key=lambda position: int(mechanism_ids[position]))
        ordered_positions.extend(tied_positions)
        tied_set = set(tied_positions)
        remaining = [
            position
            for position in remaining
            if position not in tied_set
        ]

    result = np.empty((summary.shape[0], 4), dtype=float)
    for rank, position in enumerate(ordered_positions, start=1):
        result[rank - 1] = (
            float(rank),
            mechanism_ids[position],
            reference_ids[position],
            scores[position],
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
similarity_summary = np.array([[1,5,0.7],[2,6,0.9],[3,7,0.9],[4,8,0.5]], dtype=float)
score_tolerance = 1e-12
""",
            "call": 'rerank_source_mechanisms(similarity_summary.copy(), score_tolerance)',
            "gold_call": '_oracle_rerank_source_mechanisms(similarity_summary.copy(), score_tolerance)',
        },
        {
            "setup": """import numpy as np
similarity_summary = np.array([[8,2,0.8000000000005],[2,7,0.8],[5,4,0.6]], dtype=float)
score_tolerance = 1e-12
""",
            "call": 'rerank_source_mechanisms(similarity_summary.copy(), score_tolerance)',
            "gold_call": '_oracle_rerank_source_mechanisms(similarity_summary.copy(), score_tolerance)',
        },
        {
            "setup": """import numpy as np
similarity_summary = np.array([[1,5,0.7],[2,6,0.9],[3,7,0.9],[4,8,0.5]], dtype=float)
order = np.array([3,1,0,2], dtype=int)
similarity_summary = similarity_summary[order]
score_tolerance = 1e-12
""",
            "call": 'rerank_source_mechanisms(similarity_summary.copy(), score_tolerance)',
            "gold_call": '_oracle_rerank_source_mechanisms(similarity_summary.copy(), score_tolerance)',
        },
        {
            "setup": """import numpy as np
similarity_summary = np.array([[1,2,0.5],[1,3,0.4]], dtype=float)
score_tolerance = 1e-12

def candidate_wrapper():
    try:
        rerank_source_mechanisms(similarity_summary, score_tolerance)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_rerank_source_mechanisms(similarity_summary, score_tolerance)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
