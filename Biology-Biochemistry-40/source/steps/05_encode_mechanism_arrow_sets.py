"""
Construct the unique arrow-environment feature collection for every collected mechanism.

mechanism_rule_counts contains non-negative integer rule counts for each collected mechanism. rule_arrow_incidence is a binary rule-by-feature array. A mechanism feature is present when at least one used rule carries that feature; repeated rules or repeated contributions do not increase the numerical code. Return a binary mechanism-by-feature array and preserve mechanism and feature order. Raise ValueError if any mechanism is empty or the public contract is violated.

Returns
-------
2D NumPy float array containing one binary arrow-environment feature row per mechanism.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def encode_mechanism_arrow_sets(
    mechanism_rule_counts: np.ndarray,
    rule_arrow_incidence: np.ndarray,
) -> np.ndarray:
    """Encode each mechanism as a binary arrow-environment collection."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_encode_mechanism_arrow_sets(mechanism_rule_counts, rule_arrow_incidence):
    import numpy as np

    counts = np.asarray(mechanism_rule_counts, dtype=float)
    incidence = np.asarray(rule_arrow_incidence, dtype=float)

    if counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[1] < 1:
        raise ValueError('mechanism_rule_counts must be a non-empty two-dimensional array')
    if incidence.ndim != 2 or incidence.shape[0] != counts.shape[1] or incidence.shape[1] < 1:
        raise ValueError('rule_arrow_incidence must align with rule columns and contain arrow columns')
    if (
        not np.all(np.isfinite(counts))
        or not np.all(counts == np.floor(counts))
        or np.any(counts < 0.0)
    ):
        raise ValueError('mechanism_rule_counts must contain non-negative integers')
    if (
        not np.all(np.isfinite(incidence))
        or not np.all((incidence == 0.0) | (incidence == 1.0))
    ):
        raise ValueError('rule_arrow_incidence must contain only 0.0 and 1.0')
    if np.any(np.sum(counts, axis=1) < 1.0):
        raise ValueError('every mechanism must contain at least one rule')

    return (
        ((counts > 0.0).astype(float) @ incidence) > 0.0
    ).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
mechanism_rule_counts = np.array([[1, 1, 0], [0, 2, 1]], dtype=float)
rule_arrow_incidence = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
], dtype=float)
""",
            "call": 'encode_mechanism_arrow_sets(mechanism_rule_counts.copy(), rule_arrow_incidence.copy())',
            "gold_call": '_oracle_encode_mechanism_arrow_sets(mechanism_rule_counts.copy(), rule_arrow_incidence.copy())',
        },
        {
            "setup": """import numpy as np
mechanism_rule_counts = np.array([[3, 0], [1, 1]], dtype=float)
rule_arrow_incidence = np.array([[1, 1, 0], [1, 0, 1]], dtype=float)
""",
            "call": 'encode_mechanism_arrow_sets(mechanism_rule_counts.copy(), rule_arrow_incidence.copy())',
            "gold_call": '_oracle_encode_mechanism_arrow_sets(mechanism_rule_counts.copy(), rule_arrow_incidence.copy())',
        },
        {
            "setup": """import numpy as np
mechanism_rule_counts = np.array([[1, 0, 2], [0, 1, 1]], dtype=float)
rule_arrow_incidence = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1]], dtype=float)
order = np.array([2, 0, 1], dtype=int)
mechanism_rule_counts = mechanism_rule_counts[:, order]
rule_arrow_incidence = rule_arrow_incidence[order]
""",
            "call": 'encode_mechanism_arrow_sets(mechanism_rule_counts.copy(), rule_arrow_incidence.copy())',
            "gold_call": '_oracle_encode_mechanism_arrow_sets(mechanism_rule_counts.copy(), rule_arrow_incidence.copy())',
        },
        {
            "setup": """import numpy as np
mechanism_rule_counts = np.array([[1.0]], dtype=float)
rule_arrow_incidence = np.array([[0.5, 1.0]], dtype=float)

def candidate_wrapper():
    try:
        encode_mechanism_arrow_sets(mechanism_rule_counts, rule_arrow_incidence)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_encode_mechanism_arrow_sets(mechanism_rule_counts, rule_arrow_incidence)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
