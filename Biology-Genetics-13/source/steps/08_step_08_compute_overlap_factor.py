"""
Traversal overlap compares repeated independent work with the distinct nodes touched by a shared pass. For visit sets V_i, the overlap factor is rho(k) = sum_i |V_i| / |union_i V_i|. A batch using word width w propagates q = ceil(k/w) words per node, so rho(k) greater than q is the paper's asymptotic criterion for shared discovery to reduce traversal work.

Inputs

------

visit_matrix: Binary matrix whose rows identify independent visit sets.

word_size: Number of update bits in one packed word.

Returns

-------

overlap: Independent and distinct visit counts, word count, overlap factor, and advantage flag.

Returns
-------
dict containing native integer counts and overlap_factor as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_overlap_factor(visit_matrix: np.ndarray, word_size: int = 32) -> dict:
    """Compute traversal overlap and the packed-word advantage criterion.

    Parameters
    ----------
    visit_matrix : np.ndarray
        Binary matrix with one independent traversal per row.
    word_size : int, optional
        Number of update bits represented by one word.

    Raises
    ------
    ValueError
        If `visit_matrix` is empty, is not two dimensional, is not binary, or
        contains a row with no visits, or if `word_size` is outside [1, 62].

    Returns
    -------
    overlap : dict
        Visit counts, packed word count, overlap factor, and advantage flag.
    """
    return overlap  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_overlap_factor(visit_matrix: np.ndarray, word_size: int = 32) -> dict:
    """Reference implementation."""
    visits = np.asarray(visit_matrix)
    if visits.ndim != 2 or visits.shape[0] == 0 or visits.shape[1] == 0:
        raise ValueError("visit_matrix must be nonempty and two dimensional")
    if not np.all((visits == 0) | (visits == 1)):
        raise ValueError("visit_matrix must be binary")
    if np.any(np.sum(visits, axis=1) == 0):
        raise ValueError("each independent traversal must visit at least one node")
    if not isinstance(word_size, (int, np.integer)) or not 1 <= int(word_size) <= 62:
        raise ValueError("word_size must be an integer in [1, 62]")

    independent_visits = int(np.sum(visits))
    distinct_visits = int(np.count_nonzero(np.any(visits == 1, axis=0)))
    word_count = (visits.shape[0] + int(word_size) - 1) // int(word_size)
    overlap_factor = float(independent_visits / distinct_visits)
    return {
        "independent_visits": independent_visits,
        "distinct_visits": distinct_visits,
        "word_count": int(word_count),
        "overlap_factor": overlap_factor,
        "shared_advantage": int(overlap_factor > word_count),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = '''
def _pack_overlap(o):
    return [
        float(o["independent_visits"]),
        float(o["distinct_visits"]),
        float(o["word_count"]),
        float(o["overlap_factor"]),
        float(o["shared_advantage"]),
    ]
'''
    return [
        {
            "setup": pack + """import numpy as np
visits = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
word_size = 2
""",
            "call": "_pack_overlap(compute_overlap_factor(visits, word_size=word_size))",
            "gold_call": "_pack_overlap(_oracle_compute_overlap_factor(visits, word_size=word_size))",
        },
        {
            "setup": pack + """import numpy as np
visits = np.array([[1]], dtype=np.uint8)
word_size = 1
""",
            "call": "_pack_overlap(compute_overlap_factor(visits, word_size=word_size))",
            "gold_call": "_pack_overlap(_oracle_compute_overlap_factor(visits, word_size=word_size))",
        },
        {
            "setup": """import numpy as np
visits = np.array([[1, 2]], dtype=int)
def run_model():
    try:
        compute_overlap_factor(visits, word_size=2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_overlap_factor(visits, word_size=2)
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
