"""
Compute the shared-batch traversal work ratio.

The independent discovery work model sums visited nodes and generated

candidates across mutations. The batched model charges q packed-word

operations per distinct visited node and the same total candidates, where q is

the ceiling of mutation count divided by word size. Their quotient measures

batched work relative to independent work.

Returns
-------
float, the batched-to-independent candidate-discovery work ratio as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_batch_work_ratio(
    audit_counts: "np.ndarray", n_nodes: int, word_size: int
) -> float:
    '''Compute the candidate-aware batched-to-independent work ratio.

    Parameters
    ----------
    audit_counts : np.ndarray
        Visited-node bitmask and candidate count for each mutation.
    n_nodes : int
        Number of nodes in the audited graph, from 1 through 62.
    word_size : int
        Number of usable update bits per packed word, from 1 through 62.

    Raises
    ------
    ValueError
        If audit_counts is not a nonempty integer array with two columns, a
        visited mask or candidate count is negative, a visited mask refers
        outside the graph, n_nodes or word_size is invalid, or independent work
        is zero.

    Returns
    -------
    work_ratio : float
        Batched candidate-discovery work divided by independent work.
    '''
    return work_ratio  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_batch_work_ratio(
    audit_counts: "np.ndarray", n_nodes: int, word_size: int
) -> float:
    """Reference implementation."""
    audit_counts = np.asarray(audit_counts)
    if audit_counts.ndim != 2 or audit_counts.shape[0] < 1 or audit_counts.shape[1] != 2:
        raise ValueError("audit_counts must be a nonempty array with two columns")
    if not np.issubdtype(audit_counts.dtype, np.integer):
        raise ValueError("audit_counts must contain integers")
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or not (1 <= n_nodes <= 62):
        raise ValueError("n_nodes must be an integer from 1 through 62")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    audit_counts = audit_counts.astype(np.int64)
    if np.any(audit_counts < 0):
        raise ValueError("visited masks and candidate counts cannot be negative")
    if np.any(audit_counts[:, 0] >> n_nodes):
        raise ValueError("a visited mask refers outside the graph")

    union_mask = 0
    independent_work = 0
    total_candidates = 0
    for visited_mask_raw, candidate_count_raw in audit_counts:
        visited_mask = int(visited_mask_raw)
        candidate_count = int(candidate_count_raw)
        union_mask |= visited_mask
        independent_work += visited_mask.bit_count() + candidate_count
        total_candidates += candidate_count
    if independent_work == 0:
        raise ValueError("independent work must be positive")
    n_updates = audit_counts.shape[0]
    packed_words = (n_updates + word_size - 1) // word_size
    batched_work = packed_words * union_mask.bit_count() + total_candidates
    return float(batched_work / independent_work)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
audit_counts = np.array([
    [42205952,1], [694272063,5], [560037948,3], [920534976,7],
    [1056768,0], [1072680959,14], [131136,0], [694272063,5],
    [1072664572,11], [3166211,1],
], dtype=np.int64)
n_nodes = 30
word_size = 4
""",
            "call": "compute_batch_work_ratio(audit_counts, n_nodes, word_size)",
            "gold_call": "_oracle_compute_batch_work_ratio(audit_counts, n_nodes, word_size)",
        },
        {
            "setup": """import numpy as np
audit_counts = np.array([[1,0]], dtype=np.int64)
n_nodes = 1
word_size = 1
""",
            "call": "compute_batch_work_ratio(audit_counts, n_nodes, word_size)",
            "gold_call": "_oracle_compute_batch_work_ratio(audit_counts, n_nodes, word_size)",
        },
        {
            "setup": """import numpy as np
audit_counts = np.array([[1,0]], dtype=np.int64)
n_nodes = 1
word_size = 0
def run_model():
    try:
        compute_batch_work_ratio(audit_counts, n_nodes, word_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_batch_work_ratio(audit_counts, n_nodes, word_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        {
            "setup": """import numpy as np
core_mask = (1 << 10) - 1
visit_masks = [
    core_mask | (1 << (10 + update)) | (1 << (10 + ((update + 3) % 10)))
    for update in range(9)
]
candidate_counts = [0,1,3,0,5,2,8,1,4]
audit_counts = np.array(list(zip(visit_masks, candidate_counts)), dtype=np.int64)
n_nodes = 20
word_size = 4
""",
            "call": "compute_batch_work_ratio(audit_counts, n_nodes, word_size)",
            "gold_call": "_oracle_compute_batch_work_ratio(audit_counts, n_nodes, word_size)",
        },
    ]
