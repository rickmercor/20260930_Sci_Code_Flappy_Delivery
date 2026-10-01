"""
Plan mutation attachments against a read-only graph snapshot.

Mutation application consumes candidates discovered on one read-only graph

snapshot. An exact-reach node is reused directly; when several nodes reach

exactly the carrier set, the smallest node identifier is reused. Otherwise

compatible internal nodes are considered by decreasing descendant count, with

equal counts taken in increasing node-identifier order, disjoint reaches

are retained, and uncovered carriers attach as leaves. Each plan stores an

exact node identifier or a bitmask of snapshot child identifiers.

Returns
-------
np.ndarray of shape (n_updates, 2), exact node identifier and child-node bitmask as int64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def plan_snapshot_attachments(
    children: list[list[int]],
    n_samples: int,
    reach_encoding: "np.ndarray",
    state_words: "np.ndarray",
    update_carriers: "np.ndarray",
    word_size: int,
) -> "np.ndarray":
    '''Plan reuse and new-node children from one discovery snapshot.

    Among several nodes whose reach equals the carrier set, the smallest
    identifier is reused. Candidates are visited by decreasing reach size,
    breaking ties by increasing node identifier.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered discovery-snapshot graph.
    n_samples : int
        Number of leading sample leaves.
    reach_encoding : np.ndarray
        Reach mask, reach size, and adaptive storage flag for each node.
    state_words : np.ndarray
        Packed compatibility state for each node.
    update_carriers : np.ndarray
        Binary carrier rows for the update batch.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.

    Raises
    ------
    ValueError
        If the snapshot has more than 62 nodes; n_samples or word_size is
        invalid; the graph is malformed; any numerical input has an incompatible
        shape, type, or value; or reach and compatibility data contradict the
        update carriers.

    Returns
    -------
    attachment_plans : np.ndarray
        Rows containing exact node id or -1, followed by a child-node bitmask.
    '''
    return attachment_plans  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_plan_snapshot_attachments(
    children: list[list[int]],
    n_samples: int,
    reach_encoding: "np.ndarray",
    state_words: "np.ndarray",
    update_carriers: "np.ndarray",
    word_size: int,
) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(children, (list, tuple)) or not (1 <= len(children) <= 62):
        raise ValueError("children must describe from 1 through 62 snapshot nodes")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < len(children)):
        raise ValueError("n_samples is invalid")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples and raw_children:
            raise ValueError("sample leaves cannot have children")
        if node >= n_samples and not raw_children:
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or isinstance(child, bool) or child < 0 or child >= node for child in raw_children):
            raise ValueError("children must be integer identifiers preceding their parent")

    reach_encoding = np.asarray(reach_encoding)
    state_words = np.asarray(state_words)
    update_carriers = np.asarray(update_carriers)
    if reach_encoding.shape != (len(children), 3) or not np.issubdtype(reach_encoding.dtype, np.integer):
        raise ValueError("reach_encoding has an incompatible shape or type")
    if update_carriers.ndim != 2 or update_carriers.shape[1] != n_samples or update_carriers.shape[0] < 1:
        raise ValueError("update_carriers has an incompatible shape")
    if not np.all((update_carriers == 0) | (update_carriers == 1)):
        raise ValueError("update_carriers must be binary")
    n_updates = update_carriers.shape[0]
    n_words = (n_updates + word_size - 1) // word_size
    if state_words.shape != (len(children), n_words) or not np.issubdtype(state_words.dtype, np.integer):
        raise ValueError("state_words has an incompatible shape or type")
    if np.any(state_words < 0) or np.any(reach_encoding[:, :2] < 0):
        raise ValueError("reach and state encodings cannot be negative")
    if not np.all((reach_encoding[:, 2] == 0) | (reach_encoding[:, 2] == 1)):
        raise ValueError("adaptive storage flags must be binary")

    reach_encoding = reach_encoding.astype(np.int64)
    state_words = state_words.astype(np.int64)
    update_carriers = update_carriers.astype(np.int64)
    plans = np.full((n_updates, 2), (-1, 0), dtype=np.int64)
    for update in range(n_updates):
        target_mask = 0
        for sample in np.flatnonzero(update_carriers[update]):
            target_mask |= 1 << int(sample)
        word = update // word_size
        bit = update % word_size
        candidates = []
        for node in range(n_samples, len(children)):
            compatible = bool((int(state_words[node, word]) >> bit) & 1)
            reach_mask = int(reach_encoding[node, 0])
            if compatible != ((reach_mask & ~target_mask) == 0):
                raise ValueError("reach and compatibility data contradict update carriers")
            if compatible:
                candidates.append(node)

        if target_mask == 0:
            continue
        if target_mask.bit_count() == 1:
            plans[update, 0] = int(target_mask.bit_length() - 1)
            continue
        exact = [node for node in candidates if int(reach_encoding[node, 0]) == target_mask]
        if exact:
            plans[update, 0] = min(exact)
            continue

        covered = 0
        selected_nodes = 0
        ordered = sorted(candidates, key=lambda node: (-int(reach_encoding[node, 1]), node))
        for node in ordered:
            reach_mask = int(reach_encoding[node, 0])
            if covered & reach_mask == 0:
                covered |= reach_mask
                selected_nodes |= 1 << node
        uncovered = target_mask & ~covered
        selected_nodes |= uncovered
        plans[update, 1] = np.int64(selected_nodes)
    return plans

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
children = [[], [], [], [], [0,1], [2,3]]
n_samples = 4
reach_encoding = np.array([[1,1,0],[2,1,0],[4,1,0],[8,1,0],[3,2,1],[12,2,1]], dtype=int)
state_words = np.array([[1],[1],[3],[2],[1],[2]], dtype=int)
update_carriers = np.array([[1,1,1,0],[0,0,1,1]], dtype=int)
word_size = 2
""",
            "call": "plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size).tolist()",
            "gold_call": "_oracle_plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size).tolist()",
        },
        {
            "setup": """import numpy as np
children = [[], [], [], [], [0,1], [2,3]]
n_samples = 4
reach_encoding = np.array([[1,1,0],[2,1,0],[4,1,0],[8,1,0],[3,2,1],[12,2,1]], dtype=int)
state_words = np.array([[0],[0],[0],[1],[0],[0]], dtype=int)
update_carriers = np.array([[0,0,0,1]], dtype=int)
word_size = 1
""",
            "call": "plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size).tolist()",
            "gold_call": "_oracle_plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size).tolist()",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
reach_encoding = np.zeros((2,3), dtype=int)
state_words = np.array([[1],[1],[1]], dtype=int)
update_carriers = np.array([[1,1]], dtype=int)
word_size = 1
def run_model():
    try:
        plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size)
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
children = [
    [], [], [], [], [], [], [], [],
    [0,1,2], [0,1], [2,3], [3,4], [4,5], [6,7],
    [8,3], [10,12], [0,1],
]
n_samples = 8
reach_masks = [1,2,4,8,16,32,64,128,7,3,12,24,48,192,15,60,3]
reach_encoding = np.array([
    [mask, int(mask).bit_count(), int(node >= n_samples)]
    for node, mask in enumerate(reach_masks)
], dtype=np.int64)
update_carriers = np.array([
    [1,1,1,1,1,1,1,0],
    [1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,0,0],
], dtype=np.int64)
word_size = 3
state_words = np.zeros((len(children), 2), dtype=np.int64)
for update, carrier_row in enumerate(update_carriers):
    target_mask = sum(1 << sample for sample in np.flatnonzero(carrier_row))
    word = update // word_size
    bit = update % word_size
    for node, reach_mask in enumerate(reach_masks):
        if reach_mask & ~target_mask == 0:
            state_words[node, word] |= 1 << bit
""",
            "call": "plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size).tolist()",
            "gold_call": "_oracle_plan_snapshot_attachments(children, n_samples, reach_encoding, state_words, update_carriers, word_size).tolist()",
        },
    ]
