"""
Batched candidate discovery starts from the union of target carriers and visits the ordered graph upward once. A leaf state is its packed update membership, while an internal state is the wordwise intersection of every child state. A nonzero internal state identifies the updates for which the node reach is compatible and propagates to parents; an all-zero state terminates that branch.

Inputs

------

children: Ordered child identifiers for the fixed graph snapshot.

n_samples: Number of sample leaves.

packed_leaf_memberships: Packed update bits for each sample leaf.

descendant_lists: Reach D(n) for every graph node.

n_updates: Number of updates represented by the packed words.

word_size: Number of active bits per word.

Returns

-------

discovery: Packed node states, per-update candidate lists, candidate sizes, and visited nodes.

Returns
-------
dict of native integer lists containing packed states, candidates, sizes, and visits
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def discover_shared_candidates(
    children: list,
    n_samples: int,
    packed_leaf_memberships: np.ndarray,
    descendant_lists: list,
    n_updates: int,
    word_size: int = 32,
) -> dict:
    """Discover reuse candidates for a batch in one upward traversal.

    Parameters
    ----------
    children : list
        Ordered graph child lists.
    n_samples : int
        Number of sample leaves.
    packed_leaf_memberships : np.ndarray
        Packed leaf states with shape (n_samples, ceil(n_updates / word_size)).
    descendant_lists : list
        Numeric reach list for every node.
    n_updates : int
        Number of target carrier sets in the batch.
    word_size : int, optional
        Number of update bits in each packed word.

    Raises
    ------
    ValueError
        If the graph is empty, `n_samples` does not define its leaf prefix, an
        internal node has no children, a leaf has children, or a child does not
        precede its parent. Also raised if `n_updates` is not positive,
        `word_size` is outside [1, 62], `descendant_lists` has the wrong length,
        the packed array has the wrong shape, a packed value is negative or
        non-integer, or a bit outside the batch is set.

    Returns
    -------
    discovery : dict
        Native numeric node states, candidates, candidate sizes, and visits.
    """
    return discovery  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_discover_shared_candidates(
    children: list,
    n_samples: int,
    packed_leaf_memberships: np.ndarray,
    descendant_lists: list,
    n_updates: int,
    word_size: int = 32,
) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify the leaf prefix")
    if not isinstance(n_updates, int) or n_updates < 1:
        raise ValueError("n_updates must be positive")
    if not isinstance(word_size, int) or not 1 <= word_size <= 62:
        raise ValueError("word_size must be in [1, 62]")
    if len(descendant_lists) != len(children):
        raise ValueError("descendant_lists must have one entry per node")
    for node, node_children in enumerate(children):
        if node < n_samples and list(node_children):
            raise ValueError("sample leaves must have no children")
        if node >= n_samples and not list(node_children):
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or child < 0 or child >= node for child in node_children):
            raise ValueError("every child must precede its parent")

    packed = np.asarray(packed_leaf_memberships)
    n_words = (n_updates + word_size - 1) // word_size
    if packed.shape != (n_samples, n_words):
        raise ValueError("packed_leaf_memberships has the wrong shape")
    if not np.issubdtype(packed.dtype, np.integer) or np.any(packed < 0):
        raise ValueError("packed memberships must be nonnegative integers")
    valid_masks = []
    for word in range(n_words):
        remaining = n_updates - word * word_size
        valid_masks.append((1 << min(word_size, remaining)) - 1)
    for word, mask in enumerate(valid_masks):
        if np.any(np.bitwise_and(packed[:, word].astype(np.int64), ~mask) != 0):
            raise ValueError("packed memberships set a bit outside the batch")

    parents = [[] for _ in children]
    for parent, node_children in enumerate(children):
        for child in node_children:
            parents[child].append(parent)
    for node_parents in parents:
        node_parents.sort()

    states = [[0] * n_words for _ in children]
    active = [False] * len(children)
    for sample in range(n_samples):
        states[sample] = [int(value) for value in packed[sample]]
        active[sample] = any(states[sample])

    candidates = [[] for _ in range(n_updates)]
    candidate_sizes = [[] for _ in range(n_updates)]
    visited = []
    for node in range(len(children)):
        if not active[node]:
            continue
        visited.append(node)
        if node >= n_samples:
            state = list(valid_masks)
            for child in children[node]:
                state = [left & right for left, right in zip(state, states[child])]
            states[node] = state
            if any(state):
                for update in range(n_updates):
                    word = update // word_size
                    offset = update % word_size
                    if state[word] & (1 << offset):
                        candidates[update].append(node)
                        candidate_sizes[update].append(len(descendant_lists[node]))
        if any(states[node]):
            for parent in parents[node]:
                active[parent] = True

    return {
        "node_states": states,
        "candidate_lists": candidates,
        "candidate_sizes": candidate_sizes,
        "visited_nodes": visited,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = '''
def _pack_lol(rows):
    out = [float(len(rows))]
    for row in rows:
        out.append(float(len(row)))
        out.extend(float(v) for v in row)
    return out
def _pack_discovery(d):
    out = []
    for key in ("node_states", "candidate_lists", "candidate_sizes"):
        out.extend(_pack_lol(d[key]))
    out.append(float(len(d["visited_nodes"])))
    out.extend(float(v) for v in d["visited_nodes"])
    return out
'''
    return [
        {
            "setup": pack + """import numpy as np
children = [[], [], [], [], [0, 1], [2, 3], [4, 5]]
n_samples = 4
packed = np.array([[1], [3], [2], [0]], dtype=np.int64)
descendants = [[0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3]]
n_updates = 2
word_size = 2
""",
            "call": "_pack_discovery(discover_shared_candidates(children, n_samples, packed, descendants, n_updates, word_size))",
            "gold_call": "_pack_discovery(_oracle_discover_shared_candidates(children, n_samples, packed, descendants, n_updates, word_size))",
        },
        {
            "setup": pack + """import numpy as np
children = [[]]
n_samples = 1
packed = np.array([[1]], dtype=np.int64)
descendants = [[0]]
n_updates = 1
word_size = 1
""",
            "call": "_pack_discovery(discover_shared_candidates(children, n_samples, packed, descendants, n_updates, word_size))",
            "gold_call": "_pack_discovery(_oracle_discover_shared_candidates(children, n_samples, packed, descendants, n_updates, word_size))",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0, 1]]
n_samples = 2
packed = np.array([[1], [1]], dtype=np.int64)
descendants = [[0], [1], [0, 1]]
n_updates = 1
word_size = 1
packed[0, 0] = 2
def run_model():
    try:
        discover_shared_candidates(children, n_samples, packed, descendants, n_updates, word_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_discover_shared_candidates(children, n_samples, packed, descendants, n_updates, word_size)
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
