"""
Discover compatible graph nodes for a shared update batch.

Batched candidate discovery propagates all mutation compatibilities in one

reverse-topological pass. Leaf state is packed carrier membership. Each

internal word starts from the valid-bit mask for that word and is intersected

with the corresponding word of every child, including a zero-state child. Thus

a node retains update bit i exactly when D(n) is contained in S_i. A zero state

prunes the branch above that node. Words use low-to-high bit order; unused high

bits in every word, including the partial final word, are invalid.

Returns
-------
np.ndarray of shape (n_nodes, ceil(n_updates / word_size)), packed compatibility state as int64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discover_shared_candidates(
    children: list[list[int]],
    n_samples: int,
    packed_leaf_memberships: "np.ndarray",
    word_size: int,
    n_updates: int,
) -> "np.ndarray":
    '''Propagate packed compatibility states through one shared traversal.

    Parameters
    ----------
    children : list[list[int]]
        Child identifiers for every graph node in canonical topological order:
        sample rows are empty, internal rows are nonempty, child identifiers
        are distinct integers, and every child precedes its parent.
    n_samples : int
        Number of leading sample leaves.
    packed_leaf_memberships : np.ndarray
        Nonnegative integer membership words for the sample leaves. Word j
        carries updates j * word_size through (j + 1) * word_size - 1 in
        low-to-high bit order; every unused high bit must be zero.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.
    n_updates : int
        Number of represented mutation updates.

    Raises
    ------
    ValueError
        If n_samples, word_size, or n_updates is invalid; the graph is empty,
        malformed, or not topologically ordered; packed_leaf_memberships has
        the wrong shape, a non-integer dtype (float and bool arrays are
        rejected even when their values are whole numbers), negative values,
        or bits outside the update range.

    Returns
    -------
    state_words : np.ndarray
        Packed compatibility words for every node as an independent int64
        array. The input array is not modified.
    '''
    return state_words  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_discover_shared_candidates(
    children: list[list[int]],
    n_samples: int,
    packed_leaf_memberships: "np.ndarray",
    word_size: int,
    n_updates: int,
) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")
    if not isinstance(n_updates, int) or isinstance(n_updates, bool) or n_updates < 1:
        raise ValueError("n_updates must be a positive integer")
    if not isinstance(children, (list, tuple)) or len(children) <= n_samples:
        raise ValueError("children must contain sample and internal nodes")
    n_words = (n_updates + word_size - 1) // word_size
    packed_leaf_memberships = np.asarray(packed_leaf_memberships)
    if packed_leaf_memberships.shape != (n_samples, n_words):
        raise ValueError("packed_leaf_memberships has an incompatible shape")
    if not np.issubdtype(packed_leaf_memberships.dtype, np.integer):
        raise ValueError("packed_leaf_memberships must contain integers")
    if np.any(packed_leaf_memberships < 0):
        raise ValueError("packed_leaf_memberships cannot be negative")
    packed_leaf_memberships = packed_leaf_memberships.astype(np.int64)
    for word in range(n_words):
        used = min(word_size, n_updates - word * word_size)
        if np.any(packed_leaf_memberships[:, word] >> used):
            raise ValueError("packed_leaf_memberships contains out-of-range bits")

    states = np.zeros((len(children), n_words), dtype=np.int64)
    states[:n_samples] = packed_leaf_memberships
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples:
            if raw_children:
                raise ValueError("sample leaves cannot have children")
            continue
        if not raw_children:
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or isinstance(child, bool) for child in raw_children):
            raise ValueError("child identifiers must be integers")
        if any(child < 0 or child >= node for child in raw_children):
            raise ValueError("children must precede their parent")
        if len(set(raw_children)) != len(raw_children):
            raise ValueError("child identifiers must be distinct within each row")
        state = states[raw_children[0]].copy()
        for child in raw_children[1:]:
            state &= states[child]
        states[node] = state
    return states

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
packed_leaf_memberships = np.array([[1], [1], [3], [2]], dtype=int)
word_size = 2
n_updates = 2
""",
            "call": "discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
            "gold_call": "_oracle_discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
packed_leaf_memberships = np.array([[1], [1]], dtype=int)
word_size = 1
n_updates = 1
""",
            "call": "discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
            "gold_call": "_oracle_discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
packed_leaf_memberships = np.array([[1,0], [1,0]], dtype=int)
word_size = 2
n_updates = 1
def run_model():
    try:
        discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates)
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
    [], [], [], [], [], [], [0,1,2], [2,3,4], [6,7], [8,5]
]
n_samples = 6
packed_leaf_memberships = np.array([
    [2305843009213693955, 7],
    [2305843009213693953, 5],
    [1152921504606846977, 6],
    [1152921504606846977, 6],
    [1152921504606846978, 7],
    [0, 0],
], dtype=np.int64)
word_size = 62
n_updates = 65
def _pin(triple):
    values, dtype_name, unchanged = triple
    flat = [float(v) for row in values for v in row]
    r = np.arange(1.0, len(flat) + 1.0)
    core = float(np.sum(np.sin(0.31*r)*np.array(flat)))
    return core + (1000.0 if unchanged else 0.0) + (0.001 * sum(ord(c) for c in dtype_name))
def run_model():
    before = packed_leaf_memberships.copy()
    result = discover_shared_candidates(
        children, n_samples, packed_leaf_memberships, word_size, n_updates
    )
    return _pin((result.tolist(), result.dtype.name, bool(np.array_equal(before, packed_leaf_memberships))))
def run_oracle():
    before = packed_leaf_memberships.copy()
    result = _oracle_discover_shared_candidates(
        children, n_samples, packed_leaf_memberships, word_size, n_updates
    )
    return _pin((result.tolist(), result.dtype.name, bool(np.array_equal(before, packed_leaf_memberships))))
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        {
            "setup": """import numpy as np
n_samples = 2
packed_leaf_memberships = np.array([[1], [1]], dtype=np.int64)
word_size = 2
n_updates = 2
bad_graphs = [
    [[], [], [0, 2]],
    [[1], [], [0, 1]],
    [[], [], []],
    [[], [], [0, True]],
    [[], [], [0, 0]],
]
def classify_model(graph):
    try:
        discover_shared_candidates(
            graph, n_samples, packed_leaf_memberships, word_size, n_updates
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def classify_oracle(graph):
    try:
        _oracle_discover_shared_candidates(
            graph, n_samples, packed_leaf_memberships, word_size, n_updates
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "[classify_model(graph) for graph in bad_graphs]",
            "gold_call": "[classify_oracle(graph) for graph in bad_graphs]",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
word_size = 62
n_updates = 65
bad_memberships = [
    np.array([[4611686018427387904, 0], [0, 0]], dtype=np.int64),
    np.array([[0, 8], [0, 0]], dtype=np.int64),
    np.array([[-1, 0], [0, 0]], dtype=np.int64),
    np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),
    np.array([[True, False], [False, False]], dtype=bool),
]
def classify_model(packed):
    try:
        discover_shared_candidates(children, n_samples, packed, word_size, n_updates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def classify_oracle(packed):
    try:
        _oracle_discover_shared_candidates(children, n_samples, packed, word_size, n_updates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "[classify_model(packed) for packed in bad_memberships]",
            "gold_call": "[classify_oracle(packed) for packed in bad_memberships]",
        },
        {
            "setup": """import numpy as np
children = [[], [], [], [0], [1,2], [3,4]]
n_samples = 3
packed_leaf_memberships = np.array([
    [255,1],
    [170,1],
    [85,0],
], dtype=np.int64)
word_size = 8
n_updates = 9
""",
            "call": "discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
            "gold_call": "_oracle_discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
        },
        {
            "setup": """import numpy as np
n_samples = 32
children = [[] for _ in range(n_samples)]
level = list(range(n_samples))
while len(level) > 1:
    next_level = []
    for offset in range(0, len(level), 2):
        children.append([level[offset], level[offset + 1]])
        next_level.append(len(children) - 1)
    level = next_level
word_size = 7
n_updates = 19
carrier_sets = [
    set(range(32)),
    set(range(16)),
    set(range(0,32,2)),
    {0,1},
    {0,2},
    set(range(8,24)),
    set(),
    set(range(16,32)),
    {31},
    set(range(0,32,3)),
    set(range(4,12)),
    {8,9,10,11,20,21,22,23},
    set(range(24,32)),
    set(range(8)) | set(range(24,32)),
    {5,6,7},
    {0,1,2,3},
    {12,13,14,15},
    {20,21,22,23},
    {28,29,30,31},
]
packed_leaf_memberships = np.zeros((n_samples, 3), dtype=np.int64)
for update, carriers in enumerate(carrier_sets):
    word = update // word_size
    bit = update % word_size
    for sample in carriers:
        packed_leaf_memberships[sample, word] |= 1 << bit
""",
            "call": "discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
            "gold_call": "_oracle_discover_shared_candidates(children, n_samples, packed_leaf_memberships, word_size, n_updates).tolist()",
        },
    ]
