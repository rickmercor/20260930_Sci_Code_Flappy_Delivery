"""
Apply a batch of attachment plans to the graph snapshot.

A batch applies plans made against a fixed read-only snapshot. Exact matches

attach to existing nodes. Every nonempty child mask appends a distinct mutation

node in row order, even when an earlier row appended an equivalent node,

because intra-batch structure was absent during discovery.

Returns
-------
tuple of a square binary int64 adjacency array and an int64 attachment-node array of length n_updates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_snapshot_batch(
    children: list[list[int]], n_samples: int, attachment_plans: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    '''Apply fixed-snapshot attachment plans in deterministic row order.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered discovery-snapshot graph.
    n_samples : int
        Number of leading sample leaves.
    attachment_plans : np.ndarray
        Exact node identifier and child-node bitmask for each update. A row
        equal to (-1, 0) appends no node and retains attachment identifier -1.

    Raises
    ------
    ValueError
        If n_samples or the graph is invalid, the snapshot has more than 62
        nodes, attachment_plans is not a nonempty integer array with two columns,
        a plan combines exact reuse with children, or a plan refers to a node
        outside the discovery snapshot.

    Returns
    -------
    edited_adjacency : np.ndarray
        Binary parent-by-child adjacency matrix after the batch.
    attachment_nodes : np.ndarray
        Node identifier assigned to each replacement mutation.
    '''
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_apply_snapshot_batch(
    children: list[list[int]], n_samples: int, attachment_plans: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    if not isinstance(children, (list, tuple)) or not (1 <= len(children) <= 62):
        raise ValueError("children must describe from 1 through 62 snapshot nodes")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < len(children)):
        raise ValueError("n_samples is invalid")
    normalized_children = []
    for node, raw_children in enumerate(children):
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError("every child row must be a sequence")
        if node < n_samples and raw_children:
            raise ValueError("sample leaves cannot have children")
        if node >= n_samples and not raw_children:
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or isinstance(child, bool) or child < 0 or child >= node for child in raw_children):
            raise ValueError("children must be integer identifiers preceding their parent")
        normalized_children.append(list(raw_children))

    attachment_plans = np.asarray(attachment_plans)
    if attachment_plans.ndim != 2 or attachment_plans.shape[0] < 1 or attachment_plans.shape[1] != 2:
        raise ValueError("attachment_plans must be a nonempty array with two columns")
    if not np.issubdtype(attachment_plans.dtype, np.integer):
        raise ValueError("attachment_plans must contain integers")
    attachment_plans = attachment_plans.astype(np.int64)
    snapshot_nodes = len(normalized_children)
    attachments = np.full(attachment_plans.shape[0], -1, dtype=np.int64)

    for update, (exact_raw, child_mask_raw) in enumerate(attachment_plans):
        exact = int(exact_raw)
        child_mask = int(child_mask_raw)
        if child_mask < 0:
            raise ValueError("child-node masks cannot be negative")
        if child_mask >> snapshot_nodes:
            raise ValueError("a child-node mask refers outside the discovery snapshot")
        if exact >= 0:
            if exact >= snapshot_nodes or child_mask != 0:
                raise ValueError("exact reuse must name one snapshot node and no children")
            attachments[update] = exact
            continue
        if exact != -1:
            raise ValueError("exact node identifiers must be -1 or a snapshot node")
        if child_mask == 0:
            continue
        child_ids = [node for node in range(snapshot_nodes) if (child_mask >> node) & 1]
        normalized_children.append(child_ids)
        attachments[update] = len(normalized_children) - 1

    adjacency = np.zeros((len(normalized_children), len(normalized_children)), dtype=np.int64)
    for parent, child_ids in enumerate(normalized_children):
        adjacency[parent, child_ids] = 1
    return adjacency, attachments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
children = [[], [], [], [], [0,1]]
n_samples = 4
attachment_plans = np.array([[4,0], [-1,20], [-1,20]], dtype=int)
def _pin(pair):
    adjacency, attachments = pair
    flat = [float(v) for row in adjacency for v in row] + [float(v) for v in attachments]
    r = np.arange(1.0, len(flat) + 1.0)
    return float(np.sum(np.sin(0.31*r)*np.array(flat)) + np.sum(np.cos(0.13*r)*np.array(flat)**2))
""",
            "call": "_pin([value.tolist() for value in apply_snapshot_batch(children, n_samples, attachment_plans)])",
            "gold_call": "_pin([value.tolist() for value in _oracle_apply_snapshot_batch(children, n_samples, attachment_plans)])",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
attachment_plans = np.array([[2,0]], dtype=int)
def _pin(pair):
    adjacency, attachments = pair
    flat = [float(v) for row in adjacency for v in row] + [float(v) for v in attachments]
    r = np.arange(1.0, len(flat) + 1.0)
    return float(np.sum(np.sin(0.31*r)*np.array(flat)) + np.sum(np.cos(0.13*r)*np.array(flat)**2))
""",
            "call": "_pin([value.tolist() for value in apply_snapshot_batch(children, n_samples, attachment_plans)])",
            "gold_call": "_pin([value.tolist() for value in _oracle_apply_snapshot_batch(children, n_samples, attachment_plans)])",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0,1]]
n_samples = 2
attachment_plans = np.array([[3,0]], dtype=int)
def run_model():
    try:
        apply_snapshot_batch(children, n_samples, attachment_plans)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_apply_snapshot_batch(children, n_samples, attachment_plans)
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
    [], [], [], [], [0,1], [2,3], [4,2], [1,3]
]
n_samples = 4
duplicate_reach_mask = (1 << 4) | (1 << 2)
attachment_plans = np.array([
    [4,0],
    [-1,duplicate_reach_mask],
    [-1,0],
    [-1,duplicate_reach_mask],
    [6,0],
    [-1,(1 << 0) | (1 << 1) | (1 << 2)],
], dtype=np.int64)
def _pin(pair):
    adjacency, attachments = pair
    flat = [float(v) for row in adjacency for v in row] + [float(v) for v in attachments]
    r = np.arange(1.0, len(flat) + 1.0)
    return float(np.sum(np.sin(0.31*r)*np.array(flat)) + np.sum(np.cos(0.13*r)*np.array(flat)**2))
""",
            "call": "_pin([value.tolist() for value in apply_snapshot_batch(children, n_samples, attachment_plans)])",
            "gold_call": "_pin([value.tolist() for value in _oracle_apply_snapshot_batch(children, n_samples, attachment_plans)])",
        },
    ]
