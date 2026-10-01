"""
Audit independent per-mutation graph discovery.

Independent discovery starts from one mutation's carrier leaves and proceeds

upward only through compatible nodes. A reached internal node is counted before

its compatibility test; a compatible node is also counted as a candidate and

activates its parents. Packing each visit set as a node bitmask makes both

per-mutation work and the shared union exact.

Returns
-------
np.ndarray of shape (n_updates, 2), visited-node mask and internal candidate count as int64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def audit_independent_discovery(
    snapshot_adjacency: "np.ndarray", n_samples: int, update_carriers: "np.ndarray"
) -> "np.ndarray":
    '''Audit independent traversal visits and candidate counts.

    Parameters
    ----------
    snapshot_adjacency : np.ndarray
        Binary parent-by-child adjacency matrix of the read-only discovery
        snapshot, in topological node order.
    n_samples : int
        Number of leading sample leaves.
    update_carriers : np.ndarray
        Binary carrier rows for the audited mutations.

    Raises
    ------
    ValueError
        If snapshot_adjacency is not a square nonempty binary matrix with at most
        62 nodes, n_samples is invalid, sample rows have children, an internal
        node has no children or a nonpreceding child, or update_carriers has an
        incompatible shape or non-binary entries.

    Returns
    -------
    audit_counts : np.ndarray
        Visited-node bitmask and compatible internal candidate count per update.
    '''
    return audit_counts  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_audit_independent_discovery(
    snapshot_adjacency: "np.ndarray", n_samples: int, update_carriers: "np.ndarray"
) -> "np.ndarray":
    """Reference implementation."""
    snapshot_adjacency = np.asarray(snapshot_adjacency)
    if snapshot_adjacency.ndim != 2 or snapshot_adjacency.shape[0] < 1:
        raise ValueError("snapshot_adjacency must be a nonempty matrix")
    if snapshot_adjacency.shape[0] != snapshot_adjacency.shape[1] or snapshot_adjacency.shape[0] > 62:
        raise ValueError("snapshot_adjacency must be square with at most 62 nodes")
    if not np.all((snapshot_adjacency == 0) | (snapshot_adjacency == 1)):
        raise ValueError("snapshot_adjacency must be binary")
    n_nodes = snapshot_adjacency.shape[0]
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or not (1 <= n_samples < n_nodes):
        raise ValueError("n_samples is invalid")
    snapshot_adjacency = snapshot_adjacency.astype(np.int64)
    child_rows = []
    for node in range(n_nodes):
        child_ids = np.flatnonzero(snapshot_adjacency[node]).tolist()
        if node < n_samples and child_ids:
            raise ValueError("sample leaves cannot have children")
        if node >= n_samples and not child_ids:
            raise ValueError("internal nodes must have children")
        if any(child >= node for child in child_ids):
            raise ValueError("children must precede their parent")
        child_rows.append(child_ids)

    update_carriers = np.asarray(update_carriers)
    if update_carriers.ndim != 2 or update_carriers.shape[0] < 1 or update_carriers.shape[1] != n_samples:
        raise ValueError("update_carriers has an incompatible shape")
    if not np.all((update_carriers == 0) | (update_carriers == 1)):
        raise ValueError("update_carriers must be binary")
    update_carriers = update_carriers.astype(np.int64)

    audit = np.zeros((update_carriers.shape[0], 2), dtype=np.int64)
    for update, carrier_row in enumerate(update_carriers):
        compatible = np.zeros(n_nodes, dtype=bool)
        visited_mask = 0
        for sample in np.flatnonzero(carrier_row):
            sample = int(sample)
            compatible[sample] = True
            visited_mask |= 1 << sample
        candidate_count = 0
        for node in range(n_samples, n_nodes):
            child_ids = child_rows[node]
            if any(compatible[child] for child in child_ids):
                visited_mask |= 1 << node
                compatible[node] = all(compatible[child] for child in child_ids)
                if compatible[node]:
                    candidate_count += 1
        audit[update] = (np.int64(visited_mask), np.int64(candidate_count))
    return audit

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
snapshot_adjacency = np.zeros((7,7), dtype=int)
snapshot_adjacency[4,[0,1]] = 1
snapshot_adjacency[5,[2,3]] = 1
snapshot_adjacency[6,[4,2]] = 1
n_samples = 4
update_carriers = np.array([[1,1,1,0], [0,0,1,1]], dtype=int)
""",
            "call": "audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers).tolist()",
            "gold_call": "_oracle_audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers).tolist()",
        },
        {
            "setup": """import numpy as np
snapshot_adjacency = np.zeros((3,3), dtype=int)
snapshot_adjacency[2,[0,1]] = 1
n_samples = 2
update_carriers = np.array([[1,0]], dtype=int)
""",
            "call": "audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers).tolist()",
            "gold_call": "_oracle_audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers).tolist()",
        },
        {
            "setup": """import numpy as np
snapshot_adjacency = np.zeros((3,3), dtype=int)
snapshot_adjacency[1,2] = 1
n_samples = 1
update_carriers = np.array([[1]], dtype=int)
def run_model():
    try:
        audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers)
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
    [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7],
    [8,9], [8,10], [8,11], [8,12], [8,13], [8,14],
    [15,16,17,18,19,20],
]
snapshot_adjacency = np.zeros((len(children), len(children)), dtype=np.int64)
for parent, child_ids in enumerate(children):
    snapshot_adjacency[parent, child_ids] = 1
n_samples = 8
update_carriers = np.array([
    [1,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1],
], dtype=np.int64)
""",
            "call": "audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers).tolist()",
            "gold_call": "_oracle_audit_independent_discovery(snapshot_adjacency, n_samples, update_carriers).tolist()",
        },
    ]
