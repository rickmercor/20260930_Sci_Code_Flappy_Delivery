"""
Independent mutation mapping runs the same upward compatibility recurrence once per target. A carrier leaf is compatible, an internal node is compatible only when every child is compatible, and only compatible nodes propagate to their parents. A reached internal node is counted as visited even when its failed compatibility test terminates that branch.

Inputs

------

children: Ordered child lists for the updated graph.

n_samples: Number of sample leaves.

carrier_matrix: Binary audit carrier sets, one row per independent traversal.

Returns

-------

traces: Visit matrix, visited counts, and compatible internal candidates.

Returns
-------
dict with a uint8 visit matrix and native integer lists for counts and candidates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def trace_independent_visits(children: list, n_samples: int, carrier_matrix: np.ndarray) -> dict:
    """Trace independent reuse-aware visits for several target carrier sets.

    Parameters
    ----------
    children : list
        Ordered child lists for the graph.
    n_samples : int
        Number of sample leaves.
    carrier_matrix : np.ndarray
        Binary target matrix with one row per independent traversal.

    Raises
    ------
    ValueError
        If the graph is empty, `n_samples` does not define its leaf prefix, a
        leaf has children, an internal node has no children, or a child does not
        precede its parent. Also raised if `carrier_matrix` has the wrong shape,
        is not binary, or contains an empty audit carrier set.

    Returns
    -------
    traces : dict
        Visit matrix, row counts, and per-target candidates.
    """
    return traces  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_trace_independent_visits(children: list, n_samples: int, carrier_matrix: np.ndarray) -> dict:
    """Reference implementation."""
    if not isinstance(children, list) or len(children) == 0:
        raise ValueError("children must be a nonempty list")
    if not isinstance(n_samples, int) or not 1 <= n_samples <= len(children):
        raise ValueError("n_samples must identify the leaf prefix")
    for node, node_children in enumerate(children):
        if node < n_samples and list(node_children):
            raise ValueError("sample leaves must have no children")
        if node >= n_samples and not list(node_children):
            raise ValueError("internal nodes must have children")
        if any(not isinstance(child, int) or child < 0 or child >= node for child in node_children):
            raise ValueError("every child must precede its parent")

    carriers = np.asarray(carrier_matrix)
    if carriers.ndim != 2 or carriers.shape[1] != n_samples or carriers.shape[0] == 0:
        raise ValueError("carrier_matrix has the wrong shape")
    if not np.all((carriers == 0) | (carriers == 1)):
        raise ValueError("carrier_matrix must be binary")
    if np.any(np.sum(carriers, axis=1) == 0):
        raise ValueError("audit carrier sets must be nonempty")

    parents = [[] for _ in children]
    for parent, node_children in enumerate(children):
        for child in node_children:
            parents[child].append(parent)
    for node_parents in parents:
        node_parents.sort()

    visit_matrix = np.zeros((carriers.shape[0], len(children)), dtype=np.uint8)
    candidate_lists = []
    for update in range(carriers.shape[0]):
        compatible = [False] * len(children)
        active = [False] * len(children)
        for sample in range(n_samples):
            compatible[sample] = bool(carriers[update, sample])
            active[sample] = compatible[sample]
        candidates = []
        for node in range(len(children)):
            if not active[node]:
                continue
            visit_matrix[update, node] = 1
            if node >= n_samples:
                compatible[node] = all(compatible[child] for child in children[node])
                if compatible[node]:
                    candidates.append(node)
            if compatible[node]:
                for parent in parents[node]:
                    active[parent] = True
        candidate_lists.append(candidates)

    return {
        "visit_matrix": visit_matrix,
        "visited_counts": np.sum(visit_matrix, axis=1).astype(int).tolist(),
        "candidate_lists": candidate_lists,
    }

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
children = [[], [], [], [], [0, 1], [2, 3], [4, 5]]
n_samples = 4
carriers = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
""",
            "call": "trace_independent_visits(children, n_samples, carriers)['visit_matrix'].tolist()",
            "gold_call": "_oracle_trace_independent_visits(children, n_samples, carriers)['visit_matrix'].tolist()",
        },
        {
            "setup": """import numpy as np
children = [[]]
n_samples = 1
carriers = np.array([[1]], dtype=np.uint8)
""",
            "call": "trace_independent_visits(children, n_samples, carriers)['visited_counts']",
            "gold_call": "_oracle_trace_independent_visits(children, n_samples, carriers)['visited_counts']",
        },
        {
            "setup": """import numpy as np
children = [[], [], [0, 1]]
n_samples = 2
carriers = np.array([[0, 0]], dtype=np.uint8)
def run_model():
    try:
        trace_independent_visits(children, n_samples, carriers)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_trace_independent_visits(children, n_samples, carriers)
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
