"""
Summarize non-isolated connected components and their identity colliders.

 

Nodes are indexed by rows of signal_metadata. Its columns are [dataset ID, trait ID, signal ID], and complete triplets must be unique. A component is a collider exactly when two of its distinct signal nodes share the same ordered (dataset ID, trait ID) pair. Exclude isolated nodes. Return [minimum node index, node count, edge count, collider flag] sorted by minimum node index.

Connected components summarize linked signals; repeated biological identity within a component may indicate ambiguous clustering.

Returns
-------
component_summary : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def summarize_collider_components(
    edges: "np.ndarray",
    signal_metadata: "np.ndarray",
) -> "np.ndarray":
    """
    Return canonical summaries of non-isolated graph components.
 
    Parameters
    ----------
    edges : np.ndarray
        Float edge rows [i, j, weight].
    signal_metadata : np.ndarray
        Integer node rows [dataset ID, trait ID, signal ID].
 
    Returns
    -------
    np.ndarray
        Integer array with shape (n_components, 4).
 
    Raises
    ------
    ValueError
        If graph or metadata inputs are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_summarize_collider_components(
    edges: "np.ndarray",
    signal_metadata: "np.ndarray",
) -> "np.ndarray":
    e = np.asarray(edges, dtype=float)
    m_raw = np.asarray(signal_metadata)
    if e.ndim != 2 or e.shape[1] != 3 or not np.all(np.isfinite(e)):
        raise ValueError("edges must be a finite array with three columns")
    if m_raw.ndim != 2 or m_raw.shape[0] < 1 or m_raw.shape[1] != 3 or m_raw.dtype.kind not in "iu":
        raise ValueError("signal_metadata must be a non-empty integer array with three columns")
    m = m_raw
    if len({tuple(row) for row in m.tolist()}) != m.shape[0]:
        raise ValueError("complete signal identities must be unique")
    if e.shape[0] == 0:
        return np.empty((0, 4), dtype=int)
    if np.any(e[:, 2] < 0.0) or np.any(e[:, 2] > 1.0):
        raise ValueError("edge weights must lie in [0,1]")
    if not np.all(e[:, :2] == np.floor(e[:, :2])):
        raise ValueError("edge endpoints must be integer-valued")
    ij = e[:, :2].astype(int)
    if np.any(ij < 0) or np.any(ij >= m.shape[0]) or np.any(ij[:, 0] == ij[:, 1]):
        raise ValueError("edge endpoints are invalid")
    canonical = np.sort(ij, axis=1)
    if len({tuple(row) for row in canonical.tolist()}) != canonical.shape[0]:
        raise ValueError("duplicate undirected edges are not allowed")
    adjacency = {}
    for i, j in canonical:
        adjacency.setdefault(int(i), set()).add(int(j))
        adjacency.setdefault(int(j), set()).add(int(i))
    seen = set()
    rows = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        node_set = set(nodes)
        edge_count = sum(int(i) in node_set and int(j) in node_set for i, j in canonical)
        identities = [tuple(m[node, :2]) for node in nodes]
        collider = int(len(set(identities)) < len(identities))
        rows.append((min(nodes), len(nodes), int(edge_count), collider))
    rows.sort(key=lambda row: row[0])
    return np.asarray(rows, dtype=int).reshape((-1, 4))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
e=np.array([[0.,1.,.9],[1.,2.,.9],[2.,3.,.9]])
m=np.array([[7,8,10],[9,8,11],[6,5,12],[7,8,13]],dtype=int)""",
            "call": "summarize_collider_components(e,m)",
            "gold_call": "_oracle_summarize_collider_components(e,m)",
        },
        {
            "setup": """import numpy as np
e=np.array([[3.,2.,.8],[1.,0.,.9],[2.,0.,.7]])
m=np.array([[1,1,10],[2,2,11],[3,3,12],[4,4,13],[1,1,14]],dtype=int)""",
            "call": "summarize_collider_components(e,m)",
            "gold_call": "_oracle_summarize_collider_components(e,m)",
        },
        {
            "setup": """import numpy as np
e=np.empty((0,3)); m=np.array([[1,2,3],[1,2,4]],dtype=int)""",
            "call": "summarize_collider_components(e,m)",
            "gold_call": "_oracle_summarize_collider_components(e,m)",
        },
        {
            "setup": """import numpy as np
e=np.array([[0.,1.,.9]]); m=np.array([[1,2,3],[1,2,3]],dtype=int)
def candidate_code():
    try: summarize_collider_components(e,m); return 0.0
    except ValueError: return 1.0
def oracle_code():
    try: _oracle_summarize_collider_components(e,m); return 0.0
    except ValueError: return 1.0""",
            "call": "candidate_code()",
            "gold_call": "oracle_code()",
        },
    ]
