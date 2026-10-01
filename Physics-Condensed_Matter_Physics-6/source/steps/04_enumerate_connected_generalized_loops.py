"""
Enumerate connected generalized-loop edge masks up to a supplied edge-weight cutoff.

Graph topology determines which finite edge patterns can support nonlocal
corrections.  Limiting their size makes the correction calculation finite while
retaining a controlled sequence of increasingly complex patterns.

The input is a simple undirected graph with canonically oriented edge rows.
Each output row is a binary mask over those rows.  Results are ordered first by
edge count and then by the integer represented by edge-list positions as binary
bit positions.  This canonical order is used by later cluster steps.

Parameters
----------
edges : np.ndarray
    Integer array of shape ``(E,2)`` describing a simple undirected graph.
    Each row must satisfy ``u<v``; rows must be unique.
max_weight : int
    Nonnegative maximum number of selected edges.

Returns
-------
loops : np.ndarray
    Integer array of shape ``(L,E)`` with entries in ``{0,1}``, sorted by
    selected-edge count and then by the edge-position bit mask.  If no loop
    qualifies, the shape is ``(0,E)``.

Raises
------
ValueError
    If ``edges`` is malformed or ``max_weight`` is negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_connected_generalized_loops(
    edges: "np.ndarray", max_weight: int,
) -> "np.ndarray":
    '''Return canonical masks for connected generalized loops within the cutoff.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` describing a simple undirected graph.
        Each row must satisfy ``u<v``; rows must be unique.
    max_weight : int
        Nonnegative maximum number of selected edges.

    Returns
    -------
    loops : np.ndarray
        Integer array of shape ``(L,E)`` with entries in ``{0,1}``, sorted by
        selected-edge count and then by the edge-position bit mask.  If no loop
        qualifies, the shape is ``(0,E)``.

    Raises
    ------
    ValueError
        If ``edges`` is malformed or ``max_weight`` is negative.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_enumerate_connected_generalized_loops(
    edges: "np.ndarray", max_weight: int,
) -> "np.ndarray":
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E,2)")
    max_weight = int(max_weight)
    if max_weight < 0:
        raise ValueError("max_weight must be nonnegative")
    E = edges.shape[0]
    if E == 0:
        return np.empty((0, 0), dtype=np.int64)
    if np.any(edges[:, 0] < 0) or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError("each edge must satisfy 0 <= u < v")
    if len({tuple(row) for row in edges.tolist()}) != E:
        raise ValueError("edges must be unique")

    n_vertices = int(np.max(edges)) + 1
    found = []
    for mask_value in range(1, 1 << E):
        weight = mask_value.bit_count()
        if weight > max_weight:
            continue
        degree = np.zeros(n_vertices, dtype=np.int64)
        adjacency = [[] for _ in range(n_vertices)]
        selected = []
        for e, (u0, v0) in enumerate(edges):
            if (mask_value >> e) & 1:
                u = int(u0)
                v = int(v0)
                selected.append(e)
                degree[u] += 1
                degree[v] += 1
                adjacency[u].append(v)
                adjacency[v].append(u)
        incident = np.flatnonzero(degree)
        if incident.size == 0 or np.any(degree[incident] < 2):
            continue
        start = int(incident[0])
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != incident.size:
            continue
        row = np.zeros(E, dtype=np.int64)
        row[selected] = 1
        found.append((weight, mask_value, row))
    found.sort(key=lambda item: (item[0], item[1]))
    if not found:
        return np.empty((0, E), dtype=np.int64)
    return np.stack([item[2] for item in found], axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return square, tree, figure-eight, and cutoff-boundary graph cases."""
    return [
        {
            "setup": "import numpy as np\nedges=np.array([[0,1],[0,2],[1,3],[2,3]],dtype=int)",
            "call": "enumerate_connected_generalized_loops(edges.copy(),4)",
            "gold_call": "_oracle_enumerate_connected_generalized_loops(edges.copy(),4)",
        },
        {
            "setup": "import numpy as np\nedges=np.array([[0,1],[1,2],[2,3]],dtype=int)",
            "call": "enumerate_connected_generalized_loops(edges.copy(),3)",
            "gold_call": "_oracle_enumerate_connected_generalized_loops(edges.copy(),3)",
        },
        {
            "setup": "import numpy as np\nedges=np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[3,4]],dtype=int)",
            "call": "enumerate_connected_generalized_loops(edges.copy(),6)",
            "gold_call": "_oracle_enumerate_connected_generalized_loops(edges.copy(),6)",
        },
        {
            "setup": "import numpy as np\nedges=np.array([[0,1],[0,2],[1,3],[2,3]],dtype=int)",
            "call": "enumerate_connected_generalized_loops(edges.copy(),3)",
            "gold_call": "_oracle_enumerate_connected_generalized_loops(edges.copy(),3)",
        },
    ]
