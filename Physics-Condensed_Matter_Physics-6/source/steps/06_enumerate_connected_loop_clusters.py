"""
Enumerate connected loop-multiset clusters through a total edge-weight cutoff.

The logarithm of a partition function organizes corrections by correlations
between loop excitations.  At finite order, repeated occurrences of a loop can
affect that logarithmic correction alongside interactions between different
loops.

Each cluster is represented by a multiplicity row over the supplied canonical
loop list.  Rows are ordered by total edge weight, then number of loop copies,
then the lexicographic expanded loop-index tuple.  Coefficients and total
weights are returned in the same row order.

Parameters
----------
edges : np.ndarray
    Integer array of shape ``(E,2)`` in canonical orientation.
loops : np.ndarray
    Integer binary array of shape ``(L,E)`` containing connected loops in
    canonical order.
max_weight : int
    Nonnegative upper bound on the sum of loop edge counts in a cluster.

Returns
-------
multiplicities : np.ndarray
    Integer array of shape ``(C,L)``.  Entry ``[c,l]`` is the number of
    copies of loop ``l`` in cluster ``c``.
ursell_coefficients : np.ndarray
    Float array of shape ``(C,)`` in cluster-row order.
cluster_weights : np.ndarray
    Integer array of shape ``(C,)`` giving total loop-edge weight.

Raises
------
ValueError
    If shapes are inconsistent, loop masks are not binary, or
    ``max_weight`` is negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_connected_loop_clusters(
    edges: "np.ndarray", loops: "np.ndarray", max_weight: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    '''Return multiplicities, connected-cluster coefficients, and total weights.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` in canonical orientation.
    loops : np.ndarray
        Integer binary array of shape ``(L,E)`` containing connected loops in
        canonical order.
    max_weight : int
        Nonnegative upper bound on the sum of loop edge counts in a cluster.

    Returns
    -------
    multiplicities : np.ndarray
        Integer array of shape ``(C,L)``.  Entry ``[c,l]`` is the number of
        copies of loop ``l`` in cluster ``c``.
    ursell_coefficients : np.ndarray
        Float array of shape ``(C,)`` in cluster-row order.
    cluster_weights : np.ndarray
        Integer array of shape ``(C,)`` giving total loop-edge weight.

    Raises
    ------
    ValueError
        If shapes are inconsistent, loop masks are not binary, or
        ``max_weight`` is negative.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _step06_interaction_graph(loop_indices, vertex_sets):
    m = len(loop_indices)
    graph_edges = []
    adjacency = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if loop_indices[i] == loop_indices[j] or (vertex_sets[loop_indices[i]] & vertex_sets[loop_indices[j]]):
                graph_edges.append((i, j))
                adjacency[i].append(j)
                adjacency[j].append(i)
    if m == 1:
        return True, graph_edges
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == m, graph_edges


def _step06_ursell(loop_indices, graph_edges):
    m = len(loop_indices)
    if m == 1:
        return 1.0
    signed_sum = 0
    q = len(graph_edges)
    for edge_mask in range(1, 1 << q):
        adjacency = [[] for _ in range(m)]
        n_selected = 0
        for k, (i, j) in enumerate(graph_edges):
            if (edge_mask >> k) & 1:
                adjacency[i].append(j)
                adjacency[j].append(i)
                n_selected += 1
        seen = {0}
        stack = [0]
        while stack:
            i = stack.pop()
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) == m:
            signed_sum += (-1) ** n_selected
    _, counts = np.unique(loop_indices, return_counts=True)
    denominator = 1
    for count in counts:
        denominator *= math.factorial(int(count))
    return float(signed_sum / denominator)


def _oracle_enumerate_connected_loop_clusters(
    edges: "np.ndarray", loops: "np.ndarray", max_weight: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    edges = np.asarray(edges, dtype=np.int64)
    loops = np.asarray(loops, dtype=np.int64)
    max_weight = int(max_weight)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E,2)")
    E = edges.shape[0]
    if loops.ndim != 2 or loops.shape[1] != E or np.any((loops != 0) & (loops != 1)):
        raise ValueError("loops must be a binary array with shape (L,E)")
    if max_weight < 0:
        raise ValueError("max_weight must be nonnegative")
    L = loops.shape[0]
    if L == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

    loop_weights = loops.sum(axis=1).astype(np.int64)
    if np.any(loop_weights <= 0):
        raise ValueError("loop masks must be nonempty")
    vertex_sets = []
    for mask in loops:
        vertices = set()
        for e, (u0, v0) in enumerate(edges):
            if mask[e]:
                vertices.add(int(u0))
                vertices.add(int(v0))
        vertex_sets.append(vertices)

    records = []
    max_copies = max_weight // int(np.min(loop_weights)) if max_weight > 0 else 0

    def grow(start, chosen, total_weight):
        for loop_index in range(start, L):
            new_weight = total_weight + int(loop_weights[loop_index])
            if new_weight > max_weight:
                continue
            expanded = chosen + [loop_index]
            connected, graph_edges = _step06_interaction_graph(expanded, vertex_sets)
            if connected:
                coefficient = _step06_ursell(expanded, graph_edges)
                records.append((new_weight, len(expanded), tuple(expanded), coefficient))
            if len(expanded) < max_copies:
                grow(loop_index, expanded, new_weight)

    grow(0, [], 0)
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    multiplicities = np.zeros((len(records), L), dtype=np.int64)
    coefficients = np.empty(len(records), dtype=float)
    cluster_weights = np.empty(len(records), dtype=np.int64)
    for row, (total_weight, _, expanded, coefficient) in enumerate(records):
        for loop_index in expanded:
            multiplicities[row, loop_index] += 1
        coefficients[row] = coefficient
        cluster_weights[row] = total_weight
    return multiplicities, coefficients, cluster_weights

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return repeated-copy, overlapping-loop, compatible-loop, and benchmark-like cases."""
    helper = """import numpy as np

def run_clusters(loop_fn, cluster_fn, edges, cutoff):
    ls = loop_fn(edges.copy(),cutoff)
    m,c,w = cluster_fn(edges.copy(),ls.copy(),cutoff)
    return (m,c,w)
"""
    return [
        {
            "setup": helper + "\nedges=np.array([[0,1],[0,2],[1,2]],dtype=int)",
            "call": "run_clusters(enumerate_connected_generalized_loops,enumerate_connected_loop_clusters,edges,9)",
            "gold_call": "run_clusters(_oracle_enumerate_connected_generalized_loops,_oracle_enumerate_connected_loop_clusters,edges,9)",
        },
        {
            "setup": helper + "\nedges=np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[3,4]],dtype=int)",
            "call": "run_clusters(enumerate_connected_generalized_loops,enumerate_connected_loop_clusters,edges,6)",
            "gold_call": "run_clusters(_oracle_enumerate_connected_generalized_loops,_oracle_enumerate_connected_loop_clusters,edges,6)",
        },
        {
            "setup": helper + "\nedges=np.array([[0,1],[0,2],[1,2],[3,4],[3,5],[4,5]],dtype=int)",
            "call": "run_clusters(enumerate_connected_generalized_loops,enumerate_connected_loop_clusters,edges,6)",
            "gold_call": "run_clusters(_oracle_enumerate_connected_generalized_loops,_oracle_enumerate_connected_loop_clusters,edges,6)",
        },
        {
            "setup": helper + "\nedges=np.array([[0,1],[0,2],[1,3],[2,3]],dtype=int)",
            "call": "run_clusters(enumerate_connected_generalized_loops,enumerate_connected_loop_clusters,edges,4)",
            "gold_call": "run_clusters(_oracle_enumerate_connected_generalized_loops,_oracle_enumerate_connected_loop_clusters,edges,4)",
        },
    ]
