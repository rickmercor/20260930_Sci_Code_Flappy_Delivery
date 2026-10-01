"""
Evaluate the source-defined weighted resource-allocation signal from the inserted query to every library candidate.

The result depends on retained library-edge weights, the inserted query neighborhood, and graph connectivity. Preserve original candidate order and follow the pinned implementation's degree convention.

Returns
-------
A float PWRA vector in original candidate order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def product_weighted_resource_allocation(
    n_nodes: int,
    edges: "np.ndarray",
    edge_hybrid: "np.ndarray",
    query_neighbors: "np.ndarray",
) -> "np.ndarray":
    """Return the source-defined PWRA vector for the augmented graph.

    Preserve the supplied node labels and the alignment between edges and
    their Hybrid weights. Scores remain in original candidate order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_product_weighted_resource_allocation(
    n_nodes: int,
    edges: "np.ndarray",
    edge_hybrid: "np.ndarray",
    query_neighbors: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    edge_array = np.asarray(edges, dtype=int)
    weights = np.asarray(edge_hybrid, dtype=float)
    neighbors = np.asarray(query_neighbors, dtype=int)

    if (
        n_nodes < 1
        or edge_array.ndim != 2
        or edge_array.shape[1:] != (2,)
    ):
        raise ValueError(
            "edges must be an n-by-2 array"
        )

    if (
        weights.shape != (len(edge_array),)
        or neighbors.ndim != 1
    ):
        raise ValueError(
            "inconsistent graph arrays"
        )

    adjacency = [
        set()
        for _ in range(n_nodes)
    ]
    edge_weight = {}

    for (i, j), weight in zip(
        edge_array,
        weights,
    ):
        if (
            i == j
            or min(i, j) < 0
            or max(i, j) >= n_nodes
        ):
            raise ValueError("invalid edge")

        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))

        key = (
            min(int(i), int(j)),
            max(int(i), int(j)),
        )
        edge_weight[key] = float(weight)

    query_set = {
        int(node)
        for node in neighbors
    }

    updated_degree = np.array(
        [
            len(adjacency[node])
            + int(node in query_set)
            for node in range(n_nodes)
        ],
        dtype=float,
    )

    scores = np.zeros(
        n_nodes,
        dtype=float,
    )

    for candidate in range(n_nodes):
        shared_neighbors = (
            query_set.intersection(
                adjacency[candidate]
            )
        )

        for shared in shared_neighbors:
            key = (
                min(candidate, shared),
                max(candidate, shared),
            )
            scores[candidate] += (
                edge_weight[key]
                / updated_degree[shared]
            )

    return scores

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'import numpy as np\n'
               'e=np.array([[0,2],[0,3],[1,2],[1,4],[2,3],[2,4],[3,4],[4,5]]); '
               'w=np.array([.2,.9,.8,.3,.5,.7,.4,.95]); q=np.array([2,4,5])',
      'call': 'product_weighted_resource_allocation(copy.deepcopy(6), copy.deepcopy(e), '
              'copy.deepcopy(w), copy.deepcopy(q))',
      'gold_call': '_oracle_product_weighted_resource_allocation(copy.deepcopy(6), '
                   'copy.deepcopy(e), copy.deepcopy(w), copy.deepcopy(q))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'e=np.empty((0,2),dtype=int); w=np.array([]); q=np.array([0,1])',
      'call': 'product_weighted_resource_allocation(copy.deepcopy(3), copy.deepcopy(e), '
              'copy.deepcopy(w), copy.deepcopy(q))',
      'gold_call': '_oracle_product_weighted_resource_allocation(copy.deepcopy(3), '
                   'copy.deepcopy(e), copy.deepcopy(w), copy.deepcopy(q))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'e=np.array([[0,1],[1,2]]); w=np.array([.7,.8]); q=np.array([1])',
      'call': 'product_weighted_resource_allocation(copy.deepcopy(3), copy.deepcopy(e), '
              'copy.deepcopy(w), copy.deepcopy(q))',
      'gold_call': '_oracle_product_weighted_resource_allocation(copy.deepcopy(3), '
                   'copy.deepcopy(e), copy.deepcopy(w), copy.deepcopy(q))'}]
