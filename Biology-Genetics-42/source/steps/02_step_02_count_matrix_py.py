"""
Return the lower-triangular matrix whose entry (i, j) counts the lineages present throughout intervals j to i without being involved in any event.

An unlabeled ranked network is represented by a triangular matrix that records lineage persistence across ranked intervals. Each edge is one lineage from the event rank at which it begins until the event or tip rank at which it ends. The diagonal is the lineage count in one interval; off-diagonal entries retain whether the same lineages survive across longer spans. Tip identities do not affect the invariant; only the ranked endpoints of lineages matter.

Returns
-------
list of list of int - a size-by-size lower-triangular matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def count_matrix(edges, size):
    """Construct the lower-triangular F-matrix of persisting lineages.

    Parameters
    ----------
    edges : list of tuple of str
        Edges as ``(ancestor, descendant)`` pairs. The root is ``"*"``,
        internal vertices are named ``"v<number>"``, and leaves are named
        ``"L<number>"``.
    size : int
        Number of ranked intervals. Internal vertex ranks must lie in
        ``1..size-1``.

    Returns
    -------
    list of list of int
        A ``size`` by ``size`` lower-triangular matrix. For ``j <= i``,
        entry ``(i, j)`` is the number of lineages that persist continuously
        through every ranked interval from ``j`` through ``i``. Entries above
        the diagonal are zero.

    Raises
    ------
    ValueError
        If ``size`` is not an integer of at least two; if ``edges`` is empty
        or malformed; if a vertex label is invalid; if an internal rank lies
        outside ``1..size-1``; or if an edge does not point from an older
        event to a younger event.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_count_matrix(edges, size):
    """Return the F-matrix of persisting lineages."""
    import numpy as np

    if not (
        isinstance(size, (int, np.integer))
        and not isinstance(size, bool)
        and int(size) >= 2
    ):
        raise ValueError("size must be an integer of at least 2")

    size = int(size)

    if not isinstance(edges, (list, tuple)) or len(edges) == 0:
        raise ValueError("edges must be a non-empty list or tuple")

    def rank(vertex):
        if not isinstance(vertex, str):
            raise ValueError("each endpoint must be a string")
        if vertex == "*":
            return 0
        if vertex.startswith("v") and vertex[1:].isdigit():
            value = int(vertex[1:])
            if not 1 <= value < size:
                raise ValueError("internal vertex rank must lie in 1..size-1")
            return value
        if vertex.startswith("L") and vertex[1:].isdigit():
            return size
        raise ValueError("vertex labels must be '*', v<number>, or L<number>")

    checked_edges = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each edge must be an ancestor-descendant pair")

        ancestor, descendant = edge
        ancestor_rank = rank(ancestor)
        descendant_rank = rank(descendant)

        if descendant == "*":
            raise ValueError("the root cannot be a descendant")
        if ancestor.startswith("L"):
            raise ValueError("a leaf cannot be an ancestor")
        if ancestor_rank >= descendant_rank:
            raise ValueError("edges must point from older to younger ranks")

        checked_edges.append((ancestor_rank, descendant_rank, tuple(edge)))

    alive = [
        {
            edge
            for ancestor_rank, descendant_rank, edge in checked_edges
            if ancestor_rank < interval <= descendant_rank
        }
        for interval in range(1, size + 1)
    ]

    return [
        [
            len(alive[i] & alive[j]) if j <= i else 0
            for j in range(size)
        ]
        for i in range(size)
    ]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return F-matrix test specifications."""
    network_a = """edges = [
    ('*', 'v1'),
    ('v1', 'v2'),
    ('v1', 'v3'),
    ('v2', 'v3'),
    ('v2', 'v6'),
    ('v3', 'v4'),
    ('v4', 'L1'),
    ('v4', 'v5'),
    ('v5', 'L2'),
    ('v5', 'L3'),
    ('v6', 'L4'),
    ('v6', 'L5'),
]
size = 7
"""

    network_b = """edges = [
    ('*', 'v1'),
    ('v1', 'v2'),
    ('v1', 'v3'),
    ('v2', 'v3'),
    ('v2', 'v5'),
    ('v3', 'v4'),
    ('v4', 'v5'),
    ('v4', 'v7'),
    ('v5', 'v6'),
    ('v6', 'L1'),
    ('v6', 'v8'),
    ('v7', 'L2'),
    ('v7', 'L3'),
    ('v8', 'L4'),
    ('v8', 'L5'),
]
size = 9
"""

    tree_control = """edges = [
    ('*', 'v1'),
    ('v1', 'v2'),
    ('v1', 'v3'),
    ('v2', 'v3'),
    ('v2', 'v4'),
    ('v3', 'L1'),
    ('v4', 'L2'),
    ('v4', 'L3'),
]
size = 5
"""

    same_diagonal_pair = """edges_1 = [
    ('v6', 'L2'),
    ('v1', 'v3'),
    ('v4', 'v6'),
    ('*', 'v1'),
    ('v7', 'L4'),
    ('v2', 'v3'),
    ('v4', 'v5'),
    ('v6', 'L1'),
    ('v1', 'v2'),
    ('v2', 'v4'),
    ('v5', 'v7'),
    ('v3', 'v5'),
    ('v7', 'L3'),
]
edges_2 = [
    ('v6', 'L2'),
    ('v4', 'v7'),
    ('v1', 'v3'),
    ('*', 'v1'),
    ('v2', 'v5'),
    ('v7', 'L4'),
    ('v2', 'v3'),
    ('v4', 'v5'),
    ('v6', 'L1'),
    ('v1', 'v2'),
    ('v5', 'v6'),
    ('v3', 'v4'),
    ('v7', 'L3'),
]
size = 8
"""

    invalid_edge = """edges = [
    ('*', 'v1'),
    ('v2', 'v1'),
    ('v1', 'L1'),
]
size = 3

def run_model():
    try:
        count_matrix(edges, size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_count_matrix(edges, size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""

    return [
        {
            "setup": network_a,
            "call": "count_matrix(edges, size)",
            "gold_call": "_oracle_count_matrix(edges, size)",
        },
        {
            "setup": network_b,
            "call": "count_matrix(edges, size)",
            "gold_call": "_oracle_count_matrix(edges, size)",
        },
        {
            "setup": tree_control,
            "call": "count_matrix(edges, size)",
            "gold_call": "_oracle_count_matrix(edges, size)",
        },
        {
            "setup": same_diagonal_pair,
            "call": (
                "[count_matrix(edges_1, size)[4][2], "
                "count_matrix(edges_1, size)[6][4], "
                "count_matrix(edges_2, size)[4][2], "
                "count_matrix(edges_2, size)[6][4]]"
            ),
            "gold_call": (
                "[_oracle_count_matrix(edges_1, size)[4][2], "
                "_oracle_count_matrix(edges_1, size)[6][4], "
                "_oracle_count_matrix(edges_2, size)[4][2], "
                "_oracle_count_matrix(edges_2, size)[6][4]]"
            ),
        },
        {
            "setup": invalid_edge,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
