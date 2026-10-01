"""
Return the number of lineages present in each ranked interval of a network, in order from the root.

A rooted ranked network is read as a sequence of intervals separated by its internal events. In the first interval only the stem descends from the root, and each later interval holds however many lineages survive the events so far. A bifurcation raises the count by one and a hybridization lowers it by one, so this sequence begins at one, ends at the number of leaves, and records the shape of the history before any timing is considered. It is the diagonal of the matrix encoding and everything downstream is built on it.

Returns
-------
list of int - one lineage count per interval, from the root outward
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lineage_counts(edges, size):
    """Return the lineage count in each ranked interval.

    An edge is present in interval ``k`` when its upper endpoint has rank
    below ``k`` and its lower endpoint has rank at least ``k``. The stem is
    written with upper endpoint ``"*"`` and has rank zero. Internal vertices
    are named ``"v<number>"`` and leaves are named ``"L<number>"``.

    Parameters
    ----------
    edges : list of tuple of str
        Edges as ``(ancestor, descendant)`` pairs.
    size : int
        Number of ranked intervals. Internal vertex ranks must lie in
        ``1..size-1``.

    Returns
    -------
    list of int
        One lineage count per interval, ordered from root to tip.

    Raises
    ------
    ValueError
        If ``size`` is not an integer of at least two; if ``edges`` is empty
        or malformed; if a vertex label is invalid; if an internal rank lies
        outside ``1..size-1``; or if an edge is not directed from an older
        event to a younger event.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_lineage_counts(edges, size):
    """Reference implementation for ranked-interval lineage counts."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least 2")

    if not isinstance(edges, (list, tuple)) or len(edges) == 0:
        raise ValueError("edges must be a non-empty list or tuple")

    def rank(vertex):
        if not isinstance(vertex, str):
            raise ValueError("vertex labels must be strings")

        if vertex == "*":
            return 0

        if vertex.startswith("v") and vertex[1:].isdigit():
            value = int(vertex[1:])
            if not 1 <= value < size:
                raise ValueError("internal vertex rank is outside 1..size-1")
            return value

        if vertex.startswith("L") and vertex[1:].isdigit():
            return size + 1

        raise ValueError("invalid vertex label")

    ranked_edges = []
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each edge must be an ancestor-descendant pair")

        ancestor, descendant = edge
        ancestor_rank = rank(ancestor)
        descendant_rank = rank(descendant)

        if descendant == "*":
            raise ValueError("the root cannot be an edge descendant")
        if isinstance(ancestor, str) and ancestor.startswith("L"):
            raise ValueError("a leaf cannot be an edge ancestor")
        if ancestor_rank >= descendant_rank:
            raise ValueError("each edge must point from an older to a younger vertex")

        ranked_edges.append((ancestor_rank, descendant_rank))

    return [
        sum(1 for ancestor_rank, descendant_rank in ranked_edges
            if ancestor_rank < interval <= descendant_rank)
        for interval in range(1, size + 1)
    ]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return lineage-count test specifications."""
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
    ('v1', 'L1'),
    ('v2', 'L2'),
    ('v2', 'L3'),
]
size = 3
"""

    invalid_rank = """edges = [
    ('*', 'v1'),
    ('v1', 'v3'),
    ('v3', 'L1'),
]
size = 3

def run_model():
    try:
        lineage_counts(edges, size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_lineage_counts(edges, size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""

    return [
        {
            # Normal: the five-leaf, one-hybridization network.
            "setup": network_a,
            "call": "lineage_counts(edges, size)",
            "gold_call": "_oracle_lineage_counts(edges, size)",
        },
        {
            # Normal: the five-leaf, two-hybridization network.
            "setup": network_b,
            "call": "lineage_counts(edges, size)",
            "gold_call": "_oracle_lineage_counts(edges, size)",
        },
        {
            # Boundary: a tree, with no hybridization events.
            "setup": tree_control,
            "call": "lineage_counts(edges, size)",
            "gold_call": "_oracle_lineage_counts(edges, size)",
        },
        {
            # Invalid: an internal vertex cannot have rank equal to size.
            "setup": invalid_rank,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
