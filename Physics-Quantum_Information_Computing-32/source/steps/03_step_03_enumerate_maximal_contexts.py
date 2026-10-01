"""
Enumerate every inclusion-maximal commuting measurement context.

A context is an independent set $S$ in the anticommutation graph. It is maximal when no additional measured observable commutes with every member of $S$; it need not have maximum cardinality. Equivalently, contexts are maximal cliques of the complementary compatibility graph. Enumerate all such sets, including smaller maximal sets, using a search that extends only compatible vertices and suppresses duplicates. Encode membership in the original measurement order and sort by the integer mask $\sum_{j\in S}2^j$.

Returns
-------
Integer membership array of shape $(C,m)$, sorted by ascending bit mask; each row is a maximal independent set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_maximal_contexts(adjacency: "np.ndarray") -> "np.ndarray":
    r"""Enumerate every inclusion-maximal commuting measurement context.

    Parameters
    ----------
    adjacency : np.ndarray
        Binary symmetric graph matrix of shape $(m,m)$, $1\le m\le32$, with zero diagonal.

    Returns
    -------
    contexts : np.ndarray
        Integer membership array of shape $(C,m)$, sorted by ascending bit mask; each row is a maximal independent set.

    Raises
    ------
    ValueError
        If adjacency is not a nonempty square binary symmetric matrix with zero diagonal, or has more than 32 vertices.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_enumerate_maximal_contexts(adjacency: "np.ndarray") -> "np.ndarray":
    graph = np.asarray(adjacency)
    if (
        graph.ndim != 2
        or not 1 <= graph.shape[0] <= 32
        or graph.shape[0] != graph.shape[1]
    ):
        raise ValueError("adjacency must be square with one to 32 vertices")
    if (
        not np.all((graph == 0) | (graph == 1))
        or not np.array_equal(graph, graph.T)
        or np.any(np.diag(graph))
    ):
        raise ValueError("adjacency must describe a simple undirected graph")
    m = len(graph)
    neighbors = [set(np.flatnonzero(graph[j] == 0)) - {j} for j in range(m)]
    masks = []

    def _visit(chosen, possible, excluded):
        if not possible and not excluded:
            masks.append(sum(1 << j for j in chosen))
            return
        pivot = min(
            possible | excluded, key=lambda j: (-len(possible & neighbors[j]), j)
        )
        for j in sorted(possible - neighbors[pivot]):
            _visit(chosen + [j], possible & neighbors[j], excluded & neighbors[j])
            possible.remove(j)
            excluded.add(j)

    _visit([], set(range(m)), set())
    return np.array(
        [[(mask >> j) & 1 for j in range(m)] for mask in sorted(masks)], dtype=np.int64
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical cases for this step."""
    return [
        {
            "setup": """import numpy as np
adjacency = np.array([[0,1,0],[1,0,1],[0,1,0]])
""",
            "call": "enumerate_maximal_contexts(adjacency.copy())",
            "gold_call": "_oracle_enumerate_maximal_contexts(adjacency.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
adjacency = np.zeros((4,4), dtype=int)
""",
            "call": "enumerate_maximal_contexts(adjacency.copy())",
            "gold_call": "_oracle_enumerate_maximal_contexts(adjacency.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
adjacency = np.ones((4,4), dtype=int) - np.eye(4, dtype=int)
""",
            "call": "enumerate_maximal_contexts(adjacency.copy())",
            "gold_call": "_oracle_enumerate_maximal_contexts(adjacency.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
adjacency = np.array([[0,1],[0,0]])

def _raises_value_error(fn):
    try:
        fn(adjacency.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(enumerate_maximal_contexts)",
            "gold_call": "_raises_value_error(_oracle_enumerate_maximal_contexts)",
            "tol": 0.0,
        },
    ]
