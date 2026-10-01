"""
Determine the query neighbors and whether the source connectivity condition permits reranking.

The experimental query has no known structural fingerprint. Apply the supplied spectral and connectivity controls according to the pinned insertion, boundary, and fallback conventions while preserving candidate order.

Returns
-------
A zero-based integer neighbor array, or an empty integer array when the source connectivity condition is not met.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def insert_query_node(
    query_distances: "np.ndarray",
    threshold: float,
    min_connections: int,
) -> "np.ndarray":
    """Return the source-defined query-neighbor array.

    Candidate indices remain zero-based and preserve input order. Apply the
    supplied connectivity controls using the pinned insertion and fallback
    conventions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_insert_query_node(
    query_distances: "np.ndarray",
    threshold: float,
    min_connections: int,
) -> "np.ndarray":
    import numpy as np

    values = np.asarray(
        query_distances,
        dtype=float,
    )

    if (
        values.ndim != 1
        or threshold < 0
        or min_connections < 1
    ):
        raise ValueError(
            "invalid query insertion input"
        )

    neighbors = np.flatnonzero(
        values < threshold
    ).astype(int)

    if len(neighbors) < min_connections:
        return np.empty(0, dtype=int)

    return neighbors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\nimport numpy as np\nd=np.array([39.9,40.,10.])',
      'call': 'insert_query_node(copy.deepcopy(d), copy.deepcopy(40.0), copy.deepcopy(2))',
      'gold_call': '_oracle_insert_query_node(copy.deepcopy(d), copy.deepcopy(40.0), '
                   'copy.deepcopy(2))'},
     {'setup': 'import copy\nimport numpy as np\nd=np.array([39.,41.,42.])',
      'call': 'insert_query_node(copy.deepcopy(d), copy.deepcopy(40.0), copy.deepcopy(2))',
      'gold_call': '_oracle_insert_query_node(copy.deepcopy(d), copy.deepcopy(40.0), '
                   'copy.deepcopy(2))'},
     {'setup': 'import copy\nimport numpy as np\nd=np.array([0.,39.999,55.])',
      'call': 'insert_query_node(copy.deepcopy(d), copy.deepcopy(40.0), copy.deepcopy(2))',
      'gold_call': '_oracle_insert_query_node(copy.deepcopy(d), copy.deepcopy(40.0), '
                   'copy.deepcopy(2))'}]
