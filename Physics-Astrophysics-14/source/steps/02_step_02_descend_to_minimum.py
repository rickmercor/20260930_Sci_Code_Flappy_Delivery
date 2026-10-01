"""
Follow strictly decreasing graph links from the lowest-potential seed to a local minimum. The potential input is a finite one-dimensional array of at least three pairwise-distinct real values. Adjacency rows are sorted unique native or NumPy integral IDs and encode a connected simple undirected graph. Seed IDs are a nonempty one-dimensional sequence of valid integral IDs. Select the seed minimizing ``(potential[id], id)``, repeatedly move to the lower-potential neighbor minimizing that pair and return the terminal node as a native Python int. Invalid input raises ValueError.

A sampled gravitational potential turns minimum finding into a deterministic descent over graph edges. Every move strictly lowers a distinct finite potential, so the walk cannot cycle, while the node-ID secondary order removes any implementation-dependent neighbor choice. The terminal node is a discrete local minimum rather than necessarily the global minimum.

Returns
-------
native Python int, the terminal local-minimum node ID
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def descend_to_minimum(potential, adjacency, seed_ids):
    """Return the local-minimum node reached by graph descent.

    Parameters
    ----------
    potential : array_like
        Finite one-dimensional pairwise-distinct potential values.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    seed_ids : array_like of int
        Nonempty valid native or NumPy integer seed IDs.

    Returns
    -------
    int
        Native Python integer ID of the terminal local minimum.

    Raises
    ------
    ValueError
        If the potentials, graph or IDs violate the stated contract.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_descend_to_minimum(potential, adjacency, seed_ids):
    try:
        values = np.asarray(potential)
    except Exception as exc:
        raise ValueError("potential must be a one-dimensional numeric array") from exc
    if values.ndim != 1 or values.size < 3:
        raise ValueError("potential must have at least three entries")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError("potential must contain real numeric values")
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("potential must contain finite real values") from exc
    if not np.all(np.isfinite(values)) or np.unique(values).size != values.size:
        raise ValueError("potential values must be finite and pairwise distinct")
    node_count = int(values.size)

    if not isinstance(adjacency, (list, tuple, np.ndarray)) or len(adjacency) != node_count:
        raise ValueError("adjacency must contain one row per node")
    rows = []
    for node, raw_row in enumerate(adjacency):
        try:
            row_array = np.asarray(raw_row, dtype=object)
        except Exception as exc:
            raise ValueError("adjacency rows must be one-dimensional") from exc
        if row_array.ndim != 1:
            raise ValueError("adjacency rows must be one-dimensional")
        row = []
        for item in row_array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError("adjacency must contain integral node IDs")
            neighbor = int(item)
            if neighbor < 0 or neighbor >= node_count:
                raise ValueError("adjacency contains an invalid node ID")
            row.append(neighbor)
        if any(left >= right for left, right in zip(row, row[1:])):
            raise ValueError("adjacency rows must be sorted and unique")
        if node in row:
            raise ValueError("the graph must not contain self-loops")
        rows.append(row)
    for node, row in enumerate(rows):
        for neighbor in row:
            if node not in rows[neighbor]:
                raise ValueError("the graph must be undirected")
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in rows[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if len(reached) != node_count:
        raise ValueError("the graph must be connected")

    try:
        seed_array = np.asarray(seed_ids, dtype=object)
    except Exception as exc:
        raise ValueError("seed_ids must be a one-dimensional integer sequence") from exc
    if seed_array.ndim != 1 or seed_array.size == 0:
        raise ValueError("seed_ids must be nonempty and one-dimensional")
    seeds = []
    for item in seed_array.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("seed_ids must contain integral node IDs")
        node = int(item)
        if node < 0 or node >= node_count:
            raise ValueError("seed_ids contains an invalid node ID")
        seeds.append(node)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed_ids must be distinct")

    current = min(seeds, key=lambda node: (values[node], node))
    while True:
        lower = [neighbor for neighbor in rows[current] if values[neighbor] < values[current]]
        if not lower:
            return int(current)
        current = min(lower, key=lambda node: (values[node], node))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "potential = [9.0, 4.0, 7.0, 1.0, 6.0, 2.0, 8.0]\nadjacency = [[1, 2], [0, 3, 4], [0, 4, 6], [1, 5], [1, 2, 5], [3, 4, 6], [2, 5]]\nseeds = np.array([0, 2, 6], dtype=np.int64)\n",
            "call": "descend_to_minimum(potential, adjacency, seeds)",
            "gold_call": "_oracle_descend_to_minimum(potential, adjacency, seeds)",
        },
        {
            "setup": "potential = [0.5, -8.0, 3.5, -1.0, 7.0, 2.0]\nadjacency = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]]\nseeds = [np.int64(1), 3, 5]\n",
            "call": "(descend_to_minimum(potential, adjacency, seeds), type(descend_to_minimum(potential, adjacency, seeds)) is int)",
            "gold_call": "(_oracle_descend_to_minimum(potential, adjacency, seeds), type(_oracle_descend_to_minimum(potential, adjacency, seeds)) is int)",
        },
        {
            "setup": "potential = [12.0, 5.0, 4.0, 9.0, 1.0, 7.0, -3.0, 8.0]\nadjacency = [[1, 3], [0, 2, 4], [1, 5, 6], [0, 4, 7], [1, 3, 5], [2, 4, 6], [2, 5, 7], [3, 6]]\nseeds = [0, 3]\n",
            "call": "descend_to_minimum(potential, adjacency, seeds)",
            "gold_call": "_oracle_descend_to_minimum(potential, adjacency, seeds)",
        },
        {
            "setup": "potential = [11.0, 6.0, 0.0, 8.0, 3.0, -2.0, 4.0]\nadjacency = [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 6], [2, 4, 6], [3, 4, 5]]\nseeds = [0, 3]\n",
            "call": "descend_to_minimum(potential, adjacency, seeds)",
            "gold_call": "_oracle_descend_to_minimum(potential, adjacency, seeds)",
        },
        {
            "setup": "potential = (2.0, -1.0, 4.0)\nadjacency = (np.array([1], dtype=np.int64), np.array([0, 2], dtype=np.int64), np.array([1], dtype=np.int64))\nseeds = (np.int64(2),)\n",
            "call": "descend_to_minimum(potential, adjacency, seeds)",
            "gold_call": "_oracle_descend_to_minimum(potential, adjacency, seeds)",
        },
        {
            "call": "run_model()",
            "gold_call": "run_gold()",
            "setup": "def invalid_status(function):\n    cases = [\n        ([3.0, np.nan, 1.0], [[1], [0, 2], [1]], [0]),\n        ([3.0, 3.0, 1.0], [[1], [0, 2], [1]], [0]),\n        ([3.0, 2.0, 1.0], [[1], [0], []], [0]),\n        ([3.0, 2.0, 1.0], [[1], [2], [1]], [0]),\n        ([3.0, 2.0, 1.0], [[2, 1], [0], [0]], [0]),\n        ([3.0, 2.0, 1.0], [[1], [0, 2], [1]], [True]),\n        ([3.0, 2.0, 1.0], [[1], [0, 2], [1]], []),\n    ]\n    for arguments in cases:\n        try:\n            function(*arguments)\n        except ValueError:\n            continue\n        except Exception:\n            return 2\n        return 0\n    return 1\n\ndef run_model():\n    return invalid_status(descend_to_minimum)\n\ndef run_gold():\n    return invalid_status(_oracle_descend_to_minimum)\n",
        },
    ]
