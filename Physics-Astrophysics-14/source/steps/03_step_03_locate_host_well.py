"""
Grow a discrete potential well from a supplied minimum until its first contour reaches a deeper basin. Potential values must be finite, one-dimensional and pairwise distinct. Adjacency rows must be sorted unique native or NumPy integral IDs for a connected simple undirected graph. The start ID must be valid and the subgroup threshold must be a positive integral scalar. Initialize the internal set with the start and its surface with external neighbors. Always select the surface node minimizing ``(potential[id], id)``. Admit it ordinarily when every strictly lower neighbor is internal. Otherwise explore its external connected branch through nodes strictly below its contour. If that branch reaches a node strictly below the starting potential, add only the contour node and return it as the true saddle. If not, the false-saddle candidate is the contour node together with the external nodes reached strictly below its contour. Include the contour node when testing subgroup_min_size. Absorb this inclusive candidate, merge its exterior frontier into the surface and record its sorted IDs when its inclusive size meets the threshold. Return sorted well IDs, a native-int saddle ID and candidate branches in discovery order. Invalid input or surface exhaustion raises ValueError.

Potential-well topology can contain a shallow subminimum behind an apparent saddle before the host connects to any genuinely deeper basin. Declaring the first such contour to be the boundary truncates the host. The conditional below-contour exploration distinguishes that false saddle from the first true exit: false branches become part of the completed well and remain available for later rebound analysis, while nodes visited beyond a true saddle stay outside. All topology comparisons are strict and the saddle itself belongs to the returned well.

Returns
-------
tuple of list[int], int and list[list[int]], the well, true saddle and candidate branches
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import heapq
import numpy as np


def locate_host_well(potential, adjacency, minimum_id, subgroup_min_size):
    """Return the completed well, its true saddle and recorded branches.

    Parameters
    ----------
    potential : array_like
        Finite one-dimensional pairwise-distinct potential values.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    minimum_id : int
        Valid native or NumPy integer ID of the starting minimum.
    subgroup_min_size : int
        Positive integral threshold on candidate size, including its contour node.

    Returns
    -------
    tuple
        Sorted native-int well IDs, native-int saddle ID and sorted native-int
        candidate branches in discovery order, each including its false saddle.

    Raises
    ------
    ValueError
        If an input violates the contract or no deeper exit is reachable.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import heapq
import numpy as np


def _oracle_locate_host_well(potential, adjacency, minimum_id, subgroup_min_size):
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

    if isinstance(minimum_id, (bool, np.bool_)) or not isinstance(minimum_id, (int, np.integer)):
        raise ValueError("minimum_id must be integral")
    minimum_id = int(minimum_id)
    if minimum_id < 0 or minimum_id >= node_count:
        raise ValueError("minimum_id is out of range")
    if isinstance(subgroup_min_size, (bool, np.bool_)) or not isinstance(subgroup_min_size, (int, np.integer)):
        raise ValueError("subgroup_min_size must be integral")
    subgroup_min_size = int(subgroup_min_size)
    if subgroup_min_size < 1:
        raise ValueError("subgroup_min_size must be positive")
    if any(values[neighbor] < values[minimum_id] for neighbor in rows[minimum_id]):
        raise ValueError("minimum_id must be a local minimum")

    internal = {minimum_id}
    surface = set(rows[minimum_id])
    groups = []
    while surface:
        contour = min(surface, key=lambda node: (values[node], node))
        lower_neighbors = [neighbor for neighbor in rows[contour] if values[neighbor] < values[contour]]
        if all(neighbor in internal for neighbor in lower_neighbors):
            internal.add(contour)
            surface.discard(contour)
            surface.update(neighbor for neighbor in rows[contour] if neighbor not in internal)
            continue

        branch = {contour}
        queued = set()
        exploratory = []
        for neighbor in rows[contour]:
            if neighbor not in internal and neighbor not in branch and neighbor not in queued:
                heapq.heappush(exploratory, (values[neighbor], neighbor))
                queued.add(neighbor)
        found_deeper = False
        while exploratory and exploratory[0][0] < values[contour]:
            _, node = heapq.heappop(exploratory)
            queued.discard(node)
            if node in internal or node in branch:
                continue
            branch.add(node)
            if values[node] < values[minimum_id]:
                found_deeper = True
                break
            for neighbor in rows[node]:
                if neighbor not in internal and neighbor not in branch and neighbor not in queued:
                    heapq.heappush(exploratory, (values[neighbor], neighbor))
                    queued.add(neighbor)
        if found_deeper:
            internal.add(contour)
            return ([int(node) for node in sorted(internal)], int(contour),
                    [[int(node) for node in group] for group in groups])

        internal.update(branch)
        surface.difference_update(branch)
        for node in branch:
            surface.update(neighbor for neighbor in rows[node] if neighbor not in internal)
        if len(branch) >= subgroup_min_size:
            groups.append([int(node) for node in sorted(branch)])
    raise ValueError("no deeper exit is reachable")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "potential = [0.0, 0.6, 0.9, 1.5, 3.3, 3.2, 0.7, 1.0, 1.1, 6.0, 4.0, 2.0, -10.0, 5.0, -2.0]\nadjacency = [[1, 2], [0, 3], [0, 5, 10], [1, 4], [3, 13], [2, 6], [5, 7, 8], [6], [6, 9], [8], [2, 11], [10, 12], [11], [4, 14], [13]]\n",
            "call": "locate_host_well(potential, adjacency, np.int64(0), np.int64(3))",
            "gold_call": "_oracle_locate_host_well(potential, adjacency, np.int64(0), np.int64(3))",
        },
        {
            "setup": "potential = [0.0, 1.0, 2.0, -1.0]\nadjacency = [[1], [0, 2], [1, 3], [2]]\n",
            "call": "locate_host_well(potential, adjacency, 0, 2)",
            "gold_call": "_oracle_locate_host_well(potential, adjacency, 0, 2)",
        },
        {
            "setup": "potential = [0.0, 4.0, 2.0, 5.0, 1.0, 3.0, -2.0]\nadjacency = [[1], [0, 2, 3], [1], [1, 4, 6], [3, 5], [4], [3]]\n",
            "call": "locate_host_well(potential, adjacency, 0, 2)",
            "gold_call": "_oracle_locate_host_well(potential, adjacency, 0, 2)",
        },
        {
            "setup": "potential = [0.0, 4.0, 2.0, 5.0, 1.0, 3.0, -2.0]\nadjacency = [[1], [0, 2, 3], [1], [1, 4, 6], [3, 5], [4], [3]]\n",
            "call": "locate_host_well(potential, adjacency, 0, 3)",
            "gold_call": "([0, 1, 2, 3], 3, [])",
        },
        {
            "setup": "potential = [0.0, 3.0, 1.0, 4.0, -2.0]\nadjacency = [np.array([1], dtype=np.int64), np.array([0, 2, 3], dtype=np.int64), np.array([1], dtype=np.int64), np.array([1, 4], dtype=np.int64), np.array([3], dtype=np.int64)]\n",
            "call": "locate_host_well(potential, adjacency, 0, 1)",
            "gold_call": "_oracle_locate_host_well(potential, adjacency, 0, 1)",
        },
        {
            "setup": "potential = [0.0, 3.0, 1.0, 4.0, -2.0]\nadjacency = [[1], [0, 2, 3], [1], [1, 4], [3]]\n",
            "call": "locate_host_well(potential, adjacency, 0, 3)",
            "gold_call": "_oracle_locate_host_well(potential, adjacency, 0, 3)",
        },
        {
            "call": "run_model()",
            "gold_call": "run_gold()",
            "setup": "def invalid_status(function):\n    cases = [\n        ([0.0, 1.0, 1.0], [[1], [0, 2], [1]], 0, 1),\n        ([0.0, 1.0, -1.0], [[1], [2, 0], [1]], 0, 1),\n        ([0.0, 1.0, 2.0], [[1], [0, 2], [1]], 0, 1),\n        ([0.0, 1.0, -1.0], [[1], [0, 2], [1]], 0, 0),\n    ]\n    for arguments in cases:\n        try:\n            function(*arguments)\n        except ValueError:\n            continue\n        except Exception:\n            return 2\n        return 0\n    return 1\n\ndef run_model():\n    return invalid_status(locate_host_well)\n\ndef run_gold():\n    return invalid_status(_oracle_locate_host_well)\n",
        },
    ]
