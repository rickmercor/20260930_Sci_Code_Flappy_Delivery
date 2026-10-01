"""
Re-analyse a recorded subgroup in its own accelerated frame and return its locally bound members and graph surface. All particle arrays are finite, one-dimensional and equally long with at least three entries. The graph is connected, simple and undirected with sorted unique native or NumPy integral adjacency IDs. Subgroup IDs are sorted, unique, integral and in range, and their count meets the positive integral candidate threshold. Build the local boosted potential using the entire subgroup as the seed, choose the subgroup member minimizing `$(local_potential[id], id)$`, run the full-graph well traversal with the unchanged threshold and apply the host-energy rule to that local well. Return the sorted subgroup IDs whose local mask is one, followed by the sorted union of all full-graph neighbors of those bound members that are not locally bound. Invalid input raises ValueError.

A false-saddle branch may be self-bound even when some of its particles fail the primary host-frame energy test. Its acceleration frame must therefore be estimated from every recorded candidate member before the topology and strict individual binding calculation are repeated. The local surface is the full-graph boundary of the rebound object; it may contain its unbound local saddle or nodes outside the original candidate. The boosted formula uses fixed coefficient 0.25 and the energy formula uses fixed coefficients 0.25 and 0.5.

Returns
-------
tuple of list[int] and list[int], the locally bound subgroup IDs and their graph surface
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rebound_subgroup(position, canonical_velocity, acceleration, raw_potential,
                     adjacency, subgroup_ids, scale_factor, hubble_rate, omega_m,
                     omega_lambda, turnaround_overdensity, candidate_threshold):
    """Return locally rebound candidate members and their full-graph surface.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities.
    acceleration : array_like
        Finite one-dimensional accelerations.
    raw_potential : array_like
        Finite one-dimensional raw potentials.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    subgroup_ids : array_like of int
        Sorted unique valid candidate IDs meeting ``candidate_threshold``.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    omega_lambda : float
        Nonnegative finite dark-energy density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.
    candidate_threshold : int
        Positive native or NumPy integral branch-size threshold.

    Returns
    -------
    tuple of list of int
        Sorted native-int locally bound IDs and sorted native-int surface IDs.

    Raises
    ------
    ValueError
        If an array, graph, ID sequence or scalar violates the contract, or if
        the local traversal has no deeper exit.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rebound_subgroup(position, canonical_velocity, acceleration,
                           raw_potential, adjacency, subgroup_ids, scale_factor,
                           hubble_rate, omega_m, omega_lambda,
                           turnaround_overdensity, candidate_threshold):
    arrays = []
    for value, name in ((position, "position"),
                        (canonical_velocity, "canonical_velocity"),
                        (acceleration, "acceleration"),
                        (raw_potential, "raw_potential")):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size < 3:
            raise ValueError(name + " must have at least three entries")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        arrays.append(array)
    position_array, velocity_array, acceleration_array, raw_array = arrays
    if any(array.shape != position_array.shape for array in arrays[1:]):
        raise ValueError("particle arrays must have equal length")
    node_count = int(position_array.size)

    if not isinstance(adjacency, (list, tuple, np.ndarray)) or len(adjacency) != node_count:
        raise ValueError("adjacency must contain one row per node")
    rows = []
    for node, raw_row in enumerate(adjacency):
        row_array = np.asarray(raw_row, dtype=object)
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
        if any(left >= right for left, right in zip(row, row[1:])) or node in row:
            raise ValueError("adjacency rows must be sorted, unique and loop-free")
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

    if isinstance(candidate_threshold, (bool, np.bool_)) or not isinstance(candidate_threshold, (int, np.integer)):
        raise ValueError("candidate_threshold must be integral")
    candidate_threshold = int(candidate_threshold)
    if candidate_threshold < 1:
        raise ValueError("candidate_threshold must be positive")
    subgroup_array = np.asarray(subgroup_ids, dtype=object)
    if subgroup_array.ndim != 1:
        raise ValueError("subgroup_ids must be one-dimensional")
    subgroup = []
    for item in subgroup_array.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("subgroup_ids must contain integral node IDs")
        node = int(item)
        if node < 0 or node >= node_count:
            raise ValueError("subgroup_ids contains an invalid node ID")
        subgroup.append(node)
    if len(subgroup) < candidate_threshold or any(left >= right for left, right in zip(subgroup, subgroup[1:])):
        raise ValueError("subgroup_ids must be sorted, unique and meet the threshold")

    for name, value, positive in (
        ("scale_factor", scale_factor, True),
        ("hubble_rate", hubble_rate, False),
        ("omega_m", omega_m, False),
        ("omega_lambda", omega_lambda, False),
        ("turnaround_overdensity", turnaround_overdensity, False),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar) or (scalar <= 0.0 if positive else scalar < 0.0):
            raise ValueError(name + " lies outside its valid range")

    local_potential = _oracle_boosted_turnaround_potential(
        position_array, acceleration_array, raw_array, subgroup, scale_factor,
        hubble_rate, omega_m, turnaround_overdensity)
    local_minimum = min(subgroup, key=lambda node: (local_potential[node], node))
    local_well, local_saddle, _ = _oracle_locate_host_well(
        local_potential, rows, local_minimum, candidate_threshold)
    local_mask, _, _, _ = _oracle_host_energy_mask(
        position_array, velocity_array, local_potential, local_well,
        local_saddle, local_minimum, scale_factor, hubble_rate, omega_m,
        omega_lambda, turnaround_overdensity)
    local_bound = [int(node) for node in subgroup if local_mask[node] == 1]
    bound_set = set(local_bound)
    surface = sorted({neighbor for node in local_bound for neighbor in rows[node]
                      if neighbor not in bound_set})
    return local_bound, [int(node) for node in surface]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "position = [10.0, 9.7, 10.4, 9.4, 8.5, 10.8, 11.0, 11.1, 10.9, 11.5, 12.5, 12.7, 13.0, 9.1, 8.9]\nvelocity = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.7, 2.7, 2.7, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0]\nacceleration = [-0.3, -0.5, -0.4, -0.2, -0.6, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\nraw = [-20.0, -19.417625, -18.758, -18.3305, -14.740625, -15.752, -17.7625, -17.183625, -17.618625, -10.840625, -7.890625, -8.627625, -18.5625, -14.438625, -21.063625]\nadjacency = [[1, 2], [0, 3], [0, 5, 10], [1, 4], [3, 13], [2, 6], [5, 7, 8], [6], [6, 9], [8], [2, 11], [10, 12], [11], [4, 14], [13]]\n",
            "call": "rebound_subgroup(position, velocity, acceleration, raw, adjacency, np.array([5, 6, 7, 8], dtype=np.int64), 1.0, 1.0, 1.0, 0.0, 4.55, np.int64(3))",
            "gold_call": "_oracle_rebound_subgroup(position, velocity, acceleration, raw, adjacency, np.array([5, 6, 7, 8], dtype=np.int64), 1.0, 1.0, 1.0, 0.0, 4.55, np.int64(3))",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0, 3.0, 4.0]\nvelocity = [0.0, 0.0, 0.0, 0.0, 0.0]\nacceleration = [0.0] * 5\nraw = [0.0, 1.0, 2.0, 4.0, -1.0]\nadjacency = [[1], [0, 2], [1, 3], [2, 4], [3]]\n",
            "call": "rebound_subgroup(position, velocity, acceleration, raw, adjacency, [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3)",
            "gold_call": "_oracle_rebound_subgroup(position, velocity, acceleration, raw, adjacency, [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3)",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0, 3.0, 4.0]\nvelocity = [0.0, 0.0, 8.0, 0.0, 0.0]\nacceleration = [0.0] * 5\nraw = [0.0, 1.0, 2.0, 4.0, -1.0]\nadjacency = [[1], [0, 2], [1, 3], [2, 4], [3]]\n",
            "call": "rebound_subgroup(position, velocity, acceleration, raw, adjacency, [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3)",
            "gold_call": "_oracle_rebound_subgroup(position, velocity, acceleration, raw, adjacency, [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3)",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0, 3.0, 4.0]\nvelocity = [0.0] * 5\nacceleration = [0.0] * 5\nraw = [0.0, 1.0, 2.0, 4.0, -1.0]\nadjacency = [[1], [0, 2], [1, 3], [2, 4], [3]]\n",
            "call": "tuple(type(value) is int for part in rebound_subgroup(position, velocity, acceleration, raw, adjacency, [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3) for value in part)",
            "gold_call": "tuple(type(value) is int for part in _oracle_rebound_subgroup(position, velocity, acceleration, raw, adjacency, [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3) for value in part)",
        },
        {
            "setup": "def run_model():\n    try:\n        rebound_subgroup([0, 1, 2, 3], [0]*4, [0]*4, [0, 1, 2, -1], [[1], [0, 2], [1, 3], [2]], [0, 1], 1.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_rebound_subgroup([0, 1, 2, 3], [0]*4, [0]*4, [0, 1, 2, -1], [[1], [0, 2], [1, 3], [2]], [0, 1], 1.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def invalid_status(function):\n    try:\n        function([0, 1, 2, 3], [0]*4, [0]*4, [0, 1, 2, -1], [[1], [0], [3], [2]], [0, 1, 2], 1.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "invalid_status(rebound_subgroup)",
            "gold_call": "invalid_status(_oracle_rebound_subgroup)",
        },
    ]
