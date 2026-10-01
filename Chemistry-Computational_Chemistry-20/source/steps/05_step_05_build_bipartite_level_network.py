"""
Recover the bipartite level network and its unique missing transition.

The returned incidence matrices, observed endpoints, absent edge and gauge
index use deterministic line-identifier ordering.

Returns
-------
return upper_incidence, lower_incidence, endpoints, missing_edge, gauge_lower
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_bipartite_level_network(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", int]:
    """Partition the stitched levels into upper and lower vertex classes.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of ``stitch_energy_level_cliques`` in
        step 4.

    Returns
    -------
    upper_incidence : np.ndarray
        Integer array of shape ``(n_upper, n_lines)``.
    lower_incidence : np.ndarray
        Integer array of shape ``(n_lower, n_lines)``.
    endpoints : np.ndarray
        Integer array of shape ``(n_lines, 2)`` containing each observed
        transition's upper and lower indices.
    missing_edge : np.ndarray
        Length-two integer array containing the unique unobserved upper and
        lower indices.
    gauge_lower : int
        Index of the lower level fixed to zero in the weighted inversion.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs, the recovered level graph is
        disconnected or non-bipartite, the colour classes have equal size, a
        line lacks one endpoint in each class, or the cross partition has
        anything other than one absent pair.
    """
    return upper_incidence, lower_incidence, endpoints, missing_edge, gauge_lower  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from collections import deque
import numpy as np

def _oracle_build_bipartite_level_network(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", int]:
    """Reference implementation for build_bipartite_level_network."""
    ids = np.asarray(line_ids)
    incidence = _oracle_stitch_energy_level_cliques(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    n_levels, n_lines = incidence.shape

    neighbours = [set() for _ in range(n_levels)]
    for line in range(n_lines):
        levels = np.flatnonzero(incidence[:, line])
        if levels.size != 2:
            raise ValueError("each line must join exactly two recovered levels")
        first, second = map(int, levels)
        neighbours[first].add(second)
        neighbours[second].add(first)

    colour = np.full(n_levels, -1, dtype=int)
    colour[0] = 0
    queue = deque([0])
    while queue:
        level = queue.popleft()
        for other in sorted(neighbours[level]):
            if colour[other] < 0:
                colour[other] = 1 - colour[level]
                queue.append(other)
            elif colour[other] == colour[level]:
                raise ValueError("the recovered level graph is not bipartite")
    if np.any(colour < 0):
        raise ValueError("the recovered level graph must be connected")

    first_class = np.flatnonzero(colour == 0).tolist()
    second_class = np.flatnonzero(colour == 1).tolist()
    if len(first_class) == len(second_class):
        raise ValueError("equal colour classes cannot be named by size")
    upper_old, lower_old = ((first_class, second_class)
                            if len(first_class) > len(second_class)
                            else (second_class, first_class))

    def incident_tuple(old_index):
        return tuple(sorted(str(ids[i])
                            for i in np.flatnonzero(incidence[old_index])))

    upper_old.sort(key=incident_tuple)
    lower_old.sort(key=incident_tuple)
    upper = incidence[upper_old].copy()
    lower = incidence[lower_old].copy()

    endpoints = np.empty((n_lines, 2), dtype=int)
    for line in range(n_lines):
        upper_index = np.flatnonzero(upper[:, line])
        lower_index = np.flatnonzero(lower[:, line])
        if upper_index.size != 1 or lower_index.size != 1:
            raise ValueError("each line must join one upper and one lower level")
        endpoints[line] = (int(upper_index[0]), int(lower_index[0]))

    observed = {tuple(map(int, edge)) for edge in endpoints.tolist()}
    absent = [(upper_index, lower_index)
              for upper_index in range(upper.shape[0])
              for lower_index in range(lower.shape[0])
              if (upper_index, lower_index) not in observed]
    if len(absent) != 1:
        raise ValueError("the cross partition must contain exactly one absent edge")

    missing = np.asarray(absent[0], dtype=int)
    gauge = min(range(lower.shape[0]),
                key=lambda row: tuple(sorted(
                    str(ids[i]) for i in np.flatnonzero(lower[row]))))
    return upper, lower, endpoints, missing, int(gauge)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nids = np.array([f'L{i:02d}' for i in range(1, 20)])\nscans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\nfreq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\n"
    return [
        {
            'setup': setup,
            'call': 'build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[0]',
            'gold_call': '_oracle_build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[0]',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[1]',
            'gold_call': '_oracle_build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[1]',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[2]',
            'gold_call': '_oracle_build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[2]',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': '(lambda result: np.array([result[3][0], result[3][1], result[4]]))(build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'gold_call': '(lambda result: np.array([result[3][0], result[3][1], result[4]]))(_oracle_build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'order = np.array([18,0,7,3,11,5,16,1,14,8,6,2,17,10,4,12,9,15,13])\nids, scans, freq = ids[order], scans[order], freq[order]\n',
            'call': '(lambda result: np.vstack([result[0], result[1]]))(build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'gold_call': '(lambda result: np.vstack([result[0], result[1]]))(_oracle_build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'tol': 0.0,
        },
        {
            'setup': setup + '\ndef run_model():\n    try:\n        build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_build_bipartite_level_network(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'tol': 0.0,
        },
    ]
