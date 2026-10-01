"""
Stitch local shared-level constraints into global energy-level cliques.

The returned incidence representation is deterministic under the supplied
line identifiers and retains both recovered incidences of every transition.

Returns
-------
return level_incidence
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stitch_energy_level_cliques(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> "np.ndarray":
    """Consolidate compatible local vertex witnesses into global levels.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of
        ``derive_local_vertex_constraints`` in step 3.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_levels, n_lines)``.  Entry ``(k, i)`` is
        1 exactly when transition line ``i`` is incident on recovered level
        ``k``. Rows are ordered lexicographically by the sorted tuple of
        incident line identifiers.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs, a stitched component has fewer
        than three lines, no global level is recovered, or a transition line
        does not belong to exactly two recovered levels.
    """
    return level_incidence  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np

def _oracle_stitch_energy_level_cliques(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> "np.ndarray":
    """Reference implementation for stitch_energy_level_cliques."""
    ids = np.asarray(line_ids)
    sharing_pairs = _oracle_derive_local_vertex_constraints(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    pair_sets = [frozenset(map(int, pair)) for pair in sharing_pairs]
    pair_lookup = set(pair_sets)
    parent = list(range(len(pair_sets)))
    size = [1] * len(pair_sets)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]

    for first_index, second_index in itertools.combinations(
            range(len(pair_sets)), 2):
        first = pair_sets[first_index]
        second = pair_sets[second_index]
        if len(first & second) != 1:
            continue
        combined = first | second
        if all(frozenset(pair) in pair_lookup
               for pair in itertools.combinations(combined, 2)):
            union(first_index, second_index)

    components = {}
    for pair_index, pair in enumerate(pair_sets):
        components.setdefault(find(pair_index), set()).update(pair)
    levels = [tuple(sorted(lines)) for lines in components.values()]
    if not levels:
        raise ValueError("no global energy level was recovered")
    if any(len(level) < 3 for level in levels):
        raise ValueError("each stitched energy level must contain at least three lines")
    levels.sort(key=lambda level: tuple(sorted(str(ids[i]) for i in level)))

    incidence = np.zeros((len(levels), ids.size), dtype=int)
    for row, level in enumerate(levels):
        incidence[row, list(level)] = 1
    if not np.all(np.sum(incidence, axis=0) == 2):
        raise ValueError("every transition line must belong to exactly two recovered levels")
    return incidence

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for stitch_energy_level_cliques."""
    production = (
        "import numpy as np\n"
        "ids = np.array([f'L{i:02d}' for i in range(1, 20)])\n"
        "scans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\n"
        "freq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\n"
    )
    call = "stitch_energy_level_cliques(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)"
    gold = "_oracle_stitch_energy_level_cliques(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)"
    return [
        {
            "setup": production,
            "call": f"np.array({call}.shape)",
            "gold_call": f"np.array({gold}.shape)",
            "tol": 0.0,
        },
        {
            "setup": production,
            "call": f"np.sum({call}, axis=1)",
            "gold_call": f"np.sum({gold}, axis=1)",
            "tol": 0.0,
        },
        {
            "setup": production,
            "call": call,
            "gold_call": gold,
            "tol": 0.0,
        },
        {
            "setup": production + "order = np.array([18,0,7,3,11,5,16,1,14,8,6,2,17,10,4,12,9,15,13])\nids, scans, freq = ids[order], scans[order], freq[order]\n",
            "call": call,
            "gold_call": gold,
            "tol": 0.0,
        },
        {
            "setup": production + "\ndef run_model():\n    try:\n        stitch_energy_level_cliques(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 1e-12)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_stitch_energy_level_cliques(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 1e-12)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 0.0,
        },
    ]
