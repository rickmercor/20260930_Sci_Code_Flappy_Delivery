"""
Recover the topology-discovery offset and recurring-difference groups.

The returned occurrence table is deterministic under the supplied line
identifiers and records the retained pair occurrences needed by the local
closure stage.  Its cluster indices follow increasing difference order.

Returns
-------
return b_seed, corrected_frequencies, occurrences, counts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cluster_recurring_differences(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple[float, "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Apply the deterministic first-value clustering rule to line differences.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi
        Non-empty aligned one-dimensional arrays of unique identifiers,
        scan labels A or B, and finite pairwise-distinct reported centres,
        followed by finite ordered bounds for the seed-offset bracket.
    delta
        Finite strictly positive maximum displacement from a cluster's first
        difference.
    min_occurrences
        Integer threshold at least 2 for retaining a cluster.

    Returns
    -------
    b_seed : float
        Ordinary median of the multiplicity-preserving closure candidates
        retained inside the supplied offset bracket.
    corrected_frequencies : np.ndarray
        Seed-corrected frequencies aligned with ``line_ids``.
    occurrences : np.ndarray
        Integer array with shape ``(n_occurrences, 3)``.  Each row is
        ``(cluster_index, i, j)`` with ``i < j``.
    counts : np.ndarray
        Number of pair occurrences in each retained cluster.

    Raises
    ------
    ValueError
        If the aligned line data, labels, reported centres, bounds, clustering
        tolerance or occurrence threshold violate the stated contracts, or if
        the bracket contains no eligible closure candidate.
    """
    return b_seed, corrected_frequencies, occurrences, counts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np

def _oracle_cluster_recurring_differences(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple[float, "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Reference implementation for cluster_recurring_differences."""

    ids = np.asarray(line_ids)
    labels = np.asarray(scans)
    reported = np.asarray(reported_frequencies, dtype=float)
    if ids.ndim != 1 or labels.ndim != 1 or reported.ndim != 1:
        raise ValueError("line_ids, scans and reported_frequencies must be one-dimensional")
    if ids.size == 0 or not (ids.size == labels.size == reported.size):
        raise ValueError("line_ids, scans and reported_frequencies must be non-empty and aligned")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("line_ids must be unique")
    labels = labels.astype(str)
    if not np.all(np.isin(labels, ["A", "B"])):
        raise ValueError("scans must contain only 'A' and 'B'")
    if not np.all(np.isfinite(reported)):
        raise ValueError("reported_frequencies must be finite")
    if np.unique(reported).size != reported.size:
        raise ValueError("reported_frequencies must be pairwise distinct")
    if not (np.isfinite(b_lo) and np.isfinite(b_hi)) or float(b_lo) > float(b_hi):
        raise ValueError("b_lo and b_hi must be finite and ordered")
    if not (np.isfinite(delta) and float(delta) > 0.0):
        raise ValueError("delta must be finite and strictly positive")
    if isinstance(min_occurrences, (bool, np.bool_)) or int(min_occurrences) != min_occurrences:
        raise ValueError("min_occurrences must be an integer at least 2")
    min_occurrences = int(min_occurrences)
    if min_occurrences < 2:
        raise ValueError("min_occurrences must be an integer at least 2")

    indicator_int = (labels == "B").astype(int)
    oriented = []
    for i, j in itertools.combinations(range(reported.size), 2):
        high, low = (i, j) if reported[i] > reported[j] else (j, i)
        oriented.append((high, low,
                         float(reported[high] - reported[low]),
                         int(indicator_int[high] - indicator_int[low])))

    retained = []
    for first, second in itertools.combinations(oriented, 2):
        if len({first[0], first[1], second[0], second[1]}) != 4:
            continue
        q_first, q_second = first[3], second[3]
        if q_first == q_second:
            continue
        candidate = ((first[2] - second[2]) /
                     float(q_first - q_second))
        if float(b_lo) <= candidate <= float(b_hi):
            retained.append(candidate)
    candidates = np.sort(np.asarray(retained, dtype=float))
    if candidates.size == 0:
        raise ValueError("the offset bracket contains no closure candidate")
    b_seed = float(np.median(candidates))

    indicator = (labels == "B").astype(float)
    corrected = reported - b_seed * indicator

    differences = [
        (float(abs(corrected[i] - corrected[j])), i, j)
        for i, j in itertools.combinations(range(corrected.size), 2)
    ]
    differences.sort(key=lambda row: (row[0], row[1], row[2]))

    raw_clusters = []
    cursor = 0
    while cursor < len(differences):
        first_value = differences[cursor][0]
        stop = cursor + 1
        while stop < len(differences) and differences[stop][0] <= first_value + float(delta):
            stop += 1
        raw_clusters.append(differences[cursor:stop])
        cursor = stop

    rows = []
    counts = []
    for cluster in raw_clusters:
        if len(cluster) < min_occurrences:
            continue
        retained_index = len(counts)
        counts.append(len(cluster))
        rows.extend((retained_index, item[1], item[2]) for item in cluster)

    occurrences = (np.asarray(rows, dtype=int).reshape(-1, 3)
                   if rows else np.empty((0, 3), dtype=int))
    return (float(b_seed), corrected,
            occurrences, np.asarray(counts, dtype=int))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nids = np.array([f'L{i:02d}' for i in range(1, 20)])\nscans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\nfreq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\n"
    return [
        {
            'setup': setup,
            'call': 'float(cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[0])',
            'gold_call': 'float(_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[0])',
            'tol': 1e-10,
        },
        {
            'setup': setup,
            'call': 'float(cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[3].size)',
            'gold_call': 'float(_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[3].size)',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'gold_call': '_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[3]',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'float(np.sum(cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[2][:, 1:] * np.array([1, 20])))',
            'gold_call': 'float(np.sum(_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[2][:, 1:] * np.array([1, 20])))',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[1] - freq',
            'gold_call': '_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[1] - freq',
            'tol': 1e-10,
        },
        {
            'setup': setup + 'order = np.array([18,0,7,3,11,5,16,1,14,8,6,2,17,10,4,12,9,15,13])\nids, scans, freq = ids[order], scans[order], freq[order]\n',
            'call': 'float(cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[0])',
            'gold_call': 'float(_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)[0])',
            'tol': 1e-10,
        },
        {
            'setup': setup,
            'call': 'cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 0.5)[3]',
            'gold_call': '_oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 0.5)[3]',
            'tol': 0.0,
        },
        {
            'setup': setup + '\ndef run_model():\n    try:\n        cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_cluster_recurring_differences(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'tol': 0.0,
        },
    ]
