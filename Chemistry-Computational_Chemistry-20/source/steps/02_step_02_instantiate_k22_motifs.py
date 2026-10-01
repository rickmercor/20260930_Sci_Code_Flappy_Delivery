"""
Recover the local four-transition closure motifs.

Each row names one local closure by the storage indices of its four lines, so the set of closures does not depend on the storage order of the spectral-line table.

Returns
-------
return motifs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def instantiate_k22_motifs(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Return the local closure motifs supported by recurrence data.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of
        ``cluster_recurring_differences`` in step 1.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_motifs, 4)``. Each row
        ``(h1, l1, h2, l2)`` holds the storage indices of the four distinct
        lines of one local closure: ``(h1, l1)`` and ``(h2, l2)`` are two
        pair occurrences of the same retained cluster, each ordered from the
        higher to the lower seed-corrected frequency. One row is returned for
        every line-disjoint combination of two pair occurrences in every
        retained cluster, so a closure whose two pairings fall in two
        retained clusters appears once for each. Within a row, the pair with
        the smaller (higher-line identifier, lower-line identifier) tuple
        comes first; rows are sorted lexicographically by the string
        identifiers of ``(h1, l1, h2, l2)``.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs or no line-disjoint local motif
        can be recovered.
    """
    return motifs  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np

def _oracle_instantiate_k22_motifs(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Reference implementation for instantiate_k22_motifs."""

    ids = np.asarray(line_ids)
    _, corrected, occurrences, counts = (
        _oracle_cluster_recurring_differences(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))

    def identifier_key(index):
        return str(ids[int(index)])

    def canonical_pair(first, second):
        first, second = int(first), int(second)
        return ((first, second) if identifier_key(first) < identifier_key(second)
                else (second, first))

    motifs = []
    for cluster_index in range(counts.size):
        cluster_rows = occurrences[occurrences[:, 0] == cluster_index]
        oriented = []
        for _, first, second in cluster_rows:
            first, second = int(first), int(second)
            if corrected[first] > corrected[second]:
                high, low = first, second
            elif corrected[second] > corrected[first]:
                high, low = second, first
            else:
                high, low = canonical_pair(first, second)
            oriented.append((high, low))
        oriented.sort(key=lambda pair: (identifier_key(pair[0]),
                                        identifier_key(pair[1])))

        for first_pair, second_pair in itertools.combinations(oriented, 2):
            h1, l1 = first_pair
            h2, l2 = second_pair
            if len({h1, l1, h2, l2}) == 4:
                motifs.append((h1, l1, h2, l2))

    if not motifs:
        raise ValueError("no line-disjoint local K(2,2) motif was instantiated")

    motifs.sort(key=lambda row: tuple(identifier_key(index) for index in row))
    return np.asarray(motifs, dtype=int).reshape(-1, 4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nids = np.array([f'L{i:02d}' for i in range(1, 20)])\nscans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\nfreq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\n"
    return [
        {
            'setup': setup,
            'call': 'np.array(instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6).shape)',
            'gold_call': 'np.array(_oracle_instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6).shape)',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'gold_call': '_oracle_instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': '(lambda result: float(np.dot(result.ravel(), np.arange(1, result.size + 1))))(instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'gold_call': '(lambda result: float(np.dot(result.ravel(), np.arange(1, result.size + 1))))(_oracle_instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'order = np.array([18,0,7,3,11,5,16,1,14,8,6,2,17,10,4,12,9,15,13])\nids, scans, freq = ids[order], scans[order], freq[order]\n',
            'call': 'instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'gold_call': '_oracle_instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'tol': 0.0,
        },
        {
            'setup': setup + '\ndef run_model():\n    try:\n        instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 1e-12)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_instantiate_k22_motifs(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 1e-12)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'tol': 0.0,
        },
    ]
