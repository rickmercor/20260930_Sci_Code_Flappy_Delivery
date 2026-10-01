"""
Derive canonical local shared-level constraints from closure motifs.

The returned pair table records the unique transition-line relationships
needed to assemble the global energy-level vertices.

Returns
-------
return sharing_pairs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def derive_local_vertex_constraints(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Return canonical line pairs that witness common local vertices.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of ``instantiate_k22_motifs`` in
        step 2.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_pairs, 2)`` containing the sorted unique
        line-index pairs implied to share a local energy-level vertex.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs or no local vertex constraint
        can be derived.

    Index and ordering convention
    -----------------------------
    Entries are zero-based positions in the supplied arrays. Order the two
    members of each pair by ascending string-valued line identifier, then sort
    the unique rows lexicographically by their two identifier values. This
    ordering is by identifier, not by integer storage position.
    """
    return sharing_pairs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_derive_local_vertex_constraints(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Reference implementation for derive_local_vertex_constraints."""

    ids = np.asarray(line_ids)
    motifs = _oracle_instantiate_k22_motifs(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)

    def identifier_key(index):
        return str(ids[int(index)])

    def canonical_pair(first, second):
        first, second = int(first), int(second)
        return ((first, second) if identifier_key(first) < identifier_key(second)
                else (second, first))

    sharing_pairs = set()
    for h1, l1, h2, l2 in motifs:
        for pair in ((h1, l1), (h2, l2), (h1, h2), (l1, l2)):
            sharing_pairs.add(canonical_pair(*pair))

    if not sharing_pairs:
        raise ValueError("no local shared-level constraint was derived")
    ordered = sorted(
        sharing_pairs,
        key=lambda pair: (identifier_key(pair[0]), identifier_key(pair[1])))
    return np.asarray(ordered, dtype=int).reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nids = np.array([f'L{i:02d}' for i in range(1, 20)])\nscans = np.array(['B','A','A','B','A','B','A','B','B','B','A','A','B','A','B','A','B','B','A'])\nfreq = np.array([6291.6002896,6305.3101133,6309.9630221,6314.7284247,6321.1374199,6323.6823052,6333.0911534,6338.3752148,6339.5096148,6352.0897687,6361.4986205,6367.9123463,6374.7508896,6397.8742901,6404.2927505,6422.2346684,6435.9539540,6445.3675344,6451.7765297])\n"
    return [
        {
            'setup': setup,
            'call': 'np.array(derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6).shape)',
            'gold_call': 'np.array(_oracle_derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6).shape)',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': 'derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'gold_call': '_oracle_derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'tol': 0.0,
        },
        {
            'setup': setup,
            'call': '(lambda result: float(np.dot(result.ravel(), np.arange(1, result.size + 1))))(derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'gold_call': '(lambda result: float(np.dot(result.ravel(), np.arange(1, result.size + 1))))(_oracle_derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-06))',
            'tol': 0.0,
        },
        {
            'setup': setup + 'order = np.array([18,0,7,3,11,5,16,1,14,8,6,2,17,10,4,12,9,15,13])\nids, scans, freq = ids[order], scans[order], freq[order]\n',
            'call': 'derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'gold_call': '_oracle_derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 8e-6)',
            'tol': 0.0,
        },
        {
            'setup': setup + '\ndef run_model():\n    try:\n        derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 1e-12)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_derive_local_vertex_constraints(ids.copy(), scans.copy(), freq.copy(), 0.0046, 0.0049, 1e-12)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'tol': 0.0,
        },
    ]
