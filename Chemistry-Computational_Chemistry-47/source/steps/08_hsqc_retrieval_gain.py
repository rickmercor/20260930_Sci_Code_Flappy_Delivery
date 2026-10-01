"""
Compose the seven preceding public functions into the source-protocol HSQC retrieval audit.

Candidate identity must remain aligned across padded spectra, graph records, and structural-score arrays. The pinned source governs stable ordering, insufficient-connectivity behavior, and the top-k comparison.

Returns
-------
Finite float containing the source-protocol top-k retrieval change.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hsqc_retrieval_gain(
    query_spectrum: "np.ndarray",
    library_peaks: "np.ndarray",
    peak_counts: "np.ndarray",
    pair_records: "np.ndarray",
    query_tanimoto: "np.ndarray",
    query_mcs: "np.ndarray",
    top_k: int = 3,
) -> float:
    """Return the source-protocol top-k retrieval change.

    Compose the seven preceding public functions while preserving candidate
    identity across padded spectra, graph records, and structural-score arrays.
    Apply the source-defined ordering and insufficient-connectivity conventions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_hsqc_retrieval_gain(
    query_spectrum: "np.ndarray",
    library_peaks: "np.ndarray",
    peak_counts: "np.ndarray",
    pair_records: "np.ndarray",
    query_tanimoto: "np.ndarray",
    query_mcs: "np.ndarray",
    top_k: int = 3,
) -> float:
    """Compose the seven preceding oracle subproblem functions."""
    import numpy as np

    library = np.asarray(library_peaks, dtype=float)
    counts = np.asarray(peak_counts, dtype=int)
    if library.ndim != 3 or library.shape[2] != 2 or counts.shape != (len(library),):
        raise ValueError("invalid padded library")
    if np.any(counts < 1) or np.any(counts > library.shape[1]):
        raise ValueError("invalid active peak count")

    n_nodes = len(library)
    tanimoto = np.asarray(query_tanimoto, dtype=float)
    mcs = np.asarray(query_mcs, dtype=float)
    if tanimoto.shape != (n_nodes,) or mcs.shape != (n_nodes,):
        raise ValueError("invalid query structural scores")
    if not 1 <= top_k <= n_nodes:
        raise ValueError("invalid top_k")

    query_distances = np.empty(n_nodes, dtype=float)
    for node in range(n_nodes):
        distance_matrix, cost_matrix, tolerance = _oracle_hsqc_cost_matrices(
            query_spectrum,
            library[node, : counts[node]],
            0.01,
            0.2,
            0.5,
            2.5,
            1.0,
        )
        query_distances[node], _ = _oracle_modified_hungarian_distance(
            distance_matrix,
            cost_matrix,
            tolerance,
        )

    direct_order, _ = _oracle_rank_hsqc_library(query_distances)
    edges, edge_hybrid = _oracle_build_hsqc_network(
        n_nodes,
        pair_records,
        30.0,
        0.6,
    )
    query_neighbors = _oracle_insert_query_node(
        query_distances,
        40.0,
        2,
    )

    if len(query_neighbors) == 0:
        corrected_order = direct_order
    else:
        pwra = _oracle_product_weighted_resource_allocation(
            n_nodes,
            edges,
            edge_hybrid,
            query_neighbors,
        )
        composite, _ = _oracle_network_corrected_scores(
            query_distances,
            pwra,
            0.8,
            0.2,
        )
        indices = np.arange(n_nodes, dtype=int)
        corrected_order = np.lexsort((indices, query_distances, -composite))

    query_hybrid = 0.5 * (tanimoto + mcs)
    maximum = float(np.max(query_hybrid))
    if maximum <= 0:
        raise ValueError("structural efficiency is undefined")

    eta_direct = float(np.max(query_hybrid[direct_order[:top_k]]) / maximum)
    eta_corrected = float(
        np.max(query_hybrid[corrected_order[:top_k]]) / maximum
    )
    if eta_direct <= 0:
        raise ValueError("structural efficiency is undefined")

    return float(100.0 * (eta_corrected - eta_direct) / eta_direct)


def _benchmark_fixture(
    spectrum_seed: int = 26082027,
    graph_seed: int = 52004,
    score_seed: int = 71009,
):
    import numpy as np

    rng = np.random.default_rng(spectrum_seed)
    n_nodes, n_peaks, max_peaks = 10, 5, 6
    p = np.arange(n_peaks, dtype=float)

    query = np.column_stack(
        (
            1.05 + 0.82 * p + 0.13 * p**2,
            16.0 + 14.0 * p + 4.5 * p**2,
        )
    )

    library = np.zeros((n_nodes, max_peaks, 2), dtype=float)
    for node in range(n_nodes):
        signs = np.where(
            (np.arange(n_peaks) + node) % 2 == 0,
            1.0,
            -1.0,
        )
        offsets = np.column_stack(
            (
                signs * (0.022 + 0.012 * node) * (1.0 + 0.07 * p),
                -signs * (0.35 + 0.16 * node) * (1.0 + 0.05 * p),
            )
        )
        shifted = query + offsets

        if node >= 7:
            shifted[:, 0] += (0.40, 0.35, 0.275)[node - 7]
        if node == 9:
            shifted[0, 0] += 0.70

        library[node, :n_peaks] = shifted[rng.permutation(n_peaks)]

    counts = np.full(n_nodes, n_peaks, dtype=int)
    counts[[4, 8]] = max_peaks
    counts[5] = n_peaks - 1
    library[4, 5] = [8.65, 184.0]
    library[8, 5] = [0.45, 172.0]

    graph_rng = np.random.default_rng(graph_seed)
    rows = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = 17.0 + 20.0 * graph_rng.random()
            hybrid = 0.45 + 0.45 * graph_rng.random()
            split = 0.08 * (2.0 * graph_rng.random() - 1.0)
            tanimoto = np.clip(hybrid + split, 0.0, 1.0)
            mcs = 2.0 * hybrid - tanimoto
            rows.append([i, j, distance, tanimoto, mcs])

    pair_records = np.asarray(rows, dtype=float)
    pair_records[0, 2:] = [30.0, 0.70, 0.74]
    pair_records[1, 2:] = [25.0, 0.58, 0.62]

    score_rng = np.random.default_rng(score_seed)
    query_hybrid = 0.28 + 0.66 * score_rng.random(n_nodes)
    split = 0.07 * (2.0 * score_rng.random(n_nodes) - 1.0)
    query_tanimoto = np.clip(query_hybrid + split, 0.0, 1.0)
    query_mcs = 2.0 * query_hybrid - query_tanimoto

    return (
        query,
        library,
        counts,
        pair_records,
        query_tanimoto,
        query_mcs,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\nimport numpy as np\nq,l,c,p,t,m=_benchmark_fixture()',
      'call': 'hsqc_retrieval_gain(copy.deepcopy(q), copy.deepcopy(l), copy.deepcopy(c), '
              'copy.deepcopy(p), copy.deepcopy(t), copy.deepcopy(m))',
      'gold_call': '_oracle_hsqc_retrieval_gain(copy.deepcopy(q), copy.deepcopy(l), '
                   'copy.deepcopy(c), copy.deepcopy(p), copy.deepcopy(t), copy.deepcopy(m))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'query = np.array([[0.0, 0.0]])\n'
               'normalized_distances = np.array(\n'
               '    [10.0, 10.0, 10.5, 11.0, 12.0, 40.0, 55.0]\n'
               ')\n'
               'library = np.zeros((7, 1, 2), dtype=float)\n'
               'library[:, 0, 0] = 0.01 * normalized_distances\n'
               'counts = np.ones(7, dtype=int)\n'
               'pair_records = np.array(\n'
               '    [\n'
               '        [4, 0, 20.0, 0.95, 0.95],\n'
               '        [1, 4, 20.0, 0.95, 0.95],\n'
               '        [4, 2, 20.0, 0.95, 0.95],\n'
               '        [3, 4, 20.0, 0.95, 0.95],\n'
               '        [1, 5, 20.0, 0.95, 0.95],\n'
               '    ],\n'
               '    dtype=float,\n'
               ')\n'
               'query_hybrid = np.array(\n'
               '    [0.20, 0.30, 0.40, 0.50, 0.95, 0.10, 0.10]\n'
               ')\n'
               'q = query\n'
               'l = library\n'
               'c = counts\n'
               'p = pair_records\n'
               't = query_hybrid.copy()\n'
               'm = query_hybrid.copy()\n',
      'call': 'hsqc_retrieval_gain(copy.deepcopy(q), copy.deepcopy(l), copy.deepcopy(c), '
              'copy.deepcopy(p), copy.deepcopy(t), copy.deepcopy(m), top_k=copy.deepcopy(1))',
      'gold_call': '_oracle_hsqc_retrieval_gain(copy.deepcopy(q), copy.deepcopy(l), '
                   'copy.deepcopy(c), copy.deepcopy(p), copy.deepcopy(t), copy.deepcopy(m), '
                   'top_k=copy.deepcopy(1))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'query = np.array([[0.0, 0.0]])\n'
               'normalized_distances = np.array(\n'
               '    [10.0, 10.0, 10.5, 11.0, 12.0, 40.0, 55.0]\n'
               ')\n'
               'library = np.zeros((7, 1, 2), dtype=float)\n'
               'library[:, 0, 0] = 0.01 * normalized_distances\n'
               'counts = np.ones(7, dtype=int)\n'
               'pair_records = np.array(\n'
               '    [\n'
               '        [4, 0, 20.0, 0.95, 0.95],\n'
               '        [1, 4, 20.0, 0.95, 0.95],\n'
               '        [4, 2, 20.0, 0.95, 0.95],\n'
               '        [3, 4, 20.0, 0.95, 0.95],\n'
               '        [1, 5, 20.0, 0.95, 0.95],\n'
               '    ],\n'
               '    dtype=float,\n'
               ')\n'
               'query_hybrid = np.array(\n'
               '    [0.20, 0.30, 0.40, 0.50, 0.95, 0.10, 0.10]\n'
               ')\n'
               'q = query\n'
               'l = library\n'
               'c = counts\n'
               'p = pair_records\n'
               't = query_hybrid.copy()\n'
               'm = query_hybrid.copy()\n'
               '\n'
               'order=np.array([1,5,4,0,6,2,3])\n'
               'inverse=np.empty_like(order)\n'
               'inverse[order]=np.arange(len(order))\n'
               'l=l[order]\n'
               'c=c[order]\n'
               't=t[order]\n'
               'm=m[order]\n'
               'p=p.copy()\n'
               'p[:,:2]=inverse[p[:,:2].astype(int)]\n',
      'call': 'hsqc_retrieval_gain(copy.deepcopy(q), copy.deepcopy(l), copy.deepcopy(c), '
              'copy.deepcopy(p), copy.deepcopy(t), copy.deepcopy(m), top_k=copy.deepcopy(1))',
      'gold_call': '_oracle_hsqc_retrieval_gain(copy.deepcopy(q), copy.deepcopy(l), '
                   'copy.deepcopy(c), copy.deepcopy(p), copy.deepcopy(t), copy.deepcopy(m), '
                   'top_k=copy.deepcopy(1))'}]
