#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def hsqc_cost_matrices(
    spectrum_a: "np.ndarray",
    spectrum_b: "np.ndarray",
    sigma_h: float,
    sigma_c: float,
    functional_h: float,
    functional_c: float,
    penalty_factor: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    import numpy as np

    a = np.asarray(spectrum_a, dtype=float)
    b = np.asarray(spectrum_b, dtype=float)

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (2,)
        or b.shape[1:] != (2,)
    ):
        raise ValueError("spectra must have shape (n_peaks, 2)")

    if len(a) == 0 or len(b) == 0:
        raise ValueError("spectra must be nonempty")

    if min(sigma_h, sigma_c) <= 0 or min(functional_h, functional_c) < 0:
        raise ValueError("invalid scale")

    delta_h = (a[:, None, 0] - b[None, :, 0]) / sigma_h
    delta_c = (a[:, None, 1] - b[None, :, 1]) / sigma_c
    distances = np.hypot(delta_h, delta_c)

    tolerance = float(
        np.hypot(
            functional_h / sigma_h,
            functional_c / sigma_c,
        )
    )

    costs = np.where(
        distances <= tolerance,
        distances,
        distances + penalty_factor,
    )

    return distances, costs, tolerance

def modified_hungarian_distance(
    distances: "np.ndarray",
    costs: "np.ndarray",
    tolerance: float,
) -> "tuple[float, float]":
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    d = np.asarray(distances, dtype=float)
    c = np.asarray(costs, dtype=float)

    if d.ndim != 2 or d.shape != c.shape or min(d.shape) == 0:
        raise ValueError("equal nonempty matrices required")

    n_a, n_b = d.shape
    size = max(n_a, n_b)

    padded_cost = np.full(
        (size, size),
        float(tolerance),
        dtype=float,
    )
    padded_distance = np.full(
        (size, size),
        float(tolerance),
        dtype=float,
    )

    padded_cost[:n_a, :n_b] = c
    padded_distance[:n_a, :n_b] = d

    rows, cols = linear_sum_assignment(padded_cost)

    return (
        float(np.mean(padded_cost[rows, cols])),
        float(
            np.mean(
                padded_distance[rows, cols] <= tolerance
            )
        ),
    )

def rank_hsqc_library(
    distances: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    values = np.asarray(distances, dtype=float)

    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(values < 0)
    ):
        raise ValueError("nonempty nonnegative vector required")

    indices = np.arange(len(values), dtype=int)
    order = np.lexsort((indices, values))
    similarities = 1.0 / (1.0 + values)

    return order.astype(int), similarities

def build_hsqc_network(
    n_nodes: int,
    pair_records: "np.ndarray",
    distance_limit: float,
    hybrid_floor: float,
) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    records = np.asarray(pair_records, dtype=float)

    if (
        n_nodes < 1
        or records.ndim != 2
        or records.shape[1:] != (5,)
    ):
        raise ValueError("records must have five columns")

    edges = []
    weights = []
    seen = set()

    for raw_i, raw_j, distance, tanimoto, mcs in records:
        i = int(raw_i)
        j = int(raw_j)

        if (
            raw_i != i
            or raw_j != j
            or not (
                0 <= i < n_nodes
                and 0 <= j < n_nodes
            )
        ):
            raise ValueError("invalid node")

        edge = (
            min(i, j),
            max(i, j),
        )

        if i == j or edge in seen:
            raise ValueError(
                "pairs must be unique non-self edges"
            )

        seen.add(edge)
        hybrid = 0.5 * (
            float(tanimoto)
            + float(mcs)
        )

        if (
            distance < distance_limit
            and hybrid > hybrid_floor
        ):
            edges.append(edge)
            weights.append(hybrid)

    if not edges:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=float),
        )

    return (
        np.asarray(edges, dtype=int),
        np.asarray(weights, dtype=float),
    )

def insert_query_node(
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

def product_weighted_resource_allocation(
    n_nodes: int,
    edges: "np.ndarray",
    edge_hybrid: "np.ndarray",
    query_neighbors: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    edge_array = np.asarray(edges, dtype=int)
    weights = np.asarray(edge_hybrid, dtype=float)
    neighbors = np.asarray(query_neighbors, dtype=int)

    if (
        n_nodes < 1
        or edge_array.ndim != 2
        or edge_array.shape[1:] != (2,)
    ):
        raise ValueError(
            "edges must be an n-by-2 array"
        )

    if (
        weights.shape != (len(edge_array),)
        or neighbors.ndim != 1
    ):
        raise ValueError(
            "inconsistent graph arrays"
        )

    adjacency = [
        set()
        for _ in range(n_nodes)
    ]
    edge_weight = {}

    for (i, j), weight in zip(
        edge_array,
        weights,
    ):
        if (
            i == j
            or min(i, j) < 0
            or max(i, j) >= n_nodes
        ):
            raise ValueError("invalid edge")

        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))

        key = (
            min(int(i), int(j)),
            max(int(i), int(j)),
        )
        edge_weight[key] = float(weight)

    query_set = {
        int(node)
        for node in neighbors
    }

    updated_degree = np.array(
        [
            len(adjacency[node])
            + int(node in query_set)
            for node in range(n_nodes)
        ],
        dtype=float,
    )

    scores = np.zeros(
        n_nodes,
        dtype=float,
    )

    for candidate in range(n_nodes):
        shared_neighbors = (
            query_set.intersection(
                adjacency[candidate]
            )
        )

        for shared in shared_neighbors:
            key = (
                min(candidate, shared),
                max(candidate, shared),
            )
            scores[candidate] += (
                edge_weight[key]
                / updated_degree[shared]
            )

    return scores

def network_corrected_scores(
    query_distances: "np.ndarray",
    pwra_scores: "np.ndarray",
    distance_weight: float,
    network_weight: float,
) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    distances = np.asarray(
        query_distances,
        dtype=float,
    )
    pwra = np.asarray(
        pwra_scores,
        dtype=float,
    )

    if (
        distances.ndim != 1
        or distances.shape != pwra.shape
        or len(distances) == 0
    ):
        raise ValueError(
            "equal nonempty vectors required"
        )

    if (
        min(distance_weight, network_weight) < 0
        or distance_weight + network_weight <= 0
    ):
        raise ValueError("invalid weights")

    total = float(
        distance_weight + network_weight
    )
    distance_weight /= total
    network_weight /= total

    inverse = 1.0 / (1.0 + distances)

    pwra_span = float(np.max(pwra) - np.min(pwra))
    if pwra_span == 0:
        normalized_pwra = np.ones_like(pwra)
    else:
        normalized_pwra = (pwra - np.min(pwra)) / pwra_span

    inverse_span = float(np.max(inverse) - np.min(inverse))
    if inverse_span == 0:
        normalized_inverse = np.ones_like(inverse)
    else:
        normalized_inverse = (
            inverse - np.min(inverse)
        ) / inverse_span

    combined = (
        distance_weight * normalized_inverse
        + network_weight * normalized_pwra
    )

    return combined, normalized_pwra

def hsqc_retrieval_gain(
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
        distance_matrix, cost_matrix, tolerance = hsqc_cost_matrices(
            query_spectrum,
            library[node, : counts[node]],
            0.01,
            0.2,
            0.5,
            2.5,
            1.0,
        )
        query_distances[node], _ = modified_hungarian_distance(
            distance_matrix,
            cost_matrix,
            tolerance,
        )

    direct_order, _ = rank_hsqc_library(query_distances)
    edges, edge_hybrid = build_hsqc_network(
        n_nodes,
        pair_records,
        30.0,
        0.6,
    )
    query_neighbors = insert_query_node(
        query_distances,
        40.0,
        2,
    )

    if len(query_neighbors) == 0:
        corrected_order = direct_order
    else:
        pwra = product_weighted_resource_allocation(
            n_nodes,
            edges,
            edge_hybrid,
            query_neighbors,
        )
        composite, _ = network_corrected_scores(
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
SCICODE_GOLD_EOF
