#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import itertools
import numpy as np

def cluster_recurring_differences(
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

import itertools
import numpy as np

def instantiate_k22_motifs(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Reference implementation for instantiate_k22_motifs."""

    ids = np.asarray(line_ids)
    _, corrected, occurrences, counts = (
        cluster_recurring_differences(
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

import numpy as np

def derive_local_vertex_constraints(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Reference implementation for derive_local_vertex_constraints."""

    ids = np.asarray(line_ids)
    motifs = instantiate_k22_motifs(
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

import itertools
import numpy as np

def stitch_energy_level_cliques(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> "np.ndarray":
    """Reference implementation for stitch_energy_level_cliques."""
    ids = np.asarray(line_ids)
    sharing_pairs = derive_local_vertex_constraints(
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

from collections import deque
import numpy as np

def build_bipartite_level_network(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", int]:
    """Reference implementation for build_bipartite_level_network."""
    ids = np.asarray(line_ids)
    incidence = stitch_energy_level_cliques(
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

import numpy as np

def invert_network_and_predict(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", int, "np.ndarray"]:
    """Reference implementation for invert_network_and_predict."""
    upper, lower, endpoints, missing, gauge = (
        build_bipartite_level_network(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))
    reported = np.asarray(reported_frequencies, dtype=float)
    sigma = np.asarray(standard_uncertainties, dtype=float)
    if sigma.ndim != 1 or sigma.shape != reported.shape:
        raise ValueError("standard_uncertainties must be one-dimensional and aligned")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("standard_uncertainties must be finite and strictly positive")

    labels = np.asarray(scans).astype(str)
    n_upper, n_lower = upper.shape[0], lower.shape[0]
    n_parameters = n_upper + n_lower
    design = np.zeros((reported.size, n_parameters), dtype=float)
    lower_columns = {}
    next_column = n_upper
    for lower_index in range(n_lower):
        if lower_index != gauge:
            lower_columns[lower_index] = next_column
            next_column += 1
    offset_column = n_parameters - 1
    for row, (upper_index, lower_index) in enumerate(endpoints):
        design[row, int(upper_index)] = 1.0
        if int(lower_index) != gauge:
            design[row, lower_columns[int(lower_index)]] = -1.0
        design[row, offset_column] = float(labels[row] == "B")

    origin = float(reported[0])
    centred_reported = reported - origin
    weighted_design = design / sigma[:, None]
    weighted_rhs = centred_reported / sigma
    centred_parameters, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_rhs, rcond=None)
    rank = int(rank)
    if rank != n_parameters:
        raise ValueError("the weighted design matrix is rank deficient")

    residuals = centred_reported - design @ centred_parameters
    parameters = centred_parameters.copy()
    parameters[:n_upper] += origin
    missing_upper, missing_lower = map(int, missing)
    upper_energy = float(parameters[missing_upper])
    if missing_lower == gauge:
        lower_energy = 0.0
    else:
        lower_before = sum(index != gauge
                           for index in range(missing_lower))
        lower_energy = float(parameters[n_upper + lower_before])
    refined_offset = float(parameters[-1])
    predicted = upper_energy - lower_energy
    prediction = np.asarray(
        [predicted, refined_offset, upper_energy, lower_energy], dtype=float)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("the fitted prediction and endpoints must be finite")
    return (np.asarray(parameters, dtype=float),
            np.asarray(residuals, dtype=float), rank, prediction)

import numpy as np

def reconstruct_missing_transition_frequency(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> float:
    """Reference implementation for reconstruct_missing_transition_frequency."""
    seed, corrected, occurrences, counts = (
        cluster_recurring_differences(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))
    motifs = instantiate_k22_motifs(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    sharing_pairs = derive_local_vertex_constraints(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    level_incidence = stitch_energy_level_cliques(
        line_ids, scans, reported_frequencies, b_lo, b_hi,
        delta, min_occurrences)
    upper, lower, endpoints, missing, gauge = (
        build_bipartite_level_network(
            line_ids, scans, reported_frequencies, b_lo, b_hi,
            delta, min_occurrences))
    parameters, residuals, rank, prediction = invert_network_and_predict(
        line_ids, scans, reported_frequencies, standard_uncertainties,
        b_lo, b_hi, delta, min_occurrences)

    n_lines = np.asarray(line_ids).size
    if not np.isfinite(seed):
        raise ValueError("the topology seed is not finite")
    if corrected.shape != (n_lines,) or occurrences.shape[0] != int(np.sum(counts)):
        raise ValueError("the retained-difference representation is inconsistent")
    if motifs.ndim != 2 or motifs.shape[1] != 4 or sharing_pairs.shape[1] != 2:
        raise ValueError("the local-motif representation is inconsistent")
    if level_incidence.shape[0] != upper.shape[0] + lower.shape[0]:
        raise ValueError("the stitched levels and colour classes disagree")
    if endpoints.shape != (n_lines, 2) or missing.shape != (2,):
        raise ValueError("the bipartite endpoint representation is inconsistent")
    if not (0 <= int(gauge) < lower.shape[0]):
        raise ValueError("the gauge lower-level index is invalid")
    if parameters.size != upper.shape[0] + lower.shape[0] or rank != parameters.size:
        raise ValueError("the fitted parameter dimension or rank is inconsistent")
    if residuals.shape != (n_lines,):
        raise ValueError("the fitted residual vector is inconsistent")
    if prediction.shape != (4,):
        raise ValueError("the fitted prediction summary is inconsistent")
    if not np.isclose(parameters[-1], prediction[1], rtol=0.0, atol=1e-8):
        raise ValueError("the independently returned fitted offsets disagree")
    if not np.isclose(prediction[0], prediction[2] - prediction[3],
                      rtol=0.0, atol=1e-8):
        raise ValueError("the endpoint difference and prediction disagree")
    if not np.isfinite(prediction[0]):
        raise ValueError("the predicted frequency is not finite")
    return float(prediction[0])
SCICODE_GOLD_EOF
