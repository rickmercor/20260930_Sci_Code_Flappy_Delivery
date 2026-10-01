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


def build_ising_tensor_network(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    K_h = np.asarray(K_h, dtype=float)
    K_v = np.asarray(K_v, dtype=float)
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 1:
        raise ValueError("h must have shape (R,C) with R,C >= 1")
    R, C = h.shape
    if K_h.shape != (R, max(C - 1, 0)):
        raise ValueError("K_h must have shape (R,C-1)")
    if K_v.shape != (max(R - 1, 0), C):
        raise ValueError("K_v must have shape (R-1,C)")
    if not (np.all(np.isfinite(K_h)) and np.all(np.isfinite(K_v)) and np.all(np.isfinite(h))):
        raise ValueError("all inputs must be finite")
    if np.any(K_h < 0.0) or np.any(K_v < 0.0):
        raise ValueError("couplings must be nonnegative")

    records = []
    for r in range(R):
        for c in range(C - 1):
            u = r * C + c
            records.append((u, u + 1, float(K_h[r, c])))
    for r in range(R - 1):
        for c in range(C):
            u = r * C + c
            records.append((u, (r + 1) * C + c, float(K_v[r, c])))
    records.sort(key=lambda x: (x[0], x[1]))

    E = len(records)
    N = R * C
    edges = np.asarray([(u, v) for u, v, _ in records], dtype=np.int64).reshape(E, 2)
    spins = np.array([-1.0, 1.0], dtype=float)
    bond_factors = np.empty((E, 2, 2), dtype=float)
    for e, (_, _, K) in enumerate(records):
        bond_factors[e, :, 0] = np.sqrt(np.cosh(K))
        bond_factors[e, :, 1] = spins * np.sqrt(np.sinh(K))

    neighbor_lists = [[] for _ in range(N)]
    for u, v in edges:
        neighbor_lists[int(u)].append(int(v))
        neighbor_lists[int(v)].append(int(u))
    for lst in neighbor_lists:
        lst.sort()
    degrees = np.asarray([len(lst) for lst in neighbor_lists], dtype=np.int64)
    neighbors = np.full((N, 4), -1, dtype=np.int64)
    for v, lst in enumerate(neighbor_lists):
        neighbors[v, : len(lst)] = lst

    edge_index = {tuple(row): i for i, row in enumerate(edges.tolist())}
    tensors = np.zeros((N, 16), dtype=float)
    for v in range(N):
        d = int(degrees[v])
        for bits in itertools.product((0, 1), repeat=d):
            flat = 0
            for b in bits:
                flat = 2 * flat + b
            value = 0.0
            for spin_index, spin in enumerate(spins):
                term = np.exp(h.flat[v] * spin)
                for j, nbr in enumerate(neighbor_lists[v]):
                    e = edge_index[(min(v, nbr), max(v, nbr))]
                    term *= bond_factors[e, spin_index, bits[j]]
                value += term
            tensors[v, flat] = value
    return edges, neighbors, degrees, tensors, bond_factors

import itertools
import numpy as np


def solve_bp_messages(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", tol: float,
) -> "np.ndarray":
    edges = np.asarray(edges, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    degrees = np.asarray(degrees, dtype=np.int64)
    tensors = np.asarray(tensors, dtype=float)
    tol = float(tol)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E,2)")
    N = degrees.size
    if neighbors.shape != (N, 4) or tensors.shape != (N, 16):
        raise ValueError("network arrays have inconsistent shapes")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    E = edges.shape[0]
    if E == 0:
        return np.empty((0, 2, 2), dtype=float)

    edge_index = {tuple(row): i for i, row in enumerate(edges.tolist())}
    messages = np.zeros((E, 2, 2), dtype=float)
    messages[:, :, 0] = 1.0
    for _ in range(100000):
        updated = np.empty_like(messages)
        for e, (u0, v0) in enumerate(edges):
            u = int(u0)
            v = int(v0)
            for direction, src, dst in ((0, u, v), (1, v, u)):
                d = int(degrees[src])
                nbrs = neighbors[src, :d]
                matches = np.flatnonzero(nbrs == dst)
                if matches.size != 1:
                    raise ValueError("edge and neighbor tables are inconsistent")
                outgoing_axis = int(matches[0])
                vector = np.zeros(2, dtype=float)
                for bits in itertools.product((0, 1), repeat=d):
                    flat = 0
                    for b in bits:
                        flat = 2 * flat + b
                    term = tensors[src, flat]
                    for j, nbr0 in enumerate(nbrs):
                        nbr = int(nbr0)
                        if nbr == dst:
                            continue
                        ee = edge_index[(min(src, nbr), max(src, nbr))]
                        incoming = messages[ee, 0] if nbr < src else messages[ee, 1]
                        term *= incoming[bits[j]]
                    vector[bits[outgoing_axis]] += term
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError("message update has zero or nonfinite norm")
                vector /= norm
                if vector[0] < 0.0:
                    vector = -vector
                updated[e, direction] = vector
        change = float(np.max(np.linalg.norm(updated - messages, axis=2)))
        messages = updated
        if change < tol:
            return messages
    raise RuntimeError("BP did not converge within the safety limit")

import itertools
import numpy as np


def compute_bp_vacuum(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", messages: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]":
    edges = np.asarray(edges, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    degrees = np.asarray(degrees, dtype=np.int64)
    tensors = np.asarray(tensors, dtype=float)
    messages = np.asarray(messages, dtype=float)
    E = edges.shape[0]
    N = degrees.size
    if edges.ndim != 2 or edges.shape[1] != 2 or neighbors.shape != (N, 4) or tensors.shape != (N, 16):
        raise ValueError("network arrays have inconsistent shapes")
    if messages.shape != (E, 2, 2):
        raise ValueError("messages must have shape (E,2,2)")

    edge_index = {tuple(row): i for i, row in enumerate(edges.tolist())}
    overlaps = np.empty(E, dtype=float)
    projectors0 = np.empty((E, 2, 2), dtype=float)
    projectors_perp = np.empty((E, 2, 2), dtype=float)
    for e in range(E):
        overlap = float(np.dot(messages[e, 0], messages[e, 1]))
        if not np.isfinite(overlap) or overlap <= 0.0:
            raise ValueError("edge overlap must be finite and positive")
        overlaps[e] = overlap
        projectors0[e] = np.outer(messages[e, 1], messages[e, 0]) / overlap
        projectors_perp[e] = np.eye(2, dtype=float) - projectors0[e]

    site_factors = np.empty(N, dtype=float)
    for v in range(N):
        d = int(degrees[v])
        nbrs = neighbors[v, :d]
        value = 0.0
        for bits in itertools.product((0, 1), repeat=d):
            flat = 0
            for b in bits:
                flat = 2 * flat + b
            term = tensors[v, flat]
            for j, nbr0 in enumerate(nbrs):
                nbr = int(nbr0)
                ee = edge_index[(min(v, nbr), max(v, nbr))]
                incoming = messages[ee, 0] if nbr < v else messages[ee, 1]
                term *= incoming[bits[j]] / np.sqrt(overlaps[ee])
            value += term
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("site factor must be finite and positive")
        site_factors[v] = value
    F0 = float(np.sum(np.log(site_factors)))
    return site_factors, overlaps, projectors0, projectors_perp, F0

import numpy as np


def enumerate_connected_generalized_loops(
    edges: "np.ndarray", max_weight: int,
) -> "np.ndarray":
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E,2)")
    max_weight = int(max_weight)
    if max_weight < 0:
        raise ValueError("max_weight must be nonnegative")
    E = edges.shape[0]
    if E == 0:
        return np.empty((0, 0), dtype=np.int64)
    if np.any(edges[:, 0] < 0) or np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError("each edge must satisfy 0 <= u < v")
    if len({tuple(row) for row in edges.tolist()}) != E:
        raise ValueError("edges must be unique")

    n_vertices = int(np.max(edges)) + 1
    found = []
    for mask_value in range(1, 1 << E):
        weight = mask_value.bit_count()
        if weight > max_weight:
            continue
        degree = np.zeros(n_vertices, dtype=np.int64)
        adjacency = [[] for _ in range(n_vertices)]
        selected = []
        for e, (u0, v0) in enumerate(edges):
            if (mask_value >> e) & 1:
                u = int(u0)
                v = int(v0)
                selected.append(e)
                degree[u] += 1
                degree[v] += 1
                adjacency[u].append(v)
                adjacency[v].append(u)
        incident = np.flatnonzero(degree)
        if incident.size == 0 or np.any(degree[incident] < 2):
            continue
        start = int(incident[0])
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != incident.size:
            continue
        row = np.zeros(E, dtype=np.int64)
        row[selected] = 1
        found.append((weight, mask_value, row))
    found.sort(key=lambda item: (item[0], item[1]))
    if not found:
        return np.empty((0, E), dtype=np.int64)
    return np.stack([item[2] for item in found], axis=0)

import numpy as np


def compute_normalized_loop_weights(
    edges: "np.ndarray", h: "np.ndarray", bond_factors: "np.ndarray",
    projectors0: "np.ndarray", projectors_perp: "np.ndarray",
    site_factors: "np.ndarray", loops: "np.ndarray",
) -> "np.ndarray":
    edges = np.asarray(edges, dtype=np.int64)
    h = np.asarray(h, dtype=float)
    bond_factors = np.asarray(bond_factors, dtype=float)
    projectors0 = np.asarray(projectors0, dtype=float)
    projectors_perp = np.asarray(projectors_perp, dtype=float)
    site_factors = np.asarray(site_factors, dtype=float)
    loops = np.asarray(loops, dtype=np.int64)
    E = edges.shape[0]
    N = h.size
    if edges.ndim != 2 or edges.shape[1] != 2 or h.ndim != 2:
        raise ValueError("edges and h have invalid shapes")
    if bond_factors.shape != (E, 2, 2) or projectors0.shape != (E, 2, 2) or projectors_perp.shape != (E, 2, 2):
        raise ValueError("edge-factor arrays have inconsistent shapes")
    if site_factors.shape != (N,) or not np.all(np.isfinite(site_factors)) or np.any(site_factors <= 0.0):
        raise ValueError("site_factors must be finite and positive")
    if loops.ndim != 2 or loops.shape[1] != E or np.any((loops != 0) & (loops != 1)):
        raise ValueError("loops must be a binary array with shape (L,E)")
    L = loops.shape[0]
    if L == 0:
        return np.empty(0, dtype=float)

    state_ids = np.arange(1 << N, dtype=np.uint64)[:, None]
    bit_positions = np.arange(N, dtype=np.uint64)
    spin_indices = ((state_ids >> bit_positions) & 1).astype(np.int8)
    spins = 2.0 * spin_indices - 1.0
    field_values = np.exp(spins @ h.reshape(-1))

    vacuum_edge_values = np.empty((E, field_values.size), dtype=float)
    excited_edge_values = np.empty_like(vacuum_edge_values)
    for e, (u0, v0) in enumerate(edges):
        u = int(u0)
        v = int(v0)
        left = bond_factors[e, spin_indices[:, u], :]
        right = bond_factors[e, spin_indices[:, v], :]
        vacuum_edge_values[e] = np.einsum("ni,ij,nj->n", left, projectors0[e], right)
        excited_edge_values[e] = np.einsum("ni,ij,nj->n", left, projectors_perp[e], right)

    Z0 = float(np.prod(site_factors))
    if not np.isfinite(Z0) or Z0 <= 0.0:
        raise ValueError("vacuum normalization must be finite and positive")
    result = np.empty(L, dtype=float)
    for ell, mask in enumerate(loops):
        values = field_values.copy()
        for e in range(E):
            values *= excited_edge_values[e] if mask[e] else vacuum_edge_values[e]
        result[ell] = float(np.sum(values) / Z0)
    return result

import math
import numpy as np


def _step06_interaction_graph(loop_indices, vertex_sets):
    m = len(loop_indices)
    graph_edges = []
    adjacency = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if loop_indices[i] == loop_indices[j] or (vertex_sets[loop_indices[i]] & vertex_sets[loop_indices[j]]):
                graph_edges.append((i, j))
                adjacency[i].append(j)
                adjacency[j].append(i)
    if m == 1:
        return True, graph_edges
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == m, graph_edges


def _step06_ursell(loop_indices, graph_edges):
    m = len(loop_indices)
    if m == 1:
        return 1.0
    signed_sum = 0
    q = len(graph_edges)
    for edge_mask in range(1, 1 << q):
        adjacency = [[] for _ in range(m)]
        n_selected = 0
        for k, (i, j) in enumerate(graph_edges):
            if (edge_mask >> k) & 1:
                adjacency[i].append(j)
                adjacency[j].append(i)
                n_selected += 1
        seen = {0}
        stack = [0]
        while stack:
            i = stack.pop()
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) == m:
            signed_sum += (-1) ** n_selected
    _, counts = np.unique(loop_indices, return_counts=True)
    denominator = 1
    for count in counts:
        denominator *= math.factorial(int(count))
    return float(signed_sum / denominator)


def enumerate_connected_loop_clusters(
    edges: "np.ndarray", loops: "np.ndarray", max_weight: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    edges = np.asarray(edges, dtype=np.int64)
    loops = np.asarray(loops, dtype=np.int64)
    max_weight = int(max_weight)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E,2)")
    E = edges.shape[0]
    if loops.ndim != 2 or loops.shape[1] != E or np.any((loops != 0) & (loops != 1)):
        raise ValueError("loops must be a binary array with shape (L,E)")
    if max_weight < 0:
        raise ValueError("max_weight must be nonnegative")
    L = loops.shape[0]
    if L == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

    loop_weights = loops.sum(axis=1).astype(np.int64)
    if np.any(loop_weights <= 0):
        raise ValueError("loop masks must be nonempty")
    vertex_sets = []
    for mask in loops:
        vertices = set()
        for e, (u0, v0) in enumerate(edges):
            if mask[e]:
                vertices.add(int(u0))
                vertices.add(int(v0))
        vertex_sets.append(vertices)

    records = []
    max_copies = max_weight // int(np.min(loop_weights)) if max_weight > 0 else 0

    def grow(start, chosen, total_weight):
        for loop_index in range(start, L):
            new_weight = total_weight + int(loop_weights[loop_index])
            if new_weight > max_weight:
                continue
            expanded = chosen + [loop_index]
            connected, graph_edges = _step06_interaction_graph(expanded, vertex_sets)
            if connected:
                coefficient = _step06_ursell(expanded, graph_edges)
                records.append((new_weight, len(expanded), tuple(expanded), coefficient))
            if len(expanded) < max_copies:
                grow(loop_index, expanded, new_weight)

    grow(0, [], 0)
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    multiplicities = np.zeros((len(records), L), dtype=np.int64)
    coefficients = np.empty(len(records), dtype=float)
    cluster_weights = np.empty(len(records), dtype=np.int64)
    for row, (total_weight, _, expanded, coefficient) in enumerate(records):
        for loop_index in expanded:
            multiplicities[row, loop_index] += 1
        coefficients[row] = coefficient
        cluster_weights[row] = total_weight
    return multiplicities, coefficients, cluster_weights

import numpy as np


def evaluate_cluster_correction(
    loop_weights: "np.ndarray", multiplicities: "np.ndarray",
    ursell_coefficients: "np.ndarray",
) -> "tuple[np.ndarray, float]":
    loop_weights = np.asarray(loop_weights, dtype=float)
    multiplicities = np.asarray(multiplicities)
    ursell_coefficients = np.asarray(ursell_coefficients, dtype=float)
    if loop_weights.ndim != 1 or multiplicities.ndim != 2 or multiplicities.shape[1] != loop_weights.size:
        raise ValueError("loop weights and multiplicities have inconsistent shapes")
    if ursell_coefficients.shape != (multiplicities.shape[0],):
        raise ValueError("coefficient length must match cluster rows")
    if not np.issubdtype(multiplicities.dtype, np.integer) or np.any(multiplicities < 0):
        raise ValueError("multiplicities must be nonnegative integers")
    if not (np.all(np.isfinite(loop_weights)) and np.all(np.isfinite(ursell_coefficients))):
        raise ValueError("weights and coefficients must be finite")
    if multiplicities.shape[0] == 0:
        return np.empty(0, dtype=float), 0.0
    contributions = ursell_coefficients * np.prod(
        np.power(loop_weights[None, :], multiplicities), axis=1
    )
    return contributions.astype(float), float(np.sum(contributions))

import numpy as np


def compute_cluster_free_energy(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray",
    max_weight: int, bp_tol: float,
) -> float:
    h_array = np.asarray(h, dtype=float)
    if h_array.ndim != 2 or h_array.size < 1:
        raise ValueError("h must be a nonempty rectangular field array")
    max_weight = int(max_weight)
    if max_weight < 0:
        raise ValueError("max_weight must be nonnegative")
    bp_tol = float(bp_tol)
    if not np.isfinite(bp_tol) or bp_tol <= 0.0:
        raise ValueError("bp_tol must be finite and positive")

    edges, neighbors, degrees, tensors, bond_factors = build_ising_tensor_network(
        K_h, K_v, h_array
    )
    messages = solve_bp_messages(
        edges, neighbors, degrees, tensors, bp_tol
    )
    site_factors, _, projectors0, projectors_perp, F0 = compute_bp_vacuum(
        edges, neighbors, degrees, tensors, messages
    )
    loops = enumerate_connected_generalized_loops(edges, max_weight)
    loop_weights = compute_normalized_loop_weights(
        edges, h_array, bond_factors, projectors0, projectors_perp,
        site_factors, loops
    )
    multiplicities, coefficients, _ = enumerate_connected_loop_clusters(
        edges, loops, max_weight
    )
    _, delta = evaluate_cluster_correction(
        loop_weights, multiplicities, coefficients
    )
    return float(-(F0 + delta) / h_array.size)
SCICODE_GOLD_EOF
