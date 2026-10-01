#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
 
def screen_signal_indices(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
) -> "np.ndarray":
    x = np.asarray(log_bf, dtype=float)
    c_raw = np.asarray(collection_id)
    t_raw = np.asarray(tested_variant_mask)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("log_bf must be a non-empty two-dimensional array")
    if c_raw.ndim != 1 or c_raw.shape[0] != x.shape[0]:
        raise ValueError("collection_id must align with signal rows")
    if c_raw.dtype.kind not in "iu" or np.any(c_raw < 0):
        raise ValueError("collection_id must contain non-negative integers")
    if t_raw.ndim != 2 or t_raw.shape[1] != x.shape[1] or t_raw.shape[0] < 1:
        raise ValueError("tested_variant_mask must be collection by variant")
    if t_raw.dtype.kind != "b":
        raise ValueError("tested_variant_mask must be boolean")
    if np.any(c_raw >= t_raw.shape[0]):
        raise ValueError("collection_id is out of range")
    c = c_raw.astype(int, copy=False)
    if np.any(np.isinf(x)):
        raise ValueError("log_bf cannot contain infinity")
    keep = []
    for i in range(x.shape[0]):
        available = t_raw[c[i]] & np.isfinite(x[i]) & (x[i] > -100000.0)
        if np.any(available) and float(np.max(x[i, available])) >= 5.0:
            keep.append(i)
    return np.asarray(keep, dtype=int)

import numpy as np
 
def prepare_signal_matrix(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    retained_indices: "np.ndarray",
) -> "np.ndarray":
    x = np.asarray(log_bf, dtype=float)
    c_raw = np.asarray(collection_id)
    t_raw = np.asarray(tested_variant_mask)
    r_raw = np.asarray(retained_indices)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("log_bf must be non-empty and two-dimensional")
    if c_raw.ndim != 1 or c_raw.shape[0] != x.shape[0] or c_raw.dtype.kind not in "iu":
        raise ValueError("collection_id must be an aligned integer vector")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    if t_raw.ndim != 2 or t_raw.shape[1] != x.shape[1] or t_raw.dtype.kind != "b":
        raise ValueError("tested_variant_mask must be a boolean collection-by-variant array")
    if np.any(c_raw >= t_raw.shape[0]):
        raise ValueError("collection identifier out of range")
    c = c_raw.astype(int, copy=False)
    if r_raw.ndim != 1 or r_raw.size < 1 or r_raw.dtype.kind not in "iu":
        raise ValueError("retained_indices must be a non-empty integer vector")
    r = r_raw.astype(int, copy=False)
    if np.any(r < 0) or np.any(r >= x.shape[0]) or np.any(np.diff(r) <= 0):
        raise ValueError("retained_indices must be strictly increasing and in range")
    if np.any(np.isinf(x)):
        raise ValueError("log_bf cannot contain infinity")
    values = np.full((r.size, x.shape[1]), -1000000.0, dtype=float)
    observed = np.zeros((r.size, x.shape[1]), dtype=float)
    for out_i, src_i in enumerate(r):
        mask = t_raw[c[src_i]] & np.isfinite(x[src_i]) & (x[src_i] > -100000.0)
        values[out_i, mask] = x[src_i, mask]
        observed[out_i, mask] = 1.0
    return np.stack((values, observed), axis=0)

import numpy as np
 
def collection_shared_masks(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
) -> "np.ndarray":
    p = np.asarray(prepared, dtype=float)
    c_raw = np.asarray(retained_collection_id)
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[1] < 1 or p.shape[2] < 1:
        raise ValueError("prepared must have shape (2, signals, variants)")
    if c_raw.ndim != 1 or c_raw.shape[0] != p.shape[1] or c_raw.dtype.kind not in "iu":
        raise ValueError("retained_collection_id must be an aligned integer vector")
    if not np.all(np.isfinite(p)):
        raise ValueError("prepared planes must be finite")
    observed = p[1]
    if np.any((observed != 0.0) & (observed != 1.0)):
        raise ValueError("observation plane must contain only zero and one")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    unique_raw = np.unique(c_raw)
    if not np.array_equal(unique_raw, np.arange(unique_raw.size, dtype=unique_raw.dtype)):
        raise ValueError("collection identifiers must be contiguous from zero")
    c = c_raw.astype(int, copy=False)
    unique = np.unique(c)
    universe = np.zeros((unique.size, p.shape[2]), dtype=bool)
    for k in range(unique.size):
        universe[k] = np.any(observed[c == k] == 1.0, axis=0)
        if not np.any(universe[k]):
            raise ValueError("every collection needs at least one observed variant")
    return universe[:, None, :] & universe[None, :, :]

import numpy as np
 
def _coverage_logsumexp(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return -np.inf
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m), dtype=float)))
 
def reciprocal_signal_coverage(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
) -> "np.ndarray":
    p = np.asarray(prepared, dtype=float)
    c_raw = np.asarray(retained_collection_id)
    s_raw = np.asarray(shared_masks)
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[1] < 1 or p.shape[2] < 1:
        raise ValueError("prepared must have shape (2, signals, variants)")
    if not np.all(np.isfinite(p)):
        raise ValueError("prepared must be finite")
    obs = p[1]
    if np.any((obs != 0.0) & (obs != 1.0)) or np.any(np.sum(obs, axis=1) == 0.0):
        raise ValueError("each signal must have a valid binary observation row")
    if c_raw.ndim != 1 or c_raw.shape[0] != p.shape[1] or c_raw.dtype.kind not in "iu":
        raise ValueError("retained_collection_id must be aligned integers")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    unique_raw = np.unique(c_raw)
    if not np.array_equal(unique_raw, np.arange(unique_raw.size, dtype=unique_raw.dtype)):
        raise ValueError("collection identifiers must be contiguous from zero")
    c = c_raw.astype(int, copy=False)
    n_collections = unique_raw.size
    if s_raw.ndim != 3 or s_raw.shape != (n_collections, n_collections, p.shape[2]):
        raise ValueError("shared_masks has the wrong shape")
    if s_raw.dtype.kind != "b" or not np.array_equal(s_raw, np.swapaxes(s_raw, 0, 1)):
        raise ValueError("shared_masks must be boolean and symmetric")
    n = p.shape[1]
    den = np.array([_coverage_logsumexp(p[0, i, obs[i] == 1.0]) for i in range(n)])
    out = np.zeros((n, n, 2), dtype=float)
    for i in range(n):
        for j in range(n):
            common = s_raw[c[i], c[j]] & (obs[i] == 1.0) & (obs[j] == 1.0)
            if np.any(common):
                out[i, j, 0] = np.exp(_coverage_logsumexp(p[0, i, common]) - den[i])
                out[i, j, 1] = np.exp(_coverage_logsumexp(p[0, j, common]) - den[j])
    return out

import numpy as np
 
def _evidence_lse(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    if a.size == 0:
        return -np.inf
    finite = np.isfinite(a)
    if not np.any(finite):
        return -np.inf
    m = float(np.max(a[finite]))
    return m + float(np.log(np.sum(np.exp(a[finite] - m), dtype=float)))
 
def batched_hypothesis_evidence(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    chunk_size: int,
) -> "np.ndarray":
    p = np.asarray(prepared, dtype=float)
    c_raw = np.asarray(retained_collection_id)
    s_raw = np.asarray(shared_masks)
    if p.ndim != 3 or p.shape[0] != 2 or p.shape[1] < 1 or p.shape[2] < 1 or not np.all(np.isfinite(p)):
        raise ValueError("prepared must be finite with shape (2, signals, variants)")
    if np.any((p[1] != 0.0) & (p[1] != 1.0)) or np.any(np.sum(p[1], axis=1) == 0.0):
        raise ValueError("observation plane must be binary")
    if c_raw.ndim != 1 or c_raw.shape[0] != p.shape[1] or c_raw.dtype.kind not in "iu":
        raise ValueError("retained_collection_id must be aligned integers")
    if np.any(c_raw < 0):
        raise ValueError("collection identifiers must be non-negative")
    unique_raw = np.unique(c_raw)
    if not np.array_equal(unique_raw, np.arange(unique_raw.size, dtype=unique_raw.dtype)):
        raise ValueError("collection identifiers must be contiguous from zero")
    c = c_raw.astype(int, copy=False)
    nc = unique_raw.size
    if s_raw.shape != (nc, nc, p.shape[2]) or s_raw.dtype.kind != "b" or not np.array_equal(s_raw, np.swapaxes(s_raw, 0, 1)):
        raise ValueError("shared_masks must have the correct boolean symmetric shape")
    observed_values = p[0][p[1] == 1.0]
    if observed_values.size == 0 or np.any(np.abs(observed_values) > np.finfo(float).max / 2.0):
        raise ValueError("observed values must permit finite pairwise sums")
    priors = np.asarray([p1, p2, p12], dtype=float)
    if not np.all(np.isfinite(priors)) or np.any(priors <= 0.0):
        raise ValueError("prior coefficients must be positive and finite")
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    n = p.shape[1]
    out = np.full((n, n, 5), -np.inf, dtype=float)
    out[:, :, 0] = 0.0
    for block_start in range(0, n, int(chunk_size)):
        for i in range(block_start, min(block_start + int(chunk_size), n)):
            for j in range(n):
                shared = s_raw[c[i], c[j]]
                if not np.any(shared):
                    continue
                a = p[0, i, shared]
                b = p[0, j, shared]
                h1_core = _evidence_lse(a)
                h2_core = _evidence_lse(b)
                h4_core = _evidence_lse(a + b)
                cross = a[:, None] + b[None, :]
                distinct = cross[~np.eye(a.size, dtype=bool)]
                h3_core = _evidence_lse(distinct)
                out[i, j, 1] = np.log(float(p1)) + h1_core
                out[i, j, 2] = np.log(float(p2)) + h2_core
                if np.isfinite(h3_core):
                    out[i, j, 3] = np.log(float(p1)) + np.log(float(p2)) + h3_core
                out[i, j, 4] = np.log(float(p12)) + h4_core
    return out

import numpy as np
 
def _edge_lse(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    if not np.any(finite):
        return -np.inf
    m = float(np.max(a[finite]))
    return m + float(np.log(np.sum(np.exp(a[finite] - m), dtype=float)))
 
def select_colocalisation_edges(
    log_evidence: "np.ndarray",
    coverage: "np.ndarray",
    overlap_min: float,
    h4_threshold: float,
) -> "np.ndarray":
    e = np.asarray(log_evidence, dtype=float)
    c = np.asarray(coverage, dtype=float)
    if e.ndim != 3 or e.shape[0] < 1 or e.shape[0] != e.shape[1] or e.shape[2] != 5:
        raise ValueError("log_evidence must have shape (n, n, 5)")
    if c.shape != (e.shape[0], e.shape[1], 2):
        raise ValueError("coverage must have shape (n, n, 2)")
    if np.any(np.isnan(e)) or np.any(np.isposinf(e)):
        raise ValueError("log evidence may contain negative infinity but not NaN or positive infinity")
    if not np.all(np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0 + 1e-12):
        raise ValueError("coverage must be finite and lie in [0,1]")
    thresholds = np.asarray([overlap_min, h4_threshold], dtype=float)
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0.0) or np.any(thresholds > 1.0):
        raise ValueError("thresholds must lie in [0,1]")
    rows = []
    for i in range(e.shape[0]):
        for j in range(i + 1, e.shape[0]):
            if min(c[i, j, 0], c[i, j, 1]) < float(overlap_min):
                continue
            normalizer = _edge_lse(e[i, j])
            posterior = 0.0 if not np.isfinite(e[i, j, 4]) else float(np.exp(e[i, j, 4] - normalizer))
            if posterior >= float(h4_threshold):
                rows.append((float(i), float(j), posterior))
    return np.asarray(rows, dtype=float).reshape((-1, 3))

import numpy as np
 
def restrict_credible_set_edges(
    edges: "np.ndarray",
    has_credible_set: "np.ndarray",
) -> "np.ndarray":
    e = np.asarray(edges, dtype=float)
    h_raw = np.asarray(has_credible_set)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError("edges must have shape (n_edges,3)")
    if h_raw.ndim != 1 or h_raw.size < 1 or h_raw.dtype.kind != "b":
        raise ValueError("has_credible_set must be a non-empty boolean vector")
    if not np.all(np.isfinite(e)):
        raise ValueError("edge values must be finite")
    if e.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    endpoints = e[:, :2]
    if not np.all(endpoints == np.floor(endpoints)):
        raise ValueError("edge endpoints must be integer-valued")
    ij = endpoints.astype(int)
    if np.any(ij < 0) or np.any(ij >= h_raw.size) or np.any(ij[:, 0] >= ij[:, 1]):
        raise ValueError("edge endpoints must satisfy 0 <= i < j < n")
    if len({tuple(row) for row in ij.tolist()}) != ij.shape[0]:
        raise ValueError("duplicate edges are not allowed")
    if np.any(e[:, 2] < 0.0) or np.any(e[:, 2] > 1.0):
        raise ValueError("edge weights must lie in [0,1]")
    keep = h_raw[ij[:, 0]] & h_raw[ij[:, 1]]
    return e[keep].astype(float, copy=True).reshape((-1, 3))

import numpy as np
 
def summarize_collider_components(
    edges: "np.ndarray",
    signal_metadata: "np.ndarray",
) -> "np.ndarray":
    e = np.asarray(edges, dtype=float)
    m_raw = np.asarray(signal_metadata)
    if e.ndim != 2 or e.shape[1] != 3 or not np.all(np.isfinite(e)):
        raise ValueError("edges must be a finite array with three columns")
    if m_raw.ndim != 2 or m_raw.shape[0] < 1 or m_raw.shape[1] != 3 or m_raw.dtype.kind not in "iu":
        raise ValueError("signal_metadata must be a non-empty integer array with three columns")
    m = m_raw
    if len({tuple(row) for row in m.tolist()}) != m.shape[0]:
        raise ValueError("complete signal identities must be unique")
    if e.shape[0] == 0:
        return np.empty((0, 4), dtype=int)
    if np.any(e[:, 2] < 0.0) or np.any(e[:, 2] > 1.0):
        raise ValueError("edge weights must lie in [0,1]")
    if not np.all(e[:, :2] == np.floor(e[:, :2])):
        raise ValueError("edge endpoints must be integer-valued")
    ij = e[:, :2].astype(int)
    if np.any(ij < 0) or np.any(ij >= m.shape[0]) or np.any(ij[:, 0] == ij[:, 1]):
        raise ValueError("edge endpoints are invalid")
    canonical = np.sort(ij, axis=1)
    if len({tuple(row) for row in canonical.tolist()}) != canonical.shape[0]:
        raise ValueError("duplicate undirected edges are not allowed")
    adjacency = {}
    for i, j in canonical:
        adjacency.setdefault(int(i), set()).add(int(j))
        adjacency.setdefault(int(j), set()).add(int(i))
    seen = set()
    rows = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        node_set = set(nodes)
        edge_count = sum(int(i) in node_set and int(j) in node_set for i, j in canonical)
        identities = [tuple(m[node, :2]) for node in nodes]
        collider = int(len(set(identities)) < len(identities))
        rows.append((min(nodes), len(nodes), int(edge_count), collider))
    rows.sort(key=lambda row: row[0])
    return np.asarray(rows, dtype=int).reshape((-1, 4))

import numpy as np
 
def _validate_component_table_rate(table, name):
    raw = np.asarray(table)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != 4 or raw.dtype.kind not in "iu":
        raise ValueError(name + " must be a non-empty integer array with four columns")
    x = raw.astype(int, copy=False)
    if np.any(x[:, 0] < 0) or np.any(x[:, 1] < 2) or np.any(x[:, 2] < 1):
        raise ValueError(name + " contains invalid component counts")
    if np.any(np.diff(x[:, 0]) <= 0):
        raise ValueError(name + " must be ordered by unique minimum node")
    if np.any(x[:, 2] < x[:, 1] - 1) or np.any(x[:, 2] > x[:, 1] * (x[:, 1] - 1) // 2):
        raise ValueError(name + " has impossible connected-component edge counts")
    if np.any((x[:, 3] != 0) & (x[:, 3] != 1)):
        raise ValueError(name + " collider flags must be binary")
    return x
 
def collider_rate_contrast(
    full_components: "np.ndarray",
    restricted_components: "np.ndarray",
) -> float:
    full = _validate_component_table_rate(full_components, "full_components")
    restricted = _validate_component_table_rate(restricted_components, "restricted_components")
    value = 100.0 * float(np.mean(restricted[:, 3], dtype=float)) - 100.0 * float(np.mean(full[:, 3], dtype=float))
    if not np.isfinite(value):
        raise ValueError("contrast must be finite")
    return float(value)

import numpy as np
 
def evaluate_colocalisation_audit(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    signal_metadata: "np.ndarray",
    has_credible_set: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    overlap_min: float,
    h4_threshold: float,
    chunk_size: int,
) -> float:
    metadata_raw = np.asarray(signal_metadata)
    credible_raw = np.asarray(has_credible_set)
    log_bf_raw = np.asarray(log_bf)
    n_signals = log_bf_raw.shape[0] if log_bf_raw.ndim == 2 else -1
    if metadata_raw.ndim != 2 or metadata_raw.shape != (n_signals, 3) or metadata_raw.dtype.kind not in "iu":
        raise ValueError("signal_metadata must be an aligned integer array with three columns")
    if credible_raw.ndim != 1 or credible_raw.shape[0] != n_signals or credible_raw.dtype.kind != "b":
        raise ValueError("has_credible_set must be an aligned boolean vector")
    retained = screen_signal_indices(log_bf, collection_id, tested_variant_mask)
    if retained.size < 2:
        raise ValueError("at least two signals must survive screening")
    prepared = prepare_signal_matrix(log_bf, collection_id, tested_variant_mask, retained)
    retained_collection_labels = np.asarray(collection_id)[retained]
    _, retained_collections = np.unique(retained_collection_labels, return_inverse=True)
    retained_collections = retained_collections.astype(int, copy=False)
    shared = collection_shared_masks(prepared, retained_collections)
    coverage = reciprocal_signal_coverage(prepared, retained_collections, shared)
    evidence = batched_hypothesis_evidence(prepared, retained_collections, shared, p1, p2, p12, chunk_size)
    edges = select_colocalisation_edges(evidence, coverage, overlap_min, h4_threshold)
    restricted_edges = restrict_credible_set_edges(edges, credible_raw[retained])
    retained_metadata = metadata_raw[retained]
    full_components = summarize_collider_components(edges, retained_metadata)
    restricted_components = summarize_collider_components(restricted_edges, retained_metadata)
    return collider_rate_contrast(full_components, restricted_components)
SCICODE_GOLD_EOF
