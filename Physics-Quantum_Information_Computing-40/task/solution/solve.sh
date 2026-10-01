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

def min_sum_bp_trace(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_iters: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int) % 2
    p = np.asarray(p, dtype=float)
    s = np.asarray(syndrome, dtype=int) % 2
    M, N = H.shape
    edges = [(i, j) for i in range(M) for j in range(N) if H[i, j]]
    edge_index = {edge: k for k, edge in enumerate(edges)}
    row_neighbors = [np.flatnonzero(H[i]).tolist() for i in range(M)]
    col_neighbors = [np.flatnonzero(H[:, j]).tolist() for j in range(N)]
    prior = np.log((1.0 - p) / p)
    edge_msgs = np.asarray([prior[j] for i, j in edges], dtype=float)
    rows = []
    for _ in range(int(max_iters)):
        detector_msgs = np.zeros(len(edges), dtype=float)
        for k, (i, j) in enumerate(edges):
            incoming = np.asarray(
                [edge_msgs[edge_index[(i, jp)]] for jp in row_neighbors[i] if jp != j],
                dtype=float,
            )
            signs = np.where(incoming >= 0.0, 1.0, -1.0)
            detector_msgs[k] = ((-1.0) ** int(s[i])) * float(np.prod(signs)) * float(np.min(np.abs(incoming)))
        new_edge_msgs = edge_msgs.copy()
        for k, (i, j) in enumerate(edges):
            new_edge_msgs[k] = prior[j] + sum(
                detector_msgs[edge_index[(ip, j)]] for ip in col_neighbors[j] if ip != i
            )
        posterior = np.asarray([
            prior[j] + sum(detector_msgs[edge_index[(i, j)]] for i in col_neighbors[j])
            for j in range(N)
        ], dtype=float)
        hard = (posterior <= 0.0).astype(int)
        success = bool(np.array_equal((H @ hard) % 2, s))
        rows.append(np.concatenate([posterior, hard.astype(float), new_edge_msgs, [float(success)]]))
        edge_msgs = new_edge_msgs
        if success:
            break
    return np.vstack(rows)

import numpy as np

def seed_beam_state(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", initial_iters: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    N = H.shape[1]
    E = int(np.count_nonzero(H))
    trace = min_sum_bp_trace(H, p, syndrome, int(initial_iters))
    posterior = trace[:, :N]
    final = trace[-1]
    hard = final[N:2 * N]
    edge_msgs = final[2 * N:2 * N + E]
    cumulative = np.sum(posterior, axis=0)
    next_pos = int(np.argmin(np.abs(cumulative)))
    masks = -np.ones(N, dtype=float)
    return np.concatenate([
        [final[-1], float(trace.shape[0]), 0.0, float(next_pos)],
        masks,
        edge_msgs,
        hard,
    ])

import numpy as np

def masked_bp_trace(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", edge_msgs: "np.ndarray", mask_values: "np.ndarray", max_iters: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int) % 2
    p = np.asarray(p, dtype=float)
    s = np.asarray(syndrome, dtype=int).copy() % 2
    masks = np.asarray(mask_values, dtype=int)
    M, N = H.shape
    edges = [(i, j) for i in range(M) for j in range(N) if H[i, j]]
    edge_index = {edge: k for k, edge in enumerate(edges)}
    row_neighbors = [np.flatnonzero(H[i]).tolist() for i in range(M)]
    col_neighbors = [np.flatnonzero(H[:, j]).tolist() for j in range(N)]
    prior = np.log((1.0 - p) / p)
    active = masks < 0
    for j in np.flatnonzero(masks == 1):
        s ^= H[:, j]
    work = np.asarray(edge_msgs, dtype=float).copy()
    rows = []
    for _ in range(int(max_iters)):
        detector_msgs = np.zeros(len(edges), dtype=float)
        for k, (i, j) in enumerate(edges):
            if not active[j]:
                continue
            incoming = np.asarray([
                work[edge_index[(i, jp)]]
                for jp in row_neighbors[i]
                if jp != j and active[jp]
            ], dtype=float)
            signs = np.where(incoming >= 0.0, 1.0, -1.0)
            detector_msgs[k] = ((-1.0) ** int(s[i])) * float(np.prod(signs)) * float(np.min(np.abs(incoming)))
        new_work = work.copy()
        for k, (i, j) in enumerate(edges):
            if not active[j]:
                continue
            new_work[k] = prior[j] + sum(
                detector_msgs[edge_index[(ip, j)]] for ip in col_neighbors[j] if ip != i
            )
        posterior = np.zeros(N, dtype=float)
        for j in np.flatnonzero(active):
            posterior[j] = prior[j] + sum(
                detector_msgs[edge_index[(i, j)]] for i in col_neighbors[j]
            )
        hard = np.zeros(N, dtype=int)
        hard[active] = (posterior[active] <= 0.0).astype(int)
        success = bool(np.array_equal((H @ hard) % 2, s))
        rows.append(np.concatenate([posterior, hard.astype(float), new_work, [float(success)]]))
        work = new_work
        if success:
            break
    return np.vstack(rows)

import numpy as np

def expand_beam_state(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", state: "np.ndarray", iters_per_round: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    N = H.shape[1]
    E = int(np.count_nonzero(H))
    parent = np.asarray(state, dtype=float)
    branch_pos = int(round(parent[3]))
    parent_masks = parent[4:4 + N].astype(int)
    parent_edge = parent[4 + N:4 + N + E]
    out = []
    for val in (0, 1):
        masks = parent_masks.copy()
        masks[branch_pos] = val
        trace = masked_bp_trace(H, p, syndrome, parent_edge.copy(), masks, int(iters_per_round))
        posterior = trace[:, :N]
        final = trace[-1]
        correction = final[N:2 * N].astype(int)
        success = int(round(final[-1])) == 1
        if success:
            correction[masks >= 0] = masks[masks >= 0]
        edge = final[2 * N:2 * N + E]
        unmasked = np.flatnonzero(masks < 0)
        if unmasked.size:
            cumulative = np.sum(posterior[:, unmasked], axis=0)
            reliability = np.abs(cumulative)
            next_pos = int(unmasked[int(np.argmin(reliability))])
            score = float(np.sum(reliability) / trace.shape[0])
        else:
            next_pos = -1
            score = 0.0
        out.append(np.concatenate([
            [float(success), float(trace.shape[0]), score, float(next_pos)],
            masks.astype(float), edge, correction.astype(float),
        ]))
    return np.vstack(out)

import numpy as np

def prune_beam_states(states: "np.ndarray", beam_width: int) -> "np.ndarray":
    x = np.asarray(states, dtype=float)
    order = np.argsort(-x[:, 2], kind="stable")
    return x[order[:min(int(beam_width), x.shape[0])]].copy()

import numpy as np

def minimum_weight_correction(corrections: "np.ndarray", p: "np.ndarray") -> "np.ndarray":
    c = np.asarray(corrections, dtype=int) % 2
    p = np.asarray(p, dtype=float)
    seen = set()
    unique = []
    for row in c:
        key = tuple(int(v) for v in row)
        if key not in seen:
            seen.add(key)
            unique.append(row.copy())
    u = np.vstack(unique)
    llr = np.log((1.0 - p) / p)
    weights = u @ llr
    k = int(np.argmin(weights))
    return np.concatenate([[float(weights[k])], u[k].astype(float)])

import numpy as np

def beam_search_decode(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_rounds: int, beam_width: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    p = np.asarray(p, dtype=float)
    s = np.asarray(syndrome, dtype=int)
    N = H.shape[1]
    seed = seed_beam_state(H, p, s, int(initial_iters))
    results = []
    seen = set()
    children_expanded = 0

    def _add_unique(bits):
        key = tuple(int(v) for v in np.asarray(bits, dtype=int))
        if key not in seen:
            seen.add(key)
            results.append(np.asarray(bits, dtype=int).copy())

    if int(round(seed[0])) == 1:
        _add_unique(seed[-N:])
        if int(num_results) == 1:
            selected = minimum_weight_correction(np.vstack(results), p)
            return np.concatenate([selected[:1], [0.0, 1.0, 0.0], selected[1:]])

    active = seed.reshape(1, -1)
    for round_index in range(1, int(max_rounds) + 1):
        active = prune_beam_states(active, active.shape[0])
        next_rows = []
        for path in active:
            children = expand_beam_state(H, p, s, path, int(iters_per_round))
            for child in children:
                children_expanded += 1
                if int(round(child[0])) == 1:
                    _add_unique(child[-N:])
                    if len(results) >= int(num_results):
                        selected = minimum_weight_correction(np.vstack(results), p)
                        return np.concatenate([
                            selected[:1],
                            [float(round_index), float(len(results)), float(children_expanded)],
                            selected[1:],
                        ])
                next_rows.append(child)
        if next_rows:
            active = prune_beam_states(np.vstack(next_rows), int(beam_width))

    if not results:
        raise ValueError("no valid correction found within search budget")
    selected = minimum_weight_correction(np.vstack(results), p)
    return np.concatenate([
        selected[:1],
        [float(max_rounds), float(len(results)), float(children_expanded)],
        selected[1:],
    ])

import numpy as np

def beam_search_profile(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    N = H.shape[1]
    rows = []
    for width in np.asarray(beam_widths, dtype=int):
        result = beam_search_decode(
            H, p, syndrome, int(max_rounds), int(width), int(initial_iters),
            int(iters_per_round), int(num_results)
        )
        correction = result[4:4 + N]
        rows.append([
            result[0], float(np.sum(correction)), result[1], result[2], result[3]
        ])
    return np.asarray(rows, dtype=float)

import numpy as np

def cumulative_beam_profile_path(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> float:
    H = np.asarray(H, dtype=int)
    p = np.asarray(p, dtype=float)
    widths = np.asarray(beam_widths, dtype=int)
    profile = beam_search_profile(
        H, p, syndrome, widths, int(max_rounds), int(initial_iters),
        int(iters_per_round), int(num_results)
    )
    prior_sum = float(np.sum(np.log((1.0 - p) / p)))
    N = H.shape[1]
    normalized = profile.copy()
    normalized[:, 0] /= prior_sum
    normalized[:, 1] /= float(N)
    normalized[:, 2] /= float(max_rounds)
    normalized[:, 3] /= float(num_results)
    normalized[:, 4] /= (2.0 * widths.astype(float) * float(max_rounds))
    return float(np.sum(np.linalg.norm(np.diff(normalized, axis=0), axis=1)))
SCICODE_GOLD_EOF
