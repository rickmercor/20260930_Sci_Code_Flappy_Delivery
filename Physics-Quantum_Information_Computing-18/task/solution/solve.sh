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


def _binary(value, ndim, name):
    array = np.asarray(value)
    if array.ndim != ndim or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be binary with {ndim} dimensions")
    return array.astype(np.int64, copy=True)


def _integer(value, low, high, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} is outside its integer range")
    return int(value)


def _graph(detectors):
    matrix = _binary(detectors, 2, "detectors")
    if not 1 <= matrix.shape[0] <= 6 or not 1 <= matrix.shape[1] <= 8:
        raise ValueError("invalid detector dimensions")
    support = matrix.sum(axis=0)
    if not np.all((support == 1) | (support == 2)):
        raise ValueError("each column must trigger one or two detectors")
    if np.any(support == 1):
        matrix = np.vstack((matrix, support == 1)).astype(np.int64)
    n, edges = matrix.shape
    if not 2 <= n <= 6 or not 0 <= edges - n + 1 <= 3:
        raise ValueError("invalid augmented graph size or cycle rank")
    endpoints = [tuple(np.flatnonzero(matrix[:, e])) for e in range(edges)]
    if len(set(endpoints)) != edges:
        raise ValueError("parallel edges are outside the graph contract")
    reached = {0}
    for _ in range(n):
        for u, v in endpoints:
            if u in reached or v in reached:
                reached.update((u, v))
    if len(reached) != n:
        raise ValueError("the augmented graph must be connected")
    return matrix


def _reference(reference, n_edges):
    reference = _binary(reference, 1, "reference")
    if reference.shape != (n_edges,):
        raise ValueError("reference shape must match the edge count")
    return reference


def _mask_data(incidence):
    masks = np.arange(1 << incidence.shape[1], dtype=np.int64)
    bits = (masks[:, None] >> np.arange(incidence.shape[1])) & 1
    return masks, bits, (bits @ incidence.T) % 2


def construct_physical_cycle_ensemble(
    detectors: "np.ndarray", reference: "np.ndarray"
) -> "np.ndarray":
    incidence = _graph(detectors)
    reference = _reference(reference, incidence.shape[1])
    _, bits, boundaries = _mask_data(incidence)
    return bits[np.all(boundaries == 0, axis=1)] ^ reference

import numpy as np


def _edge_weights(probabilities, reference):
    q = np.asarray(probabilities, dtype=float)
    if (
        q.shape != reference.shape
        or not np.all(np.isfinite(q))
        or np.any((q <= 0) | (q >= 1))
    ):
        raise ValueError("probabilities must be finite edge values in (0,1)")
    odds = q / (1.0 - q)
    weights = np.where(reference == 1, 1.0 / odds, odds)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("conditional weights must be finite and positive")
    return weights


def _oriented_states(incidence, tail):
    masks, _, boundaries = _mask_data(incidence)
    states = []
    heads = [tail] + [h for h in range(incidence.shape[0]) if h != tail]
    for head in heads:
        target = np.zeros(incidence.shape[0], dtype=int)
        target[head] ^= 1
        target[tail] ^= 1
        states.extend(
            (int(a), head) for a in masks[np.all(boundaries == target, axis=1)]
        )
    return states


def build_oriented_transition(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    tail: int,
) -> "np.ndarray":
    incidence = _graph(detectors)
    n, edges = incidence.shape
    reference = _reference(reference, edges)
    weights = _edge_weights(probabilities, reference)
    tail = _integer(tail, 0, n - 1, "tail")
    states = _oriented_states(incidence, tail)
    indices = {state: i for i, state in enumerate(states)}
    degrees = incidence.sum(axis=1)
    endpoints = [np.flatnonzero(incidence[:, e]) for e in range(edges)]
    transition = np.zeros((len(states), len(states)))
    for row, (mask, head) in enumerate(states):
        for edge in np.flatnonzero(incidence[head]):
            u, v = endpoints[edge]
            neighbor = int(v if head == u else u)
            ratio = (
                weights[edge] if not (mask >> int(edge)) & 1 else 1.0 / weights[edge]
            )
            acceptance = min(1.0, degrees[head] / degrees[neighbor] * ratio)
            transition[row, indices[(mask ^ (1 << int(edge)), neighbor)]] += (
                acceptance / degrees[head]
            )
            transition[row, row] += (1.0 - acceptance) / degrees[head]
    return transition

import numpy as np


def _jet_product(left, right):
    left, right = np.broadcast_arrays(left, right)
    out = np.zeros_like(left, dtype=float)
    for degree in range(4):
        for split in range(degree + 1):
            out[..., degree] += left[..., split] * right[..., degree - split]
    return out


def _matrix_jet_product(left, right):
    out = np.zeros((left.shape[0], right.shape[1], 4))
    for degree in range(4):
        for split in range(degree + 1):
            out[..., degree] += left[..., split] @ right[..., degree - split]
    return out


def compute_work_recording_kernel(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    spacing: int,
) -> "np.ndarray":
    incidence = _graph(detectors)
    k = len(construct_physical_cycle_ensemble(detectors, reference))
    spacing = _integer(spacing, 1, 3, "spacing")
    jets = np.zeros((k, k, 4))
    for tail in range(len(incidence)):
        transition = build_oriented_transition(
            detectors, probabilities, reference, tail
        )
        direct = transition[:k, :k]
        departure = transition[:k, k:]
        arrival = transition[k:, :k]
        transient = transition[k:, k:]
        system = np.eye(len(transient)) - transient
        try:
            resolvents = [np.linalg.solve(system, arrival)]
            for _ in range(3):
                resolvents.append(np.linalg.solve(system, transient @ resolvents[-1]))
        except np.linalg.LinAlgError as error:
            raise ValueError("excursions must have finite moments") from error
        for degree in range(4):
            term = resolvents[degree].copy()
            if degree >= 1:
                term += 2.0 * resolvents[degree - 1]
            if degree >= 2:
                term += resolvents[degree - 2]
            jets[..., degree] += departure @ term
            if degree <= 1:
                jets[..., degree] += direct
    jets /= len(incidence)
    if not np.all(np.isfinite(jets)) or np.min(jets) < -1e-8:
        raise ValueError("first-return moments are not numerically finite")
    jets = np.maximum(jets, 0.0)
    result = np.zeros_like(jets)
    result[..., 0] = np.eye(k)
    for _ in range(spacing):
        result = _matrix_jet_product(result, jets)
    return result

import numpy as np


def _work_kernel(kernel, n_states):
    kernel = np.asarray(kernel, dtype=float)
    if (
        kernel.shape != (n_states, n_states, 4)
        or not np.all(np.isfinite(kernel))
        or np.any(kernel < 0)
    ):
        raise ValueError("work kernel must be a finite nonnegative (k,k,4) array")
    if not np.allclose(kernel[..., 0].sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("zeroth kernel coefficient must be row stochastic")
    return kernel


def _observations(physical):
    physical = _binary(physical, 2, "physical_errors")
    if not 1 <= len(physical) <= 8 or not 1 <= physical.shape[1] <= 8:
        raise ValueError("physical observation dimensions exceed the contract")
    if len(np.unique(physical, axis=0)) != len(physical):
        raise ValueError("physical rows must be distinct")
    return physical


def _history_work(kernel, observations, samples):
    k = len(kernel)
    initial = np.zeros_like(kernel)
    initial[..., 0] = np.eye(k)
    history = {(0,) * observations.shape[1]: initial}
    for _ in range(samples):
        updated = {}
        for counts, mass in history.items():
            arrival = _matrix_jet_product(mass, kernel)
            for terminal in range(k):
                if not np.any(arrival[:, terminal, 0] > 0):
                    continue
                key = tuple(np.asarray(counts) + observations[terminal])
                if key not in updated:
                    updated[key] = np.zeros_like(kernel)
                updated[key][:, terminal] += arrival[:, terminal]
        history = updated
    return history


def compute_count_work_transfer(
    recording_kernel: "np.ndarray", physical_errors: "np.ndarray", n_samples: int
) -> "np.ndarray":
    physical = _observations(physical_errors)
    kernel = _work_kernel(recording_kernel, len(physical))
    samples = _integer(n_samples, 1, 3, "n_samples")
    history = _history_work(kernel, physical, samples)
    rows = []
    for counts, mass in sorted(history.items()):
        for terminal in range(len(physical)):
            if np.any(mass[:, terminal, 0] > 0):
                rows.append([*counts, terminal, *mass[:, terminal].ravel()])
    return np.asarray(rows, dtype=float)

import numpy as np


def _channel_probability(value):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("channel probability must be finite and in (0,1)")
    return float(value)


def _paired_map(value, edges):
    value = np.asarray(value)
    if (
        value.shape != (edges,)
        or not np.issubdtype(value.dtype, np.integer)
        or not np.array_equal(np.sort(value), np.arange(edges))
    ):
        raise ValueError("edge_map must be an edge permutation")
    return value.astype(int, copy=True)


def _feedback_prior(p, counts, samples, mapping):
    alpha = np.asarray(counts, dtype=float) / samples
    q = np.empty_like(alpha)
    q[mapping] = alpha / 2.0 + (1.0 - alpha) * p / (3.0 - 2.0 * p)
    return q


def _incoming_groups(incoming, edges, active_states, passive_states, samples):
    incoming = np.asarray(incoming, dtype=float)
    if (
        incoming.ndim != 2
        or incoming.shape[1] != edges + 6
        or len(incoming) == 0
        or not np.all(np.isfinite(incoming))
        or np.any(incoming < 0)
    ):
        raise ValueError("incoming law has invalid dimensions or values")
    keys = incoming[:, : edges + 2]
    if (
        np.any(keys != np.floor(keys))
        or np.any(keys[:, 0] >= passive_states)
        or np.any(keys[:, 1] >= active_states)
        or np.any(keys[:, 2:] > samples)
    ):
        raise ValueError("state indices and empirical counts must be in range")
    if not np.isclose(incoming[:, edges + 2].sum(), 1.0, atol=1e-9, rtol=0):
        raise ValueError("incoming zeroth coefficients must sum to one")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("incoming keys must be unique")
    groups = {}
    for row in incoming:
        passive, active = row[:2].astype(int)
        key = tuple(row[2 : edges + 2].astype(int))
        if key not in groups:
            groups[key] = np.zeros((passive_states, active_states, 4))
        groups[key][passive, active] = row[-4:]
    return groups


def transfer_retained_phase(
    incoming: "np.ndarray",
    source_samples: int,
    channel_probability: float,
    edge_map: "np.ndarray",
    target_detectors: "np.ndarray",
    target_reference: "np.ndarray",
    passive_states: int,
    target_samples: int,
    target_spacing: int,
) -> "np.ndarray":
    physical = construct_physical_cycle_ensemble(
        target_detectors, target_reference
    )
    k, edges = physical.shape
    passive_states = _integer(passive_states, 1, 8, "passive_states")
    source_samples = _integer(source_samples, 1, 3, "source_samples")
    target_samples = _integer(target_samples, 1, 3, "target_samples")
    target_spacing = _integer(target_spacing, 1, 3, "target_spacing")
    p = _channel_probability(channel_probability)
    mapping = _paired_map(edge_map, edges)
    groups = _incoming_groups(incoming, edges, k, passive_states, source_samples)
    accumulated = {}
    for source_counts, incoming_mass in groups.items():
        q = _feedback_prior(p, source_counts, source_samples, mapping)
        kernel = compute_work_recording_kernel(
            target_detectors, q, target_reference, target_spacing
        )
        table = compute_count_work_transfer(kernel, physical, target_samples)
        for row in table:
            counts = tuple(row[:edges].astype(int))
            terminal = int(row[edges])
            transfer = row[edges + 1 :].reshape(k, 1, 4)
            mass = _matrix_jet_product(incoming_mass, transfer)[:, 0]
            for passive in range(passive_states):
                if mass[passive, 0] <= 0:
                    continue
                key = (terminal, passive, *counts)
                if key not in accumulated:
                    accumulated[key] = np.zeros(4)
                accumulated[key] += mass[passive]
    return np.asarray([(*key, *mass) for key, mass in sorted(accumulated.items())])

import numpy as np


def _logical_classes(physical, logical):
    logical = _binary(logical, 2, "logical")
    if logical.shape[1] != physical.shape[1] or not 1 <= len(logical) <= 2:
        raise ValueError("logical must have one or two rows and E columns")
    labels = ((physical @ logical.T) % 2) @ (1 << np.arange(len(logical)))
    return labels, 1 << len(logical)


def compute_report_work_transfer(
    recording_kernel: "np.ndarray",
    physical_errors: "np.ndarray",
    logical: "np.ndarray",
    n_samples: int,
) -> "np.ndarray":
    physical = _observations(physical_errors)
    kernel = _work_kernel(recording_kernel, len(physical))
    samples = _integer(n_samples, 1, 5, "n_samples")
    labels, n_classes = _logical_classes(physical, logical)
    observations = np.eye(n_classes, dtype=int)[labels]
    history = _history_work(kernel, observations, samples)
    result = np.zeros((len(physical), n_classes, 4))
    for counts, mass in history.items():
        result[:, int(np.argmax(counts))] += mass.sum(axis=1)
    return result

import numpy as np


def _phase_schedule(value, last_high, name):
    value = np.asarray(value)
    high = np.array([3, 3, 3, last_high])
    if (
        value.shape != (4,)
        or not np.issubdtype(value.dtype, np.integer)
        or np.any(value < 1)
        or np.any(value > high)
    ):
        raise ValueError(f"{name} must contain four supported positive integers")
    return value.astype(int, copy=True)


def _model_inputs(hz, hx, mz, mx, sz, sx, logical, p, mapping, samples, spacings):
    z = construct_physical_cycle_ensemble(hz, mz)
    x = construct_physical_cycle_ensemble(hx, mx)
    if z.shape[1] != x.shape[1]:
        raise ValueError("paired graphs must have the same edge count")
    for h, m, s in [(hz, mz, sz), (hx, mx, sx)]:
        s = _binary(s, 1, "syndrome")
        h = np.asarray(h)
        if s.shape != (len(h),) or not np.array_equal(h @ m % 2, s):
            raise ValueError("reference must realize the syndrome")
    _logical_classes(x, logical)
    return (
        z,
        x,
        _channel_probability(p),
        _paired_map(mapping, z.shape[1]),
        _phase_schedule(samples, 5, "sample_counts"),
        _phase_schedule(spacings, 3, "spacings"),
    )


def compute_adaptive_work_moments(
    detectors_z: "np.ndarray",
    detectors_x: "np.ndarray",
    reference_z: "np.ndarray",
    reference_x: "np.ndarray",
    syndrome_z: "np.ndarray",
    syndrome_x: "np.ndarray",
    logical_x: "np.ndarray",
    channel_probability: float,
    edge_map: "np.ndarray",
    sample_counts: "np.ndarray",
    spacings: "np.ndarray",
) -> "np.ndarray":
    z, x, p, mapping, samples, spacings = _model_inputs(
        detectors_z,
        detectors_x,
        reference_z,
        reference_x,
        syndrome_z,
        syndrome_x,
        logical_x,
        channel_probability,
        edge_map,
        sample_counts,
        spacings,
    )
    edges = z.shape[1]
    kernel = compute_work_recording_kernel(
        detectors_z, np.full(edges, 2 * p / 3), reference_z, int(spacings[0])
    )
    table = compute_count_work_transfer(kernel, z, int(samples[0]))
    # The inactive X chain starts at relative state zero.
    incoming = np.array(
        [
            (int(row[edges]), 0, *row[:edges], *row[edges + 1 : edges + 5])
            for row in table
            if row[edges + 1] > 0
        ]
    )
    incoming = transfer_retained_phase(
        incoming,
        int(samples[0]),
        p,
        mapping,
        detectors_x,
        reference_x,
        len(z),
        int(samples[1]),
        int(spacings[1]),
    )
    incoming = transfer_retained_phase(
        incoming,
        int(samples[1]),
        p,
        np.argsort(mapping),
        detectors_z,
        reference_z,
        len(x),
        int(samples[2]),
        int(spacings[2]),
    )
    _, classes = _logical_classes(x, logical_x)
    result = np.zeros((classes, 4))
    # Marginalize the last Z terminal state only after its last use.
    groups = _incoming_groups(incoming, edges, len(x), len(z), int(samples[2]))
    for counts, mass in groups.items():
        q = _feedback_prior(p, counts, int(samples[2]), mapping)
        kernel = compute_work_recording_kernel(
            detectors_x, q, reference_x, int(spacings[3])
        )
        report = compute_report_work_transfer(
            kernel, x, logical_x, int(samples[3])
        )
        result += _matrix_jet_product(mass.sum(axis=0)[None, ...], report)[0]
    return result

import numpy as np


def _conditional_skewness(coefficients, target):
    probability, b1, b2, b3 = coefficients[target]
    if probability <= 0:
        raise ValueError("conditioning event must have positive probability")
    mean = b1 / probability
    second = (2 * b2 + b1) / probability
    third = (6 * b3 + 6 * b2 + b1) / probability
    variance = second - mean * mean
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("conditional work must have finite positive variance")
    value = (third - 3 * mean * second + 2 * mean**3) / variance**1.5
    if not np.isfinite(value):
        raise ValueError("conditional skewness must be finite")
    return float(value)


def compute_conditional_work_skewness(
    detectors_z: "np.ndarray",
    detectors_x: "np.ndarray",
    reference_z: "np.ndarray",
    reference_x: "np.ndarray",
    syndrome_z: "np.ndarray",
    syndrome_x: "np.ndarray",
    logical_x: "np.ndarray",
    channel_probability: float,
    edge_map: "np.ndarray",
    sample_counts: "np.ndarray",
    spacings: "np.ndarray",
    target_label: int,
) -> float:
    logical = _binary(logical_x, 2, "logical_x")
    if not 1 <= len(logical) <= 2:
        raise ValueError("logical_x must have one or two rows")
    target = _integer(target_label, 0, (1 << len(logical)) - 1, "target_label")
    coefficients = compute_adaptive_work_moments(
        detectors_z,
        detectors_x,
        reference_z,
        reference_x,
        syndrome_z,
        syndrome_x,
        logical,
        channel_probability,
        edge_map,
        sample_counts,
        spacings,
    )
    return _conditional_skewness(coefficients, target)
SCICODE_GOLD_EOF
