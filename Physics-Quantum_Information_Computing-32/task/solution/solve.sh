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


def encode_pauli_words(labels: tuple[str, ...]) -> "np.ndarray":
    if not isinstance(labels, tuple) or not labels:
        raise ValueError("labels must be a nonempty tuple")
    if any(not isinstance(word, str) for word in labels):
        raise ValueError("each label must be a string")
    n = len(labels[0])
    if not 1 <= n <= 6 or len(set(labels)) != len(labels):
        raise ValueError("use distinct words on one to six qubits")
    if any(
        len(word) != n or set(word) - set("IXYZ") or word == "I" * n for word in labels
    ):
        raise ValueError("invalid Pauli word")
    encoded = np.zeros((len(labels), 2 * n + 1), dtype=np.int64)
    for row, word in enumerate(labels):
        encoded[row, 1 : 1 + n] = [letter in "XY" for letter in word]
        encoded[row, 1 + n :] = [letter in "YZ" for letter in word]
        encoded[row, 0] = word.count("Y") % 4
    return encoded

import numpy as np


def _checked_encoding(encoded):
    values = np.asarray(encoded)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] not in range(3, 14, 2)
    ):
        raise ValueError("encoded must have shape (m, 2*n+1)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("encoding must be integral")
    n = (values.shape[1] - 1) // 2
    a, b = values[:, 1 : n + 1], values[:, n + 1 :]
    if not np.all((values[:, 1:] == 0) | (values[:, 1:] == 1)):
        raise ValueError("Pauli bits must be binary")
    if not np.array_equal(values[:, 0], np.sum(a * b, axis=1) % 4):
        raise ValueError("phase does not describe the unsigned Hermitian word")
    return values.astype(np.int64, copy=True), n


def build_frustration_matrix(encoded: "np.ndarray") -> "np.ndarray":
    values, n = _checked_encoding(encoded)
    a, b = values[:, 1 : n + 1], values[:, n + 1 :]
    return (a @ b.T + b @ a.T) % 2

import numpy as np


def enumerate_maximal_contexts(adjacency: "np.ndarray") -> "np.ndarray":
    graph = np.asarray(adjacency)
    if (
        graph.ndim != 2
        or not 1 <= graph.shape[0] <= 32
        or graph.shape[0] != graph.shape[1]
    ):
        raise ValueError("adjacency must be square with one to 32 vertices")
    if (
        not np.all((graph == 0) | (graph == 1))
        or not np.array_equal(graph, graph.T)
        or np.any(np.diag(graph))
    ):
        raise ValueError("adjacency must describe a simple undirected graph")
    m = len(graph)
    neighbors = [set(np.flatnonzero(graph[j] == 0)) - {j} for j in range(m)]
    masks = []

    def _visit(chosen, possible, excluded):
        if not possible and not excluded:
            masks.append(sum(1 << j for j in chosen))
            return
        pivot = min(
            possible | excluded, key=lambda j: (-len(possible & neighbors[j]), j)
        )
        for j in sorted(possible - neighbors[pivot]):
            _visit(chosen + [j], possible & neighbors[j], excluded & neighbors[j])
            possible.remove(j)
            excluded.add(j)

    _visit([], set(range(m)), set())
    return np.array(
        [[(mask >> j) & 1 for j in range(m)] for mask in sorted(masks)], dtype=np.int64
    )

import numpy as np


def _binary_rref(matrix):
    reduced = np.array(matrix, dtype=np.int64, copy=True)
    pivots = []
    row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if not candidates.size:
            continue
        pivot = row + int(candidates[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for other in range(reduced.shape[0]):
            if other != row and reduced[other, column]:
                reduced[other] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break
    return reduced, pivots


def _binary_kernel(matrix):
    reduced, pivots = _binary_rref(matrix)
    free = [j for j in range(matrix.shape[1]) if j not in pivots]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.int64)
    for row, column in enumerate(free):
        basis[row, column] = 1
        for pivot_row, pivot_column in enumerate(pivots):
            basis[row, pivot_column] = reduced[pivot_row, column]
    return basis


def _checked_contexts(contexts, m):
    values = np.asarray(contexts)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != m:
        raise ValueError("contexts must have shape (C,m)")
    if not np.all((values == 0) | (values == 1)) or np.any(values.sum(axis=1) == 0):
        raise ValueError("each context must have nonempty binary membership")
    return values.astype(np.int64, copy=True)


def derive_phase_constraints(
    encoded: "np.ndarray", contexts: "np.ndarray"
) -> "np.ndarray":
    values, n = _checked_encoding(encoded)
    m = len(values)
    contexts = _checked_contexts(contexts, m)
    adjacency = build_frustration_matrix(values)
    relations = []
    for context_index, context in enumerate(contexts):
        indices = np.flatnonzero(context)
        if np.any(adjacency[np.ix_(indices, indices)]):
            raise ValueError("context contains anticommuting observables")
        kernel = _binary_kernel(values[indices, 1:].T)
        for relation in kernel:
            phase = 0
            accumulated_b = np.zeros(n, dtype=np.int64)
            for local_index in np.flatnonzero(relation):
                word = values[indices[local_index]]
                phase = (
                    phase + int(word[0]) + 2 * int(accumulated_b @ word[1 : n + 1])
                ) % 4
                accumulated_b ^= word[n + 1 :]
            padded = np.zeros(m + 2, dtype=np.int64)
            padded[:2] = context_index, phase // 2
            padded[2 + indices] = relation
            relations.append(padded)
    return np.array(relations, dtype=np.int64).reshape(-1, m + 2)

import numpy as np


def _affine_binary_solutions(matrix, target):
    columns = matrix.shape[1]
    augmented, pivots = _binary_rref(np.column_stack((matrix, target)))
    if columns in pivots:
        raise ValueError("inconsistent binary constraints")
    particular = np.zeros(columns, dtype=np.int64)
    for row, column in enumerate(pivots):
        particular[column] = augmented[row, -1]
    kernel = _binary_kernel(matrix)
    solutions = np.empty((1 << len(kernel), columns), dtype=np.int64)
    for mask in range(len(solutions)):
        value = particular.copy()
        for j, direction in enumerate(kernel):
            if (mask >> j) & 1:
                value ^= direction
        solutions[mask] = value
    return solutions


def construct_projected_vertices(
    contexts: "np.ndarray", relations: "np.ndarray"
) -> "np.ndarray":
    raw = np.asarray(contexts)
    if raw.ndim != 2:
        raise ValueError("contexts must be a matrix")
    contexts = _checked_contexts(raw, raw.shape[1])
    m = contexts.shape[1]
    relations = np.asarray(relations)
    if (
        relations.ndim != 2
        or relations.shape[1] != m + 2
        or not np.issubdtype(relations.dtype, np.integer)
    ):
        raise ValueError("relations must be an integer (L,m+2) array")
    if not np.all((relations[:, 1:] == 0) | (relations[:, 1:] == 1)):
        raise ValueError("phase and relation entries must be binary")
    if np.any((relations[:, 0] < 0) | (relations[:, 0] >= len(contexts))):
        raise ValueError("context index out of range")
    columns = []
    for index, context in enumerate(contexts):
        active = np.flatnonzero(context)
        selected = relations[relations[:, 0] == index]
        if np.any(selected[:, 2:][:, context == 0]):
            raise ValueError("relation has support outside its context")
        signs = _affine_binary_solutions(selected[:, 2:][:, active], selected[:, 1])
        block = np.zeros((len(signs), m), dtype=np.int64)
        block[:, active] = 1 - 2 * signs
        columns.extend(block)
    return np.unique(np.array(columns), axis=0).T.copy()

import numpy as np


def _pauli_matrices(labels):
    single = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
    }
    matrices = []
    for word in labels:
        value = np.ones((1, 1), dtype=complex)
        for letter in word:
            value = np.kron(value, single[letter])
        matrices.append(value)
    return np.array(matrices)


def compute_thermal_marginals(
    labels: tuple[str, ...], coefficients: "np.ndarray", inverse_temperature: float
) -> "np.ndarray":
    encode_pauli_words(labels)
    raw = np.asarray(coefficients)
    if raw.shape != (len(labels),) or np.iscomplexobj(raw):
        raise ValueError("coefficients must be a real vector in label order")
    coefficients = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(coefficients))
        or not np.isfinite(inverse_temperature)
        or inverse_temperature < 0
    ):
        raise ValueError(
            "coefficients and nonnegative temperature parameter must be finite"
        )
    operators = _pauli_matrices(labels)
    hamiltonian = np.einsum("j,jab->ab", coefficients, operators)
    energies, vectors = np.linalg.eigh(hamiltonian)
    weights = np.exp(-float(inverse_temperature) * (energies - energies[0]))
    weights /= weights.sum()
    diagonals = np.einsum("ak,jab,bk->jk", vectors.conj(), operators, vectors)
    return np.real(diagonals @ weights)

import numpy as np
from scipy.optimize import linprog


def _affine_robustness_certificate(vertices, expectations):
    raw_v, raw_y = np.asarray(vertices), np.asarray(expectations)
    if raw_v.ndim != 2 or min(raw_v.shape) < 1 or raw_y.shape != (raw_v.shape[0],):
        raise ValueError("vertices and expectations have incompatible shapes")
    if np.iscomplexobj(raw_v) or np.iscomplexobj(raw_y):
        raise ValueError("linear program data must be real")
    vertices, expectations = raw_v.astype(float), raw_y.astype(float)
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(expectations)):
        raise ValueError("linear program data must be finite")
    count = vertices.shape[1]
    affine = np.vstack((vertices, np.ones(count)))
    target = np.append(expectations, 1.0)
    split = np.column_stack((affine, -affine))
    # Rescale coefficients, not the feasible correlations. This resolves small
    # positive weights near a face transition above the solver's absolute tolerance.
    scale = 65536.0 / max(1.0, float(np.max(np.abs(target))))
    options = {
        "primal_feasibility_tolerance": 1e-10,
        "dual_feasibility_tolerance": 1e-10,
    }
    primal = linprog(
        np.ones(2 * count),
        A_eq=split,
        b_eq=scale * target,
        bounds=(0, None),
        method="highs",
        options=options,
    )
    if primal.status == 2:
        raise ValueError("expectations lie outside the affine span")
    if not primal.success:
        raise RuntimeError("primal optimization failed")
    dual = linprog(
        -scale * target,
        A_ub=np.vstack((affine.T, -affine.T)),
        b_ub=np.ones(2 * count),
        bounds=[(None, None)] * len(target),
        method="highs",
        options=options,
    )
    if not dual.success:
        raise RuntimeError("dual optimization failed")
    coefficients = primal.x / scale
    optimum = float(primal.fun / scale)
    dual_value = float(-dual.fun / scale)
    residual = float(np.max(np.abs(split @ coefficients - target)))
    violation = float(max(0.0, np.max(np.abs(affine.T @ dual.x)) - 1.0))
    gap = abs(optimum - dual_value)
    tolerance = 1e-8 * max(1.0, abs(optimum))
    if (
        max(residual, violation, gap, -float(np.min(coefficients)), 1.0 - optimum)
        > tolerance
    ):
        raise RuntimeError("affine robustness certificate failed")
    return (
        optimum,
        dual_value,
        residual,
        violation,
        coefficients[:count] - coefficients[count:],
        dual.x,
    )


def _checked_profile_data(vertices, base, direction):
    raw_v, raw_b, raw_d = map(np.asarray, (vertices, base, direction))
    if raw_v.ndim != 2 or min(raw_v.shape) < 1:
        raise ValueError("vertices must be a nonempty matrix")
    if raw_b.shape != (raw_v.shape[0],) or raw_d.shape != raw_b.shape:
        raise ValueError("correlation vectors have incompatible dimensions")
    if any(
        np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in (raw_v, raw_b, raw_d)
    ):
        raise ValueError("profile data must be finite and real")
    vertices, base, direction = (x.astype(float) for x in (raw_v, raw_b, raw_d))
    affine = np.vstack((vertices, np.ones(vertices.shape[1])))
    if np.linalg.matrix_rank(affine) != len(base) + 1:
        raise ValueError("vertices must span the complete affine measurement space")
    return vertices, base, direction, affine


def certify_directional_support(
    vertices: "np.ndarray",
    base: "np.ndarray",
    direction: "np.ndarray",
    parameter: float,
) -> "np.ndarray":
    vertices, base, direction, affine = _checked_profile_data(vertices, base, direction)
    if not np.isfinite(parameter):
        raise ValueError("parameter must be finite")
    target = np.append(base + parameter * direction, 1.0)
    tangent = np.append(direction, 0.0)
    _, optimum, _, _, coefficients, _ = _affine_robustness_certificate(
        vertices, target[:-1]
    )
    # Complementary slackness gives the same entire optimal face. Its equalities
    # use vertex normals and remain well-conditioned as a positive weight tends
    # to zero, unlike imposing the almost-parallel objective equality directly.
    cutoff = 16.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(coefficients))))
    support = np.abs(coefficients) > cutoff
    face_matrix = affine[:, support].T
    face_values = np.sign(coefficients[support])
    inequalities = np.vstack((affine.T, -affine.T))
    limits = np.ones(len(inequalities))
    options = {
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }
    extrema = []
    for sense in (1.0, -1.0):
        result = linprog(
            sense * tangent,
            A_ub=inequalities,
            b_ub=limits,
            A_eq=face_matrix,
            b_eq=face_values,
            bounds=[(None, None)] * len(target),
            method="highs",
            options=options,
        )
        if not result.success:
            raise RuntimeError("optimal-face slope optimization failed")
        error = max(
            abs(target @ result.x - optimum),
            float(np.max(np.abs(face_matrix @ result.x - face_values))),
            float(np.max(inequalities @ result.x - limits)),
        )
        if error > 1e-8 * max(1.0, abs(optimum)):
            raise RuntimeError("optimal-face feasibility certificate failed")
        extrema.append(float(tangent @ result.x))
    if extrema[0] > extrema[1] + 1e-8 * max(1.0, *map(abs, extrema)):
        raise RuntimeError("directional derivative ordering failed")
    return np.array([optimum, extrema[0], extrema[1]])

import numpy as np
from scipy.optimize import linprog


def trace_robustness_profile(
    vertices: "np.ndarray", start: "np.ndarray", end: "np.ndarray"
) -> "np.ndarray":
    vertices, start, end, _ = _checked_profile_data(vertices, start, end)
    direction = end - start
    records = []

    def _probe(t):
        return certify_directional_support(vertices, start, direction, t)

    def _refine(left, right, left_support, right_support, depth):
        if depth > 80:
            raise RuntimeError("profile refinement did not converge")
        left_slope, right_slope = left_support[2], right_support[1]
        left_offset = left_support[0] - left_slope * left
        right_offset = right_support[0] - right_slope * right
        scale = max(
            1.0, abs(left_offset), abs(right_offset), abs(left_slope), abs(right_slope)
        )
        if (
            max(abs(left_slope - right_slope), abs(left_offset - right_offset))
            <= 1e-8 * scale
        ):
            records.append([left, right, left_offset, left_slope])
            return
        if right_slope <= left_slope:
            raise RuntimeError("support slopes violate convexity")
        crossing = (left_offset - right_offset) / (right_slope - left_slope)
        if not left + 1e-12 < crossing < right - 1e-12:
            raise RuntimeError(
                "support intersection is outside the unresolved interval"
            )
        middle_support = _probe(crossing)
        envelope = left_offset + left_slope * crossing
        if middle_support[0] - envelope <= 1e-8 * scale:
            records.extend(
                [
                    [left, crossing, left_offset, left_slope],
                    [crossing, right, right_offset, right_slope],
                ]
            )
            return
        _refine(left, crossing, left_support, middle_support, depth + 1)
        _refine(crossing, right, middle_support, right_support, depth + 1)

    _refine(0.0, 1.0, _probe(0.0), _probe(1.0), 0)
    merged = []
    for record in records:
        if merged and np.allclose(merged[-1][2:], record[2:], rtol=1e-8, atol=1e-8):
            merged[-1][1] = record[1]
        else:
            merged.append(record)
    return np.array(merged)

import numpy as np


def _checked_segments(profile):
    raw = np.asarray(profile)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != 4 or np.iscomplexobj(raw):
        raise ValueError("profile must be a nonempty real (K,4) array")
    segments = raw.astype(float)
    if not np.all(np.isfinite(segments)) or np.any(segments[:, 1] <= segments[:, 0]):
        raise ValueError(
            "profile must have finite coefficients and positive interval widths"
        )
    tolerance = 1e-7
    if abs(segments[0, 0]) > tolerance or abs(segments[-1, 1] - 1.0) > tolerance:
        raise ValueError("profile must span [0,1]")
    if np.any(np.abs(segments[1:, 0] - segments[:-1, 1]) > tolerance):
        raise ValueError("profile intervals must meet")
    left_values = segments[:, 2] + segments[:, 3] * segments[:, 0]
    right_values = segments[:, 2] + segments[:, 3] * segments[:, 1]
    if np.any(np.abs(left_values[1:] - right_values[:-1]) > tolerance):
        raise ValueError("profile must be continuous")
    if (
        np.any(np.diff(segments[:, 3]) < -tolerance)
        or min(left_values.min(), right_values.min()) < 1.0 - tolerance
    ):
        raise ValueError("profile must be convex and at least one")
    return segments


def integrate_measurement_gain(
    full_profile: "np.ndarray", reduced_profile: "np.ndarray"
) -> float:
    full = _checked_segments(full_profile)
    reduced = _checked_segments(reduced_profile)
    i = j = 0
    left = total = 0.0
    while i < len(full) and j < len(reduced):
        right = min(full[i, 1], reduced[j, 1])
        width = right - left
        offset, slope = full[i, 2:] - reduced[j, 2:]
        value_left = offset + slope * left
        value_right = offset + slope * right
        if min(value_left, value_right) < -1e-7:
            raise ValueError("reduced profile exceeds the full profile")
        total += (
            width
            * (
                value_left * value_left
                + value_left * value_right
                + value_right * value_right
            )
            / 3.0
        )
        left = right
        if full[i, 1] <= right:
            i += 1
        if reduced[j, 1] <= right:
            j += 1
    return float(total)

import numpy as np
from scipy.optimize import linprog


def _projected_family_vertices(labels):
    encoded = encode_pauli_words(labels)
    adjacency = build_frustration_matrix(encoded)
    contexts = enumerate_maximal_contexts(adjacency)
    relations = derive_phase_constraints(encoded, contexts)
    return construct_projected_vertices(contexts, relations)


def compute_integrated_measurement_gain(
    labels: tuple[str, ...],
    coefficients_start: "np.ndarray",
    coefficients_end: "np.ndarray",
    temperatures: "np.ndarray",
    measured_indices: "np.ndarray",
) -> float:
    temperatures = np.asarray(temperatures)
    indices = np.asarray(measured_indices)
    if (
        temperatures.shape != (2,)
        or np.iscomplexobj(temperatures)
        or not np.all(np.isfinite(temperatures))
        or np.any(temperatures < 0)
    ):
        raise ValueError(
            "temperatures must contain two finite nonnegative inverse temperatures"
        )
    if (
        indices.ndim != 1
        or not indices.size
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("measured indices must be a nonempty integer vector")
    if (
        np.any(indices < 0)
        or np.any(indices >= len(labels))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("measured indices must be distinct and in range")
    full_vertices = _projected_family_vertices(labels)
    reduced_labels = tuple(labels[j] for j in indices)
    reduced_vertices = _projected_family_vertices(reduced_labels)
    start = compute_thermal_marginals(
        labels, coefficients_start, float(temperatures[0])
    )
    end = compute_thermal_marginals(
        labels, coefficients_end, float(temperatures[1])
    )
    full_profile = trace_robustness_profile(full_vertices, start, end)
    reduced_profile = trace_robustness_profile(
        reduced_vertices, start[indices], end[indices]
    )
    return integrate_measurement_gain(full_profile, reduced_profile)
SCICODE_GOLD_EOF
