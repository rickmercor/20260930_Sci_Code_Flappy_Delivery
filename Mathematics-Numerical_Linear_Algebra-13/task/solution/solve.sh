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


def compute_dither_variance_coefficient(bits: int) -> float:
    """Reference coefficient with strict signed-bit validation."""
    if isinstance(bits, (bool, np.bool_)) or not isinstance(bits, (int, np.integer)):
        raise ValueError("bits must be an integer at least 2")
    if int(bits) < 2:
        raise ValueError("bits must be an integer at least 2")
    level = 2 ** (int(bits) - 1) - 1
    try:
        coefficient = 1.0 / (12.0 * float(level) ** 2)
    except OverflowError as exc:
        raise ValueError("bits must produce a finite positive coefficient") from exc
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("bits must produce a finite positive coefficient")
    return float(coefficient)

import numpy as np


def _validated_partition_inputs(
    A: np.ndarray,
    B: np.ndarray,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(bound, (bool, np.bool_)) or not np.isscalar(bound):
        raise ValueError("bound must be a finite positive scalar")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite positive scalar below bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be a finite positive scalar")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and below bound")
    return left, right, limit, tol


def _cluster_switches(points: list[float], tolerance: float) -> np.ndarray:
    points.sort()
    clusters: list[list[float]] = [[points[0]]]
    for point in points[1:]:
        if point - clusters[-1][-1] <= tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return np.array([sum(group) / len(group) for group in clusters], dtype=float)


def construct_two_channel_switch_partition(
    A: np.ndarray,
    B: np.ndarray,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference construction of the finite active-range partition."""
    left, right, limit, tol = _validated_partition_inputs(A, B, bound, tolerance)
    points = [-limit, limit]

    for first, second in np.abs(left):
        if first > 0.0 and second > 0.0:
            switch = float(0.5 * np.log(second / first))
            if -limit <= switch <= limit:
                points.append(float(np.clip(switch, -limit, limit)))

    for first, second in np.abs(right.T):
        if first > 0.0 and second > 0.0:
            switch = float(0.5 * np.log(first / second))
            if -limit <= switch <= limit:
                points.append(float(np.clip(switch, -limit, limit)))

    switches = _cluster_switches(points, tol)
    switches[0] = -limit
    switches[-1] = limit
    if switches.size < 2 or np.any(np.diff(switches) <= 0.0):
        raise ValueError("merged partition must contain increasing endpoints")
    probes = 0.5 * (switches[:-1] + switches[1:])
    result = np.concatenate(([float(switches.size)], switches, probes))
    if not np.all(np.isfinite(result)):
        raise ValueError("partition must be finite")
    return result

import numpy as np


def _validate_branch_inputs(
    A: np.ndarray,
    B: np.ndarray,
    partition: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    packed = np.asarray(partition, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if packed.ndim != 1 or packed.size < 4 or not np.all(np.isfinite(packed)):
        raise ValueError("partition must be a finite one-dimensional vector")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite positive scalar")
    try:
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a real scalar") from exc
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be a finite positive scalar")
    count = int(round(float(packed[0])))
    if count < 2 or packed[0] != float(count) or packed.size != 2 * count:
        raise ValueError("partition header or packed length is invalid")
    switches = packed[1 : 1 + count]
    probes = packed[1 + count :]
    widths = np.diff(switches)
    if np.any(widths <= tol):
        raise ValueError("partition switches must be separated by tolerance")
    if probes.shape != (count - 1,):
        raise ValueError("partition probe count is invalid")
    if np.any(probes <= switches[:-1]) or np.any(probes >= switches[1:]):
        raise ValueError("every probe must lie strictly inside its cell")
    if not np.allclose(
        probes,
        0.5 * (switches[:-1] + switches[1:]),
        rtol=0.0,
        atol=tol,
    ):
        raise ValueError("partition probes must be cell midpoints")
    return left, right, switches, probes, tol


def _active_channels(
    left: np.ndarray,
    right: np.ndarray,
    probe: float,
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.exp(np.array([probe, -probe], dtype=float))
    active_a = np.argmax(np.abs(left * scales), axis=1).astype(float)
    active_b = np.argmax(np.abs(right / scales[:, None]), axis=0).astype(float)
    return active_a, active_b


def encode_partition_active_branches(
    A: np.ndarray,
    B: np.ndarray,
    partition: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference active-set encoding on every open partition cell."""
    left, right, switches, probes, _tol = _validate_branch_inputs(
        A, B, partition, tolerance
    )
    records = []
    for index, probe in enumerate(probes):
        active_a, active_b = _active_channels(left, right, float(probe))
        records.append(
            np.concatenate(
                (
                    switches[index : index + 2],
                    [probe],
                    active_a,
                    active_b,
                )
            )
        )
    result = np.concatenate(([float(probes.size)], *records))
    if not np.all(np.isfinite(result)):
        raise ValueError("active branch table must be finite")
    return result

import numpy as np


def _validated_branch_problem(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    probe: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(probe, (bool, np.bool_)) or not np.isscalar(probe):
        raise ValueError("probe must be a finite real scalar")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite positive scalar")
    try:
        point = float(probe)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe and tolerance must be real scalars") from exc
    if not np.isfinite(point):
        raise ValueError("probe must be finite")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return (
        left,
        right,
        compute_dither_variance_coefficient(bits_a),  # noqa: F821 - step 01
        compute_dither_variance_coefficient(bits_b),  # noqa: F821 - step 01
        point,
        tol,
    )


def _reject_nonzero_ties(
    left: np.ndarray,
    right: np.ndarray,
    probe: float,
    tolerance: float,
) -> None:
    for first, second in np.abs(left):
        if first > 0.0 and second > 0.0:
            if abs(probe - 0.5 * np.log(second / first)) <= tolerance:
                raise ValueError("probe lies on a left-factor range tie")
    for first, second in np.abs(right.T):
        if first > 0.0 and second > 0.0:
            if abs(probe - 0.5 * np.log(first / second)) <= tolerance:
                raise ValueError("probe lies on a right-factor range tie")


def reconstruct_active_error_branch(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    probe: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference algebraic reconstruction of one active error branch."""
    left, right, coefficient_a, coefficient_b, point, tol = _validated_branch_problem(
        A, B, bits_a, bits_b, probe, tolerance
    )
    _reject_nonzero_ties(left, right, point, tol)

    scales = np.exp(np.array([point, -point], dtype=float))
    active_a = np.argmax(np.abs(left * scales), axis=1)
    active_b = np.argmax(np.abs(right / scales[:, None]), axis=0)
    amplitudes_a = left[np.arange(left.shape[0]), active_a] ** 2
    amplitudes_b = right[active_b, np.arange(right.shape[1])] ** 2
    exponents_a = np.where(active_a == 0, 2, -2)
    exponents_b = np.where(active_b == 0, -2, 2)
    energy_a = np.sum(left**2, axis=0)
    energy_b = np.sum(right**2, axis=1)

    coefficients = {-4: 0.0, 0: 0.0, 4: 0.0}
    for amplitude, exponent in zip(amplitudes_a, exponents_a):
        coefficients[int(exponent - 2)] += coefficient_a * amplitude * energy_b[0]
        coefficients[int(exponent + 2)] += coefficient_a * amplitude * energy_b[1]
    for amplitude, exponent in zip(amplitudes_b, exponents_b):
        coefficients[int(exponent + 2)] += coefficient_b * amplitude * energy_a[0]
        coefficients[int(exponent - 2)] += coefficient_b * amplitude * energy_a[1]

    cross_scale = 2.0 * coefficient_a * coefficient_b
    for amplitude_a, exponent_a in zip(amplitudes_a, exponents_a):
        for amplitude_b, exponent_b in zip(amplitudes_b, exponents_b):
            coefficients[int(exponent_a + exponent_b)] += (
                cross_scale * amplitude_a * amplitude_b
            )

    positive = coefficients[4]
    negative = coefficients[-4]
    constant = coefficients[0]
    exp_positive = np.exp(4.0 * point)
    exp_negative = np.exp(-4.0 * point)
    value = positive * exp_positive + negative * exp_negative + constant
    derivative = 4.0 * (positive * exp_positive - negative * exp_negative)
    curvature = 16.0 * (positive * exp_positive + negative * exp_negative)
    result = np.concatenate(
        (
            [
                positive,
                negative,
                constant,
                value,
                derivative,
                curvature,
                float(left.shape[0]),
                float(right.shape[1]),
            ],
            active_a.astype(float),
            active_b.astype(float),
        )
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("branch reconstruction must be finite")
    return result

import numpy as np


def _candidate_inputs(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(bound, (bool, np.bool_)) or not np.isscalar(bound):
        raise ValueError("bound must be a finite positive scalar")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be finite, positive, and below bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be a finite positive scalar")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and below bound")
    compute_dither_variance_coefficient(bits_a)  # noqa: F821 - step 01
    compute_dither_variance_coefficient(bits_b)  # noqa: F821 - step 01
    return left, right, limit, tol


def _cell_branch(
    left: np.ndarray,
    right: np.ndarray,
    bits_a: int,
    bits_b: int,
    lower: float,
    upper: float,
    probe: float,
    tolerance: float,
) -> tuple[float, float, float]:
    """Return one smooth cell's ``(P, Q, C)`` branch from the step-04 oracle."""
    tie_tolerance = min(float(tolerance), 0.25 * (float(upper) - float(lower)))
    branch = reconstruct_active_error_branch(  # noqa: F821 - step 04
        left, right, bits_a, bits_b, float(probe), tie_tolerance
    )
    return float(branch[0]), float(branch[1]), float(branch[2])


def _candidate_record(
    x: float,
    kind: float,
    cell: int,
    coefficients: tuple[float, float, float],
) -> np.ndarray:
    positive, negative, constant = coefficients
    exp_positive = np.exp(4.0 * x)
    exp_negative = np.exp(-4.0 * x)
    value = positive * exp_positive + negative * exp_negative + constant
    derivative = 4.0 * (positive * exp_positive - negative * exp_negative)
    return np.array(
        [x, kind, float(cell), positive, negative, constant, value, derivative],
        dtype=float,
    )


def enumerate_bounded_error_candidates(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference enumeration of switches and cell stationary points."""
    left, right, limit, tol = _candidate_inputs(A, B, bits_a, bits_b, bound, tolerance)
    partition = construct_two_channel_switch_partition(  # noqa: F821 - step 02
        left, right, limit, tol
    )
    count = int(round(float(partition[0])))
    switches = np.asarray(partition[1 : 1 + count], dtype=float)
    probes = np.asarray(partition[1 + count :], dtype=float)
    branch_coefficients = [
        _cell_branch(left, right, bits_a, bits_b, lower, upper, probe, tol)
        for lower, upper, probe in zip(switches[:-1], switches[1:], probes)
    ]

    records = []
    last_cell = len(branch_coefficients) - 1
    for index, point in enumerate(switches):
        cell = min(index, last_cell)
        kind = -1.0 if index == 0 else (1.0 if index == switches.size - 1 else 0.0)
        records.append(
            _candidate_record(float(point), kind, cell, branch_coefficients[cell])
        )

    for cell, ((lower, upper), coefficients) in enumerate(
        zip(zip(switches[:-1], switches[1:]), branch_coefficients)
    ):
        positive, negative, _constant = coefficients
        if positive > 0.0 and negative > 0.0:
            stationary = float(0.125 * np.log(negative / positive))
            if lower + tol < stationary < upper - tol:
                records.append(_candidate_record(stationary, 2.0, cell, coefficients))

    records.sort(key=lambda record: (record[0], record[1]))
    unique = []
    for record in records:
        if unique and abs(record[0] - unique[-1][0]) <= tol:
            if record[1] != 2.0 and unique[-1][1] == 2.0:
                unique[-1] = record
        else:
            unique.append(record)
    table = np.vstack(unique)
    result = np.concatenate(([float(table.shape[0])], table.ravel()))
    if not np.all(np.isfinite(result)):
        raise ValueError("candidate table must be finite")
    return result

import numpy as np


def compute_expected_product_error(
    A_tilde: np.ndarray,
    B_tilde: np.ndarray,
    variance_a: np.ndarray,
    variance_b: np.ndarray,
) -> np.ndarray:
    """Reference evaluation of the finite-dimensional identity."""
    left = np.asarray(A_tilde, dtype=float)
    right = np.asarray(B_tilde, dtype=float)
    field_a = np.asarray(variance_a, dtype=float)
    field_b = np.asarray(variance_b, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] < 1:
        raise ValueError("A_tilde must be a nonempty 2D array")
    if right.ndim != 2 or right.shape[0] < 1 or right.shape[1] < 1:
        raise ValueError("B_tilde must be a nonempty 2D array")
    if left.shape[1] != right.shape[0]:
        raise ValueError("contracted dimensions must agree")
    if field_a.shape != left.shape or field_b.shape != right.shape:
        raise ValueError("variance fields must match their factor shapes")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("factor entries must be finite")
    if not np.all(np.isfinite(field_a)) or np.any(field_a < 0.0):
        raise ValueError("variance_a must be finite and nonnegative")
    if not np.all(np.isfinite(field_b)) or np.any(field_b < 0.0):
        raise ValueError("variance_b must be finite and nonnegative")

    row_energies = np.sum(right**2, axis=1)
    column_energies = np.sum(left**2, axis=0)
    left_contribution = float(np.sum(field_a * row_energies[None, :]))
    right_contribution = float(np.sum(field_b * column_energies[:, None]))
    simultaneous = float(np.sum(np.sum(field_a, axis=0) * np.sum(field_b, axis=1)))
    total = left_contribution + right_contribution + simultaneous
    result = np.array(
        [left_contribution, right_contribution, simultaneous, total], dtype=float
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("expected error must be finite")
    return result

import numpy as np


def _certificate_inputs(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(bound, (bool, np.bool_)) or not np.isscalar(bound):
        raise ValueError("bound must be a finite positive scalar")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be finite, positive, and below bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be a finite positive scalar")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and below bound")
    return (
        left,
        right,
        compute_dither_variance_coefficient(bits_a),  # noqa: F821 - step 01
        compute_dither_variance_coefficient(bits_b),  # noqa: F821 - step 01
        limit,
        tol,
    )


def _cell_branch(
    left: np.ndarray,
    right: np.ndarray,
    bits_a: int,
    bits_b: int,
    lower: float,
    upper: float,
    probe: float,
    tolerance: float,
) -> np.ndarray:
    """Return one smooth cell's ``(P, Q, C)`` branch from the step-04 oracle."""
    tie_tolerance = min(float(tolerance), 0.25 * (float(upper) - float(lower)))
    branch = reconstruct_active_error_branch(  # noqa: F821 - step 04
        left, right, bits_a, bits_b, float(probe), tie_tolerance
    )
    return np.array([branch[0], branch[1], branch[2]], dtype=float)


def _certificate_error(
    left: np.ndarray,
    right: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
    x: float,
) -> float:
    scales = np.exp(np.array([x, -x], dtype=float))
    transformed_left = left * scales
    transformed_right = right / scales[:, None]
    variance_a = coefficient_a * np.max(np.abs(transformed_left), axis=1) ** 2
    variance_b = coefficient_b * np.max(np.abs(transformed_right), axis=0) ** 2
    field_a = np.repeat(variance_a[:, None], 2, axis=1)
    field_b = np.repeat(variance_b[None, :], 2, axis=0)
    return float(
        compute_expected_product_error(  # noqa: F821 - step 06
            transformed_left, transformed_right, field_a, field_b
        )[3]
    )


def _branch_value(coefficients: np.ndarray, x: float) -> float:
    return float(
        coefficients[0] * np.exp(4.0 * x)
        + coefficients[1] * np.exp(-4.0 * x)
        + coefficients[2]
    )


def _branch_derivative(coefficients: np.ndarray, x: float) -> float:
    return float(
        4.0 * (coefficients[0] * np.exp(4.0 * x) - coefficients[1] * np.exp(-4.0 * x))
    )


def certify_bounded_shared_scaling(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference global certificate from active cells and finite candidates."""
    left, right, coefficient_a, coefficient_b, limit, tol = _certificate_inputs(
        A, B, bits_a, bits_b, bound, tolerance
    )
    identity_error = _certificate_error(left, right, coefficient_a, coefficient_b, 0.0)
    if not np.isfinite(identity_error) or identity_error <= 0.0:
        raise ValueError("identity-transform error must be finite and positive")

    partition = construct_two_channel_switch_partition(  # noqa: F821 - step 02
        left, right, limit, tol
    )
    count = int(round(float(partition[0])))
    switches = np.asarray(partition[1 : 1 + count], dtype=float)
    probes = np.asarray(partition[1 + count :], dtype=float)
    branches = [
        _cell_branch(left, right, bits_a, bits_b, lower, upper, probe, tol)
        for lower, upper, probe in zip(switches[:-1], switches[1:], probes)
    ]
    table = enumerate_bounded_error_candidates(  # noqa: F821 - step 05
        left, right, bits_a, bits_b, limit, tol
    )
    records = np.asarray(table[1:], dtype=float).reshape(int(round(float(table[0]))), 8)
    candidates = records[:, [0, 1, 2, 6]]
    minimum = float(np.min(candidates[:, 3]))
    energy_tol = tol * max(1.0, abs(minimum))
    eligible = np.flatnonzero(candidates[:, 3] <= minimum + energy_tol)
    best = int(eligible[np.argmin(candidates[eligible, 0])])
    x_star, kind, cell_value, optimized_error = candidates[best]
    cell = int(cell_value)

    switch_distance = np.abs(switches - x_star)
    switch_index = int(np.argmin(switch_distance))
    if switch_distance[switch_index] <= tol:
        left_cell = max(0, switch_index - 1)
        right_cell = min(len(branches) - 1, switch_index)
    else:
        left_cell = right_cell = cell
    left_derivative = _branch_derivative(branches[left_cell], float(x_star))
    right_derivative = _branch_derivative(branches[right_cell], float(x_star))

    distinct = np.flatnonzero(np.abs(candidates[:, 0] - x_star) > tol)
    gap = (
        float(np.min(candidates[distinct, 3]) - optimized_error)
        if distinct.size
        else 0.0
    )
    gap = max(0.0, gap)

    max_branch_residual = 0.0
    for (lower, upper), coefficients in zip(zip(switches[:-1], switches[1:]), branches):
        for fraction in (0.25, 0.75):
            point = float(lower + fraction * (upper - lower))
            direct = _certificate_error(
                left, right, coefficient_a, coefficient_b, point
            )
            max_branch_residual = max(
                max_branch_residual,
                abs(direct - _branch_value(coefficients, point)),
            )

    scales = np.exp(np.array([x_star, -x_star], dtype=float))
    transformed_left = left * scales
    transformed_right = right / scales[:, None]
    product_residual = float(
        np.max(np.abs(transformed_left @ transformed_right - left @ right))
    )
    result = np.array(
        [
            x_star,
            scales[0],
            scales[1],
            optimized_error,
            identity_error,
            float(switches.size),
            float(candidates.shape[0]),
            kind,
            float(best),
            left_derivative,
            right_derivative,
            gap,
            max_branch_residual,
            product_residual,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("global certificate must be finite")
    return result

import numpy as np


def _local_validate(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(bound, (bool, np.bool_)) or not np.isscalar(bound):
        raise ValueError("bound must be finite and strictly positive")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be finite, positive, and smaller than bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be finite real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be finite and strictly positive")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and smaller than bound")
    return (
        left,
        right,
        compute_dither_variance_coefficient(bits_a),  # noqa: F821 - earlier pipeline step
        compute_dither_variance_coefficient(bits_b),  # noqa: F821 - earlier pipeline step
        limit,
        tol,
    )


def _local_transform(
    left: np.ndarray, right: np.ndarray, x: float
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.exp(np.array([x, -x], dtype=float))
    return left * scales, right / scales[:, None]


def _local_ranges(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.max(np.abs(left), axis=1), np.max(np.abs(right), axis=0)


def _local_fields(
    ranges_a: np.ndarray,
    ranges_b: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    field_a = np.repeat((coefficient_a * ranges_a**2)[:, None], 2, axis=1)
    field_b = np.repeat((coefficient_b * ranges_b**2)[None, :], 2, axis=0)
    return field_a, field_b


def _local_error(
    left: np.ndarray,
    right: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
    x: float,
) -> float:
    transformed_left, transformed_right = _local_transform(left, right, x)
    ranges_a, ranges_b = _local_ranges(transformed_left, transformed_right)
    field_a, field_b = _local_fields(ranges_a, ranges_b, coefficient_a, coefficient_b)
    return float(
        compute_expected_product_error(  # noqa: F821 - step 06
            transformed_left, transformed_right, field_a, field_b
        )[3]
    )


def compute_quantized_product_error_reduction(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> float:
    """Reference end-to-end reduction with all certificate stages checked."""
    left, right, coefficient_a, coefficient_b, limit, tol = _local_validate(
        A, B, bits_a, bits_b, bound, tolerance
    )

    identity_left, identity_right = _local_transform(left, right, 0.0)
    if not np.allclose(
        identity_left @ identity_right, left @ right, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("inverse-pair transform must preserve the product")
    ranges_a, ranges_b = _local_ranges(identity_left, identity_right)
    field_a, field_b = _local_fields(ranges_a, ranges_b, coefficient_a, coefficient_b)
    baseline_parts = compute_expected_product_error(  # noqa: F821 - earlier pipeline step
        identity_left, identity_right, field_a, field_b
    )
    baseline = float(baseline_parts[3])
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(
            "identity-transform expected error must be positive and finite"
        )

    partition = construct_two_channel_switch_partition(left, right, limit, tol)  # noqa: F821 - earlier pipeline step
    active_table = encode_partition_active_branches(left, right, partition, tol)  # noqa: F821 - earlier pipeline step
    switch_count = int(round(float(partition[0])))
    first_probe = float(partition[1 + switch_count])
    branch = reconstruct_active_error_branch(  # noqa: F821 - earlier pipeline step
        left, right, bits_a, bits_b, first_probe, tol
    )
    candidates = enumerate_bounded_error_candidates(  # noqa: F821 - earlier pipeline step
        left, right, bits_a, bits_b, limit, tol
    )
    certificate = certify_bounded_shared_scaling(  # noqa: F821 - earlier pipeline step
        left, right, bits_a, bits_b, limit, tol
    )
    if int(round(float(active_table[0]))) != switch_count - 1:
        raise ValueError("active-cell count does not match the partition")
    if int(round(float(candidates[0]))) != int(round(float(certificate[6]))):
        raise ValueError("candidate count does not match the certificate")
    probe_error = _local_error(left, right, coefficient_a, coefficient_b, first_probe)
    if abs(float(branch[3]) - probe_error) > 50.0 * tol * max(1.0, abs(probe_error)):
        raise ValueError("branch reconstruction does not match direct error")
    if certificate[12] > 100.0 * tol * max(1.0, baseline):
        raise ValueError("branch reconstruction residual is too large")
    if certificate[13] > 100.0 * tol * max(1.0, float(np.max(np.abs(left @ right)))):
        raise ValueError("product-preservation residual is too large")

    x_star = float(certificate[0])
    optimized_left, optimized_right = _local_transform(left, right, x_star)
    if not np.allclose(
        optimized_left @ optimized_right, left @ right, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("inverse-pair transform must preserve the product")
    ranges_a, ranges_b = _local_ranges(optimized_left, optimized_right)
    field_a, field_b = _local_fields(ranges_a, ranges_b, coefficient_a, coefficient_b)
    optimized_parts = compute_expected_product_error(  # noqa: F821 - earlier pipeline step
        optimized_left, optimized_right, field_a, field_b
    )
    optimized = float(optimized_parts[3])
    if abs(optimized - float(certificate[3])) > 50.0 * tol * max(1.0, abs(optimized)):
        raise ValueError("certified and directly evaluated optima disagree")
    if optimized > baseline and optimized - baseline <= tol * max(1.0, baseline):
        optimized = baseline
    reduction = 100.0 * (1.0 - optimized / baseline)
    if not np.isfinite(reduction):
        raise ValueError("error reduction must be finite")
    return float(reduction)
SCICODE_GOLD_EOF
