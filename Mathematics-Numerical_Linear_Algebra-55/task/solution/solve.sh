#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from math import comb

import numpy as np


def _validated_node_count(n: int) -> int:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if int(n) < 18:
        raise ValueError("n must be at least 18")
    return int(n)


def _validated_closure_jets(target_jet, functional_jet):
    if target_jet is None:
        targets = np.array([342523.0 / 518400.0, 1.0, 0.0])
    else:
        raw_targets = np.asarray(target_jet)
        if (
            raw_targets.ndim != 1
            or not 1 <= raw_targets.size <= 9
            or np.iscomplexobj(raw_targets)
            or not np.all(np.isfinite(raw_targets))
        ):
            raise ValueError(
                "target_jet must have between one and nine finite real entries"
            )
        targets = np.asarray(raw_targets, dtype=float)
    if functional_jet is None:
        functionals = np.zeros((3, 6, 9), dtype=float)
        functionals[0, 4, 5] = 1.0
    else:
        raw_functionals = np.asarray(functional_jet)
        if (
            raw_functionals.ndim != 3
            or raw_functionals.shape[1:] != (6, 9)
            or not 1 <= raw_functionals.shape[0] <= 9
            or np.iscomplexobj(raw_functionals)
            or not np.all(np.isfinite(raw_functionals))
        ):
            raise ValueError(
                "functional_jet must have finite real shape (r, 6, 9), 1 <= r <= 9"
            )
        functionals = np.asarray(raw_functionals, dtype=float)
    if targets.size != functionals.shape[0]:
        raise ValueError("target_jet and functional_jet must have the same jet length")
    return targets, functionals


_BOUNDARY_NORM = (
    13649.0 / 43200.0,
    12013.0 / 8640.0,
    2711.0 / 4320.0,
    5359.0 / 4320.0,
    7877.0 / 8640.0,
    43801.0 / 43200.0,
)
_BOUNDARY_BLOCK = np.array(
    [
        [
            -1.0 / 2.0,
            104009.0 / 172800.0,
            30443.0 / 259200.0,
            -33311.0 / 86400.0,
            5621.0 / 28800.0,
            -601.0 / 20736.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -104009.0 / 172800.0,
            0.0,
            -311.0 / 51840.0,
            6743.0 / 5760.0,
            -24337.0 / 34560.0,
            36661.0 / 259200.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -30443.0 / 259200.0,
            311.0 / 51840.0,
            0.0,
            -2231.0 / 5184.0,
            41287.0 / 51840.0,
            -7333.0 / 28800.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            33311.0 / 86400.0,
            -6743.0 / 5760.0,
            2231.0 / 5184.0,
            0.0,
            4147.0 / 17280.0,
            25427.0 / 259200.0,
            1.0 / 60.0,
            0.0,
            0.0,
        ],
        [
            -5621.0 / 28800.0,
            24337.0 / 34560.0,
            -41287.0 / 51840.0,
            -4147.0 / 17280.0,
            0.0,
            342523.0 / 518400.0,
            -3.0 / 20.0,
            1.0 / 60.0,
            0.0,
        ],
        [
            601.0 / 20736.0,
            -36661.0 / 259200.0,
            7333.0 / 28800.0,
            -25427.0 / 259200.0,
            -342523.0 / 518400.0,
            0.0,
            3.0 / 4.0,
            -3.0 / 20.0,
            1.0 / 60.0,
        ],
    ],
    dtype=float,
)
_BOUNDARY_BLOCK_SLOPE = np.array(
    [
        [0.0, 1.0, -4.0, 6.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 10.0, -20.0, 15.0, -4.0, 0.0, 0.0, 0.0],
        [4.0, -10.0, 0.0, 20.0, -20.0, 6.0, 0.0, 0.0, 0.0],
        [-6.0, 20.0, -20.0, 0.0, 10.0, -4.0, 0.0, 0.0, 0.0],
        [4.0, -15.0, 20.0, -10.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)
_INTERIOR_STENCIL = np.array(
    [-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0]
)


def build_sbp63_operator(
    n: int,
    target_jet: np.ndarray = None,
    functional_jet: np.ndarray = None,
) -> np.ndarray:
    """Reference functional selection and exact runtime-order path jet."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if int(n) < 18:
        raise ValueError("n must be at least 18")
    n = int(n)
    targets, functionals = _validated_closure_jets(target_jet, functional_jet)
    directional_values = np.einsum("kij,ij->k", functionals, _BOUNDARY_BLOCK_SLOPE)
    reference_values = np.einsum("kij,ij->k", functionals, _BOUNDARY_BLOCK)
    functional_direction = directional_values[0]
    functional_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.linalg.norm(functionals[0]))
            * float(np.linalg.norm(_BOUNDARY_BLOCK_SLOPE)),
        )
    )
    if abs(functional_direction) <= functional_tolerance:
        raise ValueError("F(0) does not select a unique family member")
    spacing = 1.0 / (n - 1)

    weights = np.ones(n, dtype=float)
    weights[:6] = _BOUNDARY_NORM
    weights[-6:] = weights[5::-1]
    h_matrix = spacing * np.diag(weights)

    jet_length = targets.size
    family_jet = np.zeros(jet_length, dtype=float)
    for order in range(jet_length):
        known = reference_values[order]
        for functional_order in range(1, order + 1):
            known += (
                comb(order, functional_order)
                * directional_values[functional_order]
                * family_jet[order - functional_order]
            )
        family_jet[order] = (targets[order] - known) / functional_direction

    output = [h_matrix]
    for order, family_derivative in enumerate(family_jet):
        boundary = family_derivative * _BOUNDARY_BLOCK_SLOPE
        if order == 0:
            boundary = _BOUNDARY_BLOCK + boundary
        q_derivative = np.zeros((n, n), dtype=float)
        q_derivative[:6, :9] = boundary
        q_derivative[-6:, -9:] = -boundary[::-1, ::-1]
        if order == 0:
            for row in range(6, n - 6):
                q_derivative[row, row - 3 : row + 4] = _INTERIOR_STENCIL
        d_derivative = q_derivative / (spacing * weights)[:, None]
        output.extend((q_derivative, d_derivative))
    return np.stack(output)

import numpy as np

_ORIGINS = (-1.0, -0.35, 0.1)
_LENGTHS = (0.9, 0.7, 0.9)


def _validated_grid_inputs(t, counts, sigmas, amplitude, period):
    counts = np.asarray(counts)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,) or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas must be a finite array of shape (3,)")
    if np.any(np.abs(sigmas) >= 1.0):
        raise ValueError("every stretching parameter must satisfy |sigma| < 1")
    values = np.asarray([t, amplitude, period], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("t, amplitude, and period must be finite")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    return float(t), counts.astype(int), sigmas, float(amplitude), float(period)




def compute_moving_grid_state(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    amplitude: float = 0.1,
    period: float = 1.0,
) -> np.ndarray:
    """Reference evaluation of the stretched translating overset geometry."""
    counts = np.asarray(counts)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,) or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas must be a finite array of shape (3,)")
    if np.any(np.abs(sigmas) >= 1.0):
        raise ValueError("every stretching parameter must satisfy |sigma| < 1")
    if not np.all(np.isfinite(np.asarray([t, amplitude, period], dtype=float))):
        raise ValueError("t, amplitude, and period must be finite")
    if float(period) <= 0.0:
        raise ValueError("period must be positive")
    t = float(t)
    counts = counts.astype(int)
    amplitude = float(amplitude)
    period = float(period)
    angle = 2.0 * np.pi * t / period
    shift = amplitude * np.sin(angle)
    velocity = 2.0 * np.pi * amplitude * np.cos(angle) / period

    pieces = [np.array([velocity])]
    for grid in range(3):
        xi = np.linspace(0.0, 1.0, counts[grid])
        stretched = xi - (sigmas[grid] / np.pi) * np.sin(np.pi * xi)
        origin = _ORIGINS[grid] + (shift if grid == 1 else 0.0)
        pieces.append(origin + _LENGTHS[grid] * stretched)
    return np.concatenate(pieces)

from math import factorial

import numpy as np

_LOCAL_NODES = np.array([0.0, 1.0, 2.0, 3.0])


def _cardinal_coefficients():
    coefficients = np.zeros((4, 4), dtype=float)
    for basis_index in range(4):
        polynomial = np.array([1.0])
        for node_index in range(4):
            if node_index != basis_index:
                factor = np.array([-_LOCAL_NODES[node_index], 1.0])
                polynomial = np.convolve(polynomial, factor) / (
                    _LOCAL_NODES[basis_index] - _LOCAL_NODES[node_index]
                )
        coefficients[basis_index] = polynomial
    return coefficients


_CARDINAL_COEFFICIENTS = _cardinal_coefficients()


def _validated_donor_data(donor_nodes, receiver_x, derivative_order):
    donor_nodes = np.asarray(donor_nodes, dtype=float)
    if donor_nodes.ndim != 1 or donor_nodes.size < 4:
        raise ValueError("donor_nodes must be a 1D array with at least four entries")
    receivers = np.asarray(receiver_x, dtype=float)
    scalar_receiver = receivers.ndim == 0
    if receivers.ndim > 1 or (receivers.ndim == 1 and receivers.size == 0):
        raise ValueError("receiver_x must be a finite scalar or nonempty 1D array")
    receivers = np.atleast_1d(receivers)
    if not np.all(np.isfinite(donor_nodes)) or not np.all(np.isfinite(receivers)):
        raise ValueError("donor data must be finite")
    if np.any(np.diff(donor_nodes) <= 0.0):
        raise ValueError("donor_nodes must be strictly increasing")
    if np.any(receivers < donor_nodes[0]) or np.any(receivers > donor_nodes[-1]):
        raise ValueError("every receiver must lie in the closed donor interval")
    if isinstance(derivative_order, (bool, np.bool_)) or not isinstance(
        derivative_order, (int, np.integer)
    ):
        raise ValueError("derivative_order must be an integer")
    if not 0 <= int(derivative_order) <= 8:
        raise ValueError("derivative_order must be between zero and eight")
    return donor_nodes, receivers, scalar_receiver, int(derivative_order)


def _cubic_basis_value_and_slope(z):
    powers = np.array([1.0, z, z * z, z * z * z])
    values = _CARDINAL_COEFFICIENTS @ powers
    slopes = (
        _CARDINAL_COEFFICIENTS[:, 1]
        + 2.0 * z * _CARDINAL_COEFFICIENTS[:, 2]
        + 3.0 * z * z * _CARDINAL_COEFFICIENTS[:, 3]
    )
    return values, slopes


def _series_product(first, second, order):
    return np.convolve(first, second)[: order + 1]


def _compose_polynomial(polynomial, series, order):
    result = np.zeros(order + 1, dtype=float)
    for coefficient in polynomial[::-1]:
        result = _series_product(result, series, order)
        result[0] += coefficient
    return result


def compute_donor_interpolation(
    donor_nodes: np.ndarray,
    receiver_x: float | np.ndarray,
    derivative_order: int = 6,
) -> np.ndarray:
    """Reference batched, safeguarded computational-space interpolation."""
    donor_nodes, receivers, scalar_receiver, derivative_order = _validated_donor_data(
        donor_nodes, receiver_x, derivative_order
    )
    size = donor_nodes.size
    result = np.zeros((receivers.size, derivative_order + 1, size), dtype=float)
    for receiver_index, receiver in enumerate(receivers):
        cell = int(np.searchsorted(donor_nodes, receiver, side="right")) - 1
        cell = min(max(cell, 0), size - 2)
        start = min(max(cell - 1, 0), size - 4)
        stencil = donor_nodes[start : start + 4]

        lower = float(cell - start)
        upper = lower + 1.0
        z = lower + (receiver - donor_nodes[cell]) / (
            donor_nodes[cell + 1] - donor_nodes[cell]
        )
        tolerance = 1e-14 * max(1.0, abs(float(receiver)))
        for _ in range(100):
            values, slopes = _cubic_basis_value_and_slope(z)
            residual = float(stencil @ values) - float(receiver)
            if abs(residual) <= tolerance:
                break
            if residual < 0.0:
                lower = z
            else:
                upper = z
            jacobian = float(stencil @ slopes)
            candidate = z - residual / jacobian if jacobian > 0.0 else np.nan
            if not np.isfinite(candidate) or candidate <= lower or candidate >= upper:
                candidate = 0.5 * (lower + upper)
            z = candidate
        else:
            raise ValueError("the donor inverse map did not converge")

        values, slopes = _cubic_basis_value_and_slope(z)
        map_slope = float(stencil @ slopes)
        if not np.isfinite(map_slope) or map_slope <= 0.0:
            raise ValueError("the donor inverse map has a nonpositive derivative")

        map_polynomial = stencil @ _CARDINAL_COEFFICIENTS
        inverse_series = np.zeros(derivative_order + 1, dtype=float)
        inverse_series[0] = z
        for order in range(1, derivative_order + 1):
            known = _compose_polynomial(map_polynomial, inverse_series, order)[order]
            target = 1.0 if order == 1 else 0.0
            inverse_series[order] = (target - known) / map_slope
        for basis_index, polynomial in enumerate(_CARDINAL_COEFFICIENTS):
            composed = _compose_polynomial(polynomial, inverse_series, derivative_order)
            for order in range(derivative_order + 1):
                result[receiver_index, order, start + basis_index] = (
                    factorial(order) * composed[order]
                )
    return result[0] if scalar_receiver else result

import numpy as np


def _validated_metric_inputs(c, grid_velocity, counts, nodes):
    counts = np.asarray(counts)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size != int(np.sum(counts)):
        raise ValueError("nodes must be a 1D array of length counts.sum()")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if not np.isfinite(c) or float(c) == 0.0:
        raise ValueError("c must be finite and nonzero")
    if not np.isfinite(grid_velocity):
        raise ValueError("grid_velocity must be finite")
    return float(c), float(grid_velocity), counts.astype(int), nodes


def compute_relative_metric_diagonals(
    c: float, grid_velocity: float, counts: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """Reference discrete-Jacobian evaluation of the nodal scales."""
    c, grid_velocity, counts, nodes = _validated_metric_inputs(
        c, grid_velocity, counts, nodes
    )
    starts = np.r_[0, np.cumsum(counts)]
    pieces = []
    for grid in range(3):
        derivative = build_sbp63_operator(int(counts[grid]))[2]
        jacobian = derivative @ nodes[starts[grid] : starts[grid + 1]]
        relative = c - (grid_velocity if grid == 1 else 0.0)
        if relative == 0.0 or np.sign(relative) != np.sign(c):
            raise ValueError("all grid-relative characteristics must share c's sign")
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = abs(relative) / jacobian
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(
                "the discrete Jacobian and metric magnitude must stay positive"
            )
        pieces.append(scale)
    return np.concatenate(pieces)

from math import factorial

import numpy as np


def _validated_system_inputs(
    t, counts, sigmas, c, amplitude, period, penalties, beta, x0, jet_order
):
    counts = np.asarray(counts)
    penalties = np.asarray(penalties, dtype=float)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    if penalties.shape != (3,) or not np.all(np.isfinite(penalties)):
        raise ValueError("penalties must be a finite array of shape (3,)")
    if np.any(penalties < 0.5):
        raise ValueError("every penalty must be at least 0.5")
    values = np.asarray([t, c, amplitude, period, beta, x0], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("all scalar inputs must be finite")
    if float(c) == 0.0 or float(period) <= 0.0 or float(beta) <= 0.0:
        raise ValueError("c must be nonzero; period and beta must be positive")
    if isinstance(jet_order, (bool, np.bool_)) or not isinstance(
        jet_order, (int, np.integer)
    ):
        raise ValueError("jet_order must be an integer")
    if not 0 <= int(jet_order) <= 8:
        raise ValueError("jet_order must be between zero and eight")
    return (
        float(t),
        counts.astype(int),
        float(c),
        float(amplitude),
        float(period),
        penalties,
        float(beta),
        float(x0),
        int(jet_order),
    )


def _jet_product(first, second):
    """Multiply two truncated Taylor-coefficient jets."""
    order = first.shape[0] - 1
    trailing_shape = np.broadcast_shapes(first.shape[1:], second.shape[1:])
    result = np.zeros((order + 1,) + trailing_shape, dtype=float)
    for total_order in range(order + 1):
        for first_order in range(total_order + 1):
            result[total_order] += (
                first[first_order] * second[total_order - first_order]
            )
    return result


def _jet_compose(outer, inner):
    """Compose a Taylor jet with a scalar zero-constant Taylor jet."""
    result = np.zeros_like(outer, dtype=float)
    for coefficient in outer[::-1]:
        result = _jet_product(result, inner)
        result[0] += coefficient
    return result


def _jet_exponential(exponent):
    """Exponentiate a scalar Taylor-coefficient jet."""
    order = exponent.size - 1
    result = np.zeros(order + 1, dtype=float)
    result[0] = np.exp(exponent[0])
    for total_order in range(1, order + 1):
        result[total_order] = (
            sum(
                exponent_order
                * exponent[exponent_order]
                * result[total_order - exponent_order]
                for exponent_order in range(1, total_order + 1)
            )
            / total_order
        )
    return result


def _jet_outer(first, second):
    """Form the Taylor jet of an outer product."""
    order = first.shape[0] - 1
    result = np.zeros((order + 1, first.shape[1], second.shape[1]), dtype=float)
    for total_order in range(order + 1):
        for first_order in range(total_order + 1):
            result[total_order] += np.outer(
                first[first_order], second[total_order - first_order]
            )
    return result


def assemble_weak_overset_system(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    beta: float = 80.0,
    x0: float = -0.4,
    jet_order: int = 6,
) -> np.ndarray:
    """Reference affine-system value and exact runtime-order time jet."""
    (
        t,
        counts,
        c,
        amplitude,
        period,
        penalties,
        beta,
        x0,
        jet_order,
    ) = _validated_system_inputs(
        t,
        counts,
        sigmas,
        c,
        amplitude,
        period,
        penalties,
        beta,
        x0,
        jet_order,
    )
    state = compute_moving_grid_state(t, counts, sigmas, amplitude, period)
    nodes = state[1:]
    angular_frequency = 2.0 * np.pi / period
    derivative_indices = np.arange(jet_order + 2)
    shift_derivatives = (
        amplitude
        * angular_frequency**derivative_indices
        * np.sin(angular_frequency * t + 0.5 * np.pi * derivative_indices)
    )
    velocity = float(shift_derivatives[1])
    metric = compute_relative_metric_diagonals(c, velocity, counts, nodes)

    direction = 1 if c > 0.0 else -1
    if np.sign(c - velocity) != direction:
        raise ValueError("all grid-relative characteristics must share c's sign")

    starts = np.r_[0, np.cumsum(counts)]
    left = nodes[starts[0] : starts[1]]
    middle = nodes[starts[1] : starts[2]]
    right = nodes[starts[2] : starts[3]]
    factorials = np.array([float(factorial(order)) for order in range(jet_order + 1)])

    metric_series = np.zeros((jet_order + 1, nodes.size), dtype=float)
    metric_series[0] = metric
    middle_derivative = build_sbp63_operator(int(counts[1]))[2]
    middle_jacobian = middle_derivative @ middle
    for order in range(1, jet_order + 1):
        metric_series[order, starts[1] : starts[2]] = (
            -direction
            * shift_derivatives[order + 1]
            / (factorials[order] * middle_jacobian)
        )

    if direction > 0:
        if middle[0] < left[0] or middle[0] > left[-1]:
            raise ValueError("the middle receiver has left the left donor grid")
        if right[0] < middle[0] or right[0] > middle[-1]:
            raise ValueError("the right receiver has left the middle donor grid")
        first_data = compute_donor_interpolation(left, middle[0], jet_order)
        second_data = compute_donor_interpolation(middle, right[0], jet_order)
        first_shift_sign = 1.0
        second_shift_sign = -1.0
    else:
        if left[-1] < middle[0] or left[-1] > middle[-1]:
            raise ValueError("the left receiver has left the middle donor grid")
        if middle[-1] < right[0] or middle[-1] > right[-1]:
            raise ValueError("the middle receiver has left the right donor grid")
        first_data = compute_donor_interpolation(middle, left[-1], jet_order)
        second_data = compute_donor_interpolation(right, middle[-1], jet_order)
        first_shift_sign = -1.0
        second_shift_sign = 1.0

    first_displacement = np.zeros(jet_order + 1, dtype=float)
    second_displacement = np.zeros(jet_order + 1, dtype=float)
    first_displacement[1:] = (
        first_shift_sign * shift_derivatives[1 : jet_order + 1] / factorials[1:]
    )
    second_displacement[1:] = (
        second_shift_sign * shift_derivatives[1 : jet_order + 1] / factorials[1:]
    )
    first_weight_series = _jet_compose(
        first_data / factorials[:, None], first_displacement
    )
    second_weight_series = _jet_compose(
        second_data / factorials[:, None], second_displacement
    )

    total = int(np.sum(counts))
    matrix_series = np.zeros((jet_order + 1, total, total), dtype=float)
    forcing_series = np.zeros((jet_order + 1, total), dtype=float)
    incoming = 0 if direction > 0 else -1
    operators = [build_sbp63_operator(int(counts[grid])) for grid in range(3)]
    corners = []
    for grid in range(3):
        h_matrix = operators[grid][0]
        derivative = operators[grid][2]
        corners.append(float(h_matrix[incoming, incoming]))
        begin, end = starts[grid], starts[grid + 1]
        for order in range(jet_order + 1):
            scale = metric_series[order, begin:end]
            block = -direction * scale[:, None] * derivative
            block[incoming, incoming] -= (
                penalties[grid] * scale[incoming] / h_matrix[incoming, incoming]
            )
            matrix_series[order, begin:end, begin:end] = block

    if direction > 0:
        first_column = np.zeros((jet_order + 1, int(counts[1])), dtype=float)
        first_column[:, 0] = penalties[1] * metric_series[:, starts[1]] / corners[1]
        second_column = np.zeros((jet_order + 1, int(counts[2])), dtype=float)
        second_column[:, 0] = penalties[2] * metric_series[:, starts[2]] / corners[2]
        matrix_series[:, starts[1] : starts[2], starts[0] : starts[1]] = _jet_outer(
            first_column, first_weight_series
        )
        matrix_series[:, starts[2] : starts[3], starts[1] : starts[2]] = _jet_outer(
            second_column, second_weight_series
        )
        inflow_index = starts[0]
        inflow_grid = 0
        inflow_x = -1.0
    else:
        first_column = np.zeros((jet_order + 1, int(counts[0])), dtype=float)
        first_column[:, -1] = (
            penalties[0] * metric_series[:, starts[1] - 1] / corners[0]
        )
        second_column = np.zeros((jet_order + 1, int(counts[1])), dtype=float)
        second_column[:, -1] = (
            penalties[1] * metric_series[:, starts[2] - 1] / corners[1]
        )
        matrix_series[:, starts[0] : starts[1], starts[1] : starts[2]] = _jet_outer(
            first_column, first_weight_series
        )
        matrix_series[:, starts[1] : starts[2], starts[2] : starts[3]] = _jet_outer(
            second_column, second_weight_series
        )
        inflow_index = starts[3] - 1
        inflow_grid = 2
        inflow_x = 1.0

    residual = inflow_x - x0 - c * t
    inflow_exponent = np.zeros(jet_order + 1, dtype=float)
    inflow_exponent[0] = -beta * residual**2
    if jet_order >= 1:
        inflow_exponent[1] = 2.0 * beta * c * residual
    if jet_order >= 2:
        inflow_exponent[2] = -beta * c**2
    inflow_series = _jet_exponential(inflow_exponent)
    coefficient_series = (
        penalties[inflow_grid] * metric_series[:, inflow_index] / corners[inflow_grid]
    )
    forcing_series[:, inflow_index] = _jet_product(coefficient_series, inflow_series)

    augmented_series = np.concatenate(
        (matrix_series, forcing_series[:, :, None]), axis=2
    )
    return augmented_series * factorials[:, None, None]

import numpy as np


def compute_weighted_stability_diagnostics(
    system_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    resolvent_shifts: np.ndarray = None,
) -> np.ndarray:
    """Reference quotient reduction and Cholesky-scaled resolvent calculation."""
    raw_matrix = np.asarray(system_matrix)
    raw_energy = np.asarray(energy_matrix)
    if np.iscomplexobj(raw_matrix) or np.iscomplexobj(raw_energy):
        raise ValueError("both inputs must be real")
    matrix = np.asarray(raw_matrix, dtype=float)
    energy = np.asarray(raw_energy, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or energy.ndim != 2
        or energy.shape != matrix.shape
    ):
        raise ValueError("both inputs must be nonempty square matrices of equal shape")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(energy)):
        raise ValueError("both inputs must be finite")
    symmetry_tolerance = 1e-12 * max(1.0, float(np.linalg.norm(energy, ord=np.inf)))
    if not np.allclose(energy, energy.T, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError("energy_matrix must be symmetric")
    try:
        lower = np.linalg.cholesky(energy)
    except np.linalg.LinAlgError as error:
        raise ValueError("energy_matrix must be positive definite") from error

    eigenvalues = np.linalg.eigvals(matrix)
    spectral_abscissa = float(np.max(eigenvalues.real))

    symmetric_part = 0.5 * (energy @ matrix + matrix.T @ energy)
    left_reduced = np.linalg.solve(lower, symmetric_part)
    reduced = np.linalg.solve(lower, left_reduced.T).T
    reduced = 0.5 * (reduced + reduced.T)
    instantaneous_growth = float(np.linalg.eigvalsh(reduced)[-1])
    result = [spectral_abscissa, instantaneous_growth]
    if resolvent_shifts is not None:
        raw_shifts = np.asarray(resolvent_shifts)
        if raw_shifts.ndim != 1 or raw_shifts.size == 0:
            raise ValueError("resolvent_shifts must be a nonempty 1D array")
        shifts = np.asarray(raw_shifts, dtype=complex)
        if not np.all(np.isfinite(shifts.real)) or not np.all(np.isfinite(shifts.imag)):
            raise ValueError("resolvent_shifts must be finite")
        identity = np.eye(matrix.shape[0])
        for shift in shifts:
            try:
                resolvent = np.linalg.solve(shift * identity - matrix, identity)
            except np.linalg.LinAlgError as error:
                raise ValueError("a requested shifted system is singular") from error
            left_scaled = lower.T @ resolvent
            metric_scaled = np.linalg.solve(lower, left_scaled.T).T
            result.append(float(np.linalg.svd(metric_scaled, compute_uv=False)[0]))
    result = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("stability diagnostics are not finite")
    return result

import numpy as np


def _validated_time_interval(dt, final_time):
    if not np.isfinite(dt) or float(dt) <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(final_time) or float(final_time) <= 0.0:
        raise ValueError("final_time must be positive and finite")
    steps = int(round(float(final_time) / float(dt)))
    tolerance = 1e-12 * max(1.0, abs(float(final_time)))
    if steps < 1 or abs(steps * float(dt) - float(final_time)) > tolerance:
        raise ValueError("final_time must be an integer multiple of dt")
    return float(dt), float(final_time), steps


def advance_weak_overset_rk4(
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    dt: float,
    final_time: float,
    beta: float = 80.0,
    x0: float = -0.4,
) -> np.ndarray:
    """Reference stage-updated classical RK4 integration."""
    dt, final_time, steps = _validated_time_interval(dt, final_time)
    if not np.isfinite(c) or float(c) == 0.0:
        raise ValueError("c must be finite and nonzero")
    if not np.isfinite(amplitude) or not np.isfinite(period) or float(period) <= 0.0:
        raise ValueError("amplitude must be finite and period must be positive")
    if abs(2.0 * np.pi * float(amplitude) / float(period)) >= abs(float(c)):
        raise ValueError("grid-speed magnitude must remain below abs(c)")
    if not np.isfinite(beta) or float(beta) <= 0.0 or not np.isfinite(x0):
        raise ValueError("beta must be positive and x0 must be finite")

    counts = np.asarray(counts)
    initial_state = compute_moving_grid_state(
        0.0, counts, sigmas, amplitude, period
    )
    state = np.exp(-float(beta) * (initial_state[1:] - float(x0)) ** 2)

    def right_hand_side(time, values):
        augmented = assemble_weak_overset_system(
            time,
            counts,
            sigmas,
            c,
            amplitude,
            period,
            penalties,
            beta,
            x0,
            jet_order=0,
        )[0]
        return augmented[:, :-1] @ values + augmented[:, -1]

    time = 0.0
    for _ in range(steps):
        k1 = right_hand_side(time, state)
        k2 = right_hand_side(time + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = right_hand_side(time + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = right_hand_side(time + dt, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += dt

    if not np.all(np.isfinite(state)):
        raise ValueError("time integration produced a non-finite state")
    return state

import numpy as np


def _validated_spectral_samples(spectral_samples):
    if isinstance(spectral_samples, (bool, np.bool_)) or not isinstance(
        spectral_samples, (int, np.integer)
    ):
        raise ValueError("spectral_samples must be an integer")
    if int(spectral_samples) < 2:
        raise ValueError("spectral_samples must be at least two")
    return int(spectral_samples)


def compute_moving_overset_error(
    counts: np.ndarray = None,
    sigmas: np.ndarray = None,
    c: float = 1.0,
    amplitude: float = 0.1,
    period: float = 1.0,
    penalties: np.ndarray = None,
    dt: float = 0.001,
    final_time: float = 1.0,
    beta: float = 80.0,
    x0: float = -0.4,
    spectral_samples: int = 41,
) -> float:
    """Reference end-to-end stability audit, integration, and norm reduction."""
    spectral_samples = _validated_spectral_samples(spectral_samples)
    if counts is None:
        counts = np.array([41, 49, 45])
    if sigmas is None:
        sigmas = np.array([0.25, -0.15, 0.20])
    if penalties is None:
        penalties = np.array([0.75, 0.75, 0.75])
    counts = np.asarray(counts)
    sigmas = np.asarray(sigmas, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    if not np.isfinite(c) or float(c) == 0.0:
        raise ValueError("c must be finite and nonzero")
    if not np.isfinite(amplitude) or not np.isfinite(period) or float(period) <= 0.0:
        raise ValueError("amplitude must be finite and period must be positive")
    if abs(2.0 * np.pi * float(amplitude) / float(period)) >= abs(float(c)):
        raise ValueError("grid-speed magnitude must remain below abs(c)")

    operators = [build_sbp63_operator(int(count)) for count in counts]
    starts = np.r_[0, np.cumsum(counts)]
    direction = 1 if float(c) > 0.0 else -1
    maximum_abscissa = -np.inf
    maximum_instantaneous_growth = -np.inf
    for time in np.linspace(0.0, float(final_time), spectral_samples):
        state = compute_moving_grid_state(
            time, counts, sigmas, amplitude, period
        )
        velocity = float(state[0])
        nodes = state[1:]
        metric = compute_relative_metric_diagonals(c, velocity, counts, nodes)
        augmented = assemble_weak_overset_system(
            time,
            counts,
            sigmas,
            c,
            amplitude,
            period,
            penalties,
            beta,
            x0,
        )
        system = augmented[0]
        if not np.all(np.isfinite(augmented[1:])):
            raise ValueError("the affine-system time derivatives are not finite")
        if direction > 0:
            first_weights = compute_donor_interpolation(
                nodes[starts[0] : starts[1]], nodes[starts[1]], 0
            )[0]
            second_weights = compute_donor_interpolation(
                nodes[starts[1] : starts[2]], nodes[starts[2]], 0
            )[0]
            first_row = starts[1]
            first_columns = slice(starts[0], starts[1])
            first_expected = (
                penalties[1] * metric[first_row] * first_weights / operators[1][0][0, 0]
            )
            second_row = starts[2]
            second_columns = slice(starts[1], starts[2])
            second_expected = (
                penalties[2]
                * metric[second_row]
                * second_weights
                / operators[2][0][0, 0]
            )
        else:
            first_weights = compute_donor_interpolation(
                nodes[starts[1] : starts[2]], nodes[starts[1] - 1], 0
            )[0]
            second_weights = compute_donor_interpolation(
                nodes[starts[2] : starts[3]], nodes[starts[2] - 1], 0
            )[0]
            first_row = starts[1] - 1
            first_columns = slice(starts[1], starts[2])
            first_expected = (
                penalties[0]
                * metric[first_row]
                * first_weights
                / operators[0][0][-1, -1]
            )
            second_row = starts[2] - 1
            second_columns = slice(starts[2], starts[3])
            second_expected = (
                penalties[1]
                * metric[second_row]
                * second_weights
                / operators[1][0][-1, -1]
            )
        if not np.allclose(
            system[first_row, first_columns],
            first_expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("first directional donor coupling is inconsistent")
        if not np.allclose(
            system[second_row, second_columns],
            second_expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("second directional donor coupling is inconsistent")
        energy_diagonal = np.empty(int(np.sum(counts)), dtype=float)
        for grid in range(3):
            section = slice(starts[grid], starts[grid + 1])
            energy_diagonal[section] = np.diag(operators[grid][0]) / metric[section]
        diagnostics = compute_weighted_stability_diagnostics(
            system[:, :-1], np.diag(energy_diagonal)
        )
        maximum_abscissa = max(maximum_abscissa, float(diagnostics[0]))
        maximum_instantaneous_growth = max(
            maximum_instantaneous_growth, float(diagnostics[1])
        )
    if not np.isfinite(maximum_abscissa) or maximum_abscissa >= 0.0:
        raise ValueError("a frozen system is not strictly stable")
    if not np.isfinite(maximum_instantaneous_growth):
        raise ValueError("the weighted instantaneous-growth audit is not finite")

    numerical = advance_weak_overset_rk4(
        counts,
        sigmas,
        c,
        amplitude,
        period,
        penalties,
        dt,
        final_time,
        beta,
        x0,
    )
    final_state = compute_moving_grid_state(
        final_time, counts, sigmas, amplitude, period
    )
    final_nodes = final_state[1:]
    exact = np.exp(
        -float(beta) * (final_nodes - float(x0) - float(c) * float(final_time)) ** 2
    )
    error = numerical - exact

    numerator = 0.0
    denominator = 0.0
    for grid in range(3):
        section = slice(starts[grid], starts[grid + 1])
        h_matrix, _, derivative = operators[grid][:3]
        jacobian = derivative @ final_nodes[section]
        weighted_norm = jacobian[:, None] * h_matrix
        numerator += float(error[section] @ weighted_norm @ error[section])
        denominator += float(exact[section] @ weighted_norm @ exact[section])
    if denominator <= 0.0:
        raise ValueError("exact-state norm must be positive")
    result = float(np.sqrt(numerator / denominator))
    if not np.isfinite(result):
        raise ValueError("relative error is not finite")
    return result
SCICODE_GOLD_EOF
