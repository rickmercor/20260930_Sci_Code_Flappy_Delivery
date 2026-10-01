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


# ORACLE SOLUTION


def evaluate_ellipse_field(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
) -> np.ndarray:
    """Reference evaluation of the quadratic field and derivatives."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    directions = np.asarray(translation_directions, dtype=np.float64)
    rotation_rates = np.asarray(rotation_rates, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if center.shape != (2,) or axes.shape != (2,):
        raise ValueError("center and axes must have shape (2,)")
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must have shape (p,) with p >= 1")
    if directions.shape != (state.size, 2) or rotation_rates.shape != state.shape:
        raise ValueError("motion arrays must have shapes (p, 2) and (p,)")
    if not all(
        np.all(np.isfinite(value))
        for value in (points, center, axes, state, directions, rotation_rates, [angle])
    ):
        raise ValueError("all inputs must be finite")
    if np.any(axes <= 0.0):
        raise ValueError("ellipse semiaxes must be positive")
    direction_norms = np.linalg.norm(directions, axis=1)
    valid_directions = (direction_norms <= 1e-14) | np.isclose(
        direction_norms, 1.0, rtol=0.0, atol=1e-10
    )
    if not np.all(valid_directions):
        raise ValueError("translation directions must be zero or unit length")

    deformed_angle = float(angle + state @ rotation_rates)
    deformed_center = center + state @ directions
    cosine = np.cos(deformed_angle)
    sine = np.sin(deformed_angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotation_rate = np.array([[-sine, -cosine], [cosine, -sine]])
    diagonal = np.diag(axes ** -2)
    metric = rotation @ diagonal @ rotation.T
    metric_rate = (
        rotation_rate @ diagonal @ rotation.T
        + rotation @ diagonal @ rotation_rate.T
    )
    metric_second = 2.0 * (
        rotation_rate @ diagonal @ rotation_rate.T - metric
    )
    offset = points - deformed_center
    gradient = 2.0 * offset @ metric
    field = np.einsum("ni,ij,nj->n", offset, metric, offset) - 1.0
    angle_derivative = np.einsum(
        "ni,ij,nj->n", offset, metric_rate, offset
    )
    state_derivative = (
        -(gradient @ directions.T)
        + angle_derivative[:, None] * rotation_rates[None, :]
    )
    direction_metric_direction = 2.0 * directions @ metric @ directions.T
    direction_metric_rate_offset = directions @ metric_rate @ offset.T
    angle_second_derivative = np.einsum(
        "ni,ij,nj->n", offset, metric_second, offset
    )
    state_hessian = np.empty(
        (points.shape[0], state.size, state.size), dtype=np.float64
    )
    for row in range(state.size):
        for column in range(state.size):
            state_hessian[:, row, column] = (
                direction_metric_direction[row, column]
                - 2.0
                * rotation_rates[column]
                * direction_metric_rate_offset[row]
                - 2.0
                * rotation_rates[row]
                * direction_metric_rate_offset[column]
                + rotation_rates[row]
                * rotation_rates[column]
                * angle_second_derivative
            )
    upper = np.triu_indices(state.size)
    packed_hessian = state_hessian[:, upper[0], upper[1]]
    return np.column_stack(
        [field, gradient, state_derivative, packed_hessian]
    ).astype(
        np.float64, copy=False
    )

import numpy as np


# ORACLE SOLUTION


def compute_fiber_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    discriminant_tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference quadratic-line intersection."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    directions = np.asarray(translation_directions, dtype=np.float64)
    rotation_rates = np.asarray(rotation_rates, dtype=np.float64)
    if (
        starts.ndim != 2
        or starts.shape[0] < 1
        or starts.shape[1] != 2
        or ends.shape != starts.shape
    ):
        raise ValueError("starts and ends must have matching shape (n_fibers, 2)")
    if center.shape != (2,) or axes.shape != (2,):
        raise ValueError("center and axes must have shape (2,)")
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must have shape (p,) with p >= 1")
    if directions.shape != (state.size, 2) or rotation_rates.shape != state.shape:
        raise ValueError("motion arrays must have shapes (p, 2) and (p,)")
    numeric = (
        starts, ends, center, axes, state, directions, rotation_rates,
        [angle, discriminant_tolerance],
    )
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("all inputs must be finite")
    if np.any(axes <= 0.0) or discriminant_tolerance <= 0.0:
        raise ValueError("axes and discriminant_tolerance must be positive")
    segments = ends - starts
    if np.any(np.linalg.norm(segments, axis=1) <= 0.0):
        raise ValueError("fibers must have positive length")
    direction_norms = np.linalg.norm(directions, axis=1)
    valid_directions = (direction_norms <= 1e-14) | np.isclose(
        direction_norms, 1.0, rtol=0.0, atol=1e-10
    )
    if not np.all(valid_directions):
        raise ValueError("translation directions must be zero or unit length")

    deformed_angle = float(angle + state @ rotation_rates)
    deformed_center = center + state @ directions
    cosine = np.cos(deformed_angle)
    sine = np.sin(deformed_angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    metric = rotation @ np.diag(axes ** -2) @ rotation.T
    offsets = starts - deformed_center
    quadratic = np.einsum("ni,ij,nj->n", segments, metric, segments)
    linear = 2.0 * np.einsum("ni,ij,nj->n", offsets, metric, segments)
    constant = np.einsum("ni,ij,nj->n", offsets, metric, offsets) - 1.0
    direct_discriminant = linear**2 - 4.0 * quadratic * constant

    normalized_offsets = (offsets @ rotation) / axes
    normalized_segments = (segments @ rotation) / axes
    closest_parameter = -np.einsum(
        "ni,ni->n", normalized_offsets, normalized_segments
    ) / quadratic
    closest_offsets = (
        normalized_offsets
        + closest_parameter[:, None] * normalized_segments
    )
    radial_margin = 1.0 - np.einsum(
        "ni,ni->n", closest_offsets, closest_offsets
    )
    geometric_discriminant = 4.0 * quadratic * radial_margin
    cancellation_scale = linear**2 + np.abs(4.0 * quadratic * constant)
    use_geometric = np.abs(direct_discriminant) <= (
        16.0 * np.finfo(np.float64).eps * cancellation_scale
    )
    discriminant = np.where(
        use_geometric, geometric_discriminant, direct_discriminant
    )
    roots = np.full((starts.shape[0], 2), np.nan, dtype=np.float64)
    crossing = discriminant >= -float(discriminant_tolerance)
    direct_crossing = crossing & ~use_geometric
    square_root = np.sqrt(np.maximum(discriminant[direct_crossing], 0.0))
    denominator = 2.0 * quadratic[direct_crossing]
    roots[direct_crossing, 0] = (
        -linear[direct_crossing] - square_root
    ) / denominator
    roots[direct_crossing, 1] = (
        -linear[direct_crossing] + square_root
    ) / denominator
    geometric_crossing = crossing & use_geometric
    half_span = np.sqrt(
        np.maximum(radial_margin[geometric_crossing], 0.0)
        / quadratic[geometric_crossing]
    )
    roots[geometric_crossing, 0] = (
        closest_parameter[geometric_crossing] - half_span
    )
    roots[geometric_crossing, 1] = (
        closest_parameter[geometric_crossing] + half_span
    )
    return roots

import numpy as np


# ORACLE SOLUTION


def intersect_inside_intervals(
    root_sets: np.ndarray, tie_tolerance: float = 1e-12
) -> np.ndarray:
    """Reference interval intersection with unique active owners."""
    roots = np.asarray(root_sets, dtype=np.float64)
    if roots.ndim != 3 or roots.shape[0] < 2 or roots.shape[1] < 1 or roots.shape[2] != 2:
        raise ValueError("root_sets must have shape (n_bodies, n_fibers, 2)")
    if not np.isfinite(tie_tolerance) or tie_tolerance <= 0.0:
        raise ValueError("tie_tolerance must be positive and finite")
    paired_nan = np.isnan(roots[..., 0]) & np.isnan(roots[..., 1])
    paired_finite = np.isfinite(roots[..., 0]) & np.isfinite(roots[..., 1])
    if not np.all(paired_nan | paired_finite):
        raise ValueError("each root row must be paired finite values or paired NaNs")
    if np.any(paired_finite & (roots[..., 0] > roots[..., 1])):
        raise ValueError("finite roots must be ascending")

    result = np.full((roots.shape[1], 4), np.nan, dtype=np.float64)
    for fiber in range(roots.shape[1]):
        if np.any(np.isnan(roots[:, fiber])):
            continue
        lower_candidates = np.concatenate(([0.0], roots[:, fiber, 0]))
        upper_candidates = np.concatenate(([1.0], roots[:, fiber, 1]))
        lower = float(np.max(lower_candidates))
        upper = float(np.min(upper_candidates))
        if upper <= lower + tie_tolerance:
            continue
        lower_matches = np.flatnonzero(
            np.abs(lower_candidates - lower) <= tie_tolerance
        )
        upper_matches = np.flatnonzero(
            np.abs(upper_candidates - upper) <= tie_tolerance
        )
        if lower_matches.size != 1 or upper_matches.size != 1:
            raise ValueError("active contact endpoint has a nondifferentiable tie")
        result[fiber] = [lower, upper, lower_matches[0], upper_matches[0]]
    return result

import numpy as np


# ORACLE SOLUTION


def estimate_overlap_area(
    interval_state: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Reference weighted line-length quadrature."""
    state = np.asarray(interval_state, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != 4:
        raise ValueError("interval_state must have shape (n_fibers, 4)")
    if starts.shape != (state.shape[0], 2) or ends.shape != starts.shape:
        raise ValueError("starts and ends must match the number of intervals")
    if weights.shape != (state.shape[0],):
        raise ValueError("weights must have shape (n_fibers,)")
    if not all(np.all(np.isfinite(value)) for value in (starts, ends, weights)):
        raise ValueError("fiber data and weights must be finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be nonnegative with at least one positive value")
    segments = ends - starts
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    if np.any(lengths <= 0.0):
        raise ValueError("fibers must have positive length")
    active = np.isfinite(state[:, 0])
    if np.any(active & ~np.all(np.isfinite(state), axis=1)):
        raise ValueError("active interval rows must be finite")
    if np.any((~active) & ~np.all(np.isnan(state), axis=1)):
        raise ValueError("inactive interval rows must contain four NaNs")
    if np.any(active & ((state[:, 0] < 0.0) | (state[:, 1] > 1.0))):
        raise ValueError("active bounds must lie in [0, 1]")
    if np.any(active & (state[:, 1] <= state[:, 0])):
        raise ValueError("active intervals must have positive length")
    fractions = np.zeros(state.shape[0], dtype=np.float64)
    fractions[active] = state[active, 1] - state[active, 0]
    area = float(np.sum(weights * lengths * fractions))
    if not np.isfinite(area) or area < 0.0:
        raise ValueError("overlap estimate must be finite and nonnegative")
    return area

import numpy as np


# ORACLE SOLUTION


def differentiate_boundary_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    roots: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    transversality_tolerance: float = 1e-10,
) -> np.ndarray:
    """Reference second-order implicit differentiation of boundary roots."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    roots = np.asarray(roots, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    directions = np.asarray(translation_directions, dtype=np.float64)
    rotation_rates = np.asarray(rotation_rates, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[0] < 1 or starts.shape[1] != 2:
        raise ValueError("starts must have shape (n_fibers, 2)")
    if ends.shape != starts.shape or roots.shape != starts.shape:
        raise ValueError("ends and roots must match starts")
    if center.shape != (2,) or axes.shape != (2,):
        raise ValueError("center and axes must have shape (2,)")
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must have shape (p,) with p >= 1")
    if directions.shape != (state.size, 2) or rotation_rates.shape != state.shape:
        raise ValueError("motion arrays must have shapes (p, 2) and (p,)")
    numeric = (
        starts,
        ends,
        center,
        axes,
        state,
        directions,
        rotation_rates,
        [angle, transversality_tolerance],
    )
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("finite geometry and parameters are required")
    if np.any(axes <= 0.0) or transversality_tolerance <= 0.0:
        raise ValueError("axes and transversality_tolerance must be positive")
    segments = ends - starts
    if np.any(np.linalg.norm(segments, axis=1) <= 0.0):
        raise ValueError("fibers must have positive length")
    paired_nan = np.isnan(roots[:, 0]) & np.isnan(roots[:, 1])
    paired_finite = np.isfinite(roots[:, 0]) & np.isfinite(roots[:, 1])
    if not np.all(paired_nan | paired_finite):
        raise ValueError("root rows must be paired finite values or paired NaNs")
    direction_norms = np.linalg.norm(directions, axis=1)
    valid_directions = (direction_norms <= 1e-14) | np.isclose(
        direction_norms, 1.0, rtol=0.0, atol=1e-10
    )
    if not np.all(valid_directions):
        raise ValueError("translation directions must be zero or unit length")

    deformed_angle = float(angle + state @ rotation_rates)
    cosine = np.cos(deformed_angle)
    sine = np.sin(deformed_angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotation_rate = np.array([[-sine, -cosine], [cosine, -sine]])
    diagonal = np.diag(axes ** -2)
    metric = rotation @ diagonal @ rotation.T
    metric_rate = (
        rotation_rate @ diagonal @ rotation.T
        + rotation @ diagonal @ rotation_rate.T
    )
    metric_second = 2.0 * (
        rotation_rate @ diagonal @ rotation_rate.T - metric
    )
    deformed_center = center + state @ directions
    sensitivities = np.full(
        roots.shape + (state.size + 1, state.size), np.nan, dtype=np.float64
    )
    for column in range(2):
        finite = np.isfinite(roots[:, column])
        boundary = starts[finite] + roots[finite, column, None] * segments[finite]
        offset = boundary - deformed_center
        gradient = 2.0 * offset @ metric
        angle_derivative = np.einsum(
            "ni,ij,nj->n", offset, metric_rate, offset
        )
        state_derivative = (
            -(gradient @ directions.T)
            + angle_derivative[:, None] * rotation_rates[None, :]
        )
        denominator = np.einsum("ni,ni->n", gradient, segments[finite])
        if np.any(np.abs(denominator) < transversality_tolerance):
            raise ValueError("boundary root is not transversal")
        root_gradient = -state_derivative / denominator[:, None]
        sensitivities[finite, column, 0, :] = root_gradient

        g_hh = 2.0 * np.einsum(
            "ni,ij,nj->n", segments[finite], metric, segments[finite]
        )
        q_metric_rate_x = np.einsum(
            "ni,ij,nj->n", segments[finite], metric_rate, offset
        )
        q_metric_directions = segments[finite] @ metric @ directions.T
        g_hi = 2.0 * (
            q_metric_rate_x[:, None] * rotation_rates[None, :]
            - q_metric_directions
        )
        direction_metric_direction = 2.0 * directions @ metric @ directions.T
        direction_metric_rate_offset = directions @ metric_rate @ offset.T
        angle_second_derivative = np.einsum(
            "ni,ij,nj->n", offset, metric_second, offset
        )
        partial_hessian = np.empty(
            (offset.shape[0], state.size, state.size), dtype=np.float64
        )
        for row in range(state.size):
            for state_column in range(state.size):
                partial_hessian[:, row, state_column] = (
                    direction_metric_direction[row, state_column]
                    - 2.0
                    * rotation_rates[state_column]
                    * direction_metric_rate_offset[row]
                    - 2.0
                    * rotation_rates[row]
                    * direction_metric_rate_offset[state_column]
                    + rotation_rates[row]
                    * rotation_rates[state_column]
                    * angle_second_derivative
                )
        root_hessian = np.empty_like(partial_hessian)
        for row in range(state.size):
            for state_column in range(state.size):
                numerator = (
                    partial_hessian[:, row, state_column]
                    + g_hi[:, row] * root_gradient[:, state_column]
                    + g_hi[:, state_column] * root_gradient[:, row]
                    + g_hh
                    * root_gradient[:, row]
                    * root_gradient[:, state_column]
                )
                root_hessian[:, row, state_column] = -numerator / denominator
        sensitivities[finite, column, 1:, :] = root_hessian
    return sensitivities

import math
import numpy as np


# ORACLE SOLUTION


def differentiate_overlap_area(
    interval_state: np.ndarray,
    root_sensitivities: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Reference endpoint-owner accumulation."""
    state = np.asarray(interval_state, dtype=np.float64)
    derivatives = np.asarray(root_sensitivities, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != 4:
        raise ValueError("interval_state must have shape (n_fibers, 4)")
    if (
        derivatives.ndim != 5
        or derivatives.shape[0] < 2
        or derivatives.shape[1:3] != (state.shape[0], 2)
        or derivatives.shape[4] < 1
        or derivatives.shape[3] != derivatives.shape[4] + 1
    ):
        raise ValueError(
            "root_sensitivities must have shape "
            "(n_bodies, n_fibers, 2, p + 1, p)"
        )
    expected_roots = (state.shape[0], 2)
    if starts.shape != expected_roots or ends.shape != expected_roots:
        raise ValueError("fiber endpoints must have shape (n_fibers, 2)")
    if weights.shape != (state.shape[0],):
        raise ValueError("weights must have shape (n_fibers,)")
    if not all(np.all(np.isfinite(value)) for value in (starts, ends, weights)):
        raise ValueError("fiber data and weights must be finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be nonnegative with at least one positive value")
    segments = ends - starts
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    if np.any(lengths <= 0.0):
        raise ValueError("fibers must have positive length")

    active = np.isfinite(state[:, 0])
    if np.any(active & ~np.all(np.isfinite(state), axis=1)):
        raise ValueError("active interval rows must be finite")
    if np.any((~active) & ~np.all(np.isnan(state), axis=1)):
        raise ValueError("inactive interval rows must contain four NaNs")
    owners = state[:, 2:4]
    valid_owner_values = np.arange(derivatives.shape[0] + 1, dtype=np.float64)
    if np.any(active & ~np.all(np.isin(owners, valid_owner_values), axis=1)):
        raise ValueError("endpoint owner is outside the available body range")

    derivative_shape = (state.shape[0], derivatives.shape[3], derivatives.shape[4])
    lower_derivative = np.zeros(derivative_shape, dtype=np.float64)
    upper_derivative = np.zeros_like(lower_derivative)
    for fiber in np.flatnonzero(active):
        lower_owner = int(state[fiber, 2])
        upper_owner = int(state[fiber, 3])
        if lower_owner > 0:
            lower_derivative[fiber] = derivatives[lower_owner - 1, fiber, 0]
        if upper_owner > 0:
            upper_derivative[fiber] = derivatives[upper_owner - 1, fiber, 1]
        if not np.all(np.isfinite(lower_derivative[fiber])) or not np.all(
            np.isfinite(upper_derivative[fiber])
        ):
            raise ValueError("an active endpoint has no finite derivative")
        for endpoint in (lower_derivative[fiber], upper_derivative[fiber]):
            if not np.allclose(
                endpoint[1:], endpoint[1:].T, rtol=1e-10, atol=1e-12
            ):
                raise ValueError("an active root Hessian is not symmetric")
    contributions = (
        weights[:, None, None] * lengths[:, None, None]
        * (upper_derivative - lower_derivative)
    )
    derivative = np.empty(contributions.shape[1:], dtype=np.float64)
    for row in range(derivative.shape[0]):
        for column in range(derivative.shape[1]):
            derivative[row, column] = math.fsum(
                contributions[:, row, column].tolist()
            )
    if not np.all(np.isfinite(derivative)):
        raise ValueError("overlap derivative must be finite")
    return derivative

import numpy as np

def compute_contact_energy_force(
    overlap_area: float,
    overlap_sensitivities: np.ndarray,
    stiffness: float,
    exponent: float = 1.0,
) -> np.ndarray:
    """Reference energy and negative-gradient force."""
    sensitivities = np.asarray(overlap_sensitivities, dtype=np.float64)
    values = np.asarray([overlap_area, stiffness, exponent], dtype=np.float64)
    if (
        sensitivities.ndim != 2
        or sensitivities.shape[1] < 1
        or sensitivities.shape[0] != sensitivities.shape[1] + 1
    ):
        raise ValueError("overlap_sensitivities must have shape (p + 1, p)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sensitivities)):
        raise ValueError("all inputs must be finite")
    gradient = sensitivities[0]
    hessian = sensitivities[1:]
    if not np.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
        raise ValueError("overlap Hessian must be symmetric")
    if overlap_area < 0.0:
        raise ValueError("overlap_area must be nonnegative")
    if stiffness <= 0.0 or exponent < 1.0:
        raise ValueError("stiffness must be positive and exponent must be at least one")
    if overlap_area == 0.0:
        return np.zeros((gradient.size + 1, gradient.size + 1), dtype=np.float64)
    log_area = np.log(overlap_area)
    log_energy = np.log(stiffness) + exponent * log_area
    log_max = np.log(np.finfo(np.float64).max)
    if log_energy > log_max:
        raise ValueError("energy and force must be finite")
    energy = float(np.exp(log_energy))
    log_first = (
        np.log(stiffness) + np.log(exponent) + (exponent - 1.0) * log_area
    )
    if log_first > log_max:
        raise ValueError("energy and force must be finite")
    first_factor = float(np.exp(log_first))
    if exponent == 1.0:
        second_factor = 0.0
    else:
        log_second = (
            np.log(stiffness)
            + np.log(exponent)
            + np.log(exponent - 1.0)
            + (exponent - 2.0) * log_area
        )
        if log_second > log_max:
            raise ValueError("force Jacobian must be finite")
        second_factor = float(np.exp(log_second))
    force = -first_factor * gradient
    force_jacobian = -(
        second_factor * np.outer(gradient, gradient) + first_factor * hessian
    )
    result = np.empty((gradient.size + 1, gradient.size + 1), dtype=np.float64)
    result[0, 0] = energy
    result[0, 1:] = force
    result[1:, 0] = force
    result[1:, 1:] = force_jacobian
    if not np.all(np.isfinite(result)):
        raise ValueError("energy and force must be finite")
    return result

import numpy as np

def _axis_fibers(box, n_horizontal, n_vertical):
    """Construct midpoint horizontal fibers followed by vertical fibers."""
    x_min, x_max, y_min, y_max = np.asarray(box, dtype=np.float64)
    y_values = y_min + (np.arange(n_horizontal) + 0.5) * (
        y_max - y_min
    ) / n_horizontal
    x_values = x_min + (np.arange(n_vertical) + 0.5) * (
        x_max - x_min
    ) / n_vertical
    starts = np.vstack(
        [
            np.column_stack([np.full(n_horizontal, x_min), y_values]),
            np.column_stack([x_values, np.full(n_vertical, y_min)]),
        ]
    )
    ends = np.vstack(
        [
            np.column_stack([np.full(n_horizontal, x_max), y_values]),
            np.column_stack([x_values, np.full(n_vertical, y_max)]),
        ]
    )
    weights = np.concatenate(
        [
            np.full(n_horizontal, (y_max - y_min) / (2.0 * n_horizontal)),
            np.full(n_vertical, (x_max - x_min) / (2.0 * n_vertical)),
        ]
    )
    return starts, ends, weights


def _experiment_data():
    """Return the three immutable experiment dictionaries."""
    return [
        {
            "box": (-1.4, 1.4, -1.1, 1.1),
            "counts": (64, 64),
            "center1": (-0.22, 0.0),
            "axes1": (0.82, 0.55),
            "angle1": 0.20,
            "center2": (0.38, 0.07),
            "axes2": (0.70, 0.46),
            "angle2": -0.32,
            "direction": (1.0, 0.0),
            "theta": 0.0,
            "velocity": -0.35,
            "mass": 1.7,
            "inertia": 0.18,
            "omega": 0.15,
            "stiffness": 480.0,
            "dt": 0.002,
            "steps": 50,
        },
        {
            "box": (-1.6, 1.6, -1.2, 1.2),
            "counts": (48, 48),
            "center1": (-0.30, -0.04),
            "axes1": (0.94, 0.48),
            "angle1": -0.28,
            "center2": (0.46, 0.12),
            "axes2": (0.63, 0.51),
            "angle2": 0.41,
            "direction": (0.9659258263, 0.2588190451),
            "theta": 0.0,
            "velocity": -0.29,
            "mass": 2.1,
            "inertia": 0.21,
            "omega": -0.12,
            "stiffness": 620.0,
            "dt": 0.0015,
            "steps": 60,
        },
        {
            "box": (-1.2, 1.2, -1.2, 1.2),
            "counts": (80, 80),
            "center1": (-0.18, 0.0),
            "axes1": (0.68, 0.52),
            "angle1": 0.15,
            "center2": (0.69, 0.03),
            "axes2": (0.44, 0.36),
            "angle2": -0.20,
            "direction": (1.0, 0.0),
            "theta": 0.0,
            "velocity": -0.22,
            "mass": 1.3,
            "inertia": 0.095,
            "omega": 0.08,
            "stiffness": 700.0,
            "dt": 0.001,
            "steps": 80,
        },
    ]


def _force_at_state(
    experiment, theta, starts, ends, weights, force_scale, pipeline
):
    """Compose all geometric and energetic steps at one state."""
    (
        field_function,
        root_function,
        interval_function,
        area_function,
        root_derivative_function,
        area_derivative_function,
        energy_force_function,
    ) = pipeline
    fixed_state = np.zeros(2, dtype=np.float64)
    fixed_directions = np.zeros((2, 2), dtype=np.float64)
    rotation_rates = np.zeros(2, dtype=np.float64)
    direction = np.asarray(experiment["direction"], dtype=np.float64)
    moving_state = np.asarray(theta, dtype=np.float64)
    moving_directions = np.vstack([direction, np.zeros(2)])
    moving_rotation_rates = np.array([0.0, 1.0])
    roots1 = root_function(
        starts,
        ends,
        experiment["center1"],
        experiment["axes1"],
        experiment["angle1"],
        fixed_state,
        fixed_directions,
        rotation_rates,
    )
    roots2 = root_function(
        starts,
        ends,
        experiment["center2"],
        experiment["axes2"],
        experiment["angle2"],
        moving_state,
        moving_directions,
        moving_rotation_rates,
    )
    intervals = interval_function(np.stack([roots1, roots2]))
    overlap = area_function(intervals, starts, ends, weights)

    finite1 = np.isfinite(roots1)
    finite2 = np.isfinite(roots2)
    segments = ends - starts
    points1 = starts[:, None, :] + roots1[:, :, None] * segments[:, None, :]
    points2 = starts[:, None, :] + roots2[:, :, None] * segments[:, None, :]
    values1 = field_function(
        points1[finite1],
        experiment["center1"],
        experiment["axes1"],
        experiment["angle1"],
        fixed_state,
        fixed_directions,
        rotation_rates,
    )[:, 0]
    values2 = field_function(
        points2[finite2],
        experiment["center2"],
        experiment["axes2"],
        experiment["angle2"],
        moving_state,
        moving_directions,
        moving_rotation_rates,
    )[:, 0]
    if np.any(np.abs(values1) > 1e-9) or np.any(np.abs(values2) > 1e-9):
        raise ValueError("computed roots do not satisfy the body fields")

    derivatives1 = root_derivative_function(
        starts,
        ends,
        roots1,
        experiment["center1"],
        experiment["axes1"],
        experiment["angle1"],
        fixed_state,
        fixed_directions,
        rotation_rates,
    )
    derivatives2 = root_derivative_function(
        starts,
        ends,
        roots2,
        experiment["center2"],
        experiment["axes2"],
        experiment["angle2"],
        moving_state,
        moving_directions,
        moving_rotation_rates,
    )
    overlap_sensitivities = area_derivative_function(
        intervals, np.stack([derivatives1, derivatives2]), starts, ends, weights
    )
    energy_force = np.asarray(energy_force_function(
        overlap, overlap_sensitivities,
        force_scale * experiment["stiffness"], 1.0,
    ), dtype=np.float64)
    if energy_force.shape != (3, 3) or not np.all(np.isfinite(energy_force)):
        raise ValueError("energy-force result must have shape (3, 3)")
    return energy_force[0, 1:].copy(), energy_force[1:, 1:].copy()


def run_contact_impulse_audit(force_scale: float = 1.0) -> float:
    """Reference end-to-end three-experiment computation."""
    if not np.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError("force_scale must be positive and finite")
    pipeline = (
        evaluate_ellipse_field,  # noqa: F821
        compute_fiber_roots,  # noqa: F821
        intersect_inside_intervals,  # noqa: F821
        estimate_overlap_area,  # noqa: F821
        differentiate_boundary_roots,  # noqa: F821
        differentiate_overlap_area,  # noqa: F821
        compute_contact_energy_force,  # noqa: F821
    )
    total_sensitivity = 0.0
    for experiment in _experiment_data():
        starts, ends, weights = _axis_fibers(
            experiment["box"], *experiment["counts"]
        )
        theta = np.array([experiment["theta"], 0.0], dtype=np.float64)
        initial_velocity = np.array([experiment["velocity"], experiment["omega"]], dtype=np.float64)
        velocity = initial_velocity.copy()
        position_tangent = np.zeros(2)
        velocity_tangent = np.zeros(2)
        mass = np.array([experiment["mass"], experiment["inertia"]])
        time_step = float(experiment["dt"])
        for _ in range(int(experiment["steps"])):
            old_force, old_jacobian = _force_at_state(
                experiment,
                theta,
                starts,
                ends,
                weights,
                float(force_scale),
                pipeline,
            )
            half_velocity_tangent = velocity_tangent + 0.5 * time_step * (
                old_force + old_jacobian @ position_tangent
            ) / mass
            half_velocity = velocity + 0.5 * time_step * old_force / mass
            position_tangent += time_step * half_velocity_tangent
            theta = theta + time_step * half_velocity
            new_force, new_jacobian = _force_at_state(
                experiment,
                theta,
                starts,
                ends,
                weights,
                float(force_scale),
                pipeline,
            )
            velocity_tangent = half_velocity_tangent + 0.5 * time_step * (
                new_force + new_jacobian @ position_tangent
            ) / mass
            velocity = half_velocity + 0.5 * time_step * new_force / mass
        momentum = mass * (velocity - initial_velocity)
        momentum_norm = np.linalg.norm(momentum)
        if momentum_norm == 0.0:
            raise ValueError("absolute impulse is not differentiable at zero")
        total_sensitivity += float(momentum @ (mass * velocity_tangent) / momentum_norm)
    if not np.isfinite(total_sensitivity):
        raise ValueError("audit scalar must be finite")
    return float(total_sensitivity)
SCICODE_GOLD_EOF
