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


def _finite_array(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numerical array") from exc
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} has invalid shape or non-finite entries")
    return result


def _finite_scalar(value, name):
    result = _finite_array(value, name, ())
    return float(result)


def _positive_scalar(value, name):
    result = _finite_scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def build_dual_partition(knots: np.ndarray, partition: float) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    knots = _finite_array(knots, "knots")
    partition = _finite_scalar(partition, "partition")
    if knots.ndim != 1 or knots.size < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be a strictly increasing vector")
    if not 0 < partition < 0.5:
        raise ValueError("partition must lie in (0, 0.5)")
    nodes = np.empty(2 * knots.size - 1)
    nodes[::2] = knots
    nodes[1::2] = (knots[:-1] + knots[1:]) / 2
    cuts = np.empty(2 * knots.size)
    cuts[0], cuts[-1] = knots[0], knots[-1]
    cuts[1:-1:2] = knots[:-1] + partition * np.diff(knots)
    cuts[2:-1:2] = knots[:-1] + (1 - partition) * np.diff(knots)
    return np.column_stack((nodes, cuts[:-1], cuts[1:]))

import numpy as np
from scipy.integrate import quad


def _decaying_ratio(fraction, exponent):
    if abs(exponent) < 1e-14:
        return fraction
    if exponent > 0:
        return (
            np.exp((fraction - 1) * exponent)
            * (-np.expm1(-fraction * exponent))
            / (-np.expm1(-exponent))
        )
    return np.expm1(fraction * exponent) / np.expm1(exponent)


def _trace_weight(exponent):
    if abs(exponent) < 1e-5:
        return 1 - exponent / 2 + exponent**2 / 12 - exponent**4 / 720
    if exponent > 0:
        return exponent * np.exp(-exponent) / (-np.expm1(-exponent))
    return exponent / np.expm1(exponent)


def compute_normal_kernel(
    half_width: float, diffusion: float, normal_drift: float
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    ell = _positive_scalar(half_width, "half_width")
    alpha = _positive_scalar(diffusion, "diffusion")
    beta = _finite_scalar(normal_drift, "normal_drift")
    z = 2 * ell * beta / alpha
    if not np.isfinite(z) or abs(z) > 1000:
        raise ValueError("normal Peclet number must satisfy abs(z) <= 1000")
    moments = []
    for degree in range(3):
        left = quad(
            lambda t: -_decaying_ratio((t + 1) / 2, -z) * t**degree,
            -1,
            0,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        right = quad(
            lambda t: _decaying_ratio((1 - t) / 2, z) * t**degree,
            0,
            1,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        moment = left + right
        for _ in range(degree + 1):
            moment *= ell
        moments.append(moment)
    result = np.array([_trace_weight(z), _trace_weight(-z), *moments])
    if not np.all(np.isfinite(result)):
        raise ValueError("kernel moments exceed the finite numerical range")
    return result

import numpy as np


def _rectangle(bounds):
    bounds = _finite_array(bounds, "bounds", (4,))
    if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
        raise ValueError("rectangle widths must be positive")
    return bounds


def _oriented_frame(axis):
    if not isinstance(axis, (int, np.integer)) or axis not in (0, 1):
        raise ValueError("axis must be zero or one")
    return (
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        if axis == 0
        else (np.array([0.0, 1.0]), np.array([-1.0, 0.0]))
    )


def _mapped_lagrange(offset, slope):
    base = np.array([[1.0, -3.0, 2.0], [0.0, 4.0, -4.0], [0.0, -1.0, 2.0]])
    return np.column_stack(
        (
            base[:, 0] + base[:, 1] * offset + base[:, 2] * offset**2,
            slope * (base[:, 1] + 2 * base[:, 2] * offset),
            base[:, 2] * slope**2,
        )
    )


def transform_quadratic_traces(
    bounds: np.ndarray, center: np.ndarray, axis: int
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    bounds = _rectangle(bounds)
    center = _finite_array(center, "center", (2,))
    _oriented_frame(axis)
    x0, x1, y0, y1 = bounds
    if not x0 <= center[0] <= x1 or not y0 <= center[1] <= y1:
        raise ValueError("center must lie in the rectangle")
    lx = _mapped_lagrange(
        (center[0] - x0) / (x1 - x0), (1 if axis == 0 else -1) / (x1 - x0)
    )
    ly = _mapped_lagrange((center[1] - y0) / (y1 - y0), 1 / (y1 - y0))
    result = np.empty((9, 3, 3))
    for j in range(3):
        for i in range(3):
            result[3 * j + i] = (
                np.outer(lx[i], ly[j]) if axis == 0 else np.outer(ly[j], lx[i])
            )
    return result

import numpy as np


def _face_inputs(coefficients, half_width, face_length, diffusion, kernel):
    coefficients = _finite_array(coefficients, "coefficients", (9, 3, 3))
    ell = _positive_scalar(half_width, "half_width")
    length = _positive_scalar(face_length, "face_length")
    alpha = _positive_scalar(diffusion, "diffusion")
    kernel = _finite_array(kernel, "kernel", (5,))
    if np.any(kernel[:2] < 0):
        raise ValueError("trace weights must be nonnegative")
    return coefficients, ell, length, alpha, kernel


def compute_homogeneous_face(
    coefficients: np.ndarray,
    half_width: float,
    face_length: float,
    diffusion: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    coefficients, ell, length, alpha, kernel = _face_inputs(
        coefficients, half_width, face_length, diffusion, kernel
    )
    tangent_integrals = np.array([length, 0.0, length**3 / 12])
    left = np.einsum(
        "kab,a,b->k", coefficients, np.array([1.0, -ell, ell**2]), tangent_integrals
    )
    right = np.einsum(
        "kab,a,b->k", coefficients, np.array([1.0, ell, ell**2]), tangent_integrals
    )
    return alpha / (2 * ell) * (kernel[0] * right - kernel[1] * left)

import numpy as np


def compute_transverse_face(
    coefficients: np.ndarray,
    face_length: float,
    diffusion: float,
    tangent_drift: float,
    moments: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    coefficients = _finite_array(coefficients, "coefficients", (9, 3, 3))
    length = _positive_scalar(face_length, "face_length")
    alpha = _positive_scalar(diffusion, "diffusion")
    beta = _finite_scalar(tangent_drift, "tangent_drift")
    moments = _finite_array(moments, "moments", (3,))
    return length * (
        (2 * alpha * coefficients[:, :, 2] - beta * coefficients[:, :, 1]) @ moments
    )

import numpy as np


def compute_source_face(
    source: np.ndarray,
    center: np.ndarray,
    axis: int,
    face_length: float,
    moments: np.ndarray,
) -> float:
    """Evaluate the reference numerical operation."""
    source = _finite_array(source, "source", (4,))
    center = _finite_array(center, "center", (2,))
    normal, _ = _oriented_frame(axis)
    length = _positive_scalar(face_length, "face_length")
    moments = _finite_array(moments, "moments", (3,))
    f0, fx, fy, fxy = source
    x, y = center
    value = f0 + fx * x + fy * y + fxy * x * y
    derivative = np.dot(normal, [fx + fxy * y, fy + fxy * x])
    return float(length * (value * moments[0] + derivative * moments[1]))

import numpy as np


def _transport_data(
    x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
):
    x = build_dual_partition(x_knots, partition)
    y = build_dual_partition(y_knots, partition)
    alpha = _positive_scalar(diffusion, "diffusion")
    drift = _finite_array(drift, "drift", (2,))
    source = _finite_array(source, "source", (4,))
    boundary = _finite_array(boundary, "boundary", (3,))
    fraction = _positive_scalar(trace_fraction, "trace_fraction")
    if fraction >= partition:
        raise ValueError("trace_fraction must be smaller than partition")
    return x, y, alpha, drift, source, boundary, fraction


def _assemble_carrier_balance(
    x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
):
    x, y, alpha, drift, source, boundary, fraction = _transport_data(
        x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
    )
    nx, ny = len(x), len(y)
    matrix = np.zeros((nx * ny, nx * ny))
    load = np.empty(nx * ny)
    for j in range(ny):
        for i in range(nx):
            xm, ym = (x[i, 1] + x[i, 2]) / 2, (y[j, 1] + y[j, 2]) / 2
            area = (x[i, 2] - x[i, 1]) * (y[j, 2] - y[j, 1])
            load[j * nx + i] = area * (
                source[0] + source[1] * xm + source[2] * ym + source[3] * xm * ym
            )
    fractions = np.array([0.0, partition, 1 - partition, 1.0])
    for ey in range((ny - 1) // 2):
        for ex in range((nx - 1) // 2):
            bounds = np.array(
                [x[2 * ex, 0], x[2 * ex + 2, 0], y[2 * ey, 0], y[2 * ey + 2, 0]]
            )
            widths = bounds[[1, 3]] - bounds[[0, 2]]
            lower = bounds[[0, 2]]
            indices = np.array(
                [(2 * ey + j) * nx + 2 * ex + i for j in range(3) for i in range(3)]
            )
            for axis in (0, 1):
                normal, tangent = _oriented_frame(axis)
                ell = fraction * widths[axis]
                kernel = compute_normal_kernel(ell, alpha, drift[axis])
                cuts = lower[axis] + widths[axis] * fractions
                other = 1 - axis
                ends = lower[other] + widths[other] * fractions
                for cut in (1, 2):
                    for segment in range(3):
                        center = np.empty(2)
                        center[axis] = cuts[cut]
                        center[other] = (ends[segment] + ends[segment + 1]) / 2
                        length = ends[segment + 1] - ends[segment]
                        coefficients = transform_quadratic_traces(
                            bounds, center, axis
                        )
                        flux = compute_homogeneous_face(
                            coefficients, ell, length, alpha, kernel
                        )
                        flux += compute_transverse_face(
                            coefficients,
                            length,
                            alpha,
                            float(drift @ tangent),
                            kernel[2:],
                        )
                        correction = compute_source_face(
                            source, center, axis, length, kernel[2:]
                        )
                        if axis == 0:
                            left = (2 * ey + segment) * nx + 2 * ex + cut - 1
                            right = left + 1
                        else:
                            left = (2 * ey + cut - 1) * nx + 2 * ex + segment
                            right = left + nx
                        matrix[left, indices] -= flux
                        matrix[right, indices] += flux
                        load[left] += correction
                        load[right] -= correction
    for j in range(ny):
        for i in range(nx):
            if i in (0, nx - 1) or j in (0, ny - 1):
                index = j * nx + i
                matrix[index] = 0
                matrix[index, index] = 1
                load[index] = (
                    boundary[0] + boundary[1] * x[i, 0] + boundary[2] * y[j, 0]
                )
    return matrix, load


def solve_carrier_balance(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    matrix, load = _assemble_carrier_balance(
        x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
    )
    try:
        result = np.linalg.solve(matrix, load)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the discrete system is singular") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("the discrete solution must be finite")
    return result.reshape(2 * len(y_knots) - 1, 2 * len(x_knots) - 1)

import numpy as np


def compute_carrier_population(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> float:
    """Evaluate the reference numerical operation."""
    field = solve_carrier_balance(
        x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
    )
    weights = np.outer([1.0, 4.0, 1.0], [1.0, 4.0, 1.0]) / 36
    total = 0.0
    for j, hy in enumerate(np.diff(y_knots)):
        for i, hx in enumerate(np.diff(x_knots)):
            local = field[2 * j : 2 * j + 3, 2 * i : 2 * i + 3]
            total += hx * hy * np.sum(weights * local)
    if not np.isfinite(total):
        raise ValueError("integrated population must be finite")
    return float(total)
SCICODE_GOLD_EOF
