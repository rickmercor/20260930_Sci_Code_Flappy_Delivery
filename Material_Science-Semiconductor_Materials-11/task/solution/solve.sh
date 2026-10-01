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


def pswf_bandwidth_sensitivity(c: float, quadrature_order: int) -> "np.ndarray":
    c = float(c)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c must be positive and finite")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order:
        raise ValueError("quadrature_order must be an integer")
    n = int(quadrature_order)
    if n < 8:
        raise ValueError("quadrature_order must be at least 8")

    nodes, weights = np.polynomial.legendre.leggauss(n)
    sqrt_w = np.sqrt(weights)
    xx = np.outer(nodes, nodes)
    matrix = sqrt_w[:, None] * np.cos(c * xx) * sqrt_w[None, :]
    derivative_matrix = sqrt_w[:, None] * (-xx * np.sin(c * xx)) * sqrt_w[None, :]

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalue = float(eigenvalues[0])
    vector = eigenvectors[:, 0].copy()
    integral = float(np.dot(weights, vector / sqrt_w))
    if integral < 0.0:
        vector = -vector
        eigenvectors[:, 0] = vector
        integral = -integral
    if integral == 0.0:
        raise RuntimeError("leading eigenvector has zero weighted integral")

    gaps = eigenvalue - eigenvalues[1:]
    if np.any(np.abs(gaps) <= 1.0e-12):
        raise RuntimeError("leading discrete eigenvalue is not simple")

    eigenvalue_derivative = float(vector @ (derivative_matrix @ vector))
    couplings = eigenvectors[:, 1:].T @ (derivative_matrix @ vector)
    vector_derivative = eigenvectors[:, 1:] @ (couplings / gaps)

    psi = vector / sqrt_w
    psi_derivative = vector_derivative / sqrt_w
    integral_derivative = float(np.dot(weights, psi_derivative))

    chi = psi / integral
    chi_derivative = (
        psi_derivative * integral - psi * integral_derivative
    ) / (integral * integral)

    return np.concatenate([
        nodes.astype(float),
        weights.astype(float),
        chi.astype(float),
        chi_derivative.astype(float),
        np.array([eigenvalue, eigenvalue_derivative], dtype=float),
    ])

import numpy as np


def prolate_transform_sensitivity(c: float, quadrature_order: int, s_values: "np.ndarray") -> "np.ndarray":
    s_values = np.asarray(s_values, dtype=float)
    if s_values.ndim != 1 or s_values.size == 0 or not np.all(np.isfinite(s_values)):
        raise ValueError("s_values must be a nonempty finite one-dimensional array")

    packed = pswf_bandwidth_sensitivity(c, quadrature_order)
    n = int(quadrature_order)
    nodes = packed[:n]
    weights = packed[n:2*n]
    chi = packed[2*n:3*n]
    chi_c = packed[3*n:4*n]
    eigenvalue = float(packed[-2])
    eigenvalue_c = float(packed[-1])

    vandermonde = np.polynomial.legendre.legvander(nodes, n - 1)
    degrees = np.arange(n, dtype=float)
    factor = 0.5 * (2.0 * degrees + 1.0)
    coefficients = factor * (vandermonde.T @ (weights * chi))
    coefficients_c = factor * (vandermonde.T @ (weights * chi_c))
    coefficients_x = np.polynomial.legendre.legder(coefficients)

    result = np.empty((s_values.size, 3), dtype=float)
    inside = np.abs(s_values) <= float(c)

    if np.any(inside):
        z = s_values[inside] / float(c)
        chi_z = np.polynomial.legendre.legval(z, coefficients)
        chi_c_z = np.polynomial.legendre.legval(z, coefficients_c)
        chi_x_z = np.polynomial.legendre.legval(z, coefficients_x)
        result[inside, 0] = eigenvalue * chi_z
        result[inside, 1] = (
            eigenvalue_c * chi_z
            + eigenvalue * (chi_c_z - chi_x_z * z / float(c))
        )
        result[inside, 2] = eigenvalue * chi_x_z / float(c)

    if np.any(~inside):
        outside_s = s_values[~inside]
        cosine = np.cos(np.outer(outside_s, nodes))
        sine = np.sin(np.outer(outside_s, nodes))
        result[~inside, 0] = cosine @ (weights * chi)
        result[~inside, 1] = cosine @ (weights * chi_c)
        result[~inside, 2] = -(sine @ (weights * chi * nodes))

    zero = s_values == 0.0
    result[zero, :] = np.array([1.0, 0.0, 0.0])
    return result

import numpy as np


def centered_reciprocal_indices(mode_count: int) -> "np.ndarray":
    if isinstance(mode_count, bool) or int(mode_count) != mode_count:
        raise ValueError("mode_count must be an integer")
    mode_count = int(mode_count)
    if mode_count < 4 or mode_count % 2 != 0:
        raise ValueError("mode_count must be even and at least 4")
    axis = np.arange(-mode_count // 2, mode_count // 2, dtype=int)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[np.any(grid != 0, axis=1)]

import numpy as np


def sheared_reciprocal_geometry(indices: "np.ndarray", box_length: float, shear: float) -> "np.ndarray":
    indices_array = np.asarray(indices)
    if indices_array.ndim != 2 or indices_array.shape[1] != 3 or indices_array.shape[0] == 0:
        raise ValueError("indices must have nonempty shape (M,3)")
    if not np.all(np.isfinite(indices_array)) or not np.all(indices_array == np.rint(indices_array)):
        raise ValueError("indices must be finite and integer-valued")
    integer_indices = indices_array.astype(int)
    if np.any(np.all(integer_indices == 0, axis=1)):
        raise ValueError("indices must exclude the zero row")

    box_length = float(box_length)
    shear = float(shear)
    if not np.isfinite(box_length) or box_length <= 0.0:
        raise ValueError("box_length must be positive and finite")
    if not np.isfinite(shear) or abs(shear) >= 0.5:
        raise ValueError("shear must satisfy |shear| < 0.5")

    cell = np.array([
        [box_length, shear * box_length, 0.0],
        [0.0, box_length, 0.0],
        [0.0, 0.0, box_length],
    ], dtype=float)
    cell_derivative = np.array([
        [0.0, box_length, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)

    inverse_transpose = np.linalg.inv(cell).T
    reciprocal = 2.0 * np.pi * inverse_transpose
    reciprocal_derivative = (
        -2.0 * np.pi
        * inverse_transpose
        @ cell_derivative.T
        @ inverse_transpose
    )

    xi = integer_indices.astype(float) @ reciprocal.T
    xi_derivative = integer_indices.astype(float) @ reciprocal_derivative.T
    norms = np.linalg.norm(xi, axis=1)
    norm_derivative = np.sum(xi * xi_derivative, axis=1) / norms

    return np.column_stack([xi, xi_derivative, norms, norm_derivative])

import numpy as np


def fractional_structure_factors(fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray") -> "np.ndarray":
    fractional_positions = np.asarray(fractional_positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    indices_array = np.asarray(indices)

    if fractional_positions.ndim != 2 or fractional_positions.shape[1] != 3 or fractional_positions.shape[0] == 0:
        raise ValueError("fractional_positions must have shape (N,3) with N > 0")
    if charges.ndim != 1 or charges.size != fractional_positions.shape[0]:
        raise ValueError("charges must be one-dimensional with length N")
    if indices_array.ndim != 2 or indices_array.shape[1] != 3:
        raise ValueError("indices must have shape (M,3)")
    if not np.all(np.isfinite(fractional_positions)) or not np.all(np.isfinite(charges)):
        raise ValueError("particle data must be finite")
    if not np.all(np.isfinite(indices_array)) or not np.all(indices_array == np.rint(indices_array)):
        raise ValueError("indices must be finite and integer-valued")

    phase = 2.0 * np.pi * (indices_array.astype(float) @ fractional_positions.T)
    return np.exp(1j * phase) @ charges

import numpy as np


def _inverse_sigmoid_coordinates(y: "np.ndarray", bounds: "np.ndarray") -> tuple:
    sigmoid = np.empty_like(y, dtype=float)
    positive = y >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-y[positive]))
    exp_y = np.exp(y[~positive])
    sigmoid[~positive] = exp_y / (1.0 + exp_y)
    span = bounds[:, 1] - bounds[:, 0]
    parameters = bounds[:, 0] + span * sigmoid
    derivative = span * sigmoid * (1.0 - sigmoid)
    return parameters, derivative


def scaled_inverse_residual(y: "np.ndarray", target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    y = np.asarray(y, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    target_scales = np.asarray(target_scales, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    fractional_positions = np.asarray(fractional_positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    indices_array = np.asarray(indices)
    box_length = float(box_length)
    reference_frequency = float(reference_frequency)

    if y.shape != (3,) or target_values.shape != (3,) or target_scales.shape != (3,):
        raise ValueError("y, target_values, and target_scales must have length 3")
    if bounds.shape != (3, 2):
        raise ValueError("bounds must have shape (3,2)")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(target_values)) or not np.all(np.isfinite(target_scales)) or not np.all(np.isfinite(bounds)):
        raise ValueError("solver vectors and bounds must be finite")
    if np.any(target_scales <= 0.0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("target scales must be positive and bounds must be ordered")
    if fractional_positions.ndim != 2 or fractional_positions.shape[1] != 3 or fractional_positions.shape[0] == 0:
        raise ValueError("fractional_positions must have shape (N,3)")
    if charges.ndim != 1 or charges.size != fractional_positions.shape[0]:
        raise ValueError("charges must have length N")
    if not np.all(np.isfinite(fractional_positions)) or not np.all(np.isfinite(charges)):
        raise ValueError("particle data must be finite")
    if abs(float(np.sum(charges))) > 1.0e-12:
        raise ValueError("charges must be neutral within 1e-12")
    if indices_array.ndim != 2 or indices_array.shape[1] != 3 or indices_array.shape[0] == 0:
        raise ValueError("indices must have nonempty shape (M,3)")
    if not np.all(np.isfinite(indices_array)) or not np.all(indices_array == np.rint(indices_array)):
        raise ValueError("indices must be finite and integer-valued")
    integer_indices = indices_array.astype(int)
    if np.any(np.all(integer_indices == 0, axis=1)):
        raise ValueError("indices must exclude zero")
    if not np.isfinite(box_length) or box_length <= 0.0:
        raise ValueError("box_length must be positive and finite")
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError("reference_frequency must be positive and finite")

    parameters, dp_dy = _inverse_sigmoid_coordinates(y, bounds)
    c, cutoff, shear = [float(value) for value in parameters]
    if not (0.0 < cutoff < box_length / 2.0):
        raise ValueError("transformed cutoff must satisfy 0 < cutoff < box_length/2")
    if abs(shear) >= 0.5:
        raise ValueError("transformed shear must satisfy |shear| < 0.5")

    diagnostic = prolate_transform_sensitivity(
        c, quadrature_order, np.array([cutoff * reference_frequency], dtype=float)
    )[0]

    geometry = sheared_reciprocal_geometry(integer_indices, box_length, shear)
    xi = geometry[:, 0:3]
    xi_shear = geometry[:, 3:6]
    norms = geometry[:, 6]
    norms_shear = geometry[:, 7]

    transform = prolate_transform_sensitivity(
        c, quadrature_order, cutoff * norms
    )
    chi_hat = transform[:, 0]
    chi_hat_c = transform[:, 1]
    chi_hat_s = transform[:, 2]

    structure = fractional_structure_factors(
        fractional_positions, charges, integer_indices
    )
    structure_power = np.abs(structure) ** 2
    volume = box_length ** 3

    weights = chi_hat / (volume * norms ** 2)
    weights_c = chi_hat_c / (volume * norms ** 2)
    weights_cutoff = chi_hat_s / (volume * norms)
    weights_shear = (
        chi_hat_s * cutoff * norms_shear / (volume * norms ** 2)
        - 2.0 * chi_hat * norms_shear / (volume * norms ** 3)
    )

    energy = 0.5 * float(np.sum(weights * structure_power))
    energy_c = 0.5 * float(np.sum(weights_c * structure_power))
    energy_cutoff = 0.5 * float(np.sum(weights_cutoff * structure_power))
    energy_shear = 0.5 * float(np.sum(weights_shear * structure_power))

    phase = 2.0 * np.pi * (integer_indices.astype(float) @ fractional_positions.T)
    first_imag = np.imag(np.exp(1j * phase[:, 0]) * np.conj(structure))
    q0 = float(charges[0])
    force_x = q0 * float(np.sum(weights * xi[:, 0] * first_imag))
    force_x_c = q0 * float(np.sum(weights_c * xi[:, 0] * first_imag))
    force_x_cutoff = q0 * float(np.sum(weights_cutoff * xi[:, 0] * first_imag))
    force_x_shear = q0 * float(np.sum(
        (weights_shear * xi[:, 0] + weights * xi_shear[:, 0]) * first_imag
    ))

    observables = np.array([diagnostic[0], energy, force_x], dtype=float)
    physical_jacobian = np.array([
        [diagnostic[1], reference_frequency * diagnostic[2], 0.0],
        [energy_c, energy_cutoff, energy_shear],
        [force_x_c, force_x_cutoff, force_x_shear],
    ], dtype=float)

    residual = (observables - target_values) / target_scales
    jacobian_y = (
        physical_jacobian * dp_dy[None, :]
    ) / target_scales[:, None]

    return np.concatenate([residual, jacobian_y.ravel(), parameters])

import numpy as np


def damped_lm_iteration(y: "np.ndarray", damping: float, trust_radius: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    y = np.asarray(y, dtype=float)
    damping = float(damping)
    trust_radius = float(trust_radius)
    if y.shape != (3,) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite vector of length 3")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")
    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("trust_radius must be positive and finite")

    current = scaled_inverse_residual(
        y, target_values, target_scales, bounds, fractional_positions, charges,
        indices, box_length, quadrature_order, reference_frequency
    )
    residual = current[:3]
    jacobian = current[3:12].reshape(3, 3)
    objective = 0.5 * float(residual @ residual)

    left, singular_values, right_t = np.linalg.svd(jacobian, full_matrices=False)
    delta = -right_t.T @ (
        (singular_values / (singular_values ** 2 + damping))
        * (left.T @ residual)
    )
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > trust_radius:
        delta = delta * (trust_radius / delta_norm)

    new_y = y.copy()
    new_damping = damping
    new_trust = trust_radius
    new_objective = objective
    accepted = 0.0
    alpha = 1.0

    for _ in range(12):
        step = alpha * delta
        trial = scaled_inverse_residual(
            y + step, target_values, target_scales, bounds, fractional_positions,
            charges, indices, box_length, quadrature_order, reference_frequency
        )
        trial_residual = trial[:3]
        trial_objective = 0.5 * float(trial_residual @ trial_residual)
        if np.isfinite(trial_objective) and trial_objective < objective:
            predicted = objective - 0.5 * float(
                (residual + jacobian @ step) @ (residual + jacobian @ step)
            )
            actual = objective - trial_objective
            rho = actual / predicted if predicted > 0.0 else -np.inf
            new_y = y + step
            new_objective = trial_objective
            accepted = 1.0
            if rho > 0.75:
                new_trust = min(2.0 * trust_radius, 4.0)
                new_damping = max(0.5 * damping, 1.0e-12)
            elif rho < 0.25:
                new_trust = max(0.5 * trust_radius, 1.0e-6)
                new_damping = min(4.0 * damping, 1.0e12)
            break
        alpha *= 0.5

    if accepted == 0.0:
        new_trust = max(0.5 * trust_radius, 1.0e-8)
        new_damping = min(10.0 * damping, 1.0e12)

    return np.concatenate([
        new_y,
        np.array([new_damping, new_trust, accepted, objective, new_objective], dtype=float),
    ])

import numpy as np


def solve_inverse_precision(fractional_positions: "np.ndarray", charges: "np.ndarray", box_length: float, mode_count: int, quadrature_order: int, reference_frequency: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", initial_parameters: "np.ndarray", tolerance: float, max_iterations: int) -> float:
    bounds = np.asarray(bounds, dtype=float)
    initial_parameters = np.asarray(initial_parameters, dtype=float)
    tolerance = float(tolerance)
    if bounds.shape != (3, 2) or initial_parameters.shape != (3,):
        raise ValueError("bounds must be (3,2) and initial_parameters length 3")
    if not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(initial_parameters)):
        raise ValueError("bounds and initial parameters must be finite")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be strictly ordered")
    if np.any(initial_parameters <= bounds[:, 0]) or np.any(initial_parameters >= bounds[:, 1]):
        raise ValueError("initial_parameters must lie strictly inside bounds")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or int(max_iterations) < 1:
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)

    indices = centered_reciprocal_indices(mode_count)
    fraction = (initial_parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y = np.log(fraction / (1.0 - fraction))
    damping = 1.0e-2
    trust_radius = 1.0

    for _ in range(max_iterations):
        state = scaled_inverse_residual(
            y, target_values, target_scales, bounds, fractional_positions, charges,
            indices, box_length, quadrature_order, reference_frequency
        )
        if float(np.max(np.abs(state[:3]))) <= tolerance:
            break

        iteration = damped_lm_iteration(
            y, damping, trust_radius, target_values, target_scales, bounds,
            fractional_positions, charges, indices, box_length, quadrature_order,
            reference_frequency
        )
        y = iteration[:3]
        damping = float(iteration[3])
        trust_radius = float(iteration[4])

    state = scaled_inverse_residual(
        y, target_values, target_scales, bounds, fractional_positions, charges,
        indices, box_length, quadrature_order, reference_frequency
    )
    if float(np.max(np.abs(state[:3]))) > tolerance:
        raise RuntimeError("inverse solve did not converge within max_iterations")
    return _prolate_edge_value(float(state[12]), quadrature_order)


def _prolate_edge_value(c: float, quadrature_order: int) -> float:
    packed = pswf_bandwidth_sensitivity(c, quadrature_order)
    n = int(quadrature_order)
    nodes = packed[:n]
    weights = packed[n:2*n]
    chi = packed[2*n:3*n]
    eigenvalue = float(packed[-2])
    psi = chi / np.sqrt(float(np.dot(weights, chi * chi)))
    return float(np.dot(weights * np.cos(c * nodes), psi) / eigenvalue)
SCICODE_GOLD_EOF
