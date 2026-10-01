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
def compute_kernel_normalization(kernel_coefficients: np.ndarray) -> float:
    """Reference analytic evaluation of the normalization ratio."""
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("kernel_coefficients must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("kernel_coefficients must be finite")
    rho = np.linspace(0.0, 1.0, 1001)
    if np.min(np.polynomial.polynomial.polyval(rho, coefficients)) < -1e-12:
        raise ValueError("the kernel must be nonnegative on [0, 1]")
    powers = np.arange(coefficients.size, dtype=float)
    numerator = np.sum(coefficients / (powers + 4.0))
    denominator = 2.0 * np.sum(coefficients / (powers + 3.0))
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("the defining kernel moments must be positive")
    result = float(numerator / denominator)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("the normalization must be positive and finite")
    return result

import numpy as np
def _validated_bond_matrix(values, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix of order at least two")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must have a zero diagonal")
    return matrix
def update_bond_phase_history(
    trial_driving_force: np.ndarray,
    previous_history: np.ndarray,
    critical_driving_force: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference irreversible maximum-history update."""
    trial = _validated_bond_matrix(trial_driving_force, "trial_driving_force")
    previous = _validated_bond_matrix(previous_history, "previous_history")
    if trial.shape != previous.shape:
        raise ValueError(
            "trial_driving_force and previous_history must have equal shapes"
        )
    if not np.isfinite(critical_driving_force) or critical_driving_force <= 0.0:
        raise ValueError("critical_driving_force must be positive and finite")
    history_new = np.maximum(previous, trial)
    phase_new = history_new / (history_new + float(critical_driving_force))
    np.fill_diagonal(history_new, 0.0)
    np.fill_diagonal(phase_new, 0.0)
    return history_new, np.minimum(1.0, phase_new)

import numpy as np
def _kinematic_factor(phase, threshold):
    factor = np.ones_like(phase, dtype=float)
    mask = phase > threshold
    factor[mask] = ((1.0 - phase[mask]) / (1.0 - threshold)) ** 2
    return factor
def _kernel_value(distance, horizon, coefficients):
    return np.polynomial.polynomial.polyval(distance / horizon, coefficients)
def build_kinematic_moments(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    """Reference moment assembly using the old-time phase field."""
    positions = np.asarray(reference_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    phase = np.asarray(previous_phase, dtype=float)
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 4:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least four"
        )
    point_count = positions.shape[0]
    if volumes.shape != (point_count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must have shape (N,) and be strictly positive")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(volumes)):
        raise ValueError("positions and volumes must be finite")
    if phase.shape != (point_count, point_count) or not np.all(np.isfinite(phase)):
        raise ValueError("previous_phase must be a finite (N, N) array")
    if np.any(phase < 0.0) or np.any(phase > 1.0):
        raise ValueError("previous_phase must lie in [0, 1]")
    if not np.allclose(phase, phase.T, rtol=0.0, atol=1e-12):
        raise ValueError("previous_phase must be symmetric")
    if not np.allclose(np.diag(phase), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("previous_phase must have a zero diagonal")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if (
        coefficients.ndim != 1
        or coefficients.size == 0
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("kernel_coefficients must be a nonempty finite vector")
    if not np.isfinite(kinematic_threshold) or not 0.0 <= kinematic_threshold < 1.0:
        raise ValueError("kinematic_threshold must lie in [0, 1)")
    factors = _kinematic_factor(phase, float(kinematic_threshold))
    moments = np.zeros((point_count, 3, 3), dtype=float)
    for k in range(point_count):
        for n in range(point_count):
            difference = positions[n] - positions[k]
            distance = float(np.linalg.norm(difference))
            if 0.0 < distance <= horizon:
                weight = _kernel_value(distance, horizon, coefficients)
                if weight < -1e-12:
                    raise ValueError("the kernel is negative on an active bond")
                moments[k] += (
                    weight
                    * factors[k, n]
                    * np.outer(difference, difference)
                    * volumes[n]
                )
        eigenvalues = np.linalg.eigvalsh(moments[k])
        if eigenvalues[0] <= 100.0 * np.finfo(float).eps * max(1.0, eigenvalues[-1]):
            raise ValueError("each moment matrix must be positive definite")
    return moments

import numpy as np
def compute_bond_shape_gradients(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    moments: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    """Reference system-solve construction of directed shape gradients."""
    positions = np.asarray(reference_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    phase = np.asarray(previous_phase, dtype=float)
    moments = np.asarray(moments, dtype=float)
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 4:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least four"
        )
    point_count = positions.shape[0]
    if volumes.shape != (point_count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive with shape (N,)")
    if (
        phase.shape != (point_count, point_count)
        or np.any(phase < 0.0)
        or np.any(phase > 1.0)
    ):
        raise ValueError("previous_phase must have shape (N, N) with values in [0, 1]")
    if not np.allclose(phase, phase.T, rtol=0.0, atol=1e-12) or not np.allclose(
        np.diag(phase), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("previous_phase must be symmetric with zero diagonal")
    if moments.shape != (point_count, 3, 3) or not np.all(np.isfinite(moments)):
        raise ValueError("moments must be a finite array of shape (N, 3, 3)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(volumes)):
        raise ValueError("positions and volumes must be finite")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if (
        coefficients.ndim != 1
        or coefficients.size == 0
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("kernel_coefficients must be a nonempty finite vector")
    if not np.isfinite(kinematic_threshold) or not 0.0 <= kinematic_threshold < 1.0:
        raise ValueError("kinematic_threshold must lie in [0, 1)")
    gradients = np.zeros((point_count, point_count, 3), dtype=float)
    for k in range(point_count):
        if not np.allclose(moments[k], moments[k].T, rtol=0.0, atol=1e-12):
            raise ValueError("each moment must be symmetric")
        eigenvalues = np.linalg.eigvalsh(moments[k])
        if eigenvalues[0] <= 100.0 * np.finfo(float).eps * max(1.0, eigenvalues[-1]):
            raise ValueError("each moment must be positive definite")
        for n in range(point_count):
            difference = positions[n] - positions[k]
            distance = float(np.linalg.norm(difference))
            if 0.0 < distance <= horizon:
                weight = np.polynomial.polynomial.polyval(
                    distance / horizon, coefficients
                )
                if weight < -1e-12:
                    raise ValueError("the kernel is negative on an active bond")
                factor = 1.0
                if phase[k, n] > kinematic_threshold:
                    factor = ((1.0 - phase[k, n]) / (1.0 - kinematic_threshold)) ** 2
                gradients[k, n] = (
                    weight
                    * factor
                    * np.linalg.solve(moments[k], difference)
                    * volumes[n]
                )
    return gradients

import numpy as np
def compute_point_deformation_gradients(
    displacements: np.ndarray,
    shape_gradients: np.ndarray,
) -> np.ndarray:
    """Reference first-order nonlocal reconstruction."""
    displacements = np.asarray(displacements, dtype=float)
    gradients = np.asarray(shape_gradients, dtype=float)
    if (
        displacements.ndim != 2
        or displacements.shape[1] != 3
        or displacements.shape[0] < 2
    ):
        raise ValueError("displacements must have shape (N, 3) with N at least two")
    point_count = displacements.shape[0]
    if gradients.shape != (point_count, point_count, 3):
        raise ValueError("shape_gradients must have shape (N, N, 3)")
    if not np.all(np.isfinite(displacements)) or not np.all(np.isfinite(gradients)):
        raise ValueError("displacements and shape_gradients must be finite")
    if not np.allclose(
        gradients[np.arange(point_count), np.arange(point_count)],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("self-bond shape gradients must be zero")
    deformation_gradients = np.repeat(np.eye(3)[None, :, :], point_count, axis=0)
    for k in range(point_count):
        for n in range(point_count):
            deformation_gradients[k] += np.outer(
                displacements[n] - displacements[k], gradients[k, n]
            )
    if not np.all(np.isfinite(deformation_gradients)):
        raise ValueError("deformation gradients must be finite")
    return deformation_gradients

import numpy as np
def update_bond_constitutive_state(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    point_gradients: np.ndarray,
    previous_history: np.ndarray,
    kernel_normalization: float,
    horizon: float,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference corrected-gradient and damage update."""
    reference = np.asarray(reference_positions, dtype=float)
    current = np.asarray(current_positions, dtype=float)
    gradients = np.asarray(point_gradients, dtype=float)
    history = np.asarray(previous_history, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or reference.shape[0] < 2:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least two"
        )
    point_count = reference.shape[0]
    if current.shape != reference.shape or gradients.shape != (point_count, 3, 3):
        raise ValueError("current positions or point gradients have invalid shapes")
    if history.shape != (point_count, point_count):
        raise ValueError("previous_history must have shape (N, N)")
    if (
        not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(gradients))
    ):
        raise ValueError("position and gradient inputs must be finite")
    if not np.all(np.isfinite(history)) or np.any(history < 0.0):
        raise ValueError("previous_history must be finite and nonnegative")
    if not np.allclose(history, history.T, rtol=0.0, atol=1e-12) or not np.allclose(
        np.diag(history), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("previous_history must be symmetric with zero diagonal")
    positive_parameters = (
        kernel_normalization,
        horizon,
        youngs_modulus,
        fracture_energy,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_parameters):
        raise ValueError(
            "c0, horizon, Young's modulus, and fracture energy must be positive and finite"
        )
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    lame_lambda = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    critical = fracture_energy / (2.0 * kernel_normalization * horizon)
    trial = np.zeros((point_count, point_count), dtype=float)
    piola_undamaged = np.zeros((point_count, point_count, 3, 3), dtype=float)
    identity = np.eye(3)
    for k in range(point_count):
        for n in range(k + 1, point_count):
            reference_bond = reference[n] - reference[k]
            distance = float(np.linalg.norm(reference_bond))
            if 0.0 < distance <= horizon:
                current_bond = current[n] - current[k]
                average = 0.5 * (gradients[k] + gradients[n])
                correction = (
                    np.outer(current_bond - average @ reference_bond, reference_bond)
                    / distance**2
                )
                corrected = average + correction
                determinant = float(np.linalg.det(corrected))
                if not np.isfinite(determinant) or determinant <= 0.0:
                    raise ValueError(
                        "every active corrected gradient must have positive determinant"
                    )
                green = 0.5 * (corrected.T @ corrected - identity)
                second_piola = (
                    lame_lambda * np.trace(green) * identity
                    + 2.0 * shear_modulus * green
                )
                first_piola = corrected @ second_piola
                cauchy = first_piola @ corrected.T / determinant
                maximum_principal = float(
                    np.linalg.eigvalsh(0.5 * (cauchy + cauchy.T))[-1]
                )
                driving_force = max(0.0, maximum_principal) ** 2 / (
                    2.0 * youngs_modulus
                )
                trial[k, n] = trial[n, k] = driving_force
                piola_undamaged[k, n] = piola_undamaged[n, k] = first_piola
    history_updater = globals()["update_bond_phase_history"]
    history_new, phase_new = history_updater(trial, history, critical)
    degraded = (1.0 - phase_new[..., None, None]) ** 2 * piola_undamaged
    return degraded, history_new, phase_new

import numpy as np
def compute_internal_force_density(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    shape_gradients: np.ndarray,
    degraded_piola: np.ndarray,
) -> np.ndarray:
    """Reference force-state assembly with normalized symmetric weights."""
    positions = np.asarray(reference_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    gradients = np.asarray(shape_gradients, dtype=float)
    stresses = np.asarray(degraded_piola, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least two"
        )
    point_count = positions.shape[0]
    if volumes.shape != (point_count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive with shape (N,)")
    if gradients.shape != (point_count, point_count, 3):
        raise ValueError("shape_gradients must have shape (N, N, 3)")
    if stresses.shape != (point_count, point_count, 3, 3):
        raise ValueError("degraded_piola must have shape (N, N, 3, 3)")
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(volumes))
        or not np.all(np.isfinite(gradients))
        or not np.all(np.isfinite(stresses))
    ):
        raise ValueError("all numerical arrays must be finite")
    if not np.allclose(
        gradients[np.arange(point_count), np.arange(point_count)],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("self-bond shape gradients must be zero")
    if not np.allclose(
        stresses[np.arange(point_count), np.arange(point_count)],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("self-bond stresses must be zero")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if (
        coefficients.ndim != 1
        or coefficients.size == 0
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("kernel_coefficients must be a nonempty finite vector")
    distances = np.linalg.norm(positions[None, :, :] - positions[:, None, :], axis=2)
    active = (distances > 0.0) & (distances <= horizon)
    weights = np.zeros((point_count, point_count), dtype=float)
    weights[active] = np.polynomial.polynomial.polyval(
        distances[active] / horizon, coefficients
    )
    if np.any(weights[active] < -1e-12):
        raise ValueError("the kernel is negative on an active bond")
    kernel_volumes = weights @ volumes
    if np.any(kernel_volumes <= 0.0) or not np.all(np.isfinite(kernel_volumes)):
        raise ValueError("each point must have positive finite kernel volume")
    normalized = (
        0.5 * weights * (1.0 / kernel_volumes[:, None] + 1.0 / kernel_volumes[None, :])
    )
    stabilization = np.zeros((point_count, 3, 3), dtype=float)
    identity = np.eye(3)
    for k in range(point_count):
        for n in range(point_count):
            if active[k, n]:
                direction = (positions[n] - positions[k]) / distances[k, n]
                stabilization[k] += (
                    normalized[k, n]
                    * stresses[k, n]
                    @ (identity - np.outer(direction, direction))
                    * volumes[n]
                )
    force_state = np.zeros((point_count, point_count, 3), dtype=float)
    for k in range(point_count):
        for n in range(point_count):
            if active[k, n]:
                direction = (positions[n] - positions[k]) / distances[k, n]
                bond_shape = gradients[k, n] / volumes[n]
                force_state[k, n] = (
                    normalized[k, n] * (stresses[k, n] @ direction)
                    + stabilization[k] @ bond_shape
                )
    internal_force = np.zeros((point_count, 3), dtype=float)
    for k in range(point_count):
        for n in range(point_count):
            internal_force[k] += (force_state[k, n] - force_state[n, k]) * volumes[n]
    return internal_force

import numpy as np
def compute_nonlocal_response_index(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    volumes: np.ndarray,
    previous_history: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
) -> float:
    """Reference end-to-end composition of the seven preceding oracles."""
    reference = np.asarray(reference_positions, dtype=float)
    current = np.asarray(current_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    history = np.asarray(previous_history, dtype=float)
    if current.shape != reference.shape:
        raise ValueError(
            "current_positions and reference_positions must have equal shapes"
        )
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if not np.isfinite(fracture_energy) or fracture_energy <= 0.0:
        raise ValueError("fracture_energy must be positive and finite")
    kernel_normalization = globals()["compute_kernel_normalization"](
        kernel_coefficients
    )
    critical = fracture_energy / (2.0 * kernel_normalization * horizon)
    _, previous_phase = globals()["update_bond_phase_history"](
        np.zeros_like(history), history, critical
    )
    moments = globals()["build_kinematic_moments"](
        reference,
        volumes,
        previous_phase,
        horizon,
        kernel_coefficients,
        kinematic_threshold,
    )
    shape_gradients = globals()["compute_bond_shape_gradients"](
        reference,
        volumes,
        previous_phase,
        moments,
        horizon,
        kernel_coefficients,
        kinematic_threshold,
    )
    point_gradients = globals()["compute_point_deformation_gradients"](
        current - reference, shape_gradients
    )
    degraded_piola, _, _ = globals()["update_bond_constitutive_state"](
        reference,
        current,
        point_gradients,
        history,
        kernel_normalization,
        horizon,
        youngs_modulus,
        poisson_ratio,
        fracture_energy,
    )
    internal_force = globals()["compute_internal_force_density"](
        reference,
        volumes,
        horizon,
        kernel_coefficients,
        shape_gradients,
        degraded_piola,
    )
    weighted_mean_square = np.sum(volumes * np.sum(internal_force**2, axis=1)) / np.sum(
        volumes
    )
    result = float(horizon / youngs_modulus * np.sqrt(weighted_mean_square))
    if not np.isfinite(result):
        raise ValueError("the response index must be finite")
    return result
SCICODE_GOLD_EOF
