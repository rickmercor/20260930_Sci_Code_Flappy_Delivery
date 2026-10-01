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


def _sample_points(n_pixels, cell_lengths):
    """Sample coordinates x_d = i L_d / n on the two axes, meshed in ij order."""
    axes = [np.arange(n_pixels) * float(cell_lengths[d]) / n_pixels for d in range(2)]
    return np.meshgrid(axes[0], axes[1], indexing="ij")


def rasterise_periodic_disc_microstructure(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
) -> dict:
    """Reference implementation."""
    if not isinstance(n_pixels, (int, np.integer)) or int(n_pixels) < 3 or int(n_pixels) % 2 == 0:
        raise ValueError("n_pixels must be an odd integer of at least three")
    n_pixels = int(n_pixels)
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    conductivities = np.asarray(phase_conductivities, dtype=float).ravel()
    if conductivities.size < 1 or np.any(conductivities <= 0.0):
        raise ValueError("phase conductivities must be strictly positive")
    grid = _sample_points(n_pixels, lengths)
    phase = np.ones((n_pixels, n_pixels), dtype=int)
    for disc in discs:
        centre_1, centre_2, radius, label = float(disc[0]), float(disc[1]), float(disc[2]), int(disc[3])
        if radius <= 0.0:
            raise ValueError("disc radius must be strictly positive")
        if label < 2 or label > conductivities.size:
            raise ValueError("disc phase label must lie between two and the number of phases")
        offsets = []
        for d, centre in enumerate((centre_1, centre_2)):
            delta = np.abs(grid[d] - centre) % lengths[d]
            offsets.append(np.minimum(delta, lengths[d] - delta))
        distance = np.sqrt(offsets[0] ** 2 + offsets[1] ** 2)
        if np.any(np.abs(distance - radius) < 1e-9):
            raise ValueError("a sample point lies on a disc boundary")
        phase[distance < radius] = label
    conductivity = conductivities[phase - 1]
    counts = tuple(int((phase == q + 1).sum()) for q in range(conductivities.size))
    total = float(n_pixels * n_pixels)
    return {
        "phase": phase,
        "conductivity": conductivity,
        "pixel_counts": counts,
        "volume_fractions": tuple(c / total for c in counts),
        "mean_conductivity": float(conductivity.mean()),
    }

import numpy as np


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def apply_periodic_conductivity_operator(
    field: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
) -> np.ndarray:
    """Reference implementation."""
    field = np.asarray(field, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    if field.ndim < 2 or field.shape[-1] != field.shape[-2] or field.shape[-1] < 3 or field.shape[-1] % 2 == 0:
        raise ValueError("field must end with two square axes of odd size at least three")
    n_pixels = field.shape[-1]
    if conductivity.shape != (n_pixels, n_pixels) or not np.all(np.isfinite(conductivity)) or np.any(conductivity < 0.0):
        raise ValueError("conductivity must be a finite non-negative array matching the grid")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    multipliers = _frequency_multipliers(n_pixels, lengths)
    gradient_1, gradient_2 = _spectral_gradient(field, multipliers)
    return -_spectral_divergence(conductivity * gradient_1, conductivity * gradient_2, multipliers)

import numpy as np


def _mean_free_trigonometric_basis(n_pixels, cell_lengths):
    """Orthonormal cosine and sine modes over a half-space of the odd frequency set."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels).astype(int)
    axes = [np.arange(n_pixels) * float(cell_lengths[d]) / n_pixels for d in range(2)]
    x_1, x_2 = np.meshgrid(axes[0], axes[1], indexing="ij")
    scale = np.sqrt(2.0 / (n_pixels * n_pixels))
    modes = []
    for k_1 in integer_frequencies:
        for k_2 in integer_frequencies:
            if k_1 > 0 or (k_1 == 0 and k_2 > 0):
                angle = 2.0 * np.pi * (k_1 * x_1 / float(cell_lengths[0]) + k_2 * x_2 / float(cell_lengths[1]))
                modes.append(scale * np.cos(angle))
                modes.append(scale * np.sin(angle))
    return np.stack(modes, axis=0)


def assemble_mean_free_operator_matrices(
    conductivity: np.ndarray,
    cell_lengths: tuple,
    reference_conductivity: float,
) -> dict:
    """Reference implementation."""
    conductivity = np.asarray(conductivity, dtype=float)
    if conductivity.ndim != 2 or conductivity.shape[0] != conductivity.shape[1] or conductivity.shape[0] < 3 or conductivity.shape[0] % 2 == 0:
        raise ValueError("conductivity must be a square array of odd size at least three")
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity < 0.0):
        raise ValueError("conductivity must be finite and non-negative")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    reference_conductivity = float(reference_conductivity)
    if reference_conductivity <= 0.0:
        raise ValueError("reference_conductivity must be strictly positive")
    n_pixels = conductivity.shape[0]
    n_points = n_pixels * n_pixels
    basis = _mean_free_trigonometric_basis(n_pixels, lengths)
    flat = basis.reshape(basis.shape[0], n_points)
    image = apply_periodic_conductivity_operator(basis, conductivity, lengths)  # noqa: F821
    operator = flat @ image.reshape(basis.shape[0], n_points).T / n_points
    uniform = np.full((n_pixels, n_pixels), reference_conductivity)
    reference_image = apply_periodic_conductivity_operator(basis, uniform, lengths)  # noqa: F821
    reference = flat @ reference_image.reshape(basis.shape[0], n_points).T / n_points
    return {
        "basis": basis,
        "operator": 0.5 * (operator + operator.T),
        "reference_operator": 0.5 * (reference + reference.T),
    }

import numpy as np
import scipy.linalg as sla


def solve_preconditioned_cell_operator_spectrum(
    operator: np.ndarray,
    reference_operator: np.ndarray,
    phase_conductivities: np.ndarray,
    reference_conductivity: float,
    plateau_tolerance: float,
) -> dict:
    """Reference implementation."""
    operator = np.asarray(operator, dtype=float)
    reference_operator = np.asarray(reference_operator, dtype=float)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1] or operator.shape != reference_operator.shape:
        raise ValueError("operator and reference_operator must be square matrices of the same shape")
    if np.abs(operator - operator.T).max() > 1e-8 or np.abs(reference_operator - reference_operator.T).max() > 1e-8:
        raise ValueError("both matrices must be symmetric")
    conductivities = np.asarray(phase_conductivities, dtype=float).ravel()
    reference_conductivity = float(reference_conductivity)
    if conductivities.size < 1 or np.any(conductivities <= 0.0) or reference_conductivity <= 0.0:
        raise ValueError("conductivities must be strictly positive")
    plateau_tolerance = float(plateau_tolerance)
    if plateau_tolerance <= 0.0:
        raise ValueError("plateau_tolerance must be strictly positive")
    try:
        shifted, vectors = sla.eigh(operator, reference_operator)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError("reference_operator must be positive definite") from error
    eigenvalues = 1.0 - shifted
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    gram = vectors.T @ reference_operator @ vectors
    plateau_values = tuple(float(1.0 - c / reference_conductivity) for c in conductivities)
    on_plateau = np.zeros(eigenvalues.size, dtype=bool)
    counts = []
    for value in plateau_values:
        near = np.abs(eigenvalues - value) < plateau_tolerance
        counts.append(int(near.sum()))
        on_plateau |= near
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": vectors,
        "bounds": (float(1.0 - conductivities.max() / reference_conductivity),
                   float(1.0 - conductivities.min() / reference_conductivity)),
        "plateau_values": plateau_values,
        "plateau_counts": tuple(counts),
        "transition_count": int((~on_plateau).sum()),
        "orthonormality_residual": float(np.abs(gram - np.eye(eigenvalues.size)).max()),
    }

import numpy as np


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def project_cell_load_onto_eigenstates(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Reference implementation."""
    basis = np.asarray(basis, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2] or conductivity.shape != basis.shape[1:]:
        raise ValueError("basis must have shape (N - 1, n, n) matching the conductivity grid")
    n_modes = basis.shape[0]
    if eigenvalues.shape != (n_modes,) or eigenvectors.shape != (n_modes, n_modes):
        raise ValueError("eigenvalues and eigenvectors must match the number of basis fields")
    if np.any(eigenvalues >= 1.0):
        raise ValueError("every eigenvalue must be strictly below one")
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    n_pixels = basis.shape[1]
    n_points = n_pixels * n_pixels
    multipliers = _frequency_multipliers(n_pixels, lengths)
    load_field = _spectral_divergence(conductivity * gradient[0], conductivity * gradient[1], multipliers)
    load = basis.reshape(n_modes, n_points) @ load_field.ravel() / n_points
    projected = eigenvectors.T @ load
    components = projected / (1.0 - eigenvalues)
    eigenstates = np.tensordot(eigenvectors.T, basis, axes=(1, 0))
    gradient_1, gradient_2 = _spectral_gradient(eigenstates, multipliers)
    spectral_vectors = np.stack([
        (conductivity * gradient_1).mean(axis=(1, 2)),
        (conductivity * gradient_2).mean(axis=(1, 2)),
    ], axis=1)
    dominant = int(np.argmax(np.abs(components)))
    return {
        "load": load,
        "components": components,
        "spectral_vectors": spectral_vectors,
        "identity_residual": float(np.abs(projected + spectral_vectors @ gradient).max()),
        "dominant_index": dominant,
        "dominant_eigenvalue": float(eigenvalues[dominant]),
        "dominant_magnitude": float(abs(components[dominant])),
    }

import numpy as np


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def reconstruct_solution_and_effective_tensor(
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    conductivity: np.ndarray,
    cell_lengths: tuple,
    macroscopic_gradient: np.ndarray,
) -> dict:
    """Reference implementation."""
    basis = np.asarray(basis, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    components = np.asarray(components, dtype=float).ravel()
    spectral_vectors = np.asarray(spectral_vectors, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2] or conductivity.shape != basis.shape[1:]:
        raise ValueError("basis must have shape (N - 1, n, n) matching the conductivity grid")
    n_modes = basis.shape[0]
    if (eigenvalues.shape != (n_modes,) or eigenvectors.shape != (n_modes, n_modes)
            or components.shape != (n_modes,) or spectral_vectors.shape != (n_modes, 2)):
        raise ValueError("spectral arrays must match the number of basis fields")
    if np.any(eigenvalues >= 1.0):
        raise ValueError("every eigenvalue must be strictly below one")
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    n_pixels = basis.shape[1]
    multipliers = _frequency_multipliers(n_pixels, lengths)
    fluctuation = np.tensordot(eigenvectors @ components, basis, axes=(0, 0))
    gradient_1, gradient_2 = _spectral_gradient(fluctuation, multipliers)
    flux_1 = conductivity * (gradient[0] + gradient_1)
    flux_2 = conductivity * (gradient[1] + gradient_2)
    effective_column = np.array([flux_1.mean(), flux_2.mean()])
    load_field = _spectral_divergence(conductivity * gradient[0], conductivity * gradient[1], multipliers)
    residual_field = _spectral_divergence(flux_1, flux_2, multipliers)
    load_scale = float(np.abs(load_field).max())
    mean_conductivity = float(conductivity.mean())
    effective_tensor = (mean_conductivity * np.eye(2)
                        + (spectral_vectors.T * (1.0 / (eigenvalues - 1.0))) @ spectral_vectors)
    return {
        "fluctuation": fluctuation,
        "effective_column": effective_column,
        "effective_tensor": effective_tensor,
        "mean_conductivity": mean_conductivity,
        "consistency_residual": float(np.abs(effective_tensor @ gradient - effective_column).max()),
        "equilibrium_residual": float(np.abs(residual_field).max() / (load_scale if load_scale > 0.0 else 1.0)),
        "energy_norm": float(np.sqrt(np.sum(components ** 2))),
    }

import numpy as np


def build_truncated_reduced_model(
    eigenvalues: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    mean_conductivity: float,
    effective_column: np.ndarray,
    macroscopic_gradient: np.ndarray,
    energy_error_threshold: float,
) -> dict:
    """Reference implementation."""
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    components = np.asarray(components, dtype=float).ravel()
    spectral_vectors = np.asarray(spectral_vectors, dtype=float)
    effective_column = np.asarray(effective_column, dtype=float).ravel()
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    n_modes = components.size
    if eigenvalues.shape != (n_modes,) or spectral_vectors.shape != (n_modes, 2):
        raise ValueError("eigenvalues and spectral_vectors must match the number of components")
    if gradient.shape != (2,) or effective_column.shape != (2,):
        raise ValueError("macroscopic_gradient and effective_column must have two components")
    mean_conductivity = float(mean_conductivity)
    if mean_conductivity <= 0.0:
        raise ValueError("mean_conductivity must be strictly positive")
    threshold = float(energy_error_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("energy_error_threshold must lie strictly between zero and one")
    total = float(np.sum(components ** 2))
    if total <= 0.0:
        raise ValueError("at least one component must be non-zero")
    order = np.argsort(-np.abs(components), kind="stable")
    squared = components[order] ** 2
    tail = total - np.concatenate([[0.0], np.cumsum(squared)])
    errors = np.sqrt(np.maximum(tail, 0.0) / total)
    n_retained = int(np.argmax(errors < threshold))
    selected = order[:n_retained]
    truncated_column = mean_conductivity * gradient + spectral_vectors[selected].T @ components[selected]
    return {
        "n_retained": n_retained,
        "selected_indices": selected,
        "selected_eigenvalues": eigenvalues[selected],
        "energy_error": float(errors[n_retained]),
        "energy_error_before": float(errors[n_retained - 1]) if n_retained > 0 else 1.0,
        "truncated_column": truncated_column,
        "relative_errors": np.abs(truncated_column - effective_column) / np.abs(effective_column),
        "retained_fraction": n_retained / float(n_modes),
    }

import numpy as np
import scipy.linalg as sla


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def compute_contrast_independent_representation(
    phase: np.ndarray,
    cell_lengths: tuple,
    matrix_phase: int,
    matrix_conductivity: float,
    contrast: float,
    macroscopic_gradient: np.ndarray,
    limit_tolerance: float,
) -> dict:
    """Reference implementation."""
    phase = np.asarray(phase)
    if (phase.ndim != 2 or phase.shape[0] != phase.shape[1] or phase.shape[0] < 3
            or phase.shape[0] % 2 == 0 or not np.issubdtype(phase.dtype, np.integer)):
        raise ValueError("phase must be a square integer array of odd size at least three")
    indicator = (phase == int(matrix_phase)).astype(float)
    if indicator.sum() == 0.0 or indicator.sum() == indicator.size:
        raise ValueError("matrix_phase must occur without filling the whole cell")
    matrix_conductivity = float(matrix_conductivity)
    contrast = float(contrast)
    if matrix_conductivity <= 0.0:
        raise ValueError("matrix_conductivity must be strictly positive")
    if contrast <= 0.0 or contrast == 1.0:
        raise ValueError("contrast must be strictly positive and different from one")
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    limit_tolerance = float(limit_tolerance)
    if limit_tolerance <= 0.0:
        raise ValueError("limit_tolerance must be strictly positive")
    n_pixels = phase.shape[0]
    n_points = n_pixels * n_pixels
    multipliers = _frequency_multipliers(n_pixels, lengths)
    geometric = assemble_mean_free_operator_matrices(indicator, lengths, 1.0)  # noqa: F821
    basis = geometric["basis"]
    mu, vectors = sla.eigh(geometric["operator"], geometric["reference_operator"])
    order = np.argsort(mu, kind="stable")
    mu = mu[order]
    vectors = vectors[:, order]
    eigenstates = np.tensordot(vectors.T, basis, axes=(1, 0))
    gradient_1, gradient_2 = _spectral_gradient(eigenstates, multipliers)
    couplings = np.stack([
        (indicator * gradient_1).mean(axis=(1, 2)),
        (indicator * gradient_2).mean(axis=(1, 2)),
    ], axis=1)
    projections = couplings @ gradient
    inclusion_conductivity = contrast * matrix_conductivity
    s_parameter = contrast / (contrast - 1.0)
    conductivity = np.where(indicator > 0.5, matrix_conductivity, inclusion_conductivity)
    effective_column = (conductivity.mean() * gradient
                        + (inclusion_conductivity - matrix_conductivity) * (couplings.T @ (projections / (mu - s_parameter))))
    direct = assemble_mean_free_operator_matrices(conductivity, lengths, inclusion_conductivity)  # noqa: F821
    load_field = _spectral_divergence(conductivity * gradient[0], conductivity * gradient[1], multipliers)
    load = basis.reshape(basis.shape[0], n_points) @ load_field.ravel() / n_points
    fluctuation = np.tensordot(np.linalg.solve(direct["operator"], load), basis, axes=(0, 0))
    fluctuation_1, fluctuation_2 = _spectral_gradient(fluctuation, multipliers)
    direct_column = np.array([
        (conductivity * (gradient[0] + fluctuation_1)).mean(),
        (conductivity * (gradient[1] + fluctuation_2)).mean(),
    ])
    return {
        "geometric_eigenvalues": mu,
        "limit_counts": (int((mu < limit_tolerance).sum()), int((mu > 1.0 - limit_tolerance).sum())),
        "normalised_projections": projections,
        "max_projection": float(np.abs(projections).max()),
        "contrast_parameter": float(s_parameter),
        "effective_column": effective_column,
        "direct_column": direct_column,
        "consistency_residual": float(np.abs(effective_column - direct_column).max()),
    }

import numpy as np


def compute_reduced_model_conductivity_error(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
    macroscopic_gradient: np.ndarray,
    plateau_tolerance: float,
    energy_error_threshold: float,
    two_phase_contrast: float,
) -> dict:
    """Reference implementation chaining every earlier step."""
    if not isinstance(n_pixels, (int, np.integer)) or int(n_pixels) < 3 or int(n_pixels) % 2 == 0:
        raise ValueError("n_pixels must be an odd integer of at least three")
    n_pixels = int(n_pixels)
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    conductivities = np.asarray(phase_conductivities, dtype=float).ravel()
    if conductivities.size < 2 or np.any(conductivities <= 0.0):
        raise ValueError("at least two strictly positive phase conductivities are required")
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    plateau_tolerance = float(plateau_tolerance)
    if plateau_tolerance <= 0.0:
        raise ValueError("plateau_tolerance must be strictly positive")
    threshold = float(energy_error_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("energy_error_threshold must lie strictly between zero and one")
    contrast = float(two_phase_contrast)
    if contrast <= 0.0 or contrast == 1.0:
        raise ValueError("two_phase_contrast must be strictly positive and different from one")
    reference_conductivity = 0.5 * (float(conductivities.min()) + float(conductivities.max()))

    microstructure = rasterise_periodic_disc_microstructure(  # noqa: F821
        n_pixels, lengths, discs, conductivities
    )
    conductivity = microstructure["conductivity"]
    matrices = assemble_mean_free_operator_matrices(  # noqa: F821
        conductivity, lengths, reference_conductivity
    )
    spectrum = solve_preconditioned_cell_operator_spectrum(  # noqa: F821
        matrices["operator"], matrices["reference_operator"], conductivities,
        reference_conductivity, plateau_tolerance,
    )
    projection = project_cell_load_onto_eigenstates(  # noqa: F821
        matrices["basis"], spectrum["eigenvalues"], spectrum["eigenvectors"],
        conductivity, lengths, gradient,
    )
    solution = reconstruct_solution_and_effective_tensor(  # noqa: F821
        matrices["basis"], spectrum["eigenvalues"], spectrum["eigenvectors"],
        projection["components"], projection["spectral_vectors"], conductivity, lengths, gradient,
    )
    reduced = build_truncated_reduced_model(  # noqa: F821
        spectrum["eigenvalues"], projection["components"], projection["spectral_vectors"],
        solution["mean_conductivity"], solution["effective_column"], gradient, threshold,
    )
    two_phase = compute_contrast_independent_representation(  # noqa: F821
        microstructure["phase"], lengths, 1, float(conductivities[0]), contrast, gradient, plateau_tolerance,
    )
    return {
        "relative_error": float(reduced["relative_errors"][0]),
        "pixel_counts": microstructure["pixel_counts"],
        "reference_conductivity": float(reference_conductivity),
        "bounds": spectrum["bounds"],
        "plateau_values": spectrum["plateau_values"],
        "plateau_counts": spectrum["plateau_counts"],
        "transition_count": spectrum["transition_count"],
        "effective_tensor": solution["effective_tensor"],
        "dominant_magnitude": projection["dominant_magnitude"],
        "dominant_eigenvalue": projection["dominant_eigenvalue"],
        "n_retained": reduced["n_retained"],
        "energy_error": reduced["energy_error"],
        "reduced_conductivity": float(reduced["truncated_column"][0]),
        "two_phase_limit_counts": two_phase["limit_counts"],
        "two_phase_max_projection": two_phase["max_projection"],
        "two_phase_conductivity": float(two_phase["effective_column"][0]),
        "identity_residual": projection["identity_residual"],
        "equilibrium_residual": solution["equilibrium_residual"],
        "two_phase_residual": two_phase["consistency_residual"],
    }
SCICODE_GOLD_EOF
