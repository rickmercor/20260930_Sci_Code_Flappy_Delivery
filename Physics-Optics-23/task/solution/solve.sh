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


def orthonormal_source_modes(
    shape: tuple,
    sigmas: "np.ndarray",
    tilts: "np.ndarray",
    coupling: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two integers of at least 2.")
    widths = np.asarray(sigmas, dtype=np.float64)
    carriers = np.asarray(tilts, dtype=np.float64)
    mix = np.asarray(coupling, dtype=np.float64)
    if widths.ndim != 1 or mix.ndim != 1 or carriers.ndim != 2:
        raise ValueError("sigmas, tilts, and coupling have inconsistent ranks.")
    n_modes = widths.shape[0]
    if n_modes < 1 or carriers.shape != (n_modes, 2) or mix.shape != (n_modes,):
        raise ValueError("sigmas, tilts, and coupling must describe the same modes.")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("sigmas must be positive and finite.")
    if not np.all(np.isfinite(carriers)) or not np.all(np.isfinite(mix)):
        raise ValueError("tilts and coupling must be finite.")
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    columns = []
    for width, (tilt_x, tilt_y), factor in zip(widths, carriers, mix):
        envelope = np.exp(-(grid_x * grid_x + grid_y * grid_y) / (2.0 * width * width))
        phase = tilt_x * grid_x + tilt_y * grid_y + factor * grid_x * grid_y
        columns.append((envelope * np.exp(1j * phase)).ravel())
    design = np.column_stack(columns).astype(np.complex128)
    basis = np.zeros_like(design)
    for index in range(n_modes):
        column = design[:, index].copy()
        for earlier in range(index):
            column = column - np.vdot(basis[:, earlier], column) * basis[:, earlier]
        norm = np.linalg.norm(column)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("mode construction produced a vanishing column.")
        column = column / norm
        pivot = int(np.argmax(np.abs(column)))
        column = column * np.exp(-1j * np.angle(column[pivot]))
        basis[:, index] = column
    return basis.T.reshape(n_modes, ny, nx)

import numpy as np


def coupled_residuals(modes: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(modes)
    if fields.ndim != 3 or min(fields.shape[1:]) < 2:
        raise ValueError("modes must have shape (n_modes, ny, nx) with ny, nx >= 2.")
    residuals = np.empty(fields.shape, dtype=np.complex128)
    for index in range(fields.shape[0]):
        left, values, right = np.linalg.svd(fields[index], full_matrices=False)
        separable = values[0] * np.outer(left[:, 0], right[0])
        residuals[index] = fields[index] - separable
    return residuals

import numpy as np


def residual_stack(
    residuals: "np.ndarray",
    eigenvalues: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(residuals)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    if fields.ndim != 3 or weights.ndim != 1 or weights.shape[0] != fields.shape[0]:
        raise ValueError("residuals and eigenvalues must describe the same modes.")
    if fields.shape[0] < 1 or min(fields.shape[1:]) < 2:
        raise ValueError("residuals must have shape (n_modes, ny, nx) with ny, nx >= 2.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("eigenvalues must be positive and finite.")
    rows = [weights[index] * fields[index].ravel() for index in range(fields.shape[0])]
    return np.vstack(rows)

import numpy as np


def compressed_residuals(
    stack: "np.ndarray",
    n_keep: int,
    field_shape: tuple,
) -> "np.ndarray":
    """Reference implementation."""
    matrix = np.asarray(stack)
    field_shape = tuple(int(value) for value in field_shape)
    if matrix.ndim != 2 or len(field_shape) != 2 or min(field_shape) < 2:
        raise ValueError("stack and field_shape must describe residual fields.")
    if matrix.shape[1] != field_shape[0] * field_shape[1]:
        raise ValueError("field_shape does not match the stacked row length.")
    keep = int(n_keep)
    rank = min(matrix.shape)
    if keep < 0 or keep > rank:
        raise ValueError("n_keep must lie between zero and the stack rank.")
    n_modes = matrix.shape[0]
    if keep == 0:
        return np.zeros((n_modes, field_shape[0], field_shape[1]), dtype=np.complex128)
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    approx = (left[:, :keep] * values[:keep]) @ right[:keep]
    return approx.reshape((n_modes, field_shape[0], field_shape[1]))

import numpy as np


def cumulative_energy_ratio(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
) -> float:
    """Reference implementation."""
    fields = np.asarray(modes)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    if fields.ndim != 3 or weights.ndim != 1 or weights.shape[0] != fields.shape[0]:
        raise ValueError("modes and eigenvalues must describe the same set of modes.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("eigenvalues must be positive and finite.")
    norms = np.linalg.norm(fields.reshape(fields.shape[0], -1), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("every mode must have unit Euclidean norm.")
    keep = int(n_keep)
    if keep < 0:
        raise ValueError("n_keep must be nonnegative.")
    leading = np.array(
        [np.linalg.svd(field, compute_uv=False)[0] for field in fields], dtype=np.float64
    )
    separable = float(np.sum(weights * leading ** 2) / np.sum(weights))
    if keep == 0:
        return separable
    residuals = coupled_residuals(fields)
    stack = residual_stack(residuals, weights)
    beta = np.linalg.svd(stack, compute_uv=False)
    if keep > beta.shape[0]:
        raise ValueError("n_keep exceeds the residual-stack rank.")
    total = float(np.sum(beta ** 2))
    if total == 0.0:
        return separable
    fraction = float(np.sum(beta[:keep] ** 2) / total)
    return float(separable + fraction * (1.0 - separable))

import numpy as np


def element_transmission(
    shape: tuple,
    widths: tuple,
    angle: float,
    phase_coupling: float,
) -> "np.ndarray":
    """Reference implementation."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two integers of at least 2.")
    half = np.asarray(widths, dtype=np.float64)
    if half.shape != (2,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
        raise ValueError("widths must be two positive finite half-widths.")
    theta = float(angle)
    kappa = float(phase_coupling)
    if not (np.isfinite(theta) and np.isfinite(kappa)):
        raise ValueError("angle and phase_coupling must be finite.")
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    rot_x = grid_x * np.cos(theta) + grid_y * np.sin(theta)
    rot_y = -grid_x * np.sin(theta) + grid_y * np.cos(theta)
    amplitude = np.exp(-((rot_x / half[0]) ** 4) - ((rot_y / half[1]) ** 4))
    return (amplitude * np.exp(1j * kappa * rot_x * rot_y)).astype(np.complex128)

import numpy as np


def element_terms(
    transmission: "np.ndarray",
    occupation: float,
) -> "np.ndarray":
    """Reference implementation."""
    field = np.asarray(transmission)
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("transmission must be a two-dimensional field.")
    target = float(occupation)
    if not np.isfinite(target) or target <= 0.0 or target > 1.0:
        raise ValueError("occupation must lie in (0, 1].")
    left, values, right = np.linalg.svd(field, full_matrices=False)
    total = float(np.sum(values ** 2))
    if total == 0.0:
        raise ValueError("transmission must not vanish identically.")
    shares = np.cumsum(values ** 2) / total
    reached = np.nonzero(shares >= target)[0]
    if target == 1.0:
        count = int(values.shape[0])
    else:
        count = int(reached[0]) + 1 if reached.size else int(values.shape[0])
    terms = np.empty((count, field.shape[0], field.shape[1]), dtype=np.complex128)
    for k in range(count):
        terms[k] = values[k] * np.outer(left[:, k], right[k])
    return terms

import numpy as np


def _paraxial_factor(length: int, fresnel_parameter: float) -> "np.ndarray":
    """One-dimensional paraxial transfer function on the DFT frequencies."""
    freq = np.fft.fftfreq(int(length), d=1.0)
    return np.exp(-1j * np.pi * float(fresnel_parameter) * freq * freq)


def _propagate_line(line: "np.ndarray", fresnel_parameter: float) -> "np.ndarray":
    """Propagate a one-dimensional complex profile."""
    spectrum = np.fft.fft(np.asarray(line, dtype=np.complex128))
    return np.fft.ifft(spectrum * _paraxial_factor(line.shape[0], fresnel_parameter))


def _propagate_field(field: "np.ndarray", fresnel_parameter: float) -> "np.ndarray":
    """Propagate a two-dimensional complex field."""
    data = np.asarray(field, dtype=np.complex128)
    kernel = np.outer(
        _paraxial_factor(data.shape[0], fresnel_parameter),
        _paraxial_factor(data.shape[1], fresnel_parameter),
    )
    return np.fft.ifft2(np.fft.fft2(data) * kernel)


def separable_transport(
    modes: "np.ndarray",
    terms: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(modes)
    pieces = np.asarray(terms)
    if fields.ndim != 3 or pieces.ndim != 3 or fields.shape[1:] != pieces.shape[1:]:
        raise ValueError("modes and terms must share the same spatial shape.")
    if min(fields.shape[1:]) < 2:
        raise ValueError("both spatial sides must be at least 2.")
    parameter = float(fresnel_parameter)
    if not np.isfinite(parameter):
        raise ValueError("fresnel_parameter must be finite.")
    factors = []
    for piece in pieces:
        left, values, right = np.linalg.svd(piece, full_matrices=False)
        factors.append((values[0] * left[:, 0], right[0]))
    transported = np.zeros(fields.shape, dtype=np.complex128)
    for index, field in enumerate(fields):
        left, values, right = np.linalg.svd(field, full_matrices=False)
        vertical = values[0] * left[:, 0]
        horizontal = right[0]
        for term_vertical, term_horizontal in factors:
            moved_vertical = _propagate_line(vertical * term_vertical, parameter)
            moved_horizontal = _propagate_line(horizontal * term_horizontal, parameter)
            transported[index] += np.outer(moved_vertical, moved_horizontal)
    return transported

import numpy as np


def coupled_transport(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
    transmission: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(modes)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    element = np.asarray(transmission)
    if fields.ndim != 3 or weights.shape != (fields.shape[0],):
        raise ValueError("modes and eigenvalues must describe the same set of modes.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("eigenvalues must be positive and finite.")
    if element.shape != fields.shape[1:]:
        raise ValueError("transmission must match the spatial shape of the modes.")
    parameter = float(fresnel_parameter)
    if not np.isfinite(parameter):
        raise ValueError("fresnel_parameter must be finite.")
    residuals = coupled_residuals(fields)
    stack = residual_stack(residuals, weights)
    compressed = compressed_residuals(stack, n_keep, fields.shape[1:])
    restored = compressed / weights[:, None, None]
    transported = np.zeros(fields.shape, dtype=np.complex128)
    for index in range(fields.shape[0]):
        transported[index] = _propagate_field(restored[index] * element, parameter)
    return transported

import numpy as np


def transport_intensity_error_percent(
    shape: tuple,
    sigmas: "np.ndarray",
    tilts: "np.ndarray",
    coupling: "np.ndarray",
    eigenvalues: "np.ndarray",
    energy_threshold: float,
    widths: tuple,
    angle: float,
    phase_coupling: float,
    occupation: float,
    fresnel_parameter: float,
) -> float:
    """Reference implementation."""
    target = float(energy_threshold)
    if not np.isfinite(target) or target <= 0.0 or target > 1.0:
        raise ValueError("energy_threshold must lie in (0, 1].")
    modes = orthonormal_source_modes(shape, sigmas, tilts, coupling)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    if weights.shape != (modes.shape[0],):
        raise ValueError("eigenvalues must match the number of modes.")
    rank = min(modes.shape[0], modes.shape[1] * modes.shape[2])
    n_keep = rank
    for count in range(rank + 1):
        if cumulative_energy_ratio(modes, weights, count) >= target:
            n_keep = count
            break
    transmission = element_transmission(shape, widths, angle, phase_coupling)
    terms = element_terms(transmission, occupation)
    separable = separable_transport(modes, terms, fresnel_parameter)
    coupled = coupled_transport(
        modes, weights, n_keep, transmission, fresnel_parameter
    )
    compressed = separable + coupled
    exact = np.zeros(modes.shape, dtype=np.complex128)
    for index in range(modes.shape[0]):
        exact[index] = _propagate_field(modes[index] * transmission, fresnel_parameter)
    intensity_compressed = np.sum(weights[:, None, None] * np.abs(compressed) ** 2, axis=0)
    intensity_exact = np.sum(weights[:, None, None] * np.abs(exact) ** 2, axis=0)
    difference = np.linalg.norm(intensity_compressed - intensity_exact)
    return float(100.0 * difference / np.linalg.norm(intensity_exact))
SCICODE_GOLD_EOF
