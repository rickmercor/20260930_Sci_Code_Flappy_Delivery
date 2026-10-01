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


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def ao_overlap_matrix(ao_centers: "np.ndarray",
                              ao_exponents: "np.ndarray") -> "np.ndarray":
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    separation2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    total = exponents[:, None] + exponents[None, :]
    product = exponents[:, None] * exponents[None, :]
    overlap = (2.0 * np.sqrt(product) / total) ** 1.5 * np.exp(-product / total * separation2)
    return overlap

import numpy as np
from scipy.special import erf


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def _boys0(argument):
    """Zeroth-order Boys function, taking the removable limit one at the origin."""
    argument = np.asarray(argument, dtype=float)
    value = np.ones_like(argument)
    large = argument > 1e-12
    scaled = argument[large]
    value[large] = 0.5 * np.sqrt(np.pi / scaled) * erf(np.sqrt(scaled))
    return value


def ao_eri_tensor(ao_centers: "np.ndarray",
                          ao_exponents: "np.ndarray") -> "np.ndarray":
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    n_ao = centers.shape[0]
    norm = (2.0 * exponents / np.pi) ** 0.75

    total = exponents[:, None] + exponents[None, :]
    separation2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    prefactor = np.exp(-(exponents[:, None] * exponents[None, :]) / total * separation2)
    midpoint = (exponents[:, None, None] * centers[:, None, :]
                + exponents[None, :, None] * centers[None, :, :]) / total[:, :, None]

    pair_exponent = total.reshape(-1)
    pair_prefactor = (prefactor * norm[:, None] * norm[None, :]).reshape(-1)
    pair_center = midpoint.reshape(-1, 3)

    separation_pq2 = np.sum((pair_center[:, None, :] - pair_center[None, :, :]) ** 2, axis=-1)
    exponent_sum = pair_exponent[:, None] + pair_exponent[None, :]
    rho = (pair_exponent[:, None] * pair_exponent[None, :]) / exponent_sum
    eri = (2.0 * np.pi ** 2.5
           / (pair_exponent[:, None] * pair_exponent[None, :] * np.sqrt(exponent_sum))
           * pair_prefactor[:, None] * pair_prefactor[None, :]
           * _boys0(rho * separation_pq2))
    return eri.reshape(n_ao, n_ao, n_ao, n_ao)

import numpy as np


def _octahedral_directions():
    """Unit vectors +x, -x, +y, -y, +z, -z in that order."""
    return np.array([[1.0, 0.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, -1.0]], dtype=float)


def atom_centered_grid(centers: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("centers must have shape (n_centers, 3) with n_centers >= 1")
    if radii.ndim != 1 or radii.shape[0] < 1:
        raise ValueError("radii must have shape (n_radii,) with n_radii >= 1")
    if not np.all(radii > 0.0):
        raise ValueError("every shell radius must be strictly positive")

    shell = (radii[:, None, None] * _octahedral_directions()[None, :, :]).reshape(-1, 3)
    offsets = np.vstack([np.zeros((1, 3), dtype=float), shell])
    grid_points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return grid_points

import numpy as np


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def _evaluate_basis(ao_centers, ao_exponents, grid_points):
    """Values of the normalized s primitives at the grid points, shape (n_ao, n_grid)."""
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    points = np.asarray(grid_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("grid_points must have shape (n_grid, 3) with n_grid >= 1")
    norm = (2.0 * exponents / np.pi) ** 0.75
    distance2 = np.sum((points[None, :, :] - centers[:, None, :]) ** 2, axis=-1)
    return norm[:, None] * np.exp(-exponents[:, None] * distance2)


def overlap_fit_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                               grid_points: "np.ndarray") -> "np.ndarray":
    values = _evaluate_basis(ao_centers, ao_exponents, grid_points)
    n_ao, n_grid = values.shape
    fit_matrix = (values[:, None, :] * values[None, :, :]).reshape(n_ao * n_ao, n_grid)
    return fit_matrix

import numpy as np
from scipy.optimize import nnls


def nnls_grid_weights(fit_matrix: "np.ndarray",
                              overlap: "np.ndarray") -> "np.ndarray":
    fit_matrix = np.asarray(fit_matrix, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    if fit_matrix.ndim != 2:
        raise ValueError("fit_matrix must be a two-dimensional array")
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be a square two-dimensional array")
    if fit_matrix.shape[0] != overlap.size:
        raise ValueError("fit_matrix must have one row per entry of the overlap matrix")
    n_grid = fit_matrix.shape[1]
    weights, _ = nnls(fit_matrix, overlap.reshape(-1), maxiter=max(1000, 50 * n_grid))
    return weights

import numpy as np


def prune_zero_weight_points(grid_points: "np.ndarray",
                                     weights: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    points = np.asarray(grid_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("grid_points must have shape (n_grid, 3)")
    if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
        raise ValueError("weights must have shape (n_grid,) matching grid_points")
    if np.any(weights < 0.0):
        raise ValueError("quadrature weights must be non-negative")
    keep = weights > 0.0
    return points[keep].copy(), weights[keep].copy()

import numpy as np


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def _evaluate_basis(ao_centers, ao_exponents, grid_points):
    """Values of the normalized s primitives at the grid points, shape (n_ao, n_grid)."""
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    points = np.asarray(grid_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("grid_points must have shape (n_grid, 3) with n_grid >= 1")
    norm = (2.0 * exponents / np.pi) ** 0.75
    distance2 = np.sum((points[None, :, :] - centers[:, None, :]) ** 2, axis=-1)
    return norm[:, None] * np.exp(-exponents[:, None] * distance2)


def thc_collocation_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                                   grid_points: "np.ndarray",
                                   weights: "np.ndarray") -> "np.ndarray":
    values = _evaluate_basis(ao_centers, ao_exponents, grid_points)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != values.shape[1]:
        raise ValueError("weights must have shape (n_grid,) matching grid_points")
    if np.any(weights < 0.0):
        raise ValueError("quadrature weights must be non-negative")
    return values * weights[None, :] ** 0.25

import numpy as np


def thc_grid_metric(collocation: "np.ndarray") -> "np.ndarray":
    collocation = np.asarray(collocation, dtype=float)
    if collocation.ndim != 2 or collocation.shape[0] < 1 or collocation.shape[1] < 1:
        raise ValueError("collocation must have shape (n_ao, n_grid) with both sizes >= 1")
    gram = collocation.T @ collocation
    return gram * gram

import numpy as np


def pivoted_cholesky_points(grid_metric: "np.ndarray", cutoff: float) -> "np.ndarray":
    metric = np.asarray(grid_metric, dtype=float)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1] or metric.shape[0] < 1:
        raise ValueError("grid_metric must be a square two-dimensional array")
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and strictly positive")

    n_grid = metric.shape[0]
    residual = np.diag(metric).copy()
    threshold = cutoff * np.sqrt(max(residual.max(), 0.0))
    factor = np.zeros((n_grid, n_grid), dtype=float)
    selected = np.zeros(n_grid, dtype=bool)
    pivots = []
    for step in range(n_grid):
        candidates = np.where(selected, -np.inf, residual)
        pivot = int(np.argmax(candidates))
        diagonal = candidates[pivot]
        if not diagonal > 0.0 or np.sqrt(diagonal) < threshold:
            break
        column = (metric[:, pivot] - factor[:, :step] @ factor[pivot, :step]) / np.sqrt(diagonal)
        factor[:, step] = column
        residual = residual - column ** 2
        selected[pivot] = True
        pivots.append(pivot)
    return np.array(pivots, dtype=int)

import numpy as np


def _truncated_pseudoinverse(matrix, rcond):
    """Moore-Penrose pseudoinverse keeping nonzero singular values >= rcond * largest."""
    left, singular, right = np.linalg.svd(matrix)
    keep = (singular >= rcond * singular[0]) & (singular > 0.0)
    return (right[keep].T / singular[keep]) @ left[:, keep].T


def thc_core_matrix(collocation: "np.ndarray", grid_metric: "np.ndarray",
                            eri_tensor: "np.ndarray", rcond: float) -> "np.ndarray":
    collocation = np.asarray(collocation, dtype=float)
    grid_metric = np.asarray(grid_metric, dtype=float)
    eri_tensor = np.asarray(eri_tensor, dtype=float)
    if collocation.ndim != 2 or collocation.shape[0] < 1 or collocation.shape[1] < 1:
        raise ValueError("collocation must have shape (n_ao, n_grid) with both sizes >= 1")
    n_ao, n_grid = collocation.shape
    if grid_metric.shape != (n_grid, n_grid):
        raise ValueError("grid_metric must have shape (n_grid, n_grid)")
    if eri_tensor.shape != (n_ao, n_ao, n_ao, n_ao):
        raise ValueError("eri_tensor must have shape (n_ao, n_ao, n_ao, n_ao)")
    rcond = float(rcond)
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and non-negative")

    pair_density = (collocation[:, None, :] * collocation[None, :, :]).reshape(n_ao * n_ao,
                                                                              n_grid)
    projected = pair_density.T @ eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao) @ pair_density
    metric_inverse = _truncated_pseudoinverse(grid_metric, rcond)
    return metric_inverse @ projected @ metric_inverse

import numpy as np


def _expand_basis(centers, exponents):
    """Per-orbital centers and exponents, exponent index fastest."""
    centers = np.asarray(centers, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("centers must have shape (n_centers, 3) with n_centers >= 1")
    if exponents.ndim != 1 or exponents.shape[0] < 1:
        raise ValueError("exponents must have shape (n_exponents,) with n_exponents >= 1")
    return (np.repeat(centers, exponents.shape[0], axis=0),
            np.tile(exponents, centers.shape[0]))


def _thc_reconstruction_rmsd(ao_centers, ao_exponents, points, weights, eri, rcond):
    """RMSD per entry of the LS-THC reconstruction of eri on a weighted grid."""
    collocation = thc_collocation_matrix(ao_centers, ao_exponents, points, weights)
    metric = thc_grid_metric(collocation)
    core = thc_core_matrix(collocation, metric, eri, rcond)
    n_ao = collocation.shape[0]
    pair_density = (collocation[:, None, :] * collocation[None, :, :]).reshape(n_ao * n_ao, -1)
    reconstructed = pair_density @ core @ pair_density.T
    error = eri.reshape(n_ao * n_ao, n_ao * n_ao) - reconstructed
    return float(np.linalg.norm(error) / n_ao ** 2)


def pruning_rmsd_ratio(centers: "np.ndarray", exponents: "np.ndarray",
                               radii: "np.ndarray", cutoff: float, rcond: float) -> float:
    cutoff = float(cutoff)
    if not (0.0 < cutoff <= 1.0):
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1")
    ao_centers, ao_exponents = _expand_basis(centers, exponents)
    overlap = ao_overlap_matrix(ao_centers, ao_exponents)
    eri = ao_eri_tensor(ao_centers, ao_exponents)
    grid_points = atom_centered_grid(centers, radii)

    fit_matrix = overlap_fit_matrix(ao_centers, ao_exponents, grid_points)
    weights = nnls_grid_weights(fit_matrix, overlap)
    nnls_points, nnls_weights = prune_zero_weight_points(grid_points, weights)
    nnls_rmsd = _thc_reconstruction_rmsd(ao_centers, ao_exponents, nnls_points, nnls_weights,
                                         eri, rcond)

    unit_weights = np.ones(grid_points.shape[0], dtype=float)
    input_collocation = thc_collocation_matrix(ao_centers, ao_exponents, grid_points,
                                                       unit_weights)
    pivots = pivoted_cholesky_points(thc_grid_metric(input_collocation), cutoff)
    cholesky_rmsd = _thc_reconstruction_rmsd(ao_centers, ao_exponents, grid_points[pivots],
                                             unit_weights[pivots], eri, rcond)
    return float(cholesky_rmsd / nnls_rmsd)
SCICODE_GOLD_EOF
