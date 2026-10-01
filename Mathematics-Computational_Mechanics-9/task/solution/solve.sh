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

_LATTICE_TOLERANCE = 1e-9


def initialize_material_points(domain: np.ndarray, point_spacing: float):
    """Reference implementation."""
    domain = np.asarray(domain, dtype=float)
    if domain.shape != (2,) or np.any(domain <= 0.0):
        raise ValueError("domain must hold two positive side lengths")
    spacing = float(point_spacing)
    if spacing <= 0.0:
        raise ValueError("point_spacing must be > 0")
    counts = domain / spacing
    if np.any(np.abs(counts - np.rint(counts)) > _LATTICE_TOLERANCE):
        raise ValueError("domain must be an integer multiple of point_spacing")
    n_x = round(float(counts[0]))
    n_y = round(float(counts[1]))
    xs = (np.arange(n_x) + 0.5) * spacing
    ys = (np.arange(n_y) + 0.5) * spacing
    points = np.stack([np.tile(xs, n_y), np.repeat(ys, n_x)], axis=1)
    volumes = np.full(n_x * n_y, spacing * spacing)
    return points, volumes

import numpy as np


def _gimp_factor(delta, cell_size, point_domain_length):
    """One-dimensional generalized interpolation weight and its derivative."""
    h = cell_size
    lp = point_domain_length
    reach = h + 0.5 * lp
    value = np.zeros_like(delta)
    slope = np.zeros_like(delta)
    left_edge = (delta > -reach) & (delta <= -h + 0.5 * lp)
    left_hat = (delta > -h + 0.5 * lp) & (delta <= -0.5 * lp)
    centre = (delta > -0.5 * lp) & (delta <= 0.5 * lp)
    right_hat = (delta > 0.5 * lp) & (delta <= h - 0.5 * lp)
    right_edge = (delta > h - 0.5 * lp) & (delta <= reach)
    value[left_edge] = (reach + delta[left_edge]) ** 2 / (2.0 * h * lp)
    slope[left_edge] = (reach + delta[left_edge]) / (h * lp)
    value[left_hat] = 1.0 + delta[left_hat] / h
    slope[left_hat] = 1.0 / h
    value[centre] = 1.0 - (delta[centre] ** 2 + 0.25 * lp ** 2) / (h * lp)
    slope[centre] = -2.0 * delta[centre] / (h * lp)
    value[right_hat] = 1.0 - delta[right_hat] / h
    slope[right_hat] = -1.0 / h
    value[right_edge] = (reach - delta[right_edge]) ** 2 / (2.0 * h * lp)
    slope[right_edge] = -(reach - delta[right_edge]) / (h * lp)
    return value, slope


def gimp_shape_functions(points: np.ndarray, nodes: np.ndarray, cell_size: float,
                                 point_domain_length: float):
    """Reference implementation."""
    points = np.asarray(points, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    h = float(cell_size)
    lp = float(point_domain_length)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")
    if h <= 0.0:
        raise ValueError("cell_size must be > 0")
    if not 0.0 < lp <= h:
        raise ValueError("point_domain_length must lie in (0, cell_size]")
    value_x, slope_x = _gimp_factor(points[:, None, 0] - nodes[None, :, 0], h, lp)
    value_y, slope_y = _gimp_factor(points[:, None, 1] - nodes[None, :, 1], h, lp)
    shape_values = value_x * value_y
    shape_gradients = np.stack([slope_x * value_y, value_x * slope_y], axis=-1)
    return shape_values, shape_gradients

import numpy as np


def simp_lame_parameters(densities: np.ndarray, youngs_modulus: float,
                                 poisson_ratio: float, penalty: float) -> np.ndarray:
    """Reference implementation."""
    densities = np.asarray(densities, dtype=float)
    modulus = float(youngs_modulus)
    ratio = float(poisson_ratio)
    exponent = float(penalty)
    if densities.ndim != 1 or densities.size == 0:
        raise ValueError("densities must be a non-empty one-dimensional array")
    if np.any(densities <= 0.0) or np.any(densities > 1.0):
        raise ValueError("densities must lie in (0, 1]")
    if modulus <= 0.0:
        raise ValueError("youngs_modulus must be > 0")
    if not -1.0 < ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    if exponent < 1.0:
        raise ValueError("penalty must be >= 1")
    lambda_solid = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
    mu_solid = modulus / (2.0 * (1.0 + ratio))
    scale = densities ** exponent
    return np.stack([scale * lambda_solid, scale * mu_solid], axis=1)

import numpy as np


def _logarithmic_strain(deformation_gradient):
    """Half the matrix logarithm of the left Cauchy-Green tensor."""
    eigenvalues, eigenvectors = np.linalg.eigh(deformation_gradient @ deformation_gradient.T)
    return 0.5 * (eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T)


def hencky_cauchy_stress(deformation_gradients: np.ndarray, lame: np.ndarray):
    """Reference implementation."""
    gradients = np.asarray(deformation_gradients, dtype=float)
    lame = np.asarray(lame, dtype=float)
    if gradients.ndim != 3 or gradients.shape[1:] != (2, 2):
        raise ValueError("deformation_gradients must have shape (n_points, 2, 2)")
    if lame.shape != (gradients.shape[0], 2):
        raise ValueError("lame must have shape (n_points, 2)")
    if np.any(lame[:, 1] <= 0.0) or np.any(lame[:, 0] + 2.0 * lame[:, 1] <= 0.0):
        raise ValueError("lame parameters must satisfy mu > 0 and lambda + 2 mu > 0")
    n_points = gradients.shape[0]
    cauchy_stress = np.empty((n_points, 2, 2))
    jacobian = np.empty(n_points)
    identity = np.eye(2)
    for p in range(n_points):
        determinant = float(np.linalg.det(gradients[p]))
        if determinant <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        lambda_p = float(lame[p, 0])
        mu_p = float(lame[p, 1])
        strain = _logarithmic_strain(gradients[p])
        in_plane_trace = strain[0, 0] + strain[1, 1]
        strain_zz = -lambda_p / (lambda_p + 2.0 * mu_p) * in_plane_trace
        kirchhoff = lambda_p * (in_plane_trace + strain_zz) * identity + 2.0 * mu_p * strain
        cauchy_stress[p] = kirchhoff / determinant
        jacobian[p] = determinant
    return cauchy_stress, jacobian

import numpy as np


def assemble_nodal_forces(cauchy_stress: np.ndarray, deformation_gradients: np.ndarray,
                                  volumes: np.ndarray, shape_values: np.ndarray,
                                  shape_gradients: np.ndarray, point_loads: np.ndarray):
    """Reference implementation."""
    stress = np.asarray(cauchy_stress, dtype=float)
    gradients = np.asarray(deformation_gradients, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    values = np.asarray(shape_values, dtype=float)
    reference_gradients = np.asarray(shape_gradients, dtype=float)
    loads = np.asarray(point_loads, dtype=float)
    if values.ndim != 2:
        raise ValueError("shape_values must have shape (n_points, n_nodes)")
    n_points, n_nodes = values.shape
    if stress.shape != (n_points, 2, 2) or gradients.shape != (n_points, 2, 2):
        raise ValueError("stress and deformation gradient arrays must have shape (n_points, 2, 2)")
    if volumes.shape != (n_points,) or loads.shape != (n_points, 2):
        raise ValueError("volumes and point_loads must be sized by n_points")
    if reference_gradients.shape != (n_points, n_nodes, 2):
        raise ValueError("shape_gradients must have shape (n_points, n_nodes, 2)")
    internal_forces = np.zeros((n_nodes, 2))
    for p in range(n_points):
        determinant = float(np.linalg.det(gradients[p]))
        if determinant <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        spatial_gradients = reference_gradients[p] @ np.linalg.inv(gradients[p])
        internal_forces += (determinant * volumes[p]) * (spatial_gradients @ stress[p].T)
    return internal_forces, values.T @ loads

import numpy as np

_EIGENVALUE_TOLERANCE = 1e-12


def _divided_difference_matrix(eigenvalues):
    """Divided differences of the logarithm, with the repeated-eigenvalue limit."""
    size = eigenvalues.shape[0]
    matrix = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            if abs(eigenvalues[i] - eigenvalues[j]) < _EIGENVALUE_TOLERANCE:
                matrix[i, j] = 1.0 / eigenvalues[i]
            else:
                matrix[i, j] = ((np.log(eigenvalues[i]) - np.log(eigenvalues[j]))
                                / (eigenvalues[i] - eigenvalues[j]))
    return matrix


def _first_piola_tangent(deformation_gradient, lambda_p, mu_p):
    """Derivative of the first Piola-Kirchhoff stress with respect to F."""
    left_cauchy_green = deformation_gradient @ deformation_gradient.T
    eigenvalues, eigenvectors = np.linalg.eigh(left_cauchy_green)
    divided = _divided_difference_matrix(eigenvalues)
    strain = 0.5 * (eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T)
    identity = np.eye(2)
    in_plane_trace = strain[0, 0] + strain[1, 1]
    strain_zz = -lambda_p / (lambda_p + 2.0 * mu_p) * in_plane_trace
    kirchhoff = lambda_p * (in_plane_trace + strain_zz) * identity + 2.0 * mu_p * strain
    lambda_plane_stress = 2.0 * lambda_p * mu_p / (lambda_p + 2.0 * mu_p)
    inverse = np.linalg.inv(deformation_gradient)
    tangent = np.empty((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            direction = np.zeros((2, 2))
            direction[a, b] = 1.0
            perturbation = direction @ deformation_gradient.T + deformation_gradient @ direction.T
            strain_rate = 0.5 * (eigenvectors
                                 @ (divided * (eigenvectors.T @ perturbation @ eigenvectors))
                                 @ eigenvectors.T)
            kirchhoff_rate = (lambda_plane_stress * (strain_rate[0, 0] + strain_rate[1, 1]) * identity
                              + 2.0 * mu_p * strain_rate)
            tangent[:, :, a, b] = (kirchhoff_rate @ inverse.T
                                   - np.outer(kirchhoff @ inverse[b, :], inverse[:, a]))
    return tangent


def assemble_tangent_stiffness(deformation_gradients: np.ndarray, lame: np.ndarray,
                                       volumes: np.ndarray, shape_gradients: np.ndarray,
                                       fixed_nodes: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    gradients = np.asarray(deformation_gradients, dtype=float)
    lame = np.asarray(lame, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    reference_gradients = np.asarray(shape_gradients, dtype=float)
    fixed = np.asarray(fixed_nodes, dtype=bool)
    if reference_gradients.ndim != 3 or reference_gradients.shape[2] != 2:
        raise ValueError("shape_gradients must have shape (n_points, n_nodes, 2)")
    n_points, n_nodes = reference_gradients.shape[0], reference_gradients.shape[1]
    if gradients.shape != (n_points, 2, 2) or lame.shape != (n_points, 2):
        raise ValueError("deformation_gradients and lame must be sized by n_points")
    if volumes.shape != (n_points,):
        raise ValueError("volumes must have shape (n_points,)")
    if fixed.shape != (n_nodes,):
        raise ValueError("fixed_nodes must have shape (n_nodes,)")
    stiffness = np.zeros((2 * n_nodes, 2 * n_nodes))
    for p in range(n_points):
        if float(np.linalg.det(gradients[p])) <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        tangent = _first_piola_tangent(gradients[p], float(lame[p, 0]), float(lame[p, 1]))
        stiffness += volumes[p] * np.einsum(
            'ijab,wb,vj->viwa', tangent, reference_gradients[p], reference_gradients[p]
        ).reshape(2 * n_nodes, 2 * n_nodes)
    constrained = np.repeat(fixed, 2)
    stiffness[constrained, :] = 0.0
    stiffness[:, constrained] = 0.0
    stiffness[constrained, constrained] = 1.0
    return stiffness

import numpy as np


def compliance_and_adjoint(stiffness: np.ndarray, external_forces: np.ndarray,
                                   displacement: np.ndarray, fixed_nodes: np.ndarray):
    """Reference implementation."""
    stiffness = np.asarray(stiffness, dtype=float)
    forces = np.asarray(external_forces, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    fixed = np.asarray(fixed_nodes, dtype=bool)
    if forces.ndim != 2 or forces.shape[1] != 2:
        raise ValueError("external_forces must have shape (n_nodes, 2)")
    n_nodes = forces.shape[0]
    if displacement.shape != forces.shape:
        raise ValueError("displacement must have the same shape as external_forces")
    if stiffness.shape != (2 * n_nodes, 2 * n_nodes):
        raise ValueError("stiffness must have shape (2 n_nodes, 2 n_nodes)")
    if fixed.shape != (n_nodes,):
        raise ValueError("fixed_nodes must have shape (n_nodes,)")
    right_hand_side = forces.ravel().copy()
    right_hand_side[np.repeat(fixed, 2)] = 0.0
    adjoint = np.linalg.solve(stiffness.T, right_hand_side).reshape(n_nodes, 2)
    compliance = float(np.sum(forces * displacement))
    return compliance, adjoint

import numpy as np


def density_sensitivity(adjoint: np.ndarray, cauchy_stress: np.ndarray,
                                deformation_gradients: np.ndarray, volumes: np.ndarray,
                                shape_gradients: np.ndarray, densities: np.ndarray,
                                penalty: float) -> np.ndarray:
    """Reference implementation."""
    adjoint = np.asarray(adjoint, dtype=float)
    stress = np.asarray(cauchy_stress, dtype=float)
    gradients = np.asarray(deformation_gradients, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    reference_gradients = np.asarray(shape_gradients, dtype=float)
    densities = np.asarray(densities, dtype=float)
    exponent = float(penalty)
    if reference_gradients.ndim != 3 or reference_gradients.shape[2] != 2:
        raise ValueError("shape_gradients must have shape (n_points, n_nodes, 2)")
    n_points, n_nodes = reference_gradients.shape[0], reference_gradients.shape[1]
    if adjoint.shape != (n_nodes, 2):
        raise ValueError("adjoint must have shape (n_nodes, 2)")
    if stress.shape != (n_points, 2, 2) or gradients.shape != (n_points, 2, 2):
        raise ValueError("stress and deformation gradient arrays must have shape (n_points, 2, 2)")
    if volumes.shape != (n_points,) or densities.shape != (n_points,):
        raise ValueError("volumes and densities must have shape (n_points,)")
    if np.any(densities <= 0.0):
        raise ValueError("densities must be strictly positive")
    if exponent < 1.0:
        raise ValueError("penalty must be >= 1")
    sensitivity = np.empty(n_points)
    for p in range(n_points):
        determinant = float(np.linalg.det(gradients[p]))
        if determinant <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        spatial_gradients = reference_gradients[p] @ np.linalg.inv(gradients[p])
        contribution = (determinant * volumes[p]) * (spatial_gradients @ stress[p].T)
        sensitivity[p] = -(exponent / densities[p]) * float(np.sum(adjoint * contribution))
    return sensitivity

import numpy as np

_DOMAIN = np.array([2.0, 1.0])
_POINT_SPACING = 0.2
_CELL_SIZE = 0.5
_POINT_DOMAIN_LENGTH = 0.2
_YOUNGS_MODULUS = 1.0
_POISSON_RATIO = 0.3
_NEWTON_TOLERANCE = 1e-10
_MAX_NEWTON_ITERATIONS = 50


def run_full_pipeline(total_load: float = 0.006, penalty: float = 3.0) -> float:
    """Reference implementation chaining every earlier step."""
    try:
        load = float(total_load)
        exponent = float(penalty)
    except (TypeError, ValueError):
        raise ValueError("total_load and penalty must be real numbers") from None
    if load <= 0.0:
        raise ValueError("total_load must be > 0")
    if exponent < 1.0:
        raise ValueError("penalty must be >= 1")

    points, volumes = initialize_material_points(_DOMAIN, _POINT_SPACING)
    n_x = round(float(_DOMAIN[0] / _CELL_SIZE)) + 1
    n_y = round(float(_DOMAIN[1] / _CELL_SIZE)) + 1
    node_x = np.arange(n_x) * _CELL_SIZE
    node_y = np.arange(n_y) * _CELL_SIZE
    nodes = np.stack([np.tile(node_x, n_y), np.repeat(node_y, n_x)], axis=1)
    shape_values, shape_gradients = gimp_shape_functions(
        points, nodes, _CELL_SIZE, _POINT_DOMAIN_LENGTH)

    densities = 0.3 + np.abs(points[:, 1] - 0.5)
    lame = simp_lame_parameters(densities, _YOUNGS_MODULUS, _POISSON_RATIO, exponent)

    loaded = np.abs(points[:, 0] - points[:, 0].max()) < 1e-12
    point_loads = np.zeros((points.shape[0], 2))
    point_loads[loaded, 1] = -load / int(loaded.sum())
    fixed_nodes = np.abs(nodes[:, 0]) < 1e-12
    constrained = np.repeat(fixed_nodes, 2)

    displacement = np.zeros((nodes.shape[0], 2))
    identity = np.eye(2)[None, :, :]
    for _ in range(_MAX_NEWTON_ITERATIONS):
        gradients = identity + np.einsum('vi,pvj->pij', displacement, shape_gradients)
        cauchy_stress, _jacobian = hencky_cauchy_stress(gradients, lame)
        internal_forces, external_forces = assemble_nodal_forces(
            cauchy_stress, gradients, volumes, shape_values, shape_gradients, point_loads)
        residual = internal_forces - external_forces
        residual.ravel()[constrained] = 0.0
        if float(np.linalg.norm(residual)) < _NEWTON_TOLERANCE:
            break
        stiffness = assemble_tangent_stiffness(
            gradients, lame, volumes, shape_gradients, fixed_nodes)
        correction = np.linalg.solve(stiffness, -residual.ravel())
        displacement = displacement + correction.reshape(nodes.shape[0], 2)

    stiffness = assemble_tangent_stiffness(
        gradients, lame, volumes, shape_gradients, fixed_nodes)
    _compliance, adjoint = compliance_and_adjoint(
        stiffness, external_forces, displacement, fixed_nodes)
    sensitivity = density_sensitivity(
        adjoint, cauchy_stress, gradients, volumes, shape_gradients, densities, exponent)
    return float(sensitivity.min())
SCICODE_GOLD_EOF
