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


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mesh_arrays(reference, cells):
    points = _finite_array(reference)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("reference must have shape (n_nodes, 3), n_nodes >= 4")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("cells must have shape (n_elements, 4)")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cell indices must be integers")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("cell index out of range")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("each cell needs four distinct vertices")
    edges = (points[cells[:, 1:]] - points[cells[:, :1]]).transpose(0, 2, 1)
    lengths = np.linalg.norm(edges, axis=1)
    scale = np.prod(lengths, axis=1)
    determinant = np.linalg.det(edges)
    if np.any(scale == 0) or np.any(np.abs(determinant) <= 1e-12 * scale):
        raise ValueError(
            "reference tetrahedra are degenerate at relative tolerance 1e-12"
        )
    return points, cells.astype(int), edges


def reference_geometry(
    reference: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    _, _, edges = _mesh_arrays(reference, cells)
    inverse = np.linalg.inv(edges)
    volumes = np.abs(np.linalg.det(edges)) / 6.0
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    if not np.all(np.isfinite(volumes)) or not np.all(np.isfinite(gradients)):
        raise ValueError("reference geometry exceeds finite arithmetic range")
    return volumes, gradients

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mesh_arrays(reference, cells):
    points = _finite_array(reference)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("reference must have shape (n_nodes, 3), n_nodes >= 4")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("cells must have shape (n_elements, 4)")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cell indices must be integers")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("cell index out of range")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("each cell needs four distinct vertices")
    edges = (points[cells[:, 1:]] - points[cells[:, :1]]).transpose(0, 2, 1)
    lengths = np.linalg.norm(edges, axis=1)
    scale = np.prod(lengths, axis=1)
    determinant = np.linalg.det(edges)
    if np.any(scale == 0) or np.any(np.abs(determinant) <= 1e-12 * scale):
        raise ValueError(
            "reference tetrahedra are degenerate at relative tolerance 1e-12"
        )
    return points, cells.astype(int), edges


def _strain_mandel_vector(tensor):
    root_two = np.sqrt(2.0)
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            root_two * tensor[1, 2],
            root_two * tensor[0, 2],
            root_two * tensor[0, 1],
        ]
    )


def corotational_kinematics(
    reference: np.ndarray,
    deformed: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    points, indices, edges = _mesh_arrays(reference, cells)
    current = _finite_array(deformed)
    if current.shape != points.shape:
        raise ValueError("deformed coordinates must match reference shape")
    spatial_edges = (current[indices[:, 1:]] - current[indices[:, :1]]).transpose(
        0, 2, 1
    )
    deform = spatial_edges @ np.linalg.inv(edges)
    if np.any(np.linalg.det(deform) <= 0):
        raise ValueError("deformation must preserve orientation")
    left, singular, right = np.linalg.svd(deform)
    rotation = left @ right
    stretch = (right.swapaxes(1, 2) * singular[:, None, :]) @ right
    strain = stretch - np.eye(3)
    deviator = (
        strain - np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    )
    equivalent = np.sqrt(2.0 / 3.0 * np.sum(deviator**2, axis=(1, 2)))
    tiny = equivalent < 1e-12
    equivalent[tiny] = 0.0
    deviator[tiny] = 0.0
    strain_jacobian = np.empty((len(deform), 6, 9))
    for element in range(len(deform)):
        eigenvalues, eigenvectors = np.linalg.eigh(stretch[element])
        for component in range(9):
            perturbation = np.zeros((3, 3))
            perturbation.flat[component] = 1.0
            local = rotation[element].T @ perturbation
            local_eigen = eigenvectors.T @ local @ eigenvectors
            spin_eigen = (local_eigen - local_eigen.T) / (
                eigenvalues[:, None] + eigenvalues[None, :]
            )
            np.fill_diagonal(spin_eigen, 0.0)
            spin = eigenvectors @ spin_eigen @ eigenvectors.T
            stretch_increment = local - spin @ stretch[element]
            stretch_increment = 0.5 * (stretch_increment + stretch_increment.T)
            strain_jacobian[element, :, component] = _strain_mandel_vector(
                stretch_increment
            )
    return deform, rotation, stretch, strain, deviator, equivalent, strain_jacobian

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def normalized_activation(
    equivalent_strain: np.ndarray,
    shear_modulus: np.ndarray,
    yield_stress: np.ndarray,
    sharpness: np.ndarray,
) -> tuple:
    q, mu, sy, beta = _matched_vectors(
        equivalent_strain, shear_modulus, yield_stress, sharpness
    )
    if np.any(q < 0) or np.any(mu <= 0) or np.any(sy <= 0) or np.any(beta <= 0):
        raise ValueError("invalid strain or activation parameters")
    threshold = sy / (3.0 * mu)
    s0 = np.exp(-np.logaddexp(0.0, beta))
    z = beta * (q / threshold - 1.0)
    candidate = (
        threshold
        / (beta * (1.0 - s0))
        * (np.logaddexp(0.0, z) - np.logaddexp(0.0, -beta) - s0 * beta * q / threshold)
    )
    sigmoid = np.exp(-np.logaddexp(0.0, -z))
    derivative = (sigmoid - s0) / (1.0 - s0)
    curvature = beta * sigmoid * (1.0 - sigmoid) / (threshold * (1.0 - s0))
    candidate = np.maximum(candidate, 0.0)
    derivative = np.clip(derivative, 0.0, 1.0)
    candidate[q == 0.0] = 0.0
    derivative[q == 0.0] = 0.0
    if not all(
        np.all(np.isfinite(value)) for value in (candidate, derivative, curvature)
    ):
        raise ValueError("activation exceeds finite arithmetic range")
    return candidate, derivative, curvature

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def irreversible_state(
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    candidate: np.ndarray,
    derivative: np.ndarray,
    previous_history: np.ndarray,
    previous_plastic: np.ndarray,
) -> tuple:
    q, candidate, derivative, old = _matched_vectors(
        equivalent_strain, candidate, derivative, previous_history
    )
    deviator = _symmetric_tensors(deviator, len(q))
    plastic_old = _symmetric_tensors(previous_plastic, len(q))
    if np.any(q < 0) or np.any(candidate < 0) or np.any(old < 0):
        raise ValueError("strains and histories must be nonnegative")
    if np.any(derivative < 0) or np.any(derivative > 1):
        raise ValueError("candidate slopes must lie in [0, 1]")
    for tensor, magnitude in [(deviator, q), (plastic_old, old)]:
        if not np.allclose(
            np.trace(tensor, axis1=1, axis2=2), 0.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError("deviatoric tensors must be traceless")
        norm = np.sqrt(2.0 / 3.0 * np.sum(tensor**2, axis=(1, 2)))
        if not np.allclose(norm, magnitude, atol=1e-10, rtol=1e-8):
            raise ValueError("tensor equivalent norm disagrees with scalar history")
    active = candidate > old
    if np.any(active & (q <= 0)):
        raise ValueError("active states require positive equivalent strain")
    history = np.maximum(old, candidate)
    plastic = plastic_old.copy()
    plastic[active] = (history[active] / q[active])[:, None, None] * deviator[active]
    branch_derivative = np.where(active, derivative, 0.0)
    return history, plastic, branch_derivative, active.astype(float)

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def inelastic_work(
    history: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    attenuation_rate: np.ndarray,
) -> tuple:
    h, sy, hardening, rate = _matched_vectors(
        history, yield_stress, hardening, attenuation_rate
    )
    if np.any(h < 0) or np.any(sy <= 0) or np.any(hardening < 0) or np.any(rate < 0):
        raise ValueError("invalid inelastic-work parameters")
    work = sy * h + 0.5 * hardening * h**2
    survival = np.exp(-rate * h)
    derivative = survival * (sy + hardening * h - rate * work)
    second_derivative = survival * (
        hardening - 2.0 * rate * (sy + hardening * h) + rate**2 * work
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (work, survival, derivative, second_derivative)
    ):
        raise ValueError("inelastic work exceeds finite arithmetic range")
    return work, survival, derivative, second_derivative

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mandel_vectors(tensors):
    root_two = np.sqrt(2.0)
    return np.column_stack(
        (
            tensors[:, 0, 0],
            tensors[:, 1, 1],
            tensors[:, 2, 2],
            root_two * tensors[:, 1, 2],
            root_two * tensors[:, 0, 2],
            root_two * tensors[:, 0, 1],
        )
    )


def branch_stress(
    strain: np.ndarray,
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    plastic: np.ndarray,
    history: np.ndarray,
    branch_derivative: np.ndarray,
    active: np.ndarray,
    bulk_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    attenuated_work_derivative: np.ndarray,
    activation_curvature: np.ndarray,
    attenuated_work_second_derivative: np.ndarray,
) -> tuple:
    q, h, gp, flags, bulk, mu, derivative, curvature, derivative_h = _matched_vectors(
        equivalent_strain,
        history,
        branch_derivative,
        active,
        bulk_modulus,
        shear_modulus,
        attenuated_work_derivative,
        activation_curvature,
        attenuated_work_second_derivative,
    )
    strain = _symmetric_tensors(strain, len(q))
    deviator = _symmetric_tensors(deviator, len(q))
    plastic = _symmetric_tensors(plastic, len(q))
    if np.any(q < 0) or np.any(h < 0) or np.any(bulk <= 0) or np.any(mu <= 0):
        raise ValueError("invalid strain, history, or moduli")
    if (
        np.any((flags != 0) & (flags != 1))
        or np.any(gp < 0)
        or np.any(gp > 1)
        or np.any(curvature < 0)
    ):
        raise ValueError("invalid branch flags or slopes")
    active_mask = flags == 1
    if (
        np.any(gp[~active_mask] != 0)
        or np.any(curvature[~active_mask] != 0)
        or np.any(q[active_mask] <= 0)
    ):
        raise ValueError("branch slope or equivalent strain inconsistent with branch")
    for tensor, magnitude in ((deviator, q), (plastic, h)):
        if not np.allclose(
            np.trace(tensor, axis1=1, axis2=2), 0.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError("deviatoric tensors must be traceless")
        norm = np.sqrt(2.0 / 3.0 * np.sum(tensor**2, axis=(1, 2)))
        if not np.allclose(norm, magnitude, atol=1e-10, rtol=1e-8):
            raise ValueError("tensor equivalent norm disagrees with scalar measure")
    stress = 2.0 * mu[:, None, None] * (deviator - plastic)
    derivative_q = 3.0 * mu * (q - h) * (1.0 - gp) + derivative * gp
    stress[active_mask] = (2.0 * derivative_q[active_mask] / (3.0 * q[active_mask]))[
        :, None, None
    ] * deviator[active_mask]
    stress += (bulk * np.trace(strain, axis1=1, axis2=2))[:, None, None] * np.eye(3)

    volumetric = np.outer(
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    deviatoric = np.eye(6) - volumetric / 3.0
    tangent = bulk[:, None, None] * volumetric + 2.0 * mu[:, None, None] * deviatoric
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size:
        q_active = q[active_mask]
        h_active = h[active_mask]
        gp_active = gp[active_mask]
        curvature_active = curvature[active_mask]
        derivative_q_prime = (
            3.0
            * mu[active_mask]
            * ((1.0 - gp_active) ** 2 - (q_active - h_active) * curvature_active)
            + derivative_h[active_mask] * gp_active**2
            + derivative[active_mask] * curvature_active
        )
        radial = 2.0 * derivative_q[active_mask] / (3.0 * q_active)
        radial_prime = (2.0 / 3.0) * (
            derivative_q_prime / q_active - derivative_q[active_mask] / q_active**2
        )
        deviator_mandel = _mandel_vectors(deviator[active_mask])
        tangent[active_mask] = (
            bulk[active_mask, None, None] * volumetric
            + radial[:, None, None] * deviatoric
            + (2.0 * radial_prime / (3.0 * q_active))[:, None, None]
            * np.einsum("ni,nj->nij", deviator_mandel, deviator_mandel)
        )
    return stress, tangent

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mandel_tensor_from_vector(vector):
    root_two = np.sqrt(2.0)
    tensor = np.array(
        [
            [vector[0], vector[5] / root_two, vector[4] / root_two],
            [vector[5] / root_two, vector[1], vector[3] / root_two],
            [vector[4] / root_two, vector[3] / root_two, vector[2]],
        ]
    )
    return tensor


def reference_forces(
    deformation: np.ndarray,
    rotation: np.ndarray,
    stretch: np.ndarray,
    stress: np.ndarray,
    material_tangent: np.ndarray,
    strain_jacobian: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    cells: np.ndarray,
    node_count: int,
) -> tuple:
    volumes = _finite_array(volumes)
    if volumes.ndim != 1 or volumes.size == 0 or np.any(volumes <= 0):
        raise ValueError("volumes must be a nonempty positive vector")
    count = len(volumes)
    if (
        isinstance(node_count, (bool, np.bool_))
        or not isinstance(node_count, (int, np.integer))
        or node_count < 4
    ):
        raise ValueError("node_count must be an integer at least four")
    deform = _finite_array(deformation)
    rotation = _finite_array(rotation)
    stretch = _symmetric_tensors(stretch, count)
    stress = _symmetric_tensors(stress, count)
    tangent = _finite_array(material_tangent)
    strain_jacobian = _finite_array(strain_jacobian)
    gradients = _finite_array(gradients)
    cells = np.asarray(cells)
    if (
        deform.shape != (count, 3, 3)
        or rotation.shape != deform.shape
        or gradients.shape != (count, 4, 3)
        or tangent.shape != (count, 6, 6)
        or strain_jacobian.shape != (count, 6, 9)
    ):
        raise ValueError("invalid tensor or gradient shape")
    if cells.shape != (count, 4) or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("invalid connectivity shape or dtype")
    if (
        np.any(cells < 0)
        or np.any(cells >= node_count)
        or np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0)
    ):
        raise ValueError("invalid connectivity indices")
    jacobian = np.linalg.det(deform)
    if np.any(jacobian <= 0) or np.any(np.linalg.eigvalsh(stretch) <= 0):
        raise ValueError("deformation and stretch must preserve orientation")
    checks = [
        (rotation @ rotation.swapaxes(1, 2), np.eye(3)),
        (np.linalg.det(rotation), np.ones(count)),
        (deform, rotation @ stretch),
        (gradients.sum(axis=1), np.zeros((count, 3))),
    ]
    if any(not np.allclose(a, b, atol=1e-10, rtol=1e-8) for a, b in checks):
        raise ValueError("inconsistent rotation, stretch, or reference gradients")
    if not np.allclose(tangent, tangent.swapaxes(1, 2), atol=1e-10, rtol=1e-8):
        raise ValueError("material tangent must be Mandel symmetric")
    piola = jacobian[:, None, None] * (rotation @ stress @ np.linalg.inv(stretch))
    local_forces = -volumes[:, None, None] * (gradients @ piola.swapaxes(1, 2))
    forces = np.zeros((node_count, 3))
    np.add.at(forces, cells.reshape(-1), local_forces.reshape(-1, 3))
    stiffness = np.zeros((3 * node_count, 3 * node_count))
    for element in range(count):
        inverse_deform = np.linalg.inv(deform[element])
        inverse_stretch = np.linalg.inv(stretch[element])
        base = rotation[element] @ stress[element] @ inverse_stretch
        for local_node, global_node in enumerate(cells[element]):
            for component in range(3):
                deform_increment = np.zeros((3, 3))
                deform_increment[component] = gradients[element, local_node]
                strain_mandel = strain_jacobian[element] @ deform_increment.reshape(-1)
                stretch_increment = _mandel_tensor_from_vector(strain_mandel)
                rotation_increment = (
                    deform_increment - rotation[element] @ stretch_increment
                ) @ inverse_stretch
                stress_increment = _mandel_tensor_from_vector(
                    tangent[element] @ strain_mandel
                )
                jacobian_increment = jacobian[element] * np.trace(
                    inverse_deform @ deform_increment
                )
                inverse_stretch_increment = (
                    -inverse_stretch @ stretch_increment @ inverse_stretch
                )
                piola_increment = jacobian_increment * base + jacobian[element] * (
                    rotation_increment @ stress[element] @ inverse_stretch
                    + rotation[element] @ stress_increment @ inverse_stretch
                    + rotation[element] @ stress[element] @ inverse_stretch_increment
                )
                force_increment = -volumes[element] * (
                    gradients[element] @ piola_increment.T
                )
                column = 3 * global_node + component
                for response_node, response_global in enumerate(cells[element]):
                    row = slice(3 * response_global, 3 * response_global + 3)
                    stiffness[row, column] += force_increment[response_node]
    if not np.all(np.isfinite(stiffness)):
        raise ValueError("force Jacobian exceeds finite arithmetic range")
    return forces, stiffness

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mesh_arrays(reference, cells):
    points = _finite_array(reference)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("reference must have shape (n_nodes, 3), n_nodes >= 4")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("cells must have shape (n_elements, 4)")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cell indices must be integers")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("cell index out of range")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("each cell needs four distinct vertices")
    edges = (points[cells[:, 1:]] - points[cells[:, :1]]).transpose(0, 2, 1)
    lengths = np.linalg.norm(edges, axis=1)
    scale = np.prod(lengths, axis=1)
    determinant = np.linalg.det(edges)
    if np.any(scale == 0) or np.any(np.abs(determinant) <= 1e-12 * scale):
        raise ValueError(
            "reference tetrahedra are degenerate at relative tolerance 1e-12"
        )
    return points, cells.astype(int), edges


def _path_rotation(angle, axis):
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == "z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def prescribed_path_work(
    reference: np.ndarray,
    cells: np.ndarray,
    materials: np.ndarray,
    amplitudes: np.ndarray,
    rotations: np.ndarray,
    direction_angles: np.ndarray,
    perturbations: np.ndarray,
) -> float:
    points, cells, _ = _mesh_arrays(reference, cells)
    material = _finite_array(materials)
    if material.shape != (len(cells), 6):
        raise ValueError("materials must have shape (n_elements, 6)")
    young, poisson, sy, hardening, beta, rate = material.T
    if (
        np.any(young <= 0)
        or np.any(poisson <= -1)
        or np.any(poisson >= 0.5)
        or np.any(sy <= 0)
        or np.any(hardening < 0)
        or np.any(beta <= 0)
        or np.any(rate < 0)
    ):
        raise ValueError("invalid material parameters")
    amplitude, angles, directions = _matched_vectors(
        amplitudes, rotations, direction_angles
    )
    if len(amplitude) < 2 or amplitude[0] != 0 or angles[0] != 0 or directions[0] != 0:
        raise ValueError(
            "path needs at least two frames and starts at zero amplitude and angles"
        )
    if np.any(1.0 + 1.1 * amplitude <= 0) or np.any(1.0 - 0.4 * amplitude <= 0):
        raise ValueError("prescribed stretch must be positive definite")

    path_rate = _finite_array(perturbations)
    if path_rate.shape != (len(amplitude), len(points), 3):
        raise ValueError("perturbations must have shape (n_frames, n_nodes, 3)")
    if np.any(path_rate[0] != 0.0):
        raise ValueError("the initial nodal perturbation must be zero")

    mu = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    volumes, gradients = reference_geometry(points, cells)
    history = np.zeros(len(cells))
    plastic = np.zeros((len(cells), 3, 3))
    previous_points = points.copy()
    previous_forces = np.zeros_like(points)
    previous_force_rate = np.zeros_like(points)
    previous_point_rate = np.zeros_like(points)
    plastic_rate = np.zeros_like(plastic)
    basis = np.diag([1.0, -0.5, -0.5])
    total = 0.0

    for frame, (amplitude_n, angle, direction) in enumerate(
        zip(amplitude[1:], angles[1:], directions[1:]), start=1
    ):
        axes = _path_rotation(direction, "y")
        prescribed_stretch = np.eye(3) + amplitude_n * (
            axes @ basis @ axes.T + 0.1 * np.eye(3)
        )
        current = points @ (_path_rotation(angle, "z") @ prescribed_stretch).T
        deform, rotation, stretch, strain, deviator, equivalent, strain_jacobian = (
            corotational_kinematics(points, current, cells)
        )
        candidate, slope, curvature = normalized_activation(
            equivalent, mu, sy, beta
        )
        history, plastic, gp, active = irreversible_state(
            deviator, equivalent, candidate, slope, history, plastic
        )
        _, _, derivative, derivative_h = inelastic_work(
            history, sy, hardening, rate
        )
        stress, material_tangent = branch_stress(
            strain,
            deviator,
            equivalent,
            plastic,
            history,
            gp,
            active,
            bulk,
            mu,
            derivative,
            np.where(active == 1.0, curvature, 0.0),
            derivative_h,
        )
        forces, stiffness = reference_forces(
            deform,
            rotation,
            stretch,
            stress,
            material_tangent,
            strain_jacobian,
            volumes,
            gradients,
            cells,
            len(points),
        )
        point_rate = path_rate[frame]
        deformation_rate = np.einsum("eia,eib->eab", point_rate[cells], gradients)
        strain_rate_mandel = np.einsum(
            "eij,ej->ei", strain_jacobian, deformation_rate.reshape(len(cells), 9)
        )
        strain_rate = np.array(
            [_mandel_tensor_from_vector(vector) for vector in strain_rate_mandel]
        )
        deviator_rate = strain_rate - (np.trace(strain_rate, axis1=1, axis2=2) / 3.0)[
            :, None, None
        ] * np.eye(3)
        loading = active == 1.0
        # The local Jacobian holds prior state fixed; unloading retains its rate.
        memory_stress_rate = -2.0 * mu[:, None, None] * plastic_rate
        memory_stress_rate[loading] = 0.0
        memory_piola_rate = np.linalg.det(deform)[:, None, None] * (
            rotation @ memory_stress_rate @ np.linalg.inv(stretch)
        )
        memory_force_rate = np.zeros_like(points)
        local_memory_rate = -volumes[:, None, None] * np.einsum(
            "eab,eib->eia", memory_piola_rate, gradients
        )
        np.add.at(
            memory_force_rate, cells.reshape(-1), local_memory_rate.reshape(-1, 3)
        )
        force_rate = (stiffness @ point_rate.reshape(-1)).reshape(points.shape)
        force_rate += memory_force_rate
        if np.any(loading):
            q = equivalent[loading]
            e = deviator[loading]
            edot = deviator_rate[loading]
            qdot = (2.0 / (3.0 * q)) * np.sum(e * edot, axis=(1, 2))
            hdot = slope[loading] * qdot
            plastic_rate[loading] = (history[loading] / q)[:, None, None] * edot + (
                hdot / q - history[loading] * qdot / q**2
            )[:, None, None] * e
        total -= 0.5 * (
            np.sum((force_rate + previous_force_rate) * (current - previous_points))
            + np.sum((forces + previous_forces) * (point_rate - previous_point_rate))
        )
        previous_force_rate = force_rate
        previous_point_rate = point_rate
        previous_points = current
        previous_forces = forces

    if not np.isfinite(total):
        raise ValueError("path-work sensitivity is not finite")
    return float(total)
SCICODE_GOLD_EOF
