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
def build_affine_strain_reconstruction(
    cell_bounds, face_segments, outward_normals
):
    """Reference affine reconstruction from discrete integration by parts."""
    

    bounds = np.asarray(cell_bounds, dtype=float)
    faces = np.asarray(face_segments, dtype=float)
    normals = np.asarray(outward_normals, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("cell_bounds must be a finite array of shape (4,)")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("cell_bounds must have positive side lengths")
    if faces.shape != (4, 2, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("face_segments must have shape (4, 2, 2)")
    if normals.shape != (4, 2) or not np.all(np.isfinite(normals)):
        raise ValueError("outward_normals must have shape (4, 2)")

    x0, x1, y0, y1 = bounds
    centroid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    side_x = x1 - x0
    side_y = y1 - y0
    area = side_x * side_y
    diameter = np.hypot(side_x, side_y)
    root = 1.0 / np.sqrt(3.0)
    gauss = np.array([-root, root])

    for face, normal in zip(faces, normals):
        tangent = face[1] - face[0]
        length = np.linalg.norm(tangent)
        midpoint = 0.5 * (face[0] + face[1])
        if length <= 0.0:
            raise ValueError("face segments must be nondegenerate")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("outward normals must be unit vectors")
        if not np.isclose(tangent @ normal, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("each normal must be perpendicular to its face")
        if (midpoint - centroid) @ normal <= 0.0:
            raise ValueError("normals must point outward")
        on_vertical = np.isclose(midpoint[0], [x0, x1], atol=1e-12).any()
        on_horizontal = np.isclose(midpoint[1], [y0, y1], atol=1e-12).any()
        if not (on_vertical or on_horizontal):
            raise ValueError("each segment must lie on the rectangle boundary")

    points = []
    for xi in gauss:
        for eta in gauss:
            points.append(
                [
                    centroid[0] + 0.5 * side_x * xi,
                    centroid[1] + 0.5 * side_y * eta,
                ]
            )
    points = np.asarray(points)
    weights = np.full(4, area / 4.0)

    def basis(coordinates):
        coordinates = np.asarray(coordinates)
        return np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                (coordinates[:, 0] - centroid[0]) / diameter,
                (coordinates[:, 1] - centroid[1]) / diameter,
            ]
        )

    cell_basis = basis(points)
    scalar_mass = cell_basis.T @ (weights[:, None] * cell_basis)
    strain_mass = np.zeros((9, 9))
    strain_mass[0:3, 0:3] = scalar_mass
    strain_mass[3:6, 3:6] = scalar_mass
    strain_mass[6:9, 6:9] = 2.0 * scalar_mass
    right_hand_side = np.zeros((9, 22))

    integrated_cell_basis = weights @ cell_basis
    for polynomial in range(3):
        derivative_x = 1.0 / diameter if polynomial == 1 else 0.0
        derivative_y = 1.0 / diameter if polynomial == 2 else 0.0
        right_hand_side[polynomial, 0:3] -= derivative_x * integrated_cell_basis
        right_hand_side[3 + polynomial, 3:6] -= derivative_y * integrated_cell_basis
        right_hand_side[6 + polynomial, 0:3] -= derivative_y * integrated_cell_basis
        right_hand_side[6 + polynomial, 3:6] -= derivative_x * integrated_cell_basis

    for local_face, (face, normal) in enumerate(zip(faces, normals)):
        length = np.linalg.norm(face[1] - face[0])
        midpoint = 0.5 * (face[0] + face[1])
        half_tangent = 0.5 * (face[1] - face[0])
        face_points = midpoint + gauss[:, None] * half_tangent
        face_weights = np.full(2, length / 2.0)
        face_basis = np.column_stack([np.ones(2), gauss])
        moments = basis(face_points).T @ (face_weights[:, None] * face_basis)
        offset = 6 + 4 * local_face
        normal_x, normal_y = normal
        right_hand_side[0:3, offset : offset + 2] += normal_x * moments
        right_hand_side[3:6, offset + 2 : offset + 4] += normal_y * moments
        right_hand_side[6:9, offset : offset + 2] += normal_y * moments
        right_hand_side[6:9, offset + 2 : offset + 4] += normal_x * moments

    coefficients = np.linalg.solve(strain_mass, right_hand_side)
    result = np.zeros((4, 4, 22))
    for node, values in enumerate(cell_basis):
        result[node, 0] = values @ coefficients[0:3]
        result[node, 1] = values @ coefficients[3:6]
        result[node, 3] = values @ coefficients[6:9]
    return result

import numpy as np
def build_quadratic_stabilization(
    cell_bounds, face_segments, outward_normals, strain_reconstruction
):
    """Reference constrained quadratic reconstruction and stabilization."""
    

    bounds = np.asarray(cell_bounds, dtype=float)
    faces = np.asarray(face_segments, dtype=float)
    normals = np.asarray(outward_normals, dtype=float)
    strain = np.asarray(strain_reconstruction, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("cell_bounds must be finite with shape (4,)")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("cell_bounds must have positive side lengths")
    if faces.shape != (4, 2, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("face_segments must be finite with shape (4, 2, 2)")
    if normals.shape != (4, 2) or not np.all(np.isfinite(normals)):
        raise ValueError("outward_normals must be finite with shape (4, 2)")
    if strain.shape != (4, 4, 22) or not np.all(np.isfinite(strain)):
        raise ValueError("strain_reconstruction must have shape (4, 4, 22)")

    x0, x1, y0, y1 = bounds
    centroid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    side_x = x1 - x0
    side_y = y1 - y0
    area = side_x * side_y
    diameter = np.hypot(side_x, side_y)
    root = 1.0 / np.sqrt(3.0)
    gauss = np.array([-root, root])
    points = np.array(
        [
            [centroid[0] + 0.5 * side_x * xi, centroid[1] + 0.5 * side_y * eta]
            for xi in gauss
            for eta in gauss
        ]
    )
    weights = np.full(4, area / 4.0)

    for face, normal in zip(faces, normals):
        tangent = face[1] - face[0]
        if np.linalg.norm(tangent) <= 0.0:
            raise ValueError("face segments must be nondegenerate")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("outward normals must be unit vectors")
        if not np.isclose(tangent @ normal, 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("normals must be perpendicular to faces")
        if (0.5 * (face[0] + face[1]) - centroid) @ normal <= 0.0:
            raise ValueError("normals must point outward")

    def p1_basis(coordinates):
        coordinates = np.asarray(coordinates)
        return np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                (coordinates[:, 0] - centroid[0]) / diameter,
                (coordinates[:, 1] - centroid[1]) / diameter,
            ]
        )

    def p2_basis(coordinates):
        coordinates = np.asarray(coordinates)
        scaled_x = (coordinates[:, 0] - centroid[0]) / diameter
        scaled_y = (coordinates[:, 1] - centroid[1]) / diameter
        return np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                scaled_x,
                scaled_y,
                scaled_x**2,
                scaled_x * scaled_y,
                scaled_y**2,
            ]
        )

    def p2_gradient(coordinates):
        coordinates = np.asarray(coordinates)
        scaled_x = (coordinates[:, 0] - centroid[0]) / diameter
        scaled_y = (coordinates[:, 1] - centroid[1]) / diameter
        derivative_x = np.column_stack(
            [
                np.zeros(coordinates.shape[0]),
                np.ones(coordinates.shape[0]) / diameter,
                np.zeros(coordinates.shape[0]),
                2.0 * scaled_x / diameter,
                scaled_y / diameter,
                np.zeros(coordinates.shape[0]),
            ]
        )
        derivative_y = np.column_stack(
            [
                np.zeros(coordinates.shape[0]),
                np.zeros(coordinates.shape[0]),
                np.ones(coordinates.shape[0]) / diameter,
                np.zeros(coordinates.shape[0]),
                scaled_x / diameter,
                2.0 * scaled_y / diameter,
            ]
        )
        return derivative_x, derivative_y

    p1_values = p1_basis(points)
    p2_values = p2_basis(points)
    derivative_x, derivative_y = p2_gradient(points)
    metric = np.diag([1.0, 1.0, 2.0])
    energy = np.zeros((12, 12))
    forcing = np.zeros((12, 22))
    for node, weight in enumerate(weights):
        symmetric_gradient = np.zeros((3, 12))
        symmetric_gradient[0, 0:6] = derivative_x[node]
        symmetric_gradient[1, 6:12] = derivative_y[node]
        symmetric_gradient[2, 0:6] = 0.5 * derivative_y[node]
        symmetric_gradient[2, 6:12] = 0.5 * derivative_x[node]
        reconstructed = strain[node, [0, 1, 3]]
        energy += weight * (symmetric_gradient.T @ metric @ symmetric_gradient)
        forcing += weight * (symmetric_gradient.T @ metric @ reconstructed)

    constraints = np.zeros((3, 12))
    constraint_data = np.zeros((3, 22))
    integrated_p2 = weights @ p2_values
    integrated_p1 = weights @ p1_values
    constraints[0, 0:6] = integrated_p2
    constraints[1, 6:12] = integrated_p2
    constraint_data[0, 0:3] = integrated_p1
    constraint_data[1, 3:6] = integrated_p1
    constraints[2, 0:6] = -0.5 * (weights @ derivative_y)
    constraints[2, 6:12] = 0.5 * (weights @ derivative_x)

    for local_face, (face, normal) in enumerate(zip(faces, normals)):
        length = np.linalg.norm(face[1] - face[0])
        face_basis = np.column_stack([np.ones(2), gauss])
        integrated_face_basis = (length / 2.0) * np.sum(face_basis, axis=0)
        offset = 6 + 4 * local_face
        constraint_data[2, offset : offset + 2] += (
            -0.5 * normal[1] * integrated_face_basis
        )
        constraint_data[2, offset + 2 : offset + 4] += (
            0.5 * normal[0] * integrated_face_basis
        )

    saddle = np.block([[energy, constraints.T], [constraints, np.zeros((3, 3))]])
    reconstruction = np.linalg.solve(saddle, np.vstack([forcing, constraint_data]))[
        0:12
    ]

    cell_mass = p1_values.T @ (weights[:, None] * p1_values)
    cell_cross = p1_values.T @ (weights[:, None] * p2_values)
    cell_projection = np.linalg.solve(cell_mass, cell_cross)
    result = np.zeros((22, 22))
    for component in range(2):
        difference = (
            cell_projection @ reconstruction[6 * component : 6 * (component + 1)]
        )
        difference[:, 3 * component : 3 * (component + 1)] -= np.eye(3)
        result += diameter**-2 * (difference.T @ cell_mass @ difference)

    for local_face, face in enumerate(faces):
        length = np.linalg.norm(face[1] - face[0])
        midpoint = 0.5 * (face[0] + face[1])
        half_tangent = 0.5 * (face[1] - face[0])
        face_points = midpoint + gauss[:, None] * half_tangent
        face_basis = np.column_stack([np.ones(2), gauss])
        face_p2 = p2_basis(face_points)
        face_weights = np.full(2, length / 2.0)
        face_mass = face_basis.T @ (face_weights[:, None] * face_basis)
        face_cross = face_basis.T @ (face_weights[:, None] * face_p2)
        face_projection = np.linalg.solve(face_mass, face_cross)
        offset = 6 + 4 * local_face
        for component in range(2):
            difference = (
                face_projection @ reconstruction[6 * component : 6 * (component + 1)]
            )
            start = offset + 2 * component
            difference[:, start : start + 2] -= np.eye(2)
            result += length**-1 * (difference.T @ face_mass @ difference)
    return 0.5 * (result + result.T)

import numpy as np
def build_phase_reconstruction(cell_bounds, face_segments, outward_normals):
    """Reference affine phase gradient and face-average jump matrix."""
    

    bounds = np.asarray(cell_bounds, dtype=float)
    faces = np.asarray(face_segments, dtype=float)
    normals = np.asarray(outward_normals, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("cell_bounds must be finite with shape (4,)")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("cell_bounds must have positive side lengths")
    if faces.shape != (4, 2, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("face_segments must be finite with shape (4, 2, 2)")
    if normals.shape != (4, 2) or not np.all(np.isfinite(normals)):
        raise ValueError("outward_normals must be finite with shape (4, 2)")

    x0, x1, y0, y1 = bounds
    centroid = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    area = (x1 - x0) * (y1 - y0)
    diameter = np.hypot(x1 - x0, y1 - y0)
    gradient = np.zeros((2, 5))
    lengths = np.empty(4)
    for local_face, (face, normal) in enumerate(zip(faces, normals)):
        tangent = face[1] - face[0]
        lengths[local_face] = np.linalg.norm(tangent)
        midpoint = 0.5 * (face[0] + face[1])
        if lengths[local_face] <= 0.0:
            raise ValueError("face segments must be nondegenerate")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("outward normals must be unit vectors")
        if not np.isclose(tangent @ normal, 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("normals must be perpendicular to faces")
        if (midpoint - centroid) @ normal <= 0.0:
            raise ValueError("normals must point outward")
        gradient[:, 1 + local_face] = lengths[local_face] / area * normal

    jump = np.zeros((5, 5))
    for local_face, face in enumerate(faces):
        midpoint = 0.5 * (face[0] + face[1])
        residual = np.zeros(5)
        residual[0] = 1.0
        residual += (midpoint - centroid) @ gradient
        residual[1 + local_face] -= 1.0
        jump += lengths[local_face] / diameter * np.outer(residual, residual)
    return np.vstack([gradient, 0.5 * (jump + jump.T)])

import numpy as np
def update_volumetric_deviatoric_history(
    strain_reconstruction, displacement, previous_history, lame_lambda, shear_modulus
):
    """Reference three-dimensional volumetric-deviatoric history update."""
   

    strain_operator = np.asarray(strain_reconstruction, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    previous = np.asarray(previous_history, dtype=float)
    if strain_operator.ndim != 3 or strain_operator.shape[1] != 4:
        raise ValueError("strain_reconstruction must have shape (n_q, 4, n_dof)")
    if strain_operator.shape[0] < 1 or strain_operator.shape[2] < 1:
        raise ValueError("strain_reconstruction must be nonempty")
    if displacement.shape != (strain_operator.shape[2],):
        raise ValueError("displacement length must match the local dof count")
    if previous.shape != (strain_operator.shape[0],):
        raise ValueError("previous_history length must match the quadrature count")
    if (
        not np.all(np.isfinite(strain_operator))
        or not np.all(np.isfinite(displacement))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError("array inputs must be finite")
    if np.any(previous < 0.0):
        raise ValueError("previous_history must be nonnegative")
    if not np.isfinite(lame_lambda) or not np.isfinite(shear_modulus):
        raise ValueError("elastic parameters must be finite")
    if shear_modulus <= 0.0 or lame_lambda + 2.0 * shear_modulus / 3.0 <= 0.0:
        raise ValueError(
            "elastic parameters must define positive bulk and shear moduli"
        )

    strains = np.einsum("qij,j->qi", strain_operator, displacement)
    bulk_modulus = lame_lambda + 2.0 * shear_modulus / 3.0
    current = np.empty(strains.shape[0])
    for node, strain in enumerate(strains):
        trace = strain[0] + strain[1] + strain[2]
        deviatoric_normal = strain[0:3] - trace / 3.0
        deviatoric_norm_squared = (
            deviatoric_normal @ deviatoric_normal + 2.0 * strain[3] ** 2
        )
        current[node] = (
            0.5 * bulk_modulus * max(trace, 0.0) ** 2
            + shear_modulus * deviatoric_norm_squared
        )
    return current if np.max(current) > np.max(previous) else previous.copy()

import numpy as np
def condense_degraded_mechanics(
    strain_reconstruction,
    quadrature_weights,
    stabilization,
    cell_phase,
    lame_lambda,
    shear_modulus,
    n_cell=6,
):
    """Reference degraded assembly and cell-block Schur complement."""
    

    strain = np.asarray(strain_reconstruction, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    stabilization = np.asarray(stabilization, dtype=float)
    if strain.ndim != 3 or strain.shape[1] != 4 or strain.shape[0] < 1:
        raise ValueError("strain_reconstruction must have shape (n_q, 4, n_dof)")
    n_dof = strain.shape[2]
    if weights.shape != (strain.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be positive with shape (n_q,)")
    if stabilization.shape != (n_dof, n_dof):
        raise ValueError("stabilization shape must match the local dof count")
    if (
        not np.all(np.isfinite(strain))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(stabilization))
    ):
        raise ValueError("array inputs must be finite")
    if not np.allclose(stabilization, stabilization.T, atol=1e-12, rtol=0.0):
        raise ValueError("stabilization must be symmetric")
    if np.min(np.linalg.eigvalsh(stabilization)) < -1e-10:
        raise ValueError("stabilization must be positive semidefinite")
    if not np.isfinite(cell_phase) or cell_phase < 0.0 or cell_phase >= 1.0:
        raise ValueError("cell_phase must lie in [0, 1)")
    if not np.isfinite(lame_lambda) or not np.isfinite(shear_modulus):
        raise ValueError("elastic parameters must be finite")
    if shear_modulus <= 0.0 or lame_lambda + 2.0 * shear_modulus / 3.0 <= 0.0:
        raise ValueError(
            "elastic parameters must define positive bulk and shear moduli"
        )
    if not isinstance(n_cell, (int, np.integer)) or not 0 < n_cell < n_dof:
        raise ValueError("n_cell must be an integer strictly between zero and n_dof")

    elasticity = np.zeros((4, 4))
    elasticity[0:3, 0:3] = lame_lambda
    elasticity[0, 0] += 2.0 * shear_modulus
    elasticity[1, 1] += 2.0 * shear_modulus
    elasticity[2, 2] += 2.0 * shear_modulus
    elasticity[3, 3] = 4.0 * shear_modulus
    local_matrix = np.zeros((n_dof, n_dof))
    for operator, weight in zip(strain, weights):
        local_matrix += weight * (operator.T @ elasticity @ operator)
    local_matrix += 2.0 * shear_modulus * stabilization
    local_matrix *= (1.0 - cell_phase) ** 2
    local_matrix = 0.5 * (local_matrix + local_matrix.T)

    cell_block = local_matrix[:n_cell, :n_cell]
    coupling = local_matrix[:n_cell, n_cell:]
    face_block = local_matrix[n_cell:, n_cell:]
    try:
        solved_coupling = np.linalg.solve(cell_block, coupling)
    except np.linalg.LinAlgError as error:
        raise ValueError("cell mechanics block is singular") from error
    schur = face_block - coupling.T @ solved_coupling
    recovery = -solved_coupling
    return np.vstack([0.5 * (schur + schur.T), recovery])

import numpy as np
def condense_phase_field(
    phase_reconstruction,
    quadrature_weights,
    history,
    previous_cell_phase,
    length_scale,
    fracture_toughness,
    viscosity=0.0,
    time_step=1.0,
):
    """Reference diffusion-reaction assembly and scalar cell elimination."""
    

    operator = np.asarray(phase_reconstruction, dtype=float)
    weights = np.asarray(quadrature_weights, dtype=float)
    history = np.asarray(history, dtype=float)
    if operator.shape != (7, 5) or not np.all(np.isfinite(operator)):
        raise ValueError("phase_reconstruction must be finite with shape (7, 5)")
    if weights.ndim != 1 or weights.shape[0] < 1 or np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be a nonempty positive vector")
    if history.shape != weights.shape:
        raise ValueError("history and quadrature_weights must have matching shapes")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(history)):
        raise ValueError("quadrature data must be finite")
    if np.any(history < 0.0):
        raise ValueError("history must be nonnegative")
    if not np.isfinite(previous_cell_phase) or not 0.0 <= previous_cell_phase <= 1.0:
        raise ValueError("previous_cell_phase must lie in [0, 1]")
    parameters = [length_scale, fracture_toughness, viscosity, time_step]
    if not np.all(np.isfinite(parameters)):
        raise ValueError("scalar parameters must be finite")
    if length_scale <= 0.0 or fracture_toughness <= 0.0 or time_step <= 0.0:
        raise ValueError(
            "length_scale, fracture_toughness, and time_step must be positive"
        )
    if viscosity < 0.0:
        raise ValueError("viscosity must be nonnegative")

    gradient = operator[0:2]
    jump = operator[2:7]
    if not np.allclose(jump, jump.T, atol=1e-12, rtol=0.0):
        raise ValueError("the packed jump matrix must be symmetric")
    area = float(np.sum(weights))
    cell_selector = np.zeros(5)
    cell_selector[0] = 1.0
    reaction = np.sum(
        weights
        * (1.0 / length_scale**2 + 2.0 * history / (length_scale * fracture_toughness))
    )
    source = np.sum(weights * (2.0 * history / (length_scale * fracture_toughness)))
    viscous_coefficient = (
        area * viscosity / (length_scale * fracture_toughness * time_step)
    )
    local_matrix = area * (gradient.T @ gradient) + jump
    local_matrix += (reaction + viscous_coefficient) * np.outer(
        cell_selector, cell_selector
    )
    local_rhs = (source + viscous_coefficient * previous_cell_phase) * cell_selector

    cell_diagonal = local_matrix[0, 0]
    if not np.isfinite(cell_diagonal) or cell_diagonal <= 0.0:
        raise ValueError("phase cell block must be positive")
    coupling = local_matrix[0, 1:]
    schur = local_matrix[1:, 1:] - np.outer(coupling, coupling) / cell_diagonal
    condensed_rhs = local_rhs[1:] - coupling * local_rhs[0] / cell_diagonal
    recovery_coefficients = -coupling / cell_diagonal
    recovery_constant = local_rhs[0] / cell_diagonal
    packed = np.empty((5, 5))
    packed[0:4, 0:4] = 0.5 * (schur + schur.T)
    packed[0:4, 4] = condensed_rhs
    packed[4, 0:4] = recovery_coefficients
    packed[4, 4] = recovery_constant
    return packed

import numpy as np
def advance_staggered_patch(
    displacement_load, previous_phase, previous_history, material
):
    """Reference one-sweep patch advance using the preceding oracle steps."""
    

    phase = np.asarray(previous_phase, dtype=float)
    history = np.asarray(previous_history, dtype=float)
    material = np.asarray(material, dtype=float)
    if not np.isscalar(displacement_load) or not np.isfinite(displacement_load):
        raise ValueError("displacement_load must be a finite scalar")
    if phase.shape != (9,) or not np.all(np.isfinite(phase)):
        raise ValueError("previous_phase must be finite with shape (9,)")
    if np.any(phase < 0.0) or np.any(phase >= 1.0):
        raise ValueError("previous_phase values must lie in [0, 1)")
    if history.shape != (2, 4) or not np.all(np.isfinite(history)):
        raise ValueError("previous_history must be finite with shape (2, 4)")
    if np.any(history < 0.0):
        raise ValueError("previous_history must be nonnegative")
    if material.shape != (6,) or not np.all(np.isfinite(material)):
        raise ValueError("material must be finite with shape (6,)")
    lame_lambda, shear_modulus, toughness, length_scale, viscosity, time_step = material
    if shear_modulus <= 0.0 or lame_lambda + 2.0 * shear_modulus / 3.0 <= 0.0:
        raise ValueError("material must define positive bulk and shear moduli")
    if toughness <= 0.0 or length_scale <= 0.0 or time_step <= 0.0:
        raise ValueError("Gc, ell, and dt must be positive")
    if viscosity < 0.0:
        raise ValueError("eta must be nonnegative")

    global_faces = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0]],
            [[0.5, 0.0], [0.5, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.5, 1.0]],
            [[0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.5, 1.0], [0.0, 1.0]],
        ]
    )
    cell_bounds = [
        np.array([0.0, 0.5, 0.0, 1.0]),
        np.array([0.5, 1.0, 0.0, 1.0]),
    ]
    cell_face_ids = [np.array([0, 1, 6, 5]), np.array([4, 2, 3, 1])]
    normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = np.full(4, 0.125)

    strain_operators = []
    mechanics_data = []
    phase_operators = []
    for cell in range(2):
        local_faces = global_faces[cell_face_ids[cell]]
        strain_operator = build_affine_strain_reconstruction(
            cell_bounds[cell], local_faces, normals
        )
        stabilization = build_quadratic_stabilization(
            cell_bounds[cell], local_faces, normals, strain_operator
        )
        mechanics = condense_degraded_mechanics(
            strain_operator,
            weights,
            stabilization,
            phase[cell],
            lame_lambda,
            shear_modulus,
        )
        phase_operator = build_phase_reconstruction(
            cell_bounds[cell], local_faces, normals
        )
        strain_operators.append(strain_operator)
        mechanics_data.append(mechanics)
        phase_operators.append(phase_operator)

    face_mechanics = np.zeros((28, 28))
    for cell in range(2):
        local_global = np.concatenate(
            [4 * face + np.arange(4) for face in cell_face_ids[cell]]
        )
        face_mechanics[np.ix_(local_global, local_global)] += mechanics_data[cell][0:16]

    face_displacement = np.zeros(28)
    left_dofs = 4 * 5 + np.arange(4)
    right_dofs = 4 * 2 + np.arange(4)
    fixed_dofs = np.concatenate([left_dofs, right_dofs])
    free_dofs = np.setdiff1d(np.arange(28), fixed_dofs)
    face_displacement[left_dofs[0]] = -float(displacement_load)
    try:
        face_displacement[free_dofs] = np.linalg.solve(
            face_mechanics[np.ix_(free_dofs, free_dofs)],
            -face_mechanics[np.ix_(free_dofs, fixed_dofs)]
            @ face_displacement[fixed_dofs],
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("condensed mechanics system is singular") from error
    mechanics_residual = face_mechanics @ face_displacement
    normalized_reaction = abs(mechanics_residual[left_dofs[0]]) / shear_modulus

    local_displacements = []
    updated_history = np.empty_like(history)
    for cell in range(2):
        local_global = np.concatenate(
            [4 * face + np.arange(4) for face in cell_face_ids[cell]]
        )
        local_face_values = face_displacement[local_global]
        local_cell_values = mechanics_data[cell][16:22] @ local_face_values
        local_displacement = np.concatenate([local_cell_values, local_face_values])
        local_displacements.append(local_displacement)
        updated_history[cell] = update_volumetric_deviatoric_history(
            strain_operators[cell],
            local_displacement,
            history[cell],
            lame_lambda,
            shear_modulus,
        )

    face_phase_matrix = np.zeros((7, 7))
    face_phase_rhs = np.zeros(7)
    phase_data = []
    for cell in range(2):
        condensed_phase = condense_phase_field(
            phase_operators[cell],
            weights,
            updated_history[cell],
            phase[cell],
            length_scale,
            toughness,
            viscosity,
            time_step,
        )
        ids = cell_face_ids[cell]
        face_phase_matrix[np.ix_(ids, ids)] += condensed_phase[0:4, 0:4]
        face_phase_rhs[ids] += condensed_phase[0:4, 4]
        phase_data.append(condensed_phase)
    try:
        updated_face_phase = np.linalg.solve(face_phase_matrix, face_phase_rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("condensed phase system is singular") from error

    updated_cell_phase = np.empty(2)
    for cell in range(2):
        recovery = phase_data[cell][4]
        updated_cell_phase[cell] = (
            recovery[0:4] @ updated_face_phase[cell_face_ids[cell]] + recovery[4]
        )
    updated_phase = np.concatenate([updated_cell_phase, updated_face_phase])
    result = np.concatenate(
        [updated_phase, updated_history.ravel(), [normalized_reaction]]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("the staggered update produced a nonfinite result")
    return result

import numpy as np
def compute_irreversible_patch_response(load_path=None, material=None):
    """Reference load-path orchestration with irreversible state transfer."""
    

    if load_path is None:
        load_path = np.array([0.002, 0.020, -0.035, 0.004, 0.015])
    if material is None:
        material = np.array([121.15, 80.77, 2.7e-3, 0.0075, 0.0, 1.0])
    loads = np.asarray(load_path, dtype=float)
    material = np.asarray(material, dtype=float)
    if loads.ndim != 1 or loads.shape[0] < 1 or not np.all(np.isfinite(loads)):
        raise ValueError("load_path must be a nonempty finite vector")
    if material.shape != (6,) or not np.all(np.isfinite(material)):
        raise ValueError("material must be finite with shape (6,)")

    # Build and validate the fixed patch contract through every preceding
    # numerical stage before advancing its state. These values are also built
    # inside the one-increment step, but the direct chain keeps the final
    # orchestrator sensitive to every public stage in the ordered pipeline.
    bounds = np.array([0.0, 0.5, 0.0, 1.0])
    faces = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0]],
            [[0.5, 0.0], [0.5, 1.0]],
            [[0.5, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = np.full(4, 0.125)
    strain = build_affine_strain_reconstruction(bounds, faces, normals)
    stabilization = build_quadratic_stabilization(
        bounds, faces, normals, strain
    )
    phase_operator = build_phase_reconstruction(bounds, faces, normals)
    zero_history = update_volumetric_deviatoric_history(
        strain, np.zeros(22), np.zeros(4), material[0], material[1]
    )
    mechanics_contract = condense_degraded_mechanics(
        strain, weights, stabilization, 0.0, material[0], material[1]
    )
    phase_contract = condense_phase_field(
        phase_operator,
        weights,
        zero_history,
        0.0,
        material[3],
        material[2],
        material[4],
        material[5],
    )
    contract_norms = np.array(
        [
            np.linalg.norm(strain),
            np.linalg.norm(stabilization),
            np.linalg.norm(phase_operator),
            np.linalg.norm(mechanics_contract),
            np.linalg.norm(phase_contract),
        ]
    )
    if not np.all(np.isfinite(contract_norms)):
        raise ValueError("the fixed patch contract produced a nonfinite operator")

    phase = np.zeros(9)
    history = np.zeros((2, 4))
    final_reaction = 0.0
    for load in loads:
        state = advance_staggered_patch(load, phase, history, material)
        phase = state[0:9]
        history = state[9:17].reshape(2, 4)
        final_reaction = float(state[17])
    if not np.isfinite(final_reaction):
        raise ValueError("the final normalized reaction is not finite")
    return final_reaction
SCICODE_GOLD_EOF
