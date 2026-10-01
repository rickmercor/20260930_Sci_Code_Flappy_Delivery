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

_CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=int,
)
_CUBE_TETS = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=int,
)


def build_periodic_sphere_mesh(
    n_voxels: int,
    radius: float,
    center: np.ndarray,
) -> dict:
    """Reference implementation."""
    if not isinstance(n_voxels, (int, np.integer)) or int(n_voxels) < 2:
        raise ValueError("n_voxels must be an integer of at least two")
    n_voxels = int(n_voxels)
    if not np.isfinite(radius) or not 0.0 < float(radius) < 0.5:
        raise ValueError("radius must lie strictly between zero and one half")
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must be a finite array with shape (3,)")
    if np.any(center <= 0.0) or np.any(center >= 1.0):
        raise ValueError("center must lie strictly inside the unit cell")

    h = 1.0 / n_voxels
    nodes = np.array(
        [
            [i * h, j * h, k * h]
            for i in range(n_voxels)
            for j in range(n_voxels)
            for k in range(n_voxels)
        ],
        dtype=float,
    )
    element_vertices = []
    element_node_ids = []
    element_levels = []
    for i in range(n_voxels):
        for j in range(n_voxels):
            for k in range(n_voxels):
                origin = np.array([i, j, k], dtype=float) * h
                cube_vertices = origin + h * _CUBE_VERTICES
                cube_node_ids = np.array(
                    [
                        np.ravel_multi_index(
                            (
                                (i + dx) % n_voxels,
                                (j + dy) % n_voxels,
                                (k + dz) % n_voxels,
                            ),
                            (n_voxels, n_voxels, n_voxels),
                        )
                        for dx, dy, dz in _CUBE_VERTICES
                    ],
                    dtype=int,
                )
                distances = np.abs(cube_vertices - center)
                distances = np.minimum(distances, 1.0 - distances)
                cube_levels = np.linalg.norm(distances, axis=1) - float(radius)
                for tetrahedron in _CUBE_TETS:
                    element_vertices.append(cube_vertices[tetrahedron])
                    element_node_ids.append(cube_node_ids[tetrahedron])
                    element_levels.append(cube_levels[tetrahedron])

    element_vertices = np.asarray(element_vertices, dtype=float)
    element_node_ids = np.asarray(element_node_ids, dtype=int)
    element_levels = np.asarray(element_levels, dtype=float)
    cut = (np.min(element_levels, axis=1) < 0.0) & (
        np.max(element_levels, axis=1) > 0.0
    )
    enriched_nodes = np.unique(element_node_ids[cut])
    return {
        "nodes": nodes,
        "vertices": element_vertices,
        "node_ids": element_node_ids,
        "levels": element_levels,
        "cut": cut,
        "enriched_nodes": enriched_nodes,
        "n_voxels": n_voxels,
        "radius": float(radius),
        "center": center,
    }

import numpy as np

_TET_FACES = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))

def _unique_rows(rows, tolerance=1e-12):
    unique = []
    for row in rows:
        if not any(np.linalg.norm(row - other) <= tolerance for other in unique):
            unique.append(row)
    return unique


def _clip_face(face, levels, phase, tolerance=1e-13):
    clipped = []
    for index, current in enumerate(face):
        previous = face[index - 1]
        level_previous = float(levels @ previous)
        level_current = float(levels @ current)
        previous_inside = phase * level_previous >= -tolerance
        current_inside = phase * level_current >= -tolerance
        if current_inside != previous_inside:
            fraction = level_previous / (level_previous - level_current)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
    return _unique_rows(clipped)


def _interface_polygon(levels):
    points = []
    for first in range(4):
        if abs(levels[first]) <= 1e-13:
            point = np.zeros(4)
            point[first] = 1.0
            points.append(point)
        for second in range(first + 1, 4):
            if levels[first] * levels[second] < 0.0:
                fraction = levels[first] / (levels[first] - levels[second])
                point = np.zeros(4)
                point[first] = 1.0 - fraction
                point[second] = fraction
                points.append(point)
    return _unique_rows(points)


def _order_polygon(barycentric_points, vertices):
    coordinates = np.asarray(barycentric_points) @ vertices
    center = np.mean(coordinates, axis=0)
    first_axis = coordinates[0] - center
    first_axis /= np.linalg.norm(first_axis)
    normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
    normal /= np.linalg.norm(normal)
    second_axis = np.cross(normal, first_axis)
    angles = np.arctan2(
        (coordinates - center) @ second_axis,
        (coordinates - center) @ first_axis,
    )
    return [barycentric_points[index] for index in np.argsort(angles)]


def _phase_subtets(vertices, levels, phase):
    boundary_polygons = []
    for face_ids in _TET_FACES:
        face = []
        for node in face_ids:
            point = np.zeros(4)
            point[node] = 1.0
            face.append(point)
        clipped = _clip_face(face, levels, phase)
        if len(clipped) >= 3:
            boundary_polygons.append(clipped)
    interface = _interface_polygon(levels)
    if len(interface) >= 3:
        boundary_polygons.append(_order_polygon(interface, vertices))
    polyhedron_vertices = _unique_rows(
        [point for polygon in boundary_polygons for point in polygon]
    )
    if len(polyhedron_vertices) < 4:
        return []
    center = np.mean(polyhedron_vertices, axis=0)
    subtetrahedra = []
    for polygon in boundary_polygons:
        for index in range(1, len(polygon) - 1):
            barycentric_tet = np.array(
                [center, polygon[0], polygon[index], polygon[index + 1]]
            )
            physical_tet = barycentric_tet @ vertices
            volume = abs(np.linalg.det((physical_tet[1:] - physical_tet[0]).T)) / 6.0
            if volume > 1e-15:
                subtetrahedra.append((barycentric_tet, volume, phase))
    return subtetrahedra


def construct_subcell_quadrature(
    vertices: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Reference implementation."""
    vertices = np.asarray(vertices, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if vertices.shape != (4, 3) or levels.shape != (4,):
        raise ValueError("vertices and levels must have shapes (4, 3) and (4,)")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(levels)):
        raise ValueError("vertices and levels must be finite")
    volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
    if volume <= 1e-14:
        raise ValueError("vertices must define a nondegenerate tetrahedron")
    if np.any(np.abs(levels) <= 1e-12):
        raise ValueError("the benchmark convention excludes interface nodes")

    high = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    low = (5.0 - np.sqrt(5.0)) / 20.0
    rule = np.array(
        [
            [high, low, low, low],
            [low, high, low, low],
            [low, low, high, low],
            [low, low, low, high],
        ]
    )
    if np.all(levels > 0.0) or np.all(levels < 0.0):
        subtetrahedra = [(np.eye(4), volume, 1 if np.mean(levels) > 0.0 else -1)]
    else:
        subtetrahedra = _phase_subtets(vertices, levels, -1)
        subtetrahedra += _phase_subtets(vertices, levels, 1)

    barycentric = []
    weights = []
    phases = []
    for barycentric_tet, subvolume, phase in subtetrahedra:
        barycentric.extend(rule @ barycentric_tet)
        weights.extend([subvolume / 4.0] * 4)
        phases.extend([phase] * 4)
    return {
        "barycentric": np.asarray(barycentric, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "phases": np.asarray(phases, dtype=int),
    }

import numpy as np

def evaluate_modified_abs_enrichment(
    barycentric: np.ndarray,
    levels: np.ndarray,
    shape_gradients: np.ndarray,
) -> dict:
    """Reference implementation."""
    barycentric = np.asarray(barycentric, dtype=float)
    levels = np.asarray(levels, dtype=float)
    shape_gradients = np.asarray(shape_gradients, dtype=float)
    if barycentric.ndim != 2 or barycentric.shape[1] != 4:
        raise ValueError("barycentric must have shape (q, 4)")
    if levels.shape != (4,) or shape_gradients.shape != (4, 3):
        raise ValueError("levels and shape_gradients have invalid shapes")
    if not np.allclose(np.sum(barycentric, axis=1), 1.0, atol=1e-12):
        raise ValueError("every barycentric row must sum to one")
    if np.any(barycentric < -1e-12):
        raise ValueError("barycentric coordinates must be nonnegative")
    if not np.allclose(np.sum(shape_gradients, axis=0), 0.0, atol=1e-12):
        raise ValueError("P1 shape gradients must sum to zero")

    interpolated_level = barycentric @ levels
    if np.any(np.abs(interpolated_level) <= 1e-14):
        raise ValueError("quadrature points may not lie on the interface")
    signs = np.sign(interpolated_level)
    rho = barycentric @ np.abs(levels) - np.abs(interpolated_level)
    interpolated_abs_gradient = np.abs(levels) @ shape_gradients
    level_gradient = levels @ shape_gradients
    grad_rho = interpolated_abs_gradient[None, :] - signs[:, None] * level_gradient
    grad_enriched_shapes = (
        rho[:, None, None] * shape_gradients[None, :, :]
        + barycentric[:, :, None] * grad_rho[:, None, :]
    )
    return {
        "rho": rho,
        "grad_rho": grad_rho,
        "grad_enriched_shapes": grad_enriched_shapes,
    }

import numpy as np

def _shape_gradients(vertices):
    interpolation = np.column_stack([np.ones(4), vertices])
    return np.linalg.inv(interpolation)[1:, :].T


def _modified_abs_gradients(barycentric, levels, gradients):
    interpolated_level = barycentric @ levels
    signs = np.where(interpolated_level >= 0.0, 1.0, -1.0)
    rho = barycentric @ np.abs(levels) - np.abs(interpolated_level)
    grad_rho = (np.abs(levels) @ gradients)[None, :] - signs[:, None] * (
        levels @ gradients
    )[None, :]
    return (
        rho[:, None, None] * gradients[None, :, :]
        + barycentric[:, :, None] * grad_rho[:, None, :]
    )


def _mandel_column(gradient, component):
    gx, gy, gz = gradient
    root_two = np.sqrt(2.0)
    if component == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / root_two, gy / root_two])
    if component == 1:
        return np.array([0.0, gy, 0.0, gz / root_two, 0.0, gx / root_two])
    return np.array([0.0, 0.0, gz, gy / root_two, gx / root_two, 0.0])


def _isotropic_elasticity(lame_pair):
    lame_lambda, lame_mu = lame_pair
    elasticity = np.zeros((6, 6), dtype=float)
    elasticity[:3, :3] = lame_lambda
    elasticity[np.arange(3), np.arange(3)] += 2.0 * lame_mu
    elasticity[3:, 3:] = 2.0 * lame_mu * np.eye(3)
    return elasticity


def _validate_inputs(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain):
    mesh_fields = {"nodes", "vertices", "node_ids", "levels", "cut", "enriched_nodes"}
    quad_fields = {"element_ptr", "barycentric", "weights", "phases"}
    if not isinstance(mesh, dict) or not mesh_fields.issubset(mesh):
        raise ValueError("mesh is missing required fields")
    if not isinstance(quadrature, dict) or not quad_fields.issubset(quadrature):
        raise ValueError("quadrature is missing required fields")
    vertices = np.asarray(mesh["vertices"], dtype=float)
    node_ids = np.asarray(mesh["node_ids"], dtype=int)
    levels = np.asarray(mesh["levels"], dtype=float)
    cut = np.asarray(mesh["cut"], dtype=bool)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError("mesh vertices must have shape (e, 4, 3)")
    if node_ids.shape != vertices.shape[:2] or levels.shape != vertices.shape[:2]:
        raise ValueError("mesh connectivity and level arrays do not align")
    if cut.shape != (vertices.shape[0],):
        raise ValueError("cut flags do not align with elements")
    pointer = np.asarray(quadrature["element_ptr"], dtype=int)
    if pointer.shape != (vertices.shape[0] + 1,) or pointer[0] != 0:
        raise ValueError("element_ptr has an invalid shape")
    if pointer[-1] != len(quadrature["weights"]):
        raise ValueError("element_ptr does not cover all quadrature points")
    matrix_lame = np.asarray(matrix_lame, dtype=float)
    inclusion_lame = np.asarray(inclusion_lame, dtype=float)
    macrostrain = np.asarray(macrostrain, dtype=float)
    if matrix_lame.shape != (2,) or inclusion_lame.shape != (2,):
        raise ValueError("each Lamé pair must have shape (2,)")
    if np.any(matrix_lame <= 0.0) or np.any(inclusion_lame <= 0.0):
        raise ValueError("Lamé parameters must be positive")
    if macrostrain.shape != (6,):
        raise ValueError("macrostrain must have shape (6,)")
    return matrix_lame, inclusion_lame, macrostrain


def scale_and_assemble_elements(
    mesh: dict,
    quadrature: dict,
    matrix_lame: np.ndarray,
    inclusion_lame: np.ndarray,
    macrostrain: np.ndarray,
) -> dict:
    """Reference implementation."""
    matrix_lame, inclusion_lame, macrostrain = _validate_inputs(
        mesh, quadrature, matrix_lame, inclusion_lame, macrostrain
    )
    n_nodes = len(mesh["nodes"])
    enriched_nodes = np.asarray(mesh["enriched_nodes"], dtype=int)
    enrichment_index = {int(node): index for index, node in enumerate(enriched_nodes)}
    scaling_energy = np.zeros((len(enriched_nodes), 3), dtype=float)

    for element, (vertices, node_ids, levels, is_cut) in enumerate(
        zip(mesh["vertices"], mesh["node_ids"], mesh["levels"], mesh["cut"])
    ):
        if not is_cut:
            continue
        start, stop = quadrature["element_ptr"][element : element + 2]
        barycentric = quadrature["barycentric"][start:stop]
        weights = quadrature["weights"][start:stop]
        gradients = _shape_gradients(vertices)
        enriched_gradients = _modified_abs_gradients(barycentric, levels, gradients)
        for point, weight in enumerate(weights):
            for local_node, global_node in enumerate(node_ids):
                enriched_node = enrichment_index[int(global_node)]
                for component in range(3):
                    column = _mandel_column(
                        enriched_gradients[point, local_node], component
                    )
                    scaling_energy[enriched_node, component] += weight * (
                        column @ column
                    )
    if scaling_energy.size and np.any(scaling_energy <= 1e-14):
        raise ValueError("an enriched component has zero scaling energy")

    n_elements = len(mesh["vertices"])
    element_matrices = np.zeros((n_elements, 24, 24), dtype=float)
    element_rhs = np.zeros((n_elements, 24), dtype=float)
    element_stress = np.zeros((n_elements, 6, 24), dtype=float)
    element_dofs = np.full((n_elements, 24), -1, dtype=int)
    stress_macro = np.zeros(6, dtype=float)
    materials = {
        -1: _isotropic_elasticity(inclusion_lame),
        1: _isotropic_elasticity(matrix_lame),
    }

    for element, (vertices, node_ids, levels, is_cut) in enumerate(
        zip(mesh["vertices"], mesh["node_ids"], mesh["levels"], mesh["cut"])
    ):
        start, stop = quadrature["element_ptr"][element : element + 2]
        barycentric = quadrature["barycentric"][start:stop]
        weights = quadrature["weights"][start:stop]
        phases = quadrature["phases"][start:stop]
        gradients = _shape_gradients(vertices)
        enriched_gradients = None
        if is_cut:
            enriched_gradients = _modified_abs_gradients(barycentric, levels, gradients)

        dofs = []
        for node in node_ids:
            dofs.extend([3 * int(node) + component for component in range(3)])
        for node in node_ids:
            if int(node) in enrichment_index:
                offset = 3 * n_nodes + 3 * enrichment_index[int(node)]
                dofs.extend([offset + component for component in range(3)])
            else:
                dofs.extend([-1, -1, -1])
        element_dofs[element] = dofs

        for point, (weight, phase) in enumerate(zip(weights, phases)):
            strain_matrix = np.zeros((6, 24), dtype=float)
            for local_node in range(4):
                for component in range(3):
                    strain_matrix[:, 3 * local_node + component] = _mandel_column(
                        gradients[local_node], component
                    )
                    if is_cut:
                        global_node = int(node_ids[local_node])
                        enriched_node = enrichment_index[global_node]
                        scale = np.sqrt(scaling_energy[enriched_node, component])
                        strain_matrix[:, 12 + 3 * local_node + component] = (
                            _mandel_column(
                                enriched_gradients[point, local_node], component
                            )
                            / scale
                        )
            elasticity = materials[int(phase)]
            element_matrices[element] += weight * (
                strain_matrix.T @ elasticity @ strain_matrix
            )
            element_rhs[element] -= weight * (
                strain_matrix.T @ elasticity @ macrostrain
            )
            element_stress[element] += weight * (elasticity @ strain_matrix)
            stress_macro += weight * (elasticity @ macrostrain)

    n_standard_dofs = 3 * n_nodes
    n_enriched_dofs = 3 * len(enriched_nodes)
    right_hand_side = np.zeros(n_standard_dofs + n_enriched_dofs, dtype=float)
    for dofs, local_rhs in zip(element_dofs, element_rhs):
        active = dofs >= 0
        np.add.at(right_hand_side, dofs[active], local_rhs[active])
    return {
        "scaling_energy": scaling_energy,
        "element_matrices": element_matrices,
        "element_rhs": element_rhs,
        "element_stress": element_stress,
        "element_dofs": element_dofs,
        "rhs": right_hand_side,
        "stress_macro": stress_macro,
        "n_standard_dofs": n_standard_dofs,
        "n_enriched_dofs": n_enriched_dofs,
    }

import numpy as np

_CUBE_VERTICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=int,
)
_CUBE_TETS = np.array(
    [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ],
    dtype=int,
)

def _shape_gradients(vertices):
    interpolation = np.column_stack([np.ones(4), vertices])
    return np.linalg.inv(interpolation)[1:, :].T


def _mandel_column(gradient, component):
    gx, gy, gz = gradient
    root_two = np.sqrt(2.0)
    if component == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / root_two, gy / root_two])
    if component == 1:
        return np.array([0.0, gy, 0.0, gz / root_two, 0.0, gx / root_two])
    return np.array([0.0, 0.0, gz, gy / root_two, gx / root_two, 0.0])


def _reference_voxel_stiffness(voxel_edge):
    coordinates = voxel_edge * _CUBE_VERTICES
    stiffness = np.zeros((24, 24), dtype=float)
    for tetrahedron in _CUBE_TETS:
        vertices = coordinates[tetrahedron]
        gradients = _shape_gradients(vertices)
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        strain_matrix = np.zeros((6, 12), dtype=float)
        for local_node in range(4):
            for component in range(3):
                strain_matrix[:, 3 * local_node + component] = _mandel_column(
                    gradients[local_node], component
                )
        local_stiffness = volume * (strain_matrix.T @ strain_matrix)
        dofs = np.array(
            [
                3 * int(node) + component
                for node in tetrahedron
                for component in range(3)
            ]
        )
        stiffness[np.ix_(dofs, dofs)] += local_stiffness
    return stiffness


def build_fourier_green_operator(n_voxels: int) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(n_voxels, (int, np.integer)) or int(n_voxels) < 2:
        raise ValueError("n_voxels must be an integer of at least two")
    n_voxels = int(n_voxels)
    voxel_stiffness = _reference_voxel_stiffness(1.0 / n_voxels)
    green = np.zeros(
        (n_voxels, n_voxels, n_voxels, 3, 3),
        dtype=complex,
    )
    for frequency in np.ndindex(n_voxels, n_voxels, n_voxels):
        if frequency == (0, 0, 0):
            continue
        wavevector = 2.0 * np.pi * np.asarray(frequency) / n_voxels
        phases = np.exp(1j * (_CUBE_VERTICES @ wavevector))
        phase_matrix = np.zeros((24, 3), dtype=complex)
        for node in range(8):
            phase_matrix[3 * node : 3 * node + 3] = phases[node] * np.eye(3)
        symbol = phase_matrix.conj().T @ voxel_stiffness @ phase_matrix
        green[frequency] = np.linalg.pinv(symbol, rcond=1e-12)
    return green

import numpy as np

def _apply_operator(elements, vector):
    result = np.zeros_like(vector)
    for matrix, dofs in zip(elements["element_matrices"], elements["element_dofs"]):
        active = dofs >= 0
        local = np.zeros(matrix.shape[0], dtype=float)
        local[active] = vector[dofs[active]]
        product = matrix @ local
        np.add.at(result, dofs[active], product[active])
    return result


def _effective_stress(elements, vector):
    result = np.asarray(elements["stress_macro"], dtype=float).copy()
    for matrix, dofs in zip(elements["element_stress"], elements["element_dofs"]):
        active = dofs >= 0
        local = np.zeros(matrix.shape[1], dtype=float)
        local[active] = vector[dofs[active]]
        result += matrix @ local
    return result


def _apply_preconditioner(green, residual, n_standard_dofs):
    n_voxels = green.shape[0]
    expected = 3 * n_voxels**3
    if n_standard_dofs != expected:
        raise ValueError("standard block size is inconsistent with green")
    standard = residual[:n_standard_dofs].reshape(n_voxels, n_voxels, n_voxels, 3)
    transformed = np.fft.fftn(standard, axes=(0, 1, 2))
    solution_hat = np.einsum("...ij,...j->...i", green, transformed)
    standard_solution = np.fft.ifftn(solution_hat, axes=(0, 1, 2)).real.reshape(-1)
    return np.concatenate([standard_solution, residual[n_standard_dofs:]])


def solve_scaled_xfft_system(
    elements: dict,
    green: np.ndarray,
    tol: float,
    max_iter: int,
) -> dict:
    """Reference implementation."""
    if not isinstance(elements, dict):
        raise ValueError("elements must be a dictionary")  # noqa: TRY004
    required = {
        "element_matrices",
        "element_dofs",
        "element_stress",
        "rhs",
        "stress_macro",
        "n_standard_dofs",
    }
    if not required.issubset(elements):
        raise ValueError("elements is missing required fields")
    green = np.asarray(green, dtype=complex)
    if green.ndim != 5 or green.shape[-2:] != (3, 3):
        raise ValueError("green must have shape (n, n, n, 3, 3)")
    if not np.isfinite(tol) or float(tol) <= 0.0:
        raise ValueError("tol must be positive")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")

    right_hand_side = np.asarray(elements["rhs"], dtype=float)
    displacement = np.zeros_like(right_hand_side)
    residual = right_hand_side - _apply_operator(elements, displacement)
    preconditioned = _apply_preconditioner(
        green, residual, int(elements["n_standard_dofs"])
    )
    direction = preconditioned.copy()
    residual_product = float(residual @ preconditioned)
    residual_trace = [np.sqrt(max(residual_product, 0.0))]
    converged = residual_trace[-1] <= float(tol) * np.linalg.norm(
        _effective_stress(elements, displacement)
    )
    iteration = 0
    while not converged and iteration < int(max_iter):
        action = _apply_operator(elements, direction)
        curvature = float(direction @ action)
        if curvature <= 0.0:
            raise ValueError("CG encountered non-positive curvature")
        step = residual_product / curvature
        displacement += step * direction
        residual -= step * action
        preconditioned = _apply_preconditioner(
            green, residual, int(elements["n_standard_dofs"])
        )
        new_product = float(residual @ preconditioned)
        residual_trace.append(np.sqrt(max(new_product, 0.0)))
        iteration += 1
        converged = residual_trace[-1] <= float(tol) * np.linalg.norm(
            _effective_stress(elements, displacement)
        )
        if converged:
            residual_product = new_product
            break
        direction = preconditioned + (new_product / residual_product) * direction
        residual_product = new_product
    return {
        "displacement": displacement,
        "residual_trace": np.asarray(residual_trace, dtype=float),
        "iterations": iteration,
        "converged": bool(converged),
        "effective_stress": _effective_stress(elements, displacement),
    }

import numpy as np

def compute_effective_axial_stress(
    element_stress: np.ndarray,
    element_dofs: np.ndarray,
    stress_macro: np.ndarray,
    displacement: np.ndarray,
) -> dict:
    """Reference implementation."""
    element_stress = np.asarray(element_stress, dtype=float)
    element_dofs = np.asarray(element_dofs, dtype=int)
    stress_macro = np.asarray(stress_macro, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if element_stress.ndim != 3 or element_stress.shape[1:] != (6, 24):
        raise ValueError("element_stress must have shape (e, 6, 24)")
    if element_dofs.shape != (element_stress.shape[0], 24):
        raise ValueError("element_dofs must have shape (e, 24)")
    if stress_macro.shape != (6,) or displacement.ndim != 1:
        raise ValueError("stress_macro and displacement have invalid shapes")
    if np.any(element_dofs >= displacement.size):
        raise ValueError("element_dofs contains an out-of-range index")
    effective_stress = stress_macro.copy()
    for matrix, dofs in zip(element_stress, element_dofs):
        active = dofs >= 0
        local = np.zeros(24, dtype=float)
        local[active] = displacement[dofs[active]]
        effective_stress += matrix @ local
    return {
        "effective_stress": effective_stress,
        "effective_axial_stiffness": float(effective_stress[0]),
    }

import numpy as np

def run_xfft_pipeline(
    n_voxels: int = 3,
    radius: float = 0.31,
    center: tuple = (0.43, 0.37, 0.52),
    matrix_lame: tuple = (3.0, 2.0),
    inclusion_lame: tuple = (30.0, 20.0),
    tol: float = 1e-10,
    max_iter: int = 500,
) -> dict:
    """Reference implementation chaining every earlier step."""
    center_array = np.asarray(center, dtype=float)
    matrix_lame_array = np.asarray(matrix_lame, dtype=float)
    inclusion_lame_array = np.asarray(inclusion_lame, dtype=float)
    macrostrain = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mesh = build_periodic_sphere_mesh(  # noqa: F821
        n_voxels, radius, center_array
    )

    pointer = [0]
    barycentric = []
    weights = []
    phases = []
    for vertices, levels in zip(mesh["vertices"], mesh["levels"]):
        local = construct_subcell_quadrature(vertices, levels)  # noqa: F821
        barycentric.extend(local["barycentric"])
        weights.extend(local["weights"])
        phases.extend(local["phases"])
        pointer.append(len(weights))
    quadrature = {
        "element_ptr": np.asarray(pointer, dtype=int),
        "barycentric": np.asarray(barycentric, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "phases": np.asarray(phases, dtype=int),
    }

    first_cut = int(np.flatnonzero(mesh["cut"])[0])
    start, stop = quadrature["element_ptr"][first_cut : first_cut + 2]
    first_vertices = mesh["vertices"][first_cut]
    first_interpolation = np.column_stack([np.ones(4), first_vertices])
    first_gradients = np.linalg.inv(first_interpolation)[1:, :].T
    enrichment = evaluate_modified_abs_enrichment(  # noqa: F821
        quadrature["barycentric"][start:stop],
        mesh["levels"][first_cut],
        first_gradients,
    )
    elements = scale_and_assemble_elements(  # noqa: F821
        mesh,
        quadrature,
        matrix_lame_array,
        inclusion_lame_array,
        macrostrain,
    )
    green = build_fourier_green_operator(n_voxels)  # noqa: F821
    solution = solve_scaled_xfft_system(  # noqa: F821
        elements, green, tol, max_iter
    )
    effective = compute_effective_axial_stress(  # noqa: F821
        elements["element_stress"],
        elements["element_dofs"],
        elements["stress_macro"],
        solution["displacement"],
    )
    return {
        "mesh": mesh,
        "quadrature": quadrature,
        "first_cut_element": first_cut,
        "first_cut_enrichment": enrichment,
        "elements": elements,
        "green": green,
        "solution": solution,
        "effective": effective,
        "final_answer": float(effective["effective_axial_stiffness"]),
    }
SCICODE_GOLD_EOF
