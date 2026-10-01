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


def _shear_from_bulk(bulk, poisson_ratio):
    return 3.0 * bulk * (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 + poisson_ratio))


def _equivalent_bulk(bulk_coating, bulk_inclusion, shear_coating, fraction):
    contrast = bulk_inclusion - bulk_coating
    denominator = 1.0 + (1.0 - fraction) * contrast / (bulk_coating + 4.0 * shear_coating / 3.0)
    return bulk_coating + fraction * contrast / denominator


def derive_neutral_inclusion_radius(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
) -> dict:
    """Reference implementation."""
    bulk_matrix = float(bulk_matrix)
    bulk_coating = float(bulk_coating)
    bulk_inclusion = float(bulk_inclusion)
    poisson_ratio = float(poisson_ratio)
    radius_coating = float(radius_coating)
    if min(bulk_matrix, bulk_coating, bulk_inclusion) <= 0.0:
        raise ValueError("bulk moduli must be strictly positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in the open interval (-1, 1/2)")
    if radius_coating <= 0.0:
        raise ValueError("radius_coating must be strictly positive")
    if bulk_inclusion == bulk_coating:
        raise ValueError("a neutral radius does not exist when the coating and the "
                         "inclusion share a bulk modulus")
    shear_coating = _shear_from_bulk(bulk_coating, poisson_ratio)
    contrast_ratio = (bulk_inclusion - bulk_coating) / (bulk_coating + 4.0 * shear_coating / 3.0)
    numerator = (bulk_matrix - bulk_coating) * (1.0 + contrast_ratio)
    denominator = (bulk_inclusion - bulk_coating) + (bulk_matrix - bulk_coating) * contrast_ratio
    fraction = numerator / denominator
    if not 0.0 < fraction < 1.0:
        raise ValueError("the neutrality condition has no admissible root in (0, 1)")
    return {
        "volume_fraction": fraction,
        "radius_inclusion": radius_coating * fraction ** (1.0 / 3.0),
        "effective_bulk": _equivalent_bulk(bulk_coating, bulk_inclusion, shear_coating, fraction),
    }

import numpy as np


def solve_coated_sphere_exact_field(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Reference implementation."""
    bulk_matrix = float(bulk_matrix)
    bulk_coating = float(bulk_coating)
    bulk_inclusion = float(bulk_inclusion)
    poisson_ratio = float(poisson_ratio)
    radius_inclusion = float(radius_inclusion)
    radius_coating = float(radius_coating)
    if not 0.0 < radius_inclusion < radius_coating:
        raise ValueError("the radii must satisfy 0 < radius_inclusion < radius_coating")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in the open interval (-1, 1/2)")
    shear_coating = 3.0 * bulk_coating * (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 + poisson_ratio))
    system = np.array([
        [radius_coating, 1.0 / radius_coating ** 2, 0.0],
        [3.0 * bulk_coating, -4.0 * shear_coating / radius_coating ** 3, 0.0],
        [radius_inclusion, 1.0 / radius_inclusion ** 2, -radius_inclusion],
    ])
    right = np.array([radius_coating, 3.0 * bulk_matrix, 0.0])
    coefficient_a, coefficient_b, coefficient_c = np.linalg.solve(system, right)
    residual = (3.0 * bulk_inclusion * coefficient_c
                - (3.0 * bulk_coating * coefficient_a
                   - 4.0 * shear_coating * coefficient_b / radius_inclusion ** 3))
    return {
        "coefficient_a": float(coefficient_a),
        "coefficient_b": float(coefficient_b),
        "coefficient_c": float(coefficient_c),
        "traction_residual": float(residual),
    }

import numpy as np

CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
TETRAHEDRA = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
              (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
MATRIX, COATING, INCLUSION = 0, 1, 2

def _build_mesh(n_voxels, cell_size, centre, radius_inclusion, radius_coating):
    """Periodic tetrahedral mesh with two concentric spherical interfaces."""
    n_voxels = int(n_voxels)
    spacing = float(cell_size) / n_voxels
    centre = np.asarray(centre, dtype=float)
    middle = 0.5 * (radius_inclusion + radius_coating)
    n_nodes = n_voxels ** 3
    index = np.arange(n_voxels)
    grid = np.stack(np.meshgrid(index, index, index, indexing="ij"), axis=-1).reshape(-1, 3)
    origins = grid * spacing
    corner_positions = origins[:, None, :] + spacing * CORNERS[None, :, :]
    corner_ids = ((grid[:, None, :] + CORNERS[None, :, :].astype(int)) % n_voxels)
    corner_ids = (corner_ids[..., 0] * n_voxels * n_voxels
                  + corner_ids[..., 1] * n_voxels + corner_ids[..., 2])
    radius = np.linalg.norm(corner_positions - centre, axis=-1)
    corner_levels = np.where(radius >= middle, radius - radius_coating, radius_inclusion - radius)
    corner_phase = np.where(radius >= radius_coating, MATRIX,
                            np.where(radius >= radius_inclusion, COATING, INCLUSION))
    order = np.array(TETRAHEDRA)
    vertices = corner_positions[:, order, :].reshape(-1, 4, 3)
    nodes = corner_ids[:, order].reshape(-1, 4)
    levels = corner_levels[:, order].reshape(-1, 4)
    phases = corner_phase[:, order].reshape(-1, 4)
    is_cut = (levels.min(axis=1) < 0.0) & (levels.max(axis=1) > 0.0)
    outer = np.where((phases == MATRIX).any(axis=1), MATRIX,
                     np.where((phases == INCLUSION).any(axis=1), INCLUSION, COATING))
    if np.any((phases == MATRIX).any(axis=1) & (phases == INCLUSION).any(axis=1)):
        raise ValueError("an element touches both interfaces; the configuration is invalid")
    enriched_nodes = np.unique(nodes[is_cut])
    enriched_index = -np.ones(n_nodes, dtype=int)
    enriched_index[enriched_nodes] = np.arange(enriched_nodes.size)
    return {
        "element_vertices": vertices,
        "element_nodes": nodes,
        "element_levels": levels,
        "is_cut": is_cut,
        "element_outer_phase": outer,
        "enriched_index": enriched_index,
        "n_nodes": n_nodes,
        "n_elements": vertices.shape[0],
        "n_enriched": int(enriched_nodes.size),
        "n_dof": 3 * n_nodes + 3 * int(enriched_nodes.size),
        "smallest_absolute_level": float(np.abs(levels).min()),
    }


def build_three_phase_periodic_mesh(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Reference implementation."""
    if int(n_voxels) < 2:
        raise ValueError("n_voxels must be at least two")
    if float(cell_size) <= 0.0:
        raise ValueError("cell_size must be strictly positive")
    if not 0.0 < float(radius_inclusion) < float(radius_coating):
        raise ValueError("the radii must satisfy 0 < radius_inclusion < radius_coating")
    return _build_mesh(int(n_voxels), float(cell_size), centre,
                       float(radius_inclusion), float(radius_coating))

import numpy as np

_RULE_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_RULE_B = (5.0 - np.sqrt(5.0)) / 20.0
RULE = np.array([[_RULE_A, _RULE_B, _RULE_B, _RULE_B],
                 [_RULE_B, _RULE_A, _RULE_B, _RULE_B],
                 [_RULE_B, _RULE_B, _RULE_A, _RULE_B],
                 [_RULE_B, _RULE_B, _RULE_B, _RULE_A]])

def _split_prism(lower, upper):
    """Three tetrahedra covering the prism with matched triangles lower and upper."""
    return [(lower[0], lower[1], lower[2], upper[0]),
            (lower[1], lower[2], upper[0], upper[1]),
            (lower[2], upper[0], upper[1], upper[2])]

def _minimal_subcells(levels):
    """Minimal barycentric decomposition of a tetrahedron cut by the plane L_h = 0.

    Returns a list of pairs (barycentric vertices of a sub-tetrahedron, side), where side
    is -1 where the interpolated level set is negative and +1 where it is positive. An
    uncut tetrahedron is returned as one sub-tetrahedron.
    """
    levels = np.asarray(levels, dtype=float)
    negative = [i for i in range(4) if levels[i] < 0.0]
    positive = [i for i in range(4) if levels[i] > 0.0]
    if len(negative) + len(positive) != 4:
        raise ValueError("a nodal level set value vanishes; the interface is not resolved")
    if not negative or not positive:
        return [(np.eye(4), -1 if not positive else 1)]

    def _vertex(index):
        point = np.zeros(4)
        point[index] = 1.0
        return point

    def _crossing(i, j):
        weight = levels[i] / (levels[i] - levels[j])
        point = np.zeros(4)
        point[i] = 1.0 - weight
        point[j] = weight
        return point

    cells = []
    if len(negative) == 1 or len(positive) == 1:
        lone = negative[0] if len(negative) == 1 else positive[0]
        others = positive if len(negative) == 1 else negative
        lone_side = -1 if len(negative) == 1 else 1
        cuts = [_crossing(lone, other) for other in others]
        cells.append((np.array([_vertex(lone), cuts[0], cuts[1], cuts[2]]), lone_side))
        for cell in _split_prism(cuts, [_vertex(other) for other in others]):
            cells.append((np.array(cell), -lone_side))
    else:
        first, second = negative
        third, fourth = positive
        cut_13, cut_14 = _crossing(first, third), _crossing(first, fourth)
        cut_23, cut_24 = _crossing(second, third), _crossing(second, fourth)
        for cell in _split_prism([_vertex(first), cut_13, cut_14],
                                 [_vertex(second), cut_23, cut_24]):
            cells.append((np.array(cell), -1))
        for cell in _split_prism([_vertex(third), cut_13, cut_23],
                                 [_vertex(fourth), cut_14, cut_24]):
            cells.append((np.array(cell), 1))
    return cells

def _element_quadrature(vertices, levels):
    """Barycentric quadrature points, weights and interface sides for one tetrahedron."""
    barycentric, weights, sides = [], [], []
    for cell, side in _minimal_subcells(levels):
        physical = cell @ vertices
        volume = abs(np.linalg.det((physical[1:] - physical[0]).T)) / 6.0
        barycentric.append(RULE @ cell)
        weights.extend([volume / 4.0] * 4)
        sides.extend([side] * 4)
    return np.vstack(barycentric), np.array(weights), np.array(sides, dtype=int)


def build_interface_subcell_quadrature(vertices, levels) -> dict:
    """Reference implementation."""
    vertices = np.asarray(vertices, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if vertices.shape != (4, 3) or levels.shape != (4,):
        raise ValueError("vertices must have shape (4, 3) and levels shape (4,)")
    barycentric, weights, sides = _element_quadrature(vertices, levels)
    return {"barycentric": barycentric, "weights": weights, "side": sides}

import numpy as np

def _enrichment(barycentric, levels, gradients):
    """Modified absolute enrichment and its phasewise gradient at the quadrature points."""
    levels = np.asarray(levels, dtype=float)
    interpolated = barycentric @ levels
    sign = np.where(interpolated >= 0.0, 1.0, -1.0)
    rho = barycentric @ np.abs(levels) - np.abs(interpolated)
    grad_rho = ((np.abs(levels) @ gradients)[None, :]
                - sign[:, None] * (levels @ gradients)[None, :])
    return rho, grad_rho


def evaluate_modified_abs_enrichment(barycentric, levels, gradients) -> dict:
    """Reference implementation."""
    barycentric = np.asarray(barycentric, dtype=float)
    levels = np.asarray(levels, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    if barycentric.ndim != 2 or barycentric.shape[1] != 4:
        raise ValueError("barycentric must have shape (q, 4)")
    if levels.shape != (4,) or gradients.shape != (4, 3):
        raise ValueError("levels must have shape (4,) and gradients shape (4, 3)")
    rho, grad_rho = _enrichment(barycentric, levels, gradients)
    return {"rho": rho, "grad_rho": grad_rho}

import numpy as np

R2 = np.sqrt(2.0)
MATRIX, COATING, INCLUSION = 0, 1, 2

def _strain_column(gradient, direction):
    """Mandel strain of the vector field N e_direction whose scalar gradient is given."""
    gx, gy, gz = gradient
    if direction == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / R2, gy / R2])
    if direction == 1:
        return np.array([0.0, gy, 0.0, gz / R2, 0.0, gx / R2])
    return np.array([0.0, 0.0, gz, gy / R2, gx / R2, 0.0])

def _shape_gradients(vertices):
    """Gradients of the four linear basis functions of a tetrahedron."""
    return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T


def _element_strain_operator(vertices, levels, is_cut, barycentric, enriched_scale=None):
    """Mandel strain operator of one element at its quadrature points, shape (q, 6, 24).

    Columns 0 to 11 hold the standard linear basis and columns 12 to 23 the enriched basis of
    the same four nodes. The enriched columns vanish on uncut elements. When enriched_scale is
    supplied it holds one factor per node and displacement direction.
    """
    gradients = _shape_gradients(vertices)
    operator = np.zeros((barycentric.shape[0], 6, 24))
    for node in range(4):
        for direction in range(3):
            operator[:, :, 3 * node + direction] = _strain_column(gradients[node], direction)
    if is_cut:
        enrichment = evaluate_modified_abs_enrichment(
            barycentric, levels, gradients)
        rho, grad_rho = enrichment["rho"], enrichment["grad_rho"]
        for node in range(4):
            gradient = (rho[:, None] * gradients[node][None, :]
                        + barycentric[:, node, None] * grad_rho)
            for direction in range(3):
                column = np.array([_strain_column(g, direction) for g in gradient])
                if enriched_scale is not None:
                    column = column / enriched_scale[node, direction]
                operator[:, :, 12 + 3 * node + direction] = column
    return operator


def _element_dofs(nodes, enriched_index, is_cut, n_nodes, n_dof):
    """Global degree-of-freedom indices of one element, inactive entries mapped to n_dof."""
    dofs = np.empty(24, dtype=int)
    for local, node in enumerate(nodes):
        for direction in range(3):
            dofs[3 * local + direction] = 3 * int(node) + direction
    for local, node in enumerate(nodes):
        slot = enriched_index[int(node)]
        for direction in range(3):
            dofs[12 + 3 * local + direction] = (
                3 * n_nodes + 3 * slot + direction if (is_cut and slot >= 0) else n_dof)
    return dofs


def _assemble(mesh, stiffness):
    """Internal scaling factors, element operators and quadrature of the whole cell."""
    n_elements = mesh["n_elements"]
    n_nodes = mesh["n_nodes"]
    n_dof = mesh["n_dof"]
    vertices = mesh["element_vertices"]
    levels = mesh["element_levels"]
    nodes = mesh["element_nodes"]
    is_cut = mesh["is_cut"]
    outer = mesh["element_outer_phase"]
    enriched_index = mesh["enriched_index"]

    quadrature = []
    for element in range(n_elements):
        rule = build_interface_subcell_quadrature(
            vertices[element], levels[element])
        phase = np.where(rule["side"] < 0, COATING, outer[element])
        quadrature.append((rule["barycentric"], rule["weights"], phase))

    scale = np.zeros((mesh["n_enriched"], 3))
    for element in range(n_elements):
        if not is_cut[element]:
            continue
        bary, weights, _ = quadrature[element]
        operator = _element_strain_operator(vertices[element], levels[element], True, bary)
        for node in range(4):
            slot = enriched_index[int(nodes[element, node])]
            for direction in range(3):
                column = operator[:, :, 12 + 3 * node + direction]
                scale[slot, direction] += np.sum(weights * np.einsum("qj,qj->q", column, column))
    scale = np.sqrt(scale)

    element_stiffness = np.zeros((n_elements, 24, 24))
    element_stress_map = np.zeros((n_elements, 6, 24))
    element_dofs = np.zeros((n_elements, 24), dtype=int)
    integrated_stiffness = np.zeros((6, 6))
    points, all_weights, all_phase, all_operator, owner = [], [], [], [], []
    for element in range(n_elements):
        bary, weights, phase = quadrature[element]
        local_scale = scale[enriched_index[nodes[element]]] if is_cut[element] else None
        operator = _element_strain_operator(vertices[element], levels[element],
                                            bool(is_cut[element]), bary, local_scale)
        dofs = _element_dofs(nodes[element], enriched_index, bool(is_cut[element]),
                             n_nodes, n_dof)
        weighted = weights[:, None, None] * stiffness[phase]
        element_stiffness[element] = np.einsum("qji,qjk,qkl->il", operator, weighted, operator)
        element_stress_map[element] = np.einsum("qij,qjk->ik", weighted, operator)
        integrated_stiffness += weighted.sum(axis=0)
        inactive = dofs >= n_dof
        element_stiffness[element][inactive, :] = 0.0
        element_stiffness[element][:, inactive] = 0.0
        element_stress_map[element][:, inactive] = 0.0
        element_dofs[element] = dofs
        points.append(bary @ vertices[element])
        all_weights.append(weights)
        all_phase.append(phase)
        all_operator.append(operator)
        owner.append(np.full(weights.size, element))
    return {
        "element_stiffness": element_stiffness,
        "element_stress_map": element_stress_map,
        "element_dofs": element_dofs,
        "scaling": scale,
        "integrated_stiffness": integrated_stiffness,
        "quadrature_points": np.concatenate(points),
        "quadrature_weights": np.concatenate(all_weights),
        "quadrature_phase": np.concatenate(all_phase),
        "strain_operator": np.concatenate(all_operator),
        "quadrature_element": np.concatenate(owner),
        "n_dof": n_dof,
        "n_nodes": n_nodes,
        "n_quadrature": int(np.concatenate(all_weights).size),
    }


def assemble_scaled_three_phase_system(mesh: dict, stiffness) -> dict:
    """Reference implementation."""
    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (3, 6, 6):
        raise ValueError("stiffness must have shape (3, 6, 6)")
    return _assemble(mesh, stiffness)

import numpy as np

R2 = np.sqrt(2.0)
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
TETRAHEDRA = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
              (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))

def _strain_column(gradient, direction):
    """Mandel strain of the vector field N e_direction whose scalar gradient is given."""
    gx, gy, gz = gradient
    if direction == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / R2, gy / R2])
    if direction == 1:
        return np.array([0.0, gy, 0.0, gz / R2, 0.0, gx / R2])
    return np.array([0.0, 0.0, gz, gy / R2, gx / R2, 0.0])

def _shape_gradients(vertices):
    """Gradients of the four linear basis functions of a tetrahedron."""
    return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T


def _voxel_reference_stiffness(spacing):
    """Constant-coefficient voxel stiffness of the six linear tetrahedra, Mandel identity."""
    matrix = np.zeros((24, 24))
    corners = spacing * CORNERS
    for tetrahedron in TETRAHEDRA:
        vertices = corners[list(tetrahedron)]
        gradients = _shape_gradients(vertices)
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        operator = np.zeros((6, 12))
        for node in range(4):
            for direction in range(3):
                operator[:, 3 * node + direction] = _strain_column(gradients[node], direction)
        local = volume * (operator.T @ operator)
        index = np.array([3 * tetrahedron[node] + direction
                          for node in range(4) for direction in range(3)])
        matrix[np.ix_(index, index)] += local
    return matrix


def _green(n_voxels, cell_size):
    """Block Fourier representation of the inverse constant-coefficient operator."""
    spacing = float(cell_size) / int(n_voxels)
    voxel = _voxel_reference_stiffness(spacing)
    green = np.zeros((n_voxels, n_voxels, n_voxels, 3, 3), dtype=complex)
    frequencies = np.arange(n_voxels)
    for k1 in frequencies:
        for k2 in frequencies:
            for k3 in frequencies:
                if k1 == 0 and k2 == 0 and k3 == 0:
                    continue
                phase = np.exp(2j * np.pi * (CORNERS @ np.array([k1, k2, k3])) / n_voxels)
                projector = np.zeros((24, 3), dtype=complex)
                for corner in range(8):
                    projector[3 * corner:3 * corner + 3, :] = phase[corner] * np.eye(3)
                green[k1, k2, k3] = np.linalg.inv(
                    projector.conj().T @ voxel @ projector)
    return green


def build_fourier_green_operator(n_voxels: int, cell_size: float):
    """Reference implementation."""
    if int(n_voxels) < 2:
        raise ValueError("n_voxels must be at least two")
    if float(cell_size) <= 0.0:
        raise ValueError("cell_size must be strictly positive")
    return _green(int(n_voxels), float(cell_size))

import numpy as np


def _apply_operator(operators, vector):
    """Matrix-free application of the assembled system operator."""
    padded = np.concatenate([vector, [0.0]])
    dofs = operators["element_dofs"]
    local = np.einsum("eij,ej->ei", operators["element_stiffness"], padded[dofs])
    return np.bincount(dofs.ravel(), weights=local.ravel(),
                       minlength=operators["n_dof"] + 1)[:operators["n_dof"]]


def _load(operators, macroscopic_strain):
    """Right-hand side of the equilibrium system for a prescribed macroscopic strain."""
    dofs = operators["element_dofs"]
    local = -np.einsum("eij,i->ej", operators["element_stress_map"], macroscopic_strain)
    return np.bincount(dofs.ravel(), weights=local.ravel(),
                       minlength=operators["n_dof"] + 1)[:operators["n_dof"]]


def _effective_stress(operators, vector, macroscopic_strain, cell_volume):
    """Volume-averaged Mandel stress of the current displacement fluctuation."""
    padded = np.concatenate([vector, [0.0]])
    integrated = (operators["integrated_stiffness"] @ macroscopic_strain
                  + np.einsum("eij,ej->i", operators["element_stress_map"],
                              padded[operators["element_dofs"]]))
    return integrated / cell_volume


def _precondition(operators, green, vector):
    """Apply the inverse block preconditioner, the standard block through the transform."""
    n_voxels = green.shape[0]
    n_standard = 3 * operators["n_nodes"]
    field = vector[:n_standard].reshape(n_voxels, n_voxels, n_voxels, 3)
    transformed = np.fft.fftn(field, axes=(0, 1, 2))
    filtered = np.einsum("...ij,...j->...i", green, transformed)
    standard = np.fft.ifftn(filtered, axes=(0, 1, 2)).real.reshape(-1)
    return np.concatenate([standard, vector[n_standard:]])


def _solve(operators, green, macroscopic_strain, cell_volume, tolerance, max_iterations):
    """Preconditioned linear conjugate gradients from a zero displacement fluctuation."""
    macroscopic_strain = np.asarray(macroscopic_strain, dtype=float)
    right = _load(operators, macroscopic_strain)
    displacement = np.zeros(operators["n_dof"])
    residual_vector = right.copy()
    preconditioned = _precondition(operators, green, residual_vector)
    inner = residual_vector @ preconditioned
    residuals = [float(np.sqrt(inner))]
    threshold = tolerance * float(np.linalg.norm(
        _effective_stress(operators, displacement, macroscopic_strain, cell_volume)))
    direction = preconditioned.copy()
    iterations = 0
    while residuals[-1] > threshold and iterations < max_iterations:
        applied = _apply_operator(operators, direction)
        step = inner / (direction @ applied)
        displacement = displacement + step * direction
        residual_vector = residual_vector - step * applied
        preconditioned = _precondition(operators, green, residual_vector)
        inner_new = residual_vector @ preconditioned
        residuals.append(float(np.sqrt(inner_new)))
        iterations += 1
        threshold = tolerance * float(np.linalg.norm(
            _effective_stress(operators, displacement, macroscopic_strain, cell_volume)))
        if residuals[-1] <= threshold:
            break
        direction = preconditioned + (inner_new / inner) * direction
        inner = inner_new
    converged = residuals[-1] <= threshold
    return {
        "displacement": displacement,
        "iterations": iterations,
        "residuals": np.array(residuals),
        "threshold": threshold,
        "effective_stress": _effective_stress(operators, displacement,
                                              macroscopic_strain, cell_volume),
        "converged": bool(converged),
    }


def solve_scaled_xfft_system(
    operators: dict,
    green_operator,
    macroscopic_strain,
    cell_volume: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    macroscopic_strain = np.asarray(macroscopic_strain, dtype=float)
    if macroscopic_strain.shape != (6,):
        raise ValueError("macroscopic_strain must have shape (6,)")
    if float(cell_volume) <= 0.0:
        raise ValueError("cell_volume must be strictly positive")
    if float(tolerance) <= 0.0 or int(max_iterations) < 1:
        raise ValueError("tolerance must be positive and max_iterations at least one")
    return _solve(operators, np.asarray(green_operator), macroscopic_strain,
                  float(cell_volume), float(tolerance), int(max_iterations))

import numpy as np

R2 = np.sqrt(2.0)
MATRIX, COATING, INCLUSION = 0, 1, 2

def _isotropic_stiffness(bulk, poisson_ratio):
    """Mandel stiffness matrix of an isotropic phase given its bulk modulus."""
    lame = 3.0 * bulk * poisson_ratio / (1.0 + poisson_ratio)
    shear = 3.0 * bulk * (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 + poisson_ratio))
    stiffness = lame * np.ones((6, 6))
    stiffness[3:, :] = 0.0
    stiffness[:, 3:] = 0.0
    return stiffness + 2.0 * shear * np.eye(6), shear

def _mandel(tensor):
    """Orthonormal Mandel vector of a symmetric second-order tensor field."""
    return np.stack([tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2],
                     R2 * tensor[..., 1, 2], R2 * tensor[..., 0, 2],
                     R2 * tensor[..., 0, 1]], axis=-1)


def _exact_strain_by_phase(points, centre, coefficients, phase):
    """Exact Mandel strain at points, each evaluated with the shell law of its own phase."""
    coefficient_a, coefficient_b, coefficient_c = coefficients
    offset = np.asarray(points, dtype=float) - np.asarray(centre, dtype=float)
    radius = np.linalg.norm(offset, axis=-1)
    normal = offset / np.maximum(radius, 1e-300)[:, None]
    dyad = normal[:, :, None] * normal[:, None, :]
    identity = np.eye(3)
    tensor = np.zeros((offset.shape[0], 3, 3))
    tensor[phase == MATRIX] = identity
    coating = phase == COATING
    tensor[coating] = (coefficient_a * identity
                       + (coefficient_b / radius[coating] ** 3)[:, None, None]
                       * (identity - 3.0 * dyad[coating]))
    tensor[phase == INCLUSION] = coefficient_c * identity
    return _mandel(tensor)


def compute_relative_l2_strain_error(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
    macroscopic_strain,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation, chaining the earlier steps in sequence."""
    n_voxels = int(n_voxels)
    cell_size = float(cell_size)
    macroscopic_strain = np.asarray(macroscopic_strain, dtype=float)
    if macroscopic_strain.shape != (6,):
        raise ValueError("macroscopic_strain must have shape (6,)")
    if n_voxels < 2 or cell_size <= 0.0:
        raise ValueError("n_voxels must be at least two and cell_size strictly positive")

    neutral = derive_neutral_inclusion_radius(
        bulk_matrix, bulk_coating, bulk_inclusion, poisson_ratio, radius_coating)
    radius_inclusion = neutral["radius_inclusion"]
    field = solve_coated_sphere_exact_field(
        bulk_matrix, bulk_coating, bulk_inclusion, poisson_ratio,
        radius_inclusion, radius_coating)
    coefficients = (field["coefficient_a"], field["coefficient_b"], field["coefficient_c"])

    mesh = build_three_phase_periodic_mesh(
        n_voxels, cell_size, centre, radius_inclusion, radius_coating)
    stiffness = np.array([_isotropic_stiffness(bulk, poisson_ratio)[0]
                          for bulk in (bulk_matrix, bulk_coating, bulk_inclusion)])
    operators = assemble_scaled_three_phase_system(mesh, stiffness)
    green = build_fourier_green_operator(n_voxels, cell_size)
    solution = solve_scaled_xfft_system(
        operators, green, macroscopic_strain, cell_size ** 3,
        float(tolerance), int(max_iterations))
    if not solution["converged"]:
        raise RuntimeError("the conjugate-gradient iteration did not reach the tolerance")

    padded = np.concatenate([solution["displacement"], [0.0]])
    local = padded[operators["element_dofs"]][operators["quadrature_element"]]
    discrete = macroscopic_strain[None, :] + np.einsum(
        "qij,qj->qi", operators["strain_operator"], local)
    exact = _exact_strain_by_phase(operators["quadrature_points"], centre, coefficients,
                                   operators["quadrature_phase"])
    weights = operators["quadrature_weights"]
    error_norm = float(np.sum(weights * np.sum((discrete - exact) ** 2, axis=1)))
    exact_norm = float(np.sum(weights * np.sum(exact ** 2, axis=1)))
    phase = operators["quadrature_phase"]
    return {
        "relative_error": float(np.sqrt(error_norm / exact_norm)),
        "effective_bulk_modulus": float(np.sum(solution["effective_stress"][:3]) / 9.0),
        "radius_inclusion": float(radius_inclusion),
        "coefficients": tuple(float(value) for value in coefficients),
        "traction_residual": float(field["traction_residual"]),
        "n_cut": int(mesh["is_cut"].sum()),
        "n_enriched": int(mesh["n_enriched"]),
        "n_quadrature": int(operators["n_quadrature"]),
        "phase_volumes": tuple(float(weights[phase == label].sum())
                               for label in (MATRIX, COATING, INCLUSION)),
        "iterations": int(solution["iterations"]),
        "error_norm_squared": error_norm,
        "exact_norm_squared": exact_norm,
    }
SCICODE_GOLD_EOF
