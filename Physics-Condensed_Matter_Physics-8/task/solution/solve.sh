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


def _q4_shape_plate(xi, eta):
    """Bilinear shape functions of the reference square and their parametric gradients."""
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    value = 0.25 * (1.0 + corners[:, 0] * xi) * (1.0 + corners[:, 1] * eta)
    d_xi = 0.25 * corners[:, 0] * (1.0 + corners[:, 1] * eta)
    d_eta = 0.25 * corners[:, 1] * (1.0 + corners[:, 0] * xi)
    return value, d_xi, d_eta


def _positive_plate_constant(value, label):
    """Return a length or a modulus as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _phase_constants(values, label):
    """Return the three constants of one phase once they are known to be admissible."""
    trio = np.asarray(values, dtype=float)
    if trio.shape != (3,):
        raise ValueError("%s wants three constants" % label)
    if not np.isfinite(trio).all():
        raise ValueError("%s holds an entry that is not finite" % label)
    modulus, poisson, density = float(trio[0]), float(trio[1]), float(trio[2])
    if modulus <= 0.0 or density <= 0.0:
        raise ValueError("%s wants a modulus and a density above zero" % label)
    if not -1.0 < poisson < 0.5:
        raise ValueError("%s wants a Poisson ratio between minus one and one half" % label)
    return modulus, poisson, density


def _bending_element(side, modulus, poisson, density, thickness, shear_factor):
    """Stiffness and inertia of one square bending element with one-point shear integration."""
    shear_modulus = modulus / (2.0 * (1.0 + poisson))
    direct = modulus / (1.0 - poisson ** 2)
    cross = modulus * poisson / (1.0 - poisson ** 2)
    curvature_moduli = (thickness ** 3 / 12.0) * np.array(
        [[direct, cross, 0.0], [cross, direct, 0.0], [0.0, 0.0, shear_modulus]])
    section_inertia = density * np.array(
        [thickness, thickness ** 3 / 12.0, thickness ** 3 / 12.0])
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    jacobian = 0.25 * side * side
    stiffness = np.zeros((12, 12))
    inertia = np.zeros((12, 12))
    for xi in gauss:
        for eta in gauss:
            value, d_xi, d_eta = _q4_shape_plate(xi, eta)
            d_x, d_y = d_xi * 2.0 / side, d_eta * 2.0 / side
            curvature = np.zeros((3, 12))
            interpolation = np.zeros((3, 12))
            for node in range(4):
                base = 3 * node
                curvature[0, base + 1] = d_x[node]
                curvature[1, base + 2] = d_y[node]
                curvature[2, base + 1] = d_y[node]
                curvature[2, base + 2] = d_x[node]
                interpolation[:, base:base + 3] = value[node] * np.eye(3)
            stiffness += curvature.T @ curvature_moduli @ curvature * jacobian
            inertia += interpolation.T @ np.diag(section_inertia) @ interpolation * jacobian
    value, d_xi, d_eta = _q4_shape_plate(0.0, 0.0)
    d_x, d_y = d_xi * 2.0 / side, d_eta * 2.0 / side
    shear = np.zeros((2, 12))
    for node in range(4):
        base = 3 * node
        shear[0, base] = d_x[node]
        shear[0, base + 1] = -value[node]
        shear[1, base] = d_y[node]
        shear[1, base + 2] = -value[node]
    shear_moduli = shear_factor * shear_modulus * thickness * np.eye(2)
    stiffness += shear.T @ shear_moduli @ shear * (side * side)
    return stiffness, inertia


def assemble_plate_bending_cell(
    n_elem: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
) -> dict:
    """Reference implementation."""
    if not isinstance(n_elem, (int, np.integer)) or int(n_elem) < 2:
        raise ValueError("n_elem wants an integer of two or more")
    n_elem = int(n_elem)
    cell_side = _positive_plate_constant(cell_side, "cell_side")
    inclusion_side = _positive_plate_constant(inclusion_side, "inclusion_side")
    plate_thickness = _positive_plate_constant(plate_thickness, "plate_thickness")
    if inclusion_side >= cell_side:
        raise ValueError("the inclusion must sit inside the cell")
    factor = float(shear_factor)
    if not np.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("shear_factor wants a value above zero and at most one")
    matrix = _phase_constants(matrix_constants, "matrix_constants")
    inclusion = _phase_constants(inclusion_constants, "inclusion_constants")

    side = cell_side / n_elem
    margin = 0.5 * (cell_side - inclusion_side) / side
    if abs(margin - round(margin)) > 1.0e-9 * max(1.0, abs(margin)):
        raise ValueError("the inclusion boundary must fall on element boundaries")
    low = int(round(margin))
    high = n_elem - low
    if high <= low:
        raise ValueError("the inclusion must span at least one element")

    n_side = n_elem + 1
    n_dof = 3 * n_side * n_side
    stiffness = np.zeros((n_dof, n_dof))
    mass = np.zeros((n_dof, n_dof))
    matrix_pair = _bending_element(side, *matrix, plate_thickness, factor)
    inclusion_pair = _bending_element(side, *inclusion, plate_thickness, factor)
    n_inclusion_element = 0
    for ex in range(n_elem):
        for ey in range(n_elem):
            inside = low <= ex < high and low <= ey < high
            n_inclusion_element += int(inside)
            element_stiffness, element_mass = inclusion_pair if inside else matrix_pair
            nodes = (ex + ey * n_side, ex + 1 + ey * n_side,
                     ex + 1 + (ey + 1) * n_side, ex + (ey + 1) * n_side)
            place = np.array([3 * node + c for node in nodes for c in range(3)])
            stiffness[np.ix_(place, place)] += element_stiffness
            mass[np.ix_(place, place)] += element_mass
    return {
        "n_dof": n_dof,
        "n_inclusion_element": n_inclusion_element,
        "stiffness": stiffness,
        "mass": mass,
    }

import numpy as np


def _counted_layer_division(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def _positive_layer_length(value, label):
    """Return a length as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _hex_gradient_form(side_x, side_y, side_z):
    """Dirichlet form of one trilinear hexahedron with edges aligned to the axes."""
    corners = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    form = np.zeros((8, 8))
    jacobian = 0.125 * side_x * side_y * side_z
    for xi in gauss:
        for eta in gauss:
            for zeta in gauss:
                gradient = np.vstack([
                    0.125 * a * (1.0 + b * eta) * (1.0 + c * zeta) * 2.0 / side_x,
                    0.125 * b * (1.0 + a * xi) * (1.0 + c * zeta) * 2.0 / side_y,
                    0.125 * c * (1.0 + a * xi) * (1.0 + b * eta) * 2.0 / side_z])
                form += gradient.T @ gradient * jacobian
    return form


def assemble_fluid_potential_stiffness(
    n_elem: int,
    cell_side: float,
    fluid_depth: float,
    n_layer: int,
) -> dict:
    """Reference implementation."""
    n_elem = _counted_layer_division(n_elem, "n_elem", 2)
    n_layer = _counted_layer_division(n_layer, "n_layer", 1)
    cell_side = _positive_layer_length(cell_side, "cell_side")
    fluid_depth = _positive_layer_length(fluid_depth, "fluid_depth")

    side = cell_side / n_elem
    slice_height = fluid_depth / n_layer
    if side <= 0.0 or slice_height <= 0.0:
        raise ValueError("the division leaves an element of no extent")
    n_side = n_elem + 1
    plane = n_side * n_side
    n_dof = plane * (n_layer + 1)
    stiffness = np.zeros((n_dof, n_dof))
    element_form = _hex_gradient_form(side, side, slice_height)
    for layer in range(n_layer):
        for ex in range(n_elem):
            for ey in range(n_elem):
                lower = (ex + ey * n_side + layer * plane,
                         ex + 1 + ey * n_side + layer * plane,
                         ex + 1 + (ey + 1) * n_side + layer * plane,
                         ex + (ey + 1) * n_side + layer * plane)
                upper = tuple(node + plane for node in lower)
                place = np.array(lower + upper)
                stiffness[np.ix_(place, place)] += element_form
    return {
        "n_dof": n_dof,
        "layer_thickness": slice_height,
        "stiffness": stiffness,
    }

import numpy as np


def _q4_shape_face(xi, eta):
    """Bilinear shape functions of the reference square and their parametric gradients."""
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    value = 0.25 * (1.0 + corners[:, 0] * xi) * (1.0 + corners[:, 1] * eta)
    d_xi = 0.25 * corners[:, 0] * (1.0 + corners[:, 1] * eta)
    d_eta = 0.25 * corners[:, 1] * (1.0 + corners[:, 0] * xi)
    return value, d_xi, d_eta


def _counted_face_division(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def _positive_face_length(value, label):
    """Return a quantity as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _face_gram(side):
    """Consistent surface integral of the bilinear shape functions against one another."""
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    gram = np.zeros((4, 4))
    for xi in gauss:
        for eta in gauss:
            value, _, _ = _q4_shape_face(xi, eta)
            gram += np.outer(value, value) * 0.25 * side * side
    return gram


def assemble_interface_operators(
    n_elem: int,
    cell_side: float,
    n_layer: int,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Reference implementation."""
    n_elem = _counted_face_division(n_elem, "n_elem", 2)
    n_layer = _counted_face_division(n_layer, "n_layer", 1)
    cell_side = _positive_face_length(cell_side, "cell_side")
    fluid_density = _positive_face_length(fluid_density, "fluid_density")
    gravity = _positive_face_length(gravity, "gravity")

    side = cell_side / n_elem
    if side <= 0.0 or not np.isfinite(fluid_density * gravity):
        raise ValueError("the division or the restoring scale leaves no usable operator")
    n_side = n_elem + 1
    plane = n_side * n_side
    n_fluid = plane * (n_layer + 1)
    gram = _face_gram(side)
    fsi_coupling = np.zeros((3 * plane, n_fluid))
    surface_coupling = np.zeros((plane, n_fluid))
    surface_stiffness = np.zeros((plane, plane))
    top = n_layer * plane
    for ex in range(n_elem):
        for ey in range(n_elem):
            face = np.array([ex + ey * n_side, ex + 1 + ey * n_side,
                             ex + 1 + (ey + 1) * n_side, ex + (ey + 1) * n_side])
            deflection = 3 * face
            fsi_coupling[np.ix_(deflection, face)] += -gram
            surface_coupling[np.ix_(face, face + top)] += gram
            surface_stiffness[np.ix_(face, face)] += fluid_density * gravity * gram
    return {
        "n_surface_dof": plane,
        "fsi_coupling": fsi_coupling,
        "surface_coupling": surface_coupling,
        "surface_stiffness": surface_stiffness,
    }

import numpy as np


def _counted_cell_division(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def _positive_lattice_length(value, label):
    """Return a quantity as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _phase_map(n_elem, n_layer, n_dof_node, phase_x, phase_y):
    """Independent-partner index and phase of every full degree of freedom of one field."""
    n_side = n_elem + 1
    index = np.arange(n_side)
    along_x = np.where(index < n_elem, 1.0 + 0.0j, phase_x)
    along_y = np.where(index < n_elem, 1.0 + 0.0j, phase_y)
    folded = index % n_elem
    plane = n_side * n_side
    partner = np.empty(plane * n_layer, dtype=np.int64)
    phase = np.empty(plane * n_layer, dtype=complex)
    for layer in range(n_layer):
        for row in range(n_side):
            span = slice((row + layer * n_side) * n_side, (row + layer * n_side) * n_side + n_side)
            partner[span] = folded + folded[row] * n_elem + layer * n_elem * n_elem
            phase[span] = along_x * along_y[row]
    if n_dof_node == 1:
        return partner, phase
    spread = np.repeat(partner, n_dof_node) * n_dof_node + np.tile(np.arange(n_dof_node), partner.size)
    return spread, np.repeat(phase, n_dof_node)


def _congruence(operator, row_partner, row_phase, n_row, col_partner, col_phase, n_col):
    """Accumulate the conjugated congruence of an operator under two phase-carrying maps."""
    weighted = (row_phase.conj()[:, None] * operator) * col_phase[None, :]
    folded_rows = np.zeros((n_row, operator.shape[1]), dtype=complex)
    np.add.at(folded_rows, row_partner, weighted)
    folded = np.zeros((n_col, n_row), dtype=complex)
    np.add.at(folded, col_partner, folded_rows.T)
    return folded.T


def reduce_cell_by_bloch_phase(
    plate_stiffness: np.ndarray,
    plate_mass: np.ndarray,
    fluid_stiffness: np.ndarray,
    fsi_coupling: np.ndarray,
    surface_coupling: np.ndarray,
    surface_stiffness: np.ndarray,
    n_elem: int,
    n_layer: int,
    cell_side: float,
    wave_vector: np.ndarray,
) -> dict:
    """Reference implementation."""
    n_elem = _counted_cell_division(n_elem, "n_elem", 2)
    n_layer = _counted_cell_division(n_layer, "n_layer", 1)
    cell_side = _positive_lattice_length(cell_side, "cell_side")
    vector = np.asarray(wave_vector, dtype=float)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("wave_vector wants two finite components")

    n_side = n_elem + 1
    plane = n_side * n_side
    full_plate = 3 * plane
    full_fluid = plane * (n_layer + 1)
    shapes = {
        "plate_stiffness": (plate_stiffness, (full_plate, full_plate)),
        "plate_mass": (plate_mass, (full_plate, full_plate)),
        "fluid_stiffness": (fluid_stiffness, (full_fluid, full_fluid)),
        "fsi_coupling": (fsi_coupling, (full_plate, full_fluid)),
        "surface_coupling": (surface_coupling, (plane, full_fluid)),
        "surface_stiffness": (surface_stiffness, (plane, plane)),
    }
    for label, (array, wanted) in shapes.items():
        if np.asarray(array).shape != wanted:
            raise ValueError("%s wants shape %s" % (label, wanted))

    phase_x = np.exp(1j * vector[0] * cell_side)
    phase_y = np.exp(1j * vector[1] * cell_side)
    plate_partner, plate_phase = _phase_map(n_elem, 1, 3, phase_x, phase_y)
    fluid_partner, fluid_phase = _phase_map(n_elem, n_layer + 1, 1, phase_x, phase_y)
    surface_partner, surface_phase = _phase_map(n_elem, 1, 1, phase_x, phase_y)
    n_plate = 3 * n_elem * n_elem
    n_fluid = n_elem * n_elem * (n_layer + 1)
    n_surface = n_elem * n_elem
    return {
        "plate_stiffness": _congruence(np.asarray(plate_stiffness, dtype=float),
                                       plate_partner, plate_phase, n_plate,
                                       plate_partner, plate_phase, n_plate),
        "plate_mass": _congruence(np.asarray(plate_mass, dtype=float),
                                  plate_partner, plate_phase, n_plate,
                                  plate_partner, plate_phase, n_plate),
        "fluid_stiffness": _congruence(np.asarray(fluid_stiffness, dtype=float),
                                       fluid_partner, fluid_phase, n_fluid,
                                       fluid_partner, fluid_phase, n_fluid),
        "fsi_coupling": _congruence(np.asarray(fsi_coupling, dtype=float),
                                    plate_partner, plate_phase, n_plate,
                                    fluid_partner, fluid_phase, n_fluid),
        "surface_coupling": _congruence(np.asarray(surface_coupling, dtype=float),
                                        surface_partner, surface_phase, n_surface,
                                        fluid_partner, fluid_phase, n_fluid),
        "surface_stiffness": _congruence(np.asarray(surface_stiffness, dtype=float),
                                         surface_partner, surface_phase, n_surface,
                                         surface_partner, surface_phase, n_surface),
    }

import numpy as np
import scipy.linalg as sla


def _positive_liquid_density(value, label):
    """Return a quantity as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _finite_operator(array, label):
    """Return an operator as a complex array once every entry is known to be finite."""
    values = np.asarray(array, dtype=complex)
    if not np.isfinite(values).all():
        raise ValueError("%s holds an entry that is not finite" % label)
    return values


def condense_potential_to_mass_operators(
    fluid_stiffness: np.ndarray,
    fsi_coupling: np.ndarray,
    surface_coupling: np.ndarray,
    fluid_density: float,
) -> dict:
    """Reference implementation."""
    liquid = _finite_operator(fluid_stiffness, "fluid_stiffness")
    plate = _finite_operator(fsi_coupling, "fsi_coupling")
    surface = _finite_operator(surface_coupling, "surface_coupling")
    density = _positive_liquid_density(fluid_density, "fluid_density")
    if liquid.ndim != 2 or liquid.shape[0] != liquid.shape[1]:
        raise ValueError("fluid_stiffness wants a square operator")
    order = liquid.shape[0]
    if plate.ndim != 2 or plate.shape[1] != order:
        raise ValueError("fsi_coupling wants as many columns as fluid_stiffness has rows")
    if surface.ndim != 2 or surface.shape[1] != order:
        raise ValueError("surface_coupling wants as many columns as fluid_stiffness has rows")
    hermitian = 0.5 * (liquid + liquid.conj().T)
    if float(np.abs(liquid - hermitian).max()) > 1.0e-8 * max(1.0, float(np.abs(liquid).max())):
        raise ValueError("fluid_stiffness wants a Hermitian operator")
    try:
        factor = sla.cho_factor(hermitian, lower=True)
    except sla.LinAlgError:
        raise ValueError("fluid_stiffness is singular, so the potential cannot be eliminated")
    pivots = np.abs(np.diag(factor[0]))
    if pivots.max() <= 0.0 or pivots.min() / pivots.max() < 1.0e-6:
        raise ValueError("fluid_stiffness is singular to working precision, so the potential cannot be eliminated")
    from_plate = sla.cho_solve(factor, plate.conj().T)
    from_surface = sla.cho_solve(factor, surface.conj().T)
    return {
        "added_mass": density * (plate @ from_plate),
        "coupling_mass": density * (plate @ from_surface),
        "surface_mass": density * (surface @ from_surface),
    }

import numpy as np
import scipy.linalg as sla


def _hermitian_part(operator):
    """Return the Hermitian part of an operator."""
    return 0.5 * (operator + operator.conj().T)


def _finite_square_operator(array, label):
    """Return an operator as a complex array once every entry is known to be finite."""
    values = np.asarray(array, dtype=complex)
    if values.ndim != 2:
        raise ValueError("%s wants a two-dimensional operator" % label)
    if not np.isfinite(values).all():
        raise ValueError("%s holds an entry that is not finite" % label)
    return values


def solve_and_index_branches(
    plate_stiffness: np.ndarray,
    plate_mass: np.ndarray,
    added_mass: np.ndarray,
    coupling_mass: np.ndarray,
    surface_mass: np.ndarray,
    surface_stiffness: np.ndarray,
    model: str,
    coupling_factor: float,
) -> dict:
    """Reference implementation."""
    if model not in ("dry", "added_mass", "hydro_elastic"):
        raise ValueError("model wants one of dry, added_mass or hydro_elastic")
    factor = float(coupling_factor)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("coupling_factor wants a finite value that is not negative")
    stiffness = _finite_square_operator(plate_stiffness, "plate_stiffness")
    inertia = _finite_square_operator(plate_mass, "plate_mass")
    added = _finite_square_operator(added_mass, "added_mass")
    coupling = _finite_square_operator(coupling_mass, "coupling_mass")
    surface_inertia = _finite_square_operator(surface_mass, "surface_mass")
    restoring = _finite_square_operator(surface_stiffness, "surface_stiffness")
    n_plate = stiffness.shape[0]
    n_surface = restoring.shape[0]
    if stiffness.shape != (n_plate, n_plate) or inertia.shape != (n_plate, n_plate):
        raise ValueError("the plate operators want matching square shapes")
    if added.shape != (n_plate, n_plate):
        raise ValueError("added_mass wants the shape of the plate operators")
    if restoring.shape != (n_surface, n_surface) or surface_inertia.shape != (n_surface, n_surface):
        raise ValueError("the surface operators want matching square shapes")
    if coupling.shape != (n_plate, n_surface):
        raise ValueError("coupling_mass wants one row per plate and one column per surface freedom")

    plate_block = _hermitian_part(stiffness)
    if model == "dry":
        left, right = plate_block, _hermitian_part(inertia)
    else:
        loaded = _hermitian_part(inertia + added)
        if model == "added_mass":
            left, right = plate_block, loaded
        else:
            surface_block = _hermitian_part(surface_inertia)
            left = np.zeros((n_plate + n_surface, n_plate + n_surface), dtype=complex)
            right = np.zeros_like(left)
            left[:n_plate, :n_plate] = plate_block
            left[n_plate:, n_plate:] = _hermitian_part(restoring)
            right[:n_plate, :n_plate] = loaded
            right[:n_plate, n_plate:] = factor * coupling
            right[n_plate:, :n_plate] = factor * coupling.conj().T
            right[n_plate:, n_plate:] = surface_block
    try:
        values, vectors = sla.eigh(left, _hermitian_part(right))
    except sla.LinAlgError:
        raise ValueError("the assembled inertia is not positive definite")
    values = np.clip(np.real(values), 0.0, None)
    order = np.argsort(values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    frequencies = np.sqrt(values) / (2.0 * np.pi)
    if model != "hydro_elastic":
        return {
            "n_surface_branch": 0,
            "structural_frequencies": frequencies,
            "surface_frequencies": np.zeros(0),
        }
    plate_share = (vectors[:n_plate].conj() * (loaded @ vectors[:n_plate])).sum(axis=0).real
    surface_share = (vectors[n_plate:].conj() * (surface_block @ vectors[n_plate:])).sum(axis=0).real
    share = surface_share / (plate_share + surface_share)
    is_surface = np.zeros(frequencies.size, dtype=bool)
    is_surface[np.argsort(-share, kind="stable")[:n_surface]] = True
    return {
        "n_surface_branch": int(n_surface),
        "structural_frequencies": frequencies[~is_surface],
        "surface_frequencies": frequencies[is_surface],
    }

import numpy as np


def _ordered_table(values):
    """Return a branch table as a float array once its rows are known to be ascending."""
    table = np.asarray(values, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("branch_table wants at least two rows and two columns")
    if not np.isfinite(table).all():
        raise ValueError("branch_table holds an entry that is not finite")
    if bool((np.diff(table, axis=1) < 0.0).any()):
        raise ValueError("every row of branch_table must list the branches in ascending order")
    return table


def locate_gap_edges(branch_table: np.ndarray, lower_branch: int) -> dict:
    """Reference implementation."""
    table = _ordered_table(branch_table)
    if not isinstance(lower_branch, (int, np.integer)) or int(lower_branch) < 1:
        raise ValueError("lower_branch wants an integer of one or more")
    lower = int(lower_branch)
    if lower >= table.shape[1]:
        raise ValueError("branch_table needs a column above lower_branch")
    below = table[:, lower - 1]
    above = table[:, lower]
    lower_row = int(np.argmax(below))
    upper_row = int(np.argmin(above))
    last = table.shape[0] - 1
    on_endpoints = int(lower_row in (0, last) and upper_row in (0, last))
    every = table[:, 1:].min(axis=0) - table[:, :-1].max(axis=0)
    widest = int(np.argmax(every))
    return {
        "widest_pair_lower_branch": widest + 1,
        "widest_pair_width": float(every[widest]),
        "signed_width": float(above[upper_row] - below[lower_row]),
        "lower_branch_top": float(below[lower_row]),
        "upper_branch_bottom": float(above[upper_row]),
        "lower_extreme_row": lower_row,
        "upper_extreme_row": upper_row,
        "edges_on_endpoints": on_endpoints,
    }

import numpy as np


def _positive_depth_constant(value, label):
    """Return a length as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _counted_report_integer(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def report_sloshing_bandgap_contribution(
    graded_depth: float,
    reference_depth: float,
    lower_branch: int,
    n_step: int,
    n_elem: int,
    n_layer: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Reference implementation."""
    graded_depth = _positive_depth_constant(graded_depth, "graded_depth")
    reference_depth = _positive_depth_constant(reference_depth, "reference_depth")
    if reference_depth <= graded_depth:
        raise ValueError("reference_depth must sit above graded_depth")
    lower_branch = _counted_report_integer(lower_branch, "lower_branch", 1)
    n_step = _counted_report_integer(n_step, "n_step", 1)

    plate = assemble_plate_bending_cell(  # noqa: F821
        n_elem, cell_side, inclusion_side, plate_thickness, shear_factor,
        matrix_constants, inclusion_constants)
    faces = assemble_interface_operators(  # noqa: F821
        n_elem, cell_side, n_layer, fluid_density, gravity)
    edge = np.pi / float(cell_side)
    start = np.array([edge, 0.0])
    finish = np.array([edge, edge])
    models = ("dry", "added_mass", "hydro_elastic")
    wanted = lower_branch + 1
    outcome = {}
    for label, depth in (("graded", graded_depth), ("reference", reference_depth)):
        liquid = assemble_fluid_potential_stiffness(  # noqa: F821
            n_elem, cell_side, depth, n_layer)
        tables = {name: [] for name in models}
        corner = None
        for step in range(n_step + 1):
            vector = start + (finish - start) * (step / n_step)
            reduced = reduce_cell_by_bloch_phase(  # noqa: F821
                plate["stiffness"], plate["mass"], liquid["stiffness"], faces["fsi_coupling"],
                faces["surface_coupling"], faces["surface_stiffness"],
                n_elem, n_layer, cell_side, vector)
            masses = condense_potential_to_mass_operators(  # noqa: F821
                reduced["fluid_stiffness"], reduced["fsi_coupling"],
                reduced["surface_coupling"], fluid_density)
            for name in models:
                branches = solve_and_index_branches(  # noqa: F821
                    reduced["plate_stiffness"], reduced["plate_mass"], masses["added_mass"],
                    masses["coupling_mass"], masses["surface_mass"],
                    reduced["surface_stiffness"], name, 1.0)
                structural = branches["structural_frequencies"]
                if structural.size < wanted:
                    raise ValueError("the cell returns fewer structural branches than the pair needs")
                tables[name].append(structural)
                if name == "hydro_elastic" and step == n_step:
                    corner = branches
        edges = {}
        for name in models:
            edges[name] = locate_gap_edges(np.array(tables[name]), lower_branch)  # noqa: F821
        outcome[label] = (edges, corner)

    graded_edges, graded_corner = outcome["graded"]
    reference_edges = outcome["reference"][0]
    dry_gap = graded_edges["dry"]["signed_width"]
    if abs(dry_gap - reference_edges["dry"]["signed_width"]) > 1.0e-9 * max(1.0, abs(dry_gap)):
        raise ValueError("the dry cell must not depend on the depth of the liquid")
    graded_hydro = graded_edges["hydro_elastic"]["signed_width"]
    graded_added = graded_edges["added_mass"]["signed_width"]
    reference_hydro = reference_edges["hydro_elastic"]["signed_width"]
    reference_added = reference_edges["added_mass"]["signed_width"]
    return {
        "n_wave_vector": n_step + 1,
        "n_surface_branch": int(graded_corner["n_surface_branch"]),
        "edges_on_endpoints": int(graded_edges["hydro_elastic"]["edges_on_endpoints"]),
        "corner_surface_low": float(graded_corner["surface_frequencies"][0]),
        "corner_surface_high": float(graded_corner["surface_frequencies"][-1]),
        "corner_structural_low": float(graded_corner["structural_frequencies"][0]),
        "dry_gap": float(dry_gap),
        "graded_hydro_gap": float(graded_hydro),
        "graded_added_mass_gap": float(graded_added),
        "reference_hydro_gap": float(reference_hydro),
        "reference_added_mass_gap": float(reference_added),
        "reference_contribution": float(reference_hydro - reference_added),
        "sloshing_contribution": float(graded_hydro - graded_added),
    }
SCICODE_GOLD_EOF
