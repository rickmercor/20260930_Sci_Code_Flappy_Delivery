#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def evaluate_kernel_weights(positions: "np.ndarray", spacing: float,
                                    grid_offsets: tuple, kernel_name: str) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _profile(name, d):
        """Return the one-dimensional kernel and its derivative in d."""
        magnitude = np.abs(d)
        if name == "compact":
            inside = magnitude <= 1.0
            value = np.where(inside,
                             1.0 - magnitude + np.sin(2.0 * np.pi * magnitude) / (2.0 * np.pi),
                             0.0)
            # The analytic profile is nonnegative and exactly zero at |d|=1.
            # Roundoff in sin(2*pi) can otherwise create a tiny negative nodal
            # mass when a particle lies exactly on a lattice node.
            value = np.maximum(value, 0.0)
            slope = np.where(inside, np.sign(d) * (np.cos(2.0 * np.pi * d) - 1.0), 0.0)
            return value, slope
        near = magnitude < 0.5
        far = (magnitude >= 0.5) & (magnitude <= 1.5)
        value = np.where(near, 0.75 - d * d, np.where(far, 0.5 * (1.5 - magnitude) ** 2, 0.0))
        slope = np.where(near, -2.0 * d, np.where(far, -np.sign(d) * (1.5 - magnitude), 0.0))
        return value, slope

    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("positions must have shape (n_particles, 2)")
    if coordinates.shape[0] == 0:
        raise ValueError("positions must hold at least one particle")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("positions must contain only finite entries")
    if not (isinstance(spacing, (int, float)) and np.isfinite(spacing) and float(spacing) > 0.0):
        raise ValueError("spacing must be a finite number > 0")
    offsets = np.asarray(grid_offsets, dtype=float).ravel()
    if offsets.size == 0:
        raise ValueError("grid_offsets must hold at least one offset")
    if not np.all(np.isfinite(offsets)) or np.any(offsets < 0.0) or np.any(offsets >= 1.0):
        raise ValueError("every entry of grid_offsets must satisfy 0 <= offset < 1")
    if kernel_name not in ("compact", "quadratic"):
        raise ValueError("kernel_name must be 'compact' or 'quadratic'")

    spacing = float(spacing)
    side = 2 if kernel_name == "compact" else 3
    shift = 0.0 if side == 2 else 0.5
    n_particles = coordinates.shape[0]
    n_grids = offsets.size
    n_slots = side * side

    node_indices = np.zeros((n_particles, n_grids, n_slots, 2), dtype=np.int64)
    weights = np.zeros((n_particles, n_grids, n_slots))
    weight_gradients = np.zeros((n_particles, n_grids, n_slots, 2))
    node_offsets = np.zeros((n_particles, n_grids, n_slots, 2))

    for grid, offset in enumerate(offsets):
        base = np.floor(coordinates / spacing - offset - shift).astype(np.int64)
        for first in range(side):
            for second in range(side):
                slot = side * first + second
                index = base + np.array([first, second], dtype=np.int64)
                separation = (index + offset) * spacing - coordinates
                reduced = separation / spacing
                value_x, slope_x = _profile(kernel_name, reduced[:, 0])
                value_y, slope_y = _profile(kernel_name, reduced[:, 1])
                node_indices[:, grid, slot, :] = index
                node_offsets[:, grid, slot, :] = separation
                weights[:, grid, slot] = value_x * value_y
                # The reduced coordinate falls as the particle advances, hence
                # the sign that turns the profile slope into a gradient in the
                # particle position.
                weight_gradients[:, grid, slot, 0] = -slope_x * value_y / spacing
                weight_gradients[:, grid, slot, 1] = -value_x * slope_y / spacing

    return node_indices, weights, weight_gradients, node_offsets

import numpy as np


def transfer_particles_to_grid(masses: "np.ndarray", velocities: "np.ndarray",
                                       affine_states: "np.ndarray", node_indices: "np.ndarray",
                                       weights: "np.ndarray", node_offsets: "np.ndarray") -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _finite(array, name):
        """Return the argument as a finite float array or raise."""
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")
        return values

    weight = _finite(weights, "weights")
    if weight.ndim != 3:
        raise ValueError("weights must have shape (n_particles, n_grids, n_slots)")
    n_particles, n_grids, n_slots = weight.shape
    if n_particles == 0 or n_grids == 0 or n_slots == 0:
        raise ValueError("weights must have a non-zero extent in every direction")

    mass = _finite(masses, "masses").ravel()
    if mass.shape[0] != n_particles:
        raise ValueError("masses must hold one entry per particle")
    if np.any(mass <= 0.0):
        raise ValueError("every particle mass must be > 0")
    velocity = _finite(velocities, "velocities")
    if velocity.shape != (n_particles, 2):
        raise ValueError("velocities must have shape (n_particles, 2)")
    affine = _finite(affine_states, "affine_states")
    if affine.shape != (n_particles, 2, 2):
        raise ValueError("affine_states must have shape (n_particles, 2, 2)")
    offset = _finite(node_offsets, "node_offsets")
    if offset.shape != (n_particles, n_grids, n_slots, 2):
        raise ValueError("node_offsets must have shape (n_particles, n_grids, n_slots, 2)")
    index = np.asarray(node_indices)
    if index.shape != (n_particles, n_grids, n_slots, 2):
        raise ValueError("node_indices must have shape (n_particles, n_grids, n_slots, 2)")
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError("node_indices must be an integer array")

    # -- Active node table: every stencil entry is labelled by its grid and its
    #    two lattice indices, and the unique labels are taken in sorted order.
    total = n_grids * n_slots
    grid_label = np.repeat(np.arange(n_grids, dtype=np.int64), n_slots)
    labels = np.concatenate(
        [np.broadcast_to(grid_label[None, :, None], (n_particles, total, 1)),
         index.reshape(n_particles, total, 2).astype(np.int64)], axis=2)
    node_keys, inverse = np.unique(labels.reshape(-1, 3), axis=0, return_inverse=True)
    node_slots = np.asarray(inverse, dtype=np.int64).reshape(n_particles, n_grids, n_slots)
    n_nodes = node_keys.shape[0]

    flat_weight = weight.reshape(n_particles, total)
    flat_offset = offset.reshape(n_particles, total, 2)

    # -- Affine moment matrix, averaged over the grids of the family.
    moment = np.einsum('pm,pma,pmb->pab', flat_weight, flat_offset, flat_offset) / n_grids
    determinant = moment[:, 0, 0] * moment[:, 1, 1] - moment[:, 0, 1] * moment[:, 1, 0]
    if np.any(np.abs(determinant) <= 0.0) or not np.all(np.isfinite(determinant)):
        raise ValueError("an affine moment matrix is singular")
    affine_map = affine @ np.linalg.inv(moment)

    # -- Deposit mass and affine momentum on the active nodes.
    node_masses = np.zeros(n_nodes)
    momentum = np.zeros((n_nodes, 2))
    rows = node_slots.reshape(n_particles, total).ravel()
    np.add.at(node_masses, rows, (flat_weight * mass[:, None]).ravel())
    carried = flat_weight[:, :, None] * mass[:, None, None] * (
        velocity[:, None, :] + np.einsum('pab,pmb->pma', affine_map, flat_offset))
    np.add.at(momentum, rows, carried.reshape(-1, 2))

    occupied = node_masses > 0.0
    node_velocities = np.zeros((n_nodes, 2))
    node_velocities[occupied] = momentum[occupied] / node_masses[occupied, None]
    return node_keys, node_slots, node_masses, node_velocities

def interpolate_grid_velocity(node_velocities: "np.ndarray", node_slots: "np.ndarray",
                                      weights: "np.ndarray", weight_gradients: "np.ndarray",
                                      node_offsets: "np.ndarray") -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _finite(array, name):
        """Return the argument as a finite float array or raise."""
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")
        return values

    nodal = _finite(node_velocities, "node_velocities")
    if nodal.ndim != 2 or nodal.shape[1] != 2:
        raise ValueError("node_velocities must have shape (n_nodes, 2)")
    if nodal.shape[0] == 0:
        raise ValueError("node_velocities must hold at least one node")

    weight = _finite(weights, "weights")
    if weight.ndim != 3:
        raise ValueError("weights must have shape (n_particles, n_grids, n_slots)")
    n_particles, n_grids, n_slots = weight.shape
    if n_particles == 0 or n_grids == 0 or n_slots == 0:
        raise ValueError("weights must have a non-zero extent in every direction")

    gradient = _finite(weight_gradients, "weight_gradients")
    if gradient.shape != (n_particles, n_grids, n_slots, 2):
        raise ValueError("weight_gradients must have shape (n_particles, n_grids, n_slots, 2)")
    offset = _finite(node_offsets, "node_offsets")
    if offset.shape != (n_particles, n_grids, n_slots, 2):
        raise ValueError("node_offsets must have shape (n_particles, n_grids, n_slots, 2)")

    slots = np.asarray(node_slots)
    if slots.shape != (n_particles, n_grids, n_slots):
        raise ValueError("node_slots must have shape (n_particles, n_grids, n_slots)")
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("node_slots must be an integer array")
    if slots.min() < 0 or slots.max() >= nodal.shape[0]:
        raise ValueError("node_slots refers to a node outside node_velocities")

    total = n_grids * n_slots
    flat_weight = weight.reshape(n_particles, total)
    flat_gradient = gradient.reshape(n_particles, total, 2)
    flat_offset = offset.reshape(n_particles, total, 2)
    gathered = nodal[slots.reshape(n_particles, total)]

    particle_velocities = np.einsum('pm,pma->pa', flat_weight, gathered) / n_grids
    velocity_gradients = np.einsum('pma,pmb->pab', gathered, flat_gradient) / n_grids
    affine_states = np.einsum('pm,pma,pmb->pab', flat_weight, gathered, flat_offset) / n_grids
    return particle_velocities, velocity_gradients, affine_states

def evaluate_corotated_response(deformation_gradients: "np.ndarray", shear_modulus: float,
                                        lame_first: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    gradients = np.asarray(deformation_gradients, dtype=float)
    if gradients.ndim != 3 or gradients.shape[1:] != (2, 2):
        raise ValueError("deformation_gradients must have shape (n_particles, 2, 2)")
    if gradients.shape[0] == 0:
        raise ValueError("deformation_gradients must hold at least one particle")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("deformation_gradients must contain only finite entries")
    if not (isinstance(shear_modulus, (int, float)) and np.isfinite(shear_modulus)
            and float(shear_modulus) > 0.0):
        raise ValueError("shear_modulus must be a finite number > 0")
    if not (isinstance(lame_first, (int, float)) and np.isfinite(lame_first)
            and float(lame_first) >= 0.0):
        raise ValueError("lame_first must be a finite number >= 0")

    shear = float(shear_modulus)
    lame = float(lame_first)
    determinant = (gradients[:, 0, 0] * gradients[:, 1, 1]
                   - gradients[:, 0, 1] * gradients[:, 1, 0])
    if np.any(determinant <= 0.0):
        raise ValueError("every deformation gradient must have a determinant > 0")

    # -- Rotation of the polar decomposition. In two dimensions the symmetric
    #    part of the trace and the antisymmetric part of the off-diagonal fix
    #    the rotation angle without any eigenvalue computation.
    trace_part = gradients[:, 0, 0] + gradients[:, 1, 1]
    spin_part = gradients[:, 0, 1] - gradients[:, 1, 0]
    scale = np.sqrt(trace_part * trace_part + spin_part * spin_part)
    if np.any(scale <= 0.0):
        raise ValueError("a deformation gradient has no admissible polar rotation")
    rotation = np.empty_like(gradients)
    rotation[:, 0, 0] = trace_part / scale
    rotation[:, 0, 1] = spin_part / scale
    rotation[:, 1, 0] = -spin_part / scale
    rotation[:, 1, 1] = trace_part / scale

    # -- Cofactor, which is the derivative of the determinant.
    cofactor = np.empty_like(gradients)
    cofactor[:, 0, 0] = gradients[:, 1, 1]
    cofactor[:, 0, 1] = -gradients[:, 1, 0]
    cofactor[:, 1, 0] = -gradients[:, 0, 1]
    cofactor[:, 1, 1] = gradients[:, 0, 0]

    departure = gradients - rotation
    energy_densities = (shear * np.sum(departure * departure, axis=(1, 2))
                        + 0.5 * lame * (determinant - 1.0) ** 2)
    first_piola = (2.0 * shear * departure
                   + lame * (determinant - 1.0)[:, None, None] * cofactor)
    return energy_densities, first_piola

def evaluate_corotated_tangent(deformation_gradients: "np.ndarray", shear_modulus: float,
                                       lame_first: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    gradients = np.asarray(deformation_gradients, dtype=float)
    if gradients.ndim != 3 or gradients.shape[1:] != (2, 2):
        raise ValueError("deformation_gradients must have shape (n_particles, 2, 2)")
    if gradients.shape[0] == 0:
        raise ValueError("deformation_gradients must hold at least one particle")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("deformation_gradients must contain only finite entries")
    if not (isinstance(shear_modulus, (int, float)) and np.isfinite(shear_modulus)
            and float(shear_modulus) > 0.0):
        raise ValueError("shear_modulus must be a finite number > 0")
    if not (isinstance(lame_first, (int, float)) and np.isfinite(lame_first)
            and float(lame_first) >= 0.0):
        raise ValueError("lame_first must be a finite number >= 0")

    shear = float(shear_modulus)
    lame = float(lame_first)
    n_particles = gradients.shape[0]
    determinant = (gradients[:, 0, 0] * gradients[:, 1, 1]
                   - gradients[:, 0, 1] * gradients[:, 1, 0])
    if np.any(determinant <= 0.0):
        raise ValueError("every deformation gradient must have a determinant > 0")

    trace_part = gradients[:, 0, 0] + gradients[:, 1, 1]
    spin_part = gradients[:, 0, 1] - gradients[:, 1, 0]
    scale = np.sqrt(trace_part * trace_part + spin_part * spin_part)
    if np.any(scale <= 0.0):
        raise ValueError("a deformation gradient has no admissible polar rotation")
    rotation = np.empty_like(gradients)
    rotation[:, 0, 0] = trace_part / scale
    rotation[:, 0, 1] = spin_part / scale
    rotation[:, 1, 0] = -spin_part / scale
    rotation[:, 1, 1] = trace_part / scale

    # -- Derivative of the polar rotation. Writing the rotation increment as an
    #    angle times the rotation turned by a quarter turn, the angle increment
    #    follows from the antisymmetric part of the rotation transpose times the
    #    increment of the deformation gradient, divided by the trace of the
    #    symmetric stretch.
    stretch_trace = np.einsum('pba,pba->p', rotation, gradients)
    if np.any(stretch_trace <= 0.0):
        raise ValueError("a deformation gradient has a non-positive stretch trace")
    quarter_turn = np.stack([rotation[:, :, 1], -rotation[:, :, 0]], axis=2)
    angle_derivative = np.empty((n_particles, 2, 2))
    angle_derivative[:, :, 0] = rotation[:, :, 1] / stretch_trace[:, None]
    angle_derivative[:, :, 1] = -rotation[:, :, 0] / stretch_trace[:, None]
    rotation_derivative = np.einsum('pab,pcd->pabcd', quarter_turn, angle_derivative)

    identity = np.zeros((2, 2, 2, 2))
    for first in range(2):
        for second in range(2):
            identity[first, second, first, second] = 1.0

    cofactor = np.empty_like(gradients)
    cofactor[:, 0, 0] = gradients[:, 1, 1]
    cofactor[:, 0, 1] = -gradients[:, 1, 0]
    cofactor[:, 1, 0] = -gradients[:, 0, 1]
    cofactor[:, 1, 1] = gradients[:, 0, 0]

    # -- Derivative of the cofactor, which in two dimensions is a constant
    #    fourth-order array exchanging the two diagonal and the two
    #    off-diagonal entries with a sign.
    cofactor_derivative = np.zeros((2, 2, 2, 2))
    cofactor_derivative[0, 0, 1, 1] = 1.0
    cofactor_derivative[0, 1, 1, 0] = -1.0
    cofactor_derivative[1, 0, 0, 1] = -1.0
    cofactor_derivative[1, 1, 0, 0] = 1.0

    tangents = 2.0 * shear * (identity[None] - rotation_derivative)
    tangents = tangents + lame * np.einsum('pab,pcd->pabcd', cofactor, cofactor)
    tangents = tangents + lame * (determinant - 1.0)[:, None, None, None, None] * cofactor_derivative[None]
    return tangents

import numpy as np
def assemble_potential_gradient(trial_velocities: "np.ndarray", initial_velocities: "np.ndarray",
                                        node_masses: "np.ndarray", node_slots: "np.ndarray",
                                        weight_gradients: "np.ndarray", reference_gradients: "np.ndarray",
                                        energy_densities: "np.ndarray", first_piola: "np.ndarray",
                                        volumes: "np.ndarray", step: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _finite(array, name):
        """Return the argument as a finite float array or raise."""
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")
        return values

    trial = _finite(trial_velocities, "trial_velocities")
    if trial.ndim != 2 or trial.shape[1] != 2:
        raise ValueError("trial_velocities must have shape (n_nodes, 2)")
    n_nodes = trial.shape[0]
    if n_nodes == 0:
        raise ValueError("trial_velocities must hold at least one node")
    start = _finite(initial_velocities, "initial_velocities")
    if start.shape != (n_nodes, 2):
        raise ValueError("initial_velocities must have shape (n_nodes, 2)")
    mass = _finite(node_masses, "node_masses").ravel()
    if mass.shape[0] != n_nodes:
        raise ValueError("node_masses must hold one entry per node")
    if np.any(mass < 0.0):
        raise ValueError("every nodal mass must be >= 0")

    gradient = _finite(weight_gradients, "weight_gradients")
    if gradient.ndim != 4 or gradient.shape[3] != 2:
        raise ValueError("weight_gradients must have shape (n_particles, n_grids, n_slots, 2)")
    n_particles, n_grids, n_slots = gradient.shape[:3]
    if n_particles == 0 or n_grids == 0 or n_slots == 0:
        raise ValueError("weight_gradients must have a non-zero extent in every direction")
    slots = np.asarray(node_slots)
    if slots.shape != (n_particles, n_grids, n_slots):
        raise ValueError("node_slots must have shape (n_particles, n_grids, n_slots)")
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("node_slots must be an integer array")
    if slots.min() < 0 or slots.max() >= n_nodes:
        raise ValueError("node_slots refers to a node outside the nodal arrays")

    reference = _finite(reference_gradients, "reference_gradients")
    if reference.shape != (n_particles, 2, 2):
        raise ValueError("reference_gradients must have shape (n_particles, 2, 2)")
    density = _finite(energy_densities, "energy_densities").ravel()
    if density.shape[0] != n_particles:
        raise ValueError("energy_densities must hold one entry per particle")
    stress = _finite(first_piola, "first_piola")
    if stress.shape != (n_particles, 2, 2):
        raise ValueError("first_piola must have shape (n_particles, 2, 2)")
    volume = _finite(volumes, "volumes").ravel()
    if volume.shape[0] != n_particles:
        raise ValueError("volumes must hold one entry per particle")
    if np.any(volume <= 0.0):
        raise ValueError("every reference volume must be > 0")
    if not (isinstance(step, (int, float)) and np.isfinite(step) and float(step) > 0.0):
        raise ValueError("step must be a finite number > 0")

    step = float(step)
    total = n_grids * n_slots
    flat_gradient = gradient.reshape(n_particles, total, 2)
    rows = slots.reshape(n_particles, total)

    # -- Pulling the weight gradient back through the reference deformation
    #    gradient is what turns the first Piola stress into a nodal force.
    pulled = np.einsum('pab,pma->pmb', reference, flat_gradient)
    nodal_force = step * volume[:, None, None] * np.einsum('pcb,pmb->pmc', stress, pulled)

    departure = trial - start
    residual = mass[:, None] * departure
    np.add.at(residual, rows.ravel(), nodal_force.reshape(-1, 2))

    potential = (0.5 * float(np.sum(mass[:, None] * departure * departure))
                 + n_grids * float(np.sum(volume * density)))
    return potential, residual

def assemble_potential_hessian(tangents: "np.ndarray", reference_gradients: "np.ndarray",
                                       weight_gradients: "np.ndarray", node_slots: "np.ndarray",
                                       node_masses: "np.ndarray", volumes: "np.ndarray",
                                       step: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _finite(array, name):
        """Return the argument as a finite float array or raise."""
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")
        return values

    material = _finite(tangents, "tangents")
    if material.ndim != 5 or material.shape[1:] != (2, 2, 2, 2):
        raise ValueError("tangents must have shape (n_particles, 2, 2, 2, 2)")
    n_particles = material.shape[0]
    if n_particles == 0:
        raise ValueError("tangents must hold at least one particle")

    gradient = _finite(weight_gradients, "weight_gradients")
    if gradient.ndim != 4 or gradient.shape[0] != n_particles or gradient.shape[3] != 2:
        raise ValueError("weight_gradients must have shape (n_particles, n_grids, n_slots, 2)")
    n_grids, n_slots = gradient.shape[1:3]
    if n_grids == 0 or n_slots == 0:
        raise ValueError("weight_gradients must have a non-zero extent in every direction")

    mass = _finite(node_masses, "node_masses").ravel()
    n_nodes = mass.shape[0]
    if n_nodes == 0:
        raise ValueError("node_masses must hold at least one node")
    if np.any(mass < 0.0):
        raise ValueError("every nodal mass must be >= 0")
    slots = np.asarray(node_slots)
    if slots.shape != (n_particles, n_grids, n_slots):
        raise ValueError("node_slots must have shape (n_particles, n_grids, n_slots)")
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("node_slots must be an integer array")
    if slots.min() < 0 or slots.max() >= n_nodes:
        raise ValueError("node_slots refers to a node outside the nodal arrays")

    reference = _finite(reference_gradients, "reference_gradients")
    if reference.shape != (n_particles, 2, 2):
        raise ValueError("reference_gradients must have shape (n_particles, 2, 2)")
    volume = _finite(volumes, "volumes").ravel()
    if volume.shape[0] != n_particles:
        raise ValueError("volumes must hold one entry per particle")
    if np.any(volume <= 0.0):
        raise ValueError("every reference volume must be > 0")
    if not (isinstance(step, (int, float)) and np.isfinite(step) and float(step) > 0.0):
        raise ValueError("step must be a finite number > 0")

    step = float(step)
    total = n_grids * n_slots
    flat_gradient = gradient.reshape(n_particles, total, 2)
    rows = slots.reshape(n_particles, total)

    # -- Pull the weight gradients back through the reference deformation
    #    gradient, then contract two of them against the material tangent. Each
    #    of the two contractions brings one factor of the step and one factor of
    #    the averaging over the grids, and the strain term of the potential
    #    contributes one factor equal to the number of grids.
    pulled = np.einsum('pab,pma->pmb', reference, flat_gradient)
    blocks = (step * step / n_grids) * volume[:, None, None, None, None] * np.einsum(
        'pcbef,pmb,pnf->pmcne', material, pulled, pulled)

    degrees = 2 * rows[:, :, None] + np.arange(2)[None, None, :]
    row_index = np.broadcast_to(degrees[:, :, :, None, None],
                                (n_particles, total, 2, total, 2))
    column_index = np.broadcast_to(degrees[:, None, None, :, :],
                                   (n_particles, total, 2, total, 2))
    hessian = np.zeros((2 * n_nodes, 2 * n_nodes))
    np.add.at(hessian, (row_index.ravel(), column_index.ravel()), blocks.ravel())
    diagonal = np.arange(2 * n_nodes)
    hessian[diagonal, diagonal] += np.repeat(mass, 2)
    return hessian

def solve_newton_direction(residual: "np.ndarray", hessian: "np.ndarray") -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    gradient = np.asarray(residual, dtype=float)
    if gradient.ndim != 2 or gradient.shape[1] != 2:
        raise ValueError("residual must have shape (n_nodes, 2)")
    n_nodes = gradient.shape[0]
    if n_nodes == 0:
        raise ValueError("residual must hold at least one node")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("residual must contain only finite entries")

    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape != (2 * n_nodes, 2 * n_nodes):
        raise ValueError("hessian must have shape (2 * n_nodes, 2 * n_nodes)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hessian must contain only finite entries")

    symmetric = 0.5 * (matrix + matrix.T)
    order = symmetric.shape[0]
    diagonal = np.arange(order)
    mean_diagonal = float(np.mean(symmetric[diagonal, diagonal]))
    if mean_diagonal > 0.0:
        base = 1.0e-3 * mean_diagonal
    else:
        largest = float(np.max(np.abs(symmetric)))
        base = 1.0e-3 * largest if largest > 0.0 else 1.0
    right_hand_side = -gradient.ravel()

    shift = 0.0
    for attempt in range(61):
        shifted = symmetric.copy()
        if shift > 0.0:
            shifted[diagonal, diagonal] += shift
        try:
            # The factorization is used only as the test for positive
            # definiteness; the increment itself comes from the shifted system.
            np.linalg.cholesky(shifted)
            increment = np.linalg.solve(shifted, right_hand_side)
        except np.linalg.LinAlgError:
            shift = base if attempt == 0 else 2.0 * shift
            continue
        if not np.all(np.isfinite(increment)):
            raise ValueError("the tangent system produced a non-finite increment")
        return increment.reshape(n_nodes, 2)

    raise ValueError("no shift within sixty doublings made the tangent factorizable")

def advance_particle_state(positions: "np.ndarray", deformation_gradients: "np.ndarray",
                                   particle_velocities: "np.ndarray", velocity_gradients: "np.ndarray",
                                   step: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _finite(array, name):
        """Return the argument as a finite float array or raise."""
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")
        return values

    coordinates = _finite(positions, "positions")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("positions must have shape (n_particles, 2)")
    n_particles = coordinates.shape[0]
    if n_particles == 0:
        raise ValueError("positions must hold at least one particle")

    gradients = _finite(deformation_gradients, "deformation_gradients")
    if gradients.shape != (n_particles, 2, 2):
        raise ValueError("deformation_gradients must have shape (n_particles, 2, 2)")
    velocity = _finite(particle_velocities, "particle_velocities")
    if velocity.shape != (n_particles, 2):
        raise ValueError("particle_velocities must have shape (n_particles, 2)")
    rate = _finite(velocity_gradients, "velocity_gradients")
    if rate.shape != (n_particles, 2, 2):
        raise ValueError("velocity_gradients must have shape (n_particles, 2, 2)")
    if not (isinstance(step, (int, float)) and np.isfinite(step) and float(step) > 0.0):
        raise ValueError("step must be a finite number > 0")

    step = float(step)
    increment = np.eye(2)[None] + step * rate
    updated_gradients = np.einsum('pab,pbc->pac', increment, gradients)
    determinant = (updated_gradients[:, 0, 0] * updated_gradients[:, 1, 1]
                   - updated_gradients[:, 0, 1] * updated_gradients[:, 1, 0])
    if np.any(determinant <= 0.0):
        raise ValueError("the update left a particle with a non-positive determinant")

    updated_positions = coordinates + step * velocity
    return updated_positions, updated_gradients

import numpy as np
def measure_mechanical_energy(masses: "np.ndarray", particle_velocities: "np.ndarray",
                                      volumes: "np.ndarray", energy_densities: "np.ndarray",
                                      positions: "np.ndarray") -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _finite(array, name):
        """Return the argument as a finite float array or raise."""
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")
        return values

    mass = _finite(masses, "masses").ravel()
    n_particles = mass.shape[0]
    if n_particles == 0:
        raise ValueError("masses must hold at least one particle")
    if np.any(mass <= 0.0):
        raise ValueError("every particle mass must be > 0")

    velocity = _finite(particle_velocities, "particle_velocities")
    if velocity.shape != (n_particles, 2):
        raise ValueError("particle_velocities must have shape (n_particles, 2)")
    volume = _finite(volumes, "volumes").ravel()
    if volume.shape[0] != n_particles:
        raise ValueError("volumes must hold one entry per particle")
    if np.any(volume <= 0.0):
        raise ValueError("every reference volume must be > 0")
    density = _finite(energy_densities, "energy_densities").ravel()
    if density.shape[0] != n_particles:
        raise ValueError("energy_densities must hold one entry per particle")
    if np.any(density < 0.0):
        raise ValueError("every strain energy density must be >= 0")
    coordinates = _finite(positions, "positions")
    if coordinates.shape != (n_particles, 2):
        raise ValueError("positions must have shape (n_particles, 2)")

    kinetic_energy = 0.5 * float(np.sum(mass * np.sum(velocity * velocity, axis=1)))
    strain_energy = float(np.sum(volume * density))
    total_energy = kinetic_energy + strain_energy

    centre = np.sum(mass[:, None] * coordinates, axis=0) / np.sum(mass)
    lever = coordinates - centre
    angular_momentum = float(np.sum(mass * (lever[:, 0] * velocity[:, 1]
                                            - lever[:, 1] * velocity[:, 0])))
    return kinetic_energy, strain_energy, total_energy, angular_momentum

# -- The end-to-end march. Every earlier step is called through its own oracle,
#    so the value returned here never depends on a candidate's steps 01 to 10.
def run_ck_mpm_energy_loss(spacing: float = 0.02, cells: int = 10,
                                   particles_per_cell: int = 4, pre_stretch: float = 1.20,
                                   density: float = 1200.0, shear_speed: float = 6.0,
                                   dilatational_speed: float = 14.0, step: float = 1.0e-3,
                                   n_steps: int = 20, tolerance: float = 1.0e-10,
                                   max_iterations: int = 8, kernel_name: str = "compact",
                                   grid_offsets: tuple = (0.0, 0.5)) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value, floor in (("cells", cells, 1), ("particles_per_cell", particles_per_cell, 1),
                               ("n_steps", n_steps, 0), ("max_iterations", max_iterations, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("spacing", spacing), ("pre_stretch", pre_stretch),
                        ("density", density), ("shear_speed", shear_speed),
                        ("dilatational_speed", dilatational_speed), ("step", step),
                        ("tolerance", tolerance)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    side = int(np.round(np.sqrt(float(particles_per_cell))))
    if side * side != int(particles_per_cell):
        raise ValueError("particles_per_cell must be a perfect square")
    if float(dilatational_speed) <= np.sqrt(2.0) * float(shear_speed):
        raise ValueError("dilatational_speed must exceed the shear speed times the square root of two")
    if kernel_name not in ("compact", "quadratic"):
        raise ValueError("kernel_name must be 'compact' or 'quadratic'")

    spacing = float(spacing)
    cells = int(cells)
    n_steps = int(n_steps)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)
    shear = float(density) * float(shear_speed) ** 2
    lame = float(density) * (float(dilatational_speed) ** 2 - 2.0 * float(shear_speed) ** 2)

    # -- Initial particle state: a square block at rest carrying an isochoric
    #    pre-strain, so all of its mechanical energy is strain energy.
    fractions = (np.arange(side) + 0.5) / side
    line = (np.arange(cells)[:, None] + fractions[None, :]).ravel() * spacing
    grid_x, grid_y = np.meshgrid(line, line, indexing='ij')
    positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    count = positions.shape[0]
    volumes = np.full(count, spacing * spacing / float(particles_per_cell))
    masses = float(density) * volumes
    gradients = np.zeros((count, 2, 2))
    gradients[:, 0, 0] = float(pre_stretch)
    gradients[:, 1, 1] = 1.0 / float(pre_stretch)
    velocities = np.zeros((count, 2))
    affine = np.zeros((count, 2, 2))
    identity = np.eye(2)[None]

    # -- Sub-problems 04 and 10: the mechanical energy before the march.
    density_before = evaluate_corotated_response(gradients, shear, lame)[0]
    initial = measure_mechanical_energy(masses, velocities, volumes,
                                                density_before, positions)
    if float(initial[2]) <= 0.0:
        raise ValueError("the initial state carries no mechanical energy")

    for _ in range(n_steps):
        # -- Sub-problems 01 and 02: the stencil and the deposit.
        stencil = evaluate_kernel_weights(positions, spacing, grid_offsets, kernel_name)
        node_indices, weights, weight_gradients, node_offsets = stencil
        deposited = transfer_particles_to_grid(masses, velocities, affine, node_indices,
                                                       weights, node_offsets)
        node_slots, node_masses, node_velocities = deposited[1], deposited[2], deposited[3]

        # -- Sub-problems 03 to 08: Newton on the incremental potential.
        trial = np.array(node_velocities, dtype=float, copy=True)
        newton_converged = False
        # Check the transferred field first, then permit exactly
        # max_iterations Newton updates and check the final update as well.
        for iteration in range(max_iterations + 1):
            rate = interpolate_grid_velocity(trial, node_slots, weights,
                                                     weight_gradients, node_offsets)[1]
            trial_gradients = np.einsum('pab,pbc->pac', identity + step * rate, gradients)
            response = evaluate_corotated_response(trial_gradients, shear, lame)
            residual = assemble_potential_gradient(trial, node_velocities, node_masses,
                                                           node_slots, weight_gradients,
                                                           gradients, response[0], response[1],
                                                           volumes, step)[1]
            if float(np.max(np.abs(residual))) < tolerance:
                newton_converged = True
                break
            if iteration == max_iterations:
                break
            tangents = evaluate_corotated_tangent(trial_gradients, shear, lame)
            hessian = assemble_potential_hessian(tangents, gradients, weight_gradients,
                                                         node_slots, node_masses, volumes, step)
            trial = trial + solve_newton_direction(residual, hessian)

        if not newton_converged:
            raise ValueError(
                "Newton iteration did not reach the requested residual "
                "tolerance within max_iterations updates"
            )

        # -- Sub-problems 03 and 09: read the converged field back and advect.
        converged = interpolate_grid_velocity(trial, node_slots, weights,
                                                      weight_gradients, node_offsets)
        velocities, affine = converged[0], converged[2]
        advanced = advance_particle_state(positions, gradients, velocities,
                                                  converged[1], step)
        positions, gradients = advanced[0], advanced[1]

    # -- Sub-problems 04 and 10: the mechanical energy after the march.
    density_after = evaluate_corotated_response(gradients, shear, lame)[0]
    final = measure_mechanical_energy(masses, velocities, volumes,
                                              density_after, positions)
    return float(1.0 - float(final[2]) / float(initial[2]))
SCICODE_GOLD_EOF
