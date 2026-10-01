"""
Collect the stencil into an active node table and carry mass and affine momentum from the material points onto every grid of the family.

The particle-to-grid map lumps mass onto the nodes and carries momentum with an affine correction, so that a velocity field varying linearly across a cell is transferred without loss. The correction is expressed through the inverse of the weighted second moment of the node offsets, which is the only object in the transfer that has to be inverted.

Returns
-------
tuple of four np.ndarray: active node labels of shape (n_nodes, 3), stencil-to-node map of shape (n_particles, n_grids, n_slots), nodal masses of shape (n_nodes,) and nodal velocities of shape (n_nodes, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def transfer_particles_to_grid(masses: "np.ndarray", velocities: "np.ndarray",
                               affine_states: "np.ndarray", node_indices: "np.ndarray",
                               weights: "np.ndarray", node_offsets: "np.ndarray") -> tuple:
    """Carry mass and affine momentum from the material points onto the grids.

    Every node that appears in any particle stencil is active, whatever its
    mass. The active nodes are labelled by the triple (grid, first lattice
    index, second lattice index) and are ordered lexicographically by that
    triple, which fixes the row order of every nodal array in the rest of the
    calculation.

    The affine moment matrix of a particle is the weighted second moment of
    its node offsets, averaged over the grids of the family with the factor
    one over the number of grids. The momentum a particle deposits on a node
    is its mass times the weight times its velocity plus the affine state
    contracted with the inverse moment matrix and with the node offset. The
    nodal velocity is the deposited momentum divided by the deposited mass,
    and is set to zero at a node whose deposited mass vanishes.

    Parameters
    ----------
    masses : "np.ndarray"
        Array of shape (n_particles,) holding the particle masses in kilograms
        per unit thickness; every entry must be > 0.
    velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities in
        m s^-1.
    affine_states : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the affine velocity states
        of the particles, in m^2 s^-1.
    node_indices : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots, 2) holding the
        lattice indices of the stencil nodes.
    weights : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots) holding the kernel
        weights.
    node_offsets : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the node
        position minus the particle position, in metres.

    Returns
    -------
    node_keys : "np.ndarray"
        Integer array of shape (n_nodes, 3) holding the labels of the active
        nodes in lexicographic order.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the row of node_keys it refers to.
    node_masses : "np.ndarray"
        Array of shape (n_nodes,) holding the nodal masses in kilograms per
        unit thickness.
    node_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the nodal velocities in m s^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if an affine moment matrix is singular.
    """
    return node_keys, node_slots, node_masses, node_velocities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_transfer_particles_to_grid(masses: "np.ndarray", velocities: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned arrays
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: staggered pair, mixed velocities and affine states (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
spacing = 0.02
fraction = np.linspace(0.13, 0.87, 6)
positions = np.stack([(2.0 + fraction) * spacing, (4.0 + fraction[::-1]) * spacing], axis=1)
n = positions.shape[0]
masses = np.full(n, 0.12)
velocities = 0.4 * rng.standard_normal((n, 2))
affine = 0.002 * rng.standard_normal((n, 2, 2))

def stencil(pos, offs, name):
    side = 2 if name == 'compact' else 3
    shift = 0.0 if side == 2 else 0.5
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), side * side, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), side * side))
    off = np.zeros((p, len(offs), side * side, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o - shift).astype(np.int64)
        for a in range(side):
            for b in range(side):
                s = side * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = np.abs(sep / spacing)
                if name == 'compact':
                    v = np.where(d <= 1.0, 1.0 - d + np.sin(2 * np.pi * d) / (2 * np.pi), 0.0)
                else:
                    v = np.where(d < 0.5, 0.75 - d ** 2,
                                 np.where(d <= 1.5, 0.5 * (1.5 - d) ** 2, 0.0))
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
    return idx, wgt, off

node_indices, weights, node_offsets = stencil(positions, (0.0, 0.5), 'compact')

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets))",
            "gold_call": "pin_all(_oracle_transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets))",
        },
        # --- Valid: nine-slot single grid, so the node table and the averaging both change ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(12)
spacing = 0.02
fraction = np.linspace(0.13, 0.87, 6)
positions = np.stack([(2.0 + fraction) * spacing, (4.0 + fraction[::-1]) * spacing], axis=1)
n = positions.shape[0]
masses = np.full(n, 0.12)
velocities = 0.4 * rng.standard_normal((n, 2))
affine = 0.002 * rng.standard_normal((n, 2, 2))

def stencil(pos, offs, name):
    side = 2 if name == 'compact' else 3
    shift = 0.0 if side == 2 else 0.5
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), side * side, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), side * side))
    off = np.zeros((p, len(offs), side * side, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o - shift).astype(np.int64)
        for a in range(side):
            for b in range(side):
                s = side * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = np.abs(sep / spacing)
                if name == 'compact':
                    v = np.where(d <= 1.0, 1.0 - d + np.sin(2 * np.pi * d) / (2 * np.pi), 0.0)
                else:
                    v = np.where(d < 0.5, 0.75 - d ** 2,
                                 np.where(d <= 1.5, 0.5 * (1.5 - d) ** 2, 0.0))
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
    return idx, wgt, off

node_indices, weights, node_offsets = stencil(positions, (0.0,), 'quadratic')

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets))",
            "gold_call": "pin_all(_oracle_transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets))",
        },
        # --- Valid: mass and momentum conservation of the transfer ---
        # The nodal mass of one grid of the family reproduces the body mass and
        # the nodal momentum summed over the family reproduces the body momentum
        # times the number of grids, whatever the affine states are.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(13)
spacing = 0.02
fraction = np.linspace(0.07, 0.93, 8)
positions = np.stack([(1.0 + fraction) * spacing, (3.0 + fraction[::-1]) * spacing], axis=1)
n = positions.shape[0]
masses = np.linspace(0.09, 0.15, n)
velocities = 0.5 * rng.standard_normal((n, 2))
affine = 0.003 * rng.standard_normal((n, 2, 2))

def stencil(pos, offs):
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), 4, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), 4))
    off = np.zeros((p, len(offs), 4, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o).astype(np.int64)
        for a in range(2):
            for b in range(2):
                s = 2 * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = np.abs(sep / spacing)
                v = np.where(d <= 1.0, 1.0 - d + np.sin(2 * np.pi * d) / (2 * np.pi), 0.0)
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
    return idx, wgt, off

node_indices, weights, node_offsets = stencil(positions, (0.0, 0.5))

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def conservation(fn):
    keys, slots, node_mass, node_velocity = fn(masses, velocities, affine,
                                               node_indices, weights, node_offsets)
    total_mass = node_mass.sum() / 2.0
    total_momentum = (node_mass[:, None] * node_velocity).sum(axis=0) / 2.0
    flags = float(int(abs(total_mass - masses.sum()) < 1.0e-12 * masses.sum())
                  + 2 * int(np.abs(total_momentum - (masses[:, None] * velocities).sum(axis=0)).max()
                            < 1.0e-12))
    return flags + pin(node_mass) + 5.0 * pin(node_velocity) + 1.0e6 * total_mass
""",
            "call": "conservation(transfer_particles_to_grid)",
            "gold_call": "conservation(_oracle_transfer_particles_to_grid)",
        },
        # --- Boundary: a single particle, so the stencil is the smallest possible ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.0431, 0.0177]])
masses = np.array([0.12])
velocities = np.array([[0.3, -0.2]])
affine = np.array([[[0.001, -0.0004], [0.0002, 0.0007]]])

def stencil(pos, offs):
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), 4, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), 4))
    off = np.zeros((p, len(offs), 4, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o).astype(np.int64)
        for a in range(2):
            for b in range(2):
                s = 2 * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = np.abs(sep / spacing)
                v = np.where(d <= 1.0, 1.0 - d + np.sin(2 * np.pi * d) / (2 * np.pi), 0.0)
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
    return idx, wgt, off

node_indices, weights, node_offsets = stencil(positions, (0.0, 0.5))

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets))",
            "gold_call": "pin_all(_oracle_transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets))",
        },
        # --- Invalid: a particle carrying zero mass ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.0431, 0.0177], [0.0611, 0.0233]])
masses = np.array([0.12, 0.0])
velocities = np.zeros((2, 2))
affine = np.zeros((2, 2, 2))
base = np.floor(positions / spacing).astype(np.int64)
node_indices = np.zeros((2, 1, 4, 2), dtype=np.int64)
weights = np.full((2, 1, 4), 0.25)
node_offsets = np.zeros((2, 1, 4, 2))
for a in range(2):
    for b in range(2):
        s = 2 * a + b
        node_indices[:, 0, s, :] = base + np.array([a, b])
        node_offsets[:, 0, s, :] = (base + np.array([a, b])) * spacing - positions
def run_model():
    try:
        transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: node offsets that collapse the affine moment matrix ---
        {
            "setup": """import numpy as np
masses = np.array([0.12])
velocities = np.zeros((1, 2))
affine = np.zeros((1, 2, 2))
node_indices = np.zeros((1, 1, 4, 2), dtype=np.int64)
weights = np.full((1, 1, 4), 0.25)
node_offsets = np.zeros((1, 1, 4, 2))
def run_model():
    try:
        transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_transfer_particles_to_grid(masses, velocities, affine, node_indices, weights, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
