"""
Read a nodal velocity field back onto the material points as a velocity, a velocity gradient and an affine velocity state.

The grid-to-particle map is the transpose of the deposit and must average the contributions of every grid of the staggered family with the factor one over the number of grids, since each grid on its own already carries the whole body mass. The velocity gradient obtained from the kernel gradients is what drives the deformation gradient forward, while the affine state preserves the locally linear part of the velocity field for the next deposit.

Returns
-------
tuple of three np.ndarray: particle velocities of shape (n_particles, 2), particle velocity gradients of shape (n_particles, 2, 2) and affine velocity states of shape (n_particles, 2, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def interpolate_grid_velocity(node_velocities: "np.ndarray", node_slots: "np.ndarray",
                              weights: "np.ndarray", weight_gradients: "np.ndarray",
                              node_offsets: "np.ndarray") -> tuple:
    """Read a nodal velocity field back onto the material points.

    Every returned quantity is a sum over the whole stencil of the particle,
    that is over every grid of the family and every local slot, divided by the
    number of grids.

    The particle velocity is the weighted sum of the nodal velocities. The
    velocity gradient has the derivative direction as its second index, so its
    entry (a, b) is the sum of the nodal velocity component a times component b
    of the weight gradient. The affine velocity state has the node offset
    direction as its second index, so its entry (a, b) is the weighted sum of
    the nodal velocity component a times component b of the node offset.

    Parameters
    ----------
    node_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the nodal velocities in m s^-1.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the row of node_velocities it refers to.
    weights : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots) holding the kernel
        weights.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    node_offsets : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the node
        position minus the particle position, in metres.

    Returns
    -------
    particle_velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities in
        m s^-1.
    velocity_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the particle velocity
        gradients in s^-1.
    affine_states : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the affine velocity states
        of the particles, in m^2 s^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if a slot refers to a node outside the
        nodal velocity array.
    """
    return particle_velocities, velocity_gradients, affine_states  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_interpolate_grid_velocity(node_velocities: "np.ndarray", node_slots: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned arrays
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: staggered pair with an unstructured nodal field (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(21)
spacing = 0.02
fraction = np.linspace(0.11, 0.89, 6)
positions = np.stack([(2.0 + fraction) * spacing, (4.0 + fraction[::-1]) * spacing], axis=1)
n = positions.shape[0]

def stencil(pos, offs):
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), 4, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), 4))
    grd = np.zeros((p, len(offs), 4, 2))
    off = np.zeros((p, len(offs), 4, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o).astype(np.int64)
        for a in range(2):
            for b in range(2):
                s = 2 * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = sep / spacing
                m = np.abs(d)
                v = np.where(m <= 1.0, 1.0 - m + np.sin(2 * np.pi * m) / (2 * np.pi), 0.0)
                sl = np.where(m <= 1.0, np.sign(d) * (np.cos(2 * np.pi * d) - 1.0), 0.0)
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
                grd[:, k, s, 0] = -sl[:, 0] * v[:, 1] / spacing
                grd[:, k, s, 1] = -v[:, 0] * sl[:, 1] / spacing
    return idx, wgt, grd, off

idx, weights, weight_gradients, node_offsets = stencil(positions, (0.0, 0.5))
labels = np.concatenate([np.broadcast_to(np.repeat(np.arange(2), 4)[None, :, None], (n, 8, 1)),
                         idx.reshape(n, 8, 2)], axis=2).reshape(-1, 3)
keys, inverse = np.unique(labels, axis=0, return_inverse=True)
node_slots = np.asarray(inverse, dtype=np.int64).reshape(n, 2, 4)
node_velocities = 0.5 * rng.standard_normal((keys.shape[0], 2))

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
            "call": "pin_all(interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets))",
            "gold_call": "pin_all(_oracle_interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets))",
        },
        # --- Valid: an affine nodal field, which the staggered pair reads back exactly ---
        # A nodal velocity field that is affine in the node position is returned as
        # the same affine field, so the interpolated velocity gradient reproduces the
        # imposed one and the particle velocity reproduces the field at the particle.
        {
            "setup": """import numpy as np
spacing = 0.02
fraction = np.linspace(0.07, 0.93, 7)
positions = np.stack([(3.0 + fraction) * spacing, (2.0 + fraction[::-1]) * spacing], axis=1)
n = positions.shape[0]
imposed = np.array([[0.9, -0.4], [0.25, -0.6]])
uniform = np.array([0.13, -0.07])

def stencil(pos, offs):
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), 4, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), 4))
    grd = np.zeros((p, len(offs), 4, 2))
    off = np.zeros((p, len(offs), 4, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o).astype(np.int64)
        for a in range(2):
            for b in range(2):
                s = 2 * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = sep / spacing
                m = np.abs(d)
                v = np.where(m <= 1.0, 1.0 - m + np.sin(2 * np.pi * m) / (2 * np.pi), 0.0)
                sl = np.where(m <= 1.0, np.sign(d) * (np.cos(2 * np.pi * d) - 1.0), 0.0)
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
                grd[:, k, s, 0] = -sl[:, 0] * v[:, 1] / spacing
                grd[:, k, s, 1] = -v[:, 0] * sl[:, 1] / spacing
    return idx, wgt, grd, off

idx, weights, weight_gradients, node_offsets = stencil(positions, (0.0, 0.5))
labels = np.concatenate([np.broadcast_to(np.repeat(np.arange(2), 4)[None, :, None], (n, 8, 1)),
                         idx.reshape(n, 8, 2)], axis=2).reshape(-1, 3)
keys, inverse = np.unique(labels, axis=0, return_inverse=True)
node_slots = np.asarray(inverse, dtype=np.int64).reshape(n, 2, 4)
node_xy = np.stack([(keys[:, 1] + 0.5 * keys[:, 0]) * spacing,
                    (keys[:, 2] + 0.5 * keys[:, 0]) * spacing], axis=1)
node_velocities = uniform + node_xy @ imposed.T

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def affine_recovery(fn):
    vp, gradv, affine = fn(node_velocities, node_slots, weights, weight_gradients, node_offsets)
    exact = uniform + positions @ imposed.T
    flags = float(int(np.abs(gradv - imposed).max() < 1.0e-10)
                  + 2 * int(np.abs(vp - exact).max() < 1.0e-12))
    return flags + pin(gradv) + 7.0 * pin(vp) + 11.0 * pin(affine)
""",
            "call": "affine_recovery(interpolate_grid_velocity)",
            "gold_call": "affine_recovery(_oracle_interpolate_grid_velocity)",
        },
        # --- Boundary: a nodal field that vanishes, so every returned array is zero ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.0431, 0.0177], [0.0611, 0.0233]])
n = positions.shape[0]
node_slots = np.zeros((n, 1, 4), dtype=np.int64)
for a in range(2):
    for b in range(2):
        node_slots[:, 0, 2 * a + b] = 2 * a + b
weights = np.full((n, 1, 4), 0.25)
weight_gradients = np.full((n, 1, 4, 2), 1.0 / spacing)
node_offsets = np.full((n, 1, 4, 2), 0.5 * spacing)
node_velocities = np.zeros((4, 2))

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
            "call": "pin_all(interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets))",
            "gold_call": "pin_all(_oracle_interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets))",
        },
        # --- Edge: three-grid family, so the averaging factor is one third ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(22)
spacing = 0.05
positions = np.array([[0.031, 0.077], [-0.019, 0.132], [0.104, -0.038]])
n = positions.shape[0]
offs = (0.0, 1.0 / 3.0, 2.0 / 3.0)

def stencil(pos, offs):
    p = pos.shape[0]
    idx = np.zeros((p, len(offs), 4, 2), dtype=np.int64)
    wgt = np.zeros((p, len(offs), 4))
    grd = np.zeros((p, len(offs), 4, 2))
    off = np.zeros((p, len(offs), 4, 2))
    for k, o in enumerate(offs):
        base = np.floor(pos / spacing - o).astype(np.int64)
        for a in range(2):
            for b in range(2):
                s = 2 * a + b
                ii = base + np.array([a, b])
                sep = (ii + o) * spacing - pos
                d = sep / spacing
                m = np.abs(d)
                v = np.where(m <= 1.0, 1.0 - m + np.sin(2 * np.pi * m) / (2 * np.pi), 0.0)
                sl = np.where(m <= 1.0, np.sign(d) * (np.cos(2 * np.pi * d) - 1.0), 0.0)
                idx[:, k, s, :] = ii
                off[:, k, s, :] = sep
                wgt[:, k, s] = v[:, 0] * v[:, 1]
                grd[:, k, s, 0] = -sl[:, 0] * v[:, 1] / spacing
                grd[:, k, s, 1] = -v[:, 0] * sl[:, 1] / spacing
    return idx, wgt, grd, off

idx, weights, weight_gradients, node_offsets = stencil(positions, offs)
labels = np.concatenate([np.broadcast_to(np.repeat(np.arange(3), 4)[None, :, None], (n, 12, 1)),
                         idx.reshape(n, 12, 2)], axis=2).reshape(-1, 3)
keys, inverse = np.unique(labels, axis=0, return_inverse=True)
node_slots = np.asarray(inverse, dtype=np.int64).reshape(n, 3, 4)
node_velocities = 0.4 * rng.standard_normal((keys.shape[0], 2))

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
            "call": "pin_all(interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets))",
            "gold_call": "pin_all(_oracle_interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets))",
        },
        # --- Invalid: a slot pointing past the end of the nodal velocity array ---
        {
            "setup": """import numpy as np
node_velocities = np.zeros((4, 2))
node_slots = np.array([[[0, 1, 2, 9]]], dtype=np.int64)
weights = np.full((1, 1, 4), 0.25)
weight_gradients = np.zeros((1, 1, 4, 2))
node_offsets = np.zeros((1, 1, 4, 2))
def run_model():
    try:
        interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: weight gradients whose shape does not match the stencil ---
        {
            "setup": """import numpy as np
node_velocities = np.zeros((4, 2))
node_slots = np.array([[[0, 1, 2, 3]]], dtype=np.int64)
weights = np.full((1, 1, 4), 0.25)
weight_gradients = np.zeros((1, 1, 3, 2))
node_offsets = np.zeros((1, 1, 4, 2))
def run_model():
    try:
        interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interpolate_grid_velocity(node_velocities, node_slots, weights, weight_gradients, node_offsets)
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
