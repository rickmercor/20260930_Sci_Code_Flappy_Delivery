"""
Assemble the exact tangent matrix of the incremental potential with respect to the nodal velocity degrees of freedom.

The tangent of the incremental potential is the lumped mass on its diagonal plus a material contribution that couples every pair of nodes sharing a material point, so the sparsity of the matrix is determined by the support of the kernel. The material contribution carries the square of the time step, which is why the mass term dominates it at small steps, and the system stays well conditioned.

Returns
-------
np.ndarray of shape (2 * n_nodes, 2 * n_nodes): the exact tangent matrix of the incremental potential with respect to the nodal velocity degrees of freedom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_potential_hessian(tangents: "np.ndarray", reference_gradients: "np.ndarray",
                               weight_gradients: "np.ndarray", node_slots: "np.ndarray",
                               node_masses: "np.ndarray", volumes: "np.ndarray",
                               step: float) -> "np.ndarray":
    """Assemble the exact tangent matrix of the incremental potential.

    The matrix returned is the second derivative of the incremental potential
    with respect to the trial nodal velocities, the potential being the one
    whose first derivative the residual assembly returns. Degrees of freedom
    are ordered node by node, so the two components of node ``i`` occupy rows
    and columns ``2 i`` and ``2 i + 1``.

    Parameters
    ----------
    tangents : "np.ndarray"
        Array of shape (n_particles, 2, 2, 2, 2) holding the derivative of the
        first Piola-Kirchhoff stress with respect to the deformation gradient,
        evaluated at the trial deformation gradients, in pascals.
    reference_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the start of the step.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the node row it refers to.
    node_masses : "np.ndarray"
        Array of shape (n_nodes,) holding the nodal masses in kilograms per
        unit thickness; every entry must be >= 0.
    volumes : "np.ndarray"
        Array of shape (n_particles,) holding the reference particle volumes in
        cubic metres per unit thickness; every entry must be > 0.
    step : float
        Time step in seconds (step > 0).

    Returns
    -------
    hessian : "np.ndarray"
        Array of shape (2 * n_nodes, 2 * n_nodes) holding the tangent matrix,
        in kilograms per unit thickness.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if a slot refers to a node outside the
        nodal arrays.
    """
    return hessian  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_potential_hessian(tangents: "np.ndarray", reference_gradients: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned array
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: staggered pair with a full material tangent (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(71)
n_particles, n_grids, n_slots, n_nodes = 6, 2, 4, 9
raw = rng.standard_normal((n_particles, 4, 4))
symmetric = 0.5 * (raw + np.einsum('pij->pji', raw))
tangents = 4.0e4 * symmetric.reshape(n_particles, 2, 2, 2, 2)
reference = np.eye(2)[None] + 0.15 * rng.standard_normal((n_particles, 2, 2))
weight_gradients = 40.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
node_masses = np.linspace(0.03, 0.19, n_nodes)
volumes = np.full(n_particles, 1.0e-4)
step = 1.0e-3

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
            "gold_call": "pin(_oracle_assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
        },
        # --- Valid: nine-slot single grid, so the averaging factor changes ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(72)
n_particles, n_grids, n_slots, n_nodes = 5, 1, 9, 11
raw = rng.standard_normal((n_particles, 4, 4))
symmetric = 0.5 * (raw + np.einsum('pij->pji', raw))
tangents = 4.0e4 * symmetric.reshape(n_particles, 2, 2, 2, 2)
reference = np.eye(2)[None] + 0.15 * rng.standard_normal((n_particles, 2, 2))
weight_gradients = 40.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
node_masses = np.linspace(0.02, 0.22, n_nodes)
volumes = np.full(n_particles, 1.0e-4)
step = 5.0e-4

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
            "gold_call": "pin(_oracle_assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
        },
        # --- Valid: the tangent matrix against a central difference of the residual ---
        # The matrix must be the derivative of the nodal residual, and it must be
        # symmetric because it is the second derivative of a scalar potential.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(73)
step = 1.0e-3
shear, lame = 43200.0, 148800.0
n_particles, n_grids, n_slots, n_nodes = 5, 2, 4, 8
reference = np.eye(2)[None] + 0.1 * rng.standard_normal((n_particles, 2, 2))
weight_gradients = 30.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
node_masses = np.linspace(0.04, 0.16, n_nodes)
volumes = np.full(n_particles, 1.0e-4)
trial = 0.05 * rng.standard_normal((n_nodes, 2))
start = np.zeros((n_nodes, 2))

def kinematics(vhat):
    gathered = vhat[node_slots.reshape(n_particles, -1)]
    grad = np.einsum('pma,pmb->pab', gathered,
                     weight_gradients.reshape(n_particles, -1, 2)) / n_grids
    return np.einsum('pab,pbc->pac', np.eye(2)[None] + step * grad, reference)

def rotation_of(F):
    trace = F[:, 0, 0] + F[:, 1, 1]
    spin = F[:, 0, 1] - F[:, 1, 0]
    scale = np.sqrt(trace ** 2 + spin ** 2)
    R = np.empty_like(F)
    R[:, 0, 0] = trace / scale
    R[:, 0, 1] = spin / scale
    R[:, 1, 0] = -spin / scale
    R[:, 1, 1] = trace / scale
    return R

def cofactor_of(F):
    C = np.empty_like(F)
    C[:, 0, 0] = F[:, 1, 1]
    C[:, 0, 1] = -F[:, 1, 0]
    C[:, 1, 0] = -F[:, 0, 1]
    C[:, 1, 1] = F[:, 0, 0]
    return C

def stress_of(F):
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    return 2.0 * shear * (F - rotation_of(F)) + lame * (J - 1.0)[:, None, None] * cofactor_of(F)

def residual_of(vhat):
    F = kinematics(vhat)
    pulled = np.einsum('pab,pma->pmb', reference,
                       weight_gradients.reshape(n_particles, -1, 2))
    force = step * volumes[:, None, None] * np.einsum('pcb,pmb->pmc', stress_of(F), pulled)
    out = node_masses[:, None] * (vhat - start)
    np.add.at(out, node_slots.reshape(n_particles, -1).ravel(), force.reshape(-1, 2))
    return out

def tangent_of(F):
    R = rotation_of(F)
    C = cofactor_of(F)
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    tr = np.einsum('pba,pba->p', R, F)
    quarter = np.stack([R[:, :, 1], -R[:, :, 0]], axis=2)
    ang = np.empty((F.shape[0], 2, 2))
    ang[:, :, 0] = R[:, :, 1] / tr[:, None]
    ang[:, :, 1] = -R[:, :, 0] / tr[:, None]
    ident = np.zeros((2, 2, 2, 2))
    dcof = np.zeros((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            ident[a, b, a, b] = 1.0
    dcof[0, 0, 1, 1] = 1.0
    dcof[0, 1, 1, 0] = -1.0
    dcof[1, 0, 0, 1] = -1.0
    dcof[1, 1, 0, 0] = 1.0
    out = 2.0 * shear * (ident[None] - np.einsum('pab,pcd->pabcd', quarter, ang))
    out = out + lame * np.einsum('pab,pcd->pabcd', C, C)
    return out + lame * (J - 1.0)[:, None, None, None, None] * dcof[None]

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def consistency(fn):
    exact = fn(tangent_of(kinematics(trial)), reference, weight_gradients,
               node_slots, node_masses, volumes, step)
    nudge = 1.0e-8
    numeric = np.zeros_like(exact)
    for row in range(n_nodes):
        for axis in range(2):
            up = trial.copy(); up[row, axis] += nudge
            down = trial.copy(); down[row, axis] -= nudge
            numeric[:, 2 * row + axis] = ((residual_of(up) - residual_of(down))
                                          / (2.0 * nudge)).ravel()
    scale = np.abs(numeric).max()
    flags = float(int(np.abs(exact - numeric).max() < 1.0e-5 * scale)
                  + 2 * int(np.abs(exact - exact.T).max() < 1.0e-9 * np.abs(exact).max()))
    return flags + pin(exact) + 3.0 * pin(numeric)
""",
            "call": "consistency(assemble_potential_hessian)",
            "gold_call": "consistency(_oracle_assemble_potential_hessian)",
        },
        # --- Boundary: a vanishing material tangent, leaving only the lumped mass ---
        {
            "setup": """import numpy as np
n_particles, n_grids, n_slots, n_nodes = 3, 2, 4, 6
tangents = np.zeros((n_particles, 2, 2, 2, 2))
reference = np.tile(np.eye(2), (n_particles, 1, 1))
weight_gradients = np.full((n_particles, n_grids, n_slots, 2), 25.0)
node_slots = np.zeros((n_particles, n_grids, n_slots), dtype=np.int64)
for p in range(n_particles):
    node_slots[p] = np.arange(n_grids * n_slots).reshape(n_grids, n_slots) % n_nodes
node_masses = np.linspace(0.05, 0.15, n_nodes)
volumes = np.full(n_particles, 1.0e-4)
step = 1.0e-3

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
            "gold_call": "pin(_oracle_assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
        },
        # --- Edge: a large step, where the material term dominates the mass term ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(74)
n_particles, n_grids, n_slots, n_nodes = 4, 2, 4, 7
raw = rng.standard_normal((n_particles, 4, 4))
symmetric = 0.5 * (raw + np.einsum('pij->pji', raw))
tangents = 4.0e4 * symmetric.reshape(n_particles, 2, 2, 2, 2)
reference = np.eye(2)[None] + 0.1 * rng.standard_normal((n_particles, 2, 2))
weight_gradients = 50.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
node_masses = np.linspace(0.01, 0.05, n_nodes)
volumes = np.full(n_particles, 1.0e-4)
step = 5.0e-2

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
            "gold_call": "pin(_oracle_assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, step))",
        },
        # --- Invalid: a material tangent of the wrong shape ---
        {
            "setup": """import numpy as np
n_particles, n_grids, n_slots, n_nodes = 2, 1, 4, 4
tangents = np.zeros((n_particles, 4, 4))
reference = np.tile(np.eye(2), (n_particles, 1, 1))
weight_gradients = np.zeros((n_particles, n_grids, n_slots, 2))
node_slots = np.zeros((n_particles, n_grids, n_slots), dtype=np.int64)
node_masses = np.full(n_nodes, 0.1)
volumes = np.full(n_particles, 1.0e-4)
def run_model():
    try:
        assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative nodal mass ---
        {
            "setup": """import numpy as np
n_particles, n_grids, n_slots, n_nodes = 2, 1, 4, 4
tangents = np.zeros((n_particles, 2, 2, 2, 2))
reference = np.tile(np.eye(2), (n_particles, 1, 1))
weight_gradients = np.zeros((n_particles, n_grids, n_slots, 2))
node_slots = np.zeros((n_particles, n_grids, n_slots), dtype=np.int64)
node_masses = np.array([0.1, 0.1, -0.1, 0.1])
volumes = np.full(n_particles, 1.0e-4)
def run_model():
    try:
        assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_potential_hessian(tangents, reference, weight_gradients, node_slots, node_masses, volumes, 1.0e-3)
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
