"""
Evaluate the incremental potential of the implicit step at a trial nodal velocity and assemble its gradient with respect to the nodal degrees of freedom.

Backward Euler on the nodal momentum balance is equivalent to minimising an incremental potential that adds the kinetic penalty of departing from the transferred velocity to the strain energy the resulting deformation would store. The stationary point of that potential is exactly the implicit momentum balance, so the gradient assembled here is the residual the nonlinear solve drives to zero.

Returns
-------
tuple containing a native Python float for the incremental potential and an np.ndarray of shape (n_nodes, 2) for its gradient with respect to the trial nodal velocities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_potential_gradient(trial_velocities: "np.ndarray", initial_velocities: "np.ndarray",
                                node_masses: "np.ndarray", node_slots: "np.ndarray",
                                weight_gradients: "np.ndarray", reference_gradients: "np.ndarray",
                                energy_densities: "np.ndarray", first_piola: "np.ndarray",
                                volumes: "np.ndarray", step: float) -> tuple:
    """Evaluate the incremental potential and its nodal gradient.

    The incremental potential at a trial nodal velocity field ``vhat`` is

        ``E = (1/2) sum_i m_i |vhat_i - v_i|^2 + n_grids * sum_p V_p psi_p``,

    where ``m_i`` are the nodal masses, ``v_i`` the velocities the transfer
    deposited, ``V_p`` the reference volumes and ``psi_p`` the strain energy
    densities of the trial deformation gradients. The factor equal to the
    number of grids compensates the fact that each grid of the family already
    carries the whole body mass, so that the nodal kinetic term and the strain
    term are weighted alike.

    The trial deformation gradient of a particle is ``(I + step * L_p) F_p``,
    with ``F_p`` its reference deformation gradient and ``L_p`` the velocity
    gradient the trial field induces at the particle. The returned residual is
    the gradient of the potential with respect to the trial nodal velocities.

    Parameters
    ----------
    trial_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the trial nodal velocities in
        m s^-1.
    initial_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the nodal velocities the transfer
        deposited, in m s^-1.
    node_masses : "np.ndarray"
        Array of shape (n_nodes,) holding the nodal masses in kilograms per
        unit thickness; every entry must be >= 0.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the node row it refers to.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    reference_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the start of the step.
    energy_densities : "np.ndarray"
        Array of shape (n_particles,) holding the strain energy densities of
        the trial deformation gradients, in joules per cubic metre.
    first_piola : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the first Piola-Kirchhoff
        stresses of the trial deformation gradients, in pascals.
    volumes : "np.ndarray"
        Array of shape (n_particles,) holding the reference particle volumes in
        cubic metres per unit thickness; every entry must be > 0.
    step : float
        Time step in seconds (step > 0).

    Returns
    -------
    potential : float
        Value of the incremental potential in joules per unit thickness, as a
        native Python float.
    residual : "np.ndarray"
        Array of shape (n_nodes, 2) holding the gradient of the potential with
        respect to the trial nodal velocities, in kilogram metre per second.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if a slot refers to a node outside the
        nodal arrays.
    """
    return potential, residual  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_potential_gradient(trial_velocities: "np.ndarray", initial_velocities: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned value
    # and array through a position weighted digest, the invalid cases return a
    # status code.
    return [
        # --- Valid: staggered pair away from the transferred velocity (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(61)
n_particles, n_grids, n_slots, n_nodes = 7, 2, 4, 9
trial = 0.4 * rng.standard_normal((n_nodes, 2))
start = 0.4 * rng.standard_normal((n_nodes, 2))
node_masses = np.linspace(0.03, 0.19, n_nodes)
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
weight_gradients = 40.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
reference = np.eye(2)[None] + 0.15 * rng.standard_normal((n_particles, 2, 2))
density = 500.0 + 300.0 * rng.random(n_particles)
piola = 2.0e4 * rng.standard_normal((n_particles, 2, 2))
volumes = np.full(n_particles, 1.0e-4)
step = 1.0e-3

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def reduce_pair(parts):
    return float((1.0 + 1.0e6 * float(parts[0])) * (1.0 + pin(parts[1])) ** 0.5)
""",
            "call": "reduce_pair(assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
            "gold_call": "reduce_pair(_oracle_assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
        },
        # --- Valid: nine-slot single grid, so the strain term carries a different factor ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(62)
n_particles, n_grids, n_slots, n_nodes = 6, 1, 9, 12
trial = 0.4 * rng.standard_normal((n_nodes, 2))
start = 0.4 * rng.standard_normal((n_nodes, 2))
node_masses = np.linspace(0.02, 0.21, n_nodes)
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
weight_gradients = 40.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
reference = np.eye(2)[None] + 0.15 * rng.standard_normal((n_particles, 2, 2))
density = 400.0 + 200.0 * rng.random(n_particles)
piola = 1.5e4 * rng.standard_normal((n_particles, 2, 2))
volumes = np.full(n_particles, 1.0e-4)
step = 5.0e-4

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def reduce_pair(parts):
    return float((1.0 + 1.0e6 * float(parts[0])) * (1.0 + pin(parts[1])) ** 0.5)
""",
            "call": "reduce_pair(assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
            "gold_call": "reduce_pair(_oracle_assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
        },
        # --- Valid: the gradient against a central difference of the potential ---
        # The residual must be the derivative of the potential with respect to the
        # trial nodal velocities, which fixes the chain rule through the trial
        # deformation gradient and the averaging factor of the staggered family.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(63)
spacing, step = 0.02, 1.0e-3
shear, lame = 43200.0, 148800.0
n_particles, n_grids, n_slots = 6, 2, 4
n_nodes = 10
start = 0.3 * rng.standard_normal((n_nodes, 2))
node_masses = np.linspace(0.04, 0.16, n_nodes)
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
weight_gradients = 30.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
reference = np.eye(2)[None] + 0.1 * rng.standard_normal((n_particles, 2, 2))
volumes = np.full(n_particles, 1.0e-4)
trial = start + 0.05 * rng.standard_normal((n_nodes, 2))

def response(F):
    trace = F[:, 0, 0] + F[:, 1, 1]
    spin = F[:, 0, 1] - F[:, 1, 0]
    scale = np.sqrt(trace ** 2 + spin ** 2)
    R = np.empty_like(F)
    R[:, 0, 0] = trace / scale
    R[:, 0, 1] = spin / scale
    R[:, 1, 0] = -spin / scale
    R[:, 1, 1] = trace / scale
    C = np.empty_like(F)
    C[:, 0, 0] = F[:, 1, 1]
    C[:, 0, 1] = -F[:, 1, 0]
    C[:, 1, 0] = -F[:, 0, 1]
    C[:, 1, 1] = F[:, 0, 0]
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    psi = shear * np.sum((F - R) ** 2, axis=(1, 2)) + 0.5 * lame * (J - 1.0) ** 2
    return psi, 2.0 * shear * (F - R) + lame * (J - 1.0)[:, None, None] * C

def state(vhat):
    gathered = vhat[node_slots.reshape(n_particles, -1)]
    grad = np.einsum('pma,pmb->pab', gathered,
                     weight_gradients.reshape(n_particles, -1, 2)) / n_grids
    trial_F = np.einsum('pab,pbc->pac', np.eye(2)[None] + step * grad, reference)
    return response(trial_F)

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def stationarity(fn):
    density, piola = state(trial)
    value, residual = fn(trial, start, node_masses, node_slots, weight_gradients,
                         reference, density, piola, volumes, step)
    nudge = 1.0e-7
    numeric = np.zeros_like(residual)
    for row in range(n_nodes):
        for axis in range(2):
            up = trial.copy(); up[row, axis] += nudge
            down = trial.copy(); down[row, axis] -= nudge
            du, pu = state(up)
            dd, pd = state(down)
            upper = fn(up, start, node_masses, node_slots, weight_gradients,
                       reference, du, pu, volumes, step)[0]
            lower = fn(down, start, node_masses, node_slots, weight_gradients,
                       reference, dd, pd, volumes, step)[0]
            numeric[row, axis] = (upper - lower) / (2.0 * nudge)
    flags = float(int(np.abs(residual - numeric).max() < 1.0e-5 * np.abs(numeric).max()))
    return flags + pin(residual) + 3.0 * pin(numeric) + 1.0e3 * float(value)
""",
            "call": "stationarity(assemble_potential_gradient)",
            "gold_call": "stationarity(_oracle_assemble_potential_gradient)",
        },
        # --- Boundary: the trial field equals the transferred field, so only the strain term acts ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(64)
n_particles, n_grids, n_slots, n_nodes = 5, 2, 4, 8
start = 0.4 * rng.standard_normal((n_nodes, 2))
trial = start.copy()
node_masses = np.linspace(0.03, 0.19, n_nodes)
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
weight_gradients = 40.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
reference = np.tile(np.eye(2), (n_particles, 1, 1))
density = np.zeros(n_particles)
piola = 1.0e4 * rng.standard_normal((n_particles, 2, 2))
volumes = np.full(n_particles, 1.0e-4)
step = 1.0e-3

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def reduce_pair(parts):
    return float((1.0 + 1.0e6 * float(parts[0])) * (1.0 + pin(parts[1])) ** 0.5)
""",
            "call": "reduce_pair(assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
            "gold_call": "reduce_pair(_oracle_assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
        },
        # --- Edge: a node of vanishing mass, which contributes no kinetic penalty ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(65)
n_particles, n_grids, n_slots, n_nodes = 5, 2, 4, 8
trial = 0.4 * rng.standard_normal((n_nodes, 2))
start = 0.4 * rng.standard_normal((n_nodes, 2))
node_masses = np.linspace(0.03, 0.19, n_nodes)
node_masses[3] = 0.0
node_slots = rng.integers(0, n_nodes, size=(n_particles, n_grids, n_slots))
weight_gradients = 40.0 * rng.standard_normal((n_particles, n_grids, n_slots, 2))
reference = np.eye(2)[None] + 0.12 * rng.standard_normal((n_particles, 2, 2))
density = 300.0 + 100.0 * rng.random(n_particles)
piola = 1.0e4 * rng.standard_normal((n_particles, 2, 2))
volumes = np.full(n_particles, 1.0e-4)
step = 2.0e-3

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def reduce_pair(parts):
    return float((1.0 + 1.0e6 * float(parts[0])) * (1.0 + pin(parts[1])) ** 0.5)
""",
            "call": "reduce_pair(assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
            "gold_call": "reduce_pair(_oracle_assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, step))",
        },
        # --- Invalid: a non-positive time step ---
        {
            "setup": """import numpy as np
n_particles, n_grids, n_slots, n_nodes = 2, 1, 4, 4
trial = np.zeros((n_nodes, 2))
start = np.zeros((n_nodes, 2))
node_masses = np.full(n_nodes, 0.1)
node_slots = np.zeros((n_particles, n_grids, n_slots), dtype=np.int64)
weight_gradients = np.zeros((n_particles, n_grids, n_slots, 2))
reference = np.tile(np.eye(2), (n_particles, 1, 1))
density = np.zeros(n_particles)
piola = np.zeros((n_particles, 2, 2))
volumes = np.full(n_particles, 1.0e-4)
def run_model():
    try:
        assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a reference volume of zero ---
        {
            "setup": """import numpy as np
n_particles, n_grids, n_slots, n_nodes = 2, 1, 4, 4
trial = np.zeros((n_nodes, 2))
start = np.zeros((n_nodes, 2))
node_masses = np.full(n_nodes, 0.1)
node_slots = np.zeros((n_particles, n_grids, n_slots), dtype=np.int64)
weight_gradients = np.zeros((n_particles, n_grids, n_slots, 2))
reference = np.tile(np.eye(2), (n_particles, 1, 1))
density = np.zeros(n_particles)
piola = np.zeros((n_particles, 2, 2))
volumes = np.array([1.0e-4, 0.0])
def run_model():
    try:
        assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_potential_gradient(trial, start, node_masses, node_slots, weight_gradients, reference, density, piola, volumes, 1.0e-3)
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
