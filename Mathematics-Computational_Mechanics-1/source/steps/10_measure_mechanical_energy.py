"""
Reduce the particle state to the kinetic, strain and total mechanical energy of the body and to its angular momentum about the centre of mass.

The mechanical energy the material points carry is the sum of the kinetic energy of their velocities and the strain energy stored in their deformation gradients, and for a body under no external loading, the continuum motion holds that sum fixed. Measuring it before and after a march therefore isolates whatever the discretization has removed, while the angular momentum about the centre of mass provides an independent conserved quantity to watch.

Returns
-------
tuple of four native Python floats: kinetic energy, strain energy, total mechanical energy and angular momentum about the centre of mass.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def measure_mechanical_energy(masses: "np.ndarray", particle_velocities: "np.ndarray",
                              volumes: "np.ndarray", energy_densities: "np.ndarray",
                              positions: "np.ndarray") -> tuple:
    """Reduce the particle state to energies and to an angular momentum.

    The kinetic energy is one half of the mass weighted sum of the squared
    particle speeds. The strain energy is the sum of the strain energy
    densities weighted by the reference volumes. The total is their sum. The
    angular momentum is taken about the centre of mass of the particles, and
    in two dimensions it is the scalar out-of-plane component, that is the sum
    over particles of the mass times the cross product of the position
    measured from the centre of mass with the velocity.

    Parameters
    ----------
    masses : "np.ndarray"
        Array of shape (n_particles,) holding the particle masses in kilograms
        per unit thickness; every entry must be > 0.
    particle_velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities in
        m s^-1.
    volumes : "np.ndarray"
        Array of shape (n_particles,) holding the reference particle volumes in
        cubic metres per unit thickness; every entry must be > 0.
    energy_densities : "np.ndarray"
        Array of shape (n_particles,) holding the strain energy densities in
        joules per cubic metre; every entry must be >= 0.
    positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle positions in
        metres.

    Returns
    -------
    kinetic_energy : float
        Kinetic energy in joules per unit thickness, as a native Python float.
    strain_energy : float
        Strain energy in joules per unit thickness, as a native Python float.
    total_energy : float
        Sum of the kinetic and the strain energy, as a native Python float.
    angular_momentum : float
        Out-of-plane angular momentum about the centre of mass, in kilogram
        square metre per second per unit thickness, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above or if the shapes
        are mutually inconsistent.
    """
    return kinetic_energy, strain_energy, total_energy, angular_momentum  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_measure_mechanical_energy(masses: "np.ndarray", particle_velocities: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases combine the four returned
    # scalars with distinct weights, the invalid cases return a status code.
    return [
        # --- Valid: benchmark body carrying both kinds of energy (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(101)
n = 12
masses = np.full(n, 0.12)
velocities = 0.6 * rng.standard_normal((n, 2))
volumes = np.full(n, 1.0e-4)
density = 2000.0 + 900.0 * rng.random(n)
positions = 0.2 * rng.random((n, 2))

def reduce_four(parts):
    return float(sum((order + 1.7) * (1.0 + 1.0e6 * float(value))
                     for order, value in enumerate(parts)))
""",
            "call": "reduce_four(measure_mechanical_energy(masses, velocities, volumes, density, positions))",
            "gold_call": "reduce_four(_oracle_measure_mechanical_energy(masses, velocities, volumes, density, positions))",
        },
        # --- Valid: the pre-strained benchmark block at rest ---
        # The block starts at rest, so the kinetic energy and the angular momentum
        # both vanish and the total is the strain energy alone.
        {
            "setup": """import numpy as np
spacing, cells = 0.02, 10
offsets = (np.arange(2) + 0.5) / 2.0
line = (np.arange(cells)[:, None] + offsets[None, :]).ravel() * spacing
grid_x, grid_y = np.meshgrid(line, line, indexing='ij')
positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
n = positions.shape[0]
volumes = np.full(n, spacing * spacing / 4.0)
masses = 1200.0 * volumes
velocities = np.zeros((n, 2))
stretch = 1.20
density = np.full(n, 43200.0 * ((stretch - 1.0) ** 2 + (1.0 / stretch - 1.0) ** 2))

def reduce_four(parts):
    return float(sum((order + 1.7) * (1.0 + 1.0e6 * float(value))
                     for order, value in enumerate(parts)))
""",
            "call": "reduce_four(measure_mechanical_energy(masses, velocities, volumes, density, positions))",
            "gold_call": "reduce_four(_oracle_measure_mechanical_energy(masses, velocities, volumes, density, positions))",
        },
        # --- Valid: invariance of the angular momentum under a rigid translation ---
        # Taking the moment about the centre of mass makes the angular momentum
        # independent of where the body sits and of any uniform velocity added to
        # it, while the kinetic energy is not.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(102)
n = 9
masses = np.linspace(0.08, 0.16, n)
velocities = 0.5 * rng.standard_normal((n, 2))
volumes = np.full(n, 1.0e-4)
density = 1500.0 * rng.random(n)
positions = 0.2 * rng.random((n, 2))
displacement = np.array([3.7, -1.9])
drift = np.array([0.23, 0.41])

def objectivity(fn):
    base = fn(masses, velocities, volumes, density, positions)
    moved = fn(masses, velocities + drift, volumes, density, positions + displacement)
    flags = float(int(abs(moved[3] - base[3]) < 1.0e-12 * max(1.0, abs(base[3])))
                  + 2 * int(moved[0] > base[0]))
    return flags + sum((order + 1.7) * (1.0 + 1.0e6 * float(value))
                       for order, value in enumerate(base)) + 1.0e3 * float(moved[0])
""",
            "call": "objectivity(measure_mechanical_energy)",
            "gold_call": "objectivity(_oracle_measure_mechanical_energy)",
        },
        # --- Boundary: an unstrained body, so the total is purely kinetic ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(103)
n = 7
masses = np.full(n, 0.12)
velocities = 0.4 * rng.standard_normal((n, 2))
volumes = np.full(n, 1.0e-4)
density = np.zeros(n)
positions = 0.2 * rng.random((n, 2))

def reduce_four(parts):
    return float(sum((order + 1.7) * (1.0 + 1.0e6 * float(value))
                     for order, value in enumerate(parts)))
""",
            "call": "reduce_four(measure_mechanical_energy(masses, velocities, volumes, density, positions))",
            "gold_call": "reduce_four(_oracle_measure_mechanical_energy(masses, velocities, volumes, density, positions))",
        },
        # --- Edge: a rigidly spinning body, whose angular momentum is large ---
        {
            "setup": """import numpy as np
spacing, cells = 0.02, 6
offsets = (np.arange(2) + 0.5) / 2.0
line = (np.arange(cells)[:, None] + offsets[None, :]).ravel() * spacing
grid_x, grid_y = np.meshgrid(line, line, indexing='ij')
positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
n = positions.shape[0]
volumes = np.full(n, spacing * spacing / 4.0)
masses = 1200.0 * volumes
centre = positions.mean(axis=0)
spin = 12.0
lever = positions - centre
velocities = spin * np.stack([-lever[:, 1], lever[:, 0]], axis=1)
density = np.zeros(n)

def reduce_four(parts):
    return float(sum((order + 1.7) * (1.0 + 1.0e6 * float(value))
                     for order, value in enumerate(parts)))
""",
            "call": "reduce_four(measure_mechanical_energy(masses, velocities, volumes, density, positions))",
            "gold_call": "reduce_four(_oracle_measure_mechanical_energy(masses, velocities, volumes, density, positions))",
        },
        # --- Invalid: a negative strain energy density ---
        {
            "setup": """import numpy as np
masses = np.full(3, 0.12)
velocities = np.zeros((3, 2))
volumes = np.full(3, 1.0e-4)
density = np.array([100.0, -5.0, 200.0])
positions = np.zeros((3, 2))
def run_model():
    try:
        measure_mechanical_energy(masses, velocities, volumes, density, positions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_measure_mechanical_energy(masses, velocities, volumes, density, positions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a position array that does not match the masses ---
        {
            "setup": """import numpy as np
masses = np.full(3, 0.12)
velocities = np.zeros((3, 2))
volumes = np.full(3, 1.0e-4)
density = np.zeros(3)
positions = np.zeros((4, 2))
def run_model():
    try:
        measure_mechanical_energy(masses, velocities, volumes, density, positions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_measure_mechanical_energy(masses, velocities, volumes, density, positions)
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
