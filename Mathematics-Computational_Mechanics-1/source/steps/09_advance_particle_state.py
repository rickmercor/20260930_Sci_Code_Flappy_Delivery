"""
Advect the material points and update their deformation gradients from the converged velocity field of the step.

The deformation gradient of a material point is advanced multiplicatively by the increment the velocity gradient accumulates over the step, which keeps the update objective and lets arbitrarily large total deformation build up from small increments. The positions follow the interpolated particle velocity, so the two updates are driven by the same converged nodal field.

Returns
-------
tuple of two np.ndarray: particle positions of shape (n_particles, 2) and deformation gradients of shape (n_particles, 2, 2), both at the end of the step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_particle_state(positions: "np.ndarray", deformation_gradients: "np.ndarray",
                           particle_velocities: "np.ndarray", velocity_gradients: "np.ndarray",
                           step: float) -> tuple:
    """Advect the material points and update their deformation gradients.

    The deformation gradient of a particle is multiplied on the left by the
    identity plus the step times the velocity gradient, in that order, so that
    the update composes the incremental motion with the deformation already
    accumulated. The position is advanced by the step times the particle
    velocity.

    Parameters
    ----------
    positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle positions at the
        start of the step, in metres.
    deformation_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the start of the step.
    particle_velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities at the
        end of the step, in m s^-1.
    velocity_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the particle velocity
        gradients at the end of the step, in s^-1, with the derivative
        direction as the second index.
    step : float
        Time step in seconds (step > 0).

    Returns
    -------
    updated_positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle positions at the
        end of the step, in metres.
    updated_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the end of the step.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if the update would leave a particle with
        a non-positive determinant.
    """
    return updated_positions, updated_gradients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_advance_particle_state(positions: "np.ndarray", deformation_gradients: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned arrays
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: benchmark step with a mixed velocity gradient (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(91)
n = 8
positions = 0.2 * rng.random((n, 2))
gradients = np.eye(2)[None] + 0.15 * rng.standard_normal((n, 2, 2))
velocities = 0.5 * rng.standard_normal((n, 2))
rates = 30.0 * rng.standard_normal((n, 2, 2))
step = 1.0e-3

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
            "call": "pin_all(advance_particle_state(positions, gradients, velocities, rates, step))",
            "gold_call": "pin_all(_oracle_advance_particle_state(positions, gradients, velocities, rates, step))",
        },
        # --- Valid: composition of the multiplicative update over two half steps ---
        # Applying the same constant velocity gradient over two half steps is not
        # the same as applying it once over the whole step, and the difference is
        # of the order of the square of the step, which pins the ordering of the
        # multiplication as well as its size.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(92)
n = 6
positions = 0.2 * rng.random((n, 2))
gradients = np.eye(2)[None] + 0.2 * rng.standard_normal((n, 2, 2))
velocities = 0.4 * rng.standard_normal((n, 2))
rates = 25.0 * rng.standard_normal((n, 2, 2))
step = 2.0e-3

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def splitting(fn):
    once = fn(positions, gradients, velocities, rates, step)
    first = fn(positions, gradients, velocities, rates, 0.5 * step)
    twice = fn(first[0], first[1], velocities, rates, 0.5 * step)
    gap = np.abs(once[1] - twice[1]).max()
    drift = np.abs(once[0] - twice[0]).max()
    flags = float(int(gap > 1.0e-9) + 2 * int(drift < 1.0e-15))
    return flags + pin(once[1]) + 3.0 * pin(twice[1]) + 1.0e6 * gap
""",
            "call": "splitting(advance_particle_state)",
            "gold_call": "splitting(_oracle_advance_particle_state)",
        },
        # --- Boundary: a vanishing velocity gradient, so only the positions move ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(93)
n = 5
positions = 0.2 * rng.random((n, 2))
gradients = np.eye(2)[None] + 0.1 * rng.standard_normal((n, 2, 2))
velocities = 0.6 * rng.standard_normal((n, 2))
rates = np.zeros((n, 2, 2))
step = 1.0e-3

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
            "call": "pin_all(advance_particle_state(positions, gradients, velocities, rates, step))",
            "gold_call": "pin_all(_oracle_advance_particle_state(positions, gradients, velocities, rates, step))",
        },
        # --- Edge: a strongly compressive rate that nearly closes the volume ---
        {
            "setup": """import numpy as np
n = 3
positions = np.array([[0.01, 0.02], [0.05, 0.07], [0.11, 0.13]])
gradients = np.stack([np.diag([1.2, 1.0 / 1.2]), np.eye(2), np.diag([0.7, 1.3])])
velocities = np.array([[0.1, -0.2], [0.0, 0.0], [-0.3, 0.4]])
rates = np.tile(np.array([[-450.0, 20.0], [-15.0, -430.0]]), (n, 1, 1))
step = 1.0e-3

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
            "call": "pin_all(advance_particle_state(positions, gradients, velocities, rates, step))",
            "gold_call": "pin_all(_oracle_advance_particle_state(positions, gradients, velocities, rates, step))",
        },
        # --- Invalid: a rate that inverts a particle over the step ---
        {
            "setup": """import numpy as np
positions = np.array([[0.01, 0.02]])
gradients = np.tile(np.eye(2), (1, 1, 1))
velocities = np.zeros((1, 2))
rates = np.array([[[-3000.0, 0.0], [0.0, -100.0]]])
def run_model():
    try:
        advance_particle_state(positions, gradients, velocities, rates, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_advance_particle_state(positions, gradients, velocities, rates, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a velocity array that does not match the positions ---
        {
            "setup": """import numpy as np
positions = np.zeros((3, 2))
gradients = np.tile(np.eye(2), (3, 1, 1))
velocities = np.zeros((2, 2))
rates = np.zeros((3, 2, 2))
def run_model():
    try:
        advance_particle_state(positions, gradients, velocities, rates, 1.0e-3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_advance_particle_state(positions, gradients, velocities, rates, 1.0e-3)
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
