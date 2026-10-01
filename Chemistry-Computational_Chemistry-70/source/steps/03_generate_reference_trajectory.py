"""
Implement a function that integrates the exact, unapproximated Lennard-Jones

equations of motion forward from a given initial state using velocity-Verlet,

producing the deterministic reference trajectory that every approximate

solver in this problem (the RBL stochastic-force solver and the DINaMo-style

physics-informed solver) is judged against.

Both RBMD 2.0's random batch list estimator and DINaMo's physics-informed

neural solver are approximations to the same underlying microcanonical

initial-value problem: given (r0, v0), integrate Newton's equations under the

Lennard-Jones force law of Step 1. Before either approximation can be

evaluated, this exact trajectory must exist as ground truth. Velocity-Verlet

is used because it is the symplectic, time-reversible integrator both source

papers use to generate their own reference dynamics, so any error later

attributed to RBL's stochastic force estimator or to DINaMo's learned

correction is measured relative to the same numerical integration scheme,

not confounded by a difference in integrator.

Returns
-------
np.ndarray of shape (n_steps + 1, N, 3), the wrapped exact reference positions at every integration frame from t=0 through t=n_steps*dt
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def generate_reference_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                   box_length: float, n_steps: int, dt: float = 5e-4,
                                   cutoff: float = 2.937, epsilon: float = 1.0,
                                   sigma: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    positions0 : np.ndarray
        Array of shape (N, 3) with N >= 2, the initial particle positions.
    velocities0 : np.ndarray
        Array of shape (N, 3), matching `positions0`, the initial particle
        velocities.
    box_length : float
        Side length of the cubic periodic box.
    n_steps : int
        Number of integration steps to take, >= 0.
    dt : float, optional
        Integration timestep. Default 5e-4.
    cutoff : float, optional
        Lennard-Jones interaction cutoff. Default 2.937.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_steps + 1, N, 3): the wrapped particle positions at
        t = 0, dt, 2*dt, ..., n_steps*dt, with `trajectory[0]` equal to
        `positions0` wrapped into `[0, box_length)`.

    Raises
    ------
    ValueError
        If `positions0` does not have shape (N, 3) with N >= 2, or
        contains non-finite values; if `velocities0` does not match
        `positions0`'s shape or contains non-finite values; if
        `box_length`, `dt`, `epsilon`, or `sigma` is not finite and > 0;
        if `n_steps` is not a non-negative integer; or if `cutoff` is not
        finite, > 0, and < `box_length` / 2.
    """
    n = positions0.shape[0]
    trajectory = np.zeros((n_steps + 1, n, 3))  # placeholder
    return trajectory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_reference_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                           box_length: float, n_steps: int, dt: float = 5e-4,
                                           cutoff: float = 2.937, epsilon: float = 1.0,
                                           sigma: float = 1.0) -> np.ndarray:
    """Reference implementation."""

    import numpy as np

    def _is_nonneg_int(x):
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 0

    positions0 = np.asarray(positions0, dtype=float)
    velocities0 = np.asarray(velocities0, dtype=float)

    if positions0.ndim != 2 or positions0.shape[1] != 3 or positions0.shape[0] < 2:
        raise ValueError("positions0 must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(positions0)):
        raise ValueError("positions0 must contain only finite values.")
    if velocities0.shape != positions0.shape:
        raise ValueError("velocities0 must have the same shape as positions0.")
    if not np.all(np.isfinite(velocities0)):
        raise ValueError("velocities0 must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")
    if not _is_nonneg_int(n_steps):
        raise ValueError("n_steps must be a non-negative integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions0.shape[0]
    pos = positions0.copy() % box_length
    vel = velocities0.copy()

    trajectory = np.zeros((n_steps + 1, n, 3))
    trajectory[0] = pos

    force, _ = _oracle_lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
    for step in range(1, n_steps + 1):
        pos = (pos + vel * dt + 0.5 * force * dt * dt) % box_length
        force_new, _ = _oracle_lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
        vel = vel + 0.5 * (force + force_new) * dt
        force = force_new
        trajectory[step] = pos

    return trajectory

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: 4 well-separated atoms, several integration steps.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [2.2, 1.3, 1.1],\n"
                "    [1.4, 2.5, 1.6],\n"
                "    [2.6, 2.3, 2.0],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.3, -0.1, 0.2],\n"
                "    [-0.2, 0.4, -0.1],\n"
                "    [0.1, 0.1, -0.3],\n"
                "    [-0.2, -0.4, 0.2],\n"
                "])\n"
                "box_length = 5.0\n"
                "cutoff = 2.0\n"
                "n_steps = 5\n"
                "dt = 1e-3\n"
            ),
            "call": "generate_reference_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff)",
            "gold_call": "_oracle_generate_reference_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff)",
        },
        {
            # Boundary case: n_steps = 0, so the trajectory must contain
            # exactly one frame equal to the (wrapped) initial positions.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [0.5, 0.5, 0.5],\n"
                "    [6.2, -1.3, 2.7],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.1, 0.0, -0.1],\n"
                "    [-0.1, 0.2, 0.0],\n"
                "])\n"
                "box_length = 6.0\n"
                "cutoff = 2.5\n"
                "n_steps = 0\n"
            ),
            "call": "generate_reference_trajectory(positions0, velocities0, box_length, n_steps, cutoff=cutoff)",
            "gold_call": "_oracle_generate_reference_trajectory(positions0, velocities0, box_length, n_steps, cutoff=cutoff)",
        },
        {
            # Edge case: two atoms in the strongly repulsive regime with a
            # small timestep, checking the integrator stays finite and
            # well-behaved under a stiff force.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [4.0, 4.0, 4.0],\n"
                "    [4.9, 4.0, 4.0],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.0, 0.0, 0.0],\n"
                "    [0.0, 0.0, 0.0],\n"
                "])\n"
                "box_length = 8.0\n"
                "cutoff = 3.0\n"
                "n_steps = 10\n"
                "dt = 1e-4\n"
            ),
            "call": "generate_reference_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff)",
            "gold_call": "_oracle_generate_reference_trajectory(positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff)",
        },
    ]
