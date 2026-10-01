"""
Implement a function that integrates the Lennard-Jones equations of motion

forward from a given initial state using velocity-Verlet, exactly as in

Step 3, but replacing the exact pairwise force evaluation at every step with

RBMD 2.0's random batch list (RBL) estimator from Step 4, threading a single

continuing pseudo-random stream through the whole trajectory rather than

reseeding at each step.

RBL is a per-step force approximation, so its effect on trajectory accuracy

can only be assessed by actually integrating with it over many steps, not by

evaluating it once. Using one continuing `np.random.default_rng(seed)`

stream across the trajectory (rather than a fresh seed each step) matches

how RBMD 2.0 is actually deployed: the random batch is redrawn every

timestep, but the sequence of draws forms a single reproducible stream for

the whole run. The resulting trajectory is what Step 7 will compare against

the exact reference trajectory of Step 3 to quantify RBL's approximation

error under this problem's shared initial condition.

Returns
-------
np.ndarray of shape (n_steps + 1, N, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def generate_rbl_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                             box_length: float, n_steps: int, core_cutoff: float,
                             cutoff: float, batch_size: int, seed: int,
                             dt: float = 5e-4, epsilon: float = 1.0,
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
    core_cutoff : float
        RBL core cutoff; neighbors within this distance are always included
        exactly at every step.
    cutoff : float
        Full interaction cutoff at which the underlying shifted-force
        Lennard-Jones law vanishes; must exceed `core_cutoff`.
    batch_size : int
        Number of shell neighbors sampled per particle per step when the
        shell is larger than `batch_size`.
    seed : int
        Seed for the single `np.random.default_rng` stream used across all
        steps.
    dt : float, optional
        Integration timestep. Default 5e-4.
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
        `box_length`, `core_cutoff`, `dt`, `epsilon`, or `sigma` is not
        finite and > 0; if `n_steps` is not a non-negative integer; if
        `cutoff` is not finite, > `core_cutoff`, and < `box_length` / 2;
        if `batch_size` is not a positive integer; or if `seed` is not a
        non-negative integer.
    """
    n = positions0.shape[0]
    trajectory = np.zeros((n_steps + 1, n, 3))  # placeholder
    return trajectory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_rbl_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                     box_length: float, n_steps: int, core_cutoff: float,
                                     cutoff: float, batch_size: int, seed: int,
                                     dt: float = 5e-4, epsilon: float = 1.0,
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
    if not np.isfinite(core_cutoff) or core_cutoff <= 0:
        raise ValueError("core_cutoff must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= core_cutoff or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > core_cutoff, and < box_length / 2.")
    if not (isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, bool) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer.")
    if not _is_nonneg_int(seed):
        raise ValueError("seed must be a non-negative integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions0.shape[0]
    rng = np.random.default_rng(seed)
    pos = positions0.copy() % box_length
    vel = velocities0.copy()

    trajectory = np.zeros((n_steps + 1, n, 3))
    trajectory[0] = pos

    force = _oracle_rbl_force(pos, box_length, core_cutoff, cutoff, batch_size, rng, epsilon, sigma)
    for step in range(1, n_steps + 1):
        pos = (pos + vel * dt + 0.5 * force * dt * dt) % box_length
        force_new = _oracle_rbl_force(pos, box_length, core_cutoff, cutoff, batch_size, rng, epsilon, sigma)
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
            # Normal scenario: 5 particles, several steps, shell larger than
            # batch_size so stochastic subsampling is exercised every step.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.7, 1.1, 1.0],\n"
                "    [1.0, 1.7, 1.2],\n"
                "    [1.9, 1.9, 1.1],\n"
                "    [1.2, 1.3, 1.9],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.2, -0.1, 0.1],\n"
                "    [-0.1, 0.2, -0.1],\n"
                "    [0.1, 0.1, -0.2],\n"
                "    [-0.2, -0.1, 0.2],\n"
                "    [0.0, -0.1, 0.1],\n"
                "])\n"
                "box_length = 10.0\n"
                "core_cutoff = 1.0\n"
                "cutoff = 2.5\n"
                "batch_size = 2\n"
                "n_steps = 5\n"
                "dt = 1e-3\n"
                "seed = 123\n"
            ),
            "call": (
                "generate_rbl_trajectory(positions0, velocities0, box_length, n_steps, "
                "core_cutoff, cutoff, batch_size, seed, dt=dt)"
            ),
            "gold_call": (
                "_oracle_generate_rbl_trajectory(positions0, velocities0, box_length, n_steps, "
                "core_cutoff, cutoff, batch_size, seed, dt=dt)"
            ),
        },
        {
            # Boundary case: n_steps = 0, so the trajectory must contain
            # exactly one frame equal to the wrapped initial positions and no
            # random draws are consumed.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [0.5, 0.5, 0.5],\n"
                "    [11.2, -1.3, 2.7],\n"
                "])\n"
                "velocities0 = np.array([\n"
                "    [0.1, 0.0, -0.1],\n"
                "    [-0.1, 0.2, 0.0],\n"
                "])\n"
                "box_length = 12.0\n"
                "core_cutoff = 1.0\n"
                "cutoff = 3.0\n"
                "batch_size = 4\n"
                "n_steps = 0\n"
                "seed = 99\n"
            ),
            "call": (
                "generate_rbl_trajectory(positions0, velocities0, box_length, n_steps, "
                "core_cutoff, cutoff, batch_size, seed)"
            ),
            "gold_call": (
                "_oracle_generate_rbl_trajectory(positions0, velocities0, box_length, n_steps, "
                "core_cutoff, cutoff, batch_size, seed)"
            ),
        },
        {
            # Edge case: batch_size = 1, the most aggressive subsampling,
            # over several steps with a larger shell.
            "setup": (
                "import numpy as np\n"
                "positions0 = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.6, 1.0, 1.0],\n"
                "    [1.0, 1.6, 1.0],\n"
                "    [1.6, 1.6, 1.0],\n"
                "    [1.3, 1.3, 1.6],\n"
                "])\n"
                "velocities0 = np.zeros((5, 3))\n"
                "box_length = 10.0\n"
                "core_cutoff = 0.8\n"
                "cutoff = 2.5\n"
                "batch_size = 1\n"
                "n_steps = 8\n"
                "dt = 5e-4\n"
                "seed = 2026\n"
            ),
            "call": (
                "generate_rbl_trajectory(positions0, velocities0, box_length, n_steps, "
                "core_cutoff, cutoff, batch_size, seed, dt=dt)"
            ),
            "gold_call": (
                "_oracle_generate_rbl_trajectory(positions0, velocities0, box_length, n_steps, "
                "core_cutoff, cutoff, batch_size, seed, dt=dt)"
            ),
        },
    ]
