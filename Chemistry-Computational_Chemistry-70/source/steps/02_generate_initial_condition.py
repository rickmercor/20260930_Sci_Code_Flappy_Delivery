"""
Implement a function that builds the single shared Lennard-Jones initial

condition (particle positions, velocities, and box length) used by every

downstream solver in this problem. Positions must be produced by

overlap-guarded random insertion into a cubic periodic box and then

energy-relaxed; velocities must be drawn at a target reduced temperature,

have their center-of-mass component removed, and then be carried through a

short thermostatted dynamics burn-in so they become dynamically consistent

with the relaxed configuration.

RBMD 2.0's random batch list estimator and DINaMo's physics-informed

trajectory solver are compared on one shared initial-value problem, so both

must start from exactly the same deterministic state. A naive random packing

followed by independently-sampled velocities is not adequate: velocities that

are uncorrelated with the local force environment can, under DINaMo's purely

ballistic initial-condition ansatz (Step 6), extrapolate two atoms into an

unphysically close approach within the evaluation window even when the

starting configuration itself is force-balanced. A short thermostatted burn-in

resolves this by letting real pairwise repulsion (Step 1) shape the

velocity-position correlations before the production window begins, exactly

as the equilibrate-then-simulate protocol used by both source papers requires.

Returns
-------
tuple of (np.ndarray of shape (n_atoms, 3),  np.ndarray of shape (n_atoms, 3), float, float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def generate_initial_condition(n_atoms: int, density: float, temperature: float,
                                pos_seed: int, vel_seed: int,
                                min_separation: float = 0.8, cutoff: float = 2.937,
                                epsilon: float = 1.0, sigma: float = 1.0,
                                minimize_steps: int = 100, burn_in_steps: int = 1000,
                                dt: float = 5e-4, rescale_interval: int = 20,
                                max_attempts: int = 200000) -> tuple:
    """

    Parameters
    ----------
    n_atoms : int
        Number of atoms, must be >= 2.
    density : float
        Reduced number density rho = n_atoms / box_length**3.
    temperature : float
        Target reduced temperature Theta for the Maxwell-Boltzmann velocities.
    pos_seed : int
        Seed for the position-insertion random number generator.
    vel_seed : int
        Seed for the velocity-sampling random number generator.
    min_separation : float, optional
        Minimum allowed minimum-image separation between inserted atoms, in
        units of sigma. Default 0.8.
    cutoff : float, optional
        Lennard-Jones interaction cutoff used for minimization, burn-in, and
        energy evaluation. Default 2.937.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.
    minimize_steps : int, optional
        Number of fixed-step, always-accepted steepest-descent relaxation
        iterations (see notes below). Default 100. A value of 0 skips
        minimization entirely.
    burn_in_steps : int, optional
        Number of thermostatted velocity-Verlet burn-in steps. Default 1000.
        A value of 0 skips the burn-in entirely.
    dt : float, optional
        Burn-in integration timestep. Default 5e-4.
    rescale_interval : int, optional
        Number of burn-in steps between temperature rescalings. Default 20.
    max_attempts : int, optional
        Maximum number of candidate positions tried during insertion before
        giving up. Default 200000.

    Returns
    -------
    positions : np.ndarray
        Array of shape (n_atoms, 3), the relaxed, wrapped particle positions.
    velocities : np.ndarray
        Array of shape (n_atoms, 3), the burned-in particle velocities.
    box_length : float
        Side length of the cubic periodic box, (n_atoms / density) ** (1/3),
        as a native Python float.

    Raises
    ------
    ValueError
        If `n_atoms` is not an integer >= 2; if `density`, `temperature`,
        or `dt` is not finite and > 0; if `pos_seed` or `vel_seed` is not
        a non-negative integer; if `minimize_steps` or `burn_in_steps` is
        not a non-negative integer; if `rescale_interval` or
        `max_attempts` is not a positive integer; if `min_separation` or
        `cutoff` is not finite, > 0, and < `box_length` / 2; if `epsilon`
        or `sigma` is not finite and > 0; or if no valid configuration of
        `n_atoms` particles satisfying `min_separation` can be placed
        within `max_attempts` insertion attempts.
    """
    positions = np.zeros((n_atoms, 3))  # placeholder
    velocities = np.zeros((n_atoms, 3))  # placeholder
    box_length = (n_atoms / density) ** (1.0 / 3.0)  # placeholder
    return positions, velocities, box_length

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_initial_condition(n_atoms: int, density: float, temperature: float,
                                        pos_seed: int, vel_seed: int,
                                        min_separation: float = 0.8, cutoff: float = 2.937,
                                        epsilon: float = 1.0, sigma: float = 1.0,
                                        minimize_steps: int = 100, burn_in_steps: int = 1000,
                                        dt: float = 5e-4, rescale_interval: int = 20,
                                        max_attempts: int = 200000) -> tuple:
    """Reference implementation."""
    import numpy as np

    def _is_nonneg_int(x):
        return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 0

    if not (isinstance(n_atoms, (int, np.integer)) and not isinstance(n_atoms, bool) and n_atoms >= 2):
        raise ValueError("n_atoms must be an integer >= 2.")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and > 0.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0.")
    if not _is_nonneg_int(pos_seed):
        raise ValueError("pos_seed must be a non-negative integer.")
    if not _is_nonneg_int(vel_seed):
        raise ValueError("vel_seed must be a non-negative integer.")
    if not _is_nonneg_int(minimize_steps):
        raise ValueError("minimize_steps must be a non-negative integer.")
    if not _is_nonneg_int(burn_in_steps):
        raise ValueError("burn_in_steps must be a non-negative integer.")
    if not (isinstance(rescale_interval, (int, np.integer)) and rescale_interval > 0):
        raise ValueError("rescale_interval must be a positive integer.")
    if not (isinstance(max_attempts, (int, np.integer)) and max_attempts > 0):
        raise ValueError("max_attempts must be a positive integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0.")

    box_length = (n_atoms / density) ** (1.0 / 3.0)

    if not np.isfinite(min_separation) or min_separation <= 0 or min_separation >= box_length / 2:
        raise ValueError("min_separation must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    def _min_image(vec, box):
        return vec - box * np.round(vec / box)

    def _minimize(pos, n_iter):
        if n_iter == 0:
            return pos
        pos = pos.copy()
        max_disp = 0.005
        for _ in range(n_iter):
            F, _ = _oracle_lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
            fmax = np.max(np.linalg.norm(F, axis=1))
            if fmax < 1e-6:
                break
            s = max_disp / max(fmax, 1e-12)
            pos = (pos + s * F) % box_length
        return pos

    def _rescale(v, dof):
        ke = 0.5 * np.sum(v ** 2)
        t_inst = 2.0 * ke / dof
        return v * np.sqrt(temperature / t_inst)

    def _burn_in(pos, vel, n_steps, dof):
        if n_steps == 0:
            return pos, vel
        pos = pos.copy()
        vel = vel.copy()
        F, _ = _oracle_lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
        for step in range(1, n_steps + 1):
            pos = (pos + vel * dt + 0.5 * F * dt * dt) % box_length
            F_new, _ = _oracle_lj_force_energy(pos, box_length, cutoff, epsilon, sigma)
            vel = vel + 0.5 * (F + F_new) * dt
            F = F_new
            if step % rescale_interval == 0:
                vel = _rescale(vel, dof)
        vel = vel - vel.mean(axis=0, keepdims=True)
        vel = _rescale(vel, dof)
        return pos, vel

    rng_pos = np.random.default_rng(pos_seed)
    positions = np.zeros((n_atoms, 3))
    placed = 0
    attempts = 0
    while placed < n_atoms:
        candidate = rng_pos.random(3) * box_length
        ok = True
        for k in range(placed):
            d = np.linalg.norm(_min_image(candidate - positions[k], box_length))
            if d < min_separation:
                ok = False
                break
        if ok:
            positions[placed] = candidate
            placed += 1
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Could not place {n_atoms} particles with min_separation="
                f"{min_separation} in box_length={box_length} within "
                f"{max_attempts} attempts; density may be too high."
            )

    positions = _minimize(positions, minimize_steps)

    rng_vel = np.random.default_rng(vel_seed)
    velocities = rng_vel.normal(loc=0.0, scale=np.sqrt(temperature), size=(n_atoms, 3))
    velocities = velocities - velocities.mean(axis=0, keepdims=True)
    dof = 3 * n_atoms - 3
    velocities = _rescale(velocities, dof)

    positions, velocities = _burn_in(positions, velocities, burn_in_steps, dof)

    return positions, velocities, float(box_length)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: 6 atoms, moderate density, short but nonzero
            # minimization and burn-in.
            "setup": (
                "n_atoms = 6\n"
                "density = 0.15\n"
                "temperature = 1.5\n"
                "pos_seed = 1\n"
                "vel_seed = 2\n"
                "cutoff = 1.5\n"
                "minimize_steps = 50\n"
                "burn_in_steps = 50\n"
            ),
            "call": (
                "generate_initial_condition(n_atoms, density, temperature, pos_seed, vel_seed, "
                "cutoff=cutoff, minimize_steps=minimize_steps, burn_in_steps=burn_in_steps)"
            ),
            "gold_call": (
                "_oracle_generate_initial_condition(n_atoms, density, temperature, pos_seed, vel_seed, "
                "cutoff=cutoff, minimize_steps=minimize_steps, burn_in_steps=burn_in_steps)"
            ),
        },
        {
            # Boundary case: the minimum allowed n_atoms (2), with
            # minimize_steps and burn_in_steps at their boundary value of 0,
            # exercising the skip-relaxation code paths.
            "setup": (
                "n_atoms = 2\n"
                "density = 0.1\n"
                "temperature = 1.0\n"
                "pos_seed = 5\n"
                "vel_seed = 7\n"
                "cutoff = 1.2\n"
            ),
            "call": (
                "generate_initial_condition(n_atoms, density, temperature, pos_seed, vel_seed, "
                "cutoff=cutoff, minimize_steps=0, burn_in_steps=0)"
            ),
            "gold_call": (
                "_oracle_generate_initial_condition(n_atoms, density, temperature, pos_seed, vel_seed, "
                "cutoff=cutoff, minimize_steps=0, burn_in_steps=0)"
            ),
        },
        {
            # Edge case: tighter packing (higher density, larger
            # min_separation relative to spacing) that still succeeds, with a
            # non-default epsilon/sigma and rescale_interval.
            "setup": (
                "n_atoms = 10\n"
                "density = 0.3\n"
                "temperature = 2.0\n"
                "pos_seed = 11\n"
                "vel_seed = 13\n"
                "min_separation = 0.85\n"
                "cutoff = 1.5\n"
                "epsilon = 1.2\n"
                "sigma = 1.0\n"
            ),
            "call": (
                "generate_initial_condition(n_atoms, density, temperature, pos_seed, vel_seed, "
                "min_separation=min_separation, cutoff=cutoff, epsilon=epsilon, sigma=sigma, "
                "minimize_steps=20, burn_in_steps=20, rescale_interval=5)"
            ),
            "gold_call": (
                "_oracle_generate_initial_condition(n_atoms, density, temperature, pos_seed, vel_seed, "
                "min_separation=min_separation, cutoff=cutoff, epsilon=epsilon, sigma=sigma, "
                "minimize_steps=20, burn_in_steps=20, rescale_interval=5)"
            ),
        },
    ]
