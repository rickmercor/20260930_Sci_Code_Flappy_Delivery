"""
Implement resample_particles to create new particles from the updated
macroscopic distribution after a UGKWP time step.

Resampling converts the share of the macroscopic distribution assigned to
particles into new particles, cell by cell. For positive assigned cell
mass, the particle count is n_i = round(total_mass / m_e). Cells with
non-positive mass or zero rounded count produce no particles. When n_i > 0,
each particle has mass total_mass / n_i, preserving the assigned mass in
that sampled cell. Particle positions are uniform within each cell, and
velocities are isotropic.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], (x_resamp, xi_resamp, w_resamp), positions, velocities and masses of the resampled particles
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def resample_particles(
    psi_ma: np.ndarray,
    dt: float,
    tau: float,
    dx: float,
    m_e: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    psi_ma : numpy.ndarray
        Macroscopic part of the scalar flux, shape (N_x,).
    dt : float
        Nonnegative time step.
    tau : float
        Positive characteristic collision time.
    dx : float
        Positive cell width.
    m_e : float
        Positive target mass per particle.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (x_resamp, xi_resamp, w_resamp) where x_resamp and xi_resamp are the
        positions and velocities of resampled particles, and w_resamp is the
        per-particle mass for each resampled particle.

    Raises
    ------
    ValueError
        If dt is negative, or tau, m_e or dx is not positive.

    Notes
    -----
    Random draws must follow this exact order for reproducibility: cells are
    processed in ascending order, and for each particle one uniform position
    draw is taken first, then one uniform velocity draw on [-1, 1].
    """
    return x_resamp, xi_resamp, w_resamp

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_resample_particles(
    psi_ma: np.ndarray,
    dt: float,
    tau: float,
    dx: float,
    m_e: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent positive cell masses whose rounded particle counts are nonzero."""
    if m_e <= 0 or dx <= 0 or dt < 0 or tau <= 0:
        raise ValueError("dt must be nonnegative; tau, m_e and dx must be positive")
    N_x = len(psi_ma)
    free_fraction = np.exp(-dt / tau)

    total = 0
    for i in range(N_x):
        total_mass = free_fraction * psi_ma[i] * dx
        if total_mass > 0.0:
            n_i = int(np.round(total_mass / m_e))
            total += n_i

    x_resamp = np.empty(total)
    xi_resamp = np.empty(total)
    w_resamp = np.empty(total)

    idx = 0
    for i in range(N_x):
        total_mass = free_fraction * psi_ma[i] * dx
        if total_mass <= 0.0:
            continue
        n_i = int(np.round(total_mass / m_e))
        if n_i == 0:
            continue
        actual_w = total_mass / n_i
        x_lo = i * dx
        for k in range(n_i):
            x_resamp[idx] = x_lo + rng.random() * dx
            xi_resamp[idx] = 2.0 * rng.random() - 1.0
            w_resamp[idx] = actual_w
            idx += 1

    return x_resamp, xi_resamp, w_resamp

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # Zero elapsed time retains the full available mass in this cell
        {
            "setup": """import numpy as np

def _zero_time_mass(fn):
    rng = np.random.default_rng(42)
    _, _, weights = fn(np.ones(1), 0.0, 1.0, 1.0, 0.1, rng)
    return float(np.sum(weights))
""",
            "call": "_zero_time_mass(resample_particles)",
            "gold_call": "_zero_time_mass(_oracle_resample_particles)",
        },
        # Negative elapsed time must raise the documented invalid-data exception
        {
            "setup": """import numpy as np

def _negative_time(fn):
    try:
        fn(np.ones(1), -0.1, 1.0, 1.0, 0.1, np.random.default_rng(42))
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_negative_time(resample_particles)",
            "gold_call": "_negative_time(_oracle_resample_particles)",
        },
        # Zero collision time must raise ValueError before the survival calculation
        {
            "setup": """import numpy as np

def _zero_collision_time(fn):
    try:
        fn(np.ones(1), 0.1, 0.0, 1.0, 0.1, np.random.default_rng(42))
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_zero_collision_time(resample_particles)",
            "gold_call": "_zero_collision_time(_oracle_resample_particles)",
        },
        # Normal: moderate free fraction, check per-cell placement, mass and velocities
        {
            "setup": """import numpy as np

def run_model():
    rng = np.random.default_rng(42)
    psi_ma = np.full(10, 1.5)
    dx = 0.1
    tau = 1.0 / 11.0
    dt = 0.2 * dx
    m_e = dx / 200
    x_r, xi_r, w_r = resample_particles(psi_ma, dt, tau, dx, m_e, rng)
    total_mass = float(np.sum(w_r)) if len(w_r) > 0 else 0.0
    cell_counts = np.bincount(np.floor(x_r / dx).astype(int), minlength=len(psi_ma))
    summary = np.array([len(x_r), round(total_mass, 6), round(float(np.mean(xi_r)), 6),
                        round(float(np.min(xi_r)), 6), round(float(np.max(xi_r)), 6)])
    return np.concatenate((summary, cell_counts))

def run_gold():
    rng = np.random.default_rng(42)
    psi_ma = np.full(10, 1.5)
    dx = 0.1
    tau = 1.0 / 11.0
    dt = 0.2 * dx
    m_e = dx / 200
    x_r, xi_r, w_r = _oracle_resample_particles(psi_ma, dt, tau, dx, m_e, rng)
    total_mass = float(np.sum(w_r)) if len(w_r) > 0 else 0.0
    cell_counts = np.bincount(np.floor(x_r / dx).astype(int), minlength=len(psi_ma))
    summary = np.array([len(x_r), round(total_mass, 6), round(float(np.mean(xi_r)), 6),
                        round(float(np.min(xi_r)), 6), round(float(np.max(xi_r)), 6)])
    return np.concatenate((summary, cell_counts))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Boundary: diffusion limit, nearly zero particles resampled
        {
            "setup": """import numpy as np

def run_model():
    rng = np.random.default_rng(42)
    psi_ma = np.full(5, 1.0)
    dx = 0.2
    tau = 0.001
    dt = 1.0
    m_e = 0.01
    x_r, _, w_r = resample_particles(psi_ma, dt, tau, dx, m_e, rng)
    return len(x_r)

def run_gold():
    rng = np.random.default_rng(42)
    psi_ma = np.full(5, 1.0)
    dx = 0.2
    tau = 0.001
    dt = 1.0
    m_e = 0.01
    x_r, _, w_r = _oracle_resample_particles(psi_ma, dt, tau, dx, m_e, rng)
    return len(x_r)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Edge: some cells with zero psi_ma, check no particles created there
        {
            "setup": """import numpy as np

def run_model():
    rng = np.random.default_rng(42)
    psi_ma = np.array([1.0, 0.0, 2.0, 0.0, 1.0])
    dx = 0.2
    tau = 0.5
    dt = 0.1
    m_e = 0.01
    x_r, xi_r, w_r = resample_particles(psi_ma, dt, tau, dx, m_e, rng)
    in_cell1 = int(np.sum((x_r >= 0.2) & (x_r < 0.4)))
    in_cell3 = int(np.sum((x_r >= 0.6) & (x_r < 0.8)))
    return np.array([in_cell1, in_cell3, float(len(x_r) > 0)])

def run_gold():
    rng = np.random.default_rng(42)
    psi_ma = np.array([1.0, 0.0, 2.0, 0.0, 1.0])
    dx = 0.2
    tau = 0.5
    dt = 0.1
    m_e = 0.01
    x_r, xi_r, w_r = _oracle_resample_particles(psi_ma, dt, tau, dx, m_e, rng)
    in_cell1 = int(np.sum((x_r >= 0.2) & (x_r < 0.4)))
    in_cell3 = int(np.sum((x_r >= 0.6) & (x_r < 0.8)))
    return np.array([in_cell1, in_cell3, float(len(x_r) > 0)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
