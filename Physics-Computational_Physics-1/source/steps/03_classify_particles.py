"""
Implement classify_particles to sample the free-transport time of each
particle and classify it as collisional or collisionless.

Collisionless particles survive the entire time step without interaction.

Collisional particles undergo at least one collision during the step.

The collision-time distribution determines how the particle population is

split between these two contributions.

Returns
-------
tuple[np.ndarray, np.ndarray], (t_f, is_collisionless), free-transport times capped at dt and the collisionless mask
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def classify_particles(
    tau: float, dt: float, rng: np.random.Generator, n_particles: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    tau : float
        Characteristic collision time, tau = 1 / (v * Sigma).
    dt : float
        Time step size.
    rng : numpy.random.Generator
        Random number generator. Draw one uniform survival probability per
        particle from rng.random in particle order.
    n_particles : int
        Number of particles to classify.

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray]
        (t_f, is_collisionless) where t_f has shape (n_particles,) giving the
        free transport time for each particle (capped at dt), and
        is_collisionless is a boolean array (True if t_f == dt).

    Raises
    ------
    ValueError
        If n_particles < 0, or tau or dt is not positive.
    """
    return t_f, is_collisionless

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_classify_particles(
    tau: float, dt: float, rng: np.random.Generator, n_particles: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample free transport times and classify particles."""
    if n_particles < 0 or tau <= 0 or dt <= 0:
        raise ValueError("need n_particles >= 0 and positive tau and dt")
    r = rng.random(n_particles)
    t_f = np.minimum(-tau * np.log(r), dt)
    is_collisionless = t_f >= dt
    return t_f, is_collisionless

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # Normal: moderate tau/dt, check moments and time/mask consistency
        {
            "setup": """import numpy as np

def run_model():
    rng = np.random.default_rng(42)
    t_f, mask = classify_particles(0.5, 0.1, rng, 100000)
    frac = float(np.mean(mask))
    mean_tf = float(np.mean(t_f))
    consistent = float(np.array_equal(mask, t_f == 0.1))
    return np.array([round(frac, 2), round(mean_tf, 4), consistent])

def run_gold():
    rng = np.random.default_rng(42)
    t_f, mask = _oracle_classify_particles(0.5, 0.1, rng, 100000)
    frac = float(np.mean(mask))
    mean_tf = float(np.mean(t_f))
    consistent = float(np.array_equal(mask, t_f == 0.1))
    return np.array([round(frac, 2), round(mean_tf, 4), consistent])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Boundary: free-streaming limit (tau >> dt), nearly all collisionless
        {
            "setup": """import numpy as np

def run_model():
    rng = np.random.default_rng(42)
    _, mask = classify_particles(1000.0, 0.01, rng, 50000)
    return round(float(np.mean(mask)), 4)

def run_gold():
    rng = np.random.default_rng(42)
    _, mask = _oracle_classify_particles(1000.0, 0.01, rng, 50000)
    return round(float(np.mean(mask)), 4)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Edge: diffusion limit (tau << dt), nearly all collisional
        {
            "setup": """import numpy as np

def run_model():
    rng = np.random.default_rng(42)
    t_f, mask = classify_particles(0.001, 1.0, rng, 50000)
    frac_cl = float(np.mean(mask))
    max_tf = float(np.max(t_f))
    return np.array([round(frac_cl, 4), float(max_tf <= 1.0)])

def run_gold():
    rng = np.random.default_rng(42)
    t_f, mask = _oracle_classify_particles(0.001, 1.0, rng, 50000)
    frac_cl = float(np.mean(mask))
    max_tf = float(np.max(t_f))
    return np.array([round(frac_cl, 4), float(max_tf <= 1.0)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
