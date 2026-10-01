"""
The shared environmental integral is supplied by the preceding stage. Individual cells draw association rates and carrying capacities from replicate-specific log-normal laws, then convert continuous particle counts to empirical fluorescence.

Simulate fluorescence snapshots for one experimental replicate.

Returns
-------
np.ndarray, one replicate's fluorescence with axes (time, cell).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_hierarchical_fluorescence(
    log_params: np.ndarray,
    environment_integral: np.ndarray,
    n_cells: int,
    cell_calibration: np.ndarray,
    particle_calibration: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate fluorescence snapshots for one replicate.

    Parameters
    ----------
    log_params : np.ndarray
        Length-4 vector $(m_r,s_r,m_K,s_K)$ for one replicate.
    environment_integral : np.ndarray
        Shared integrated free-particle density at each observation time.
    n_cells : int
        Number of observed cells at each time.
    cell_calibration : np.ndarray
        Empirical cell-autofluorescence sample.
    particle_calibration : np.ndarray
        Empirical single-particle fluorescence sample.
    rng : np.random.Generator
        Shared PCG64 random-number stream, already advanced through this
        replicate's pilot-cell draws.

    Notes
    -----
    At each observation time, consume the stream by drawing direct log-normal
    rates, direct log-normal capacities, integer autofluorescence indices and one
    concatenated vector of integer particle-fluorescence indices, in that order.

    Returns
    -------
    result : np.ndarray
        Array of shape (time, cell).

    Raises
    ------
    ValueError
        If log_params does not have shape (4,), any log-normal scale is
        negative, the environment integral is empty or negative, n_cells is not
        positive, or either calibration array is empty.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _sample_particle_fluorescence_vectorised(particle_count, particle_calibration, rng):
    p = np.maximum(0.0, np.asarray(particle_count, dtype=float).ravel())
    calibration = np.asarray(particle_calibration, dtype=float).ravel()
    if calibration.size == 0:
        raise ValueError("particle_calibration must be non-empty")

    n_full = np.floor(p).astype(np.int64)
    frac = p - n_full
    has_fraction = frac > 0.0
    n_draw = n_full + has_fraction.astype(np.int64)
    total_draws = int(np.sum(n_draw))
    if total_draws == 0:
        return np.zeros_like(p)

    draws = calibration[rng.integers(0, calibration.size, size=total_draws)]
    offsets = np.empty(p.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(n_draw, out=offsets[1:])

    cumulative = np.empty(total_draws + 1, dtype=float)
    cumulative[0] = 0.0
    np.cumsum(draws, out=cumulative[1:])
    segment_sum = cumulative[offsets[1:]] - cumulative[offsets[:-1]]

    last = np.zeros_like(p)
    nonempty = n_draw > 0
    last[nonempty] = draws[offsets[1:][nonempty] - 1]
    correction = np.zeros_like(p)
    correction[has_fraction] = (1.0 - frac[has_fraction]) * last[has_fraction]
    return segment_sum - correction


def _simulate_replicate_fluorescence(
    log_params,
    environment_integral,
    n_cells,
    cell_calibration,
    particle_calibration,
    rng,
):
    log_params = np.asarray(log_params, dtype=float)
    integral = np.asarray(environment_integral, dtype=float)
    cell_calibration = np.asarray(cell_calibration, dtype=float).ravel()
    particle_calibration = np.asarray(particle_calibration, dtype=float).ravel()
    if log_params.shape != (4,):
        raise ValueError("log_params must have shape (4,)")
    if integral.ndim != 1 or integral.size == 0 or np.any(integral < 0.0):
        raise ValueError("environment_integral must be a non-empty non-negative vector")
    if int(n_cells) < 1:
        raise ValueError("n_cells must be positive")
    if cell_calibration.size == 0 or particle_calibration.size == 0:
        raise ValueError("calibration arrays must be non-empty")

    mr, sr, mk, sk = log_params
    if sr < 0.0 or sk < 0.0:
        raise ValueError("log-normal scales must be non-negative")

    coverage = 1.0
    surface_area = 5.31e-5
    result = np.empty((integral.size, int(n_cells)), dtype=float)
    for it, integrated_environment in enumerate(integral):
        r = rng.lognormal(mean=mr, sigma=sr, size=int(n_cells))
        k = rng.lognormal(mean=mk, sigma=sk, size=int(n_cells))
        dcell = cell_calibration[
            rng.integers(0, cell_calibration.size, size=int(n_cells))
        ]
        particle_count = k * (
            1.0 - np.exp(-coverage * surface_area * r * integrated_environment / k)
        )
        result[it] = dcell + _sample_particle_fluorescence_vectorised(
            particle_count, particle_calibration, rng
        )
    return result


# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import ndtr, ndtri

def _oracle_simulate_hierarchical_fluorescence(
    log_params: np.ndarray,
    environment_integral: np.ndarray,
    n_cells: int,
    cell_calibration: np.ndarray,
    particle_calibration: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.asarray(log_params, dtype=float).shape != (4,):
        raise ValueError("log_params must have shape (4,)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return _simulate_replicate_fluorescence(
        log_params,
        environment_integral,
        n_cells,
        cell_calibration,
        particle_calibration,
        rng,
    ).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic tests for one replicate's fluorescence simulator."""
    return [
        {
            "setup": """import numpy as np
from scipy.special import ndtri
log_params = np.array([-14.035551324356856, 0.4789159649348846, 2.282974736417405, 0.19804220043536505])
environment_integral = np.array([2.8e11, 5.1e11, 8.7e11])
q = (np.arange(64) + 0.5) / 64.0
z = ndtri(q)
cell_calibration = np.exp(5.3 + 0.35 * z)
particle_calibration = np.exp(3.0 + 0.25 * z)
rng_call = np.random.default_rng(1701)
rng_gold = np.random.default_rng(1701)
""",
            "call": "simulate_hierarchical_fluorescence(log_params, environment_integral, 20, cell_calibration, particle_calibration, rng_call)",
            "gold_call": "_oracle_simulate_hierarchical_fluorescence(log_params, environment_integral, 20, cell_calibration, particle_calibration, rng_gold)",
        },
        {
            "setup": """import numpy as np
log_params = np.array([np.log(4.0e-7), 0.0, np.log(8.0), 0.0])
environment_integral = np.array([1.5e11, 3.0e11])
cell_calibration = np.array([100.0, 120.0, 140.0])
particle_calibration = np.array([10.0, 20.0, 30.0])
rng_call = np.random.default_rng(9)
rng_gold = np.random.default_rng(9)
""",
            "call": "simulate_hierarchical_fluorescence(log_params, environment_integral, 8, cell_calibration, particle_calibration, rng_call)",
            "gold_call": "_oracle_simulate_hierarchical_fluorescence(log_params, environment_integral, 8, cell_calibration, particle_calibration, rng_gold)",
        },
        {
            "setup": """import numpy as np
log_params = np.array([np.log(7.0e-7), 0.2, np.log(15.0), 0.15])
environment_integral = np.array([0.0, 4.0e12, 9.0e12])
cell_calibration = np.linspace(80.0, 180.0, 11)
particle_calibration = np.linspace(5.0, 35.0, 13)
rng_call = np.random.default_rng(2026)
rng_gold = np.random.default_rng(2026)
""",
            "call": "simulate_hierarchical_fluorescence(log_params, environment_integral, 11, cell_calibration, particle_calibration, rng_call)",
            "gold_call": "_oracle_simulate_hierarchical_fluorescence(log_params, environment_integral, 11, cell_calibration, particle_calibration, rng_gold)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
log_params = np.array([np.log(3.0e-7), 0.2, np.log(9.0), 0.1])
environment_integral = np.array([1.0e11])
cell_calibration = np.array([])
particle_calibration = np.array([10.0, 20.0])
rng_call = np.random.default_rng(12)
rng_gold = np.random.default_rng(12)
""",
            "call": "value_error_code(lambda: simulate_hierarchical_fluorescence(log_params, environment_integral, 4, cell_calibration, particle_calibration, rng_call))",
            "gold_call": "value_error_code(lambda: _oracle_simulate_hierarchical_fluorescence(log_params, environment_integral, 4, cell_calibration, particle_calibration, rng_gold))",
        },
    ]
