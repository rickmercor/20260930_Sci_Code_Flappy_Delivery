"""
Run the complete two-population hierarchical affinity ABC-SMC benchmark.

This final orchestrator starts from the top-level benchmark inputs, regenerates the observed data, initialises the hierarchical particle population, reruns the complete forward-model and discrepancy chain, and performs two adaptive resample-move updates.

Returns
-------
float, the second-replicate carrying-capacity mean after population two.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_hierarchical_affinity_benchmark(
    theta_true: np.ndarray,
    times: np.ndarray,
    n_cells: int,
    n_prep: int,
    fine_step: float,
    n_grid: int,
    n_particles: int,
    r_trial: int,
    c_smc: float,
    split_fraction: float,
    max_moves: int,
    data_seed: int,
    inference_seed: int,
) -> float:
    """Run two hierarchical ABC-SMC populations from the raw benchmark inputs.

    Parameters
    ----------
    theta_true : np.ndarray
        Generating vector ordered as replicate means followed by six hyperparameters.
    times : np.ndarray
        Observation times in seconds.
    n_cells : int
        Observed and simulated cells per replicate and time.
    n_prep : int
        Pilot cells per fine-grid time.
    fine_step : float
        Fine-grid spacing in seconds.
    n_grid : int
        Empirical-CDF grid size.
    n_particles : int
        Number of ABC-SMC particles.
    r_trial : int
        Trial ABC-MCMC moves per resampled particle.
    c_smc : float
        Target bound used to adapt the move count.
    split_fraction : float
        Fraction of particles resampled and moved.
    max_moves : int
        Maximum total ABC-MCMC moves per resampled particle.
    data_seed : int
        Random seed for generating the observed synthetic dataset.
    inference_seed : int
        Seed for one shared inference random-number stream.

    Notes
    -----
    Generate observations from the independent data stream. On the inference
    stream, initialise particles one at a time by drawing the prior and then its
    simulated data. Each simulation consumes replicate-major pilot and observation
    draws. Carry that stream through both updates, resampling rejected indices first
    and processing moved particles in ascending position order; draw each proposal
    before its support check and a Metropolis uniform only for a finite-prior
    proposal, then simulate only after the screen passes.

    Returns
    -------
    result : float
        Particle mean of the second replicate carrying-capacity mean after two
        sequential ABC-SMC updates.

    Raises
    ------
    ValueError
        If theta_true does not have dimension 2M+6 with at least two
        replicates, or n_particles is below 4.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_empirical_calibration():
    q = (np.arange(64, dtype=float) + 0.5) / 64.0
    z = ndtri(q)
    cell_calibration = np.exp(5.3 + 0.35 * z)
    particle_calibration = np.exp(3.0 + 0.25 * z)
    return cell_calibration, particle_calibration


# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ndtri

def _oracle_run_hierarchical_affinity_benchmark(
    theta_true: np.ndarray,
    times: np.ndarray,
    n_cells: int,
    n_prep: int,
    fine_step: float,
    n_grid: int,
    n_particles: int,
    r_trial: int,
    c_smc: float,
    split_fraction: float,
    max_moves: int,
    data_seed: int,
    inference_seed: int,
) -> float:
    theta_true = np.asarray(theta_true, dtype=float)
    times = np.asarray(times, dtype=float)
    if theta_true.ndim != 1 or (theta_true.size - 6) % 2 != 0:
        raise ValueError("theta_true must have dimension 2*M+6")
    m = (theta_true.size - 6) // 2
    if m < 2:
        raise ValueError("at least two replicates are required")
    if int(n_particles) < 4:
        raise ValueError("n_particles must be at least 4")

    cell_calibration, particle_calibration = _build_empirical_calibration()
    log_params_true = _oracle_parameterise_hierarchical_lognormals(
        theta_true[:m],
        theta_true[m : 2 * m],
        theta_true[2 * m],
        theta_true[2 * m + 1],
    )
    data_rng = np.random.default_rng(int(data_seed))
    observed = np.empty((times.size, m, int(n_cells)), dtype=float)
    for j in range(m):
        environment = _oracle_compute_shared_environment_integral(
            log_params_true[j], times, n_prep, fine_step, data_rng
        )
        observed[:, j] = _oracle_simulate_hierarchical_fluorescence(
            log_params_true[j],
            environment,
            n_cells,
            cell_calibration,
            particle_calibration,
            data_rng,
        )

    inference_rng = np.random.default_rng(int(inference_seed))
    dimension = theta_true.size
    theta = np.empty((dimension, int(n_particles)), dtype=float)
    rho = np.empty(int(n_particles), dtype=float)
    reference_particles = np.repeat(theta_true[:, None], 2, axis=1)
    for i in range(int(n_particles)):
        prior_package = _oracle_evaluate_hierarchical_prior_and_proposal(
            theta_true, reference_particles, inference_rng
        )
        theta[:, i] = prior_package[:dimension]
        log_params = _oracle_parameterise_hierarchical_lognormals(
            theta[:m, i],
            theta[m : 2 * m, i],
            theta[2 * m, i],
            theta[2 * m + 1, i],
        )
        simulated = np.empty_like(observed)
        for j in range(m):
            environment = _oracle_compute_shared_environment_integral(
                log_params[j], times, n_prep, fine_step, inference_rng
            )
            simulated[:, j] = _oracle_simulate_hierarchical_fluorescence(
                log_params[j],
                environment,
                n_cells,
                cell_calibration,
                particle_calibration,
                inference_rng,
            )
        rho[i] = _oracle_compute_snapshot_discrepancy(observed, simulated, n_grid)

    for _ in range(2):
        packed = _oracle_advance_abc_smc_population(
            theta,
            rho,
            observed,
            times,
            n_cells,
            n_prep,
            fine_step,
            cell_calibration,
            particle_calibration,
            n_grid,
            r_trial,
            c_smc,
            split_fraction,
            max_moves,
            inference_rng,
        )
        theta = packed[4 : 4 + dimension * int(n_particles)].reshape(
            dimension, int(n_particles)
        )
        rho = packed[4 + dimension * int(n_particles) :]

    return float(np.mean(theta[m + 1]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return whole-pipeline tests for the final orchestrator."""
    return [
        {
            "setup": """import numpy as np
theta_true = np.array([9.0e-7, 3.86125e-7, 3.86125e-7,
                       10.0, 20.0, 10.0,
                       4.56962e-7, 2.0, 3.86125e-7, 10.0, 0.0, 0.0])
times = 3600.0 * np.array([1.0, 2.0, 4.0])
""",
            "call": "run_hierarchical_affinity_benchmark(theta_true, times, 60, 30, 1800.0, 1000, 120, 20, 0.01, 0.5, 5000, 1701, 424242)",
            "gold_call": "_oracle_run_hierarchical_affinity_benchmark(theta_true, times, 60, 30, 1800.0, 1000, 120, 20, 0.01, 0.5, 5000, 1701, 424242)",
        },
        {
            "setup": """import numpy as np
theta_true = np.array([8.0e-7, 4.0e-7, 4.0e-7,
                       9.0, 16.0, 11.0,
                       3.5e-7, 1.5, 4.0e-7, 10.0, 0.0, 0.0])
times = np.array([1800.0, 3600.0])
""",
            "call": "run_hierarchical_affinity_benchmark(theta_true, times, 8, 5, 900.0, 101, 18, 2, 0.5, 0.5, 6, 41, 42)",
            "gold_call": "_oracle_run_hierarchical_affinity_benchmark(theta_true, times, 8, 5, 900.0, 101, 18, 2, 0.5, 0.5, 6, 41, 42)",
        },
        {
            "setup": """import numpy as np
theta_true = np.array([5.5e-7, 7.5e-7, 3.5e-7,
                       7.0, 13.0, 9.0,
                       2.5e-7, 1.0, 4.0e-7, 9.0, 0.0, 0.0])
times = np.array([900.0, 1800.0, 2700.0])
""",
            "call": "run_hierarchical_affinity_benchmark(theta_true, times, 7, 4, 900.0, 81, 16, 2, 0.4, 0.5, 5, 99, 100)",
            "gold_call": "_oracle_run_hierarchical_affinity_benchmark(theta_true, times, 7, 4, 900.0, 81, 16, 2, 0.4, 0.5, 5, 99, 100)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
theta_true = np.array([5.0e-7, 6.0e-7, 8.0, 9.0, 2.0e-7, 1.0, 5.0e-7, 8.0, 0.0, 0.0])
times = np.array([900.0])
""",
            "call": "value_error_code(lambda: run_hierarchical_affinity_benchmark(theta_true, times, 4, 2, 900.0, 21, 3, 1, 0.5, 0.5, 2, 3, 4))",
            "gold_call": "value_error_code(lambda: _oracle_run_hierarchical_affinity_benchmark(theta_true, times, 4, 2, 900.0, 21, 3, 1, 0.5, 0.5, 2, 3, 4))",
        },
    ]
