"""
Orchestrator: run the full three-stage pipeline end to end and report the minimum total training hours.

Complete the sparse, separable exposure landscape from its observed cells, jointly recover the coupled health matrix from the completed landscape and the observed health cells, seed the initial worker-skill levels from that health matrix rank for rank, and compute the minimum total training hours that satisfy the certification mandate under the skill dynamics and both capacity limits. Conventions the tests depend on: the returned scalar is a native float; the seeding uses row w as worker w and column k as skill rung k with no relabeling.

Returns
-------
float: the minimum total training hours across all workers, skills and shifts, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_minimum_training_hours(
    landscape_obs: dict,
    landscape_dims: tuple,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters_palm: int,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    """Run the full pipeline and report the minimum total training hours.

    Parameters
    ----------
    landscape_obs : dict
        Sparse, exact observations of the multiplicatively separable landscape.
    landscape_dims : tuple
        Shape of the full landscape.
    health_obs : dict
        Sparse, noisy observations of the health matrix.
    health_dims : tuple
        (n_workers, n_skills) shape of the health matrix.
    lam_landscape, lam_health, lam_coupling, delta : float
        Weights of the coupled-recovery objective.
    n_iters_palm : int
        Number of block sweeps for the coupled recovery.
    decay_rates : list
        Length-n_skills per-skill decay rates.
    threshold : float
        Minimum level required for certification.
    train_rate : float
        Level gained per hour of training.
    n_shifts : int
        Total number of shifts.
    mandate_start : int
        First shift (1-indexed) at which certification is required.
    worker_budget : float
        Maximum total training hours per worker per shift.
    skill_budget : float
        Maximum total instructor-hours per skill per shift.

    Returns
    -------
    total_hours : float
        The minimum feasible total training hours across every worker, skill
        and shift.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_minimum_training_hours(
    landscape_obs: dict,
    landscape_dims: tuple,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters_palm: int,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    landscape = _oracle_complete_separable_landscape(landscape_obs, landscape_dims)
    health_matrix = _oracle_recover_coupled_matrices(
        landscape, health_obs, health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm
    )
    return _oracle_solve_training_schedule(
        health_matrix, decay_rates, threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
landscape_obs = {(0,0,0):0.3,(0,0,1):0.33,(1,0,0):0.48,(0,1,0):0.45,(1,1,1):0.792}
landscape_dims = (2, 2, 2)
health_obs = {(0,0): 0.45, (1,1): 0.70}
health_dims = (2, 2)
lam_landscape = lam_health = lam_coupling = 0.15
delta = 0.08
n_iters_palm = 3000
decay_rates = [0.02, 0.03]
threshold = 0.60
train_rate = 0.10
n_shifts = 10
mandate_start = 4
worker_budget = 3.0
skill_budget = 3.0
""",
            "call": "compute_minimum_training_hours(landscape_obs.copy(), landscape_dims, health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm, decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
            "gold_call": "_oracle_compute_minimum_training_hours(landscape_obs.copy(), landscape_dims, health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm, decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
        },
        {
            "setup": """
landscape_obs = {(0,0,0):0.3,(0,0,1):0.33,(1,0,0):0.48,(0,1,0):0.45,(1,1,1):0.792}
landscape_dims = (2, 2, 2)
health_obs = {(0,0): 0.75, (0,1): 0.80, (1,0): 0.78, (1,1): 0.82}
health_dims = (2, 2)
lam_landscape = lam_health = lam_coupling = 0.15
delta = 0.08
n_iters_palm = 3000
decay_rates = [0.02, 0.03]
threshold = 0.60
train_rate = 0.10
n_shifts = 10
mandate_start = 4
worker_budget = 3.0
skill_budget = 3.0
""",
            "call": "compute_minimum_training_hours(landscape_obs.copy(), landscape_dims, health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm, decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
            "gold_call": "_oracle_compute_minimum_training_hours(landscape_obs.copy(), landscape_dims, health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm, decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
        },
        {
            "setup": """
landscape_obs = {(0,0,1):0.4864,(0,0,2):0.5472,(0,1,1):0.7296,(0,2,2):0.9576,
                  (1,0,1):0.3584,(1,1,1):0.5376,(2,0,2):0.7416,(2,1,0):1.3596,
                  (2,1,1):0.9888,(2,1,2):1.1124,(2,2,1):1.1536,(2,3,1):0.9064,
                  (3,0,0):0.8976,(3,1,1):0.9792,(3,2,2):1.2852,(4,1,1):0.6816,
                  (5,0,1):0.5120}
landscape_dims = (6, 4, 3)
health_obs = {(0,2):0.5971,(0,3):0.6371,(1,1):0.6107,(1,3):0.6874,(2,2):0.5769,
              (2,3):0.6285,(3,0):0.6851,(3,2):0.7550,(4,1):0.6332,(4,3):0.7840,
              (5,2):0.5173,(5,3):0.5639}
health_dims = (6, 4)
lam_landscape = lam_health = lam_coupling = 0.20
delta = 0.10
n_iters_palm = 5000
decay_rates = [0.012, 0.030, 0.020, 0.008]
threshold = 0.60
train_rate = 0.05
n_shifts = 24
mandate_start = 8
worker_budget = 3.0
skill_budget = 3.5
""",
            "call": "compute_minimum_training_hours(landscape_obs.copy(), landscape_dims, health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm, decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
            "gold_call": "_oracle_compute_minimum_training_hours(landscape_obs.copy(), landscape_dims, health_obs.copy(), health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm, decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
        },
    ]
