"""
Compute the minimum total training hours, summed over every worker, skill and shift, that keeps every worker certified in every skill from the mandate start through the last shift, under per-shift skill decay, additive training gains, level bounds and both capacity limits.

Conventions the tests depend on: worker w's initial level in skill k is health_matrix[w, k] clipped to [0, 1] (rows are workers, columns are skill rungs, no relabeling); within each shift the level in skill k is first multiplied by 1 - decay_rates[k] and then raised by train_rate per hour trained on that skill in that shift; a worker is certified in a skill at the end of a shift when that post-update level is at least threshold, required for every shift from mandate_start (1-indexed) through n_shifts inclusive; levels never exceed 1, imposed as a constraint on every shift's post-update level; per shift, no worker trains more than worker_budget hours in total and no skill receives more than skill_budget hours in total; hours are non-negative reals; the returned value is the exact optimum as a native float.

Returns
-------
float: the minimum total training hours as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_training_schedule(
    health_matrix: np.ndarray,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    """Minimum total training hours meeting the certification mandate.

    Parameters
    ----------
    health_matrix : np.ndarray
        Array of shape (n_workers, n_skills); entry [w, k] clipped to [0, 1]
        is worker w's level in skill k entering shift 1.
    decay_rates : list
        Length-n_skills list of per-skill per-shift decay rates in [0, 1).
    threshold : float
        Minimum level required for certification.
    train_rate : float
        Level gained per hour of training (positive).
    n_shifts : int
        Number of shifts (shifts are numbered 1..n_shifts).
    mandate_start : int
        First shift (1-indexed) at whose end certification is required; the
        requirement holds through shift n_shifts inclusive.
    worker_budget : float
        Maximum total training hours for one worker in one shift.
    skill_budget : float
        Maximum total instructor-hours for one skill in one shift.

    Returns
    -------
    total_hours : float
        The minimum total training hours over all workers, skills and shifts.

    Raises
    ------
    ValueError
        If health_matrix is not 2-D, decay_rates has the wrong length or an
        entry outside [0, 1), train_rate is not positive, n_shifts is not
        positive, mandate_start is outside [1, n_shifts], a budget is negative,
        or the mandate cannot be met under the limits (infeasible).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog


def _schedule_constraints(S0, decay_rates, threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget):
    n_workers, n_skills = S0.shape
    n_vars = n_workers * n_skills * n_shifts

    def ix(w, k, t):
        return (w * n_skills + k) * n_shifts + t

    rows_A = []
    rows_b = []
    for w in range(n_workers):
        for k in range(n_skills):
            d = decay_rates[k]
            retention = [(1 - d) ** e for e in range(n_shifts + 1)]
            for t in range(mandate_start, n_shifts + 1):
                row = np.zeros(n_vars)
                for s in range(t):
                    row[ix(w, k, s)] = -train_rate * retention[t - 1 - s]
                rows_A.append(row)
                rows_b.append(retention[t] * S0[w, k] - threshold)
            for t in range(1, n_shifts + 1):
                row = np.zeros(n_vars)
                for s in range(t):
                    row[ix(w, k, s)] = train_rate * retention[t - 1 - s]
                rows_A.append(row)
                rows_b.append(1.0 - retention[t] * S0[w, k])
    for w in range(n_workers):
        for t in range(n_shifts):
            row = np.zeros(n_vars)
            for k in range(n_skills):
                row[ix(w, k, t)] = 1.0
            rows_A.append(row)
            rows_b.append(worker_budget)
    for k in range(n_skills):
        for t in range(n_shifts):
            row = np.zeros(n_vars)
            for w in range(n_workers):
                row[ix(w, k, t)] = 1.0
            rows_A.append(row)
            rows_b.append(skill_budget)
    return np.array(rows_A), np.array(rows_b), n_vars


def _oracle_solve_training_schedule(
    health_matrix: np.ndarray,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    S0 = np.asarray(health_matrix, dtype=float)
    if S0.ndim != 2:
        raise ValueError("health_matrix must be a 2D array")
    n_workers, n_skills = S0.shape
    if len(decay_rates) != n_skills:
        raise ValueError("decay_rates length must match the number of skills")
    if any(d < 0 or d >= 1 for d in decay_rates):
        raise ValueError("decay_rates entries must lie in [0, 1)")
    if train_rate <= 0:
        raise ValueError("train_rate must be positive")
    if n_shifts <= 0:
        raise ValueError("n_shifts must be positive")
    if not (1 <= mandate_start <= n_shifts):
        raise ValueError("mandate_start must be in [1, n_shifts]")
    if worker_budget < 0 or skill_budget < 0:
        raise ValueError("worker_budget and skill_budget must be non-negative")
    S0 = np.clip(S0, 0.0, 1.0)
    A, b, n_vars = _schedule_constraints(
        S0, list(decay_rates), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget
    )
    result = linprog(np.ones(n_vars), A_ub=A, b_ub=b, bounds=[(0, None)] * n_vars, method="highs")
    if not result.success:
        raise ValueError(f"the mandate cannot be met under the limits: {result.message}")
    return float(result.fun)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
health_matrix = np.array([[0.55, 0.40], [0.70, 0.65]])
decay_rates = [0.05, 0.03]
threshold = 0.60
train_rate = 0.10
n_shifts = 4
mandate_start = 2
worker_budget = 2.0
skill_budget = 1.5
""",
            "call": "solve_training_schedule(health_matrix.copy(), decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
            "gold_call": "_oracle_solve_training_schedule(health_matrix.copy(), decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
        },
        {
            "setup": """
import numpy as np
health_matrix = np.array([[0.95]])
decay_rates = [0.02]
threshold = 0.60
train_rate = 0.05
n_shifts = 3
mandate_start = 1
worker_budget = 3.0
skill_budget = 3.0
""",
            "call": "solve_training_schedule(health_matrix.copy(), decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
            "gold_call": "_oracle_solve_training_schedule(health_matrix.copy(), decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
        },
        {
            "setup": """
import numpy as np
health_matrix = np.array([[0.494168,0.661784,0.530122,0.634609],
               [0.403877,0.543089,0.435170,0.535810],
               [0.604703,0.808310,0.647410,0.764968],
               [0.658899,0.882348,0.706803,0.845833],
               [0.477356,0.641974,0.514410,0.633898],
               [0.489775,0.655254,0.524854,0.623974]])
decay_rates = [0.012, 0.030, 0.020, 0.008]
threshold = 0.60
train_rate = 0.05
n_shifts = 24
mandate_start = 8
worker_budget = 3.0
skill_budget = 3.5
""",
            "call": "solve_training_schedule(health_matrix.copy(), decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
            "gold_call": "_oracle_solve_training_schedule(health_matrix.copy(), decay_rates.copy(), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget)",
        },
    ]
