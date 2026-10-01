"""
Resolve the complete shared-parameter domain and return the smallest worst-right-hand-side left-to-right residual ratio under the storage budget.

The final step combines the exact adaptive partitions with the benchmark's feasibility condition and fixed-budget solver comparison. It evaluates one representative per distinct feasible pair of final supports; it does not replace the continuous search with a grid.




$$

Q=\min_{(\varepsilon,\delta)\le K}\max_j\frac{R_{L,j}}{R_{R,j}}.

$$

Returns
-------
float, the unrounded global minimax left/right true-residual ratio; only this step is the final orchestrator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_shared_budget(
    A: np.ndarray,
    rhs: np.ndarray,
    box: tuple,
    budget: int,
    c: int = 2,
    restart: int = 3,
    cycles: int = 2,
) -> float:
    """Return the global shared-budget minimax residual ratio.

    Parameters
    ----------
    A : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        exactly after binary64 conversion.
    rhs : np.ndarray
        Finite real (n, m) array, m >= 1, with no zero column.
    box : tuple
        Domain for (epsilon**2, delta**2), encoded as in column_regions:
        ((eta_low_pair, eta_high_pair, eta_low_closed, eta_high_closed),
         (xi_low_pair, xi_high_pair, xi_low_closed, xi_high_closed)).
        Rational pairs must be reduced with positive denominators; closure
        flags are integer 0/1; eta >= 0 and 0 < xi <= 1. Closed singletons
        are allowed. No rounding of thresholds is permitted.
    budget : int
        Nonnegative non-Boolean combined active-position budget.
    c : int
        Positive non-Boolean selection cap.
    restart, cycles : int
        Positive non-Boolean GMRES counts, with restart <= n.

    Returns
    -------
    minimum : float
        Unrounded native Python float. The same parameter pair determines
        both preconditioners. Each feasible configuration is scored by the
        maximum same-right-hand-side left/right true residual ratio; the
        smallest score is returned. Scientific answer tolerance is 1e-3.
        The final displayed three-decimal answer is a separate formatting step.

    Raises
    ------
    ValueError
        If any input violates the stated contracts, no feasible joint pattern
        exists, any feasible right relative residual is zero, or numerical
        evaluation produces a nonfinite residual ratio.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_shared_budget(
    A: np.ndarray,
    rhs: np.ndarray,
    box: tuple,
    budget: int,
    c: int = 2,
    restart: int = 3,
    cycles: int = 2,
) -> float:
    data = _nr_matrix(A)
    n = data['n']
    A = _nr_real_array(A, 'A', 2)
    rhs = _nr_real_array(rhs, 'rhs', 2)
    if rhs.shape[0] != n or any(not np.any(b) for b in rhs.T):
        raise ValueError("rhs must have n rows and no zero column")
    _nr_box(box)
    budget = _nr_integer(budget, 'budget', 0)
    c = _nr_integer(c, 'c', 1)
    restart = _nr_integer(restart, 'restart', 1, n)
    cycles = _nr_integer(cycles, 'cycles', 1)
    partitions = tuple(
        _oracle_column_regions(C, k, box, c)
        for C in (A, A.T) for k in range(n)
    )
    configurations = _oracle_joint_patterns(partitions, budget)
    if not configurations:
        raise ValueError("the storage budget has no feasible configuration")
    best = float('inf')
    for _region, supports, _cost in configurations:
        residuals = _oracle_evaluate_patterns(A, rhs, supports, restart, cycles)
        if np.any(residuals[1] == 0):
            raise ValueError("a right residual is zero, so a ratio is undefined")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            score = float(np.max(residuals[0] / residuals[1]))
        if not np.isfinite(score):
            raise ValueError("nonfinite residual ratio")
        best = min(best, score)
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: complete 48-by-48 benchmark
        {
            "setup": """import numpy as np
n = 48
i = np.arange(n)
B = np.zeros((n, n), dtype=float)
B[i, i] = 7/4 + (i % 5)/16
B[i, (i + 1) % n] = -5/4
B[i, (i - 1) % n] = -1/2
B[i, (i + 5) % n] = (7/8) * (-1.0)**i
dl = 2.0**((7*i) % 9 - 4)
dr = 2.0**((i % 3)//2) / dl
A = dl[:, None] * B * dr[None, :]
rhs = np.column_stack((
    1 + (-1.0)**i/4 + (i % 7)/32,
    (-1.0)**i * (1 + (i % 5)/16),
    (((3*i) % 11) - 5)/8 + (i % 2)/32,
))
box = (((1, 25), (3, 25), 1, 1), ((1, 50), (3, 25), 1, 1))

expected = 47.58731289680045

def _check_result(value):
    return int(type(value) is float and np.isfinite(value)
               and abs(value - expected) <= 0.001)
""",
            "call": '_check_result(solve_shared_budget(A, rhs, box, 980, 2, 3, 2))',
            "gold_call": '_check_result(_oracle_solve_shared_budget(A, rhs, box, 980, 2, 3, 2))',
        },
        # boundary: closed singleton domain
        {
            "setup": """import numpy as np
n = 8
i = np.arange(n)
B = 3.0 * np.eye(n)
B[i, (i + 1) % n] = -1.0
dl = 2.0**((3*i) % 7 - 3)
dr = 1.0 / dl
A = dl[:, None] * B * dr[None, :]
rhs = np.column_stack((1.0 + i/8, (-1.0)**i + i/16))
box = (((1, 4), (1, 4), 1, 1), ((1, 16), (1, 16), 1, 1))

expected = 6.786379659114657

def _check_result(value):
    return int(type(value) is float and np.isfinite(value)
               and abs(value - expected) <= 0.001)
""",
            "call": '_check_result(solve_shared_budget(A, rhs, box, 64, 2, 1, 1))',
            "gold_call": '_check_result(_oracle_solve_shared_budget(A, rhs, box, 64, 2, 1, 1))',
        },
        # edge: one residual row selected per expansion
        {
            "setup": """import numpy as np
n = 8
i = np.arange(n)
B = 3.0 * np.eye(n)
B[i, (i + 1) % n] = -1.0
dl = 2.0**((3*i) % 7 - 3)
dr = 1.0 / dl
A = dl[:, None] * B * dr[None, :]
rhs = np.column_stack((1.0 + i/8, (-1.0)**i + i/16))
box = (((1, 10000), (1, 4), 1, 1), ((1, 100), (1, 2), 1, 1))

expected = 0.3686681108908046

def _check_result(value):
    return int(type(value) is float and np.isfinite(value)
               and abs(value - expected) <= 0.001)
""",
            "call": '_check_result(solve_shared_budget(A, rhs, box, 70, 1, 1, 2))',
            "gold_call": '_check_result(_oracle_solve_shared_budget(A, rhs, box, 70, 1, 1, 2))',
        },
        # invalid: shared budget is infeasible
        {
            "setup": """import numpy as np
n = 8
i = np.arange(n)
B = 3.0 * np.eye(n)
B[i, (i + 1) % n] = -1.0
dl = 2.0**((3*i) % 7 - 3)
dr = 1.0 / dl
A = dl[:, None] * B * dr[None, :]
rhs = np.column_stack((1.0 + i/8, (-1.0)**i + i/16))
box = (((1, 4), (1, 4), 1, 1), ((1, 16), (1, 16), 1, 1))

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: solve_shared_budget(A, rhs, box, 0, 2, 1, 1))',
            "gold_call": '_capture_value_error(lambda: _oracle_solve_shared_budget(A, rhs, box, 0, 2, 1, 1))',
        },
        # invalid: exact zero right residual makes a quotient undefined
        {
            "setup": """import numpy as np
A=np.eye(2)
rhs=np.array([[1.0],[2.0]])
box=(((1,4),(1,4),1,1),((1,4),(1,4),1,1))

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: solve_shared_budget(A, rhs, box, 8, 2, 1, 1))',
            "gold_call": '_capture_value_error(lambda: _oracle_solve_shared_budget(A, rhs, box, 8, 2, 1, 1))',
        },
    ]
