"""
Find the robust randomized hardware policy.

This constructed design problem maximizes min_r quality[r] dot x, with x>=0, sum(x)=1, costs dot x<=budget and tails[s] dot x>=tail_min for every scenario. Use the lexicographically smallest x among exact maximizers.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def optimize_policy(quality: np.ndarray, tails: np.ndarray, costs: np.ndarray, budget: float, tail_min: float) -> np.ndarray:
    """Find the robust randomized hardware policy.

    quality : ndarray, shape (R,C)
        Finite advantage rows, scenario-major and importance-minor when applicable.
    tails : ndarray, shape (S,C)
        High-fidelity probabilities in [0,1] for each scenario and profile.
    costs : ndarray, shape (C,)
        Nonnegative profile costs. R,S,C are positive.
    budget : float
        Finite nonnegative expected-cost ceiling.
    tail_min : float
        Minimum required tail probability in [0,1].
    Returns
    -------
    ndarray, shape (C+3,), float
        [optimal signed advantage, x[0], ..., x[C-1], minimum tail probability,
        expected cost]. An infeasible policy problem raises ValueError.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def _oracle_optimize_policy(quality: np.ndarray, tails: np.ndarray, costs: np.ndarray, budget: float, tail_min: float) -> np.ndarray:
    quality = np.asarray(quality, dtype=float)
    tails = np.asarray(tails, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if quality.ndim != 2 or tails.ndim != 2 or costs.ndim != 1 or (quality.shape[1] != len(costs)) or (tails.shape[1] != len(costs)) or (not len(costs)) or (not len(quality)) or (not len(tails)):
        raise ValueError('Incompatible nonempty quality, tail, and cost arrays.')
    if not all((np.isfinite(v).all() for v in [quality, tails, costs, np.array([budget, tail_min])])) or np.any((tails < 0) | (tails > 1)) or np.any(costs < 0) or (budget < 0) or (not 0 <= tail_min <= 1):
        raise ValueError('Invalid finite budget or probability data.')
    n = len(costs)
    objective = np.r_[np.zeros(n), -1.0]
    A = np.vstack([np.c_[-quality, np.ones(len(quality))], np.c_[-tails, np.zeros(len(tails))], np.r_[costs, 0.0]])
    b = np.r_[np.zeros(len(quality)), -np.full(len(tails), tail_min), budget]
    equality = np.array([np.r_[np.ones(n), 0.0]])
    eq_values = np.array([1.0])
    bounds = [(0, 1)] * n + [(None, None)]
    opts = {'dual_feasibility_tolerance': 1e-09, 'primal_feasibility_tolerance': 1e-09}
    sol = linprog(objective, A_ub=A, b_ub=b, A_eq=equality, b_eq=eq_values, bounds=bounds, method='highs', options=opts)
    if not sol.success:
        raise ValueError('No feasible policy.')
    equality = np.vstack([equality, np.r_[np.zeros(n), 1.0]])
    eq_values = np.r_[eq_values, sol.x[-1]]
    for k in range(n - 1):
        obj = np.zeros(n + 1)
        obj[k] = 1
        trial = linprog(obj, A_ub=A, b_ub=b, A_eq=equality, b_eq=eq_values, bounds=bounds, method='highs', options=opts)
        if not trial.success:
            raise ValueError('Optimal-face refinement failed.')
        sol = trial
        equality = np.vstack([equality, obj])
        eq_values = np.r_[eq_values, sol.x[k]]
    x = np.maximum(sol.x[:n], 0)
    return np.r_[np.min(quality @ x), x, np.min(tails @ x), costs @ x]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'quality=np.array([[0.2, 0.8]], dtype=float)\n'
               'tails=np.array([[0.5, 0.5]], dtype=float)\n'
               'costs=np.array([0, 1], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0.4, 0)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0.4, 0)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[0.8, 0.1], [0.2, 0.9]], dtype=float)\n'
               'tails=np.array([[0.7, 0.7], [0.8, 0.8]], dtype=float)\n'
               'costs=np.array([0, 0], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0, 0.6)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0, 0.6)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[0.8, 0.2]], dtype=float)\n'
               'tails=np.array([[0.1, 0.9]], dtype=float)\n'
               'costs=np.array([0, 0], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0, 0.6)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0, 0.6)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[-0.3, -0.1], [-0.15, -0.4]], dtype=float)\n'
               'tails=np.array([[0.8, 0.8]], dtype=float)\n'
               'costs=np.array([0, 0], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0, 0)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0, 0)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[0.4, 0.4, 0.4]], dtype=float)\n'
               'tails=np.array([[0.6, 0.6, 0.6]], dtype=float)\n'
               'costs=np.array([0, 0, 0], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0, 0.5)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0, 0.5)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[0.1, 0.3]], dtype=float)\n'
               'tails=np.array([[0.7, 0.2], [0.2, 0.7]], dtype=float)\n'
               'costs=np.array([0, 0], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0, 0.4)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0, 0.4)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[0.6, 0.4, 0.2], [0.1, 0.5, 0.8]], dtype=float)\n'
               'tails=np.array([[0.2, 0.6, 0.9]], dtype=float)\n'
               'costs=np.array([0.0, 0.5, 1.0], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0.55, 0.6)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0.55, 0.6)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'quality=np.array([[0.27], [0.32]], dtype=float)\n'
               'tails=np.array([[0.61], [0.84]], dtype=float)\n'
               'costs=np.array([0.2], dtype=float)',
      'call': 'optimize_policy(quality, tails, costs, 0.2, 0.61)',
      'gold_call': '_oracle_optimize_policy(quality, tails, costs, 0.2, 0.61)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'def _exception_code(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0',
      'call': '_exception_code(lambda: optimize_policy(np.array([[.2,.3]]), np.array([[.1,.1]]), '
              'np.array([0.,0.]), 0., .5))',
      'gold_call': '_exception_code(lambda: _oracle_optimize_policy(np.array([[.2,.3]]), '
                   'np.array([[.1,.1]]), np.array([0.,0.]), 0., .5))',
      'tol': 0.0}]
