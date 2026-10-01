"""
Select the regret-controlled management plan.

Compare the stored common-effort condition scores across plans. Average each plan's two largest condition regrets, apply the regret limit and penalty, and use the earliest-plan tie rule. Return [winner, winner decision score, winner robust score, winner tail regret, winner effort], followed by one row-major audit row [decision score, robust score, tail regret, effort] for every plan. Initialize an unavailable plan's complete audit row to -1e12. For an otherwise available plan whose tail regret exceeds the limit, set only its decision score to -1e12 and retain its robust score, tail regret, and effort.

Returns
-------
return a float64 array of shape (5 + 4 × plans): selected-plan summary followed by the row-major plan audit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_regret_controlled_plan(profile_table: "np.ndarray", regret_weight: float, regret_limit: float, tie_tolerance: float) -> "np.ndarray":
    """Apply cross-plan tail regret and return the selected summary followed by the complete plan audit.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _oracle_select_regret_controlled_plan(profile_table: "np.ndarray", regret_weight: float, regret_limit: float, tie_tolerance: float) -> "np.ndarray":
    table = np.asarray(profile_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 8:
        raise ValueError("profile_table must contain at least two plans and three conditions")
    if regret_weight < 0.0 or regret_limit < 0.0 or tie_tolerance < 0.0 or not np.isfinite([regret_weight, regret_limit, tie_tolerance]).all():
        raise ValueError("invalid regret-selection scalars")
    conditions = table.shape[1] - 5
    scenario = table[:, 1:1 + conditions]
    robust = table[:, 1 + conditions]
    available = np.isfinite(scenario).all(axis=1) & (robust > -1.0e11)
    if not np.any(available):
        raise ValueError("no plan has a usable common-effort profile")
    best_by_condition = np.max(scenario[available], axis=0)
    audit = np.full((table.shape[0], 4), -1.0e12, dtype=np.float64)
    for plan in np.where(available)[0]:
        regrets = best_by_condition - scenario[plan]
        tail = float(np.sort(regrets)[-2:].mean())
        decision = float(robust[plan] - regret_weight * tail) if tail <= regret_limit else -1.0e12
        audit[plan] = [decision, robust[plan], tail, table[plan, 0]]
    best = float(np.max(audit[:, 0]))
    eligible = np.where(audit[:, 0] >= best - tie_tolerance)[0]
    if eligible.size == 0 or best <= -1.0e11:
        raise ValueError("no plan satisfies the regret limit")
    winner = int(eligible[0])
    head = np.array([float(winner), audit[winner, 0], audit[winner, 1], audit[winner, 2], audit[winner, 3]], dtype=np.float64)
    return np.concatenate([head, audit.ravel()]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nt=np.array([[1.0,2.0,1.5,1.0,1.0,2.0,0.0,0.5],[1.2,1.7,1.8,1.3,1.3,2.0,0.0,0.5],[0.9,1.4,1.4,1.4,1.4,2.0,0.0,0.5]])", "call": "select_regret_controlled_plan(t,0.2,1.0,1e-9)", "gold_call": "_oracle_select_regret_controlled_plan(t,0.2,1.0,1e-9)", "tol": 1e-12},
        {"setup": "import numpy as np\nt=np.array([[1.0,2.0,2.0,2.0,2.0,0.0,0.0,0.4],[1.0,2.0,2.0,2.0,2.0,0.0,0.0,0.4]])", "call": "select_regret_controlled_plan(t,0.2,1.0,1e-9)", "gold_call": "_oracle_select_regret_controlled_plan(t,0.2,1.0,1e-9)", "tol": 1e-12},
        {"setup": "import numpy as np\nt=np.array([[1.0,-1e12,-1e12,-1e12,-1e12,-1.0,-1.0,-1.0],[1.1,1.0,1.1,1.2,1.0,2.0,0.0,0.5],[1.2,1.2,1.0,1.05,1.0,2.0,0.0,0.5]])", "call": "select_regret_controlled_plan(t,0.1,1.0,1e-9)", "gold_call": "_oracle_select_regret_controlled_plan(t,0.1,1.0,1e-9)", "tol": 1e-12},
        {"setup": "import numpy as np\nt=np.ones((1,8))\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: select_regret_controlled_plan(t,0.1,1.0,1e-9))", "gold_call": "caught(lambda: _oracle_select_regret_controlled_plan(t,0.1,1.0,1e-9))", "tol": 0.0},
    ]
