"""
Advance the value function at one joint factor node back over a single time step. Blend the child value functions with the death benefit if the step closes a policy year, then form the continuation value by mapping each branch's fund multiplier onto the child value function it leads to, weighting by branch probability and discount factor, and subtracting any premium payable at the date. The fund contribution due at the date is credited before the branch multiplier is applied. If the date admits surrender, apply the obstacle pointwise at the fund level before that contribution. Child values are evaluated by linear interpolation on the supplied fund grid, extended beyond either end by the slope of the outermost interval.

The contract value at a joint factor node is a function of the accumulated fund. Advancing it back over one step composes four operations, and their order is contractual rather than arbitrary.

Mortality is independent of the financial factors, so it enters as a deterministic weight rather than a state. When the step closes a policy year, the value carried back from the child nodes is the survival-weighted mixture of the continuing contract value and the death benefit, the latter being the greater of the fund and the death guarantee level at the anniversary reached. On steps interior to a policy year no such blending occurs.

The continuation value at fund level F is the discounted expectation over the branch set. Each branch carries a probability, a discount factor, and a multiplier applied to the fund. The contribution due at the date is credited before the fund evolves, so the argument passed to the child value function is the multiplier applied to the post-contribution fund. Any premium payable at the date is an outflow and enters as a pure additive shift of the continuation value, independent of F.

Where surrender is admitted, the value is the greater of continuing and exercising. The surrender payoff compares a fraction of the fund with the surrender guarantee level, and the decision is taken before the premium and contribution of that anniversary, so the obstacle is applied at the fund level as it stands on arrival rather than after crediting. Because the obstacle is a pointwise maximum against a quantity that does not depend on the value function, it is non-expansive and does not disturb the stability of the recursion; but it does destroy affinity in the premium, which is why the fair premium for a surrenderable contract has no closed form.

The child value function must be evaluated at fund levels that do not coincide with the stored ones, since the branch multipliers move the argument off any fixed set. Beyond the outermost stored levels the value function of these contracts is affine in the fund, the payoff having no further kinks there, so extending by the slope of the outermost interval introduces no error of principle at the upper end.

Returns
-------
np.ndarray of shape (n,), the contract value at this factor node evaluated on fund_grid, as native floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def backward_step_with_obstacle(fund_grid: np.ndarray, child_values: np.ndarray,
                                child_index: np.ndarray, branches: np.ndarray,
                                contribution: float, premium: float, death_prob: float,
                                g_death: float, g_surrender: float, alpha_s: float,
                                apply_obstacle: bool) -> np.ndarray:
    '''Advance the value function at one factor node back over one time step.

    Parameters
    ----------
    fund_grid : np.ndarray
        Shape (n,), strictly increasing fund levels on which values are stored.
    child_values : np.ndarray
        Shape (m, n), the value function at each child factor node.
    child_index : np.ndarray
        Shape (k,), integer row of child_values reached by each branch.
    branches : np.ndarray
        Shape (3, k): branch probabilities, discount factors and fund
        multipliers.
    contribution : float
        Amount credited to the fund at this date, zero if none is due.
    premium : float
        Premium payable at this date, zero if none is due.
    death_prob : float
        Probability of death during the policy year closed by this step, in
        [0, 1]; zero on steps interior to a policy year.
    g_death : float
        Death guarantee level at the anniversary reached by this step.
    g_surrender : float
        Surrender guarantee level applying at this date.
    alpha_s : float
        Fraction of the fund returned by the fund-based surrender component,
        in (0, 1].
    apply_obstacle : bool
        Whether surrender is admitted at this date.

    Returns
    -------
    values : np.ndarray
        Shape (n,), the value function at this node on fund_grid.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid node, branch set and contract
        date.
    '''
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _interp(grid, vals, F):
    F = np.asarray(F, dtype=float)
    out = np.interp(F, grid, vals)
    lo = F < grid[0]
    if lo.any():
        s = (vals[1] - vals[0]) / (grid[1] - grid[0])
        out = np.where(lo, vals[0] + s * (F - grid[0]), out)
    hi = F > grid[-1]
    if hi.any():
        s = (vals[-1] - vals[-2]) / (grid[-1] - grid[-2])
        out = np.where(hi, vals[-1] + s * (F - grid[-1]), out)
    return out


def _oracle_backward_step_with_obstacle(fund_grid: np.ndarray, child_values: np.ndarray,
                                        child_index: np.ndarray, branches: np.ndarray,
                                        contribution: float, premium: float, death_prob: float,
                                        g_death: float, g_surrender: float, alpha_s: float,
                                        apply_obstacle: bool) -> np.ndarray:
    F = np.asarray(fund_grid, dtype=float)
    U = np.asarray(child_values, dtype=float)
    idx = np.asarray(child_index)
    B = np.asarray(branches, dtype=float)

    if F.ndim != 1 or F.size < 2 or not np.all(np.isfinite(F)) or np.any(np.diff(F) <= 0.0):
        raise ValueError("fund_grid must be a strictly increasing finite 1-D array")
    if U.ndim != 2 or U.shape[1] != F.size or not np.all(np.isfinite(U)):
        raise ValueError("child_values must be a finite 2-D array with one column per fund node")
    if B.ndim != 2 or B.shape[0] != 3 or not np.all(np.isfinite(B)):
        raise ValueError("branches must be a finite array with three rows")
    if idx.ndim != 1 or idx.size != B.shape[1] or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("child_index must be an integer array with one entry per branch")
    if idx.min() < 0 or idx.max() >= U.shape[0]:
        raise ValueError("child_index refers to a value function that was not supplied")
    for name, val in (("contribution", contribution), ("premium", premium),
                      ("death_prob", death_prob), ("g_death", g_death),
                      ("g_surrender", g_surrender), ("alpha_s", alpha_s)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
    if not 0.0 <= float(death_prob) <= 1.0:
        raise ValueError("death_prob must lie in [0, 1]")
    if not 0.0 < float(alpha_s) <= 1.0:
        raise ValueError("alpha_s must lie in (0, 1]")
    if B.shape[1] < 1 or B[0].min() < 0.0:
        raise ValueError("branch probabilities must be non-negative")

    D, P = float(contribution), float(premium)
    qd, gd = float(death_prob), float(g_death)
    gs, a_s = float(g_surrender), float(alpha_s)

    if qd > 0.0:
        benefit = np.maximum(F, gd)
        Um = (1.0 - qd) * U + qd * benefit[None, :]
    else:
        Um = U

    cont = np.zeros_like(F)
    for b in range(B.shape[1]):
        cont += B[0, b] * B[1, b] * _interp(F, Um[int(idx[b])], (F + D) * B[2, b])
    cont -= P

    if apply_obstacle:
        cont = np.maximum(cont, np.maximum(a_s * F, gs))
    return cont

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = """import numpy as np
F = np.linspace(1.0, 1200.0, 601)
U = np.vstack([np.maximum(F, 312.2848660765) + 3.0 * k for k in range(4)])
pr = np.array([0.05, 0.12, 0.18, 0.22, 0.16, 0.11, 0.08, 0.05, 0.03])
disc = np.full(9, 0.99512)
mult = np.array([0.86, 0.91, 0.95, 0.99, 1.02, 1.06, 1.11, 1.17, 1.24])
B = np.vstack([pr, disc, mult])
ci = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])
"""
    err = pre + """
def run_model(**kw):
    a = dict(fund_grid=F, child_values=U, child_index=ci, branches=B, contribution=100.0,
             premium=107.5, death_prob=0.0016725, g_death=206.1012114219,
             g_surrender=306.0706041064, alpha_s=0.95, apply_obstacle=True)
    a.update(kw)
    try:
        backward_step_with_obstacle(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(fund_grid=F, child_values=U, child_index=ci, branches=B, contribution=100.0,
             premium=107.5, death_prob=0.0016725, g_death=206.1012114219,
             g_surrender=306.0706041064, alpha_s=0.95, apply_obstacle=True)
    a.update(kw)
    try:
        _oracle_backward_step_with_obstacle(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    A = "F, U, ci, B, 100.0, 107.5, 0.0016725, 206.1012114219, 306.0706041064, 0.95"
    return [
        {"setup": pre, "call": f"backward_step_with_obstacle({A}, True)",
         "gold_call": f"_oracle_backward_step_with_obstacle({A}, True)"},
        {"setup": pre, "call": f"backward_step_with_obstacle({A}, False)",
         "gold_call": f"_oracle_backward_step_with_obstacle({A}, False)"},
        {"setup": pre,
         "call": "backward_step_with_obstacle(F, U, ci, B, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, False)",
         "gold_call": "_oracle_backward_step_with_obstacle(F, U, ci, B, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, False)"},
        {"setup": pre,
         "call": "backward_step_with_obstacle(F, U, ci, B, 100.0, 107.5, 0.0016725, 206.1012114219, 5000.0, 0.95, True)",
         "gold_call": "_oracle_backward_step_with_obstacle(F, U, ci, B, 100.0, 107.5, 0.0016725, 206.1012114219, 5000.0, 0.95, True)"},
        {"setup": pre,
         "call": "backward_step_with_obstacle(F, U, ci, B, 100.0, 107.5, 0.0016725, 206.1012114219, 0.0, 1.0, True)",
         "gold_call": "_oracle_backward_step_with_obstacle(F, U, ci, B, 100.0, 107.5, 0.0016725, 206.1012114219, 0.0, 1.0, True)"},
        {"setup": pre,
         "call": "backward_step_with_obstacle(F, U, ci, B, 0.0, 0.0, 1.0, 206.1012114219, 0.0, 0.95, False)",
         "gold_call": "_oracle_backward_step_with_obstacle(F, U, ci, B, 0.0, 0.0, 1.0, 206.1012114219, 0.0, 0.95, False)"},
        {"setup": err, "call": "run_model(death_prob=1.5)", "gold_call": "run_gold(death_prob=1.5)"},
        {"setup": err, "call": "run_model(alpha_s=0.0)", "gold_call": "run_gold(alpha_s=0.0)"},
        {"setup": err, "call": "run_model(child_index=np.array([0, 1, 2, 3, 0, 1, 2, 3, 9]))",
         "gold_call": "run_gold(child_index=np.array([0, 1, 2, 3, 0, 1, 2, 3, 9]))"},
    ]
