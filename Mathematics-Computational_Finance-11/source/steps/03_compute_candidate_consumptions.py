"""
From a value function tabulated on the wealth grid, produce the two candidate consumption policies that a monotone upwind treatment of the recursive-utility Hamiltonian requires at every node.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with one node per row of the supplied value function and node zero at the borrowing limit. The value function has two columns, the two income states in the order low income then high income, and is strictly negative everywhere.

Return a single 2-D array with twice as many rows as the value function and the same two columns, obtained by stacking the two candidate policies vertically: the forward candidate in the first block of rows and the backward candidate in the second.

The returned pair must satisfy the following contract at every node, where the zero-saving consumption at a node is the interest income plus the income of that state:

  the saving rate implied by the forward candidate is greater than or equal to zero

  the saving rate implied by the backward candidate is less than or equal to zero

  at the first node the backward candidate equals the zero-saving consumption

  at the last node the forward candidate equals the zero-saving consumption

  wherever the relevant one-sided difference of the value function is not strictly positive, the forward candidate takes the zero-saving consumption and the backward candidate takes the supplied cap

The cap is a regularising upper bound on the backward candidate only. It must exceed the largest zero-saving consumption on the grid.

Inputs are the value function, the risk aversion, the elasticity of intertemporal substitution, the discount rate, the interest rate, the two income levels, the borrowing limit, the upper truncation point and the cap.

Raises ValueError if the value function is not a two-dimensional finite array with exactly two columns and at least two rows, if any of its entries is not strictly negative, if any real input is not finite, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, if the discount rate or the interest rate is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, if the zero-saving consumption is not strictly positive at every node, or if the cap does not exceed the largest zero-saving consumption on the grid.

A monotone scheme for a first-order Hamilton-Jacobi-Bellman equation must differentiate the value function in the direction the state is actually moving. Since the direction of motion is itself the object being solved for, the standard device is to form a candidate policy from each one-sided difference and to admit each candidate only on the side where it is consistent with its own drift: the forward difference is used where the state is moving up, the backward difference where it is moving down. The two candidates are constructed so that the forward one can only produce non-negative saving and the backward one only non-positive saving, which is what makes the resulting operator monotone and the associated comparison principle available.

Under a concave value function the two candidates cannot both be active at the same node, since the forward difference of a concave function is smaller than the backward one and the optimal consumption is decreasing in the marginal value of wealth. The construction therefore reduces at the solution to ordinary drift-sign upwinding, but the two-candidate form is what makes the intermediate iterates well behaved, and it is also how the state constraint enters. Rather than imposing a boundary condition at the borrowing limit, the scheme simply denies the agent the option of saving negatively there, and symmetrically denies positive saving at the artificial upper truncation. Both restrictions are expressed as restrictions on the admissible consumption at those nodes.

Recursive utility complicates the candidate policies in one specific way. Under time-additive utility the optimal consumption depends only on the marginal value of wealth. Under Epstein-Zin preferences the felicity aggregator depends on the level of the value function as well, so the consumption rule carries an extra factor built from that level, and the exponent on that factor is fixed by the same combination of risk aversion and elasticity that governs the timing regime. A consequence is that intermediate iterates can leave the region where the rule is well defined, because the factor involves a fractional power of a quantity that must keep its sign, and because a one-sided difference of a not-yet-converged value function can fail to be positive. The scheme is therefore regularised by truncating the downward candidate from above, with a bound chosen large enough to be inactive at any converged solution while keeping the iteration finite along the way.

Returns
-------
np.ndarray, shape (2 * n, 2) of dtype float, the forward candidate consumption stacked above the backward candidate consumption, on the same grid and with the same income-state column order as the supplied value function
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_candidate_consumptions(V: np.ndarray, gamma: float, psi: float, rho: float,
                                   r: float, y1: float, y2: float, xlow: float,
                                   xbar: float, cap: float) -> np.ndarray:
    '''Return the stacked forward and backward candidate consumption policies.

    Parameters
    ----------
    V : np.ndarray
        Shape (n, 2) strictly negative value function on the wealth grid, columns being
        the low and high income states.
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate, strictly positive.
    r : float
        Interest rate, strictly positive.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    cap : float
        Regularising upper bound on the backward candidate.

    Returns
    -------
    out : np.ndarray
        Shape (2 * n, 2) float array holding the forward candidate in its first n rows and
        the backward candidate in its last n rows.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_candidate_consumptions(V: np.ndarray, gamma: float, psi: float, rho: float,
                                           r: float, y1: float, y2: float, xlow: float,
                                           xbar: float, cap: float) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[1] != 2 or V.shape[0] < 2:
        raise ValueError("V must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(V)):
        raise ValueError("V must be finite")
    if not np.all(V < 0.0):
        raise ValueError("V must be strictly negative")
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r),
                      ("y1", y1), ("y2", y2), ("xlow", xlow), ("xbar", xbar), ("cap", cap)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    y1, y2, xlow, xbar, cap = float(y1), float(y2), float(xlow), float(xbar), float(cap)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not rho > 0.0:
        raise ValueError("rho must be strictly positive")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    n = V.shape[0]
    nint = n - 1
    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(n)
    cbar = r * x[:, None] + np.array([y1, y2])[None, :]
    if not np.all(cbar > 0.0):
        raise ValueError("the zero-saving consumption must be strictly positive at every node")
    if not cap > float(cbar.max()):
        raise ValueError("cap must exceed the largest zero-saving consumption on the grid")

    d = (V[1:, :] - V[:nint, :]) / dx
    unconstrained = np.where(d > 0.0,
                             rho ** psi * np.maximum(d, 1e-300) ** (-psi), np.inf)
    weight = ((1.0 - gamma) * V) ** ((1.0 - gamma * psi) / (1.0 - gamma))

    cF = cbar.copy()
    cF[:nint, :] = np.minimum(unconstrained * weight[:nint, :], cbar[:nint, :])
    cB = cbar.copy()
    cB[1:, :] = np.maximum(np.minimum(cap, unconstrained * weight[1:, :]), cbar[1:, :])
    return np.vstack([cF, cB]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.021410049888
y1, y2, xlow, xbar, nint = 0.4, 1.2, -0.25, 24.75, 1250
cap = 5.0
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
V = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
""",
            "call": "compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
            "gold_call": "_oracle_compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 3.0, 0.4, 0.05, 0.02
y1, y2, xlow, xbar, nint = 0.5, 1.5, -0.15, 5.85, 6
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
cap = 4.0
V = np.column_stack([-1.0 / (1.0 + x - xlow), -0.8 / (1.0 + x - xlow)])
""",
            "call": "compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
            "gold_call": "_oracle_compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.03
y1, y2, xlow, xbar, nint = 0.4, 1.2, -0.25, 2.75, 10
cap = 3.0
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
V = np.repeat((-1.0 / (1.0 + x - xlow))[:, None], 2, axis=1)
V[4, 0] = V[3, 0] - 0.05
V[7, 1] = V[6, 1]
""",
            "call": "compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
            "gold_call": "_oracle_compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.03
y1, y2, xlow, xbar, nint = 0.4, 1.2, 0.0, 1.0, 1
cap = 3.0
V = np.array([[-2.0, -1.8], [-1.999999, -1.799999]])
""",
            "call": "compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
            "gold_call": "_oracle_compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2, xlow, xbar, cap)",
        },
        {
            "setup": """import numpy as np
V = np.array([[-1.0, -0.9], [-0.8, -0.7]])
def run_model():
    try:
        compute_candidate_consumptions(V, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_candidate_consumptions(V, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
V = np.array([[-1.0, -0.9], [0.5, -0.7]])
def run_model():
    try:
        compute_candidate_consumptions(V, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_candidate_consumptions(V, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
V = np.array([-1.0, -0.9, -0.8])
def run_model():
    try:
        compute_candidate_consumptions(V, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_candidate_consumptions(V, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
V = np.array([[-1.0, -0.9], [-0.8, -0.7]])
def run_model():
    try:
        compute_candidate_consumptions(V, 1.5, 1.4, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_candidate_consumptions(V, 1.5, 1.4, 0.045, 0.03, 0.4, 1.2, 0.0, 1.0, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
