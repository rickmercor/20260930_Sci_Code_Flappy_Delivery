"""
Solve for the value function that satisfies the discretised stationary equation exactly at a given, fixed pair of candidate consumption policies.

Holding the policies fixed removes the maximisation from the equation but does not make it linear: under recursive utility the felicity aggregator depends on the value function itself, so the resulting square system is nonlinear and must be solved iteratively. The value function is required to stay strictly negative throughout, because the aggregator involves a fractional power of a quantity built from it.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with node zero at the borrowing limit. The policies are supplied stacked, forward candidate in the first half of the rows and backward candidate in the second half, with two columns holding the two income states in the order low income then high income. The initial guess and the returned value function are single blocks of that same node count and column order.

The two income states are coupled by the Poisson switching terms, which are solved together with the drift terms rather than lagged.

Return the value function as a 2-D array with one row per node and two columns.

The iteration stops when the maximum absolute residual of the system falls below the supplied tolerance. Convergence must be reached within the supplied iteration budget.

Inputs are the stacked policies, the initial guess, the risk aversion, the elasticity of intertemporal substitution, the discount rate, the interest rate, the two income levels, the two switching rates, the borrowing limit, the upper truncation point, the residual tolerance and the iteration budget.

Raises ValueError if the stacked policies are not a two-dimensional finite array with two columns and an even number of rows of at least four, if the initial guess is not a two-dimensional finite array with two columns whose row count is half that of the policies, if any entry of the initial guess is not strictly negative, if any real input is not finite, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, if the discount rate or the interest rate is not strictly positive, if either switching rate is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, if the zero-saving consumption is not strictly positive at every node, if the forward candidate implies a negative saving rate anywhere or the backward candidate implies a positive saving rate anywhere, if the tolerance or the iteration budget is not strictly positive, or if the residual tolerance is not reached within the iteration budget.

Policy iteration alternates two steps: evaluate the value of a fixed policy, then improve the policy against that value. Under time-additive utility the evaluation step is a linear system, because the instantaneous payoff depends only on the control, and the whole method reduces to repeated linear solves. Recursive utility breaks this. The felicity aggregator is a function of both consumption and the value function, so freezing the control leaves an equation that is still nonlinear in the unknown, and the evaluation step becomes a root-find rather than a linear solve.

The nonlinearity is mild in form but not in consequence. It enters through a single fractional power of a quantity built from the value function, whose exponent is fixed by the combination of risk aversion and elasticity that also selects the timing regime. In the late-resolution regime that power makes the aggregator decreasing in the value function, which is what preserves the monotonicity the comparison principle needs, and it also means the Jacobian of the evaluation system remains diagonally dominant so a Newton iteration is well behaved. Getting the sign of the exponent wrong reverses that monotonicity and destroys the property, so the iteration fails to converge rather than converging to a slightly different answer.

Two practical points follow. First, the fractional power is only real while the argument keeps its sign, so a Newton step that overshoots past zero has to be shortened; a step restriction that keeps the iterate on the correct side, combined with a requirement that the residual decrease, is enough. Second, the Poisson switching terms couple the two income states, and how they are treated changes the convergence rate rather than the answer: solving them together with the drift gives the quadratic convergence Newton promises, while lagging them onto the right-hand side gives a linearly convergent scheme that reaches the same root but needs many more iterations to reach the same accuracy.

Returns
-------
np.ndarray, shape (n, 2) of dtype float, the strictly negative value function on the wealth grid, with columns holding the low and high income states in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_fixed_policy(policies: np.ndarray, v_init: np.ndarray, gamma: float, psi: float,
                          rho: float, r: float, y1: float, y2: float, lam1: float,
                          lam2: float, xlow: float, xbar: float, tol: float,
                          max_iter: int) -> np.ndarray:
    '''Return the value function consistent with a fixed pair of consumption policies.

    Parameters
    ----------
    policies : np.ndarray
        Shape (2 * n, 2) stacked candidate consumptions, forward block then backward block.
    v_init : np.ndarray
        Shape (n, 2) strictly negative starting guess for the value function.
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
    lam1 : float
        Switching rate out of the low income state, strictly positive.
    lam2 : float
        Switching rate out of the high income state, strictly positive.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    tol : float
        Maximum absolute residual accepted as convergence.
    max_iter : int
        Maximum number of iterations allowed.

    Returns
    -------
    out : np.ndarray
        Shape (n, 2) float value function on the grid, strictly negative.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _oracle_evaluate_fixed_policy(policies: np.ndarray, v_init: np.ndarray, gamma: float,
                                  psi: float, rho: float, r: float, y1: float, y2: float,
                                  lam1: float, lam2: float, xlow: float, xbar: float,
                                  tol: float, max_iter: int) -> np.ndarray:
    policies = np.asarray(policies, dtype=float)
    v_init = np.asarray(v_init, dtype=float)
    if policies.ndim != 2 or policies.shape[1] != 2 or policies.shape[0] < 4 or policies.shape[0] % 2:
        raise ValueError("policies must be a 2-D array with two columns and an even row count of at least 4")
    if not np.all(np.isfinite(policies)):
        raise ValueError("policies must be finite")
    n = policies.shape[0] // 2
    if v_init.ndim != 2 or v_init.shape != (n, 2):
        raise ValueError("v_init must have shape (n, 2) with n half the policy row count")
    if not np.all(np.isfinite(v_init)):
        raise ValueError("v_init must be finite")
    if not np.all(v_init < 0.0):
        raise ValueError("v_init must be strictly negative")
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r), ("y1", y1),
                      ("y2", y2), ("lam1", lam1), ("lam2", lam2), ("xlow", xlow),
                      ("xbar", xbar), ("tol", tol)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("max_iter must be an integer")
    max_iter = int(max_iter)
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    y1, y2, lam1, lam2 = float(y1), float(y2), float(lam1), float(lam2)
    xlow, xbar, tol = float(xlow), float(xbar), float(tol)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not rho > 0.0:
        raise ValueError("rho must be strictly positive")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not (lam1 > 0.0 and lam2 > 0.0):
        raise ValueError("both switching rates must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")
    if not tol > 0.0:
        raise ValueError("tol must be strictly positive")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nint = n - 1
    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(n)
    lam = np.array([lam1, lam2])
    cbar = r * x[:, None] + np.array([y1, y2])[None, :]
    if not np.all(cbar > 0.0):
        raise ValueError("the zero-saving consumption must be strictly positive at every node")
    cF, cB = policies[:n, :], policies[n:, :]
    sF, sB = cbar - cF, cbar - cB
    if np.any(sF < -1e-12) or np.any(sB > 1e-12):
        raise ValueError("forward saving must be non-negative and backward saving non-positive")
    sF = np.maximum(sF, 0.0); sF[nint, :] = 0.0
    sB = np.minimum(sB, 0.0); sB[0, :] = 0.0

    theta = (1.0 - 1.0 / psi) / (1.0 - gamma)
    powv = 1.0 - theta
    nu = 1.0 - 1.0 / psi
    phi = (rho / nu) * (cF ** nu + cB ** nu - cbar ** nu)

    blocks = []
    for j in (0, 1):
        main = (sF[:, j] - sB[:, j]) / dx + lam[j] + rho / theta
        blocks.append(sp.diags([sB[1:, j] / dx, main, -sF[:nint, j] / dx],
                               [-1, 0, 1], format="csr"))
    eye = sp.eye(n, format="csr")
    A = sp.bmat([[blocks[0], -lam[0] * eye], [-lam[1] * eye, blocks[1]]], format="csc")

    vf = v_init.T.reshape(-1).copy()
    pf = phi.T.reshape(-1)
    converged = False
    for _ in range(max_iter):
        w = (1.0 - gamma) * vf
        res = A @ vf - pf * w ** powv
        rn = float(np.max(np.abs(res)))
        if rn < tol:
            converged = True
            break
        jac = (A - sp.diags(pf * powv * (1.0 - gamma) * w ** (powv - 1.0))).tocsc()
        step = spl.spsolve(jac, res)
        t = 1.0
        for _ in range(60):
            trial = vf - t * step
            if np.max(trial) < 0.0:
                rt = np.max(np.abs(A @ trial - pf * ((1.0 - gamma) * trial) ** powv))
                if rt < (1.0 - 1e-4 * t) * rn:
                    break
            t *= 0.5
        vf = vf - t * step
    if not converged:
        w = (1.0 - gamma) * vf
        if float(np.max(np.abs(A @ vf - pf * w ** powv))) >= tol:
            raise ValueError("the residual tolerance was not reached within max_iter")
    return vf.reshape(2, n).T.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.021410049888
y1, y2, lam1, lam2 = 0.4, 1.2, 0.25, 0.15
xlow, xbar, nint = -0.25, 24.75, 1250
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
cbar = r * x[:, None] + np.array([y1, y2])[None, :]
cF = 0.85 * cbar
cF[nint, :] = cbar[nint, :]
cB = 1.40 * cbar
cB[0, :] = cbar[0, :]
policies = np.vstack([cF, cB])
v_init = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
""",
            "call": "evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-12, 100)",
            "gold_call": "_oracle_evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-12, 100)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.021410049888
y1, y2, lam1, lam2 = 0.4, 1.2, 0.25, 0.15
xlow, xbar, nint = -0.25, 24.75, 1250
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
cbar = r * x[:, None] + np.array([y1, y2])[None, :]
policies = np.vstack([cbar, cbar])
v_init = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
""",
            "call": "evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-12, 100)",
            "gold_call": "_oracle_evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-12, 100)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 3.0, 0.4, 0.05, 0.02
y1, y2, lam1, lam2 = 0.5, 1.5, 0.4, 0.4
xlow, xbar, nint = -0.15, 5.85, 6
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
cbar = r * x[:, None] + np.array([y1, y2])[None, :]
cF = 0.6 * cbar
cF[nint, :] = cbar[nint, :]
cB = 2.0 * cbar
cB[0, :] = cbar[0, :]
policies = np.vstack([cF, cB])
v_init = np.repeat((-3.0 / (1.0 + x - xlow))[:, None], 2, axis=1)
""",
            "call": "evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-13, 100)",
            "gold_call": "_oracle_evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-13, 100)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.03
y1, y2, lam1, lam2 = 0.4, 1.2, 0.05, 0.9
xlow, xbar, nint = 0.0, 1.0, 1
x = np.array([0.0, 1.0])
cbar = r * x[:, None] + np.array([y1, y2])[None, :]
policies = np.vstack([cbar, cbar])
v_init = np.array([[-9.0, -0.05], [-9.0, -0.05]])
""",
            "call": "evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-13, 200)",
            "gold_call": "_oracle_evaluate_fixed_policy(policies, v_init, gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, 1e-13, 200)",
        },
        {
            "setup": """import numpy as np
x = np.array([0.0, 1.0])
cbar = 0.03 * x[:, None] + np.array([0.4, 1.2])[None, :]
policies = np.vstack([1.5 * cbar, cbar])
v_init = np.array([[-2.0, -1.8], [-1.9, -1.7]])
def run_model():
    try:
        evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, 1e-12, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, 1e-12, 50)
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
x = np.array([0.0, 1.0])
cbar = 0.03 * x[:, None] + np.array([0.4, 1.2])[None, :]
policies = np.vstack([cbar, cbar])
v_init = np.array([[-2.0, -1.8], [-1.9, -1.7]])
def run_model():
    try:
        evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, 1e-30, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, 1e-30, 3)
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
x = np.array([0.0, 1.0])
cbar = 0.03 * x[:, None] + np.array([0.4, 1.2])[None, :]
policies = np.vstack([cbar, cbar])
v_init = np.array([[-2.0, 1.8], [-1.9, -1.7]])
def run_model():
    try:
        evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, 1e-12, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, 1e-12, 50)
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
x = np.array([0.0, 1.0])
cbar = 0.03 * x[:, None] + np.array([0.4, 1.2])[None, :]
policies = np.vstack([cbar, cbar])
v_init = np.array([[-2.0, -1.8], [-1.9, -1.7]])
def run_model():
    try:
        evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 0.15, 0.0, 1.0, 1e-12, 50)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_fixed_policy(policies, v_init, 1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.0, 0.15, 0.0, 1.0, 1e-12, 50)
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
