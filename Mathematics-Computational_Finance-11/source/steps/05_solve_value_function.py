"""
Solve the discretised stationary Hamilton-Jacobi-Bellman equation by policy iteration and return both the converged value function and the resulting saving policy.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with node zero at the borrowing limit; its node count is taken from the supplied initial guess, which has one row per node and two columns holding the two income states in the order low income then high income.

The outer loop alternates two operations. It improves the pair of candidate consumption policies against the current value function, using the same forward and backward candidates and the same regularising cap as the candidate-policy construction, and it then re-solves the equation exactly at those fixed candidates. It stops when the largest absolute change in either candidate between successive outer iterations, maximised over nodes and summed over the two income states, falls below the supplied policy tolerance. The loop is initialised with both candidates equal to the zero-saving consumption, and the value function is carried from one outer iteration to the next as the starting point of the inner solve.

Return a single 2-D array with twice as many rows as the grid and two columns, obtained by stacking vertically the converged value function in the first block of rows and the saving policy in the second. The saving policy at a node is the sum of the saving rate implied by the forward candidate and the saving rate implied by the backward candidate at that node.

Inputs are the risk aversion, the elasticity of intertemporal substitution, the discount rate, the interest rate, the two income levels, the two switching rates, the borrowing limit, the upper truncation point, the initial guess for the value function, the regularising cap, the policy tolerance, the residual tolerance of the inner solve, the outer iteration budget and the inner iteration budget.

Raises ValueError if the initial guess is not a two-dimensional finite array with exactly two columns and at least two rows, if any of its entries is not strictly negative, if any real input is not finite, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, if the discount rate or the interest rate is not strictly positive, if either switching rate is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, if the zero-saving consumption is not strictly positive at every node, if the cap does not exceed the largest zero-saving consumption on the grid, if either tolerance or either iteration budget is not strictly positive, if an inner solve fails to reach its residual tolerance within the inner budget, or if the outer loop fails to reach the policy tolerance within the outer budget.

Policy iteration for a monotone scheme converges because each sweep produces a subsolution of the equation, the sweeps are monotone in the ordering of grid functions, and the barriers bound the sequence. In the late-resolution regime all three ingredients survive the passage to recursive utility, so the method inherits its classical guarantee and terminates: once the discrete maximiser stops changing, the iterate is an exact solution of the scheme, not merely an approximate one. That is why the stopping test is placed on the policy rather than on the value function, and why the tolerance chosen for it turns out not to affect the converged answer over a wide range.

Two features of the loop are specific to this setting. The evaluation step is a nonlinear solve rather than a linear one, because the aggregator depends on the value function, so each sweep contains an inner iteration of its own; carrying the value function from one sweep to the next rather than restarting it keeps that inner cost to a handful of iterations. And the regularising cap on the downward candidate is needed for the sweeps, not for the answer. Intermediate iterates can have a one-sided difference of the value function pass through zero, at which point the unconstrained candidate consumption is unbounded; the cap keeps the iterate finite, and because it is chosen above any value the converged policy can take, it leaves the fixed point untouched.

The converged object of interest is the saving policy rather than the value function itself, because it is the saving policy that drives the distribution of wealth. Its qualitative shape is known in advance and is worth checking: the low-income agent dissaves everywhere above the borrowing limit and saves exactly nothing at it, while the high-income agent saves at the limit, provided the switching rate out of the high state is large enough for a precautionary motive to be present, and crosses into dissaving at a single interior wealth level beyond which no stationary mass can accumulate.

Returns
-------
np.ndarray, shape (2 * n, 2) of dtype float, the converged value function stacked above the saving policy, both on the wealth grid with columns holding the low and high income states in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_value_function(gamma: float, psi: float, rho: float, r: float, y1: float, y2: float,
                         lam1: float, lam2: float, xlow: float, xbar: float,
                         v_init: np.ndarray, cap: float, tol_policy: float,
                         tol_residual: float, max_outer: int, max_inner: int) -> np.ndarray:
    '''Return the converged value function stacked above the saving policy.

    Parameters
    ----------
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
    v_init : np.ndarray
        Shape (n, 2) strictly negative starting guess, which fixes the grid node count.
    cap : float
        Regularising upper bound on the backward candidate consumption.
    tol_policy : float
        Convergence tolerance on the change in the candidate policies.
    tol_residual : float
        Residual tolerance of each inner solve.
    max_outer : int
        Maximum number of outer policy iterations.
    max_inner : int
        Maximum number of iterations allowed in each inner solve.

    Returns
    -------
    out : np.ndarray
        Shape (2 * n, 2) float array holding the converged value function in its first n
        rows and the saving policy in its last n rows.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _h_candidates(V, cbar, dx, gamma, psi, rho, cap):
    n = V.shape[0]
    nint = n - 1
    d = (V[1:, :] - V[:nint, :]) / dx
    unconstrained = np.where(d > 0.0, rho ** psi * np.maximum(d, 1e-300) ** (-psi), np.inf)
    weight = ((1.0 - gamma) * V) ** ((1.0 - gamma * psi) / (1.0 - gamma))
    cF = cbar.copy()
    cF[:nint, :] = np.minimum(unconstrained * weight[:nint, :], cbar[:nint, :])
    cB = cbar.copy()
    cB[1:, :] = np.maximum(np.minimum(cap, unconstrained * weight[1:, :]), cbar[1:, :])
    return cF, cB


def _h_operator(sF, sB, lam, dx, rho, theta, n):
    nint = n - 1
    blocks = []
    for j in (0, 1):
        main = (sF[:, j] - sB[:, j]) / dx + lam[j] + rho / theta
        blocks.append(sp.diags([sB[1:, j] / dx, main, -sF[:nint, j] / dx],
                               [-1, 0, 1], format="csr"))
    eye = sp.eye(n, format="csr")
    return sp.bmat([[blocks[0], -lam[0] * eye], [-lam[1] * eye, blocks[1]]], format="csc")


def _h_evaluate(A, phi, v, gamma, powv, tol, max_iter, n):
    vf = v.T.reshape(-1).copy()
    pf = phi.T.reshape(-1)
    for _ in range(max_iter):
        w = (1.0 - gamma) * vf
        res = A @ vf - pf * w ** powv
        rn = float(np.max(np.abs(res)))
        if rn < tol:
            return vf.reshape(2, n).T.copy()
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
    w = (1.0 - gamma) * vf
    if float(np.max(np.abs(A @ vf - pf * w ** powv))) >= tol:
        raise ValueError("an inner solve failed to reach its residual tolerance")
    return vf.reshape(2, n).T.copy()


def _oracle_solve_value_function(gamma: float, psi: float, rho: float, r: float, y1: float,
                                 y2: float, lam1: float, lam2: float, xlow: float,
                                 xbar: float, v_init: np.ndarray, cap: float,
                                 tol_policy: float, tol_residual: float, max_outer: int,
                                 max_inner: int) -> np.ndarray:
    v_init = np.asarray(v_init, dtype=float)
    if v_init.ndim != 2 or v_init.shape[1] != 2 or v_init.shape[0] < 2:
        raise ValueError("v_init must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(v_init)):
        raise ValueError("v_init must be finite")
    if not np.all(v_init < 0.0):
        raise ValueError("v_init must be strictly negative")
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r), ("y1", y1),
                      ("y2", y2), ("lam1", lam1), ("lam2", lam2), ("xlow", xlow),
                      ("xbar", xbar), ("cap", cap), ("tol_policy", tol_policy),
                      ("tol_residual", tol_residual)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    for name, val in (("max_outer", max_outer), ("max_inner", max_inner)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if int(val) < 1:
            raise ValueError(name + " must be a positive integer")
    max_outer, max_inner = int(max_outer), int(max_inner)
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    y1, y2, lam1, lam2 = float(y1), float(y2), float(lam1), float(lam2)
    xlow, xbar, cap = float(xlow), float(xbar), float(cap)
    tol_policy, tol_residual = float(tol_policy), float(tol_residual)
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
    if not (tol_policy > 0.0 and tol_residual > 0.0):
        raise ValueError("both tolerances must be strictly positive")

    n = v_init.shape[0]
    nint = n - 1
    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(n)
    lam = np.array([lam1, lam2])
    cbar = r * x[:, None] + np.array([y1, y2])[None, :]
    if not np.all(cbar > 0.0):
        raise ValueError("the zero-saving consumption must be strictly positive at every node")
    if not cap > float(cbar.max()):
        raise ValueError("cap must exceed the largest zero-saving consumption on the grid")

    theta = (1.0 - 1.0 / psi) / (1.0 - gamma)
    powv = 1.0 - theta
    nu = 1.0 - 1.0 / psi
    cF, cB = cbar.copy(), cbar.copy()
    V = v_init.copy()
    converged = False
    for _ in range(max_outer):
        sF = np.maximum(cbar - cF, 0.0); sF[nint, :] = 0.0
        sB = np.minimum(cbar - cB, 0.0); sB[0, :] = 0.0
        A = _h_operator(sF, sB, lam, dx, rho, theta, n)
        phi = (rho / nu) * (cF ** nu + cB ** nu - cbar ** nu)
        V = _h_evaluate(A, phi, V, gamma, powv, tol_residual, max_inner, n)
        cFn, cBn = _h_candidates(V, cbar, dx, gamma, psi, rho, cap)
        change = sum(float(np.max(np.abs(cFn[:, j] - cF[:, j])))
                     + float(np.max(np.abs(cBn[:, j] - cB[:, j]))) for j in (0, 1))
        cF, cB = cFn, cBn
        if change < tol_policy:
            converged = True
            break
    if not converged:
        raise ValueError("the outer policy iteration failed to converge within max_outer")

    sF = np.maximum(cbar - cF, 0.0); sF[nint, :] = 0.0
    sB = np.minimum(cbar - cB, 0.0); sB[0, :] = 0.0
    return np.vstack([V, sF + sB]).astype(float)

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
v_init = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
""",
            "call": "solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 5.0, 1e-7, 1e-12, 400, 100)",
            "gold_call": "_oracle_solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 5.0, 1e-7, 1e-12, 400, 100)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.001
y1, y2, lam1, lam2 = 0.4, 1.2, 0.25, 0.15
xlow, xbar, nint = -0.25, 24.75, 500
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
v_init = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
""",
            "call": "solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 5.0, 1e-7, 1e-12, 400, 100)",
            "gold_call": "_oracle_solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 5.0, 1e-7, 1e-12, 400, 100)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 4.0, 0.2, 0.06, 0.03
y1, y2, lam1, lam2 = 0.5, 1.5, 0.4, 0.05
xlow, xbar, nint = -0.15, 9.85, 40
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
v_init = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
""",
            "call": "solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 6.0, 1e-8, 1e-13, 400, 100)",
            "gold_call": "_oracle_solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 6.0, 1e-8, 1e-13, 400, 100)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.021410049888
y1, y2, lam1, lam2 = 0.4, 1.2, 0.25, 0.15
xlow, xbar, nint = -0.25, 24.75, 200
x = xlow + (xbar - xlow) / nint * np.arange(nint + 1)
v_init = np.repeat(((r * x + y1) ** (1 - gamma) / (1 - gamma))[:, None], 2, axis=1)
out_a = solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 3.0, 1e-7, 1e-12, 400, 100)
out_b = solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 20.0, 1e-7, 1e-12, 400, 100)
def cap_gap():
    return float(np.max(np.abs(out_a - out_b)))
gold_a = _oracle_solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 3.0, 1e-7, 1e-12, 400, 100)
gold_b = _oracle_solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar, v_init, 20.0, 1e-7, 1e-12, 400, 100)
def gold_cap_gap():
    return float(np.max(np.abs(gold_a - gold_b)))
""",
            "call": "cap_gap()",
            "gold_call": "gold_cap_gap()",
        },
        {
            "setup": """import numpy as np
x = np.linspace(0.0, 1.0, 6)
v_init = np.repeat((-1.0 / (1.0 + x))[:, None], 2, axis=1)
def run_model():
    try:
        solve_value_function(1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, v_init, 1.0, 1e-7, 1e-12, 400, 100)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_value_function(1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, v_init, 1.0, 1e-7, 1e-12, 400, 100)
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
x = np.linspace(0.0, 1.0, 6)
v_init = np.repeat((-1.0 / (1.0 + x))[:, None], 2, axis=1)
def run_model():
    try:
        solve_value_function(1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, v_init, 3.0, 1e-14, 1e-12, 2, 100)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_value_function(1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, v_init, 3.0, 1e-14, 1e-12, 2, 100)
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
x = np.linspace(0.0, 1.0, 6)
v_init = np.repeat((1.0 / (1.0 + x))[:, None], 2, axis=1)
def run_model():
    try:
        solve_value_function(1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, v_init, 3.0, 1e-7, 1e-12, 400, 100)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_value_function(1.5, 0.6, 0.045, 0.03, 0.4, 1.2, 0.25, 0.15, 0.0, 1.0, v_init, 3.0, 1e-7, 1e-12, 400, 100)
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
x = np.linspace(0.0, 1.0, 6)
v_init = np.repeat((-1.0 / (1.0 + x))[:, None], 2, axis=1)
def run_model():
    try:
        solve_value_function(1.5, 0.6, 0.045, 0.03, 1.2, 0.4, 0.25, 0.15, 0.0, 1.0, v_init, 3.0, 1e-7, 1e-12, 400, 100)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_value_function(1.5, 0.6, 0.045, 0.03, 1.2, 0.4, 0.25, 0.15, 0.0, 1.0, v_init, 3.0, 1e-7, 1e-12, 400, 100)
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
