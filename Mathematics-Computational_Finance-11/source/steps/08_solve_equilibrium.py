"""
Solve the stationary equilibrium end to end and return the total weight the wealth distribution places on the borrowing limit.

This step orchestrates the whole pipeline. For a trial interest rate it derives the scalar constants of the recursive-utility formulation, builds the discrete barriers, solves the Hamilton-Jacobi-Bellman equation by policy iteration, forms the stationary distribution of the resulting saving policy at the given step length, and reduces that distribution to aggregate capital, aggregate labour supply and the two weights at the borrowing limit. The interest rate is then required to equal the net marginal product of capital of a Cobb-Douglas technology with the supplied capital share, depreciation rate and total factor productivity.

The equilibrium interest rate is located by bisection on the supplied bracket, halving until the bracket width falls below the supplied tolerance and taking the midpoint. The bracketing values must produce residuals of opposite sign, with the residual defined as the net marginal product of capital minus the trial interest rate.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point with the supplied number of intervals, and node zero is at the borrowing limit.

Three consistency checks are performed at the equilibrium and each raises on failure: the converged value function must lie between the two discrete barriers at every node and income state; re-deriving the candidate consumption policies from the converged value function must reproduce the saving policy returned by the policy iteration; and re-solving the equation at those candidate policies must reproduce the converged value function.

The policy iteration is run with a policy tolerance of 1e-7, an inner residual tolerance of 1e-12, an outer budget of 400 iterations and an inner budget of 100 iterations.

Return the sum of the two weights at the borrowing limit as a single float.

Inputs are the risk aversion, the elasticity of intertemporal substitution, the discount rate, the two income levels, the two switching rates, the borrowing limit, the upper truncation point, the number of grid intervals, the step length of the forward process, the capital share, the depreciation rate, the total factor productivity, the two bracketing interest rates, the bracket tolerance and the regularising cap.

Raises ValueError if any real input is not finite, if the number of grid intervals is not a positive integer, if the capital share is not strictly between zero and one, if the depreciation rate is negative, if the total factor productivity is not strictly positive, if the bracket is not ordered strictly increasing with both endpoints strictly positive and strictly below the discount rate, if the bracket tolerance is not strictly positive, if the residuals at the two bracketing interest rates do not have opposite signs, if aggregate capital is not strictly positive at any evaluated interest rate, if the recursive-utility parameter is below one so that the late-resolution policy iteration is not justified, or if any of the three consistency checks fails. Any error raised by the underlying stages is propagated.

A stationary recursive equilibrium is a fixed point in a single scalar. Given a price, individual optimisation delivers a policy; the policy delivers a distribution; the distribution delivers aggregates; the aggregates deliver a price. Nothing about that loop is guaranteed to be well behaved in general, but in this model it is: aggregate capital rises with the interest rate, the marginal product of capital falls in aggregate capital, so the excess between the implied and the trial rate is monotone and crosses zero exactly once on the admissible range. That makes bisection both sufficient and safe, and it means the located rate does not depend on where the bracket was placed.

The economics of the loop are worth keeping in view. A higher interest rate makes saving more attractive and pushes the wealth distribution to the right, which raises aggregate capital and, through diminishing returns, lowers the marginal product. The borrowing constraint is what keeps aggregate capital finite as the interest rate approaches the discount rate: without it, agents would accumulate without bound. The quantity of interest, the share of the population pressed against that constraint, therefore moves in the opposite direction to the interest rate and is sensitive to every part of the pipeline at once, which is why it is a demanding end-to-end target rather than a local diagnostic.

Because the pipeline composes many stages, it is worth building in checks that consume the intermediate results rather than discarding them. The barriers bracket the value function by construction, so a converged value function outside them indicates a failure of the iteration rather than of the economics. Re-deriving the candidate policies from the converged value function must return the policy the iteration reported, since a converged policy iteration is by definition a fixed point of exactly that map. And re-solving the equation at those policies must return the same value function, since that is what convergence means. Each of these is cheap, each can genuinely fail, and each localises a fault to one stage instead of leaving only a wrong final number.

Returns
-------
float, the total weight the equilibrium stationary distribution places on the grid node at the borrowing limit, summed over the two income states, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_equilibrium(gamma: float, psi: float, rho: float, y1: float, y2: float,
                      lam1: float, lam2: float, xlow: float, xbar: float, nint: int,
                      h: float, alpha: float, delta: float, tfp: float, r_lo: float,
                      r_hi: float, tol_r: float, cap: float) -> float:
    '''Return the total stationary weight at the borrowing limit in equilibrium.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    lam1 : float
        Switching rate out of the low income state.
    lam2 : float
        Switching rate out of the high income state.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    nint : int
        Number of grid intervals.
    h : float
        Step length of the forward process.
    alpha : float
        Capital share of the Cobb-Douglas technology.
    delta : float
        Depreciation rate.
    tfp : float
        Total factor productivity.
    r_lo : float
        Lower bracketing interest rate.
    r_hi : float
        Upper bracketing interest rate.
    tol_r : float
        Bracket width below which the bisection stops.
    cap : float
        Regularising upper bound on the backward candidate consumption.

    Returns
    -------
    out : float
        Total weight the equilibrium stationary distribution places on the borrowing limit,
        summed over the two income states.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_equilibrium(gamma: float, psi: float, rho: float, y1: float, y2: float,
                              lam1: float, lam2: float, xlow: float, xbar: float, nint: int,
                              h: float, alpha: float, delta: float, tfp: float, r_lo: float,
                              r_hi: float, tol_r: float, cap: float) -> float:
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("y1", y1), ("y2", y2),
                      ("lam1", lam1), ("lam2", lam2), ("xlow", xlow), ("xbar", xbar),
                      ("h", h), ("alpha", alpha), ("delta", delta), ("tfp", tfp),
                      ("r_lo", r_lo), ("r_hi", r_hi), ("tol_r", tol_r), ("cap", cap)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    if isinstance(nint, (bool, np.bool_)) or not isinstance(nint, (int, np.integer)) or int(nint) < 1:
        raise ValueError("nint must be a positive integer")
    nint = int(nint)
    gamma, psi, rho = float(gamma), float(psi), float(rho)
    y1, y2, lam1, lam2 = float(y1), float(y2), float(lam1), float(lam2)
    xlow, xbar, h = float(xlow), float(xbar), float(h)
    alpha, delta, tfp = float(alpha), float(delta), float(tfp)
    r_lo, r_hi, tol_r, cap = float(r_lo), float(r_hi), float(tol_r), float(cap)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if not tfp > 0.0:
        raise ValueError("tfp must be strictly positive")
    if not (0.0 < r_lo < r_hi < rho):
        raise ValueError("the bracket must satisfy 0 < r_lo < r_hi < rho")
    if not tol_r > 0.0:
        raise ValueError("tol_r must be strictly positive")

    tol_policy, tol_residual, max_outer, max_inner = 1e-7, 1e-12, 400, 100
    n = nint + 1
    x = xlow + (xbar - xlow) / nint * np.arange(n)

    def stage(r):
        constants = _oracle_compute_model_constants(gamma, psi, rho, r)
        theta, b = float(constants[0]), float(constants[4])
        if theta < 1.0:
            raise ValueError("theta below one: the late-resolution policy iteration is not justified")
        barriers = _oracle_compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)
        lower, upper = barriers[:n, :], barriers[n:, :]
        sol = _oracle_solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar,
                                           lower, cap, tol_policy, tol_residual,
                                           max_outer, max_inner)
        V, s = sol[:n, :], sol[n:, :]
        if np.any(V < lower - 1e-8) or np.any(V > upper + 1e-8):
            raise ValueError("the converged value function left the discrete barriers")
        policies = _oracle_compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2,
                                                          xlow, xbar, cap)
        czero = r * x[:, None] + np.array([y1, y2])[None, :]
        sF = np.maximum(czero - policies[:n, :], 0.0)
        sF[nint, :] = 0.0
        sB = np.minimum(czero - policies[n:, :], 0.0)
        sB[0, :] = 0.0
        if float(np.max(np.abs(sF + sB - s))) > 1e-9:
            raise ValueError("re-derived candidates do not reproduce the returned saving policy")
        V_again = _oracle_evaluate_fixed_policy(policies, V, gamma, psi, rho, r, y1, y2,
                                                lam1, lam2, xlow, xbar, tol_residual, max_inner)
        if float(np.max(np.abs(V_again - V))) > 1e-9:
            raise ValueError("re-evaluating the converged policy does not reproduce the value function")
        G = _oracle_stationary_distribution(s, lam1, lam2, h, xlow, xbar)
        agg = _oracle_compute_aggregates(G, y1, y2, xlow, xbar)
        K, N = float(agg[0]), float(agg[1])
        if not K > 0.0:
            raise ValueError("aggregate capital must be strictly positive")
        residual = tfp * alpha * (K / N) ** (alpha - 1.0) - delta - r
        return residual, float(agg[2] + agg[3])

    f_lo = stage(r_lo)[0]
    f_hi = stage(r_hi)[0]
    if f_lo * f_hi > 0.0:
        raise ValueError("the residuals at the bracket endpoints must have opposite signs")

    lo, hi = (r_lo, r_hi) if f_lo > 0.0 else (r_hi, r_lo)
    while abs(hi - lo) > tol_r:
        mid = 0.5 * (lo + hi)
        if stage(mid)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    return float(stage(0.5 * (lo + hi))[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
args = (1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 1250, 1.0,
        0.36, 0.08, 0.5, 0.005, 0.035, 1e-12, 5.0)
""",
            "call": "solve_equilibrium(*args)",
            "gold_call": "_oracle_solve_equilibrium(*args)",
        },
        {
            "setup": """import numpy as np
args = (1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
        0.36, 0.08, 0.5, 0.005, 0.035, 1e-10, 5.0)
""",
            "call": "solve_equilibrium(*args)",
            "gold_call": "_oracle_solve_equilibrium(*args)",
        },
        {
            "setup": """import numpy as np
args = (2.5, 0.35, 0.05, 0.5, 1.5, 0.3, 0.3, -0.1, 19.9, 200, 0.25,
        0.33, 0.06, 0.6, 0.004, 0.040, 1e-10, 6.0)
""",
            "call": "solve_equilibrium(*args)",
            "gold_call": "_oracle_solve_equilibrium(*args)",
        },
        {
            "setup": """import numpy as np
base = (1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200)
tail = (0.36, 0.08, 0.5, 0.005, 0.035, 1e-10, 5.0)
def step_gap():
    a = solve_equilibrium(*base, 1.0, *tail)
    b = solve_equilibrium(*base, 0.02, *tail)
    return [float(a), float(b)]
def gold_step_gap():
    a = _oracle_solve_equilibrium(*base, 1.0, *tail)
    b = _oracle_solve_equilibrium(*base, 0.02, *tail)
    return [float(a), float(b)]
""",
            "call": "step_gap()",
            "gold_call": "gold_step_gap()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_equilibrium(1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                          0.36, 0.08, 0.5, 0.030, 0.035, 1e-10, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_equilibrium(1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                                  0.36, 0.08, 0.5, 0.030, 0.035, 1e-10, 5.0)
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
def run_model():
    try:
        solve_equilibrium(1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                          1.4, 0.08, 0.5, 0.005, 0.035, 1e-10, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_equilibrium(1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                                  1.4, 0.08, 0.5, 0.005, 0.035, 1e-10, 5.0)
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
def run_model():
    try:
        solve_equilibrium(1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                          0.36, 0.08, 0.5, 0.005, 0.055, 1e-10, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_equilibrium(1.5, 0.6, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                                  0.36, 0.08, 0.5, 0.005, 0.055, 1e-10, 5.0)
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
def run_model():
    try:
        solve_equilibrium(1.5, 0.75, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                          0.36, 0.08, 0.5, 0.005, 0.035, 1e-10, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_equilibrium(1.5, 0.75, 0.045, 0.4, 1.2, 0.25, 0.15, -0.25, 24.75, 200, 1.0,
                                  0.36, 0.08, 0.5, 0.005, 0.035, 1e-10, 5.0)
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
