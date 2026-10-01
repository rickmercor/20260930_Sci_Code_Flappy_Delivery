"""
Compute the intrinsic propagation speed c(alpha, B, phi) of an edge: the asymptotic speed, in nodes per unit time, at which the front launched by the constant input x_in into the uniform resting state x_init travels through an infinitely long uniform pathway in which every edge carries the parameters (alpha, B, phi); that is, the speed at which the front's profile translates without change of shape once the boundary transient has decayed. The returned value must be accurate to a relative error of rel_tol or better; how the infinite-pathway limit is reached to that accuracy is part of the task.

Because the transition region of the front spans only a few nodes, the delay a heterogeneous pathway imposes can be attributed edge by edge. Each edge is therefore credited with the speed a front would have if the whole pathway were built from that edge. In a long uniform pathway the front settles into a shape-invariant travelling wave, and its speed is the reciprocal of the limiting delay between the switching of consecutive nodes; the approach to that limit is geometric, and it slows down as the bias approaches the boundary of the bistable window.

Returns
-------
float, the asymptotic front speed c(alpha, B, phi) in nodes per unit time, accurate to rel_tol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def intrinsic_edge_speed(alpha: float, B: float, phi: float, x_in: float, x_init: float,
                                 rel_tol: float) -> float:
    """Compute the intrinsic propagation speed c(alpha, B, phi) of an edge: the asymptotic speed, in nodes per unit time, at which the front launched by the constant input x_in into the uniform resting state x_init travels through an infinitely long uniform pathway in which every edge carries the parameters (alpha, B, phi); that is, the speed at which the front's profile translates without change of shape once the boundary transient has decayed. The returned value must be accurate to a relative error of rel_tol or better; how the infinite-pathway limit is reached to that accuracy is part of the task.

    Parameters
    ----------
    alpha : float
        Edge timescale parameter, positive.
    B : float
        Edge saturation parameter, greater than 1.
    phi : float
        Edge bias parameter, with |phi| < 1 / B (bistable edge).
    x_in : float
        Constant upstream input, +1 or -1.
    x_init : float
        Uniform resting state of the pathway, +1 or -1, different from x_in.
    rel_tol : float
        Required relative accuracy of the returned speed, in (0, 1e-6].

    Returns
    -------
    c : float
        Asymptotic front speed.

    Raises
    ------
    ValueError
        If alpha is not positive, B is not greater than 1, |phi| >= 1 / B, x_in or x_init is not +1 or -1, x_in equals x_init, rel_tol is outside (0, 1e-6], or the requested accuracy cannot be certified.
    """
    return c

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_params(alpha, B, phi):
    a = _as_vector(alpha, "alpha"); b = _as_vector(B, "B"); p = _as_vector(phi, "phi")
    if not (a.size == b.size == p.size):
        raise ValueError("alpha, B and phi must have the same length")
    if np.any(a <= 0.0) or np.any(b <= 1.0) or np.any(np.abs(p) > 1.0):
        raise ValueError("need alpha > 0, B > 1 and |phi| <= 1 on every edge")
    return a, b, p


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def _uniform_chain_rhs(t, y, x_in, av, bv, pv):
    """Right-hand side of a uniform reference chain (module-level so the oracle stays a single function)."""
    return _oracle_cascade_rate(y, x_in, av, bv, pv)


def _uniform_chain_delays(n, alpha, B, phi, x_in, x_init, sgn):
    """Zero-crossing delays between consecutive nodes of an n-node uniform chain, from a tight DOP853 integration
    with root-located crossing times (dense output + Brent)."""
    av, bv, pv = np.full(n, alpha), np.full(n, B), np.full(n, phi)
    event = (lambda t, y, *args: sgn * y[-1])
    event.terminal = True
    event.direction = 1.0
    sol = solve_ivp(_uniform_chain_rhs, (0.0, 400.0 * n / alpha), np.full(n, x_init), args=(x_in, av, bv, pv),
                    method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True, events=event)
    if sol.t_events[0].size == 0:
        raise ValueError("the front did not reach the end of the reference chain")
    grid = np.linspace(0.0, float(sol.t[-1]), 20001)
    X = sol.sol(grid)
    crossing = np.full(n, np.nan)
    for i in range(n):
        g = sgn * X[i]
        idx = np.where((g[:-1] < 0.0) & (g[1:] >= 0.0))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        crossing[i] = brentq(lambda t, i=i: sgn * float(sol.sol(t)[i]), grid[k], grid[k + 1], xtol=1e-14)
    delays = np.diff(crossing)
    return delays[np.isfinite(delays)]


def _oracle_intrinsic_edge_speed(alpha: float, B: float, phi: float, x_in: float, x_init: float,
                                 rel_tol: float) -> float:
    """Property-defined intrinsic speed c(alpha, B, phi): the asymptotic speed, in nodes per unit time, at which
    the front launched by the constant input x_in into the uniform resting state x_init travels through an
    infinitely long uniform pathway built from this edge, i.e. the speed at which the front profile translates
    without change of shape once the boundary transient has decayed. Returned to a relative accuracy of rel_tol
    or better.

    Construction: the reciprocal of the limiting per-node delay of the front. The delay between the zero crossings
    of consecutive nodes converges geometrically along the chain (contraction factor between 0.55 and 0.95 per node
    for bistable edges, slowest close to the bistability boundary), so a chain of 240, 480 or 960 nodes integrated
    with a tight tolerance and root-located crossing times (dense output + Brent, never sample interpolation) is
    extended until the last ten delays agree to rel_tol / 10; a ValueError is raised if that never happens."""
    for name, v in (("alpha", alpha), ("B", B), ("phi", phi), ("x_in", x_in), ("x_init", x_init), ("rel_tol", rel_tol)):
        if isinstance(v, bool) or not np.isfinite(float(v)):
            raise ValueError(f"{name} must be a finite number")
    a, b, p = float(alpha), float(B), float(phi)
    xin, x0, tol = float(x_in), float(x_init), float(rel_tol)
    if a <= 0.0 or b <= 1.0 or abs(p) >= 1.0 / b:
        raise ValueError("need alpha > 0, B > 1 and a bistable edge |phi| < 1 / B")
    if not (0.0 < tol <= 1e-6):
        raise ValueError("rel_tol must lie in (0, 1e-6]")
    if abs(abs(xin) - 1.0) > 0.0 or abs(abs(x0) - 1.0) > 0.0 or xin == x0:
        raise ValueError("the front must be launched from one saturated state into the other: x_in, x_init in {-1, +1}, x_in != x_init")
    sgn = 1.0 if xin > x0 else -1.0
    for n in (240, 480, 960):
        delays = _uniform_chain_delays(n, a, b, p, xin, x0, sgn)
        if delays.size < 20:
            raise ValueError("too few nodes crossed on the reference chain")
        tail = delays[-10:]
        # certificate: with a contraction factor below 0.95 per node, a last-ten spread under tol/10 bounds the
        # remaining deviation of the limiting delay by about tol/4
        if (tail.max() - tail.min()) <= 0.1 * tol * tail.mean():
            return float(1.0 / tail.mean())
    raise ValueError("the per-node delay did not converge to the requested accuracy on a 960-node chain")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx_in, x_init = 1.0, -1.0\nalpha, B, phi = 1.0, 3.0, 0.0\n",
            "call": "intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "gold_call": "_oracle_intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "tol": 5e-09,
        },
        {
            "setup": "import numpy as np\nx_in, x_init = 1.0, -1.0\nalpha, B, phi = 1.0, 8.0, 0.0\n",
            "call": "intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "gold_call": "_oracle_intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "tol": 5e-09,
        },
        {
            "setup": "import numpy as np\nx_in, x_init = 1.0, -1.0\nalpha, B, phi = 1.0, 8.0, 0.12\n",
            "call": "intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "gold_call": "_oracle_intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "tol": 5e-09,
        },
        {
            "setup": "import numpy as np\nx_in, x_init = 1.0, -1.0\nalpha, B, phi = 1.0, 3.0, 0.2\nx_in, x_init = -1.0, 1.0\n",
            "call": "intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "gold_call": "_oracle_intrinsic_edge_speed(alpha, B, phi, x_in, x_init, 1e-9)",
            "tol": 5e-09,
        },
        {
            "setup": "import numpy as np\nx_in, x_init = 1.0, -1.0\ndef run_model():\n    try:\n        intrinsic_edge_speed(1.0, 3.0, 0.5, 1.0, -1.0, 1e-9)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_intrinsic_edge_speed(1.0, 3.0, 0.5, 1.0, -1.0, 1e-9)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
