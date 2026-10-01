"""
Advance the NSFD scheme for an American put by one implicit time step. Given the solution at the previous time level, the interior tridiagonal coefficients, the time denominator psi1, the exercise payoff on the grid and the two Dirichlet boundary values at the new level, return the exact solution of the discrete linear complementarity problem on the full grid.

Early exercise turns the implicit equation into an obstacle problem. For a candidate new level v with v_0 = left_value and v_M = right_value, let R_m(v) = lower_m v_{m-1} + centre_m v_m + upper_m v_{m+1} + v_m^{old} / psi1 be the residual of the European implicit equation at interior node m. The American level is the vector whose interior values satisfy, at every m = 1, ..., M-1, the complementarity conditions

R_m(v) <= 0,  v_m >= g_m,  R_m(v) (v_m - g_m) = 0,

where g is the exercise payoff. Because the off-diagonal coefficients are positive and every row sum is negative, the negated system matrix is a strictly diagonally dominant M-matrix, so these conditions have exactly one solution for any obstacle. The returned interior values are that solution to double precision: all three conditions hold to rounding error, not merely to an iteration tolerance.

Returns
-------
np.ndarray, float64 array of shape (M+1,): the new American time level (both Dirichlet values and the exact complementarity solution at the interior nodes)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nsfd_american_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, obstacle: "np.ndarray", left_value: float, right_value: float) -> "np.ndarray":
    """One implicit NSFD step of an American put (exact complementarity solve).

    Parameters
    ----------
    v_prev : np.ndarray
        Solution at the previous time level, shape (M+1,), finite, M >= 2.
    lower, centre, upper : np.ndarray
        Coefficients Bl_m, Bc_m, Br_m at m = 1, ..., M-1, each of shape (M-1,).
    psi1 : float
        Time denominator exp(r dt) - 1, finite and strictly positive.
    obstacle : np.ndarray
        Exercise payoff g on the full grid, shape (M+1,), finite; only the
        interior entries g_1, ..., g_{M-1} enter the complementarity problem.
    left_value : float
        Dirichlet value v_0 at the new time level.
    right_value : float
        Dirichlet value v_M at the new time level.

    Returns
    -------
    v_new : np.ndarray
        Float64 array of shape (M+1,): left_value, the M-1 interior values
        solving the linear complementarity problem, and right_value.

    Raises
    ------
    ValueError
        If v_prev is not one-dimensional with at least 3 finite entries, any
        coefficient array does not have shape (M-1,), obstacle does not have
        shape (M+1,) or is not finite, psi1 is not finite and positive, or a
        boundary value is not finite.
    """
    return v_new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded


def _oracle_nsfd_american_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, obstacle: "np.ndarray", left_value: float, right_value: float) -> "np.ndarray":
    v_prev = np.asarray(v_prev, dtype=float)
    if v_prev.ndim != 1 or v_prev.size < 3 or not np.all(np.isfinite(v_prev)):
        raise ValueError("v_prev must be a finite 1-D array with at least 3 entries.")
    n = v_prev.size - 2
    lower, centre, upper = (np.asarray(a, dtype=float) for a in (lower, centre, upper))
    if lower.shape != (n,) or centre.shape != (n,) or upper.shape != (n,):
        raise ValueError("coefficient arrays must have shape (M-1,).")
    obstacle = np.asarray(obstacle, dtype=float)
    if obstacle.shape != v_prev.shape or not np.all(np.isfinite(obstacle)):
        raise ValueError("obstacle must be a finite array of shape (M+1,).")
    psi1, left_value, right_value = float(psi1), float(left_value), float(right_value)
    if not np.isfinite(psi1) or psi1 <= 0.0:
        raise ValueError("psi1 must be finite and positive.")
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        raise ValueError("boundary values must be finite.")
    b = v_prev[1:-1] / psi1
    b[0] += lower[0] * left_value
    b[-1] += upper[-1] * right_value
    g = obstacle[1:-1]
    a_lo, a_ce, a_up = -lower, -centre, -upper
    exercise = np.zeros(n, dtype=bool)
    # Policy iteration on min(A v - b, v - g) = 0; for an M-matrix it terminates in at most n + 1 solves.
    for _ in range(n + 2):
        bands = np.zeros((3, n))
        bands[0, 1:] = np.where(exercise[:-1], 0.0, a_up[:-1])
        bands[1] = np.where(exercise, 1.0, a_ce)
        bands[2, :-1] = np.where(exercise[1:], 0.0, a_lo[1:])
        v = solve_banded((1, 1), bands, np.where(exercise, g, b))
        residual = a_ce * v - b
        residual[1:] += a_lo[1:] * v[:-1]
        residual[:-1] += a_up[:-1] * v[1:]
        new_exercise = residual > v - g
        if np.array_equal(new_exercise, exercise):
            break
        exercise = new_exercise
    else:
        raise RuntimeError("policy iteration did not terminate.")
    return np.concatenate(([left_value], v, [right_value]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    coeffs = (
        "import numpy as np\n"
        "def coeffs(M, dt, r, sigma):\n"
        "    al = 2.0 * r / sigma ** 2\n"
        "    j = np.arange(1, M, dtype=float)\n"
        "    a1 = (j + 1) / j ** al - j / (j + 1) ** al\n"
        "    a2 = (j + 2) / j ** al - j / (j + 2) ** al\n"
        "    a3 = (j + 2) / (j + 1) ** al - (j + 1) / (j + 2) ** al\n"
        "    q = a2 - a1 - a3\n"
        "    p1 = np.expm1(r * dt)\n"
        "    return (a2 - a1) / q - 1.0, -a2 / q - 1.0 / p1, a1 / q, p1\n"
    )
    return [
        # First American step of a put on 64 intervals (r = 0.1, sigma = 0.3, dt = 1/64) from the put payoff; exercise binds below the strike.
        {
            "setup": coeffs + "s = np.linspace(0.0, 4.0, 65)\ng = np.maximum(1.0 - s, 0.0)\nlo, ce, up, p1 = coeffs(64, 1.0 / 64, 0.1, 0.3)\n",
            "call": "nsfd_american_step(g, lo, ce, up, p1, g, 1.0, 0.0)",
            "gold_call": "_oracle_nsfd_american_step(g.copy(), lo.copy(), ce.copy(), up.copy(), p1, g.copy(), 1.0, 0.0)",
            "tol": 1e-12,
        },
        # A large time step (dt = 0.5) with a strike of 2 on 20 intervals: the exercise region spans many nodes.
        {
            "setup": coeffs + "s = np.linspace(0.0, 5.0, 21)\ng = np.maximum(2.0 - s, 0.0)\nv0 = g + 0.02 * np.exp(-(s - 2.0) ** 2)\nlo, ce, up, p1 = coeffs(20, 0.5, 0.12, 0.2)\n",
            "call": "nsfd_american_step(v0, lo, ce, up, p1, g, 2.0, 0.0)",
            "gold_call": "_oracle_nsfd_american_step(v0.copy(), lo.copy(), ce.copy(), up.copy(), p1, g.copy(), 2.0, 0.0)",
            "tol": 1e-12,
        },
        # Edge: a zero obstacle never binds for a positive previous level, so the step reduces to the European solve.
        {
            "setup": coeffs + "s = np.linspace(0.0, 3.0, 13)\nv0 = np.exp(-s)\ng = np.zeros(13)\nlo, ce, up, p1 = coeffs(12, 0.05, 0.05, 0.4)\n",
            "call": "nsfd_american_step(v0, lo, ce, up, p1, g, 0.9, 0.05)",
            "gold_call": "_oracle_nsfd_american_step(v0.copy(), lo.copy(), ce.copy(), up.copy(), p1, g.copy(), 0.9, 0.05)",
            "tol": 1e-12,
        },
        # Invalid: obstacle of the wrong length.
        {
            "setup": "import numpy as np\ndef code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: nsfd_american_step(np.zeros(5), np.ones(3), -3.0 * np.ones(3), np.ones(3), 0.1, np.zeros(4), 1.0, 0.0))",
            "gold_call": "code(lambda: _oracle_nsfd_american_step(np.zeros(5), np.ones(3), -3.0 * np.ones(3), np.ones(3), 0.1, np.zeros(4), 1.0, 0.0))",
        },
    ]
