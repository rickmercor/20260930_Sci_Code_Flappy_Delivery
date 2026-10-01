"""
Advance the implicit NSFD scheme by one time step. Given the solution on the full asset grid at the previous time level, the interior tridiagonal coefficients, the time denominator psi1 and the two Dirichlet boundary values at the new time level, return the solution on the full grid at the new level.

At every interior node m = 1, ..., M-1 the new values satisfy Bl_m v_{m-1} + Bc_m v_m + Br_m v_{m+1} = -v_m^{old} / psi1. The boundary values v_0 and v_M are known at the new level, so the equations at m = 1 and m = M-1 carry the terms Bl_1 v_0 and Br_{M-1} v_M to the right-hand side, and the remaining (M-1) x (M-1) tridiagonal system is solved for the interior values. Because the negated matrix is an M-matrix, the solve is stable without pivoting.

Returns
-------
np.ndarray, float64 array of shape (M+1,): the new time level including both Dirichlet values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nsfd_time_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, left_value: float, right_value: float) -> "np.ndarray":
    """One implicit NSFD step on the full asset grid.

    Parameters
    ----------
    v_prev : np.ndarray
        Solution at the previous time level, shape (M+1,), finite, M >= 2.
    lower, centre, upper : np.ndarray
        Coefficients Bl_m, Bc_m, Br_m at m = 1, ..., M-1, each of shape (M-1,).
    psi1 : float
        Time denominator exp(r dt) - 1, finite and strictly positive.
    left_value : float
        Dirichlet value v_0 at the new time level.
    right_value : float
        Dirichlet value v_M at the new time level.

    Returns
    -------
    v_new : np.ndarray
        Float64 array of shape (M+1,): left_value, the M-1 interior values
        solving the tridiagonal system, and right_value.

    Raises
    ------
    ValueError
        If v_prev is not one-dimensional with at least 3 finite entries, any
        coefficient array does not have shape (M-1,), psi1 is not finite and
        positive, or a boundary value is not finite.
    """
    return v_new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded


def _oracle_nsfd_time_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, left_value: float, right_value: float) -> "np.ndarray":
    v_prev = np.asarray(v_prev, dtype=float)
    if v_prev.ndim != 1 or v_prev.size < 3 or not np.all(np.isfinite(v_prev)):
        raise ValueError("v_prev must be a finite 1-D array with at least 3 entries.")
    n = v_prev.size - 2
    lower, centre, upper = (np.asarray(a, dtype=float) for a in (lower, centre, upper))
    if lower.shape != (n,) or centre.shape != (n,) or upper.shape != (n,):
        raise ValueError("coefficient arrays must have shape (M-1,).")
    psi1, left_value, right_value = float(psi1), float(left_value), float(right_value)
    if not np.isfinite(psi1) or psi1 <= 0.0:
        raise ValueError("psi1 must be finite and positive.")
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        raise ValueError("boundary values must be finite.")
    rhs = -v_prev[1:-1] / psi1
    rhs[0] -= lower[0] * left_value
    rhs[-1] -= upper[-1] * right_value
    bands = np.zeros((3, n))
    bands[0, 1:] = upper[:-1]
    bands[1] = centre
    bands[2, :-1] = lower[1:]
    interior = solve_banded((1, 1), bands, rhs)
    return np.concatenate(([left_value], interior, [right_value]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # A 64-interval step from the unsmoothed put payoff with synthetic coefficient ramps.
        {
            "setup": "import numpy as np\ns = np.linspace(0.0, 4.0, 65)\nv0 = np.maximum(1.0 - s, 0.0)\nlo = np.linspace(0.02, 30.0, 63)\nce = -np.linspace(1400.0, 1500.0, 63)\nup = np.linspace(1.5, 32.0, 63)\n",
            "call": "nsfd_time_step(v0, lo, ce, up, 0.000390701, 0.99960943, 0.0)",
            "gold_call": "_oracle_nsfd_time_step(v0.copy(), lo.copy(), ce.copy(), up.copy(), 0.000390701, 0.99960943, 0.0)",
            "tol": 1e-12,
        },
        # Non-zero right boundary value with non-uniform coefficients.
        {
            "setup": "import numpy as np\nv0 = np.array([0.0, 0.3, 0.9, 1.4, 2.2, 3.1])\nlo = np.array([0.4, 1.1, 2.0, 3.2])\nce = np.array([-12.0, -14.5, -17.0, -21.0])\nup = np.array([0.9, 1.8, 2.9, 4.1])\n",
            "call": "nsfd_time_step(v0, lo, ce, up, 0.1, 0.0, 3.05)",
            "gold_call": "_oracle_nsfd_time_step(v0.copy(), lo.copy(), ce.copy(), up.copy(), 0.1, 0.0, 3.05)",
            "tol": 1e-12,
        },
        # Boundary: a single interior node, where both boundary terms enter the same equation.
        {
            "setup": "import numpy as np\nv0 = np.array([1.0, 0.5, 0.0])\nlo = np.array([0.2])\nce = np.array([-5.0])\nup = np.array([1.0])\n",
            "call": "nsfd_time_step(v0, lo, ce, up, 0.25, 0.98, 0.0)",
            "gold_call": "_oracle_nsfd_time_step(v0.copy(), lo.copy(), ce.copy(), up.copy(), 0.25, 0.98, 0.0)",
            "tol": 1e-12,
        },
        # Invalid: coefficient arrays of the wrong length.
        {
            "setup": "import numpy as np\ndef code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: nsfd_time_step(np.zeros(5), np.ones(4), -np.ones(4), np.ones(4), 0.1, 1.0, 0.0))",
            "gold_call": "code(lambda: _oracle_nsfd_time_step(np.zeros(5), np.ones(4), -np.ones(4), np.ones(4), 0.1, 1.0, 0.0))",
        },
    ]
