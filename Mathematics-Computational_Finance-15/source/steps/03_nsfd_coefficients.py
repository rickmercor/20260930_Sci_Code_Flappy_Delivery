"""
Assemble the coefficients of the implicit non-standard finite-difference (NSFD) scheme for the Black-Scholes equation. Given the number of asset intervals, the time step, the rate and the volatility, return the left, centre and right coefficients of the tridiagonal system at every interior node and the scaled time denominator psi1.

With t the time to maturity, the Black-Scholes equation reads u_t = L u with L u = (sigma^2/2) s^2 u_ss + r s u_s - r u, and the payoff is its initial condition at t = 0. The scheme is the implicit NSFD scheme that combines the exact finite-difference scheme of the Cauchy-Euler equation L u = 0 with the exact finite-difference scheme of the decay equation u_t = -r u, whose time denominator phi(dt) takes the place of dt. At interior node m it uses the weights of step 02 with j = m (a correction of the source's index m - 1, which is singular at m = 1): A1_m goes with v_{m+1}, A2_m with v_m and A3_m with v_{m-1}.

Return the coefficients of the equation at node m divided by r, written as lower_m v_{m-1}^k + centre_m v_m^k + upper_m v_{m+1}^k = -v_m^{k-1} / psi1 with psi1 = r phi(dt). The off-diagonal coefficients are then positive and every row sum is negative, so the negated system matrix is an M-matrix for every step size.

Returns
-------
tuple (lower, centre, upper, psi1): three float64 arrays of shape (M-1,) with the tridiagonal coefficients at the interior nodes m = 1, ..., M-1, and psi1 = r phi(dt) as a Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nsfd_coefficients(n_intervals: int, time_step: float, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """Tridiagonal NSFD coefficients at m = 1, ..., M-1 and psi1.

    Parameters
    ----------
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    time_step : float
        Time step dt, finite and strictly positive.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.

    Returns
    -------
    lower, centre, upper : np.ndarray
        Float64 arrays of shape (M-1,) holding lower_m, centre_m and upper_m
        at the interior nodes m = 1, ..., M-1.
    psi1 : float
        rate times the exact decay-scheme denominator phi(time_step), as a
        Python float.

    Raises
    ------
    ValueError
        If n_intervals is not an integer of at least 2, or time_step, rate or
        volatility is not finite and strictly positive.
    """
    return lower, centre, upper, psi1

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_nsfd_coefficients(n_intervals: int, time_step: float, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    time_step = float(time_step)
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive.")
    a1, a2, a3 = _oracle_exact_difference_weights(n_intervals, rate, volatility)
    psi1 = float(np.expm1(float(rate) * time_step))
    q = a2 - a1 - a3
    upper = a1 / q
    centre = -a2 / q - 1.0 / psi1
    lower = (a2 - a1) / q - 1.0
    return lower, centre, upper, psi1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = "import numpy as np\ndef pack(out):\n    return np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in out])\n"
    return [
        # Benchmark grid: 128 intervals and 128 time steps over half a year.
        {
            "setup": pack,
            "call": "pack(nsfd_coefficients(128, 0.5 / 128, 0.05, 0.3))",
            "gold_call": "pack(_oracle_nsfd_coefficients(128, 0.5 / 128, 0.05, 0.3))",
            "tol": 1e-9,
        },
        # Large time step: psi1 is far from r * dt, so the exact decay denominator matters.
        {
            "setup": pack,
            "call": "pack(nsfd_coefficients(10, 2.0, 0.1, 0.25))",
            "gold_call": "pack(_oracle_nsfd_coefficients(10, 2.0, 0.1, 0.25))",
            "tol": 1e-10,
        },
        # Boundary: a single interior node.
        {
            "setup": pack,
            "call": "pack(nsfd_coefficients(2, 0.01, 0.03, 0.2))",
            "gold_call": "pack(_oracle_nsfd_coefficients(2, 0.01, 0.03, 0.2))",
            "tol": 1e-10,
        },
        # Invalid: non-positive time step.
        {
            "setup": "def code(thunk):\n    try:\n        thunk()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "code(lambda: nsfd_coefficients(8, -0.1, 0.05, 0.3))",
            "gold_call": "code(lambda: _oracle_nsfd_coefficients(8, -0.1, 0.05, 0.3))",
        },
    ]
