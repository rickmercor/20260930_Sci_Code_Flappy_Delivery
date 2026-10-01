"""
Chains the lift construction, state matrix, conditional mean, cross moment, projection slope, constrained slope, and one-step simulation of the preceding steps into the complete lifted-Heston march at the requested configuration, and returns the accumulated variance increment over the horizon.

End-to-end march and result extraction.

Running the lift construction, the state matrix, the conditional mean, the cross moment, the projection slope, the constrained slope, and the one-step simulation in sequence for every step of the horizon, carrying the factor state forward and accumulating the realized variance increment each time, is exactly the same computation as the fused march expressed one stage at a time instead of as a single loop. The draws are pinned: one generator seeded once produces all the standard normal and uniform variates up front, and step k consumes the k-th of each, so the whole path is a deterministic function of the configuration. Because the march is fully deterministic given its inputs, a single run at the pinned configuration produces the accumulated increment with no post-processing beyond summing the per-step realizations. Running the whole pipeline with no arguments reproduces the benchmark configuration exactly, which is what makes the accumulated increment a fixed, reproducible number rather than one more input a caller has to supply.

Returns
-------
X_T : float -- the accumulated variance increment over the horizon, the sum of the per-step realized increments (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_clp_march(H: float = 0.3, N: int = 5, lam: float = 0.25, nu: float = 0.1, v0: float = 0.02, theta: float = 0.5, T: float = 5.0, M: int = 10, seed: int = 12345, n_grid: int = 4001) -> float:
    """Run the full lifted-Heston march and return the accumulated variance increment.

    Parameters
    ----------
    H : float
        Hurst exponent of the rough-volatility kernel (must satisfy
        0 < H < 0.5).
    N : int
        Number of lift factors (must be an integer >= 1).
    lam : float
        Mean-reversion speed of the variance level (must be > 0).
    nu : float
        Volatility-of-variance coefficient (must be > 0).
    v0 : float
        Initial variance level (must be > 0).
    theta : float
        Long-run variance level (must be > 0).
    T : float
        Length of the horizon (must be > 0).
    M : int
        Number of time steps in the march (must be an integer >= 1).
    seed : int
        Seed of the generator that produces the pinned draws (must be an
        integer).
    n_grid : int
        Number of uniform grid points used by the fixed-step integrator on
        each step (must be an integer >= 2).

    Returns
    -------
    X_T : float
        The accumulated variance increment over the horizon, the sum of the
        per-step realized increments.

    Raises
    ------
    ValueError
        If M is not an integer >= 1, if seed is not an integer, if T is not
        > 0, or if n_grid is not an integer >= 2.
    """
    return X_T

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _g0_t6_05(t, omega, x, lam, v0, theta):
    """Initial variance curve evaluated at t."""
    return float(v0 + lam * theta * np.sum(omega / x * (1.0 - np.exp(-x * t))))


def _oracle_run_clp_march(H: float = 0.3, N: int = 5, lam: float = 0.25, nu: float = 0.1, v0: float = 0.02, theta: float = 0.5, T: float = 5.0, M: int = 10, seed: int = 12345, n_grid: int = 4001) -> float:
    """Reference implementation of run_clp_march: chains
    _oracle_lift_parameters, _oracle_state_matrix, _oracle_conditional_mean,
    _oracle_cross_moment, _oracle_projection_slope, _oracle_constrained_slope,
    and _oracle_simulate_step step by step over the horizon, accumulating the
    realized variance increment."""
    if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or int(M) < 1:
        raise ValueError("M must be an integer >= 1")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(T, (int, float, np.integer, np.floating)) or isinstance(T, bool) or not (float(T) > 0.0):
        raise ValueError("T must be a real number > 0")
    if not isinstance(n_grid, (int, np.integer)) or isinstance(n_grid, bool) or int(n_grid) < 2:
        raise ValueError("n_grid must be an integer >= 2")
    M = int(M)
    seed = int(seed)
    T = float(T)
    n_grid = int(n_grid)

    omega, x = _oracle_lift_parameters(H, N)
    A = _oracle_state_matrix(lam, omega, x)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(M)
    u = rng.random(M)

    dt = T / M
    U = np.zeros(N)
    X_T = 0.0
    for k in range(M):
        s = k * dt
        t = (k + 1) * dt
        alpha, mu = _oracle_conditional_mean(lam, v0, theta, omega, x, A, s, t, U, n_grid)
        EXZ, kappa = _oracle_cross_moment(lam, nu, v0, theta, omega, x, A, s, t, U, n_grid)
        beta, betaL, C0, c, _feasible = _oracle_projection_slope(lam, nu, v0, theta, omega, x, U, s, t, mu, alpha, kappa, EXZ)
        beta_tilde = _oracle_constrained_slope(nu, omega, alpha, c, beta, betaL, C0)
        g0_t = _g0_t6_05(t, omega, x, lam, v0, theta)
        Xhat, Zhat, U_next, V_next = _oracle_simulate_step(alpha, beta_tilde, mu, kappa, EXZ, x, omega, lam, nu, U, float(z[k]), float(u[k]), g0_t)
        U = U_next
        X_T += Xhat
    return float(X_T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 4 cases: the pinned benchmark configuration, a reduced two-step
    horizon, a single-step boundary, and two invalid-input edges.
    """
    shared_setup = "import numpy as np\n"
    guard_def = (
        "def _guard(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    except Exception:\n"
        "        return 1"
    )
    return [
        {
            # benchmark configuration, zero required args: the pinned march
            "setup": shared_setup,
            "call": "float(run_clp_march())",
            "gold_call": "float(_oracle_run_clp_march())",
        },
        {
            # reduced horizon: two steps of length 0.5
            "setup": shared_setup,
            "call": "float(run_clp_march(M=2, T=1.0))",
            "gold_call": "float(_oracle_run_clp_march(M=2, T=1.0))",
        },
        {
            # boundary: a single step over the shortest admissible horizon
            "setup": shared_setup,
            "call": "float(run_clp_march(M=1, T=0.5))",
            "gold_call": "float(_oracle_run_clp_march(M=1, T=0.5))",
        },
        {
            # edge: a non-positive step count is invalid
            "setup": shared_setup + "\n" + guard_def,
            "call": "_guard(lambda: run_clp_march(M=0))",
            "gold_call": "_guard(lambda: _oracle_run_clp_march(M=0))",
        },
        {
            # edge: a non-integer seed is invalid
            "setup": shared_setup + "\n" + guard_def,
            "call": "_guard(lambda: run_clp_march(seed=1.5))",
            "gold_call": "_guard(lambda: _oracle_run_clp_march(seed=1.5))",
        },
    ]
