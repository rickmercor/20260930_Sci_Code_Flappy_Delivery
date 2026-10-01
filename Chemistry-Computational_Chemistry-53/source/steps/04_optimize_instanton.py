"""
Implement optimize_instanton, which returns the ring-polymer instanton of the two-state surface:
the folded half ring polymer at which the discretized imaginary-time action is stationary, with
exactly one unstable direction.

In ring-polymer instanton theory the dominant tunnelling pathway at a given inverse temperature is
a stationary point of the discretized Euclidean action of a ring polymer that runs from one well to
the other and back.

Returns
-------
np.ndarray of shape (N/2+1,): half-ring instanton bead positions y_0 ... y_{N/2}, with y_0 on the lower-well side
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimize_instanton(params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    '''Folded half-ring instanton of the lower adiabatic surface.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2}, shape (N/2 + 1,), in the folded convention
        of half_ring_action. They form a stationary point of S_half with respect to all
        N/2 + 1 beads (the end beads are free), converged to max_j |dS_half/dy_j| < 1e-10, at
        which the Hessian of S_half has exactly one negative eigenvalue and the path crosses
        the barrier: y_0 is the end on the side of the lower minimum of
        locate_stationary_points and y_{N/2} the end on the side of the higher minimum. When
        the surface is mirror-symmetric (omega_l == omega_r and eps == 0), the returned
        stationary point is the mirror-symmetric one, y_{N/2-j} = -y_j for every j.

    Raises
    ------
    ValueError
        If params is invalid (as in two_diabat_potential), n_beads is not an even integer
        >= 4, beta is not positive, the surface does not have exactly two minima, or no such
        stationary point is found.
    '''
    return beads

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.optimize import brentq


def _tridiagonal_solve(diag: "np.ndarray", off: "np.ndarray", rhs: "np.ndarray") -> "np.ndarray":
    """Solve the symmetric tridiagonal system with diagonal diag and off-diagonal off."""
    band = np.zeros((3, diag.size))
    band[0, 1:] = off
    band[1] = diag
    band[2, :-1] = off
    return solve_banded((1, 1), band, rhs)


def _inverted_orbit(params: "np.ndarray", start: float, speed: float, times: "np.ndarray", target: float) -> "np.ndarray":
    """Path of x'' = V'(x) (m = 1) leaving start towards target with the given speed, sampled
    at times; samples after the path stops approaching target are placed at target."""
    direction = float(np.sign(target - start))

    def _rhs(_t, u):
        return [u[1], float(_oracle_two_diabat_potential(u[0], params)[1, 0])]

    def _passes_target(_t, u):
        return direction * (u[0] - target)
    _passes_target.terminal = True

    def _turns_back(_t, u):
        return direction * u[1] if _t > 0.0 else 1.0
    _turns_back.terminal = True
    _turns_back.direction = -1

    solution = solve_ivp(_rhs, (0.0, float(times[-1]) + 1.0), [start, direction * speed],
                         method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True,
                         events=(_passes_target, _turns_back))
    path = np.full(times.size, target)
    reached = times <= solution.t[-1]
    path[reached] = solution.sol(times[reached])[0]
    return path


def _oracle_optimize_instanton(params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    tau = _ring_step(beta, n_beads)
    x_low, x_top, x_high, v_low, v_top, v_high = _oracle_locate_stationary_points(params)[:6]
    last = int(n_beads) // 2
    index = np.arange(last + 1)
    symmetric = _is_mirror_symmetric(params)
    free = index[: (last + 1) // 2]
    if symmetric:
        # orbit at the energy of the minima, centred on the middle of the half ring
        right = index >= 0.5 * last
        y = np.empty(last + 1)
        y[right] = _inverted_orbit(params, 0.0, np.sqrt(2.0 * (v_top - v_high)),
                                   (index[right] - 0.5 * last) * tau, x_high)
        y[free] = -y[last - free]
    else:
        # orbit at the energy of the higher minimum, starting at rest at its turning point
        # inside the lower well
        x_turn = brentq(lambda z: float(_oracle_two_diabat_potential(z, params)[0, 0]) - v_high,
                        x_low, x_top, xtol=1e-15, rtol=1e-15, maxiter=500)
        y = _inverted_orbit(params, x_turn, 0.0, index * tau, x_high)
    gtol = max(1e-12, 16.0 * np.finfo(float).eps * float(np.max(np.abs(y))) / tau)
    for _ in range(100):
        if symmetric:
            y[last - free] = -y[free]
            if last % 2 == 0:
                y[last // 2] = 0.0
        _, grad, hess_diag, hess_off = _oracle_half_ring_action(y, params, beta, n_beads)
        converged = float(np.max(np.abs(grad))) < gtol
        if symmetric:
            if converged:
                break
            # Newton step within the mirror-symmetric subspace (the two halves move oppositely)
            diag = hess_diag[: free.size].copy()
            if last % 2:
                diag[-1] -= hess_off[0]
            step = np.zeros(last + 1)
            step[free] = _tridiagonal_solve(diag, hess_off[: free.size - 1], -grad[free])
        else:
            lowest = eigh_tridiagonal(hess_diag, hess_off, eigvals_only=True, select="i",
                                      select_range=(0, 1))
            if converged and lowest[0] < 0.0 < lowest[1]:
                break
            step = _tridiagonal_solve(hess_diag, hess_off, -grad)
        largest = float(np.max(np.abs(step)))
        if largest > 0.2:
            step *= 0.2 / largest
        y = y + step
    else:
        raise ValueError("the instanton search did not converge")
    if not ((y[0] - x_top) * (x_low - x_top) > 0.0 and (y[-1] - x_top) * (x_high - x_top) > 0.0):
        raise ValueError("the stationary point found does not connect the two wells")
    return y

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: the asymmetric model at beta = 300 with N = 1024 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
""",
            "call": "optimize_instanton(params.copy(), 300.0, 1024)",
            "gold_call": "_oracle_optimize_instanton(params.copy(), 300.0, 1024)",
            "tol": 1e-7,
        },
        # --- Typical: the mirror image (x0 < 0); the lower well lies at positive x, so y_0 does too ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, -4.6, -0.19, 1.5])
""",
            "call": "optimize_instanton(params.copy(), 300.0, 512)",
            "gold_call": "_oracle_optimize_instanton(params.copy(), 300.0, 512)",
            "tol": 1e-7,
        },
        # --- Boundary: coarse ring (N = 256 at beta = 300), only a few beads in the lower well ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
""",
            "call": "optimize_instanton(params.copy(), 300.0, 256)",
            "gold_call": "_oracle_optimize_instanton(params.copy(), 300.0, 256)",
            "tol": 1e-7,
        },
        # --- Edge: the wide right well is the deeper one, at a higher temperature (beta = 200) ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -1.5, 1.5])
""",
            "call": "optimize_instanton(params.copy(), 200.0, 512)",
            "gold_call": "_oracle_optimize_instanton(params.copy(), 200.0, 512)",
            "tol": 1e-7,
        },
        # --- Edge: mirror-symmetric surface with degenerate wells ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
""",
            "call": "optimize_instanton(params.copy(), 300.0, 512)",
            "gold_call": "_oracle_optimize_instanton(params.copy(), 300.0, 512)",
            "tol": 1e-7,
        },
        # --- Invalid: odd number of ring beads ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
def run_model():
    try:
        optimize_instanton(params.copy(), 300.0, 1023)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_optimize_instanton(params.copy(), 300.0, 1023)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
