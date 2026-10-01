"""
Advance the concatenated three-grid state while rebuilding every moving-grid quantity at each of four stage times.

For $F(t,\\Phi)=M(t)\\Phi+b(t)$, use stage times $t$, $t+\\Delta t/2$, $t+\\Delta t/2$, and $t+\\Delta t$, with stage states $\\Phi$, $\\Phi+\\Delta t k_1/2$, $\\Phi+\\Delta t k_2/2$, and $\\Phi+\\Delta t k_3$. Combine the four derivatives with weights $(1,2,2,1)/6$.



At every stage, rebuild the moving nodes, discrete metrics, donor rows, affine matrix, and forcing. Initialize each overlapping copy independently from $\\exp[-\\beta(x-x_0)^2]$. Require a positive finite step size and terminal time, an integer number of updates, and a grid-speed magnitude strictly below $|c|$ so the incoming side cannot switch during a run.

Returns
-------
np.ndarray of length counts.sum(), containing the final left-middle-right state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_weak_overset_rk4(
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    dt: float,
    final_time: float,
    beta: float = 80.0,
    x0: float = -0.4,
) -> np.ndarray:
    r"""Advance $\Phi=[u_L^T,u_M^T,u_R^T]^T$ to ``final_time``.

    Parameters
    ----------
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``, each at least eighteen.
    sigmas : np.ndarray
        Finite stretching parameters ``[sigma_L, sigma_M, sigma_R]``, each of
        magnitude below one.
    c : float
        Finite nonzero physical advection speed. Its sign selects the incoming
        side on every component grid.
    amplitude : float
        Middle-grid translation amplitude.
    period : float
        Positive translation period.
    penalties : np.ndarray
        Three penalties, each at least $1/2$.
    dt : float
        Positive RK4 step size dividing ``final_time`` exactly.
    final_time : float
        Positive terminal time.
    beta : float
        Positive Gaussian scaling parameter.
    x0 : float
        Initial Gaussian centre.

    Returns
    -------
    np.ndarray
        Final concatenated state of length ``counts.sum()``.

    Raises
    ------
    ValueError
        If ``dt`` or ``final_time`` is not positive and finite;
        ``final_time`` is not an integer multiple of ``dt``; ``c`` is zero or
        non-finite; ``period`` is not positive; the grid-speed magnitude is not
        below ``abs(c)``; or the integration produces a non-finite state.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_time_interval(dt, final_time):
    if not np.isfinite(dt) or float(dt) <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(final_time) or float(final_time) <= 0.0:
        raise ValueError("final_time must be positive and finite")
    steps = int(round(float(final_time) / float(dt)))
    tolerance = 1e-12 * max(1.0, abs(float(final_time)))
    if steps < 1 or abs(steps * float(dt) - float(final_time)) > tolerance:
        raise ValueError("final_time must be an integer multiple of dt")
    return float(dt), float(final_time), steps


def _oracle_advance_weak_overset_rk4(
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    dt: float,
    final_time: float,
    beta: float = 80.0,
    x0: float = -0.4,
) -> np.ndarray:
    """Reference stage-updated classical RK4 integration."""
    dt, final_time, steps = _validated_time_interval(dt, final_time)
    if not np.isfinite(c) or float(c) == 0.0:
        raise ValueError("c must be finite and nonzero")
    if not np.isfinite(amplitude) or not np.isfinite(period) or float(period) <= 0.0:
        raise ValueError("amplitude must be finite and period must be positive")
    if abs(2.0 * np.pi * float(amplitude) / float(period)) >= abs(float(c)):
        raise ValueError("grid-speed magnitude must remain below abs(c)")
    if not np.isfinite(beta) or float(beta) <= 0.0 or not np.isfinite(x0):
        raise ValueError("beta must be positive and x0 must be finite")

    counts = np.asarray(counts)
    initial_state = _oracle_compute_moving_grid_state(
        0.0, counts, sigmas, amplitude, period
    )
    state = np.exp(-float(beta) * (initial_state[1:] - float(x0)) ** 2)

    def right_hand_side(time, values):
        augmented = _oracle_assemble_weak_overset_system(
            time,
            counts,
            sigmas,
            c,
            amplitude,
            period,
            penalties,
            beta,
            x0,
            jet_order=0,
        )[0]
        return augmented[:, :-1] @ values + augmented[:, -1]

    time = 0.0
    for _ in range(steps):
        k1 = right_hand_side(time, state)
        k2 = right_hand_side(time + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = right_hand_side(time + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = right_hand_side(time + dt, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += dt

    if not np.all(np.isfinite(state)):
        raise ValueError("time integration produced a non-finite state")
    return state

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return static, positive, negative, and nonintegral-step cases."""
    return [
        {
            "setup": """import numpy as np
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -0.15, 0.20])
c, amplitude, period = 1.0, 0.0, 1.0
penalties = np.array([0.75, 0.75, 0.75])
dt, final_time = 0.01, 0.05
""",
            "call": "advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time)",
            "gold_call": "_oracle_advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time)",
        },
        {
            "setup": """import numpy as np
counts = np.array([20, 25, 22])
sigmas = np.array([-0.2, 0.35, 0.1])
c, amplitude, period = 1.0, 0.05, 0.8
penalties = np.array([0.6, 0.7, 0.8])
dt, final_time = 0.005, 0.05
""",
            "call": "advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time)",
            "gold_call": "_oracle_advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time)",
        },
        {
            "setup": """import numpy as np
counts = np.array([20, 25, 22])
sigmas = np.array([-0.2, 0.35, 0.1])
c, amplitude, period = -1.0, 0.05, 0.8
penalties = np.array([0.6, 0.7, 0.8])
dt, final_time = 0.005, 0.05
beta, x0 = 60.0, 0.35
""",
            "call": "advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time, beta, x0)",
            "gold_call": "_oracle_advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time, beta, x0)",
        },
        {
            "setup": """import numpy as np
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -0.15, 0.20])
c, amplitude, period = 1.0, 0.0, 1.0
penalties = np.array([0.75, 0.75, 0.75])
dt, final_time = 0.03, 0.05
def run_model():
    try:
        advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_advance_weak_overset_rk4(counts, sigmas, c, amplitude, period, penalties, dt, final_time)
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
