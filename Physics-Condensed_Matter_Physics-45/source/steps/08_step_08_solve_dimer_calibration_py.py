"""
Calibrate $J$ and $\eta$ from the four measurements and return the first post-calibration crossing of the supplied bond-correlation target.

The complete calculation first determines the bounded global minimum of the four-measurement calibration objective. The fitted exchange constant $J_*$ and damping parameter $\eta_*$ are then used to continue the same dissipative dimer trajectory.

The final result is the smallest time $t>t_{\mathrm{start}}$ satisfying

$$

C(t)=C_{\mathrm{target}}.

$$

This orchestrator combines the complete correlation-dynamics, calibration, and prediction workflow without redefining the calculations performed by the preceding subproblems.

Returns
-------
float, the first post-calibration bond-correlation crossing time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_dimer_calibration(
    S: float,
    hbar: float,
    state0: "np.ndarray",
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    J_bounds: tuple[float, float],
    eta_bounds: tuple[float, float],
    target: float,
    t_start: float,
    t_end: float,
) -> float:
    """Calibrate the dimer and return the first post-calibration crossing.

    Parameters
    ----------
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant.
    state0 : np.ndarray
        Length-4 initial state ordered as $[N^1,N^2,N^z,M^z]$.
    observation_times : np.ndarray
        Length-4 calibration times.
    observations : np.ndarray
        Length-4 measurements ordered as $N^z$, $N^2$, $N^z$, and $C$.
    J_bounds : tuple[float, float]
        Lower and upper bounds for $J$.
    eta_bounds : tuple[float, float]
        Lower and upper bounds for $\eta$.
    target : float
        Target equal-time bond correlation.
    t_start : float
        Lower boundary for the post-calibration crossing search.
    t_end : float
        Upper boundary for the crossing search.

    Returns
    -------
    crossing_time : float
        Earliest time $t>t_{\mathrm{start}}$ at which the calibrated
        trajectory satisfies $C(t)=\mathrm{target}$.
    """
    return crossing_time

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_dimer_calibration(
    S: float,
    hbar: float,
    state0: "np.ndarray",
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    J_bounds: tuple[float, float],
    eta_bounds: tuple[float, float],
    target: float,
    t_start: float,
    t_end: float,
) -> float:
    """Reference implementation."""
    fit = _oracle_fit_dimer_parameters(
        J_bounds,
        eta_bounds,
        observation_times,
        observations,
        state0,
        S,
        hbar,
    )

    J_fit = float(fit[0])
    eta_fit = float(fit[1])

    crossing_time = _oracle_find_first_correlation_crossing(
        J_fit,
        eta_fit,
        target,
        t_start,
        t_end,
        state0,
        S,
        hbar,
    )

    return float(crossing_time)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
times = np.array([0.70, 1.35, 2.40, 2.80], dtype=float)
obs = np.array([0.96653326, 0.89872454, -1.20908983, -3.52539157], dtype=float)
""",
            "call": "round(solve_dimer_calibration(1.5, 1.0, state0.copy(), times.copy(), obs.copy(), (0.4, 1.8), (0.02, 0.15), -3.6, 2.8, 8.0), 8)",
            "gold_call": "round(_oracle_solve_dimer_calibration(1.5, 1.0, state0.copy(), times.copy(), obs.copy(), (0.4, 1.8), (0.02, 0.15), -3.6, 2.8, 8.0), 8)",
        },
        {
            "setup": """import numpy as np
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
times = np.array([0.70, 1.35, 2.40, 2.80], dtype=float)
obs = np.array([1.43413956, 0.54926223, 0.78676773, -3.60211297], dtype=float)
""",
            "call": "round(solve_dimer_calibration(1.5, 1.0, state0.copy(), times.copy(), obs.copy(), (0.4, 1.8), (0.02, 0.15), -3.65, 2.8, 8.0), 8)",
            "gold_call": "round(_oracle_solve_dimer_calibration(1.5, 1.0, state0.copy(), times.copy(), obs.copy(), (0.4, 1.8), (0.02, 0.15), -3.65, 2.8, 8.0), 8)",
        },
        {
            "setup": """import numpy as np
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
times = np.array([0.70, 1.35, 2.40, 2.80], dtype=float)
obs = np.array([-0.67555562, -0.47170711, 1.39434805, -3.35821543], dtype=float)
""",
            "call": "round(solve_dimer_calibration(1.5, 1.0, state0.copy(), times.copy(), obs.copy(), (0.4, 1.8), (0.02, 0.15), -3.5, 2.8, 8.0), 8)",
            "gold_call": "round(_oracle_solve_dimer_calibration(1.5, 1.0, state0.copy(), times.copy(), obs.copy(), (0.4, 1.8), (0.02, 0.15), -3.5, 2.8, 8.0), 8)",
        },
    ]
