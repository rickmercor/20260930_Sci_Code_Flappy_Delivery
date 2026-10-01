"""
Continue the dimer trajectory and determine the earliest time after a supplied starting time at which the bond correlation reaches a target value.

The prediction stage uses the same equal-time bond correlation as the calibration,

$$

C(t)=2N^1(t)+[M^z(t)]^2-[N^z(t)]^2.

$$

For a supplied target $C_{\mathrm{target}}$, the crossing condition is

$$

C(t)-C_{\mathrm{target}}=0.

$$

The required result is the smallest root satisfying $t>t_{\mathrm{start}}$ within the supplied search interval.

Returns
-------
float, the smallest $t>t_{\mathrm{start}}$ satisfying $C(t)=C_{\mathrm{target}}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_first_correlation_crossing(
    J: float,
    eta: float,
    target: float,
    t_start: float,
    t_end: float,
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> float:
    """Find the first post-start crossing of the bond-correlation target.

    Parameters
    ----------
    J : float
        Heisenberg exchange constant.
    eta : float
        Correlation-level damping parameter.
    target : float
        Target value of the equal-time bond correlation.
    t_start : float
        Lower search boundary. The returned root must satisfy
        $t>t_{\mathrm{start}}$.
    t_end : float
        Upper search boundary.
    state0 : np.ndarray
        Length-4 initial state ordered as $[N^1,N^2,N^z,M^z]$ at $t=0$.
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    crossing_time : float
        Smallest time $t>t_{\mathrm{start}}$ satisfying
        $C(t)=\mathrm{target}$.

    Raises
    ------
    ValueError
        If no crossing occurs in the requested interval.
    """
    return crossing_time

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_find_first_correlation_crossing(
    J: float,
    eta: float,
    target: float,
    t_start: float,
    t_end: float,
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> float:
    """Reference implementation."""
    lower = np.nextafter(
        float(t_start),
        float(t_end),
    )

    n_grid = max(
        1001,
        int(
            np.ceil(
                (float(t_end) - lower) * 500.0
            )
        ) + 1,
    )

    grid = np.linspace(
        lower,
        float(t_end),
        n_grid,
    )

    trajectory = _oracle_integrate_dimer(
        grid,
        state0,
        J,
        eta,
        S,
        hbar,
    )

    values = np.array(
        [
            _oracle_compute_bond_quantities(
                row,
                J,
            )[0] - target
            for row in trajectory
        ],
        dtype=float,
    )

    bracket = None

    for i in range(grid.size - 1):
        if values[i] * values[i + 1] <= 0.0:
            bracket = (
                float(grid[i]),
                float(grid[i + 1]),
            )
            break

    if bracket is None:
        raise ValueError(
            "No correlation crossing found in the requested interval"
        )

    def _root_value(t):
        state = _oracle_integrate_dimer(
            np.array([float(t)], dtype=float),
            state0,
            J,
            eta,
            S,
            hbar,
        )[0]

        return float(
            _oracle_compute_bond_quantities(
                state,
                J,
            )[0] - target
        )

    return float(
        brentq(
            _root_value,
            bracket[0],
            bracket[1],
            xtol=1e-13,
            rtol=1e-13,
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "round(find_first_correlation_crossing(0.83, 0.06, -3.6, 2.8, 8.0, state0.copy(), 1.5, 1.0), 9)",
            "gold_call": "round(_oracle_find_first_correlation_crossing(0.83, 0.06, -3.6, 2.8, 8.0, state0.copy(), 1.5, 1.0), 9)",
        },
        {
            "setup": """import numpy as np
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "round(find_first_correlation_crossing(0.83, 0.06, -3.53, 2.8, 4.0, state0.copy(), 1.5, 1.0), 9)",
            "gold_call": "round(_oracle_find_first_correlation_crossing(0.83, 0.06, -3.53, 2.8, 4.0, state0.copy(), 1.5, 1.0), 9)",
        },
        {
            "setup": """import numpy as np
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "round(find_first_correlation_crossing(1.8, 0.02, -3.5, 2.8, 8.0, state0.copy(), 1.5, 1.0), 9)",
            "gold_call": "round(_oracle_find_first_correlation_crossing(1.8, 0.02, -3.5, 2.8, 8.0, state0.copy(), 1.5, 1.0), 9)",
        },
    ]
