"""
Propagate the four semiclassical correlation variables from the initial state to specified observation times.

The nonlinear dissipative correlation equations define an initial-value problem for

$$

\mathbf y(t)=\left(N^1,N^2,N^z,M^z\right).

$$

For fixed $J$, $\eta$, $S$, and $\hbar$, the trajectory satisfies

$$

\dot{\mathbf y}=\mathbf f(\mathbf y;J,\eta,S,\hbar).

$$

The state must be evaluated accurately at specified times because later calibration compares different components of the trajectory and the reconstructed bond correlation with time-resolved measurements.

Returns
-------
np.ndarray of shape (len(times), 4), containing the dimer state at each requested time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_dimer(
    times: "np.ndarray",
    state0: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Integrate the dimer state from $t=0$ to the requested times.

    Parameters
    ----------
    times : np.ndarray
        Nonempty one-dimensional nondecreasing array of nonnegative
        evaluation times.
    state0 : np.ndarray
        Length-4 initial state ordered as $[N^1,N^2,N^z,M^z]$.
    J : float
        Heisenberg exchange constant.
    eta : float
        Correlation-level damping parameter.
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape $(\mathrm{len}(times),4)$ containing the state
        at each requested time.
    """
    return trajectory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_integrate_dimer(
    times: "np.ndarray",
    state0: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Reference implementation."""
    times = np.asarray(times, dtype=float)
    state0 = np.asarray(state0, dtype=float)

    t_max = float(times[-1])

    if t_max == 0.0:
        return np.repeat(
            state0[None, :],
            times.size,
            axis=0,
        ).astype(float)

    unique_times, inverse = np.unique(times, return_inverse=True)

    solution = solve_ivp(
        lambda t, y: _oracle_compute_dimer_rhs(
            t,
            y,
            J,
            eta,
            S,
            hbar,
        ),
        (0.0, t_max),
        state0,
        t_eval=unique_times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y.T[inverse].astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    cases = [
        {
            "setup": """import numpy as np
times = np.array([0.70, 1.35, 2.40, 2.80], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(integrate_dimer(times.copy(), state0.copy(), 0.83, 0.06, 1.5, 1.0), 9)",
            "gold_call": "np.round(_oracle_integrate_dimer(times.copy(), state0.copy(), 0.83, 0.06, 1.5, 1.0), 9)",
        },
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.25, 0.50], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(integrate_dimer(times.copy(), state0.copy(), 0.83, 0.0, 1.5, 1.0), 9)",
            "gold_call": "np.round(_oracle_integrate_dimer(times.copy(), state0.copy(), 0.83, 0.0, 1.5, 1.0), 9)",
        },
        {
            "setup": """import numpy as np
times = np.array([0.0, 0.8, 1.6], dtype=float)
state0 = np.array([0.2, -0.1, 0.7, 0.3], dtype=float)
""",
            "call": "np.round(integrate_dimer(times.copy(), state0.copy(), 0.0, 0.10, 1.0, 1.0), 9)",
            "gold_call": "np.round(_oracle_integrate_dimer(times.copy(), state0.copy(), 0.0, 0.10, 1.0, 1.0), 9)",
        },
    ]

    cases.extend([{'setup': 'import numpy as np\ntimes = np.array([0.0, 0.0, 0.0], dtype=float)\nstate0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)\n', 'call': 'np.round(integrate_dimer(times.copy(), state0.copy(), 0.83, 0.06, 1.5, 1.0), 9)', 'gold_call': 'np.round(_oracle_integrate_dimer(times.copy(), state0.copy(), 0.83, 0.06, 1.5, 1.0), 9)'}, {'setup': 'import numpy as np\ntimes = np.array([0.0, 0.70, 0.70, 1.35], dtype=float)\nstate0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)\n', 'call': 'np.round(integrate_dimer(times.copy(), state0.copy(), 0.83, 0.06, 1.5, 1.0), 9)', 'gold_call': 'np.round(_oracle_integrate_dimer(times.copy(), state0.copy(), 0.83, 0.06, 1.5, 1.0), 9)'}])
    return cases
