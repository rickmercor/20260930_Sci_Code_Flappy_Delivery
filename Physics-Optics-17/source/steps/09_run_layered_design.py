"""
Final orchestrator: the fixed differentiable spectral design experiment.

The experiment composes the differentiable spectral objective of the layered

sphere with a fixed number of Adam updates from a given start, then evaluates

the objective and its gradient once more at the final variables. Reporting the

base-ten logarithm of the initial-to-final loss ratio, together with the final

physical parameters and gradient norm, tests whether the electromagnetic

sensitivities remain consistent through the whole design chain.

Returns
-------
ndarray, shape (22,), float64: log10(L_initial/L_final), L_initial, L_final, Euclidean norm of the final u-gradient, nine final physical parameters (core radius, two thicknesses, three real indices, three extinction coefficients), then nine final u entries. Zero steps gives a log reduction of zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_layered_design(
    steps: int,
    learning_rate: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    """Run the whole differentiable design using the earlier step functions.

    Use the preceding spectral loss and its full u-gradient at each old
    state, with zero first and second moments initially. Apply exactly
    steps Adam updates, numbered 1 through steps. Evaluate the loss and
    gradient again at the final updated u. Use the preceding sigmoid bounds
    to report physical parameters p. The benchmark is steps=40,
    learning_rate=0.06 and initial=(0.15,-0.4,0.35,-0.25,0.3,-0.1,
    -0.8,0.2,-0.5). No stopping test, randomization or extra polishing is used.
    Every earlier function contributes through the spectral-loss chain and
    Adam step. The state arrays must be carried between all updates.

    Parameters
    ----------
    steps : int
        Number of updates, from 0 to 60 inclusive.
    learning_rate : float
        Positive learning rate at most 0.06.
    initial : ndarray, shape (9,), float64
        Initial u; entries in [-1,1], with generated states required to stay
        in the spectral-loss domain [-6,6].

    Returns
    -------
    ndarray, shape (22,), float64
        log10(L_initial/L_final), L_initial, L_final, Euclidean norm of the
        final u-gradient, nine final physical parameters (core radius,
        two thicknesses, three real indices, three extinction coefficients),
        then nine final u entries. Zero steps gives a log reduction of zero.

    Raises
    ------
    ValueError
        If steps is not an integer in [0,60], the learning rate is outside
        (0,0.06], or initial has the wrong shape, nonfinite entries or an
        entry outside [-1,1].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _material_design(u):
    lower = np.array([35, 12, 15, 2.8, 1.35, 2.1, 0, 0.015, 0.005])
    upper = np.array([85, 48, 65, 4.2, 2.15, 3.3, 0.07, 0.16, 0.08])
    s = 1 / (1 + np.exp(-u))
    params = lower + (upper - lower) * s
    jac = np.diag((upper - lower) * s * (1 - s))
    radii = np.cumsum(params[:3])
    dr = np.cumsum(jac[:3], axis=0)
    index = params[3:6] + 1j * params[6:9]
    dm = jac[3:6] + 1j * jac[6:9]
    return params, radii, index, dr, dm


def _oracle_run_layered_design(
    steps: int,
    learning_rate: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    if (
        not isinstance(steps, (int, np.integer))
        or not 0 <= steps <= 60
        or not 0 < learning_rate <= 0.06
    ):
        raise ValueError("invalid update count or learning rate")
    initial = np.asarray(initial, dtype=float)
    if (
        initial.shape != (9,)
        or not np.all(np.isfinite(initial))
        or np.any(abs(initial) > 1)
    ):
        raise ValueError("initial must have nine finite entries in [-1, 1]")
    state = np.array([initial, np.zeros(9), np.zeros(9)])
    initial_loss = _oracle_spectral_loss_gradient(state[0])[0]
    for step in range(1, steps + 1):
        data = _oracle_spectral_loss_gradient(state[0])
        state = _oracle_adam_design_step(state, data[1:], step, learning_rate)
    final = _oracle_spectral_loss_gradient(state[0])
    params = _material_design(state[0])[0]
    return np.r_[
        np.log10(initial_loss / final[0]),
        initial_loss,
        final[0],
        np.linalg.norm(final[1:]),
        params,
        state[0],
    ]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "u = np.array([0.15, -0.4, 0.35, -0.25, 0.3, -0.1, -0"
                ".8, 0.2, -0.5])\n"
            ),
            "call": ("run_layered_design(1, 0.06, u.copy())\n"),
            "gold_call": ("_oracle_run_layered_design(1, 0.06, u.copy())\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "u = np.array([0.15, -0.4, 0.35, -0.25, 0.3, -0.1, -0"
                ".8, 0.2, -0.5])\n"
            ),
            "call": ("run_layered_design(0, 0.06, u.copy())\n"),
            "gold_call": ("_oracle_run_layered_design(0, 0.06, u.copy())\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n" "\n" "u = np.linspace(-0.5, 0.5, 9)\n"
            ),
            "call": ("run_layered_design(4, 0.04, u.copy())\n"),
            "gold_call": ("_oracle_run_layered_design(4, 0.04, u.copy())\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def raises(fn):\n"
                "    try:\n"
                "        fn(-1, 0.06, np.zeros(9))\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(run_layered_design)\n"),
            "gold_call": ("raises(_oracle_run_layered_design)\n"),
            "tol": 0.0,
        },
    ]
