"""
Bias-corrected Adam step for the physical design variables.

Gradient-based spectral design updates all unconstrained variables at once

from the full objective gradient. Adam keeps exponential moving averages of the

gradient and of its square, corrects both for their zero initialization and

scales each coordinate by the corrected root-mean-square gradient, so the

geometric and material variables advance at comparable rates despite very

different sensitivities.

Returns
-------
ndarray, shape (3, 9), float64: Updated u, m and v rows.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adam_design_step(
    state: "np.ndarray",
    grad: "np.ndarray",
    step: int,
    learning_rate: float,
) -> "np.ndarray":
    """Return one Adam update of variables and their two moment vectors.

    Set m_new=0.9*m_old+0.1*g and v_new=0.999*v_old+0.001*g**2.
    With iteration t=step, update u_new=u_old-learning_rate*
    [m_new/(1-0.9**t)]/[sqrt(v_new/(1-0.999**t))+1e-8].
    The small constant is outside the square root. There is no weight decay,
    projection, convergence stopping or moment resetting.

    Parameters
    ----------
    state : ndarray, shape (3, 9), float64
        Rows are u, first moment m and nonnegative second moment v.
    grad : ndarray, shape (9,), float64
        Gradient with respect to u at the old state. All data are finite.
    step : int
        One-based iteration number in [1, 60].
    learning_rate : float
        Positive learning rate at most 0.06.

    Returns
    -------
    ndarray, shape (3, 9), float64
        Updated u, m and v rows.

    Raises
    ------
    ValueError
        If step is not an integer in [1,60] or the learning rate is outside
        (0,0.06].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adam_design_step(
    state: "np.ndarray",
    grad: "np.ndarray",
    step: int,
    learning_rate: float,
) -> "np.ndarray":
    if (
        not isinstance(step, (int, np.integer))
        or not 1 <= step <= 60
        or not 0 < learning_rate <= 0.06
    ):
        raise ValueError("invalid Adam iteration or learning rate")
    u, first, second = np.asarray(state)
    first = 0.9 * first + 0.1 * grad
    second = 0.999 * second + 0.001 * grad**2
    correction = (first / (1 - 0.9**step)) / (
        np.sqrt(second / (1 - 0.999**step)) + 1e-8
    )
    return np.array([u - learning_rate * correction, first, second])

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
                "state = np.zeros((3, 9))\n"
                "state[0] = np.linspace(-0.4, 0.5, 9)\n"
                "g = np.array([0.3, -0.4, 0.02, 0.05, -0.11, 0.004, -"
                "0.8, 0.12, -0.3])\n"
            ),
            "call": ("adam_design_step(state.copy(), g.copy(), 1, 0.06)\n"),
            "gold_call": (
                "_oracle_adam_design_step(state.copy(), g.copy(), 1, "
                "0.06)\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "state = np.zeros((3, 9))\n"
                "state[0] = np.linspace(-0.4, 0.5, 9)\n"
                "g = np.array([0.3, -0.4, 0.02, 0.05, -0.11, 0.004, -"
                "0.8, 0.12, -0.3])\n"
                "g *= 0\n"
            ),
            "call": ("adam_design_step(state.copy(), g.copy(), 1, 0.04)\n"),
            "gold_call": (
                "_oracle_adam_design_step(state.copy(), g.copy(), 1, "
                "0.04)\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "state = np.zeros((3, 9))\n"
                "state[0] = np.linspace(-0.4, 0.5, 9)\n"
                "g = np.array([0.3, -0.4, 0.02, 0.05, -0.11, 0.004, -"
                "0.8, 0.12, -0.3])\n"
                "state[1] = 0.1 * g\n"
                "state[2] = 0.003 * g**2\n"
            ),
            "call": ("adam_design_step(state.copy(), g.copy(), 7, 0.045)\n"),
            "gold_call": (
                "_oracle_adam_design_step(state.copy(), g.copy(), 7, "
                "0.045)\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def raises(fn):\n"
                "    try:\n"
                "        fn(np.zeros((3, 9)), np.ones(9), 0, 0.06)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(adam_design_step)\n"),
            "gold_call": ("raises(_oracle_adam_design_step)\n"),
            "tol": 0.0,
        },
    ]
