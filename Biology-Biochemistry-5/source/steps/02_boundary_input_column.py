"""
Return the input column through which the boundary conditional moment enters the truncated backward equation. Exactly one truncated state exchanges probability with the boundary in this network; the column carries that state's rate into the boundary and zero elsewhere.

Truncating the state space leaves the conditional expectations at the boundary unknown. The source treats them as an external input to a linear system, so the coupling between the truncated states and the boundary has to be written down explicitly.

Returns
-------
ndarray of float64, shape (N,), the boundary input column.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_input_column(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Return the input column through which the boundary conditional moment enters the truncated backward equation. Exactly one truncated state exchanges probability with the boundary in this network; the column carries that state's rate into the boundary and zero elsewhere.

    Parameters
    ----------
    n_states : int
        Number N of truncated states.
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3).

    Returns
    -------
    b : np.ndarray
        Boundary input column of shape (N,).

    Raises
    ------
    ValueError
        If n_states is not an integer >= 2, or theta is not three finite positive rates.
    """
    return b

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb
from scipy.linalg import expm


def _check_theta(theta):
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (3,):
        raise ValueError("theta must be the three rate constants (theta_1, theta_2, theta_3)")
    if not np.all(np.isfinite(th)) or np.any(th <= 0.0):
        raise ValueError("every rate constant must be finite and strictly positive")
    return th


def _oracle_boundary_input_column(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Eq. (21): the column through which the boundary conditional moment drives q."""
    if isinstance(n_states, bool) or int(n_states) != n_states or int(n_states) < 2:
        raise ValueError("n_states must be an integer >= 2")
    N = int(n_states)
    th = _check_theta(theta)
    b = np.zeros(N, dtype=np.float64)
    # only R1 from x = N-1 reaches the boundary state x = N, with rate theta_1
    b[N - 1] = th[0]
    return b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states = 20\n",
            "call": "np.asarray(boundary_input_column(n_states, theta))",
            "gold_call": "np.asarray(_oracle_boundary_input_column(n_states, theta))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([2.0, 0.1, 0.05])\nn_states = 5\n",
            "call": "np.asarray(boundary_input_column(n_states, theta))",
            "gold_call": "np.asarray(_oracle_boundary_input_column(n_states, theta))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([12.0, 0.5, 0.3])\nn_states = 2\n",
            "call": "np.asarray(boundary_input_column(n_states, theta))",
            "gold_call": "np.asarray(_oracle_boundary_input_column(n_states, theta))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\ndef run_model():\n    try:\n        boundary_input_column(0, theta)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_boundary_input_column(0, theta)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
