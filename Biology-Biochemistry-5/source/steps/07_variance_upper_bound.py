"""
Orchestrate the whole pipeline: build the truncated generator and boundary column, construct the two upper polynomials and the input-bound system, integrate the bounding models for the first and second moments, and return the guaranteed upper bound on the variance of the copy number at time t for an initial copy number of zero. Call the earlier step functions rather than reimplementing any of them.

A guaranteed upper bound on a transient variance certifies the size of intrinsic noise without moment closure or sampling error; its tightness is controlled by the truncation size and the quality of the polynomial input bounds.

Returns
-------
float, the upper bound V_plus(t) on the variance of the copy number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variance_upper_bound(theta: "np.ndarray", n_states: int, t: float) -> float:
    """Orchestrate the whole pipeline: build the truncated generator and boundary column, construct the two upper polynomials and the input-bound system, integrate the bounding models for the first and second moments, and return the guaranteed upper bound on the variance of the copy number at time t for an initial copy number of zero. Call the earlier step functions rather than reimplementing any of them.

    Parameters
    ----------
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3).
    n_states : int
        Number N of truncated states.
    t : float
        Nonnegative evaluation time.

    Returns
    -------
    v_plus : float
        Upper bound on V[X(t)].

    Raises
    ------
    ValueError
        If theta is not three finite positive rates, n_states is not an integer >= 2, or t is negative.
    """
    return v_plus

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


def _check_n(n_states):
    if isinstance(n_states, bool) or int(n_states) != n_states or int(n_states) < 2:
        raise ValueError("n_states must be an integer >= 2")
    return int(n_states)


def _oracle_variance_upper_bound(theta: "np.ndarray", n_states: int, t: float) -> float:
    """ORCHESTRATOR: the guaranteed upper bound V+(t) on V[X(t)] for X(0) = 0."""
    th = _check_theta(theta)
    N = _check_n(n_states)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be a finite nonnegative time")
    L = _oracle_truncated_generator(N, th)
    b = _oracle_boundary_input_column(N, th)
    c1 = _oracle_upper_input_polynomial(th, 1)
    c2 = _oracle_upper_input_polynomial(th, 2)
    M = _oracle_input_bound_ode_matrix(c1, c2)
    y1 = _oracle_bounding_output_at_time(L, b, M, 1, t)
    y2 = _oracle_bounding_output_at_time(L, b, M, 2, t)
    v = _oracle_variance_bracket(float(y1[0]), float(y1[1]), float(y2[0]), float(y2[1]))
    return float(v[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states, t = 20, 20.0\n",
            "call": "variance_upper_bound(theta, n_states, t)",
            "gold_call": "_oracle_variance_upper_bound(theta, n_states, t)",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states, t = 25, 5.0\n",
            "call": "variance_upper_bound(theta, n_states, t)",
            "gold_call": "_oracle_variance_upper_bound(theta, n_states, t)",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([2.0, 0.1, 0.05])\nn_states, t = 8, 2.0\n",
            "call": "variance_upper_bound(theta, n_states, t)",
            "gold_call": "_oracle_variance_upper_bound(theta, n_states, t)",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\ndef run_model():\n    try:\n        variance_upper_bound(theta, 20, -1.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_variance_upper_bound(theta, 20, -1.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
