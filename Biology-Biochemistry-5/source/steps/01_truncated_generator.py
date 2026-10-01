"""
Assemble the generator of the dimerization reaction network of Table I restricted to the truncated state space, namely the copy numbers 0 through N-1. Each row holds the transition rates out of one state into the other truncated states, with the diagonal carrying the total outflow that the source's construction prescribes for a truncated state, so that the matrix is exactly the operator the backward equation runs on.

The Kolmogorov backward equation propagates conditional expectations with the adjoint of the master-equation operator. Restricting it to a finite window of copy numbers turns it into a finite matrix, and how that matrix treats jumps that leave the window is what makes the later bounds valid rather than merely approximate.

Returns
-------
ndarray of float64, shape (N, N), the truncated generator with rows indexed by copy number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def truncated_generator(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Assemble the generator of the dimerization reaction network of Table I restricted to the truncated state space, namely the copy numbers 0 through N-1. Each row holds the transition rates out of one state into the other truncated states, with the diagonal carrying the total outflow that the source's construction prescribes for a truncated state, so that the matrix is exactly the operator the backward equation runs on.

    Parameters
    ----------
    n_states : int
        Number N of truncated states; the window is {0, ..., N-1}.
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3) of Table I.

    Returns
    -------
    L : np.ndarray
        Truncated generator of shape (N, N).

    Raises
    ------
    ValueError
        If n_states is not an integer >= 2, or theta is not three finite positive rates.
    """
    return L

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


def _propensities(x, th):
    """lambda_j(x) for the three reactions of Table I at copy number x."""
    return np.array([th[0], th[1] * x, th[2] * x * (x - 1.0)], dtype=np.float64)


def _oracle_truncated_generator(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Eq. (7) restricted to the truncated state space S = {0, ..., N-1} of eq. (20)."""
    if isinstance(n_states, bool) or int(n_states) != n_states or int(n_states) < 2:
        raise ValueError("n_states must be an integer >= 2")
    N = int(n_states)
    th = _check_theta(theta)
    stoich = (1, -1, -2)
    L = np.zeros((N, N), dtype=np.float64)
    for x in range(N):
        lam = _propensities(float(x), th)
        for lj, s in zip(lam, stoich):
            # the diagonal carries the outflow of every reaction, including the R1 jump
            # from x = N-1 that leaves S: that lost mass is exactly the boundary input
            L[x, x] -= lj
            if 0 <= x + s < N:
                L[x, x + s] += lj
    return L

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states = 20\n",
            "call": "np.asarray(truncated_generator(n_states, theta))",
            "gold_call": "np.asarray(_oracle_truncated_generator(n_states, theta))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([2.0, 0.1, 0.05])\nn_states = 4\n",
            "call": "np.asarray(truncated_generator(n_states, theta))",
            "gold_call": "np.asarray(_oracle_truncated_generator(n_states, theta))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([12.0, 0.5, 0.3])\nn_states = 2\n",
            "call": "np.asarray(truncated_generator(n_states, theta))",
            "gold_call": "np.asarray(_oracle_truncated_generator(n_states, theta))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\ndef run_model():\n    try:\n        truncated_generator(1, theta)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_truncated_generator(1, theta)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
