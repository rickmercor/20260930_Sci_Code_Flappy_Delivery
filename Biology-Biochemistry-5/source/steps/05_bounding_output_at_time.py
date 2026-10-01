"""
Integrate the two bounding state-space models for the test function x^alpha and return their outputs [y_plus(t), y_minus(t)] for an initial copy number of zero. The upper system is driven by the input-bound system through the boundary column; use the source's initial conditions for the boundary inputs and its lower input choice.

Because the truncated backward equation is monotone in its boundary input, replacing the unknown input by an upper or lower bound yields a state-space model whose output bounds the true transient moment from above or below. Both models are linear and time-invariant, so they can be integrated exactly.

Returns
-------
ndarray of float64, shape (2,), [y_plus(t), y_minus(t)] bounding E[X(t)^alpha].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bounding_output_at_time(generator: "np.ndarray", boundary_column: "np.ndarray",
                                     u_matrix: "np.ndarray", alpha: int, t: float) -> "np.ndarray":
    """Integrate the two bounding state-space models for the test function x^alpha and return their outputs [y_plus(t), y_minus(t)] for an initial copy number of zero. The upper system is driven by the input-bound system through the boundary column; use the source's initial conditions for the boundary inputs and its lower input choice.

    Parameters
    ----------
    generator : np.ndarray
        Truncated generator of shape (N, N).
    boundary_column : np.ndarray
        Boundary input column of shape (N,).
    u_matrix : np.ndarray
        Augmented-state matrix (3, 3) of the input-bound system.
    alpha : int
        Monomial order of the test function, 1 or 2.
    t : float
        Nonnegative time at which the bounds are evaluated.

    Returns
    -------
    y : np.ndarray
        Array [y_plus, y_minus].

    Raises
    ------
    ValueError
        If the generator is not square of size >= 2, the column or matrix shapes do not match, alpha is not 1 or 2, or t is negative.
    """
    return y

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb
from scipy.linalg import expm


def _oracle_bounding_output_at_time(generator: "np.ndarray", boundary_column: "np.ndarray",
                                     u_matrix: "np.ndarray", alpha: int, t: float) -> "np.ndarray":
    """Eqs. (12), (13): outputs [y+(t), y-(t)] of B+ and B- for f(x) = x^alpha, p_0 = e_0.

    B+ is integrated as one LTI on the augmented state [q, u_1+, u_2+, 1] with
    u_mu+(0) = N^mu (Lemma 2); B- uses u- = 0 so q- evolves under the generator alone.
    """
    L = np.asarray(generator, dtype=np.float64)
    b = np.asarray(boundary_column, dtype=np.float64)
    M = np.asarray(u_matrix, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 2:
        raise ValueError("generator must be a square matrix of size >= 2")
    N = L.shape[0]
    if b.shape != (N,):
        raise ValueError("boundary_column must have one entry per truncated state")
    if M.shape != (3, 3):
        raise ValueError("u_matrix must be 3 x 3")
    if isinstance(alpha, bool) or int(alpha) != alpha or int(alpha) not in (1, 2):
        raise ValueError("alpha must be 1 or 2")
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be a finite nonnegative time")
    alpha = int(alpha)
    f0 = np.arange(N, dtype=np.float64) ** alpha
    A = np.zeros((N + 3, N + 3), dtype=np.float64)
    A[:N, :N] = L
    A[:N, N + alpha - 1] += b            # boundary input theta_1 u_alpha+(t) into row N-1
    A[N:, N:] = M
    z0 = np.concatenate([f0, [float(N), float(N) ** 2, 1.0]])
    y_plus = float((expm(A * float(t)) @ z0)[0])
    y_minus = float((expm(L * float(t)) @ f0)[0])
    return np.array([y_plus, y_minus], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states = 20\nL = _oracle_truncated_generator(n_states, theta)\nb = _oracle_boundary_input_column(n_states, theta)\nM = _oracle_input_bound_ode_matrix(_oracle_upper_input_polynomial(theta, 1), _oracle_upper_input_polynomial(theta, 2))\nalpha, t = 1, 20.0\n",
            "call": "np.asarray(bounding_output_at_time(L, b, M, alpha, t))",
            "gold_call": "np.asarray(_oracle_bounding_output_at_time(L, b, M, alpha, t))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states = 20\nL = _oracle_truncated_generator(n_states, theta)\nb = _oracle_boundary_input_column(n_states, theta)\nM = _oracle_input_bound_ode_matrix(_oracle_upper_input_polynomial(theta, 1), _oracle_upper_input_polynomial(theta, 2))\nalpha, t = 2, 20.0\n",
            "call": "np.asarray(bounding_output_at_time(L, b, M, alpha, t))",
            "gold_call": "np.asarray(_oracle_bounding_output_at_time(L, b, M, alpha, t))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([2.0, 0.1, 0.05])\nn_states = 6\nL = _oracle_truncated_generator(n_states, theta)\nb = _oracle_boundary_input_column(n_states, theta)\nM = _oracle_input_bound_ode_matrix(_oracle_upper_input_polynomial(theta, 1), _oracle_upper_input_polynomial(theta, 2))\nalpha, t = 1, 0.0\n",
            "call": "np.asarray(bounding_output_at_time(L, b, M, alpha, t))",
            "gold_call": "np.asarray(_oracle_bounding_output_at_time(L, b, M, alpha, t))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nn_states = 6\nL = np.array([[-5.0, 5.0, 0.0, 0.0, 0.0, 0.0], [0.03465735902799726, -5.034657359027997, 5.0, 0.0, 0.0, 0.0], [0.04, 0.06931471805599453, -5.109314718055995, 5.0, 0.0, 0.0], [0.0, 0.12, 0.10397207708399178, -5.2239720770839915, 5.0, 0.0], [0.0, 0.0, 0.24, 0.13862943611198905, -5.378629436111989, 5.0], [0.0, 0.0, 0.0, 0.4, 0.17328679513998632, -5.573286795139986]])\nb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])\nM = np.array([[-0.03465735902799726, 0.0, 5.0], [9.954657359027998, 0.010685281944005476, 5.0], [0.0, 0.0, 0.0]])\ndef run_model():\n    try:\n        bounding_output_at_time(L, b, M, 3, 1.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_bounding_output_at_time(L, b, M, 3, 1.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
