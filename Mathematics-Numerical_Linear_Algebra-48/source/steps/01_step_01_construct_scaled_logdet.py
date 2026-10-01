"""
Build the scaled algebraic SPSD diagonal spectrum and return its exact regularized log-determinant tr log(A+I).

Applications such as Gaussian-process likelihoods need log det(H + mu I) for large SPSD H. Dividing through by the regularization turns this into a regularized log-determinant of a rescaled matrix. Instances with algebraic spectral decay are the standard stress test, since the tail decays slowly enough that low-rank methods are not trivially exact.

Returns
-------
float, exact tr log(A+I) for A=diag(i^{-2}/mu) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def construct_scaled_logdet(n: int, mu: float) -> float:
    """Build A = diag(i^{-2}/mu) and return tr log(A+I).

    Parameters
    ----------
    n : int
        Matrix order.
    mu : float
        Positive regularization scale used in A = H/mu.

    Returns
    -------
    float
        Exact value of sum_i log(1 + lambda_i(A)).
        
    Raises
    ------
    ValueError
        If ``n < 1``, or if ``mu <= 0``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_construct_scaled_logdet(n: int, mu: float) -> float:
    

    if n < 1:
        raise ValueError("n must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    idx = np.arange(1, n + 1, dtype=float)
    lam = (idx ** (-2)) / mu
    return float(np.sum(np.log1p(lam)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "construct_scaled_logdet(32, 1e-2)",
            "gold_call": "_oracle_construct_scaled_logdet(32, 1e-2)",
        },
        {
            "setup": "import numpy as np",
            "call": "construct_scaled_logdet(1, 1.0)",
            "gold_call": "_oracle_construct_scaled_logdet(1, 1.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "construct_scaled_logdet(8, 1e-4)",
            "gold_call": "_oracle_construct_scaled_logdet(8, 1e-4)",
        },
    ]
