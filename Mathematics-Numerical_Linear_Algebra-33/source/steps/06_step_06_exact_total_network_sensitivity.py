"""
Exact total-network communicability edge sensitivity from the block identity.

Exact total-network communicability sensitivity of an edge via the classical block-triangular matrix-function identity. Reference only; not the ratio numerator.

Returns
-------
float, exact total-network sensitivity for edge (i, j)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm

def exact_total_network_sensitivity(A: np.ndarray, i: int, j: int) -> float:
    """Exact total-network communicability sensitivity for edge (i, j).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    i : int
        Row index (0-based).
    j : int
        Column index (0-based).

    Returns
    -------
    float
        Exact total-network sensitivity.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_exact_total_network_sensitivity(A: np.ndarray, i: int, j: int) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n):
        raise ValueError("invalid inputs")
    M = A.T
    E = np.ones((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    b[j] = 1.0
    big = np.block([[M, E], [np.zeros((n, n)), M]])
    out = expm(big) @ np.concatenate([np.zeros(n), b])
    return float(out[i])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [0.0, 0.0]])\ni, j = 0, 1",
            "call": "exact_total_network_sensitivity(A, i, j)",
            "gold_call": "_oracle_exact_total_network_sensitivity(A, i, j)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\ni, j = 2, 0",
            "call": "exact_total_network_sensitivity(A, i, j)",
            "gold_call": "_oracle_exact_total_network_sensitivity(A, i, j)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.3, 0.0, 0.4, 0.0, 0.2], [0.0, 0.0, 0.9, 0.0, 0.5, 0.0], [0.6, 0.0, 0.0, 1.1, 0.0, 0.3], [0.0, 0.7, 0.0, 0.0, 0.8, 0.0], [0.2, 0.0, 0.5, 0.0, 0.0, 1.4], [0.0, 0.3, 0.0, 0.6, 0.0, 0.0]])\ni, j = 2, 5",
            "call": "exact_total_network_sensitivity(A, i, j)",
            "gold_call": "_oracle_exact_total_network_sensitivity(A, i, j)",
        },
    ]
