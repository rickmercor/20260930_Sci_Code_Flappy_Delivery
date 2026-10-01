"""
Estrada-index edge sensitivity for a directed edge

Estrada-index edge sensitivity for edge (i, j). Retrieve the closed identity from the secondary network-sensitivity literature; do not use a Frechet action or an entry of exp(A).

Returns
-------
float, Estrada-index edge sensitivity for edge (i, j)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm

def estrada_edge_sensitivity(A: np.ndarray, i: int, j: int) -> float:
    """Estrada-index edge sensitivity for edge (i, j).

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
        Estrada-index edge sensitivity.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_estrada_edge_sensitivity(A: np.ndarray, i: int, j: int) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n):
        raise ValueError("invalid inputs")
    return float(expm(A.T)[i, j])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [1.0, 0.0]])\ni, j = 0, 1",
            "call": "estrada_edge_sensitivity(A, i, j)",
            "gold_call": "_oracle_estrada_edge_sensitivity(A, i, j)",
        },
        {
            "setup": "import numpy as np\nA = np.zeros((3, 3))\ni, j = 1, 1",
            "call": "estrada_edge_sensitivity(A, i, j)",
            "gold_call": "_oracle_estrada_edge_sensitivity(A, i, j)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.3, 0.0, 0.4, 0.0, 0.2], [0.0, 0.0, 0.9, 0.0, 0.5, 0.0], [0.6, 0.0, 0.0, 1.1, 0.0, 0.3], [0.0, 0.7, 0.0, 0.0, 0.8, 0.0], [0.2, 0.0, 0.5, 0.0, 0.0, 1.4], [0.0, 0.3, 0.0, 0.6, 0.0, 0.0]])\ni, j = 2, 5",
            "call": "estrada_edge_sensitivity(A, i, j)",
            "gold_call": "_oracle_estrada_edge_sensitivity(A, i, j)",
        },
    ]
