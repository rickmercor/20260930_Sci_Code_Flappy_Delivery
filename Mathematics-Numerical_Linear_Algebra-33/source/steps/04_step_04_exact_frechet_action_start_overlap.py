"""
Return the overlap of the exact Frechet action with the starting vector

Using the classical block-triangular embedding identity, form the exact Frechet action and return its overlap with b. Do not return a Euclidean norm.

Returns
-------
float, overlap of the exact Frechet action with b
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm

def exact_frechet_action_start_overlap(
    A: np.ndarray, E: np.ndarray, b: np.ndarray
) -> float:
    """Overlap of the exact Frechet action with the start vector.

    Parameters
    ----------
    A : np.ndarray
        Square matrix.
    E : np.ndarray
        Direction matrix.
    b : np.ndarray
        Starting / action vector.

    Returns
    -------
    float
        v_ex^T b for the exact Frechet action.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_exact_frechet_action_start_overlap(
    A: np.ndarray, E: np.ndarray, b: np.ndarray
) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.shape[0]
    if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,):
        raise ValueError("incompatible shapes")
    big = np.block([[A, E], [np.zeros((n, n)), A]])
    v = (expm(big) @ np.concatenate([np.zeros(n), b]))[:n]
    return float(v @ b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.5, 0.0], [0.0, 1.0]])\nE = np.array([[0.0, 1.0], [0.0, 0.0]])\nb = np.array([1.0, 0.0])",
            "call": "exact_frechet_action_start_overlap(A, E, b)",
            "gold_call": "_oracle_exact_frechet_action_start_overlap(A, E, b)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [0.5, 0.0]])\nE = np.eye(2)\nb = np.array([1.0, 0.0])",
            "call": "exact_frechet_action_start_overlap(A, E, b)",
            "gold_call": "_oracle_exact_frechet_action_start_overlap(A, E, b)",
        },
        {
            "setup": "import numpy as np\nA = np.zeros((2, 2))\nE = np.eye(2)\nb = np.array([1.0, -1.0])",
            "call": "exact_frechet_action_start_overlap(A, E, b)",
            "gold_call": "_oracle_exact_frechet_action_start_overlap(A, E, b)",
        },
    ]
