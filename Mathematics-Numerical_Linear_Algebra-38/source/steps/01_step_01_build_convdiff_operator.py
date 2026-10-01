"""
Construct the convection-diffusion operator and the normalized starting vector for the deterministic matrix-function computation.

The benchmark uses a sparse two-dimensional convection-diffusion discretization on a uniform square grid. The resulting operator is non-Hermitian and is paired with a normalized starting vector for a matrix-function computation.

Returns
-------
float, Frobenius norm of A as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_convdiff_operator(
    state: dict,
    n: int,
    D: float,
) -> float:
    """Construct the operator and normalized starting vector.

    Parameters
    ----------
    state : dict
        Mutable state for intermediate pipeline data.
    n : int
        Number of interior grid points per dimension.
    D : float
        Diffusion coefficient.

    Returns
    -------
    float
        Frobenius norm of the constructed operator.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_convdiff_operator(
    state: dict,
    n: int,
    D: float,
) -> float:
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    if not (isinstance(n, (int, np.integer)) and n >= 1):
        raise ValueError("n must be a positive integer")
    if not (isinstance(D, (int, float, np.integer, np.floating)) and float(D) > 0.0):
        raise ValueError("D must be positive")

    h = 1.0 / (n + 1)

    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = 2.0
        if i > 0:
            L[i, i - 1] = -1.0
        if i + 1 < n:
            L[i, i + 1] = -1.0

    C = np.zeros((n, n), dtype=float)
    for i in range(n):
        C[i, i] = 1.0
        if i > 0:
            C[i, i - 1] = -1.0

    I = np.eye(n)

    A = (
        float(D) / h**2 * (np.kron(L, I) + np.kron(I, L))
        + 1.0 / h * (np.kron(C, I) + np.kron(I, C.T))
    )

    b = np.ones(n * n, dtype=float)
    b /= np.linalg.norm(b)

    state["A"] = A
    state["b"] = b
    state["n"] = n
    state["N"] = n * n
    state["h"] = h
    state["D"] = float(D)

    return float(np.linalg.norm(A, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
state = {}
n = 3
D = 1e-3
""",
            "call": "build_convdiff_operator(state, n, D)",
            "gold_call": "_oracle_build_convdiff_operator(state, n, D)",
        },
        {
            "setup": """import numpy as np
state = {}
n = 7
D = 1e-3
""",
            "call": "build_convdiff_operator(state, n, D)",
            "gold_call": "_oracle_build_convdiff_operator(state, n, D)",
        },
        {
            "setup": """import numpy as np
state = {}
n = 1
D = 0.5
""",
            "call": "build_convdiff_operator(state, n, D)",
            "gold_call": "_oracle_build_convdiff_operator(state, n, D)",
        },
    ]
