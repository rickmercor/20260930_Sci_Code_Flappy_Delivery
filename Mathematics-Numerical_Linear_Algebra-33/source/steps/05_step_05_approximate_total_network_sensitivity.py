"""
Approximate total-network communicability edge sensitivity via Krylov Frechet action

Approximate the total-network communicability sensitivity of a directed edge using the structure-preserving Krylov Frechet process at the requested depth. Do not substitute the exact block-triangular embedding

Returns
-------
float, approximate total-network sensitivity for edge (i, j)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm

def approximate_total_network_sensitivity(
    A: np.ndarray, i: int, j: int, k: int
) -> float:
    """Approximate total-network communicability sensitivity for edge (i, j).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    i : int
        Row index (0-based).
    j : int
        Column index (0-based).
    k : int
        Krylov depth.

    Returns
    -------
    float
        Approximate total-network sensitivity.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_approximate_total_network_sensitivity(
    A: np.ndarray, i: int, j: int, k: int
) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    k = int(k)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    M = A.T
    E = np.ones((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    b[j] = 1.0

    X = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k + 1), dtype=float)
    AX = np.zeros((n, k), dtype=float)
    EY = np.zeros((n, k + 1), dtype=float)
    R = np.zeros((k, k + 1), dtype=float)
    Hbot = np.zeros((k + 1, k), dtype=float)
    Y[:, 0] = b / np.linalg.norm(b)

    for it in range(k):
        wE = E @ Y[:, it]
        EY[:, it] = wE
        wX = AX[:, :it] @ R[:it, it] + wE
        wY = M @ Y[:, it]
        h = Y[:, : it + 1].T @ wY
        Hbot[: it + 1, it] = h
        wY = wY - Y[:, : it + 1] @ h
        for _ in range(2):
            corr = Y[:, : it + 1].T @ wY
            wY = wY - Y[:, : it + 1] @ corr
        beta = float(np.linalg.norm(wY))
        if beta < 1e-15:
            raise ValueError("lucky breakdown in bottom orthonormalization")
        Hbot[it + 1, it] = beta
        Y[:, it + 1] = wY / beta
        R[:it, it + 1] = -R[:it, : it + 1] @ h / beta
        R[it, it + 1] = 1.0 / beta
        g = X[:, :it].T @ wX
        wX = wX - X[:, :it] @ g
        for _ in range(2):
            corr = X[:, :it].T @ wX
            wX = wX - X[:, :it] @ corr
        alpha = float(np.linalg.norm(wX))
        if alpha < 1e-15:
            raise ValueError("lucky breakdown in top orthonormalization")
        X[:, it] = wX / alpha
        AX[:, it] = M @ X[:, it]
        if it == 0:
            preR = np.array([[alpha]], dtype=float)
        else:
            preR = np.block(
                [
                    [np.eye(it), g.reshape(-1, 1)],
                    [np.zeros((1, it)), np.array([[alpha]])],
                ]
            )
        R[: it + 1, it + 1] = preR @ R[: it + 1, it + 1]

    Yk = Y[:, :-1]
    H = np.block(
        [[X.T @ AX, X.T @ EY[:, :-1]], [np.zeros((k, k)), Hbot[:-1].copy()]]
    )
    U = np.block([[X, np.zeros((n, k))], [np.zeros((n, k)), Yk]])
    rhs = np.zeros(2 * k, dtype=float)
    rhs[k:] = Yk.T @ b
    v = (U @ (expm(H) @ rhs))[:n]
    return float(v[i])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [0.5, 0.0]])\ni, j, k = 0, 1, 1",
            "call": "approximate_total_network_sensitivity(A, i, j, k)",
            "gold_call": "_oracle_approximate_total_network_sensitivity(A, i, j, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.4], [0.2, 0.0, 1.2]])\ni, j, k = 1, 0, 2",
            "call": "approximate_total_network_sensitivity(A, i, j, k)",
            "gold_call": "_oracle_approximate_total_network_sensitivity(A, i, j, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.3, 0.0, 0.4, 0.0, 0.2], [0.0, 0.0, 0.9, 0.0, 0.5, 0.0], [0.6, 0.0, 0.0, 1.1, 0.0, 0.3], [0.0, 0.7, 0.0, 0.0, 0.8, 0.0], [0.2, 0.0, 0.5, 0.0, 0.0, 1.4], [0.0, 0.3, 0.0, 0.6, 0.0, 0.0]])\ni, j, k = 2, 5, 2",
            "call": "approximate_total_network_sensitivity(A, i, j, k)",
            "gold_call": "_oracle_approximate_total_network_sensitivity(A, i, j, k)",
        },
    ]
