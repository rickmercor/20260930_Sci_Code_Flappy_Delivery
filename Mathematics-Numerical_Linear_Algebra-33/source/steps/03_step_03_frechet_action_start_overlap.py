"""
Return the overlap of the approximate Frechet action with the starting vecto

Approximate L_exp(A, E)b by the structure-preserving compressed path and return the scalar overlap v^T b. Do not return ||v||_2 or an entry-sum.

Returns
-------
float, overlap v^T b of the approximate Frechet action
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm


def frechet_action_start_overlap(
    A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int
) -> float:
    """Overlap of the approximate Frechet action with the start vector.

    Parameters
    ----------
    A : np.ndarray
        Square matrix.
    E : np.ndarray
        Direction matrix.
    b : np.ndarray
        Starting / action vector.
    k : int
        Depth.

    Returns
    -------
    float
        v^T b where v approximates L_exp(A, E)b.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_frechet_action_start_overlap(
    A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int
) -> float:
    import numpy as np
    from scipy.linalg import expm

    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    k = int(k)
    n = A.shape[0]
    if A.shape != (n, n) or E.shape != (n, n) or b.shape != (n,) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    X = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k + 1), dtype=float)
    AX = np.zeros((n, k), dtype=float)
    EY = np.zeros((n, k + 1), dtype=float)
    R = np.zeros((k, k + 1), dtype=float)
    Hbot = np.zeros((k + 1, k), dtype=float)
    bn = np.linalg.norm(b)
    if bn < 1e-15:
        raise ValueError("b must be nonzero")
    Y[:, 0] = b / bn

    for it in range(k):
        wE = E @ Y[:, it]
        EY[:, it] = wE
        wX = AX[:, :it] @ R[:it, it] + wE
        wY = A @ Y[:, it]
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
        AX[:, it] = A @ X[:, it]
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
    return float(v @ b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [0.5, 0.0]])\nE = np.eye(2)\nb = np.array([1.0, 0.0])\nk = 1",
            "call": "frechet_action_start_overlap(A, E, b, k)",
            "gold_call": "_oracle_frechet_action_start_overlap(A, E, b, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 2.0, 0.5], [0.1, 0.0, 1.0], [0.0, 1.5, 0.0]])\nE = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])\nb = np.array([0.0, 1.0, 0.0])\nk = 2",
            "call": "frechet_action_start_overlap(A, E, b, k)",
            "gold_call": "_oracle_frechet_action_start_overlap(A, E, b, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [1.0, 0.0]])\nE = np.ones((2, 2))\nb = np.array([1.0, 0.0])\nk = 1",
            "call": "frechet_action_start_overlap(A, E, b, k)",
            "gold_call": "_oracle_frechet_action_start_overlap(A, E, b, k)",
        },
    ]
