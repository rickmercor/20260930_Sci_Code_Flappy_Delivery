"""
Return the leading entry of the last column of the separate-orthonormalization coupling factor

From the structure-preserving Frechet Krylov process, return R[0, k] (0-based), the leading entry of the last column of the triangular coupling factor. A global Frobenius norm is not accepted.

Returns
-------
float, R[0, k] from separate top/bottom orthonormalization
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sep_orth_R_lead_entry(A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int) -> float:
    """Leading entry of the last column of the coupling factor R.

    Parameters
    ----------
    A : np.ndarray
        Square matrix.
    E : np.ndarray
        Direction matrix.
    b : np.ndarray
        Starting vector.
    k : int
        Depth.

    Returns
    -------
    float
        The entry R[0, k] of the k-by-(k+1) coupling factor.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sep_orth_R_lead_entry(A: np.ndarray, E: np.ndarray, b: np.ndarray, k: int) -> float:
    import numpy as np

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
    R = np.zeros((k, k + 1), dtype=float)
    bn = np.linalg.norm(b)
    if bn < 1e-15:
        raise ValueError("b must be nonzero")
    Y[:, 0] = b / bn

    for it in range(k):
        wX = AX[:, :it] @ R[:it, it] + E @ Y[:, it]
        wY = A @ Y[:, it]
        h = Y[:, : it + 1].T @ wY
        wY = wY - Y[:, : it + 1] @ h
        for _ in range(2):
            corr = Y[:, : it + 1].T @ wY
            wY = wY - Y[:, : it + 1] @ corr
        beta = float(np.linalg.norm(wY))
        if beta < 1e-15:
            raise ValueError("lucky breakdown in bottom orthonormalization")
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

    return float(R[0, -1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [1.0, 0.0]])\nE = np.ones((2, 2))\nb = np.array([1.0, 0.0])\nk = 1",
            "call": "sep_orth_R_lead_entry(A, E, b, k)",
            "gold_call": "_oracle_sep_orth_R_lead_entry(A, E, b, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 2.0, 0.5], [0.1, 0.0, 1.0], [0.0, 1.5, 0.0]])\nE = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])\nb = np.array([0.0, 1.0, 0.0])\nk = 2",
            "call": "sep_orth_R_lead_entry(A, E, b, k)",
            "gold_call": "_oracle_sep_orth_R_lead_entry(A, E, b, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[1.0, 0.2], [0.3, 1.5]])\nE = np.array([[0.0, 1.0], [1.0, 0.0]])\nb = np.array([1.0, 0.0])\nk = 1",
            "call": "sep_orth_R_lead_entry(A, E, b, k)",
            "gold_call": "_oracle_sep_orth_R_lead_entry(A, E, b, k)",
        },
    ]
