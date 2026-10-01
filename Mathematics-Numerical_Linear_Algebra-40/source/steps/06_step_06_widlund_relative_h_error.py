"""
Relative H-norm error of a fixed-length Widlund iterate for the split system.

Run the energy-optimal short-recurrence method whose estimate was quantified previously for exactly the requested number of steps on the dissipative system with the given load, and report how far its iterate sits from the exact solution, measured relatively in the energy norm induced by the coercive part. Fidelity to that specific method matters: the iterate is the energy-orthogonal projection onto the Krylov space of the preconditioned transport operator built from the preconditioned load, and the reduced operator inherited from that space is skew with vanishing diagonal, which must be respected exactly rather than approximately.

Returns
-------
float, H-norm relative error of the k-step Widlund iterate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def widlund_relative_h_error(
    n: int,
    nu: float,
    b_adv: float,
    c: float,
    b: np.ndarray,
    k: int,
) -> float:
    """Return the H-norm relative error of the k-step Widlund iterate.

    

    Returns
    -------
    float
        Relative H-norm error.

    Raises
    ------
    ValueError
        If n < 2, nu <= 0, c < 0, k < 1, or an even-iterate bound is invalid.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_widlund_relative_h_error(
    n: int,
    nu: float,
    b_adv: float,
    c: float,
    b: np.ndarray,
    k: int,
) -> float:
    if n < 2:
        raise ValueError("n must be at least 2")
    if nu <= 0:
        raise ValueError("nu must be positive")
    if c < 0:
        raise ValueError("c must be nonnegative")
    if k < 1:
        raise ValueError("k must be at least 1")
    bound = None
    if k % 2 == 0:
        bound = _oracle_widlund_even_iterate_bound(n, nu, b_adv, c, k)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("invalid even-iterate bound from the preceding step")
    b = np.asarray(b, dtype=float).reshape(n)
    h = 1.0 / (n + 1)
    main = -2.0 / h**2
    off = 1.0 / h**2
    lap = np.diag(main * np.ones(n)) + np.diag(off * np.ones(n - 1), 1)
    lap = lap + np.diag(off * np.ones(n - 1), -1)
    H = -nu * lap + c * np.eye(n)
    S = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            S[i, i - 1] = -b_adv / (2.0 * h)
        if i < n - 1:
            S[i, i + 1] = b_adv / (2.0 * h)
    A = H + S
    b_hat = np.linalg.solve(H, b)
    r0h = float(np.sqrt(max(b_hat @ H @ b_hat, 0.0)))
    if r0h < 1e-30:
        return 0.0

    # H-MGS Krylov basis for M = H^{-1}S applied matrix-free via H-solves.
    V = np.zeros((n, k))
    V[:, 0] = b_hat / r0h
    dim = 1
    for j in range(1, k):
        w = np.linalg.solve(H, S @ V[:, j - 1])
        for i in range(j):
            w = w - float(w @ H @ V[:, i]) * V[:, i]
        beta = float(np.sqrt(max(w @ H @ w, 0.0)))
        if beta < 1e-14:
            break
        V[:, j] = w / beta
        dim = j + 1
    V = V[:, :dim]
    MV = np.column_stack([np.linalg.solve(H, S @ V[:, j]) for j in range(dim)])
    T = V.T @ (H @ MV)
    # Force skew + zero diagonal (H-skew Galerkin projection).
    T = 0.5 * (T - T.T)
    np.fill_diagonal(T, 0.0)
    rhs = np.zeros(dim)
    rhs[0] = r0h
    y = np.linalg.solve(np.eye(dim) + T, rhs)
    x_k = V @ y
    x_exact = np.linalg.solve(A, b)
    denom = float(np.sqrt(x_exact @ H @ x_exact))
    if denom < 1e-30:
        return 0.0
    diff = x_k - x_exact
    err = float(np.sqrt(max(diff @ H @ diff, 0.0)) / denom)
    if not np.isfinite(err):
        raise ValueError("non-finite Widlund relative error")
    # Cross-step check: a vanishing even-iterate bound forces a vanishing error,
    # since both are driven by the same spectral width.
    if bound is not None and bound == 0.0 and err > 1e-12:
        raise ValueError("zero even-iterate bound with nonzero realized error")
    return err

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nb = np.array([0.7, -1.1, 0.4, 1.2, -0.85, 0.3, 0.55, -0.2])",
            "call": "widlund_relative_h_error(8, 0.37, 1.85, 0.18, b, 6)",
            "gold_call": "_oracle_widlund_relative_h_error(8, 0.37, 1.85, 0.18, b, 6)",
        },
        {
            "setup": "import numpy as np\nb = np.array([0.0, 1.0, 0.0])",
            "call": "widlund_relative_h_error(3, 1.0, 0.0, 0.5, b, 3)",
            "gold_call": "_oracle_widlund_relative_h_error(3, 1.0, 0.0, 0.5, b, 3)",
        },
        {
            "setup": "import numpy as np\nb = np.zeros(5)",
            "call": "widlund_relative_h_error(5, 0.25, 1.5, 0.1, b, 4)",
            "gold_call": "_oracle_widlund_relative_h_error(5, 0.25, 1.5, 0.1, b, 4)",
        },
        {
            "setup": "import numpy as np\nb = np.linspace(1.0, -0.5, 8)",
            "call": "widlund_relative_h_error(8, 0.05, 2.0, 0.02, b, 5)",
            "gold_call": "_oracle_widlund_relative_h_error(8, 0.05, 2.0, 0.02, b, 5)",
        },
        {
            "setup": "import numpy as np\nb = np.array([1e-12, -1e-12, 1e-12, -1e-12])",
            "call": "widlund_relative_h_error(4, 0.5, 0.75, 0.25, b, 2)",
            "gold_call": "_oracle_widlund_relative_h_error(4, 0.5, 0.75, 0.25, b, 2)",
        },
        {
            "setup": "import numpy as np\nb = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])",
            "call": "widlund_relative_h_error(7, 0.02, 3.0, 0.0, b, 7)",
            "gold_call": "_oracle_widlund_relative_h_error(7, 0.02, 3.0, 0.0, b, 7)",
        },
        {
            "setup": "import numpy as np\nb = np.eye(5)[0]",
            "call": "widlund_relative_h_error(5, 1e-3, -2.5, 0.05, b, 1)",
            "gold_call": "_oracle_widlund_relative_h_error(5, 1e-3, -2.5, 0.05, b, 1)",
        },
        {
            "setup": "import numpy as np\nb = np.array([0.5, -1.0, 0.25, 0.75, -0.5, 1.0, -0.25, 0.1])",
            "call": "widlund_relative_h_error(8, 0.03, 2.8, 0.05, b, 6)",
            "gold_call": "_oracle_widlund_relative_h_error(8, 0.03, 2.8, 0.05, b, 6)",
        },
    ]
