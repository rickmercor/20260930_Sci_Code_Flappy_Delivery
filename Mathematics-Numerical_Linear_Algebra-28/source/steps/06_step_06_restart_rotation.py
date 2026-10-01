"""
Restart the search space onto the retained Ritz states as the source study prescribes. The input is the generalized Ritz state of Step 03 at the maximal search-space dimension; the output is the restarted three-column basis together with its stored sketch and its operator image, which begin the next cycle. The fixed sketch is carried forward unchanged and no new sketch is drawn.

Restarted eigensolvers bound the memory and the cost of each iteration by periodically compressing the search space onto the most relevant approximate eigenvectors. In a sketched solver the compressed space must also carry a sketch and an operator image consistent with the new basis, and the way the restart treats the basis determines the geometry that later expansions build on.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], containing the unchanged operator A and sketch S together with the restarted basis V_restart, its stored sketch Q_restart = S @ V_restart, and its operator image W_restart = A @ V_restart
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def restart_rotation(state: tuple) -> tuple:
    """Restart the search space onto the retained Ritz states.

    Parameters
    ----------
    state : tuple
        Generalized Ritz state produced by Step 03 at the maximal
        dimension: (A, S, Vt, Q, W, theta, Y, U, R).

    Returns
    -------
    tuple
        (A, S, V_restart, Q_restart, W_restart): the restarted basis, its
        stored sketch and its operator image, with one column per retained
        Ritz state.

    Raises
    ------
    ValueError
        If the state is malformed or its dimensions are inconsistent.

    Notes
    -----
    The retained Ritz coefficients are used with the signs they carry in
    the input state.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_restart_rotation(state: tuple) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 9:
        raise ValueError("state must be the Step 03 tuple")

    A, S, Vt, Q, W, theta, Y, U, R = [np.asarray(x, dtype=np.float64) for x in state]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if Vt.ndim != 2 or Vt.shape[0] != A.shape[0] or W.shape != Vt.shape:
        raise ValueError("Vt and W must have shape (n,j)")

    if Q.ndim != 2 or Q.shape[1] != Vt.shape[1]:
        raise ValueError("Q and Vt have inconsistent dimensions")

    if Y.ndim != 2 or Y.shape[0] != Vt.shape[1]:
        raise ValueError("Y has inconsistent shape")

    if not all(np.all(np.isfinite(x)) for x in (A, S, Vt, Q, W, Y)):
        raise ValueError("state contains non-finite values")

    return A, S, Vt @ Y, Q @ Y, W @ Y

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge test specifications."""
    common = (
        "import copy\nimport numpy as np\n"
        "from scipy.linalg import eigh\n"
        "def _family(n):\n"
        "    i = np.arange(1, n + 1, dtype=float)\n"
        "    A = np.zeros((n, n))\n"
        "    A[np.diag_indices(n)] = np.log(99.0 + i)\n"
        "    off = 0.19*np.sin(0.73*i[:-1]) + 0.004*np.cos(0.41*i[:-1])\n"
        "    A[np.arange(n-1), np.arange(1, n)] = off\n"
        "    A[np.arange(1, n), np.arange(n-1)] = off\n"
        "    return A\n"
        "def _sketch(s, n, zeta, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    S = np.zeros((s, n))\n"
        "    for col in range(n):\n"
        "        rows = rng.choice(s, zeta, replace=False)\n"
        "        signs = 2*rng.integers(0, 2, zeta) - 1\n"
        "        S[rows, col] = signs / np.sqrt(zeta)\n"
        "    return S\n"
        "def _extraction(A, S, V, k, flip=()):\n"
        "    Qs, Rf = np.linalg.qr(S @ V)\n"
        "    d = np.sign(np.diag(Rf))\n"
        "    Vt = np.linalg.solve((d[:, None]*Rf).T, V.T).T\n"
        "    Q = S @ Vt\n"
        "    W = A @ Vt\n"
        "    vals, vecs = eigh(Vt.T @ W, Vt.T @ Vt)\n"
        "    theta, Y = vals[:k].copy(), vecs[:, :k].copy()\n"
        "    U = Vt @ Y\n"
        "    for c in range(k):\n"
        "        nz = np.flatnonzero(np.abs(U[:, c]) > 1e-15)\n"
        "        if nz.size and U[nz[0], c] < 0.0:\n"
        "            U[:, c] *= -1.0\n"
        "            Y[:, c] *= -1.0\n"
        "    for c in flip:\n"
        "        U[:, c] *= -1.0\n"
        "        Y[:, c] *= -1.0\n"
        "    R = W @ Y - U * theta[None, :]\n"
        "    return (A, S, Vt, Q, W, theta, Y, U, R)\n"
    )
    return [
        # Normal: restart from a nine-column sketch-orthonormal basis (j_max = 9).
        {
            "setup": common + (
                "n = 24\n"
                "A = _family(n)\n"
                "S = _sketch(45, n, 4, 2026)\n"
                "state = _extraction(A, S, np.random.default_rng(9).standard_normal((n, 9)), 3)\n"
            ),
            "call": "restart_rotation(copy.deepcopy(state))",
            "gold_call": "_oracle_restart_rotation(state)",
        },

        # Boundary: smallest maximal dimension, j_max = 2k = 6.
        {
            "setup": common + (
                "n = 16\n"
                "A = _family(n)\n"
                "S = _sketch(30, n, 4, 11)\n"
                "state = _extraction(A, S, np.random.default_rng(10).standard_normal((n, 6)), 3)\n"
            ),
            "call": "restart_rotation(copy.deepcopy(state))",
            "gold_call": "_oracle_restart_rotation(state)",
        },

        # Edge: retained coefficient columns carrying flipped signs.
        {
            "setup": common + (
                "n = 24\n"
                "A = _family(n)\n"
                "S = _sketch(45, n, 4, 2027)\n"
                "state = _extraction(A, S, np.random.default_rng(12).standard_normal((n, 9)), 3, flip=(0, 2))\n"
            ),
            "call": "restart_rotation(copy.deepcopy(state))",
            "gold_call": "_oracle_restart_rotation(state)",
        },
    ]
