"""
Use the current search-space state (from Step 02 or from an expansion in Step 05) to perform the source study's exact full-space generalized Rayleigh-Ritz extraction and obtain the selected approximate spectral states of the benchmark operator. The extraction operates on the reduced representation associated with the current basis and returns the selected states together with the quantities required for the next expansion stage. The operator, sketch, basis, and auxiliary representation are carried forward unchanged as part of the numerical chain.

Iterative eigensolvers approximate selected parts of a large spectrum by restricting the problem to a much smaller search space. Information from this reduced space is used to identify approximate eigenvalues and corresponding physical-space vectors, after which the quality of those approximations can be assessed relative to the original operator. This separation between search-space construction and spectral extraction allows the method to maintain a compact iterative representation while still evaluating the resulting states in the original problem space.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], containing the operator A, sketch S, current search basis Vt, stored sketch Q, operator image W = A @ Vt, selected Ritz values theta, generalized Ritz coefficient matrix Y, Ritz vectors U = Vt @ Y, and residual block R
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generalized_ritz(state: tuple, k: int = 3) -> tuple:
    """Extract the selected approximate spectral states.

    Parameters
    ----------
    state : tuple
        Numerical state (A, S, Vt, Q) produced by Step 02 or by Step 05:
        operator, sketch, current full-space basis and its stored sketch.
    k : int
        Positive number of spectral states to retain, bounded by the
        available search-space dimension.

    Returns
    -------
    tuple
        Numerical state containing the propagated search information and
        the selected Ritz quantities and residual information.

    Raises
    ------
    ValueError
        If the chained state is malformed, dimensions are inconsistent,
        k is invalid, or the computed quantities are not finite.

    Notes
    -----
    The selected states must remain compatible with the generalized
    spectral problem defined by the preceding stage.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh


def _oracle_generalized_ritz(state: tuple, k: int = 3) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 4:
        raise ValueError("state must be (A, S, Vt, Q)")

    A, S, Vt, Q = [
        np.asarray(x, dtype=np.float64)
        for x in state
    ]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = int(A.shape[0])

    if n < 1:
        raise ValueError("A must have positive dimension")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if Vt.ndim != 2 or Vt.shape[0] != n or Vt.shape[1] < 1:
        raise ValueError("Vt must have shape (n,j) with j >= 1")

    if Q.ndim != 2:
        raise ValueError("Q must be two-dimensional")

    if Q.shape[0] != S.shape[0] or Q.shape[1] != Vt.shape[1]:
        raise ValueError("inconsistent Vt and Q shapes")

    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")

    if not np.all(np.isfinite(Vt)):
        raise ValueError("Vt must contain only finite values")

    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must contain only finite values")

    if not isinstance(k, (int, np.integer)) or isinstance(
        k, (bool, np.bool_)
    ):
        raise ValueError("k must be an integer")

    k = int(k)

    if k < 1 or k > Vt.shape[1]:
        raise ValueError("invalid k")

    # Operator action on the current search basis.
    W = A @ Vt

    if not np.all(np.isfinite(W)):
        raise ValueError("operator-applied basis contains non-finite values")

    # Full-space reduced quantities.
    G = Vt.T @ Vt
    H = Vt.T @ W

    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("reduced matrices contain non-finite values")

    # The reduced problem is solved through the positive-definite Gram matrix.
    try:
        vals, vecs = eigh(
            H,
            G,
            lower=True,
            check_finite=True,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("spectral extraction failed") from exc

    theta = vals[:k].copy()
    Y = vecs[:, :k].copy()

    # Construct physical-space vectors and impose a deterministic global
    # sign convention.
    U = Vt @ Y

    for c in range(k):
        nz = np.flatnonzero(np.abs(U[:, c]) > 1e-15)
        if nz.size and U[nz[0], c] < 0.0:
            U[:, c] *= -1.0
            Y[:, c] *= -1.0

    # Residuals in the original space.
    R = W @ Y - U * theta[None, :]

    if (
        not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(Y))
        or not np.all(np.isfinite(U))
        or not np.all(np.isfinite(R))
    ):
        raise ValueError("spectral output contains non-finite values")

    return A, S, Vt, Q, W, theta, Y, U, R

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test specifications."""
    return [
        # Normal: prompt-family tridiagonal A (n = 8), non-orthonormal 5-column basis, k = 3 < j = 5,
        # so the generalized normalization, the lowest-k selection and nonzero residuals are all compared.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "n = 8\n"
                "i = np.arange(1, n + 1, dtype=float)\n"
                "A = np.zeros((n, n), dtype=float)\n"
                "A[np.diag_indices(n)] = np.log(99.0 + i)\n"
                "off = 0.19*np.sin(0.73*i[:-1]) + 0.004*np.cos(0.41*i[:-1])\n"
                "A[np.arange(n-1), np.arange(1,n)] = off\n"
                "A[np.arange(1,n), np.arange(n-1)] = off\n"
                "V = np.random.default_rng(11).standard_normal((n, 5))\n"
                "S = np.zeros((6, n), dtype=float)\n"
                "rng = np.random.default_rng(5)\n"
                "for col in range(n):\n"
                "    rows = rng.choice(6, 2, replace=False)\n"
                "    S[rows, col] = (2*rng.integers(0, 2, 2) - 1) / np.sqrt(2.0)\n"
                "Q = S @ V\n"
                "state = (A, S, V, Q)\n"
            ),
            "call": "generalized_ritz(copy.deepcopy(state), k=3)",
            "gold_call": "_oracle_generalized_ritz(state, k=3)",
        },
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.array([[2.0, 0.0], [0.0, 5.0]])\n"
                "S = np.eye(2, dtype=float)\n"
                "V = np.array([[1.0], [0.0]])\n"
                "Q = S @ V\n"
                "state = (A, S, V, Q)\n"
            ),
            "call": "generalized_ritz(copy.deepcopy(state), k=1)",
            "gold_call": "_oracle_generalized_ritz(state, k=1)",
        },
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.eye(3, dtype=float)\n"
                "S = np.eye(3, dtype=float)\n"
                "V = np.eye(3, dtype=float)\n"
                "Q = S @ V\n"
                "state = (A, S, V, Q)\n"
            ),
            "call": "generalized_ritz(copy.deepcopy(state), k=3)",
            "gold_call": "_oracle_generalized_ritz(state, k=3)",
        },
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.eye(3, dtype=float)\n"
                "S = np.eye(3, dtype=float)\n"
                "V = np.eye(3, dtype=float)[:, :2]\n"
                "Q = S @ V\n"
                "state = (A, S, V, Q)\n"
                "def run_model():\n"
                "    try:\n"
                "        generalized_ritz(copy.deepcopy(state), k=3)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_generalized_ritz(state, k=3)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
