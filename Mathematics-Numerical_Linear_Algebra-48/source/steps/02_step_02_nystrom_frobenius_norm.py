"""
Stabilized Nyström approximation; return the Frobenius norm of Ahat.

The randomized Nystrom approximation compresses an SPSD matrix onto the range of a random sketch and is the low-rank building block of the preconditioner used later. The core matrix it inverts is severely ill-conditioned in floating point, so a stabilizing perturbation is applied before inversion. The Frobenius norm of the resulting approximant is a compact fingerprint of sketch quality.

Returns
-------
float, Frobenius norm ||Ahat||_F of the stabilized Nyström approximant as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nystrom_frobenius_norm(A: np.ndarray, Omega: np.ndarray) -> float:
    """Compute ||Ahat||_F for the stabilized Nyström sketch of A with sketch Omega.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    Omega : ndarray, shape (n, s)
        Gaussian sketching matrix.

    Returns
    -------
    float
        Frobenius norm of the Nyström approximant Ahat.

    Raises
    ------
    ValueError
        If ``A`` is not a two-dimensional square array; if ``Omega`` is not
        two-dimensional with ``Omega.shape[0] == A.shape[0]``; if ``Omega``
        has no columns; or if ``A`` or ``Omega`` contains a non-finite entry.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_nystrom_frobenius_norm(A: np.ndarray, Omega: np.ndarray) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if Omega.shape[1] < 1:
        raise ValueError("Omega must have at least one column")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")

    Y = A @ Omega
    s = Omega.shape[1]
    nu = np.finfo(float).eps * norm(Y, 2)
    C = Omega.T @ Y + nu * np.eye(s)
    try:
        CinvYT = solve(C, Y.T)
    except np.linalg.LinAlgError:
        CinvYT = pinv(C) @ Y.T
    Ahat = Y @ CinvYT
    Ahat = 0.5 * (Ahat + Ahat.T)
    evals, evecs = eigh(Ahat)
    evals = np.maximum(evals, 0.0)
    Ahat = (evecs * evals) @ evecs.T
    return float(norm(Ahat, "fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "rng = np.random.default_rng(0)\n"
                "A = np.diag((np.arange(1, 9, dtype=float) ** (-2)) / 1e-2)\n"
                "Omega = rng.standard_normal((8, 4))"
            ),
            "call": "nystrom_frobenius_norm(A, Omega)",
            "gold_call": "_oracle_nystrom_frobenius_norm(A, Omega)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "A = np.diag([10.0, 1.0, 0.1, 0.01])\n"
                "Omega = np.eye(4)[:, :2]"
            ),
            "call": "nystrom_frobenius_norm(A, Omega)",
            "gold_call": "_oracle_nystrom_frobenius_norm(A, Omega)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "rng = np.random.default_rng(1)\n"
                "A = np.diag(np.linspace(5.0, 0.05, 6))\n"
                "Omega = rng.standard_normal((6, 1))"
            ),
            "call": "nystrom_frobenius_norm(A, Omega)",
            "gold_call": "_oracle_nystrom_frobenius_norm(A, Omega)",
        },
    ]
