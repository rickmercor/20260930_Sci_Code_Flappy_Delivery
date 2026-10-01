"""
Exact preconditioner contribution t1 = tr log(Ahat + I).

The matvec budget can be spent two ways: almost entirely on the preconditioner with a single residual probe, or split so several probes average down the stochastic error. The right choice depends on whether enriching the sketch is still reducing the residual. Comparing the diagnostic at two sketch widths, weighted by how the budget would be reallocated, decides the branch.

Returns
-------
float, t1 = tr log(Ahat+I) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def preconditioner_logdet(A: np.ndarray, Omega: np.ndarray) -> float:
    """Return tr log(Ahat + I) for the Nyström approximant of A with sketch Omega.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    Omega : ndarray, shape (n, s)
        Sketching matrix.

    Returns
    -------
    float
        The preconditioner log-determinant tr log(Ahat + I).

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

def _oracle_preconditioner_logdet(A: np.ndarray, Omega: np.ndarray) -> float:
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
    Ahat = 0.5 * ((Y @ CinvYT) + (Y @ CinvYT).T)
    evals = np.maximum(eigh(Ahat)[0], 0.0)
    return float(np.sum(np.log1p(evals)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "rng = np.random.default_rng(0)\n"
                "A = np.diag((np.arange(1, 33, dtype=float) ** (-2)) / 1e-2)\n"
                "Om = rng.standard_normal((32, 12))\n"
                "Psi = rng.standard_normal((32, 4))\n"
                "Omega = np.hstack([Om, Psi])"
            ),
            "call": "preconditioner_logdet(A, Omega)",
            "gold_call": "_oracle_preconditioner_logdet(A, Omega)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "A = np.diag([3.0, 1.0, 0.2])\n"
                "Omega = np.eye(3)[:, :2]"
            ),
            "call": "preconditioner_logdet(A, Omega)",
            "gold_call": "_oracle_preconditioner_logdet(A, Omega)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "A = np.diag([3.0, 0.0, 0.0, 0.0])\n"
                "Omega = np.eye(4)[:, :1]"
            ),
            "call": "preconditioner_logdet(A, Omega)",
            "gold_call": "_oracle_preconditioner_logdet(A, Omega)",
        },
    ]
