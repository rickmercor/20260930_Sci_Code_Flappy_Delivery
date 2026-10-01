"""
Leave-one-out squared Frobenius Nyström residual estimator.

Choosing between sketching strategies requires knowing how much of the matrix the current sketch has already captured, but forming the residual directly would cost more matrix-vector products than the budget allows. A resampling diagnostic reuses the sketch already drawn: each column is held out in turn and scored against an approximation built without it, at no additional matvec cost.

Returns
-------
float, leave-one-out estimate of ||A-Ahat||_F^2 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def leave_one_out_errF2(A: np.ndarray, Omega: np.ndarray) -> float:
    """Leave-one-out estimate of ||A - Ahat||_F^2 for Nyström sketch Omega.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    Omega : ndarray, shape (n, s)
        Sketching matrix with s >= 2.

    Returns
    -------
    float
        Estimated squared Frobenius residual of the Nyström approximation.

    Raises
    ------
    ValueError
        If ``A`` is not a two-dimensional square array; if ``Omega`` is not
        two-dimensional with ``Omega.shape[0] == A.shape[0]``; if ``A`` or
        ``Omega`` contains a non-finite entry; or if ``Omega`` has fewer than
        two columns, since each replicate leaves one column out.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_leave_one_out_errF2(A: np.ndarray, Omega: np.ndarray) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")
    s = Omega.shape[1]
    if s < 2:
        raise ValueError("leave-one-out requires at least 2 sketch columns")
    errs = []
    for i in range(s):
        mask = np.ones(s, dtype=bool)
        mask[i] = False
        Om_i = Omega[:, mask]
        Y = A @ Om_i
        si = Om_i.shape[1]
        nu = np.finfo(float).eps * norm(Y, 2)
        C = Om_i.T @ Y + nu * np.eye(si)
        try:
            CinvYT = solve(C, Y.T)
        except np.linalg.LinAlgError:
            CinvYT = pinv(C) @ Y.T
        Ahat_i = 0.5 * ((Y @ CinvYT) + (Y @ CinvYT).T)
        evals, evecs = eigh(Ahat_i)
        evals = np.maximum(evals, 0.0)
        Ahat_i = (evecs * evals) @ evecs.T
        errs.append(norm((A - Ahat_i) @ Omega[:, i]) ** 2)
    return float(np.mean(errs))

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
                "Omega = rng.standard_normal((32, 12))[:, :9]"
            ),
            "call": "leave_one_out_errF2(A, Omega)",
            "gold_call": "_oracle_leave_one_out_errF2(A, Omega)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "A = np.diag([4.0, 2.0, 1.0, 0.5])\n"
                "Omega = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])"
            ),
            "call": "leave_one_out_errF2(A, Omega)",
            "gold_call": "_oracle_leave_one_out_errF2(A, Omega)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "rng = np.random.default_rng(2)\n"
                "A = np.diag(np.logspace(1, -2, 10))\n"
                "Omega = rng.standard_normal((10, 3))"
            ),
            "call": "leave_one_out_errF2(A, Omega)",
            "gold_call": "_oracle_leave_one_out_errF2(A, Omega)",
        },
    ]
