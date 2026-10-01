"""
One-probe Lanczos quadrature for w^T log(M) w on the preconditioned matrix.

The residual term is a trace of a matrix function, estimated from quadratic forms in random probe vectors. Each probe is handled by a short Kryloviteration that reduces the operator to a small tridiagonal matrix, on which the function is evaluated by Gaussian quadrature. Effective preconditioning clusters the spectrum, so few iterations and few probes suffice.

Returns
-------
float, Lanczos approximation of w^T log(M) w on the preconditioned matrix as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def slq_preconditioned_quadratic(
    A: np.ndarray, Omega: np.ndarray, w: np.ndarray, m: int
) -> float:
    """Approximate w^T log(M) w with m Lanczos steps, M = P^{-1/2}(A+I)P^{-1/2}.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        SPSD matrix.
    Omega : ndarray, shape (n, s)
        Sketch defining Ahat and P = Ahat + I.
    w : ndarray, shape (n,)
        Probe vector.
    m : int
        Number of Lanczos steps.

    Returns
    -------
    float
        Quadrature approximation of w^T log(M) w.

    Raises
    ------
    ValueError
        If ``m < 1``; if ``A`` is not a two-dimensional square array; if
        ``Omega`` is not two-dimensional with ``Omega.shape[0] == A.shape[0]``;
        if ``Omega`` has no columns; if ``w`` does not have length
        ``A.shape[0]``; or if ``A``, ``Omega`` or ``w`` contains a non-finite
        entry.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_slq_preconditioned_quadratic(
    A: np.ndarray, Omega: np.ndarray, w: np.ndarray, m: int
) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if m < 1:
        raise ValueError("m must be positive")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if Omega.shape[1] < 1:
        raise ValueError("Omega must have at least one column")
    if w.shape[0] != A.shape[0]:
        raise ValueError("w must have length n")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")
    if not np.all(np.isfinite(w)):
        raise ValueError("w must be finite")

    Y = A @ Omega
    s = Omega.shape[1]
    nu = np.finfo(float).eps * norm(Y, 2)
    C = Omega.T @ Y + nu * np.eye(s)
    try:
        CinvYT = solve(C, Y.T)
    except np.linalg.LinAlgError:
        CinvYT = pinv(C) @ Y.T
    Ahat = 0.5 * ((Y @ CinvYT) + (Y @ CinvYT).T)
    evals, evecs = eigh(Ahat)
    evals = np.maximum(evals, 0.0)
    inv_sqrt = 1.0 / np.sqrt(evals + 1.0)

    beta0 = norm(w)
    if beta0 == 0.0:
        return 0.0
    Q = [w / beta0]
    alphas = []
    betas = []
    beta = 0.0
    for j in range(m):
        v = evecs @ (inv_sqrt * (evecs.T @ Q[j]))
        v = (A + np.eye(A.shape[0])) @ v
        v = evecs @ (inv_sqrt * (evecs.T @ v))
        if j > 0:
            v = v - beta * Q[j - 1]
        alpha = float(np.dot(Q[j], v))
        v = v - alpha * Q[j]
        for qi in Q:
            v = v - np.dot(qi, v) * qi
        beta_next = norm(v)
        alphas.append(alpha)
        betas.append(beta_next)
        if beta_next < 1e-14:
            break
        Q.append(v / beta_next)
        beta = beta_next
    k = len(alphas)
    T = np.diag(alphas)
    for i in range(k - 1):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    tevals, tevecs = eigh(T)
    tevals = np.maximum(tevals, 1e-300)
    logT_11 = float(tevecs[0] @ (np.log(tevals) * tevecs[0]))
    return float((beta0 ** 2) * logT_11)

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
                "Omega = np.hstack([Om, Psi])\n"
                "w = rng.standard_normal(32)"
            ),
            "call": "slq_preconditioned_quadratic(A, Omega, w, 5)",
            "gold_call": "_oracle_slq_preconditioned_quadratic(A, Omega, w, 5)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "A = np.diag([2.0, 1.0, 0.5, 0.25])\n"
                "Omega = np.eye(4)[:, :2]\n"
                "w = np.ones(4)"
            ),
            "call": "slq_preconditioned_quadratic(A, Omega, w, 3)",
            "gold_call": "_oracle_slq_preconditioned_quadratic(A, Omega, w, 3)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "A = np.diag([1.0, 1.0])\n"
                "Omega = np.array([[1.0], [0.0]])\n"
                "w = np.zeros(2)"
            ),
            "call": "slq_preconditioned_quadratic(A, Omega, w, 2)",
            "gold_call": "_oracle_slq_preconditioned_quadratic(A, Omega, w, 2)",
        },
    ]
