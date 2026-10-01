"""
Decompose a template covariance into eigenmodes and keep the leading modes that capture a target fraction of the total variance.

The eigendecomposition $\Sigma = \sum_i \lambda_i v_i v_i^{\mathsf T}$ of the combined covariance concentrates the template uncertainty into a small number of leading modes: for a smooth kernel the eigenvalue spectrum decays rapidly, at a rate set by the ratio of the length scale to the bin spacing, and every systematic source adds at most one significant mode. Retaining the leading modes up to a fixed fraction of the total variance replaces the full set of per-bin nuisance parameters by a handful of Gaussian-constrained amplitudes, and the number of retained modes is itself a diagnostic of how compressible the template uncertainty is.

Returns
-------
np.ndarray, Array of shape (k,) holding the eigenvalues of Sigma in decreasing order, truncated to the smallest k for which their sum is at least fraction times the trace of Sigma (the sum of all eigenvalues). Eigenvalues are those of the symmetrised matrix (Sigma + Sigma^T) / 2; the comparison uses the cumulative sum divided by the trace, so that a fraction of exactly one returns every eigenvalue.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def truncated_eigenvalues(Sigma: "np.ndarray", fraction: float) -> "np.ndarray":
    r"""Return the leading eigenvalues that together reach the target variance fraction.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance of shape (N, N), N at least one, finite, symmetric to
        within an absolute tolerance of 1e-9, with a trace above zero and
        no eigenvalue below -1e-9 times the trace.
    fraction : float
        Target fraction of the total variance, above zero and at most
        one.

    Returns
    -------
    eigenvalues : np.ndarray
        Array of shape (k,) holding the eigenvalues of Sigma in decreasing
        order, truncated to the smallest k for which their sum is at least
        fraction times the trace of Sigma (the sum of all eigenvalues).
        Eigenvalues are those of the symmetrised matrix (Sigma + Sigma^T)
        / 2; the comparison uses the cumulative sum divided by the trace,
        so that a fraction of exactly one returns every eigenvalue.

    Raises
    ------
    ValueError
        If Sigma is not a finite square matrix that is symmetric to within
        1e-9, if its trace is not above zero or an eigenvalue lies below
        -1e-9 times the trace, or if fraction is not finite, not above
        zero or above one.
    """
    return eigenvalues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _sorted_eigenpairs(Sigma) -> tuple:
    """Eigenvalues in decreasing order with matching eigenvectors as columns, after validation."""
    matrix = np.asarray(Sigma, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("Sigma must be a finite square matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9):
        raise ValueError("Sigma must be symmetric to within 1e-9")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    trace = float(np.sum(values))
    if not trace > 0.0:
        raise ValueError("Sigma must have a trace above zero")
    if np.any(values < -1e-9 * trace):
        raise ValueError("Sigma must not have an eigenvalue below -1e-9 times its trace")
    return values, vectors, trace


def _oracle_truncated_eigenvalues(Sigma: "np.ndarray", fraction: float) -> "np.ndarray":
    target = float(fraction)
    if not (math.isfinite(target) and 0.0 < target <= 1.0):
        raise ValueError("fraction must be finite, above zero and at most one")
    values, _, trace = _sorted_eigenpairs(Sigma)
    cumulative = np.cumsum(values) / trace
    count = int(np.searchsorted(cumulative, target, side="left")) + 1
    return values[:min(count, values.shape[0])]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "truncated_eigenvalues"),
                           ("run_gold", "_oracle_truncated_eigenvalues")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    matrix = ("import numpy as np\n"
              "x = (np.arange(12) + 0.5) / 12.0\n"
              "r = np.abs(x[:, None] - x[None, :]) / 0.3\n"
              "stat = 0.002 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)\n"
              "d1 = 0.02 * np.array([0.14, 0.12, 0.11, 0.07, -0.04, -0.09, 0.02, 0.19, 0.32, 0.35, 0.21, 0.09]) * 10\n"
              "d2 = 0.06 * np.ones(12)\n"
              "d3 = 0.01 * np.array([0.1, 0.0, -0.2, -0.3, 0.5, 0.8, 0.3, -0.4, -0.6, -0.2, 0.1, 0.2]) * 10\n"
              "S = stat + np.outer(d1, d1) + np.outer(d2, d2) + np.outer(d3, d3)\n")
    return [
        # a statistical plus three-systematic covariance at the benchmark fraction
        {
            "setup": matrix,
            "call": "truncated_eigenvalues(S, 0.95)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.95)",
            "tol": 1e-9,
        },
        # a tighter fraction retains more modes
        {
            "setup": matrix,
            "call": "truncated_eigenvalues(S, 0.99)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.99)",
            "tol": 1e-9,
        },
        # a loose fraction retains the leading mode only
        {
            "setup": matrix,
            "call": "truncated_eigenvalues(S, 0.5)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.5)",
            "tol": 1e-9,
        },
        # boundary: a fraction of one returns every eigenvalue, including those of order the rounding error
        {
            "setup": matrix,
            "call": "truncated_eigenvalues(S, 1.0)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 1.0)",
            "tol": 1e-9,
        },
        # boundary: the fraction exactly equals the cumulative share of the first two modes
        {
            "setup": "import numpy as np\nS = np.diag([0.5, 0.3, 0.15, 0.05])\n",
            "call": "truncated_eigenvalues(S, 0.8)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.8)",
            "tol": 1e-9,
        },
        # edge: a rank-one covariance, where one mode carries all the variance
        {
            "setup": "import numpy as np\nd = np.array([1.0, -2.0, 0.5, 3.0])\nS = np.outer(d, d)\n",
            "call": "truncated_eigenvalues(S, 0.95)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.95)",
            "tol": 1e-9,
        },
        # edge: degenerate eigenvalues, where the truncation must count the repeated values in order
        {
            "setup": "import numpy as np\nS = np.diag([0.25, 0.25, 0.25, 0.25])\n",
            "call": "truncated_eigenvalues(S, 0.6)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.6)",
            "tol": 1e-9,
        },
        # edge: a covariance supplied with a tiny asymmetry within the tolerance is symmetrised
        {
            "setup": matrix + "S = S.copy(); S[0, 1] += 5e-10\n",
            "call": "truncated_eigenvalues(S, 0.9)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.9)",
            "tol": 1e-9,
        },
        # boundary: a one by one covariance
        {
            "setup": "import numpy as np\nS = np.array([[0.7]])\n",
            "call": "truncated_eigenvalues(S, 0.3)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.3)",
            "tol": 1e-9,
        },
        # stress: a forty-bin covariance with a slowly decaying spectrum
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "r = np.abs(x[:, None] - x[None, :]) / 0.05\n"
                     "S = 0.001 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + 0.0004 * np.outer(np.sin(6 * x), np.sin(6 * x))\n",
            "call": "truncated_eigenvalues(S, 0.95)",
            "gold_call": "_oracle_truncated_eigenvalues(S, 0.95)",
            "tol": 1e-9,
        },
        {
            "setup": matrix + "# invalid: a fraction above one\nargs = (S, 1.2)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": matrix + "# invalid: a fraction of zero\nargs = (S, 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": matrix + "# invalid: an asymmetric matrix\nS = S.copy(); S[0, 1] += 1e-3\nargs = (S, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np\n# invalid: an indefinite matrix\nS = np.diag([1.0, -0.5])\nargs = (S, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "import numpy as np\n# invalid: a zero matrix, whose trace is not above zero\nargs = (np.zeros((3, 3)), 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
