"""
Compute the posterior mode of the log-rate of a log-Gaussian Cox process fitted to binned Monte Carlo counts, the penalised maximum-likelihood estimate of the Laplace approximation.

The raw Monte Carlo counts $a_j$ in bins of width $w_j$ are Poisson distributed with means $e^{f_j} w_j$, and the log-rate vector $f$ carries a zero-mean Gaussian prior with covariance $K$. The Laplace approximation replaces the posterior by a Gaussian centred at the mode $\hat f$, which maximises the Poisson log-likelihood plus the Gaussian log-prior. Because the Poisson log-likelihood is concave in $f$ and the prior term is strictly concave, the objective has a unique maximiser; Newton's method converges in a handful of iterations for typical templates.

The mode defines the smooth template: the fitted rate $e^{\hat f_j} w_j$ replaces the raw count of every bin, with a value that pools information from the neighbouring bins according to the kernel.

Returns
-------
np.ndarray, Array of shape (N,) maximising sum_j (a_j f_j - exp(f_j) w_j) - f^T K^{-1} f / 2 over f, with a_j the counts and w_j the widths. The objective is strictly concave, so the maximiser is unique; it is returned with every component converged to within 1e-10 of the exact maximiser (the gradient a_j - exp(f_j) w_j - (K^{-1} f)_j vanishes to that accuracy). A bin with zero counts is valid and is kept finite by the prior rather than driven to minus infinity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lgcp_posterior_mode(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    r"""Return the log-rate vector maximising the penalised Poisson log-likelihood.

    Parameters
    ----------
    counts : np.ndarray
        Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers (an integer or float dtype).
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite.

    Returns
    -------
    f_hat : np.ndarray
        Array of shape (N,) maximising sum_j (a_j f_j - exp(f_j) w_j)
        - f^T K^{-1} f / 2 over f, with a_j the counts and w_j the widths.
        The objective is strictly concave, so the maximiser is unique;
        it is returned with every component converged to within 1e-10 of
        the exact maximiser (the gradient a_j - exp(f_j) w_j
        - (K^{-1} f)_j vanishes to that accuracy). A bin with zero counts
        is valid and is kept finite by the prior rather than driven to
        minus infinity.

    Raises
    ------
    ValueError
        If counts is not a finite one-dimensional array of non-negative
        integer values, if widths is not a finite one-dimensional array of
        the same length with every entry above zero, or if K is not a
        finite (N, N) matrix that is symmetric to within 1e-9 and positive
        definite.
    """
    return f_hat

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_counts_widths(counts, widths) -> tuple:
    a = np.asarray(counts, dtype=float)
    w = np.asarray(widths, dtype=float)
    if a.ndim != 1 or a.shape[0] < 1 or not np.all(np.isfinite(a)) or np.any(a < 0.0) or np.any(a != np.round(a)):
        raise ValueError("counts must be a finite one-dimensional array of non-negative integers")
    if w.shape != a.shape or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("widths must be a finite one-dimensional array of the same length with positive entries")
    return a, w


def _check_covariance(K, size: int) -> "np.ndarray":
    """Validate a symmetric positive definite (size, size) matrix and return it as float."""
    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite square matrix matching the number of bins")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9):
        raise ValueError("K must be symmetric to within 1e-9")
    try:
        np.linalg.cholesky(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        raise ValueError("K must be positive definite")
    return matrix


def _oracle_lgcp_posterior_mode(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    a, w = _check_counts_widths(counts, widths)
    prior = _check_covariance(K, a.shape[0])
    precision = np.linalg.inv(prior)
    f = np.log((a + 0.5) / w)
    for _ in range(500):
        rate = np.exp(f) * w
        gradient = a - rate - precision @ f
        step = np.linalg.solve(np.diag(rate) + precision, gradient)
        # damp the Newton step so that the rate cannot overflow far from the mode
        scale = min(1.0, 5.0 / float(np.max(np.abs(step))))
        f = f + scale * step
        if np.max(np.abs(step)) < 1e-13:
            break
    return f

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "lgcp_posterior_mode"),
                           ("run_gold", "_oracle_lgcp_posterior_mode")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    kernel = ("import numpy as np\n"
              "def kernel(x, sigma, ell, b, n_basis=4):\n"
              "    from scipy.interpolate import BSpline\n"
              "    lo, hi = 0.0, 1.0\n"
              "    interior = np.linspace(lo, hi, n_basis - 2)[1:-1]\n"
              "    knots = np.concatenate([np.full(4, lo), interior, np.full(4, hi)])\n"
              "    H = BSpline.design_matrix(x, knots, 3).toarray()\n"
              "    r = np.abs(x[:, None] - x[None, :]) / ell\n"
              "    return sigma ** 2 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + b ** 2 * H @ H.T\n"
              "x = (np.arange(12) + 0.5) / 12.0\n"
              "w = np.full(12, 1.0 / 12.0)\n"
              "a = np.array([2947, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n")
    return [
        # the benchmark template with the selected length scale
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 10.0)\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # a short length scale, where the mode follows the counts closely
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.1, 10.0)\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # a long length scale with a small amplitude, where the prior pulls the mode towards the mean-function subspace
        {
            "setup": kernel + "K = kernel(x, 0.2, 0.8, 10.0)\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # a systematic variation template of the benchmark
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 10.0)\n"
                     "a = np.array([2557, 1887, 1339, 1056, 1115, 1471, 1441, 939, 397, 151, 92, 67])\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # edge: a sparse template with empty bins, where the prior keeps the mode finite
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.2, 3.0)\n"
                     "a = np.array([9, 4, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0])\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # edge: non-uniform bin widths change the rate per unit width but not the counts
        {
            "setup": kernel + "w = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])\n"
                     "K = kernel(x, 1.0, 0.3, 10.0)\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # boundary: a single bin, where the mode solves a scalar equation
        {
            "setup": "import numpy as np\n"
                     "a = np.array([37])\nw = np.array([0.25])\nK = np.array([[4.0]])\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # edge: a strong prior (tiny variance) pins the mode near zero log-rate regardless of the counts
        {
            "setup": kernel + "K = 1e-4 * np.eye(12)\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # edge: integer-valued counts of a small dtype are valid input and must not be modified in place
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 10.0)\n"
                     "a = np.array([2947, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74], dtype=np.int32)\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        # stress: forty bins of the paper's grid with counts spanning three orders of magnitude
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "w = np.full(40, 0.025)\n"
                     "a = np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2) + 12 * np.cos(9 * x) ** 2).astype(int)\n"
                     "r = np.abs(x[:, None] - x[None, :]) / 0.15\n"
                     "K = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + 25.0 * np.outer(np.ones(40), np.ones(40))\n",
            "call": "lgcp_posterior_mode(a, w, K)",
            "gold_call": "_oracle_lgcp_posterior_mode(a, w, K)",
            "tol": 1e-7,
        },
        {
            "setup": kernel + "# invalid: a negative count\n"
                     "a = np.array([2947, 2193, -1, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
                     "args = (a, w, kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a non-integer count\n"
                     "a = np.array([2947.5, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
                     "args = (a, w, kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a covariance that is not positive definite\n"
                     "args = (a, w, -kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a bin width of zero\n"
                     "w = np.full(12, 1.0 / 12.0); w[3] = 0.0\n"
                     "args = (a, w, kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: an asymmetric matrix\n"
                     "K = kernel(x, 1.0, 0.3, 10.0); K[0, 1] += 1e-3\n"
                     "args = (a, w, K)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
