"""
Evaluate the Laplace approximation to the log marginal likelihood of binned Monte Carlo counts under a log-Gaussian Cox process with a given prior covariance.

The marginal likelihood integrates the Poisson likelihood of the counts against the Gaussian prior of the log-rate. The integral has no closed form for a Poisson observation model; the Laplace approximation expands the log-posterior to second order around its mode $\hat f$ and integrates the resulting Gaussian, which gives the log-likelihood of the counts at the mode, minus the prior penalty of the mode, minus one half of the log-determinant of $I + K W$ with $W$ the diagonal matrix of fitted rates. This quantity is the objective used to select the hyperparameters of the kernel: it rewards a prior that explains the counts while penalising the volume of log-rate configurations the prior admits, so that neither a rigid nor an overly flexible kernel is preferred.

Returns
-------
float, the value, as a native Python float, of sum_j (a_j log mu_j - mu_j - log Gamma(a_j + 1)) - f^T K^{-1} f / 2 - log det(I + K W) / 2, evaluated at the posterior mode f of the log-rate for this K, with mu_j = exp(f_j) * w_j the fitted rates and W their diagonal matrix. The first term is the Poisson log-likelihood of the counts at the mode including the log-factorial of every count, so that the value is a proper log-probability.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def laplace_log_marginal_likelihood(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> float:
    r"""Return the Laplace approximation to the log marginal likelihood of the counts.

    Parameters
    ----------
    counts : np.ndarray
        Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite.

    Returns
    -------
    log_marginal : float
        The value, as a native Python float, of sum_j (a_j log mu_j - mu_j
        - log Gamma(a_j + 1)) - f^T K^{-1} f / 2 - log det(I + K W) / 2,
        evaluated at the posterior mode f of the log-rate for this K, with
        mu_j = exp(f_j) * w_j the fitted rates and W their diagonal matrix.
        The first term is the Poisson log-likelihood of the counts at the
        mode including the log-factorial of every count, so that the value
        is a proper log-probability.

    Raises
    ------
    ValueError
        If counts is not a finite one-dimensional array of non-negative
        integer values, if widths is not a finite one-dimensional array of
        the same length with every entry above zero, or if K is not a
        finite (N, N) matrix that is symmetric to within 1e-9 and positive
        definite.
    """
    return log_marginal

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammaln


def _oracle_laplace_log_marginal_likelihood(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> float:
    a, w = _check_counts_widths(counts, widths)
    prior = _check_covariance(K, a.shape[0])
    f = _oracle_lgcp_posterior_mode(a, w, prior)
    rates = np.exp(f) * w
    log_likelihood = float(np.sum(a * np.log(rates) - rates - gammaln(a + 1.0)))
    penalty = 0.5 * float(f @ np.linalg.solve(prior, f))
    _, log_det = np.linalg.slogdet(np.eye(a.shape[0]) + prior * rates[None, :])
    return log_likelihood - penalty - 0.5 * float(log_det)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "laplace_log_marginal_likelihood"),
                           ("run_gold", "_oracle_laplace_log_marginal_likelihood")):
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
        # the benchmark template at the selected length scale
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 10.0)\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # the shortest length scale of the benchmark grid
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.1, 10.0)\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # the longest length scale of the benchmark grid, strongly disfavoured
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.8, 10.0)\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # a small amplitude and a narrow mean-function prior
        {
            "setup": kernel + "K = kernel(x, 0.2, 0.3, 1.0)\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # edge: a sparse template with empty bins, where the log-factorials vanish for those bins
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.2, 3.0)\n"
                     "a = np.array([9, 4, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0])\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # boundary: a single bin
        {
            "setup": "import numpy as np\na = np.array([37])\nw = np.array([0.25])\nK = np.array([[4.0]])\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # edge: non-uniform widths
        {
            "setup": kernel + "w = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])\n"
                     "K = kernel(x, 1.0, 0.3, 10.0)\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        # edge: a very broad mean-function prior, where the log-determinant is dominated by the basis directions
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 1000.0)\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-7,
        },
        # stress: forty bins with large counts, where the log-factorials are of order ten thousand
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "w = np.full(40, 0.025)\n"
                     "a = np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2) + 12 * np.cos(9 * x) ** 2).astype(int)\n"
                     "r = np.abs(x[:, None] - x[None, :]) / 0.15\n"
                     "K = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + 25.0 * np.outer(np.ones(40), np.ones(40))\n",
            "call": "laplace_log_marginal_likelihood(a, w, K)",
            "gold_call": "_oracle_laplace_log_marginal_likelihood(a, w, K)",
            "tol": 1e-9,
        },
        {
            "setup": kernel + "# invalid: a negative count\n"
                     "a = np.array([2947, 2193, -1, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
                     "args = (a, w, kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: widths of the wrong length\nargs = (a, w[:11], kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a covariance that is not positive definite\nargs = (a, w, np.zeros((12, 12)))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
