"""
Compute the Laplace approximation to the posterior covariance of the log-rate of a log-Gaussian Cox process at its mode.

At the posterior mode the negative Hessian of the penalised log-likelihood is the sum of the prior precision and the diagonal matrix of fitted Poisson rates, so the Gaussian approximation to the posterior has covariance $\Sigma = (K^{-1} + W)^{-1}$ with $W = \mathrm{diag}(\hat\mu_j)$ and $\hat\mu_j = e^{\hat f_j} w_j$. This covariance encodes correlated uncertainty across all bins: the smoothness prior pools the information of neighbouring bins, and the matrix identity $(K^{-1} + W)^{-1} \preceq W^{-1}$ shows that the posterior variance of every bin is bounded above by the inverse fitted rate.

Returns
-------
np.ndarray, Array of shape (N, N) equal to the inverse of K^{-1} + W, with W the diagonal matrix of the fitted rates exp(f_hat_j) * w_j. The result is symmetric and positive definite, and its diagonal entries never exceed 1 / (exp(f_hat_j) * w_j).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def laplace_posterior_covariance(f_hat: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    r"""Return the Laplace posterior covariance of the log-rate at the given mode.

    Parameters
    ----------
    f_hat : np.ndarray
        Posterior mode of the log-rate, shape (N,), N at least one, finite.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite.

    Returns
    -------
    Sigma : np.ndarray
        Array of shape (N, N) equal to the inverse of K^{-1} + W, with W
        the diagonal matrix of the fitted rates exp(f_hat_j) * w_j. The
        result is symmetric and positive definite, and its diagonal
        entries never exceed 1 / (exp(f_hat_j) * w_j).

    Raises
    ------
    ValueError
        If f_hat is not a finite one-dimensional array, if widths is not
        a finite one-dimensional array of the same length with every entry
        above zero, or if K is not a finite (N, N) matrix that is
        symmetric to within 1e-9 and positive definite.
    """
    return Sigma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_laplace_posterior_covariance(f_hat: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    f = np.asarray(f_hat, dtype=float)
    w = np.asarray(widths, dtype=float)
    if f.ndim != 1 or f.shape[0] < 1 or not np.all(np.isfinite(f)):
        raise ValueError("f_hat must be a finite one-dimensional array")
    if w.shape != f.shape or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("widths must be a finite one-dimensional array of the same length with positive entries")
    prior = _check_covariance(K, f.shape[0])
    rates = np.exp(f) * w
    covariance = np.linalg.inv(np.linalg.inv(prior) + np.diag(rates))
    return 0.5 * (covariance + covariance.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "laplace_posterior_covariance"),
                           ("run_gold", "_oracle_laplace_posterior_covariance")):
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
              "f = np.array([10.47, 10.17, 9.81, 9.49, 9.42, 9.62, 9.84, 9.51, 8.79, 7.98, 7.20, 6.75])\n")
    return [
        # a log-rate close to the benchmark mode with the selected kernel
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 10.0)\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-8,
        },
        # a short length scale, where the covariance approaches the diagonal inverse rates
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.05, 10.0)\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-8,
        },
        # a low log-rate, where the prior dominates and the covariance approaches the prior
        {
            "setup": kernel + "K = kernel(x, 0.5, 0.3, 2.0)\nf = f - 9.0\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-8,
        },
        # boundary: a single bin, where the variance is 1 / (1 / K + rate)
        {
            "setup": "import numpy as np\nf = np.array([2.5])\nw = np.array([0.4])\nK = np.array([[0.7]])\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-9,
        },
        # edge: non-uniform widths and a diagonal prior, where the result is diagonal
        {
            "setup": kernel + "w = np.linspace(0.02, 0.2, 12)\nK = np.diag(np.linspace(0.1, 2.0, 12))\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-9,
        },
        # edge: an ill-conditioned prior with a broad mean-function term, where a naive inversion loses digits
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 1000.0)\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-6,
        },
        # edge: a large log-rate makes the rates huge and the posterior variance tiny
        {
            "setup": kernel + "K = kernel(x, 1.0, 0.3, 10.0)\nf = f + 6.0\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-8,
        },
        # stress: forty bins of the paper's grid
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "w = np.full(40, 0.025)\n"
                     "f = np.log(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2)) - np.log(w)\n"
                     "r = np.abs(x[:, None] - x[None, :]) / 0.15\n"
                     "K = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + 25.0 * np.outer(np.ones(40), np.ones(40))\n",
            "call": "laplace_posterior_covariance(f, w, K)",
            "gold_call": "_oracle_laplace_posterior_covariance(f, w, K)",
            "tol": 1e-8,
        },
        {
            "setup": kernel + "# invalid: a non-finite mode entry\nf[4] = np.inf\nargs = (f, w, kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a covariance of the wrong size\nargs = (f, w, kernel(x[:11], 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a covariance that is not positive definite\n"
                     "K = kernel(x, 1.0, 0.3, 10.0); K[5, 5] = -K[5, 5]\nargs = (f, w, K)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": kernel + "# invalid: a negative bin width\nw[0] = -w[0]\nargs = (f, w, kernel(x, 1.0, 0.3, 10.0))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
