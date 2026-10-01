"""
Build the combined covariance of the log-rate of a template, the Laplace posterior covariance of the nominal Monte Carlo sample plus one rank-1 contribution for every systematic shape variation.

Systematic shape variations are provided as Monte Carlo templates generated at the $+1\sigma$ and $-1\sigma$ points of every source. Fitting the same Gaussian process to each variation template and differencing the posterior modes gives the direction of the variation in log-rate space, free of the bin-by-bin noise that contaminates a histogram-based estimate because the smoothing is applied before the differencing. Every direction contributes an outer product to the covariance, so that the combined matrix encodes statistical and systematic template uncertainty in one object whose eigenmodes span both. In the limit of negligible statistical uncertainty a single systematic reduces the covariance to one outer product and the exponential mode parametrisation of the template reproduces the piecewise-exponential interpolation of histogram templates.

Returns
-------
np.ndarray, Array of shape (N, N): the Laplace posterior covariance of the log-rate at the posterior mode of the nominal counts, plus for every systematic source the outer product delta delta^T with delta = (f_plus - f_minus) / 2, where f_plus and f_minus are the posterior modes of the log-rate for the two variation templates under the same prior covariance K. The result is symmetric and positive definite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def combined_template_covariance(counts: "np.ndarray", variations: list, widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    r"""Return the combined statistical and systematic covariance of the log-rate.

    Parameters
    ----------
    counts : np.ndarray
        Nominal Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers.
    variations : list
        Sequence of pairs (counts_plus, counts_minus), possibly empty, one
        pair per systematic source, each entry a count array with the
        same requirements and length as counts.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite,
        used for the nominal and for every variation template.

    Returns
    -------
    Sigma_comb : np.ndarray
        Array of shape (N, N): the Laplace posterior covariance of the
        log-rate at the posterior mode of the nominal counts, plus for
        every systematic source the outer product delta delta^T with
        delta = (f_plus - f_minus) / 2, where f_plus and f_minus are the
        posterior modes of the log-rate for the two variation templates
        under the same prior covariance K. The result is symmetric and
        positive definite.

    Raises
    ------
    ValueError
        If counts, widths or K violate their stated requirements, if
        variations is not a sequence of pairs, or if any variation array
        is not a finite one-dimensional array of non-negative integers of
        the same length as counts.
    """
    return Sigma_comb

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_combined_template_covariance(counts: "np.ndarray", variations: list, widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    a, w = _check_counts_widths(counts, widths)
    prior = _check_covariance(K, a.shape[0])
    pairs = list(variations)
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("every variation must be a pair of count arrays")
        for entry in pair:
            varied, _ = _check_counts_widths(entry, w)
            if varied.shape != a.shape:
                raise ValueError("every variation array must match the length of counts")
    mode = _oracle_lgcp_posterior_mode(a, w, prior)
    covariance = _oracle_laplace_posterior_covariance(mode, w, prior)
    for plus, minus in pairs:
        direction = 0.5 * (_oracle_lgcp_posterior_mode(plus, w, prior) - _oracle_lgcp_posterior_mode(minus, w, prior))
        covariance = covariance + np.outer(direction, direction)
    return covariance

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "combined_template_covariance"),
                           ("run_gold", "_oracle_combined_template_covariance")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    sample = ("import numpy as np\n"
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
              "K = kernel(x, 1.0, 0.3, 10.0)\n"
              "a = np.array([2947, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
              "cal = (np.array([3375, 2420, 1672, 1202, 1037, 1242, 1498, 1384, 750, 301, 140, 79]), np.array([2557, 1887, 1339, 1056, 1115, 1471, 1441, 939, 397, 151, 92, 67]))\n"
              "res = (np.array([3016, 2107, 1507, 1120, 1136, 1348, 1623, 1175, 578, 257, 138, 80]), np.array([2985, 2086, 1556, 1141, 1032, 1299, 1514, 1126, 526, 230, 108, 83]))\n"
              "norm = (np.array([3098, 2180, 1550, 1186, 1162, 1414, 1727, 1246, 594, 209, 125, 69]), np.array([2938, 2160, 1463, 1093, 1053, 1262, 1389, 1029, 477, 210, 127, 86]))\n")
    return [
        # the benchmark: three systematic sources
        {
            "setup": sample,
            "call": "combined_template_covariance(a, [cal, res, norm], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [cal, res, norm], w, K)",
            "tol": 1e-8,
        },
        # a single source
        {
            "setup": sample,
            "call": "combined_template_covariance(a, [cal], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [cal], w, K)",
            "tol": 1e-8,
        },
        # boundary: no systematic source, the statistical covariance alone
        {
            "setup": sample,
            "call": "combined_template_covariance(a, [], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [], w, K)",
            "tol": 1e-8,
        },
        # the variation pair given in the opposite order flips the sign of the direction but not the outer product
        {
            "setup": sample,
            "call": "combined_template_covariance(a, [(cal[1], cal[0])], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [(cal[1], cal[0])], w, K)",
            "tol": 1e-8,
        },
        # edge: a variation identical to the nominal template contributes nothing
        {
            "setup": sample,
            "call": "combined_template_covariance(a, [(a.copy(), a.copy())], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [(a.copy(), a.copy())], w, K)",
            "tol": 1e-8,
        },
        # edge: a pure normalisation variation, whose direction is nearly constant across bins
        {
            "setup": sample + "up = np.round(a * 1.2).astype(int)\ndown = np.round(a * 0.8).astype(int)\n",
            "call": "combined_template_covariance(a, [(up, down)], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [(up, down)], w, K)",
            "tol": 1e-8,
        },
        # edge: the same source listed twice doubles its contribution
        {
            "setup": sample,
            "call": "combined_template_covariance(a, [res, res], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [res, res], w, K)",
            "tol": 1e-8,
        },
        # a short length scale, where the statistical part is nearly diagonal and the systematic parts dense
        {
            "setup": sample + "K = kernel(x, 1.0, 0.05, 10.0)\n",
            "call": "combined_template_covariance(a, [cal, norm], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [cal, norm], w, K)",
            "tol": 1e-8,
        },
        # edge: sparse variation templates with empty bins
        {
            "setup": sample + "K = kernel(x, 1.0, 0.2, 3.0)\n"
                     "a = np.array([9, 4, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0])\n"
                     "sparse = (np.array([12, 5, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0]), np.array([7, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]))\n",
            "call": "combined_template_covariance(a, [sparse], w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, [sparse], w, K)",
            "tol": 1e-8,
        },
        # stress: forty bins with four sources
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "w = np.full(40, 0.025)\n"
                     "r = np.abs(x[:, None] - x[None, :]) / 0.15\n"
                     "K = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + 25.0 * np.outer(np.ones(40), np.ones(40))\n"
                     "base = 4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2)\n"
                     "a = np.round(base).astype(int)\n"
                     "pairs = [(np.round(base * (1 + 0.1 * x)).astype(int), np.round(base * (1 - 0.1 * x)).astype(int)),\n"
                     "         (np.round(base * 1.15).astype(int), np.round(base * 0.85).astype(int)),\n"
                     "         (np.round(4000 * np.exp(-4 * (x - 0.02)) + 1500 * np.exp(-0.5 * ((x - 0.57) / 0.1) ** 2)).astype(int), np.round(4000 * np.exp(-4 * (x + 0.02)) + 1500 * np.exp(-0.5 * ((x - 0.53) / 0.1) ** 2)).astype(int)),\n"
                     "         (np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.11) ** 2)).astype(int), np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.09) ** 2)).astype(int))]\n",
            "call": "combined_template_covariance(a, pairs, w, K)",
            "gold_call": "_oracle_combined_template_covariance(a, pairs, w, K)",
            "tol": 1e-8,
        },
        {
            "setup": sample + "# invalid: a variation array of the wrong length\nargs = (a, [(cal[0][:11], cal[1])], w, K)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a variation with a negative count\nbad = cal[0].copy(); bad[0] = -1\nargs = (a, [(bad, cal[1])], w, K)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a variation that is not a pair\nargs = (a, [(cal[0], cal[1], cal[0])], w, K)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a covariance that is not positive definite\nargs = (a, [cal], w, -K)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
