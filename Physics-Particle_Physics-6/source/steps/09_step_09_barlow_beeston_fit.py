"""
Fit the signal strength of a single-channel template likelihood with the histogram background template, the Barlow–Beeston treatment of its Monte Carlo statistical uncertainty and the interpolation of its systematic shape variations by the standard framework's code 4.

The histogram template of bin $j$ is the raw Monte Carlo count scaled to the data luminosity, $a_j / \tau$, and its statistical uncertainty enters the likelihood through one multiplicative factor $\gamma_j$ per bin with a Gaussian constraint centred at unity whose width $1/\sqrt{a_j}$ reflects the raw count. The factors are independent: the constraint on $\gamma_j$ involves bin $j$ alone, with no coupling to its neighbours, which is exact for a histogram of independent Poisson counts but discards the smoothness of the underlying distribution. Every systematic source adds one nuisance parameter $\alpha_k$ with a unit Gaussian constraint that interpolates between the nominal histogram and the histograms of the two variation templates: exponential extrapolation beyond one standard deviation, and inside it the sixth-order polynomial that matches the value, slope and curvature of the two exponential branches at $\alpha_k = \pm 1$, so that the factor is twice continuously differentiable. The variation templates enter with their own bin-by-bin Monte Carlo noise. The number of nuisance parameters equals the number of bins plus the number of sources, and the joint profile over all of them attains the semiparametric efficiency bound at the price of a rich, weakly constrained nuisance space.

Returns
-------
tuple, Two native Python floats (mu_hat, sigma_mu). With h_j = a_j / tau the histogram template and h_j^{k,+}, h_j^{k,-} the variation templates of source k scaled in the same way, the expected count of bin j is nu_j = mu * s_j + gamma_j * h_j * prod_k F_jk(alpha_k). With r_+ = h_j^{k,+} / h_j and r_- = h_j^{k,-} / h_j, the interpolation factor is F_jk = r_+ ** alpha_k for alpha_k >= 1, r_- ** (-alpha_k) for alpha_k <= -1, and for |alpha_k| < 1 the polynomial 1 + sum_{i=1}^{6} c_i alpha_k ** i whose six coefficients are fixed by matching the value, the first and the second derivative of the two exponential branches at alpha_k = 1 and alpha_k = -1. The negative log-likelihood sum_j (nu_j - n_j log nu_j) + sum_j a_j (gamma_j - 1)^2 / 2 + sum_k alpha_k^2 / 2 is minimised jointly over the signal strength mu, which may take any real value that keeps every nu_j above zero, the N factors gamma_j and the source parameters alpha_k. mu_hat is the minimising signal strength, converged to within 1e-9, and sigma_mu is the square root of the (mu, mu) entry of the inverse Hessian of the negative log-likelihood at the minimum, with the Hessian taken over (mu, gamma_1, ..., gamma_N, alpha_1, ..., alpha_K).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def barlow_beeston_fit(observed: "np.ndarray", signal: "np.ndarray", counts: "np.ndarray", variations: list, tau: float) -> tuple:
    r"""Return the fitted signal strength and its parabolic uncertainty with per-bin gamma factors and interpolated shape systematics.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts of shape (N,), N at least one, finite,
        non-negative integers.
    signal : np.ndarray
        Expected signal counts at unit signal strength, shape (N,), finite,
        non-negative, with at least one entry above zero.
    counts : np.ndarray
        Raw Monte Carlo background counts of shape (N,), finite, integers
        above zero.
    variations : list
        Sequence of pairs (counts_plus, counts_minus), possibly empty, one
        pair per systematic source, each a finite array of integers above
        zero of length N.
    tau : float
        Ratio of the Monte Carlo to the data luminosity, finite and above
        zero.

    Returns
    -------
    result : tuple
        Two native Python floats (mu_hat, sigma_mu). With h_j = a_j / tau
        the histogram template and h_j^{k,+}, h_j^{k,-} the variation
        templates of source k scaled in the same way, the expected count
        of bin j is nu_j = mu * s_j + gamma_j * h_j * prod_k F_jk(alpha_k).
        With r_+ = h_j^{k,+} / h_j and r_- = h_j^{k,-} / h_j, the
        interpolation factor is F_jk = r_+ ** alpha_k for alpha_k >= 1,
        r_- ** (-alpha_k) for alpha_k <= -1, and for |alpha_k| < 1 the
        polynomial 1 + sum_{i=1}^{6} c_i alpha_k ** i whose six
        coefficients are fixed by matching the value, the first and the
        second derivative of the two exponential branches at alpha_k = 1
        and alpha_k = -1. The negative log-likelihood sum_j (nu_j
        - n_j log nu_j) + sum_j a_j (gamma_j - 1)^2 / 2 + sum_k alpha_k^2
        / 2 is minimised jointly over the signal strength mu, which may
        take any real value that keeps every nu_j above zero, the N
        factors gamma_j and the source parameters alpha_k. mu_hat is the
        minimising signal strength, converged to within 1e-9, and sigma_mu
        is the square root of the (mu, mu) entry of the inverse Hessian of
        the negative log-likelihood at the minimum, with the Hessian taken
        over (mu, gamma_1, ..., gamma_N, alpha_1, ..., alpha_K).

    Raises
    ------
    ValueError
        If observed is not a finite one-dimensional array of non-negative
        integers, if signal is not a finite non-negative array of the same
        length with an entry above zero, if counts or any variation array
        is not a finite array of the same length of integers above zero,
        if variations is not a sequence of pairs, or if tau is not finite
        or not above zero.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _positive_counts(values, size: int, label: str) -> "np.ndarray":
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0) or np.any(array != np.round(array)):
        raise ValueError(f"{label} must be a finite array of integers above zero with one entry per bin")
    return array


def _code4_coefficients(ratio_up: "np.ndarray", ratio_down: "np.ndarray") -> "np.ndarray":
    """Polynomial coefficients c_1..c_6 per bin matching value, slope and curvature of the exponential branches at +-1."""
    log_up, log_down = np.log(ratio_up), np.log(ratio_down)
    # rows: F(1), F'(1), F''(1), F(-1), F'(-1), F''(-1) of the polynomial 1 + sum c_i alpha^i
    system = np.array([[1, 1, 1, 1, 1, 1],
                       [1, 2, 3, 4, 5, 6],
                       [0, 2, 6, 12, 20, 30],
                       [-1, 1, -1, 1, -1, 1],
                       [1, -2, 3, -4, 5, -6],
                       [0, 2, -6, 12, -20, 30]], dtype=float)
    targets = np.stack([ratio_up - 1.0, ratio_up * log_up, ratio_up * log_up ** 2,
                        ratio_down - 1.0, -ratio_down * log_down, ratio_down * log_down ** 2])
    return np.linalg.solve(system, targets)          # shape (6, N)


def _code4_factor(alpha: float, ratio_up: "np.ndarray", ratio_down: "np.ndarray", coefficients: "np.ndarray") -> tuple:
    """Interpolation factor per bin with its first and second derivative in alpha."""
    if alpha >= 1.0:
        log_ratio = np.log(ratio_up)
        value = ratio_up ** alpha
        return value, value * log_ratio, value * log_ratio ** 2
    if alpha <= -1.0:
        log_ratio = np.log(ratio_down)
        value = ratio_down ** (-alpha)
        return value, -value * log_ratio, value * log_ratio ** 2
    powers = alpha ** np.arange(1, 7)
    degrees = np.arange(1, 7, dtype=float)
    value = 1.0 + coefficients.T @ powers
    first = coefficients.T @ (degrees * alpha ** np.arange(0, 6))
    second = coefficients.T @ (degrees * (degrees - 1.0) * np.concatenate([[0.0], alpha ** np.arange(0, 5)]))
    return value, first, second


def _oracle_barlow_beeston_fit(observed: "np.ndarray", signal: "np.ndarray", counts: "np.ndarray", variations: list, tau: float) -> tuple:
    a = np.asarray(counts, dtype=float)
    if a.ndim != 1 or a.shape[0] < 1:
        raise ValueError("counts must be a one-dimensional array")
    a = _positive_counts(a, a.shape[0], "counts")
    ratio = float(tau)
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError("tau must be finite and above zero")
    n, s, template = _check_channel(observed, signal, a / ratio)
    pairs = list(variations)
    ratios_up, ratios_down, coefficients = [], [], []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("every variation must be a pair of count arrays")
        up = _positive_counts(pair[0], a.shape[0], "counts_plus") / a
        down = _positive_counts(pair[1], a.shape[0], "counts_minus") / a
        ratios_up.append(up)
        ratios_down.append(down)
        coefficients.append(_code4_coefficients(up, down))
    size = a.shape[0]
    n_sources = len(pairs)

    def _objective(point):
        mu, gamma, alpha = point[0], point[1:size + 1], point[size + 1:]
        factors = np.ones((n_sources, size))
        first = np.zeros((n_sources, size))
        second = np.zeros((n_sources, size))
        for k in range(n_sources):
            factors[k], first[k], second[k] = _code4_factor(float(alpha[k]), ratios_up[k], ratios_down[k], coefficients[k])
        background = gamma * template * np.prod(factors, axis=0)
        nu = mu * s + background
        if np.any(nu <= 0.0):
            return math.inf, None, None
        value = float(np.sum(nu - n * np.log(nu)) + 0.5 * np.sum(a * (gamma - 1.0) ** 2) + 0.5 * np.sum(alpha ** 2))
        residual = 1.0 - n / nu
        curvature = n / nu ** 2
        slope = first / factors                                   # d log F / d alpha per source and bin
        d_nu = np.column_stack([s, np.diag(background / gamma), (background[None, :] * slope).T])
        gradient = d_nu.T @ residual
        gradient[1:size + 1] += a * (gamma - 1.0)
        gradient[size + 1:] += alpha
        hessian = d_nu.T @ (curvature[:, None] * d_nu)
        weighted = residual * background
        hessian[1:size + 1, size + 1:] += (slope * (weighted / gamma)[None, :]).T
        hessian[size + 1:, 1:size + 1] += slope * (weighted / gamma)[None, :]
        cross = (slope * weighted[None, :]) @ slope.T             # products of first derivatives of different sources
        np.fill_diagonal(cross, (second / factors) @ weighted)     # second derivative of one source
        hessian[size + 1:, size + 1:] += cross
        hessian[1:size + 1, 1:size + 1] += np.diag(a)
        hessian[size + 1:, size + 1:] += np.eye(n_sources)
        return value, gradient, hessian

    start = np.concatenate([[1.0], np.ones(size), np.zeros(n_sources)])
    point, hessian = _newton_minimise(_objective, start)
    covariance = np.linalg.inv(hessian)
    return (float(point[0]), float(math.sqrt(covariance[0, 0])))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "barlow_beeston_fit"),
                           ("run_gold", "_oracle_barlow_beeston_fit")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    channel = ("import numpy as np\n"
               "n = np.array([295, 238, 137, 169, 110, 132, 151, 115, 50, 10, 13, 11])\n"
               "s = np.array([0.0, 0.29, 10.09, 40.82, 28.79, 12.63, 3.67, 0.41, 0.02, 0.0, 0.0, 0.0])\n"
               "a = np.array([2947, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
               "cal = (np.array([3375, 2420, 1672, 1202, 1037, 1242, 1498, 1384, 750, 301, 140, 79]), np.array([2557, 1887, 1339, 1056, 1115, 1471, 1441, 939, 397, 151, 92, 67]))\n"
               "res = (np.array([3016, 2107, 1507, 1120, 1136, 1348, 1623, 1175, 578, 257, 138, 80]), np.array([2985, 2086, 1556, 1141, 1032, 1299, 1514, 1126, 526, 230, 108, 83]))\n"
               "norm = (np.array([3098, 2180, 1550, 1186, 1162, 1414, 1727, 1246, 594, 209, 125, 69]), np.array([2938, 2160, 1463, 1093, 1053, 1262, 1389, 1029, 477, 210, 127, 86]))\n")
    return [
        # the benchmark channel with its three systematic sources
        {
            "setup": channel,
            "call": "barlow_beeston_fit(n, s, a, [cal, res, norm], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [cal, res, norm], 10.0)",
            "tol": 1e-7,
        },
        # the statistical factors alone
        {
            "setup": channel,
            "call": "barlow_beeston_fit(n, s, a, [], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [], 10.0)",
            "tol": 1e-7,
        },
        # a single source
        {
            "setup": channel,
            "call": "barlow_beeston_fit(n, s, a, [cal], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [cal], 10.0)",
            "tol": 1e-7,
        },
        # the variation pair in the opposite order, which changes the interpolation branches and the fitted values
        {
            "setup": channel,
            "call": "barlow_beeston_fit(n, s, a, [(cal[1], cal[0])], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [(cal[1], cal[0])], 10.0)",
            "tol": 1e-7,
        },
        # a small Monte Carlo sample, where the gamma factors are loosely constrained
        {
            "setup": channel + "a = np.array([59, 44, 31, 22, 21, 26, 31, 23, 11, 5, 2, 1])\n"
                     "up = np.array([68, 48, 34, 24, 21, 25, 30, 28, 15, 6, 3, 2]); down = np.array([51, 38, 27, 21, 22, 29, 29, 19, 8, 3, 2, 1])\n",
            "call": "barlow_beeston_fit(n, s, a, [(up, down)], 0.2)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [(up, down)], 0.2)",
            "tol": 1e-7,
        },
        # a huge Monte Carlo sample, where the fit approaches the fixed-template fit with interpolated systematics
        {
            "setup": channel + "big = a * 1000\nbig_cal = (cal[0] * 1000, cal[1] * 1000)\n",
            "call": "barlow_beeston_fit(n, s, big, [big_cal], 10000.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, big, [big_cal], 10000.0)",
            "tol": 1e-7,
        },
        # data without signal, where the fitted strength is negative
        {
            "setup": channel + "n = np.array([292, 221, 150, 108, 101, 125, 160, 118, 55, 22, 12, 6])\n",
            "call": "barlow_beeston_fit(n, s, a, [cal, res, norm], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [cal, res, norm], 10.0)",
            "tol": 1e-7,
        },
        # a large injected signal
        {
            "setup": channel + "n = np.array([300, 218, 190, 260, 210, 175, 162, 116, 54, 26, 11, 7])\n",
            "call": "barlow_beeston_fit(n, s, a, [cal, res, norm], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [cal, res, norm], 10.0)",
            "tol": 1e-7,
        },
        # edge: a pure normalisation source, whose log-ratios are the same in every bin
        {
            "setup": channel + "up = np.round(a * 1.2).astype(int); down = np.round(a * 0.8).astype(int)\n",
            "call": "barlow_beeston_fit(n, s, a, [(up, down)], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [(up, down)], 10.0)",
            "tol": 1e-7,
        },
        # edge: an asymmetric source whose upward variation is far larger than the downward one, so that the branch matters
        {
            "setup": channel + "up = np.round(a * 1.5).astype(int); down = np.round(a * 0.98).astype(int)\n",
            "call": "barlow_beeston_fit(n, s, a, [(up, down)], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [(up, down)], 10.0)",
            "tol": 1e-7,
        },
        # edge: empty observed bins pull the corresponding gamma factors below unity
        {
            "setup": channel + "n = np.array([295, 238, 137, 169, 110, 132, 151, 115, 50, 0, 0, 0])\n",
            "call": "barlow_beeston_fit(n, s, a, [cal], 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [cal], 10.0)",
            "tol": 1e-7,
        },
        # boundary: a single bin with one source
        {
            "setup": "import numpy as np\nn = np.array([130])\ns = np.array([20.0])\na = np.array([500])\n",
            "call": "barlow_beeston_fit(n, s, a, [(np.array([560]), np.array([450]))], 5.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [(np.array([560]), np.array([450]))], 5.0)",
            "tol": 1e-7,
        },
        # edge: a luminosity ratio below one, the Monte Carlo sample smaller than the data
        {
            "setup": channel + "a = np.array([148, 110, 76, 55, 51, 64, 79, 57, 28, 12, 6, 4])\n"
                     "up = np.array([169, 121, 84, 60, 52, 62, 75, 69, 38, 15, 7, 4]); down = np.array([128, 94, 67, 53, 56, 74, 72, 47, 20, 8, 5, 3])\n",
            "call": "barlow_beeston_fit(n, s, a, [(up, down)], 0.5)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, [(up, down)], 0.5)",
            "tol": 1e-7,
        },
        # stress: forty bins with four sources
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "base = 4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2)\n"
                     "a = np.round(base + 12 * np.cos(9 * x) ** 2).astype(int)\n"
                     "pairs = [(np.round(base * (1 + 0.1 * x)).astype(int), np.round(base * (1 - 0.1 * x)).astype(int)),\n"
                     "         (np.round(base * 1.15).astype(int), np.round(base * 0.85).astype(int)),\n"
                     "         (np.round(4000 * np.exp(-4 * (x - 0.02)) + 1500 * np.exp(-0.5 * ((x - 0.57) / 0.1) ** 2)).astype(int), np.round(4000 * np.exp(-4 * (x + 0.02)) + 1500 * np.exp(-0.5 * ((x - 0.53) / 0.1) ** 2)).astype(int)),\n"
                     "         (np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.11) ** 2)).astype(int), np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.09) ** 2)).astype(int))]\n"
                     "s = 30 * np.exp(-0.5 * ((x - 0.3) / 0.06) ** 2)\n"
                     "n = np.round(base / 10 + 0.8 * s + 3 * np.sin(11 * x)).astype(int)\n",
            "call": "barlow_beeston_fit(n, s, a, pairs, 10.0)",
            "gold_call": "_oracle_barlow_beeston_fit(n, s, a, pairs, 10.0)",
            "tol": 1e-7,
        },
        {
            "setup": channel + "# invalid: an empty Monte Carlo bin\na = a.copy(); a[11] = 0\nargs = (n, s, a, [cal], 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: an empty bin in a variation template\nbad = cal[0].copy(); bad[11] = 0\nargs = (n, s, a, [(bad, cal[1])], 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a luminosity ratio of zero\nargs = (n, s, a, [cal], 0.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a negative observed count\nn = n.copy(); n[3] = -1\nargs = (n, s, a, [cal], 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a variation that is not a pair\nargs = (n, s, a, [(cal[0],)], 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
