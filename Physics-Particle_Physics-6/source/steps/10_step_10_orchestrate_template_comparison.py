"""
Run the complete pipeline: select the kernel length scale from the nominal Monte Carlo template, build the smooth template and its combined statistical and systematic covariance, fit the signal strength with the truncated eigenmodes, fit it again with the histogram template that uses per-bin Barlow–Beeston factors and interpolated shape systematics, and report the residual between the two fitted signal strengths.

The Gaussian process template replaces the histogram and the whole chain of per-bin modifiers by one posterior mean and one covariance whose leading eigenmodes carry both statistical and systematic uncertainty. The smooth template is the posterior rate scaled to the data luminosity, and the comparison with the histogram treatment on the same data, the same signal template, the same Monte Carlo samples and the same systematic sources isolates the effect of the template representation. The residual between the signal strength of the eigenmode fit and that of the histogram fit on the same data is the pseudo-experiment-level comparison of the two representations, whose mean over an ensemble measures the plug-in penalty of smoothing the template once before the fit. The exact variance bound compares the posterior log-rate standard deviation with the inverse square root of the fitted Monte Carlo mean. The returned ratio instead uses the raw-count histogram uncertainty; it is below one in every bin of the benchmark, but that is not a universal consequence of the bound.

Returns
-------
tuple, Eight native Python floats (mu_gp, sigma_gp, n_modes, mu_bb, sigma_bb, ell, ratio, residual). The bin centres are the midpoints of the edges and the widths their differences. ell is the length scale of ell_grid selected on the nominal counts by the Laplace log marginal likelihood with the given amplitude, mean-function basis on the range [edges[0], edges[-1]] and prior scale. With the effective prior covariance at that length scale, the smooth template of bin j is exp(f_j) w_j / tau with f the posterior mode of the nominal counts, and the combined covariance adds one rank-1 term per systematic source to the Laplace posterior covariance. mu_gp and sigma_gp are the fitted signal strength and its parabolic uncertainty from the eigenmode template fit at the given fraction, n_modes the number of retained eigenmodes as a float, mu_bb and sigma_bb the corresponding results of the Barlow–Beeston histogram fit with the same signal, observed counts, nominal counts, variation pairs and tau, ratio the mean over bins of the posterior standard deviation of the log-rate (square root of the diagonal of the Laplace posterior covariance of the nominal template) divided by the histogram relative uncertainty 1 / sqrt(a_j), and residual the difference mu_gp - mu_bb between the two fitted signal strengths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orchestrate_template_comparison(edges: "np.ndarray", counts: "np.ndarray", variations: list, signal: "np.ndarray", observed: "np.ndarray", tau: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray", fraction: float) -> tuple:
    r"""Return the Gaussian process eigenmode fit, the histogram fit, their residual and the template diagnostics.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges of shape (N + 1,), N at least one, finite and strictly
        increasing.
    counts : np.ndarray
        Nominal Monte Carlo background counts of shape (N,), finite,
        integers above zero.
    variations : list
        Sequence of pairs (counts_plus, counts_minus), possibly empty, one
        pair per systematic source, each a finite array of integers above
        zero of length N.
    signal : np.ndarray
        Expected signal counts at unit signal strength, shape (N,), finite,
        non-negative, with at least one entry above zero.
    observed : np.ndarray
        Observed counts of shape (N,), finite, non-negative integers.
    tau : float
        Ratio of the Monte Carlo to the data luminosity, finite and above
        zero.
    n_basis : int
        Number of cubic B-spline basis functions of the mean function, at
        least four.
    sigma : float
        Amplitude of the Matérn 5/2 kernel, finite and above zero.
    prior_scale : float
        Standard deviation of the prior on the mean-function coefficients,
        finite and above zero.
    ell_grid : np.ndarray
        Candidate length scales, finite and above zero, at least one.
    fraction : float
        Fraction of the total variance retained in the eigenmodes, above
        zero and at most one.

    Returns
    -------
    result : tuple
        Eight native Python floats (mu_gp, sigma_gp, n_modes, mu_bb,
        sigma_bb, ell, ratio, residual). The bin centres are the midpoints of the
        edges and the widths their differences. ell is the length scale
        of ell_grid selected on the nominal counts by the Laplace log
        marginal likelihood with the given amplitude, mean-function basis
        on the range [edges[0], edges[-1]] and prior scale. With the
        effective prior covariance at that length scale, the smooth
        template of bin j is exp(f_j) w_j / tau with f the posterior mode
        of the nominal counts, and the combined covariance adds one
        rank-1 term per systematic source to the Laplace posterior
        covariance. mu_gp and sigma_gp are the fitted signal strength and
        its parabolic uncertainty from the eigenmode template fit at the
        given fraction, n_modes the number of retained eigenmodes as a
        float, mu_bb and sigma_bb the corresponding results of the
        Barlow–Beeston histogram fit with the same signal, observed counts,
        nominal counts, variation pairs and tau, ratio the mean over bins of the
        posterior standard deviation of the log-rate (square root of the
        diagonal of the Laplace posterior covariance of the nominal
        template) divided by the histogram relative uncertainty
        1 / sqrt(a_j), and residual the difference mu_gp - mu_bb between
        the two fitted signal strengths.

    Raises
    ------
    ValueError
        If edges is not a finite strictly increasing one-dimensional array
        with at least two entries, or if any other argument violates the
        requirements stated for the steps that consume it.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orchestrate_template_comparison(edges: "np.ndarray", counts: "np.ndarray", variations: list, signal: "np.ndarray", observed: "np.ndarray", tau: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray", fraction: float) -> tuple:
    boundaries = np.asarray(edges, dtype=float)
    if boundaries.ndim != 1 or boundaries.shape[0] < 2 or not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("edges must be a finite strictly increasing one-dimensional array with at least two entries")
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])
    widths = np.diff(boundaries)
    a = np.asarray(counts, dtype=float)
    if a.shape != centres.shape or not np.all(np.isfinite(a)) or np.any(a <= 0.0) or np.any(a != np.round(a)):
        raise ValueError("counts must be integers above zero, one per bin")
    lo, hi = float(boundaries[0]), float(boundaries[-1])
    ell = _oracle_select_length_scale(a, widths, centres, lo, hi, n_basis, sigma, prior_scale, ell_grid)
    prior = _oracle_effective_kernel_matrix(centres, lo, hi, n_basis, sigma, ell, prior_scale)
    mode = _oracle_lgcp_posterior_mode(a, widths, prior)
    statistical = _oracle_laplace_posterior_covariance(mode, widths, prior)
    template = np.exp(mode) * widths / float(tau)
    combined = _oracle_combined_template_covariance(a, variations, widths, prior)
    n_modes = _oracle_truncated_eigenvalues(combined, fraction).shape[0]
    mu_gp, sigma_gp = _oracle_eigenmode_template_fit(observed, signal, template, combined, fraction)
    mu_bb, sigma_bb = _oracle_barlow_beeston_fit(observed, signal, a, variations, tau)
    ratio = float(np.mean(np.sqrt(np.diag(statistical)) * np.sqrt(a)))
    return (float(mu_gp), float(sigma_gp), float(n_modes), float(mu_bb), float(sigma_bb), float(ell), ratio, float(mu_gp - mu_bb))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "orchestrate_template_comparison"),
                           ("run_gold", "_oracle_orchestrate_template_comparison")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    sample = ("import numpy as np\n"
              "edges = np.linspace(0.0, 1.0, 13)\n"
              "a = np.array([2947, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
              "cal = (np.array([3375, 2420, 1672, 1202, 1037, 1242, 1498, 1384, 750, 301, 140, 79]), np.array([2557, 1887, 1339, 1056, 1115, 1471, 1441, 939, 397, 151, 92, 67]))\n"
              "res = (np.array([3016, 2107, 1507, 1120, 1136, 1348, 1623, 1175, 578, 257, 138, 80]), np.array([2985, 2086, 1556, 1141, 1032, 1299, 1514, 1126, 526, 230, 108, 83]))\n"
              "norm = (np.array([3098, 2180, 1550, 1186, 1162, 1414, 1727, 1246, 594, 209, 125, 69]), np.array([2938, 2160, 1463, 1093, 1053, 1262, 1389, 1029, 477, 210, 127, 86]))\n"
              "s = np.array([0.0, 0.29, 10.09, 40.82, 28.79, 12.63, 3.67, 0.41, 0.02, 0.0, 0.0, 0.0])\n"
              "n = np.array([295, 238, 137, 169, 110, 132, 151, 115, 50, 10, 13, 11])\n"
              "grid = np.array([0.1, 0.2, 0.3, 0.5, 0.8])\n")
    return [
        # the benchmark configuration
        {
            "setup": sample,
            "call": "orchestrate_template_comparison(edges, a, [cal, res, norm], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [cal, res, norm], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)",
            "tol": 1e-7,
        },
        # statistical uncertainty only, where more modes are needed and the fit moves
        {
            "setup": sample,
            "call": "orchestrate_template_comparison(edges, a, [], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)",
            "tol": 1e-7,
        },
        # the tighter variance threshold of the paper's alternative
        {
            "setup": sample,
            "call": "orchestrate_template_comparison(edges, a, [cal, res, norm], s, n, 10.0, 4, 1.0, 10.0, grid, 0.99)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [cal, res, norm], s, n, 10.0, 4, 1.0, 10.0, grid, 0.99)",
            "tol": 1e-7,
        },
        # a single-candidate grid, a richer mean function and a smaller amplitude
        {
            "setup": sample,
            "call": "orchestrate_template_comparison(edges, a, [cal, norm], s, n, 10.0, 6, 0.5, 3.0, np.array([0.2]), 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [cal, norm], s, n, 10.0, 6, 0.5, 3.0, np.array([0.2]), 0.95)",
            "tol": 1e-7,
        },
        # a small Monte Carlo sample at a luminosity ratio of one half, where the templates differ most
        {
            "setup": sample + "a = np.array([148, 110, 76, 55, 51, 64, 79, 57, 28, 12, 6, 4])\n"
                     "cal = (np.array([169, 121, 84, 60, 52, 62, 75, 69, 38, 15, 7, 4]), np.array([128, 94, 67, 53, 56, 74, 72, 47, 20, 8, 5, 3]))\n",
            "call": "orchestrate_template_comparison(edges, a, [cal], s, n, 0.5, 4, 1.0, 10.0, grid, 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [cal], s, n, 0.5, 4, 1.0, 10.0, grid, 0.95)",
            "tol": 1e-7,
        },
        # a template range in GeV with non-uniform edges
        {
            "setup": "import numpy as np\n"
                     "edges = np.array([105.0, 110.0, 115.0, 120.0, 124.0, 128.0, 132.0, 136.0, 140.0, 145.0, 150.0, 160.0])\n"
                     "a = np.array([412, 361, 318, 229, 208, 189, 171, 158, 172, 148, 241])\n"
                     "cal = (np.array([440, 370, 330, 240, 200, 185, 175, 165, 180, 150, 250]), np.array([390, 350, 305, 220, 215, 195, 165, 150, 165, 145, 230]))\n"
                     "s = np.array([0.0, 0.0, 0.1, 0.6, 2.4, 4.6, 2.2, 0.5, 0.1, 0.0, 0.0])\n"
                     "n = np.array([170, 140, 120, 95, 88, 84, 70, 60, 66, 55, 95])\n"
                     "grid = np.array([5.0, 10.0, 20.0, 40.0])\n",
            "call": "orchestrate_template_comparison(edges, a, [cal], s, n, 2.5, 4, 1.0, 10.0, grid, 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [cal], s, n, 2.5, 4, 1.0, 10.0, grid, 0.95)",
            "tol": 1e-7,
        },
        # boundary: two bins with one systematic source
        {
            "setup": "import numpy as np\n"
                     "edges = np.array([0.0, 0.5, 1.0])\n"
                     "a = np.array([2000, 800])\ncal = (np.array([2100, 720]), np.array([1900, 880]))\n"
                     "s = np.array([15.0, 5.0])\nn = np.array([215, 84])\n",
            "call": "orchestrate_template_comparison(edges, a, [cal], s, n, 10.0, 4, 1.0, 10.0, np.array([0.2, 0.5]), 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, [cal], s, n, 10.0, 4, 1.0, 10.0, np.array([0.2, 0.5]), 0.95)",
            "tol": 1e-7,
        },
        # stress: forty bins with four sources
        {
            "setup": "import numpy as np\n"
                     "edges = np.linspace(0.0, 1.0, 41)\n"
                     "x = 0.5 * (edges[:-1] + edges[1:])\n"
                     "base = 4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2)\n"
                     "a = np.round(base + 12 * np.cos(9 * x) ** 2).astype(int)\n"
                     "pairs = [(np.round(base * (1 + 0.1 * x)).astype(int), np.round(base * (1 - 0.1 * x)).astype(int)),\n"
                     "         (np.round(base * 1.15).astype(int), np.round(base * 0.85).astype(int)),\n"
                     "         (np.round(4000 * np.exp(-4 * (x - 0.02)) + 1500 * np.exp(-0.5 * ((x - 0.57) / 0.1) ** 2)).astype(int), np.round(4000 * np.exp(-4 * (x + 0.02)) + 1500 * np.exp(-0.5 * ((x - 0.53) / 0.1) ** 2)).astype(int)),\n"
                     "         (np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.11) ** 2)).astype(int), np.round(4000 * np.exp(-4 * x) + 1500 * np.exp(-0.5 * ((x - 0.55) / 0.09) ** 2)).astype(int))]\n"
                     "s = 30 * np.exp(-0.5 * ((x - 0.3) / 0.06) ** 2)\n"
                     "n = np.round(base / 10 + 0.8 * s + 3 * np.sin(11 * x)).astype(int)\n"
                     "grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3])\n",
            "call": "orchestrate_template_comparison(edges, a, pairs, s, n, 10.0, 8, 1.0, 10.0, grid, 0.95)",
            "gold_call": "_oracle_orchestrate_template_comparison(edges, a, pairs, s, n, 10.0, 8, 1.0, 10.0, grid, 0.95)",
            "tol": 1e-7,
        },
        {
            "setup": sample + "# invalid: edges that are not increasing\nargs = (edges[::-1], a, [cal], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: an empty Monte Carlo bin\na = a.copy(); a[11] = 0\nargs = (edges, a, [cal], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: an empty length-scale grid\nargs = (edges, a, [cal], s, n, 10.0, 4, 1.0, 10.0, np.array([]), 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a signal template without any positive entry\nargs = (edges, a, [cal], np.zeros(12), n, 10.0, 4, 1.0, 10.0, grid, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a variation array of the wrong length\nargs = (edges, a, [(cal[0][:11], cal[1])], s, n, 10.0, 4, 1.0, 10.0, grid, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
