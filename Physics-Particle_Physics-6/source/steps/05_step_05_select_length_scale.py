"""
Select the length scale of the kernel from a grid of candidates by maximising the Laplace approximation to the log marginal likelihood of the Monte Carlo counts.

The hyperparameters of the Gaussian process are chosen by evidence maximisation: for every candidate the effective prior covariance is built, the posterior mode of the log-rate is found, and the Laplace approximation to the log marginal likelihood is evaluated; the candidate with the largest value is kept. The marginal likelihood balances the fit to the counts against the volume of functions the prior admits, so that the selected length scale reflects the correlation length of the template rather than the bin spacing. The amplitude is held fixed here, because the evidence is nearly degenerate along a ridge of amplitudes and length scales that produce similar posteriors, and a one-dimensional grid gives a determinate choice.

Returns
-------
float, the candidate, as a native Python float, whose effective prior covariance (Matérn 5/2 kernel with the given amplitude and that length scale plus the integrated mean function) gives the largest Laplace approximation to the log marginal likelihood of the counts. On an exact tie the candidate listed first in ell_grid is returned.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_length_scale(counts: "np.ndarray", widths: "np.ndarray", x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray") -> float:
    r"""Return the length scale of the grid with the largest Laplace log marginal likelihood.

    Parameters
    ----------
    counts : np.ndarray
        Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    x : np.ndarray
        Bin centres of shape (N,), finite, within [lo, hi].
    lo : float
        Lower end of the template range, finite.
    hi : float
        Upper end of the template range, finite and above lo.
    n_basis : int
        Number of cubic B-spline basis functions of the mean function, at
        least four.
    sigma : float
        Amplitude of the Matérn 5/2 kernel, finite and above zero.
    prior_scale : float
        Standard deviation of the prior on the mean-function coefficients,
        finite and above zero.
    ell_grid : np.ndarray
        Candidate length scales of shape (M,), M at least one, finite and
        above zero.

    Returns
    -------
    ell : float
        The candidate, as a native Python float, whose effective prior
        covariance (Matérn 5/2 kernel with the given amplitude and that
        length scale plus the integrated mean function) gives the largest
        Laplace approximation to the log marginal likelihood of the
        counts. On an exact tie the candidate listed first in ell_grid is
        returned.

    Raises
    ------
    ValueError
        If counts, widths or x violate their stated requirements, if lo
        or hi is not finite or hi is not above lo, if n_basis is below
        four, if sigma or prior_scale is not finite or not above zero, or
        if ell_grid is empty or contains a value that is not finite or
        not above zero.
    """
    return ell

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_length_scale(counts: "np.ndarray", widths: "np.ndarray", x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray") -> float:
    a, w = _check_counts_widths(counts, widths)
    grid = np.asarray(ell_grid, dtype=float)
    if grid.ndim != 1 or grid.shape[0] < 1 or not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("ell_grid must be a non-empty one-dimensional array of finite positive values")
    best_value, best_ell = None, None
    for ell in grid:
        prior = _oracle_effective_kernel_matrix(x, lo, hi, n_basis, sigma, float(ell), prior_scale)
        if prior.shape[0] != a.shape[0]:
            raise ValueError("x must have one centre per bin")
        value = _oracle_laplace_log_marginal_likelihood(a, w, prior)
        # a strict comparison keeps the first candidate on an exact tie
        if best_value is None or value > best_value:
            best_value, best_ell = value, float(ell)
    return best_ell

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "select_length_scale"),
                           ("run_gold", "_oracle_select_length_scale")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    sample = ("import numpy as np\n"
              "x = (np.arange(12) + 0.5) / 12.0\n"
              "w = np.full(12, 1.0 / 12.0)\n"
              "a = np.array([2947, 2193, 1526, 1094, 1026, 1288, 1572, 1133, 561, 244, 110, 74])\n"
              "grid = np.array([0.1, 0.2, 0.3, 0.5, 0.8])\n")
    return [
        # the benchmark selection
        {
            "setup": sample,
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # a finer grid around the benchmark optimum
        {
            "setup": sample + "grid = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45])\n",
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # the grid in reverse order, which must not change the choice
        {
            "setup": sample + "grid = grid[::-1]\n",
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # a larger amplitude shifts the preferred length scale along the evidence ridge
        {
            "setup": sample,
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 3.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 3.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # a richer mean function absorbs the shape, and the evidence prefers a different scale
        {
            "setup": sample,
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 6, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 6, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # a sparse template, where the evidence favours a long length scale
        {
            "setup": sample + "a = np.array([9, 4, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0])\n",
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 3.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 3.0, grid)",
            "tol": 1e-12,
        },
        # a wiggly template, where the evidence favours a short length scale
        {
            "setup": sample + "a = np.array([500, 900, 480, 950, 510, 880, 470, 930, 520, 900, 490, 940])\n",
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # boundary: a grid with a single candidate
        {
            "setup": sample + "grid = np.array([0.42])\n",
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # edge: a repeated candidate, where the first occurrence is returned on the exact tie
        {
            "setup": sample + "grid = np.array([0.3, 0.3, 0.1])\n",
            "call": "select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        # edge: a template range not starting at zero, with the bin centres and grid in those units
        {
            "setup": "import numpy as np\n"
                     "x = 105.0 + (np.arange(11) + 0.5) * 5.0\n"
                     "w = np.full(11, 5.0)\n"
                     "a = np.array([412, 361, 318, 279, 260, 231, 201, 173, 165, 138, 121])\n"
                     "grid = np.array([5.0, 10.0, 20.0, 40.0, 80.0])\n",
            "call": "select_length_scale(a, w, x, 105.0, 160.0, 4, 1.0, 10.0, grid)",
            "gold_call": "_oracle_select_length_scale(a, w, x, 105.0, 160.0, 4, 1.0, 10.0, grid)",
            "tol": 1e-12,
        },
        {
            "setup": sample + "# invalid: an empty grid\nargs = (a, w, x, 0.0, 1.0, 4, 1.0, 10.0, np.array([]))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a candidate length scale of zero\nargs = (a, w, x, 0.0, 1.0, 4, 1.0, 10.0, np.array([0.0, 0.3]))\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: bin centres of the wrong length\nargs = (a, w, x[:11], 0.0, 1.0, 4, 1.0, 10.0, grid)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": sample + "# invalid: a negative count\na[2] = -3\nargs = (a, w, x, 0.0, 1.0, 4, 1.0, 10.0, grid)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
