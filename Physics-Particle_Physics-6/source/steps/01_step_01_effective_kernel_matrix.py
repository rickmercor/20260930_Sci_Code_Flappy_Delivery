"""
Build the effective prior covariance matrix of the log-rate at the bin centres of a template: a Matérn 5/2 kernel plus the contribution of a cubic B-spline mean function whose coefficients are integrated out under a broad Gaussian prior.

A smooth template is modelled as a log-Gaussian Cox process: the logarithm of the event rate is a Gaussian process with a covariance kernel that encodes the assumption of a twice differentiable log-rate, the Matérn kernel of order $5/2$ with amplitude $\sigma$ and length scale $\ell$. Large-scale trends such as exponential decays or broad peaks depart from the stationarity of the kernel; they are absorbed by a parametric mean function $h(x)^{\mathsf T}\beta$ built from cubic B-splines. Integrating the coefficients $\beta$ out under the prior $\beta \sim \mathcal N(0, B)$ with $B = b^2 I$ turns the mean function into an additional kernel term, so that the log-rate has the effective prior covariance $K_{\mathrm{eff}} = K + b^2 H H^{\mathsf T}$, with $H$ the design matrix of the basis at the bin centres.

The B-spline basis uses an open uniform knot vector on the template range: the two boundary knots are repeated four times and the interior knots are equally spaced, so that the basis functions form a partition of unity on the range and the number of functions exceeds the number of interior knots by four.

Returns
-------
np.ndarray, Array of shape (N, N) with entries k(x_i, x_j) + b ** 2 * sum_m H_im H_jm, where k(x, x') = sigma ** 2 * (1 + sqrt(5) r + 5 r ** 2 / 3) * exp(-sqrt(5) r) with r = |x - x'| / ell is the Matérn 5/2 kernel and H_im is the value at x_i of the m-th cubic B-spline of the open uniform knot vector on [lo, hi] (boundary knots of multiplicity four, n_basis - 4 interior knots equally spaced between them, so that the basis functions sum to one at every position of the closed range, including at hi). The result is symmetric and positive definite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_kernel_matrix(x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, ell: float, prior_scale: float) -> "np.ndarray":
    r"""Return the effective prior covariance matrix of the log-rate at the given positions.

    Parameters
    ----------
    x : np.ndarray
        Positions of shape (N,), N at least one, finite, each within the
        closed range [lo, hi].
    lo : float
        Lower end of the template range, finite.
    hi : float
        Upper end of the template range, finite and above lo.
    n_basis : int
        Number of cubic B-spline basis functions of the mean function, at
        least four.
    sigma : float
        Amplitude of the Matérn 5/2 kernel, finite and above zero.
    ell : float
        Length scale of the Matérn 5/2 kernel, finite and above zero.
    prior_scale : float
        Standard deviation b of the independent Gaussian prior on every
        coefficient of the mean function, finite and above zero.

    Returns
    -------
    K_eff : np.ndarray
        Array of shape (N, N) with entries k(x_i, x_j) + b ** 2 * sum_m
        H_im H_jm, where k(x, x') = sigma ** 2 * (1 + sqrt(5) r + 5 r ** 2 / 3)
        * exp(-sqrt(5) r) with r = |x - x'| / ell is the Matérn 5/2 kernel
        and H_im is the value at x_i of the m-th cubic B-spline of the open
        uniform knot vector on [lo, hi] (boundary knots of multiplicity four,
        n_basis - 4 interior knots equally spaced between them, so that the
        basis functions sum to one at every position of the closed range,
        including at hi). The result is symmetric and positive definite.

    Raises
    ------
    ValueError
        If x is not a finite one-dimensional array with at least one entry
        inside [lo, hi], if lo or hi is not finite or hi is not above lo,
        if n_basis is below four, or if sigma, ell or prior_scale is not
        finite or not above zero.
    """
    return K_eff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.interpolate import BSpline


def _check_positive(value: float, label: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{label} must be finite and above zero")
    return value


def _positions_in_range(x, lo: float, hi: float) -> tuple:
    positions = np.asarray(x, dtype=float)
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError("lo and hi must be finite with hi above lo")
    if positions.ndim != 1 or positions.shape[0] < 1 or not np.all(np.isfinite(positions)):
        raise ValueError("x must be a finite one-dimensional array with at least one entry")
    if np.any(positions < lo) or np.any(positions > hi):
        raise ValueError("every position must lie inside [lo, hi]")
    return positions, lo, hi


def _bspline_design_matrix(positions: "np.ndarray", lo: float, hi: float, n_basis: int) -> "np.ndarray":
    """Cubic B-spline values on the open uniform knot vector, one row per position."""
    degree = 3
    interior = np.linspace(lo, hi, n_basis - degree + 1)[1:-1]
    knots = np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])
    return BSpline.design_matrix(positions, knots, degree).toarray()


def _matern52(positions: "np.ndarray", sigma: float, ell: float) -> "np.ndarray":
    r = np.abs(positions[:, None] - positions[None, :]) / ell
    return sigma ** 2 * (1.0 + math.sqrt(5.0) * r + 5.0 * r ** 2 / 3.0) * np.exp(-math.sqrt(5.0) * r)


def _oracle_effective_kernel_matrix(x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, ell: float, prior_scale: float) -> "np.ndarray":
    positions, lo, hi = _positions_in_range(x, lo, hi)
    if int(n_basis) < 4:
        raise ValueError("n_basis must be at least four")
    sigma = _check_positive(sigma, "sigma")
    ell = _check_positive(ell, "ell")
    scale = _check_positive(prior_scale, "prior_scale")
    design = _bspline_design_matrix(positions, lo, hi, int(n_basis))
    return _matern52(positions, sigma, ell) + scale ** 2 * design @ design.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "effective_kernel_matrix"),
                           ("run_gold", "_oracle_effective_kernel_matrix")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    centres = "import numpy as np\nx = (np.arange(12) + 0.5) / 12.0\n"
    return [
        # the benchmark configuration: twelve bin centres, four basis functions (no interior knot), selected length scale
        {
            "setup": centres,
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 4, 1.0, 0.3, 10.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 4, 1.0, 0.3, 10.0)",
            "tol": 1e-9,
        },
        # six basis functions, two interior knots at one third and two thirds
        {
            "setup": centres,
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 6, 0.5, 0.2, 3.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 6, 0.5, 0.2, 3.0)",
            "tol": 1e-9,
        },
        # a template range that does not start at zero, with positions at both ends of the range
        {
            "setup": "import numpy as np\nx = np.array([105.0, 112.5, 121.0, 130.0, 138.5, 147.0, 155.5, 160.0])\n",
            "call": "effective_kernel_matrix(x, 105.0, 160.0, 5, 0.8, 12.0, 4.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 105.0, 160.0, 5, 0.8, 12.0, 4.0)",
            "tol": 1e-9,
        },
        # boundary: a single position, where the result is the 1 by 1 matrix sigma ** 2 + b ** 2 (partition of unity)
        {
            "setup": "import numpy as np\nx = np.array([0.37])\n",
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 7, 1.5, 0.25, 2.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 7, 1.5, 0.25, 2.0)",
            "tol": 1e-9,
        },
        # edge: a very short length scale makes the kernel part nearly diagonal while the basis part stays dense
        {
            "setup": centres,
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 4, 2.0, 0.01, 1.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 4, 2.0, 0.01, 1.0)",
            "tol": 1e-9,
        },
        # edge: a very long length scale makes the kernel part nearly constant
        {
            "setup": centres,
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 5, 1.0, 50.0, 0.5)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 5, 1.0, 50.0, 0.5)",
            "tol": 1e-9,
        },
        # edge: unevenly spaced positions, including one exactly on an interior knot
        {
            "setup": "import numpy as np\nx = np.array([0.0, 0.05, 0.2, 0.25, 0.5, 0.51, 0.75, 0.99, 1.0])\n",
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 8, 1.0, 0.3, 10.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 8, 1.0, 0.3, 10.0)",
            "tol": 1e-9,
        },
        # edge: integer-valued positions are valid input and must not be modified in place
        {
            "setup": "import numpy as np\nx = np.array([1, 2, 3, 5, 8])\n",
            "call": "effective_kernel_matrix(x, 0, 10, 4, 1.0, 2.0, 1.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0, 10, 4, 1.0, 2.0, 1.0)",
            "tol": 1e-9,
        },
        # stress: forty positions and twelve basis functions, the paper's Experiment B grid
        {
            "setup": "import numpy as np\nx = (np.arange(40) + 0.5) / 40.0\n",
            "call": "effective_kernel_matrix(x, 0.0, 1.0, 12, 1.0, 0.15, 10.0)",
            "gold_call": "_oracle_effective_kernel_matrix(x, 0.0, 1.0, 12, 1.0, 0.15, 10.0)",
            "tol": 1e-9,
        },
        {
            "setup": centres + "# invalid: fewer than four basis functions\nargs = (x, 0.0, 1.0, 3, 1.0, 0.3, 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": centres + "# invalid: a position outside the range\nargs = (np.append(x, 1.2), 0.0, 1.0, 4, 1.0, 0.3, 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": centres + "# invalid: a length scale of zero\nargs = (x, 0.0, 1.0, 4, 1.0, 0.0, 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": centres + "# invalid: a negative prior scale\nargs = (x, 0.0, 1.0, 4, 1.0, 0.3, -1.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": centres + "# invalid: the upper end of the range below the lower end\nargs = (x, 1.0, 0.0, 4, 1.0, 0.3, 10.0)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
