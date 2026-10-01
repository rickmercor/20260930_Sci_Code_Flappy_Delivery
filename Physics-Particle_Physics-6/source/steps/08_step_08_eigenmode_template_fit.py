"""
Fit the signal strength of a single-channel template likelihood in which the background template is deformed along the leading eigenmodes of its covariance, with one unit-Gaussian constrained amplitude per retained mode.

The expected count in bin $j$ is $\nu_j = \mu s_j + b_j \exp(\sum_{i=1}^{k} \sqrt{\lambda_i} z_i [v_i]_j)$, with $s_j$ the signal template, $b_j$ the smooth background template, $(\lambda_i, v_i)$ the retained eigenpairs of the log-rate covariance and $z_i$ the mode amplitudes. The exponential form keeps every expected count positive and reduces, when the covariance is diagonal, to the per-bin multiplicative factors of the histogram treatment, and, when a single systematic dominates, to the piecewise-exponential interpolation of histogram templates. The likelihood is the product of Poisson terms over bins times unit Gaussian constraints on the amplitudes; its maximum gives the fitted signal strength, and the curvature of the negative log-likelihood at the maximum gives the parabolic uncertainty that the coverage studies of the method rely on.

Returns
-------
tuple, Two native Python floats (mu_hat, sigma_mu). The k leading eigenpairs (lambda_i, v_i) of Sigma are retained, k being the smallest number of modes whose eigenvalues sum to at least fraction times the trace, and the expected count of bin j is nu_j = mu * s_j + b_j * exp(sum_i sqrt(lambda_i) z_i v_ij). The negative log-likelihood sum_j (nu_j - n_j log nu_j) + sum_i z_i^2 / 2 is minimised jointly over the signal strength mu, which may take any real value that keeps every nu_j above zero, and the k amplitudes z_i. mu_hat is the minimising signal strength, converged to within 1e-9, and sigma_mu is the square root of the (mu, mu) entry of the inverse Hessian of the negative log-likelihood at the minimum, with the Hessian taken over (mu, z_1, ..., z_k). Eigenvector signs do not affect the result.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def eigenmode_template_fit(observed: "np.ndarray", signal: "np.ndarray", template: "np.ndarray", Sigma: "np.ndarray", fraction: float) -> tuple:
    r"""Return the fitted signal strength and its parabolic uncertainty.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts of shape (N,), N at least one, finite,
        non-negative integers.
    signal : np.ndarray
        Expected signal counts at unit signal strength, shape (N,), finite,
        non-negative, with at least one entry above zero.
    template : np.ndarray
        Smooth background template, shape (N,), finite and above zero.
    Sigma : np.ndarray
        Covariance of the background log-rate, shape (N, N), with the
        requirements of the eigenmode truncation (finite, symmetric to
        within 1e-9, trace above zero, no eigenvalue below -1e-9 times the
        trace).
    fraction : float
        Fraction of the total variance retained in the eigenmodes, above
        zero and at most one.

    Returns
    -------
    result : tuple
        Two native Python floats (mu_hat, sigma_mu). The k leading
        eigenpairs (lambda_i, v_i) of Sigma are retained, k being the
        smallest number of modes whose eigenvalues sum to at least
        fraction times the trace, and the expected count of bin j is
        nu_j = mu * s_j + b_j * exp(sum_i sqrt(lambda_i) z_i v_ij). The
        negative log-likelihood sum_j (nu_j - n_j log nu_j) + sum_i z_i^2
        / 2 is minimised jointly over the signal strength mu, which may
        take any real value that keeps every nu_j above zero, and the k
        amplitudes z_i. mu_hat is the minimising signal strength,
        converged to within 1e-9, and sigma_mu is the square root of the
        (mu, mu) entry of the inverse Hessian of the negative
        log-likelihood at the minimum, with the Hessian taken over
        (mu, z_1, ..., z_k). Eigenvector signs do not affect the result.

    Raises
    ------
    ValueError
        If observed is not a finite one-dimensional array of non-negative
        integers, if signal or template is not a finite one-dimensional
        array of the same length satisfying the stated positivity
        requirements, if Sigma is not a valid covariance of matching size,
        or if fraction is not finite, not above zero or above one.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _check_channel(observed, signal, template) -> tuple:
    n = np.asarray(observed, dtype=float)
    s = np.asarray(signal, dtype=float)
    b = np.asarray(template, dtype=float)
    if n.ndim != 1 or n.shape[0] < 1 or not np.all(np.isfinite(n)) or np.any(n < 0.0) or np.any(n != np.round(n)):
        raise ValueError("observed must be a finite one-dimensional array of non-negative integers")
    if s.shape != n.shape or not np.all(np.isfinite(s)) or np.any(s < 0.0) or not np.any(s > 0.0):
        raise ValueError("signal must be a finite non-negative array of the same length with an entry above zero")
    if b.shape != n.shape or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("template must be a finite one-dimensional array of the same length with positive entries")
    return n, s, b


def _descent_direction(hessian: "np.ndarray", gradient: "np.ndarray") -> "np.ndarray":
    """Newton step, with the Hessian shifted towards the identity until it is positive definite."""
    shift = 0.0
    for _ in range(60):
        try:
            factor = np.linalg.cholesky(hessian + shift * np.eye(hessian.shape[0]))
            return -np.linalg.solve(factor.T, np.linalg.solve(factor, gradient))
        except np.linalg.LinAlgError:
            shift = max(1e-8, 4.0 * shift)
    return -gradient


def _newton_minimise(value_gradient_hessian, start: "np.ndarray") -> tuple:
    """Damped Newton descent to the minimiser; returns the point and the Hessian there."""
    point = np.asarray(start, dtype=float)
    value, gradient, hessian = value_gradient_hessian(point)
    for _ in range(500):
        step = _descent_direction(hessian, gradient)
        scale = 1.0
        while scale > 1e-12:
            candidate = point + scale * step
            new_value, new_gradient, new_hessian = value_gradient_hessian(candidate)
            if math.isfinite(new_value) and new_value <= value + 1e-4 * scale * float(gradient @ step):
                break
            scale *= 0.5
        if not math.isfinite(new_value) or new_value > value:
            break                                   # no descent possible along the Newton direction
        stalled = value - new_value <= 1e-15 * max(1.0, abs(value))
        point, value, gradient, hessian = candidate, new_value, new_gradient, new_hessian
        if np.max(np.abs(scale * step)) < 1e-13 or stalled:
            break
    return point, hessian


def _oracle_eigenmode_template_fit(observed: "np.ndarray", signal: "np.ndarray", template: "np.ndarray", Sigma: "np.ndarray", fraction: float) -> tuple:
    n, s, b = _check_channel(observed, signal, template)
    values, vectors, _ = _sorted_eigenpairs(Sigma)
    if values.shape[0] != n.shape[0]:
        raise ValueError("Sigma must match the number of bins")
    kept = _oracle_truncated_eigenvalues(Sigma, fraction)
    modes = vectors[:, :kept.shape[0]] * np.sqrt(kept)[None, :]   # columns sqrt(lambda_i) v_i

    def _objective(point):
        mu, z = point[0], point[1:]
        deformed = b * np.exp(modes @ z)
        nu = mu * s + deformed
        if np.any(nu <= 0.0):
            return math.inf, None, None
        value = float(np.sum(nu - n * np.log(nu)) + 0.5 * np.sum(z ** 2))
        residual = 1.0 - n / nu
        curvature = n / nu ** 2
        gradient = np.concatenate([[float(residual @ s)], modes.T @ (residual * deformed) + z])
        d_nu = np.column_stack([s, modes * deformed[:, None]])       # d nu_j / d parameter
        hessian = d_nu.T @ (curvature[:, None] * d_nu)
        hessian[1:, 1:] += modes.T @ ((residual * deformed)[:, None] * modes) + np.eye(z.shape[0])
        return value, gradient, hessian

    start = np.concatenate([[1.0], np.zeros(kept.shape[0])])
    point, hessian = _newton_minimise(_objective, start)
    covariance = np.linalg.inv(hessian)
    return (float(point[0]), float(math.sqrt(covariance[0, 0])))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "eigenmode_template_fit"),
                           ("run_gold", "_oracle_eigenmode_template_fit")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    channel = ("import numpy as np\n"
               "n = np.array([295, 238, 137, 169, 110, 132, 151, 115, 50, 10, 13, 11])\n"
               "s = np.array([0.0, 0.29, 10.09, 40.82, 28.79, 12.63, 3.67, 0.41, 0.02, 0.0, 0.0, 0.0])\n"
               "b = np.array([292.6, 217.7, 154.6, 110.4, 103.5, 127.3, 156.2, 113.9, 56.3, 24.2, 11.1, 7.2])\n"
               "x = (np.arange(12) + 0.5) / 12.0\n"
               "r = np.abs(x[:, None] - x[None, :]) / 0.3\n"
               "stat = 0.002 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)\n"
               "d1 = 0.2 * np.array([0.14, 0.12, 0.11, 0.07, -0.04, -0.09, 0.02, 0.19, 0.32, 0.35, 0.21, 0.09])\n"
               "d2 = 0.06 * np.ones(12)\n"
               "d3 = 0.1 * np.array([0.1, 0.0, -0.2, -0.3, 0.5, 0.8, 0.3, -0.4, -0.6, -0.2, 0.1, 0.2])\n"
               "S = stat + np.outer(d1, d1) + np.outer(d2, d2) + np.outer(d3, d3)\n")
    return [
        # a benchmark-like channel at the benchmark fraction
        {
            "setup": channel,
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # every mode retained
        {
            "setup": channel,
            "call": "eigenmode_template_fit(n, s, b, S, 1.0)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 1.0)",
            "tol": 1e-7,
        },
        # the leading mode only
        {
            "setup": channel,
            "call": "eigenmode_template_fit(n, s, b, S, 0.5)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.5)",
            "tol": 1e-7,
        },
        # data without signal, where the fitted strength is negative
        {
            "setup": channel + "n = np.array([292, 221, 150, 108, 101, 125, 160, 118, 55, 22, 12, 6])\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # a large injected signal
        {
            "setup": channel + "n = np.array([300, 218, 190, 260, 210, 175, 162, 116, 54, 26, 11, 7])\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # a diagonal covariance, the histogram limit with independent per-bin factors
        {
            "setup": channel + "S = np.diag(1.0 / (10.0 * b))\n",
            "call": "eigenmode_template_fit(n, s, b, S, 1.0)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 1.0)",
            "tol": 1e-7,
        },
        # a single systematic direction, the piecewise-exponential interpolation limit
        {
            "setup": channel + "S = np.outer(d1, d1)\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # edge: a tiny covariance, where the fit reduces to a one-parameter Poisson fit
        {
            "setup": channel + "S = 1e-12 * np.eye(12)\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # edge: empty observed bins
        {
            "setup": channel + "n = np.array([295, 238, 137, 169, 110, 132, 151, 115, 50, 0, 0, 0])\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # boundary: a two-bin channel with one mode
        {
            "setup": "import numpy as np\nn = np.array([120, 80])\ns = np.array([10.0, 2.0])\nb = np.array([100.0, 90.0])\nS = np.array([[0.01, 0.004], [0.004, 0.02]])\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.7)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.7)",
            "tol": 1e-7,
        },
        # edge: a large systematic variance, where the amplitudes move far from zero
        {
            "setup": channel + "S = stat + 25.0 * np.outer(d1, d1)\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        # stress: forty bins with a dense covariance and a broad signal
        {
            "setup": "import numpy as np\n"
                     "x = (np.arange(40) + 0.5) / 40.0\n"
                     "b = 400 * np.exp(-4 * x) + 150 * np.exp(-0.5 * ((x - 0.55) / 0.1) ** 2)\n"
                     "s = 30 * np.exp(-0.5 * ((x - 0.3) / 0.06) ** 2)\n"
                     "n = np.round(b + 0.8 * s + 3 * np.sin(11 * x)).astype(int)\n"
                     "r = np.abs(x[:, None] - x[None, :]) / 0.15\n"
                     "S = 0.001 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r) + np.outer(0.03 * x, 0.03 * x) + 0.0025 * np.ones((40, 40))\n",
            "call": "eigenmode_template_fit(n, s, b, S, 0.95)",
            "gold_call": "_oracle_eigenmode_template_fit(n, s, b, S, 0.95)",
            "tol": 1e-7,
        },
        {
            "setup": channel + "# invalid: a template entry of zero\nb = b.copy(); b[11] = 0.0\nargs = (n, s, b, S, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a signal template without any positive entry\nargs = (n, np.zeros(12), b, S, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a non-integer observed count\nn = n.astype(float); n[0] = 295.5\nargs = (n, s, b, S, 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a covariance of the wrong size\nargs = (n, s, b, S[:11, :11], 0.95)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": channel + "# invalid: a fraction above one\nargs = (n, s, b, S, 1.5)\n" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
