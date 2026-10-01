"""
The adaptive Gaussian random walk is based on $(2.38^2/d)$ times the empirical particle covariance and is factorised through a dimensionless correlation matrix with relative diagonal jitter $10^{-12}$, including for zero-variance coordinates.

Evaluate the hierarchical target and construct a scale-aware proposal factor. The hierarchical target contains uniform and half-Cauchy hyperpriors together with ordinary Normal density factors for positive replicate means.

Returns
-------
np.ndarray, the packed prior draw, log density, and proposal factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_hierarchical_prior_and_proposal(
    theta: np.ndarray,
    particles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Return one prior draw, one log-prior value, and the proposal factor.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector whose hierarchical log prior is evaluated.
    particles : np.ndarray
        Current particle matrix with shape (dimension, n_particles).
    rng : np.random.Generator
        Shared PCG64 random-number stream.

    Notes
    -----
    Draw the two within-replicate spreads and two population means directly, then
    draw each half-Cauchy scale as its scale times the absolute value of one
    standard Cauchy variate. Draw each positive replicate mean by inverse CDF from
    one uniform on $[\Phi(-m/s),1]$. Form the lower proposal factor from
    $(2.38^2/d)$ times the sample covariance with divisor $n-1$, factorised
    through the correlation matrix with relative diagonal jitter $10^{-12}$; a
    coordinate with zero sample variance receives the diagonal factor entry
    $\sqrt{\text{tiny}}$, the square root of the smallest positive float, and
    zero off-diagonal entries.

    Returns
    -------
    result : np.ndarray
        Numeric vector containing the prior draw, log prior, and flattened
        scale-aware Gaussian proposal factor.

    Raises
    ------
    ValueError
        If theta does not have dimension 2M+6 matching the particle matrix, or
        the particle matrix has fewer than two particles.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _safe_uniform(rng, low, high, size=None):
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise ValueError("invalid uniform probability interval")
    if high == low:
        if size is None:
            return float(low)
        return np.full(size, low, dtype=float)
    lo = np.nextafter(float(low), float(high))
    hi = np.nextafter(float(high), float(low))
    return rng.uniform(lo, hi, size=size)


def _sample_positive_normal(mean, sd, size, rng):
    mean = float(mean)
    sd = float(sd)
    size = int(size)
    if sd < 0.0 or size < 1:
        raise ValueError("sd must be non-negative and size must be positive")
    if sd == 0.0:
        if mean < 0.0:
            raise ValueError("degenerate positive Normal requires non-negative mean")
        return np.full(size, mean, dtype=float)
    lower_cdf = float(ndtr((0.0 - mean) / sd))
    u = _safe_uniform(rng, lower_cdf, 1.0, size=size)
    return mean + sd * ndtri(u)


def _sample_half_cauchy(scale, rng):
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("half-Cauchy scale must be positive")
    return float(scale * abs(rng.standard_cauchy()))


def _sample_hierarchical_prior(n_replicates, rng):
    m = int(n_replicates)
    if m < 1:
        raise ValueError("n_replicates must be positive")
    sigma_r = rng.uniform(0.0, 2.0e-6)
    sigma_k = rng.uniform(0.0, 5.0)
    mean_r = rng.uniform(0.0, 1.0e-6)
    mean_k = rng.uniform(0.0, 50.0)
    sd_r = _sample_half_cauchy(3.0e-7, rng)
    sd_k = _sample_half_cauchy(5.0, rng)

    theta = np.empty(2 * m + 6, dtype=float)
    theta[:m] = _sample_positive_normal(mean_r, sd_r, m, rng)
    theta[m : 2 * m] = _sample_positive_normal(mean_k, sd_k, m, rng)
    theta[2 * m :] = [sigma_r, sigma_k, mean_r, mean_k, sd_r, sd_k]
    return theta


def _uniform_logpdf(x, lo, hi):
    if not (lo <= x <= hi):
        return -np.inf
    return -math.log(hi - lo)


def _half_cauchy_logpdf(x, scale):
    if x < 0.0 or scale <= 0.0:
        return -np.inf
    return math.log(2.0) - math.log(math.pi * scale) - math.log1p((x / scale) ** 2)


def _normal_logpdf_sum(values, mean, sd):
    values = np.asarray(values, dtype=float)
    if sd < 0.0:
        return -np.inf
    if sd == 0.0:
        return 0.0 if np.allclose(values, mean, rtol=0.0, atol=0.0) else -np.inf
    z = (values - mean) / sd
    return float(-0.5 * np.sum(z**2) - values.size * math.log(sd * math.sqrt(2.0 * math.pi)))


def _log_hierarchical_prior(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 8 or (theta.size - 6) % 2 != 0:
        return -np.inf
    m = (theta.size - 6) // 2
    mu_r = theta[:m]
    mu_k = theta[m : 2 * m]
    sigma_r, sigma_k, mean_r, mean_k, sd_r, sd_k = theta[2 * m :]

    if np.any(~np.isfinite(theta)) or np.any(mu_r < 0.0) or np.any(mu_k < 0.0):
        return -np.inf

    lp = 0.0
    lp += _uniform_logpdf(float(sigma_r), 0.0, 2.0e-6)
    lp += _uniform_logpdf(float(sigma_k), 0.0, 5.0)
    lp += _uniform_logpdf(float(mean_r), 0.0, 1.0e-6)
    lp += _uniform_logpdf(float(mean_k), 0.0, 50.0)
    lp += _half_cauchy_logpdf(float(sd_r), 3.0e-7)
    lp += _half_cauchy_logpdf(float(sd_k), 5.0)
    if not np.isfinite(lp):
        return -np.inf

    lp += _normal_logpdf_sum(mu_r, float(mean_r), float(sd_r))
    lp += _normal_logpdf_sum(mu_k, float(mean_k), float(sd_k))
    return float(lp)


def _regularise_covariance(covariance, relative_jitter=1.0e-12):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    jitter = float(relative_jitter) * diagonal
    jitter = np.where(diagonal > 0.0, jitter, np.finfo(float).tiny)
    return covariance + np.diag(jitter)


def _scaled_cholesky(covariance, relative_jitter=1.0e-12, max_attempts=8):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    sd = np.sqrt(diagonal)
    positive = sd > 0.0
    safe_sd = np.where(positive, sd, 1.0)

    correlation = covariance / np.outer(safe_sd, safe_sd)
    correlation[~positive, :] = 0.0
    correlation[:, ~positive] = 0.0
    np.fill_diagonal(correlation, 1.0)
    correlation = 0.5 * (correlation + correlation.T)

    eye = np.eye(correlation.shape[0])
    for attempt in range(int(max_attempts)):
        jitter = float(relative_jitter) * (10.0**attempt)
        try:
            factor_corr = np.linalg.cholesky(correlation + jitter * eye)
            physical_sd = np.where(positive, sd, math.sqrt(np.finfo(float).tiny))
            return physical_sd[:, None] * factor_corr
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("proposal covariance is not factorisable")


def _proposal_factor(particles):
    particles = np.asarray(particles, dtype=float)
    if particles.ndim != 2 or particles.shape[0] < 1 or particles.shape[1] < 2:
        raise ValueError("particles must have shape (dimension, n_particles) with n_particles >= 2")
    d = particles.shape[0]
    covariance = (2.38**2 / d) * np.cov(particles, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance = _regularise_covariance(covariance, 1.0e-12)
    return _scaled_cholesky(covariance)


# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import ndtr, ndtri

def _oracle_evaluate_hierarchical_prior_and_proposal(
    theta: np.ndarray,
    particles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    particles = np.asarray(particles, dtype=float)
    if theta.ndim != 1 or particles.ndim != 2 or particles.shape[0] != theta.size:
        raise ValueError("theta dimension must match the particle matrix")
    m = (theta.size - 6) // 2
    if 2 * m + 6 != theta.size or m < 1:
        raise ValueError("theta must have dimension 2*M+6")

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    prior_draw = _sample_hierarchical_prior(m, rng)
    log_prior = _log_hierarchical_prior(theta)
    factor = _proposal_factor(particles)
    return np.concatenate((prior_draw, np.array([log_prior]), factor.ravel())).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic tests for the hierarchical target and proposal."""
    return [
        {
            "setup": """import numpy as np
fixture_rng = np.random.default_rng(17)
theta = np.array([4.0e-7, 5.0e-7, 6.0e-7, 8.0, 12.0, 16.0,
                  8.0e-7, 2.0, 5.0e-7, 12.0, 2.0e-7, 4.0])
scales = np.array([2e-7, 2e-7, 2e-7, 3.0, 3.0, 3.0, 2e-7, 0.5, 1e-7, 4.0, 1e-7, 1.0])[:, None]
particles = theta[:, None] + scales * fixture_rng.normal(size=(12, 30))
rng_call = np.random.default_rng(424242)
rng_gold = np.random.default_rng(424242)
""",
            "call": "evaluate_hierarchical_prior_and_proposal(theta, particles, rng_call)",
            "gold_call": "_oracle_evaluate_hierarchical_prior_and_proposal(theta, particles, rng_gold)",
        },
        {
            "setup": """import numpy as np
theta = np.array([3.0e-7, 3.0e-7, 10.0, 10.0, 5.0e-7, 1.0,
                  3.0e-7, 10.0, 0.0, 0.0])
base = theta[:, None]
particles = np.repeat(base, 8, axis=1)
rng_call = np.random.default_rng(3)
rng_gold = np.random.default_rng(3)
""",
            "call": "evaluate_hierarchical_prior_and_proposal(theta, particles, rng_call)",
            "gold_call": "_oracle_evaluate_hierarchical_prior_and_proposal(theta, particles, rng_gold)",
        },
        {
            "setup": """import numpy as np
fixture_rng = np.random.default_rng(2)
theta = np.array([4.0e-7, 5.0e-7, 8.0, 9.0, 3.0e-6, 2.0,
                  4.0e-7, 10.0, 1.0e-7, 2.0])
particles = np.vstack([fixture_rng.normal(4.0e-7, 1.0e-7, 20),
                       fixture_rng.normal(5.0e-7, 1.0e-7, 20),
                       fixture_rng.normal(8.0, 1.0, 20),
                       fixture_rng.normal(9.0, 1.0, 20),
                       fixture_rng.normal(8.0e-7, 1.0e-7, 20),
                       fixture_rng.normal(2.0, 0.2, 20),
                       fixture_rng.normal(4.0e-7, 1.0e-7, 20),
                       fixture_rng.normal(10.0, 1.0, 20),
                       fixture_rng.normal(1.0e-7, 2.0e-8, 20),
                       fixture_rng.normal(2.0, 0.2, 20)])
rng_call = np.random.default_rng(99)
rng_gold = np.random.default_rng(99)
""",
            "call": "evaluate_hierarchical_prior_and_proposal(theta, particles, rng_call)",
            "gold_call": "_oracle_evaluate_hierarchical_prior_and_proposal(theta, particles, rng_gold)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
theta = np.ones(10)
particles = np.ones((9, 6))
rng_call = np.random.default_rng(5)
rng_gold = np.random.default_rng(5)
""",
            "call": "value_error_code(lambda: evaluate_hierarchical_prior_and_proposal(theta, particles, rng_call))",
            "gold_call": "value_error_code(lambda: _oracle_evaluate_hierarchical_prior_and_proposal(theta, particles, rng_gold))",
        },
    ]
