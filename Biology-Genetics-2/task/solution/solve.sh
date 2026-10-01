#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def focal_branch_intervals(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    t_archaic: float = 15000.0,
) -> "np.ndarray":
    """Reference implementation."""
    coal_times = np.asarray(coal_times, dtype=float)
    focal_mask = np.asarray(focal_mask, dtype=float)
    if coal_times.ndim != 2 or coal_times.shape[1] < 1:
        raise ValueError("coal_times must be a two dimensional array of shape (m, c)")
    if focal_mask.shape != coal_times.shape:
        raise ValueError("focal_mask must have the same shape as coal_times")
    if not np.all(np.isin(focal_mask, (0.0, 1.0))):
        raise ValueError("focal_mask entries must be 0 or 1")
    if np.any(focal_mask.sum(axis=1) < 1.0):
        raise ValueError("every marginal tree must have at least one focal coalescence")
    if np.any(coal_times <= 0.0):
        raise ValueError("coalescence times must be strictly positive")
    if np.any(np.diff(coal_times, axis=1) <= 0.0):
        raise ValueError("coal_times must increase strictly along each row")
    if not np.isfinite(t_archaic) or t_archaic <= 0.0:
        raise ValueError("t_archaic must be a positive finite time")

    m = coal_times.shape[0]
    intervals = np.zeros((m, 2), dtype=float)
    for i in range(m):
        path = np.concatenate(([0.0], coal_times[i][focal_mask[i] == 1.0]))
        root = path[-1]
        lower, upper = root, root
        for j in range(path.size - 1):
            if path[j] <= t_archaic < path[j + 1]:
                lower, upper = path[j], path[j + 1]
                break
        intervals[i] = (lower, upper)
    return intervals

import numpy as np


def tree_observation_statistic(
    coal_times: "np.ndarray",
    intervals: "np.ndarray",
    x_floor: float = 1e-10,
) -> tuple:
    """Reference implementation."""
    coal_times = np.asarray(coal_times, dtype=float)
    intervals = np.asarray(intervals, dtype=float)
    if coal_times.ndim != 2:
        raise ValueError("coal_times must be a two dimensional array of shape (m, c)")
    if intervals.shape != (coal_times.shape[0], 2):
        raise ValueError("intervals must have shape (m, 2) matching coal_times")
    if np.any(intervals[:, 1] < intervals[:, 0]):
        raise ValueError("every upper endpoint must be at least its lower endpoint")
    if not np.isfinite(x_floor) or x_floor <= 0.0:
        raise ValueError("x_floor must be strictly positive")

    lower = intervals[:, 0][:, None]
    upper = intervals[:, 1][:, None]
    n_coal = np.sum((coal_times > lower) & (coal_times < upper), axis=1).astype(float)
    observations = n_coal * (intervals[:, 1] - intervals[:, 0]) + x_floor
    return n_coal, observations

import numpy as np
from scipy.stats import t as student_t


def esd_outlier_flags(
    observations: "np.ndarray",
    alpha: float = 0.05,
    max_outlier_fraction: float = 0.2,
) -> "np.ndarray":
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1:
        raise ValueError("observations must be one dimensional")
    if observations.size < 4:
        raise ValueError("observations must hold at least four points")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not (0.0 < max_outlier_fraction < 1.0):
        raise ValueError("max_outlier_fraction must lie strictly between 0 and 1")

    n = observations.size
    flags = np.zeros(n, dtype=float)
    remaining = observations.copy()
    index = np.arange(n)
    max_outliers = max(1, int(n * max_outlier_fraction))
    for i in range(1, max_outliers + 1):
        sd = remaining.std(ddof=1)
        if sd <= 0.0:
            break
        deviation = remaining - remaining.mean()
        j = int(np.argmax(deviation))
        statistic = deviation[j] / sd
        prob = 1.0 - alpha / (n - i + 1)
        t_c = student_t.ppf(prob, df=n - i - 1)
        critical = (n - i) * t_c / np.sqrt((n - i - 1 + t_c ** 2) * (n - i + 1))
        if statistic <= critical:
            break
        flags[index[j]] = 1.0
        remaining = np.delete(remaining, j)
        index = np.delete(index, j)
    return flags

import numpy as np
from scipy.special import digamma


def weighted_gamma_mle(
    observations: np.ndarray,
    weights: np.ndarray,
    bracket: tuple = (1e-6, 1e6),
    n_bisect: int = 200,
) -> tuple:
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if observations.ndim != 1 or weights.ndim != 1:
        raise ValueError("observations and weights must be one dimensional")
    if observations.size != weights.size:
        raise ValueError("observations and weights must have the same length")
    if np.any(observations <= 0.0):
        raise ValueError("observations must be strictly positive")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    if int(n_bisect) < 1:
        raise ValueError("n_bisect must be a positive integer")
    low, high = float(bracket[0]), float(bracket[1])
    if not (0.0 < low < high):
        raise ValueError("bracket must be an increasing pair of positive numbers")

    mean = float((weights * observations).sum() / total)
    mean_log = float((weights * np.log(observations)).sum() / total)
    s = np.log(mean) - mean_log
    if s <= 0.0:
        raise ValueError("the weighted sample has no dispersion in the log domain")

    def _shape_equation(a):
        return np.log(a) - digamma(a) - s

    for _ in range(int(n_bisect)):
        mid = 0.5 * (low + high)
        if _shape_equation(mid) > 0.0:
            low = mid
        else:
            high = mid
    shape = 0.5 * (low + high)
    rate = shape / mean
    return float(shape), float(rate)

import numpy as np
from scipy.special import gammaln


def gamma_emission_logpdf(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
) -> np.ndarray:
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1:
        raise ValueError("observations must be one dimensional")
    if np.any(observations <= 0.0):
        raise ValueError("observations must be strictly positive")
    parameters = (shape_null, rate_null, shape_archaic, rate_archaic)
    if any((not np.isfinite(v)) or v <= 0.0 for v in parameters):
        raise ValueError("gamma shapes and rates must be strictly positive")

    log_x = np.log(observations)
    log_emissions = np.empty((2, observations.size), dtype=float)
    for row, (a, b) in enumerate(((shape_null, rate_null), (shape_archaic, rate_archaic))):
        log_emissions[row] = a * np.log(b) - gammaln(a) + (a - 1.0) * log_x - b * observations
    return log_emissions

import numpy as np


def forward_backward_posteriors(
    log_emissions: np.ndarray,
    p: float,
    q: float,
    pi_archaic: float = 0.05,
) -> tuple:
    """Reference implementation."""
    log_emissions = np.asarray(log_emissions, dtype=float)
    if log_emissions.ndim != 2 or log_emissions.shape[0] != 2:
        raise ValueError("log_emissions must have shape (2, m)")
    if log_emissions.shape[1] < 2:
        raise ValueError("at least two marginal trees are required")
    for name, value in (("p", p), ("q", q), ("pi_archaic", pi_archaic)):
        if not np.isfinite(value) or not (0.0 < value < 1.0):
            raise ValueError("%s must lie strictly between 0 and 1" % name)

    m = log_emissions.shape[1]
    log_t = np.log(np.array([[1.0 - p, p], [q, 1.0 - q]]))
    log_pi = np.log(np.array([1.0 - pi_archaic, pi_archaic]))

    alphas = np.zeros((2, m), dtype=float)
    scalers = np.zeros(m, dtype=float)
    alphas[:, 0] = log_pi + log_emissions[:, 0]
    scalers[0] = np.logaddexp(alphas[0, 0], alphas[1, 0])
    alphas[:, 0] -= scalers[0]
    for i in range(1, m):
        for z in range(2):
            alphas[z, i] = log_emissions[z, i] + np.logaddexp(
                alphas[0, i - 1] + log_t[0, z], alphas[1, i - 1] + log_t[1, z]
            )
        scalers[i] = np.logaddexp(alphas[0, i], alphas[1, i])
        alphas[:, i] -= scalers[i]

    betas = np.zeros((2, m), dtype=float)
    for i in range(m - 2, -1, -1):
        for z in range(2):
            betas[z, i] = np.logaddexp(
                betas[0, i + 1] + log_emissions[0, i + 1] + log_t[z, 0],
                betas[1, i + 1] + log_emissions[1, i + 1] + log_t[z, 1],
            )
        betas[:, i] -= np.logaddexp(betas[0, i], betas[1, i])

    log_gamma = alphas + betas
    log_gamma -= np.logaddexp(log_gamma[0], log_gamma[1])
    return np.exp(log_gamma), float(scalers.sum())

import numpy as np
from scipy.special import digamma, gammaln


def _log_emissions(observations, a0, b0, a1, b1):
    """Two-row log emission matrix of the gamma densities in shape and rate form."""
    log_x = np.log(observations)
    rows = []
    for a, b in ((a0, b0), (a1, b1)):
        rows.append(a * np.log(b) - gammaln(a) + (a - 1.0) * log_x - b * observations)
    return np.vstack(rows)


def _forward_backward(log_emissions, p, q, pi_archaic):
    """Log-domain forward and backward recursions of the two-state chain."""
    m = log_emissions.shape[1]
    log_t = np.log(np.array([[1.0 - p, p], [q, 1.0 - q]]))
    log_pi = np.log(np.array([1.0 - pi_archaic, pi_archaic]))
    alphas = np.zeros((2, m))
    scalers = np.zeros(m)
    alphas[:, 0] = log_pi + log_emissions[:, 0]
    scalers[0] = np.logaddexp(alphas[0, 0], alphas[1, 0])
    alphas[:, 0] -= scalers[0]
    for i in range(1, m):
        for z in range(2):
            alphas[z, i] = log_emissions[z, i] + np.logaddexp(
                alphas[0, i - 1] + log_t[0, z], alphas[1, i - 1] + log_t[1, z]
            )
        scalers[i] = np.logaddexp(alphas[0, i], alphas[1, i])
        alphas[:, i] -= scalers[i]
    betas = np.zeros((2, m))
    for i in range(m - 2, -1, -1):
        for z in range(2):
            betas[z, i] = np.logaddexp(
                betas[0, i + 1] + log_emissions[0, i + 1] + log_t[z, 0],
                betas[1, i + 1] + log_emissions[1, i + 1] + log_t[z, 1],
            )
        betas[:, i] -= np.logaddexp(betas[0, i], betas[1, i])
    log_gamma = alphas + betas
    log_gamma -= np.logaddexp(log_gamma[0], log_gamma[1])
    return np.exp(log_gamma), alphas, betas, log_t, float(scalers.sum())


def _weighted_gamma_mle(observations, weights, bracket=(1e-6, 1e6), n_bisect=200):
    """Maximum-likelihood gamma shape and rate of a weighted positive sample."""
    total = weights.sum()
    mean = float((weights * observations).sum() / total)
    mean_log = float((weights * np.log(observations)).sum() / total)
    s = np.log(mean) - mean_log
    low, high = bracket
    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        if np.log(mid) - digamma(mid) - s > 0.0:
            low = mid
        else:
            high = mid
    shape = 0.5 * (low + high)
    return float(shape), float(shape / mean)


def baum_welch_parameters(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
    p: float = 0.01,
    q: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
) -> dict:
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size < 2:
        raise ValueError("observations must be one dimensional with at least two points")
    if np.any(observations <= 0.0):
        raise ValueError("observations must be strictly positive")
    for value in (shape_null, rate_null, shape_archaic, rate_archaic):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("gamma shapes and rates must be strictly positive")
    for name, value in (("p", p), ("q", q), ("pi_archaic", pi_archaic)):
        if not np.isfinite(value) or not (0.0 < value < 1.0):
            raise ValueError("%s must lie strictly between 0 and 1" % name)
    if int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(loglik_tol) or loglik_tol <= 0.0:
        raise ValueError("loglik_tol must be strictly positive")

    m = observations.size
    previous = None
    loglik = 0.0
    n_iter = 0
    for n_iter in range(int(max_iter)):
        log_emissions = _log_emissions(observations, shape_null, rate_null, shape_archaic, rate_archaic)
        gammas, alphas, betas, log_t, loglik = _forward_backward(
            log_emissions, p, q, pi_archaic
        )
        if previous is not None and abs(loglik - previous) < loglik_tol:
            break
        previous = loglik
        xi_01 = np.zeros(m - 1)
        xi_10 = np.zeros(m - 1)
        for i in range(m - 1):
            terms = np.array([
                alphas[0, i] + log_t[0, 0] + log_emissions[0, i + 1] + betas[0, i + 1],
                alphas[0, i] + log_t[0, 1] + log_emissions[1, i + 1] + betas[1, i + 1],
                alphas[1, i] + log_t[1, 0] + log_emissions[0, i + 1] + betas[0, i + 1],
                alphas[1, i] + log_t[1, 1] + log_emissions[1, i + 1] + betas[1, i + 1],
            ])
            peak = terms.max()
            norm = peak + np.log(np.exp(terms - peak).sum())
            xi_01[i] = np.exp(terms[1] - norm)
            xi_10[i] = np.exp(terms[2] - norm)
        p = float(xi_01.sum() / gammas[0, :-1].sum())
        q = float(xi_10.sum() / gammas[1, :-1].sum())
        shape_archaic, rate_archaic = _weighted_gamma_mle(observations, gammas[1])

    return {
        "shape_archaic": float(shape_archaic),
        "rate_archaic": float(rate_archaic),
        "p": float(p),
        "q": float(q),
        "n_iter": int(n_iter),
        "loglik": float(loglik),
    }

import numpy as np


def archaic_segments(
    posterior_archaic: np.ndarray,
    span_bp: np.ndarray,
    span_cm: np.ndarray,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> list:
    """Reference implementation."""
    posterior_archaic = np.asarray(posterior_archaic, dtype=float)
    span_bp = np.asarray(span_bp, dtype=float)
    span_cm = np.asarray(span_cm, dtype=float)
    if posterior_archaic.ndim != 1 or span_bp.ndim != 1 or span_cm.ndim != 1:
        raise ValueError("posterior_archaic, span_bp and span_cm must be one dimensional")
    if not (posterior_archaic.size == span_bp.size == span_cm.size):
        raise ValueError("posterior_archaic, span_bp and span_cm must have equal length")
    if np.any(span_bp < 0.0) or np.any(span_cm < 0.0):
        raise ValueError("spans must be non-negative")
    if not np.isfinite(post_threshold) or not (0.0 < post_threshold < 1.0):
        raise ValueError("post_threshold must lie strictly between 0 and 1")
    if min_bp < 0.0 or min_cm < 0.0:
        raise ValueError("min_bp and min_cm must be non-negative")

    called = posterior_archaic > post_threshold
    segments = []
    i = 0
    m = called.size
    while i < m:
        if not called[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and called[j + 1]:
            j += 1
        if span_bp[i:j + 1].sum() >= min_bp and span_cm[i:j + 1].sum() >= min_cm:
            segments.append((int(i), int(j)))
        i = j + 1
    return segments

import numpy as np


def admixture_time_estimate(
    segments: list,
    intervals: "np.ndarray",
    span_bp: "np.ndarray",
) -> float:
    """Reference implementation."""
    intervals = np.asarray(intervals, dtype=float)
    span_bp = np.asarray(span_bp, dtype=float)
    if span_bp.ndim != 1:
        raise ValueError("span_bp must be one dimensional")
    if intervals.shape != (span_bp.size, 2):
        raise ValueError("intervals must have shape (m, 2) matching span_bp")
    if len(segments) == 0:
        raise ValueError("at least one retained tract is required")

    covered = []
    for start, end in segments:
        start, end = int(start), int(end)
        if end < start:
            raise ValueError("a segment end must not precede its start")
        if start < 0 or end >= span_bp.size:
            raise ValueError("segment indices must lie within the range of trees")
        covered.extend(range(start, end + 1))
    covered = np.array(sorted(set(covered)), dtype=int)

    weights = span_bp[covered]
    if weights.sum() <= 0.0:
        raise ValueError("the covered trees carry no positive physical span")
    return float((weights * intervals[covered, 0]).sum() / weights.sum())

import numpy as np


def run_full_pipeline(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    span_bp: "np.ndarray",
    span_cm: "np.ndarray",
    t_archaic: float = 15000.0,
    x_floor: float = 1e-10,
    esd_alpha: float = 0.05,
    esd_max_outlier_fraction: float = 0.2,
    p_init: float = 0.01,
    q_init: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> float:
    """Reference implementation."""
    coal_times = np.asarray(coal_times, dtype=float)
    span_bp = np.asarray(span_bp, dtype=float)
    span_cm = np.asarray(span_cm, dtype=float)
    if coal_times.ndim != 2:
        raise ValueError("coal_times must be a two dimensional array of shape (m, c)")
    if span_bp.ndim != 1 or span_bp.size != coal_times.shape[0]:
        raise ValueError("span_bp must be one dimensional of length m")
    if span_cm.ndim != 1 or span_cm.size != coal_times.shape[0]:
        raise ValueError("span_cm must be one dimensional of length m")

    intervals = focal_branch_intervals(coal_times, focal_mask, t_archaic=t_archaic)
    _, observations = tree_observation_statistic(
        coal_times, intervals, x_floor=x_floor
    )
    flags = esd_outlier_flags(
        observations, alpha=esd_alpha, max_outlier_fraction=esd_max_outlier_fraction
    )
    if flags.sum() <= 0.0:
        raise ValueError("the outlier test flagged no tree, so the archaic state has no seed")

    shape_null, rate_null = weighted_gamma_mle(observations, 1.0 - flags)
    shape_archaic, rate_archaic = weighted_gamma_mle(observations, flags)

    fitted = baum_welch_parameters(
        observations,
        shape_null,
        rate_null,
        shape_archaic,
        rate_archaic,
        p=p_init,
        q=q_init,
        pi_archaic=pi_archaic,
        max_iter=max_iter,
        loglik_tol=loglik_tol,
    )
    log_emissions = gamma_emission_logpdf(
        observations,
        shape_null,
        rate_null,
        fitted["shape_archaic"],
        fitted["rate_archaic"],
    )
    posteriors, _ = forward_backward_posteriors(
        log_emissions, fitted["p"], fitted["q"], pi_archaic=pi_archaic
    )
    segments = archaic_segments(
        posteriors[1],
        span_bp,
        span_cm,
        post_threshold=post_threshold,
        min_bp=min_bp,
        min_cm=min_cm,
    )
    return admixture_time_estimate(segments, intervals, span_bp)
SCICODE_GOLD_EOF
