#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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


def effective_kernel_matrix(x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, ell: float, prior_scale: float) -> "np.ndarray":
    positions, lo, hi = _positions_in_range(x, lo, hi)
    if int(n_basis) < 4:
        raise ValueError("n_basis must be at least four")
    sigma = _check_positive(sigma, "sigma")
    ell = _check_positive(ell, "ell")
    scale = _check_positive(prior_scale, "prior_scale")
    design = _bspline_design_matrix(positions, lo, hi, int(n_basis))
    return _matern52(positions, sigma, ell) + scale ** 2 * design @ design.T

import numpy as np


def _check_counts_widths(counts, widths) -> tuple:
    a = np.asarray(counts, dtype=float)
    w = np.asarray(widths, dtype=float)
    if a.ndim != 1 or a.shape[0] < 1 or not np.all(np.isfinite(a)) or np.any(a < 0.0) or np.any(a != np.round(a)):
        raise ValueError("counts must be a finite one-dimensional array of non-negative integers")
    if w.shape != a.shape or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("widths must be a finite one-dimensional array of the same length with positive entries")
    return a, w


def _check_covariance(K, size: int) -> "np.ndarray":
    """Validate a symmetric positive definite (size, size) matrix and return it as float."""
    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError("K must be a finite square matrix matching the number of bins")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9):
        raise ValueError("K must be symmetric to within 1e-9")
    try:
        np.linalg.cholesky(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        raise ValueError("K must be positive definite")
    return matrix


def lgcp_posterior_mode(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    a, w = _check_counts_widths(counts, widths)
    prior = _check_covariance(K, a.shape[0])
    precision = np.linalg.inv(prior)
    f = np.log((a + 0.5) / w)
    for _ in range(500):
        rate = np.exp(f) * w
        gradient = a - rate - precision @ f
        step = np.linalg.solve(np.diag(rate) + precision, gradient)
        # damp the Newton step so that the rate cannot overflow far from the mode
        scale = min(1.0, 5.0 / float(np.max(np.abs(step))))
        f = f + scale * step
        if np.max(np.abs(step)) < 1e-13:
            break
    return f

import numpy as np


def laplace_posterior_covariance(f_hat: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
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

import numpy as np
from scipy.special import gammaln


def laplace_log_marginal_likelihood(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> float:
    a, w = _check_counts_widths(counts, widths)
    prior = _check_covariance(K, a.shape[0])
    f = lgcp_posterior_mode(a, w, prior)
    rates = np.exp(f) * w
    log_likelihood = float(np.sum(a * np.log(rates) - rates - gammaln(a + 1.0)))
    penalty = 0.5 * float(f @ np.linalg.solve(prior, f))
    _, log_det = np.linalg.slogdet(np.eye(a.shape[0]) + prior * rates[None, :])
    return log_likelihood - penalty - 0.5 * float(log_det)

import numpy as np


def select_length_scale(counts: "np.ndarray", widths: "np.ndarray", x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray") -> float:
    a, w = _check_counts_widths(counts, widths)
    grid = np.asarray(ell_grid, dtype=float)
    if grid.ndim != 1 or grid.shape[0] < 1 or not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("ell_grid must be a non-empty one-dimensional array of finite positive values")
    best_value, best_ell = None, None
    for ell in grid:
        prior = effective_kernel_matrix(x, lo, hi, n_basis, sigma, float(ell), prior_scale)
        if prior.shape[0] != a.shape[0]:
            raise ValueError("x must have one centre per bin")
        value = laplace_log_marginal_likelihood(a, w, prior)
        # a strict comparison keeps the first candidate on an exact tie
        if best_value is None or value > best_value:
            best_value, best_ell = value, float(ell)
    return best_ell

import numpy as np


def combined_template_covariance(counts: "np.ndarray", variations: list, widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
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
    mode = lgcp_posterior_mode(a, w, prior)
    covariance = laplace_posterior_covariance(mode, w, prior)
    for plus, minus in pairs:
        direction = 0.5 * (lgcp_posterior_mode(plus, w, prior) - lgcp_posterior_mode(minus, w, prior))
        covariance = covariance + np.outer(direction, direction)
    return covariance

import math

import numpy as np


def _sorted_eigenpairs(Sigma) -> tuple:
    """Eigenvalues in decreasing order with matching eigenvectors as columns, after validation."""
    matrix = np.asarray(Sigma, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("Sigma must be a finite square matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-9):
        raise ValueError("Sigma must be symmetric to within 1e-9")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    trace = float(np.sum(values))
    if not trace > 0.0:
        raise ValueError("Sigma must have a trace above zero")
    if np.any(values < -1e-9 * trace):
        raise ValueError("Sigma must not have an eigenvalue below -1e-9 times its trace")
    return values, vectors, trace


def truncated_eigenvalues(Sigma: "np.ndarray", fraction: float) -> "np.ndarray":
    target = float(fraction)
    if not (math.isfinite(target) and 0.0 < target <= 1.0):
        raise ValueError("fraction must be finite, above zero and at most one")
    values, _, trace = _sorted_eigenpairs(Sigma)
    cumulative = np.cumsum(values) / trace
    count = int(np.searchsorted(cumulative, target, side="left")) + 1
    return values[:min(count, values.shape[0])]

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


def eigenmode_template_fit(observed: "np.ndarray", signal: "np.ndarray", template: "np.ndarray", Sigma: "np.ndarray", fraction: float) -> tuple:
    n, s, b = _check_channel(observed, signal, template)
    values, vectors, _ = _sorted_eigenpairs(Sigma)
    if values.shape[0] != n.shape[0]:
        raise ValueError("Sigma must match the number of bins")
    kept = truncated_eigenvalues(Sigma, fraction)
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


def barlow_beeston_fit(observed: "np.ndarray", signal: "np.ndarray", counts: "np.ndarray", variations: list, tau: float) -> tuple:
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

import numpy as np


def orchestrate_template_comparison(edges: "np.ndarray", counts: "np.ndarray", variations: list, signal: "np.ndarray", observed: "np.ndarray", tau: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray", fraction: float) -> tuple:
    boundaries = np.asarray(edges, dtype=float)
    if boundaries.ndim != 1 or boundaries.shape[0] < 2 or not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("edges must be a finite strictly increasing one-dimensional array with at least two entries")
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])
    widths = np.diff(boundaries)
    a = np.asarray(counts, dtype=float)
    if a.shape != centres.shape or not np.all(np.isfinite(a)) or np.any(a <= 0.0) or np.any(a != np.round(a)):
        raise ValueError("counts must be integers above zero, one per bin")
    lo, hi = float(boundaries[0]), float(boundaries[-1])
    ell = select_length_scale(a, widths, centres, lo, hi, n_basis, sigma, prior_scale, ell_grid)
    prior = effective_kernel_matrix(centres, lo, hi, n_basis, sigma, ell, prior_scale)
    mode = lgcp_posterior_mode(a, widths, prior)
    statistical = laplace_posterior_covariance(mode, widths, prior)
    template = np.exp(mode) * widths / float(tau)
    combined = combined_template_covariance(a, variations, widths, prior)
    n_modes = truncated_eigenvalues(combined, fraction).shape[0]
    mu_gp, sigma_gp = eigenmode_template_fit(observed, signal, template, combined, fraction)
    mu_bb, sigma_bb = barlow_beeston_fit(observed, signal, a, variations, tau)
    ratio = float(np.mean(np.sqrt(np.diag(statistical)) * np.sqrt(a)))
    return (float(mu_gp), float(sigma_gp), float(n_modes), float(mu_bb), float(sigma_bb), float(ell), ratio, float(mu_gp - mu_bb))
SCICODE_GOLD_EOF
