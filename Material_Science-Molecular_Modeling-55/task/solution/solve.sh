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

def _validate_panel(n_cal, n_test, d):
    if not isinstance(n_cal, int) or not isinstance(n_test, int) or not isinstance(d, int):
        raise ValueError("sizes must be integers")
    if n_cal < 40 or n_test < 24 or d < 6:
        raise ValueError("n_cal >= 40, n_test >= 24, and d >= 6 are required")


def generate_uncertainty_panel(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    _validate_panel(n_cal, n_test, d)
    rng = np.random.default_rng(seed)
    w = np.sin((np.arange(d * 3, dtype=float).reshape(d, 3) + 1.0) * 0.371)
    x_cal = rng.normal(0.0, 1.0, (n_cal, d))
    x_test = rng.normal(0.0, 1.0, (n_test, d))
    x_test += 0.18 * np.tanh(x_test[:, [0]]) * np.cos(np.arange(d)[None, :] + 0.5)

    def make_forces(x, z):
        pred = x @ w / np.sqrt(d) + 0.12 * np.column_stack(
            (np.sin(x[:, 0] * x[:, 1]), np.cos(x[:, 2] - x[:, 3]), np.tanh(x[:, 4] + x[:, 5]))
        )
        sigma = 0.075 + 0.019 * np.linalg.norm(x[:, :3], axis=1) + 0.013 * np.abs(np.sin(x[:, 3] + 0.4 * x[:, 5]))
        local = 0.72 + 0.28 * np.exp(0.35 * x[:, 0] - 0.18 * x[:, 2]) + 0.11 * np.abs(x[:, 6 % d])
        direction = z / np.linalg.norm(z, axis=1, keepdims=True)
        magnitude = sigma * local * (0.62 + 0.28 * np.abs(z[:, 0]) + 0.10 * np.abs(z[:, 1]))
        ref = pred + magnitude[:, None] * direction
        return pred, ref, sigma

    pred_cal, ref_cal, sigma_cal = make_forces(x_cal, rng.normal(size=(n_cal, 3)))
    pred_test, ref_test, sigma_test = make_forces(x_test, rng.normal(size=(n_test, 3)))
    return x_cal, pred_cal, ref_cal, sigma_cal, x_test, pred_test, ref_test, sigma_test

import numpy as np

def compute_site_scores(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    panel = generate_uncertainty_panel(seed, n_cal, n_test, d)
    x_cal, pred_cal, ref_cal, sigma_cal = panel[:4]
    error = np.linalg.norm(pred_cal - ref_cal, axis=1)
    score = error / sigma_cal
    checksum = float(np.dot(score, 1.0 + (np.arange(n_cal) % 13) / 17.0))
    return score, error, checksum

import numpy as np

def _finite_quantile(values, alpha):
    v = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(v) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(v))
    return float(v[rank - 1])


def fit_global_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, alpha: float = 0.5) -> tuple:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    score, error, checksum = compute_site_scores(seed, n_cal, n_test, d)
    q = _finite_quantile(score, alpha)
    pinball = float(np.mean(np.where(score >= q, (1.0 - alpha) * (score - q), alpha * (q - score))))
    return q, pinball, checksum

import numpy as np

def _finite_quantile(values, alpha):
    v = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(v) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(v))
    return float(v[rank - 1])


def _gmm_partition(x, n_classes, iterations=17):
    n, d = x.shape
    if n_classes < 2 or n_classes > min(20, n // 4):
        raise ValueError("n_classes must be between 2 and min(20, n//4)")
    mu0 = x.mean(axis=0)
    sd0 = x.std(axis=0) + 1e-12
    z = (x - mu0) / sd0
    projection = z @ (np.cos(np.arange(d) * 0.73 + 0.2) / np.sqrt(d))
    order = np.argsort(projection, kind="mergesort")
    starts = ((np.arange(n_classes) + 0.5) * n / n_classes).astype(int)
    means = z[order[np.clip(starts, 0, n - 1)]].copy()
    variances = np.tile(np.var(z, axis=0) + 0.15, (n_classes, 1))
    priors = np.full(n_classes, 1.0 / n_classes)
    for _ in range(iterations):
        logp = np.empty((n, n_classes))
        for k in range(n_classes):
            logp[:, k] = np.log(priors[k] + 1e-15) - 0.5 * np.sum(
                np.log(2.0 * np.pi * variances[k]) + (z - means[k]) ** 2 / variances[k], axis=1
            )
        logp -= logp.max(axis=1, keepdims=True)
        resp = np.exp(logp)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-9
        priors = nk / nk.sum()
        means = (resp.T @ z) / nk[:, None]
        for k in range(n_classes):
            variances[k] = (resp[:, k, None] * (z - means[k]) ** 2).sum(axis=0) / nk[k] + 0.025
    # Assign labels using the mixture returned after the last M-step.
    for k in range(n_classes):
        logp[:, k] = np.log(priors[k] + 1e-15) - 0.5 * np.sum(
            np.log(2.0 * np.pi * variances[k]) + (z - means[k]) ** 2 / variances[k], axis=1
        )
    labels = np.argmax(logp, axis=1)
    return labels, means, variances, priors, mu0, sd0


def fit_class_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, n_classes: int = 7, alpha: float = 0.5) -> tuple:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    x_cal = generate_uncertainty_panel(seed, n_cal, n_test, d)[0]
    score = compute_site_scores(seed, n_cal, n_test, d)[0]
    labels, means, variances, priors, mu0, sd0 = _gmm_partition(x_cal, n_classes)
    q_class = np.empty(n_classes)
    global_q = _finite_quantile(score, alpha)
    for k in range(n_classes):
        local = score[labels == k]
        q_class[k] = _finite_quantile(local, alpha) if len(local) >= 3 else global_q
    checksum = float(np.dot(q_class, np.arange(1, n_classes + 1)) + np.dot(priors, np.arange(n_classes) ** 2))
    return q_class, means, variances, priors, mu0, sd0, checksum

import numpy as np

def _features(x, mu, sd, width):
    z = (x - mu) / sd
    d = z.shape[1]
    grid = np.arange(d * width, dtype=float).reshape(d, width)
    w = np.sin(0.173 * (grid + 1.0)) + 0.37 * np.cos(0.113 * (grid + 3.0))
    w /= np.sqrt(d)
    b = 0.31 * np.cos(np.arange(width) * 0.47)
    return np.column_stack((np.ones(len(z)), z, np.tanh(z @ w + b)))


def fit_flexible_quantile(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, width: int = 13, iterations: int = 31, ridge: float = 0.065) -> tuple:
    if width < 5 or iterations < 5 or ridge <= 0:
        raise ValueError("width, iterations, and ridge are invalid")
    x_cal = generate_uncertainty_panel(seed, n_cal, n_test, d)[0]
    score, error, _ = compute_site_scores(seed, n_cal, n_test, d)
    mu = x_cal.mean(axis=0)
    sd = x_cal.std(axis=0) + 1e-12
    h = _features(x_cal, mu, sd, width)
    target = np.log(np.maximum(score, 1e-12))
    physical = 0.25 + error / np.median(error)
    beta = np.linalg.solve(h.T @ (physical[:, None] * h) + ridge * np.eye(h.shape[1]), h.T @ (physical * target))
    # The log-score fit is initialization only, not the optimized objective.
    for _ in range(iterations):
        residual = target - h @ beta
        irls = physical / np.sqrt(residual * residual + 2.5e-5)
        lhs = h.T @ (irls[:, None] * h) + ridge * np.eye(h.shape[1])
        rhs = h.T @ (irls * target)
        beta = np.linalg.solve(lhs, rhs)
    sigma = generate_uncertainty_panel(seed, n_cal, n_test, d)[3]

    def direct_loss(coefficients):
        q = np.exp(np.clip(h @ coefficients, -2.5, 2.5))
        residual = sigma * q - error
        return float(np.mean(physical * np.abs(residual)))

    # Deterministic subgradient Adam on the physical force-space L1 objective.
    # Ridge stabilizes the warm start; it does not alter the direct objective.
    moments = np.zeros_like(beta)
    squares = np.zeros_like(beta)
    best = beta.copy()
    best_value = direct_loss(beta)
    for step in range(1, iterations * 10 + 1):
        raw = h @ beta
        clipped = np.clip(raw, -2.5, 2.5)
        predicted = sigma * np.exp(clipped)
        residual = predicted - error
        derivative = (raw > -2.5) & (raw < 2.5)
        gradient = h.T @ (physical * np.sign(residual) * predicted * derivative) / len(error)
        moments = 0.9 * moments + 0.1 * gradient
        squares = 0.999 * squares + 0.001 * gradient * gradient
        corrected_m = moments / (1.0 - 0.9 ** step)
        corrected_v = squares / (1.0 - 0.999 ** step)
        beta = beta - 0.001 * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        value = direct_loss(beta)
        if value < best_value:
            best_value = value
            best = beta.copy()
    beta = best
    objective = best_value
    checksum = float(np.dot(beta, np.sin(np.arange(len(beta)) + 0.7)))
    return beta, mu, sd, objective, checksum

import numpy as np

def _features(x, mu, sd, width):
    z = (x - mu) / sd
    d = z.shape[1]
    grid = np.arange(d * width, dtype=float).reshape(d, width)
    w = np.sin(0.173 * (grid + 1.0)) + 0.37 * np.cos(0.113 * (grid + 3.0))
    w /= np.sqrt(d)
    b = 0.31 * np.cos(np.arange(width) * 0.47)
    return np.column_stack((np.ones(len(z)), z, np.tanh(z @ w + b)))


def _spearman(a, b):
    ra = np.empty(len(a), dtype=float)
    rb = np.empty(len(b), dtype=float)
    ra[np.argsort(a, kind="mergesort")] = np.arange(len(a), dtype=float)
    rb[np.argsort(b, kind="mergesort")] = np.arange(len(b), dtype=float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(np.dot(ra, rb) / np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))


def evaluate_transfer_tensor(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8,
                             shifts: tuple = (-0.45, -0.15, 0.2, 0.55),
                             uncertainty_scales: tuple = (0.78, 1.0, 1.24),
                             correction_scales: tuple = (0.88, 1.07, 1.31)) -> tuple:
    if min(len(shifts), len(uncertainty_scales), len(correction_scales)) < 2:
        raise ValueError("each tensor axis must contain at least two values")
    panel = generate_uncertainty_panel(seed, n_cal, n_test, d)
    x_test, pred_test, ref_test, sigma_test = panel[4:]
    source_error_vector = pred_test - ref_test
    q_global = fit_global_conformal(seed, n_cal, n_test, d)[0]
    q_class, means, variances, priors, class_mu, class_sd, _ = fit_class_conformal(seed, n_cal, n_test, d)
    beta, flex_mu, flex_sd, _, _ = fit_flexible_quantile(seed, n_cal, n_test, d)
    direction = np.cos(np.arange(d) * 0.61 + 0.4)
    rows = []
    for shift in shifts:
        x = x_test + shift * direction
        z = (x - class_mu) / class_sd
        logp = np.empty((len(x), len(q_class)))
        for k in range(len(q_class)):
            logp[:, k] = np.log(priors[k] + 1e-15) - 0.5 * np.sum(
                np.log(2.0 * np.pi * variances[k]) + (z - means[k]) ** 2 / variances[k], axis=1
            )
        q_cb = q_class[np.argmax(logp, axis=1)]
        h = _features(x, flex_mu, flex_sd, 13)
        q_flex = np.exp(np.clip(h @ beta, -2.5, 2.5))
        for us in uncertainty_scales:
            sigma = sigma_test * us * (1.0 + 0.045 * shift * shift)
            for es in correction_scales:
                correction_direction = np.column_stack((
                    np.sin(x[:, 0] + x[:, 2]),
                    np.cos(x[:, 1] - x[:, 3]),
                    np.tanh(x[:, 4] + x[:, 5]),
                ))
                correction = (0.08 * es * (1.0 + 0.25 * shift * np.tanh(x[:, 0])))[:, None] * correction_direction
                error = np.linalg.norm(source_error_vector - correction, axis=1)
                predictions = (q_global * sigma, q_cb * sigma, q_flex * sigma)
                metrics = []
                for prediction in predictions:
                    alignment = np.mean(np.abs(prediction - error)) / np.mean(error)
                    coverage = np.mean(error <= prediction)
                    rho = _spearman(prediction, error)
                    hits = []
                    for window_size in (6, 9):
                        for start in range(0, len(error), window_size):
                            stop = min(start + window_size, len(error))
                            if stop - start >= 2:
                                hits.append(float(np.argmax(prediction[start:stop]) == np.argmax(error[start:stop])))
                    window_accuracy = float(np.mean(hits))
                    top_k = min(max(2, int(np.ceil(len(error) / 6))), len(error) - 1)
                    pred_top = np.argsort(-prediction, kind="mergesort")[:top_k]
                    error_top = np.argsort(-error, kind="mergesort")[:top_k]
                    top_k_precision = len(np.intersect1d(pred_top, error_top)) / top_k
                    risk = (0.35 * alignment + 0.20 * abs(coverage - 0.5)
                            + 0.15 * (1.0 - rho) / 2.0
                            + 0.15 * (1.0 - window_accuracy)
                            + 0.15 * (1.0 - top_k_precision))
                    metrics.append((risk, coverage, rho, window_accuracy, top_k_precision))
                rows.append([v for triple in metrics for v in triple])
    tensor = np.asarray(rows, dtype=float)
    checksum = float(np.dot(tensor.ravel(), 1.0 + (np.arange(tensor.size) % 29) / 31.0))
    return tensor, checksum

import numpy as np

def compute_uncertainty_calibration_audit(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    panel = generate_uncertainty_panel(int(seed), n_cal, n_test, d)
    score, error, score_checksum = compute_site_scores(int(seed), n_cal, n_test, d)
    q_global, pinball, _ = fit_global_conformal(int(seed), n_cal, n_test, d)
    q_class, _, _, _, _, _, class_checksum = fit_class_conformal(int(seed), n_cal, n_test, d)
    beta, _, _, objective, flex_checksum = fit_flexible_quantile(int(seed), n_cal, n_test, d)
    tensor, tensor_checksum = evaluate_transfer_tensor(int(seed), n_cal, n_test, d)
    risks = tensor[:, [0, 5, 10]]
    flex = risks[:, 2]
    improvement = float(np.mean(risks[:, 0] - flex))
    median = float(np.median(flex))
    spread = float(np.std(flex, ddof=0))
    worst = float(np.max(flex))
    singular = float(np.linalg.svd(risks - risks.mean(axis=0), compute_uv=False)[0])
    transfer = float(np.mean(flex.reshape(4, 3, 3)[-1]))
    j = float(np.exp(-(1.7 * improvement + 0.8 * median + 0.6 * spread + 0.25 * worst + 0.15 * transfer + 0.04 * singular)) / (1.0 + objective + pinball))
    j = float(np.round(j, 8))
    return (j, q_global, score_checksum, class_checksum, objective, flex_checksum,
            tensor_checksum, improvement, median, spread, worst, singular, transfer)
SCICODE_GOLD_EOF
