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
def build_homolog_pair(
    seed: int,
    length: int,
    delete_probability: float,
    substitute_probability: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation: three ordered draws, then one masked copy."""
    import numpy as np

    for name, value, low in (("seed", seed, 0), ("length", length, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}")
    probabilities = []
    for name, value in (("delete_probability", delete_probability),
                        ("substitute_probability", substitute_probability)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a number")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        probabilities.append(value)
    delete, substitute = probabilities
    if delete + substitute > 1.0:
        raise ValueError("the deletion and substitution probabilities sum to more than 1")

    rng = np.random.default_rng(int(seed))
    x = rng.integers(0, 4, int(length)).astype(np.int64)
    u = rng.random(int(length))
    v = rng.integers(0, 4, int(length)).astype(np.int64)
    redrawn = np.where(u < delete + substitute, v, x)
    y = redrawn[u >= delete].astype(np.int64)
    if y.size == 0:
        raise ValueError("every base was deleted, so the homolog is empty")
    return x, y

import numpy as np
def compute_log_forward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Reference implementation sweeping anti-diagonals in the log domain."""
    import numpy as np

    xs, ys, log_m, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    table = np.full((3, n + 1, m + 1), -np.inf)
    fm, fx, fy = table  # views into the returned table
    fm[0, 0] = 0.0
    # Every cell on anti-diagonal d depends only on diagonals d - 1 and d - 2.
    for d in range(1, n + m + 1):
        i = np.arange(max(0, d - m), min(n, d) + 1)
        j = d - i
        keep = (i > 0) & (j > 0)
        a, b = i[keep], j[keep]
        fm[a, b] = log_m[a - 1, b - 1] + np.logaddexp(
            np.logaddexp(fm[a - 1, b - 1], fx[a - 1, b - 1]), fy[a - 1, b - 1])
        keep = i > 0
        a, b = i[keep], j[keep]
        fx[a, b] = log_x[a - 1] + np.logaddexp(log_open + fm[a - 1, b], log_extend + fx[a - 1, b])
        keep = j > 0
        a, b = i[keep], j[keep]
        fy[a, b] = log_y[b - 1] + np.logaddexp(log_open + fm[a, b - 1], log_extend + fy[a, b - 1])
    return table


def _validate_pair_model(x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend):
    """Return validated codes and the natural-log factors of the pair model."""
    import numpy as np

    xs, ys = (np.asarray(s) for s in (x, y))
    for name, a in (("x", xs), ("y", ys)):
        if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 3:
            raise ValueError(f"{name} must be a non-empty 1-D integer array of codes 0 to 3")
    em, ex, ey = (np.asarray(v, dtype=float) for v in (match_factors, x_gap_factors, y_gap_factors))
    if (em.shape, ex.shape, ey.shape) != ((4, 4), (4,), (4,)):
        raise ValueError("factor arrays must have shapes (4, 4), (4,) and (4,)")
    every = np.concatenate([em.ravel(), ex, ey, np.array([gap_open, gap_extend], dtype=float)])
    if not np.all(np.isfinite(every)) or np.any(every <= 0.0):
        raise ValueError("every factor must be finite and positive")
    xs, ys = xs.astype(np.int64), ys.astype(np.int64)
    return (xs, ys, np.log(em)[xs][:, ys], np.log(ex)[xs], np.log(ey)[ys],
            float(np.log(every[-2])), float(np.log(every[-1])))

import numpy as np
def compute_log_backward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Reference implementation sweeping anti-diagonals backward in the log domain."""
    import numpy as np

    xs, ys, log_m, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    table = np.full((3, n + 1, m + 1), -np.inf)
    bm, bx, by = table  # views into the returned table
    table[:, n, m] = 0.0
    # Cells on anti-diagonal d depend only on diagonals d + 1 and d + 2.
    for d in range(n + m - 1, -1, -1):
        i = np.arange(max(0, d - m), min(n, d) + 1)
        j = d - i
        to_match = np.full(i.size, -np.inf)
        to_xgap = np.full(i.size, -np.inf)
        to_ygap = np.full(i.size, -np.inf)
        keep = (i < n) & (j < m)
        to_match[keep] = log_m[i[keep], j[keep]] + bm[i[keep] + 1, j[keep] + 1]
        keep = i < n
        to_xgap[keep] = log_x[i[keep]] + bx[i[keep] + 1, j[keep]]
        keep = j < m
        to_ygap[keep] = log_y[j[keep]] + by[i[keep], j[keep] + 1]
        bm[i, j] = np.logaddexp(to_match, np.logaddexp(log_open + to_xgap, log_open + to_ygap))
        bx[i, j] = np.logaddexp(to_match, log_extend + to_xgap)
        by[i, j] = np.logaddexp(to_match, log_extend + to_ygap)
    return table

import numpy as np
def compute_expected_gap_transition_counts(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation summing local-event posteriors of the gap transitions."""
    import numpy as np

    xs, ys, _, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    f, b = _validate_log_tables(log_forward, log_backward, n, m)
    log_z = np.logaddexp.reduce(f[:, n, m])
    # A column in X at (i, j) follows lattice point (i - 1, j); in Y at (i, j) it follows (i, j - 1).
    into_x = log_x[:, None] + b[1, 1:, :]
    into_y = log_y[None, :] + b[2, :, 1:]
    opened = np.logaddexp(np.logaddexp.reduce(log_open + f[0, :-1, :] + into_x, axis=None),
                          np.logaddexp.reduce(log_open + f[0, :, :-1] + into_y, axis=None))
    extended = np.logaddexp(np.logaddexp.reduce(log_extend + f[1, :-1, :] + into_x, axis=None),
                            np.logaddexp.reduce(log_extend + f[2, :, :-1] + into_y, axis=None))
    return np.exp(np.array([opened, extended]) - log_z)


def _validate_log_tables(log_forward, log_backward, n, m):
    """Return validated float tables of shape (3, n + 1, m + 1)."""
    import numpy as np

    tables = [np.asarray(t, dtype=float) for t in (log_forward, log_backward)]
    for t in tables:
        if t.shape != (3, n + 1, m + 1) or np.any(np.isnan(t)) or np.any(t == np.inf):
            raise ValueError("log tables must have shape (3, L + 1, K + 1) without NaN or +inf")
    if not np.isfinite(np.logaddexp.reduce(tables[0][:, n, m])):
        raise ValueError("the reference total weight must be positive")
    return tables[0], tables[1]

import numpy as np
def calibrate_gap_factors(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    open_count: float,
    extend_count: float,
) -> "np.ndarray":
    """Reference implementation: damped quasi-Newton iteration in the log factors."""
    import numpy as np

    _validate_pair_model(x, y, match_factors, x_gap_factors, y_gap_factors, 1.0, 1.0)
    targets = np.array([open_count, extend_count], dtype=float)
    if targets.shape != (2,) or not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("target counts must be finite and positive")
    log_targets = np.log(targets)

    def _residual_at(theta):
        # Log counts minus log targets; infinite where the factors leave the float range.
        if np.any(np.abs(theta) > 690.0):
            return np.full(2, np.inf)
        args = (x, y, match_factors, x_gap_factors, y_gap_factors, *np.exp(theta))
        counts = compute_expected_gap_transition_counts(
            *args, compute_log_forward_tables(*args), compute_log_backward_tables(*args))
        with np.errstate(divide="ignore"):
            return np.log(counts) - log_targets

    def _jacobian_at(theta, residual, step=1.0e-6):
        columns = [(_residual_at(theta + step * np.eye(2)[k]) - residual) / step for k in range(2)]
        return np.column_stack(columns)

    theta = np.log([0.05, 0.5])
    residual = _residual_at(theta)
    if not np.all(np.isfinite(residual)):
        raise ValueError("no positive gap factors reproduce the target counts")
    jacobian, fresh = _jacobian_at(theta, residual), True
    for _ in range(100):
        norm = np.max(np.abs(residual))
        # Summed counts carry rounding near 1e-12 for long sequences, so stop well above it.
        if norm <= 1.0e-11:
            break
        accepted = False
        if np.all(np.isfinite(jacobian)) and abs(np.linalg.det(jacobian)) > 1.0e-14:
            step = -np.linalg.solve(jacobian, residual)
            for scale in 0.5 ** np.arange(12):
                trial = theta + scale * step
                trial_residual = _residual_at(trial)
                if np.all(np.isfinite(trial_residual)) and np.max(np.abs(trial_residual)) < norm:
                    accepted = True
                    break
        if accepted:
            # Broyden rank-one update keeps most iterations at one model evaluation.
            delta, change = trial - theta, trial_residual - residual
            jacobian = jacobian + np.outer(change - jacobian @ delta, delta) / (delta @ delta)
            theta, residual, fresh = trial, trial_residual, False
        elif not fresh:
            jacobian, fresh = _jacobian_at(theta, residual), True
        else:
            break
    if np.all(np.isfinite(residual)) and np.max(np.abs(residual)) <= 1.0e-10:
        return np.exp(theta)
    raise ValueError("no positive gap factors reproduce the target counts")

import numpy as np
def evaluate_single_substitution_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation contracting per-base coefficients with new factors."""
    import numpy as np

    # The y-gap and transition factors do not enter this contraction, so
    # placeholders stand in for them during validation.
    xs, ys, log_m, log_x, _, _, _ = _validate_pair_model(
        x, y, match_factors, x_gap_factors, x_gap_factors, 1.0, 1.0)
    f, b = _validate_log_tables(log_forward, log_backward, xs.size, ys.size)
    new_match = np.log(np.asarray(match_factors, dtype=float))[:, ys]
    new_xgap = np.log(np.asarray(x_gap_factors, dtype=float))
    # Each complete alignment uses base i once: in M at some column j, or in X.
    match_coefficient = f[0, 1:, 1:] + b[0, 1:, 1:] - log_m
    xgap_coefficient = np.logaddexp.reduce(f[1, 1:, :] + b[1, 1:, :], axis=1) - log_x
    result = np.empty((xs.size, 4))
    for code in range(4):
        through_match = np.logaddexp.reduce(match_coefficient + new_match[code], axis=1)
        result[:, code] = np.logaddexp(through_match, xgap_coefficient + new_xgap[code])
    return result

import numpy as np
def compute_neighbour_double_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation: outside context, joint two-row block, inside context."""
    import numpy as np

    xs, ys, log_m, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    if n < 2:
        raise ValueError("x must have at least two bases")
    f, b = _validate_log_tables(log_forward, log_backward, n, m)
    new_match = np.log(np.asarray(match_factors, dtype=float))[:, ys]
    new_xgap = np.log(np.asarray(x_gap_factors, dtype=float))
    result = np.empty((n - 1, 4, 4))
    for start in range(0, n - 1, 256):
        i = np.arange(start, min(n - 1, start + 256))
        # Weight up to and including base i placed in M (columns 1..m) or X (0..m).
        into_i_match = np.full((i.size, 4, m + 1), -np.inf)
        into_i_match[:, :, 1:] = (f[0, i + 1, 1:] - log_m[i])[:, None, :] + new_match[None]
        into_i_xgap = (f[1, i + 1, :] - log_x[i][:, None])[:, None, :] + new_xgap[None, :, None]
        # Columns of y placed against gaps between the two bases follow only M.
        y_run = np.full((i.size, 4, m + 1), -np.inf)
        for col in range(1, m + 1):
            y_run[:, :, col] = log_y[col - 1] + np.logaddexp(
                log_open + into_i_match[:, :, col - 1], log_extend + y_run[:, :, col - 1])
        before_match = np.logaddexp(np.logaddexp(into_i_match[:, :, :-1], into_i_xgap[:, :, :-1]),
                                    y_run[:, :, :-1])
        before_xgap = np.logaddexp(log_open + into_i_match, log_extend + into_i_xgap)
        after_match, after_xgap = b[0, i + 2, 1:], b[1, i + 2, :]
        for code in range(4):
            via_match = np.logaddexp.reduce(
                before_match + (new_match[code] + after_match)[:, None, :], axis=2)
            via_xgap = np.logaddexp.reduce(
                before_xgap + (new_xgap[code] + after_xgap)[:, None, :], axis=2)
            result[i, :, code] = np.logaddexp(via_match, via_xgap)
    return result

import numpy as np
def summarize_neighbour_epistasis(
    x: "np.ndarray",
    single_log_weights: "np.ndarray",
    double_log_weights: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation on the natural-log scale."""
    import numpy as np

    codes = np.asarray(x)
    if (codes.ndim != 1 or codes.size < 2 or not np.issubdtype(codes.dtype, np.integer)
            or codes.min() < 0 or codes.max() > 3):
        raise ValueError("x must be a 1-D integer array of codes 0 to 3 with at least two bases")
    n = codes.size
    single = np.asarray(single_log_weights, dtype=float)
    double = np.asarray(double_log_weights, dtype=float)
    if single.shape != (n, 4) or double.shape != (n - 1, 4, 4):
        raise ValueError("log weights must have shapes (L, 4) and (L - 1, 4, 4)")
    if not (np.all(np.isfinite(single)) and np.all(np.isfinite(double))):
        raise ValueError("log weights must be finite")
    positions = np.arange(n)
    log_z = single[0, codes[0]]
    reference = np.concatenate([single[positions, codes],
                                double[positions[:-1], codes[:-1], codes[1:]]])
    if np.max(np.abs(reference - log_z)) > 1e-8 * max(1.0, abs(log_z)):
        raise ValueError("unsubstituted entries disagree with the reference total weight")

    epistasis = double + log_z - single[:-1, :, None] - single[1:, None, :]
    genuine = np.ones(epistasis.shape, dtype=bool)
    genuine[positions[:-1], codes[:-1], :] = False
    genuine[positions[:-1], :, codes[1:]] = False
    values = epistasis[genuine]
    flat = np.argmax(np.where(genuine, np.abs(epistasis), -1.0))
    i, c, d = np.unravel_index(flat, epistasis.shape)
    return np.array([values.mean(), epistasis[i, c, d], i + 1, c, d,
                     np.mean(values > 0.0), values.size], dtype=float)

import numpy as np
def estimate_mean_neighbour_epistasis(
    seed: int = 20260914,
    length: int = 2000,
    delete_probability: float = 0.06,
    substitute_probability: float = 0.16,
    match_factors: "tuple" = ((0.16, 0.03, 0.05, 0.03), (0.03, 0.16, 0.03, 0.05),
                              (0.05, 0.03, 0.16, 0.03), (0.03, 0.05, 0.03, 0.16)),
    x_gap_factors: "tuple" = (0.12, 0.38, 0.38, 0.12),
    y_gap_factors: "tuple" = (0.16, 0.34, 0.34, 0.16),
    open_count: float = 115.0,
    extend_count: float = 32.0,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    x, y = build_homolog_pair(seed, length, delete_probability, substitute_probability)
    if x.size < 2:
        raise ValueError("length must be at least 2")
    emissions = (np.asarray(match_factors, dtype=float), np.asarray(x_gap_factors, dtype=float),
                 np.asarray(y_gap_factors, dtype=float))
    gap_open, gap_extend = calibrate_gap_factors(x, y, *emissions, open_count, extend_count)
    model = (*emissions, float(gap_open), float(gap_extend))
    log_forward = compute_log_forward_tables(x, y, *model)
    log_backward = compute_log_backward_tables(x, y, *model)
    counts = compute_expected_gap_transition_counts(x, y, *model, log_forward, log_backward)
    if np.max(np.abs(counts / np.array([open_count, extend_count], dtype=float) - 1.0)) > 1.0e-8:
        raise ValueError("the fitted gap factors do not reproduce the target counts")
    single = evaluate_single_substitution_log_weights(
        x, y, emissions[0], emissions[1], log_forward, log_backward)
    double = compute_neighbour_double_log_weights(x, y, *model, log_forward, log_backward)
    summary = summarize_neighbour_epistasis(x, single, double)
    return float(summary[0])
SCICODE_GOLD_EOF
