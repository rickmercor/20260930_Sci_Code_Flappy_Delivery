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


def _ref_check_box(xmin, xmax):
    """Validate the box bounds and return them as float arrays."""
    xmin = np.asarray(xmin, dtype=float)
    xmax = np.asarray(xmax, dtype=float)
    if xmin.ndim != 1 or xmax.ndim != 1:
        raise ValueError("xmin and xmax must be one-dimensional arrays")
    if xmin.size != xmax.size or xmin.size < 1:
        raise ValueError("xmin and xmax must have the same length n >= 1")
    if not (np.all(np.isfinite(xmin)) and np.all(np.isfinite(xmax))):
        raise ValueError("xmin and xmax must be finite")
    if np.any(xmax <= xmin):
        raise ValueError("the box must be nonempty in every coordinate")
    if np.any(xmin <= -1.0):
        raise ValueError("every lower bound must exceed -1")
    return xmin, xmax


def attainable_return_range(xmin: np.ndarray, xmax: np.ndarray,
                                    cap: float) -> np.ndarray:
    """Reference implementation."""
    xmin, xmax = _ref_check_box(xmin, xmax)
    if not np.isscalar(cap) or not np.isfinite(float(cap)):
        raise ValueError("cap must be a finite scalar")
    cap = float(cap)
    n = xmin.size
    if cap <= 0.0 or n * cap < 1.0:
        raise ValueError("the admissible portfolio set must be nonempty")

    endpoints = []
    for values, largest in ((xmin, False), (xmax, True)):
        order = np.argsort(-values if largest else values, kind="stable")
        remaining = 1.0
        total = 0.0
        for idx in order:
            if remaining <= 0.0:
                break
            take = cap if cap < remaining else remaining
            total += take * values[idx]
            remaining -= take
        endpoints.append(total)
    return np.array(endpoints, dtype=float)

import numpy as np


def tangent_coefficients(y_range: np.ndarray, M: int) -> np.ndarray:
    """Reference implementation."""
    y_range = np.asarray(y_range, dtype=float)
    if y_range.shape != (2,):
        raise ValueError("y_range must have shape (2,)")
    if not np.all(np.isfinite(y_range)):
        raise ValueError("y_range must be finite")
    y_lo, y_hi = float(y_range[0]), float(y_range[1])
    if y_lo >= y_hi:
        raise ValueError("y_range must be a nondegenerate interval")
    if y_lo <= -1.0:
        raise ValueError("the utility must be finite on the interval")
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)):
        raise ValueError("M must be an integer")
    M = int(M)
    if M < 2:
        raise ValueError("at least two tangent lines are required")

    points = np.linspace(y_lo, y_hi, M)
    alpha = 1.0 / (1.0 + points)
    beta = np.log1p(points) - alpha * points
    return np.vstack([alpha, beta])

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _ref_observations(returns, xmin, xmax):
    """Validate the observed returns against the box and return them as floats."""
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] != xmin.size:
        raise ValueError("returns must have shape (N, n) with N >= 1")
    if not np.all(np.isfinite(X)):
        raise ValueError("returns must be finite")
    if np.any(X < xmin) or np.any(X > xmax):
        raise ValueError("every observation must lie in the box")
    return X


def _ref_unit_weights(weights, n):
    """Validate a long-only, fully invested weight vector of length n."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be a finite nonnegative (n,) array")
    if abs(float(np.sum(w)) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one")
    return w


def _ref_surrogate_pieces(coeffs):
    """Validate the affine pieces and return (slopes, intercepts)."""
    coeffs = np.asarray(coeffs, dtype=float)
    if (coeffs.ndim != 2 or coeffs.shape[0] != 2 or coeffs.shape[1] < 1
            or not np.all(np.isfinite(coeffs))):
        raise ValueError("coeffs must be a finite (2, M) array")
    if np.any(coeffs[0] < 0.0):
        raise ValueError("the slopes must be nonnegative")
    return coeffs[0], coeffs[1]


def _ref_positive_radius(radius):
    """Validate the radius of the Wasserstein ball."""
    if not np.isscalar(radius) or not np.isfinite(float(radius)) or float(radius) <= 0.0:
        raise ValueError("radius must be a finite positive scalar")
    return float(radius)


def surrogate_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                                 returns: np.ndarray, coeffs: np.ndarray,
                                 radius: float) -> float:
    """Reference implementation."""
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    w = _ref_unit_weights(weights, xmin.size)
    alpha, beta = _ref_surrogate_pieces(coeffs)
    radius = _ref_positive_radius(radius)
    N, M = X.shape[0], alpha.size

    # Transport dual with the box support dualised: for each piece the inner
    # minimum over the box is alpha_m <w, x_j> + beta_m minus the l1 cost of
    # moving each held asset to its lower bound where alpha_m w_i exceeds lam.
    held = np.flatnonzero(w > 0.0)
    k = held.size
    D = (X - xmin)[:, held]
    base = alpha[None, :] * (X @ w)[:, None] + beta[None, :]
    nv = 1 + N + M * k
    c = np.zeros(nv)
    c[0] = radius
    c[1:1 + N] = -1.0 / N
    r1 = np.arange(N * M)
    jj, mm = np.divmod(r1, M)
    rows = [r1, np.repeat(r1, k)]
    cols = [1 + jj, (1 + N + mm[:, None] * k + np.arange(k)[None, :]).ravel()]
    vals = [np.ones(N * M), D[jj].ravel()]
    r2 = N * M + np.arange(M * k)
    rows += [r2, r2]
    cols += [np.zeros(M * k, dtype=int), 1 + N + np.arange(M * k)]
    vals += [-np.ones(M * k), -np.ones(M * k)]
    A_ub = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N * M + M * k, nv))
    b_ub = np.concatenate([base.ravel(), -(alpha[:, None] * w[held][None, :]).ravel()])
    bounds = [(0.0, None)] + [(None, None)] * N + [(0.0, None)] * (M * k)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    lam = float(res.x[0])
    excess = np.maximum(alpha[:, None] * w[held][None, :] - lam, 0.0)
    inner = base - D @ excess.T
    return float(-lam * radius + inner.min(axis=1).mean())

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def surrogate_portfolio(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                                coeffs: np.ndarray, cap: float, radius: float) -> np.ndarray:
    """Reference implementation."""
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    alpha, beta = _ref_surrogate_pieces(coeffs)
    if not np.isscalar(cap) or not np.isfinite(float(cap)):
        raise ValueError("cap must be a finite scalar")
    cap = float(cap)
    N, n = X.shape
    if cap <= 0.0 or n * cap < 1.0:
        raise ValueError("the admissible portfolio set must be nonempty")
    radius = _ref_positive_radius(radius)
    M = alpha.size

    # Box-specialised hyperplane-dual linear programme. Variables
    # z = [w (n), lam, a (N), s^1..s^M (n each)] with s^m >= alpha_m w - lam,
    # s^m >= 0 shared by all observations.
    il, ia, i_s = n, n + 1, n + 1 + N
    nv = n + 1 + N + M * n
    c = np.zeros(nv)
    c[il] = radius
    c[ia:ia + N] = -1.0 / N
    D = X - xmin
    r1 = np.arange(N * M)
    jj, mm = np.divmod(r1, M)
    rows_w = np.repeat(r1, n)
    cols_w = np.tile(np.arange(n), N * M)
    vals_w = (-alpha[mm][:, None] * X[jj]).ravel()
    cols_s = (i_s + mm[:, None] * n + np.arange(n)[None, :]).ravel()
    vals_s = D[jj].ravel()
    k2 = np.arange(M * n)
    m2, i2 = np.divmod(k2, n)
    r2 = N * M + k2
    rows = np.concatenate([rows_w, r1, rows_w, r2, r2, r2])
    cols = np.concatenate([cols_w, ia + jj, cols_s, i2, np.full(M * n, il), i_s + k2])
    vals = np.concatenate([vals_w, np.ones(N * M), vals_s, alpha[m2],
                           -np.ones(M * n), -np.ones(M * n)])
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(N * M + M * n, nv))
    b_ub = np.concatenate([beta[mm], np.zeros(M * n)])
    A_eq = sparse.csr_matrix((np.ones(n), (np.zeros(n, dtype=int), np.arange(n))),
                             shape=(1, nv))
    bounds = ([(0.0, cap)] * n + [(0.0, None)] + [(None, None)] * N
              + [(0.0, None)] * (M * n))
    # The optimum is flat (a small loss of value lets w move far), so the vertex is
    # located at the tightest feasibility tolerances HiGHS accepts.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds,
                  method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    return np.clip(res.x[:n], 0.0, cap)

import numpy as np


def _ref_prefix_corners(w, X, xmin):
    """Portfolio return and l1 distance of each row's candidate corners."""
    order = np.argsort(-w, kind="stable")
    d = (X - xmin)[:, order]
    zero = np.zeros((X.shape[0], 1))
    distance = np.hstack([zero, np.cumsum(d, axis=1)])
    reduction = np.hstack([zero, np.cumsum(d * w[order], axis=1)])
    return (X @ w)[:, None] - reduction, distance


def inner_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                             returns: np.ndarray, lam: float) -> np.ndarray:
    """Reference implementation."""
    if not np.isscalar(lam) or not np.isfinite(float(lam)) or float(lam) < 0.0:
        raise ValueError("lam must be a finite nonnegative scalar")
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    w = _ref_unit_weights(weights, xmin.size)
    portfolio_return, distance = _ref_prefix_corners(w, X, xmin)
    return np.min(np.log1p(portfolio_return) + float(lam) * distance, axis=1)

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def delivered_value(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                            returns: np.ndarray, radius: float) -> float:
    """Reference implementation."""
    radius = _ref_positive_radius(radius)
    xmin, xmax = _ref_check_box(xmin, xmax)
    X = _ref_observations(returns, xmin, xmax)
    w = _ref_unit_weights(weights, xmin.size)
    portfolio_return, distance = _ref_prefix_corners(w, X, xmin)
    N, K = portfolio_return.shape
    utility = np.log1p(portfolio_return).ravel()
    distance = distance.ravel()
    # Transport dual with the portfolio fixed: maximise -lam*radius + mean_j a_j
    # subject to a_j <= log(1 + y_jk) + lam * d_jk over every prefix corner k.
    rows = np.arange(N * K)
    A_ub = sparse.csr_matrix(
        (np.concatenate([-distance, np.ones(N * K)]),
         (np.concatenate([rows, rows]),
          np.concatenate([np.zeros(N * K, dtype=int), 1 + rows // K]))),
        shape=(N * K, 1 + N))
    c = np.concatenate([[radius], -np.ones(N) / N])
    res = linprog(c, A_ub=A_ub, b_ub=utility, bounds=[(0.0, None)] + [(None, None)] * N,
                  method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    lam = float(res.x[0])
    return float(-lam * radius + np.mean(inner_worst_case(w, xmin, xmax, X, lam)))

import numpy as np


def surrogate_shortfall(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                                cap: float, radius: float, M: int) -> float:
    """Reference implementation."""
    y_range = attainable_return_range(xmin, xmax, cap)
    coeffs = tangent_coefficients(y_range, M)
    weights = surrogate_portfolio(xmin, xmax, returns, coeffs, cap, radius)
    surrogate_value = surrogate_worst_case(weights, xmin, xmax, returns, coeffs, radius)
    delivered = delivered_value(weights, xmin, xmax, returns, radius)
    return float(surrogate_value - delivered)
SCICODE_GOLD_EOF
