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


def simulate_correlated_antithetic_gbm_paths(s0: "np.ndarray", r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, seed: int) -> "np.ndarray":
    s0_arr = np.asarray(s0, dtype=float)
    half = int(n_paths) // 2
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((int(steps), half, 2))
    corr = np.empty_like(raw)
    corr[..., 0] = raw[..., 0]
    corr[..., 1] = float(rho) * raw[..., 0] + np.sqrt(1.0 - float(rho) ** 2) * raw[..., 1]
    shocks = np.concatenate((corr, -corr), axis=1)
    dt = float(maturity) / int(steps)
    drift = (float(r) - float(q) - 0.5 * float(sigma) ** 2) * dt
    vol = float(sigma) * np.sqrt(dt)
    paths = np.empty((int(steps) + 1, int(n_paths), 2), dtype=float)
    paths[0] = s0_arr
    for t in range(int(steps)):
        paths[t + 1] = paths[t] * np.exp(drift + vol * shocks[t])
    return paths

import numpy as np


def _basis_exponents(degree: int):
    exponents = []
    for total in range(int(degree) + 1):
        for a in range(total, -1, -1):
            exponents.append((a, total - a))
    return exponents


def min_put_polynomial_basis(states: "np.ndarray", strike: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(states, dtype=float)
    if bool(sort_state):
        x = np.sort(x, axis=1)[:, ::-1]
    z = x / float(strike) - 1.0
    return np.column_stack([(z[:, 0] ** a) * (z[:, 1] ** b) for a, b in _basis_exponents(int(degree))])

import numpy as np


def _min_put_payoff(states: "np.ndarray", strike: float) -> "np.ndarray":
    x = np.asarray(states, dtype=float)
    return np.maximum(float(strike) - np.minimum(x[..., 0], x[..., 1]), 0.0)


def _ols(Phi: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    return np.linalg.lstsq(np.asarray(Phi, dtype=float), np.asarray(y, dtype=float), rcond=None)[0]


def fit_min_put_primal_policy(paths: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    steps = x.shape[0] - 1
    p = len(_basis_exponents(int(degree)))
    theta = np.zeros((steps + 1, p), dtype=float)
    realized = _min_put_payoff(x[steps], strike)
    disc = np.exp(-float(r) * float(dt))
    for t in range(steps - 1, 0, -1):
        regressand = disc * realized
        Phi = min_put_polynomial_basis(x[t], strike, degree, sort_state)
        theta[t] = _ols(Phi, regressand)
        continuation = Phi @ theta[t]
        immediate = _min_put_payoff(x[t], strike)
        exercise = (immediate > 0.0) & (immediate >= continuation)
        realized = np.where(exercise, immediate, regressand)
    return theta

import numpy as np


def evaluate_min_put_payoff_process(paths: "np.ndarray", theta: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    th = np.asarray(theta, dtype=float)
    steps = x.shape[0] - 1
    H = np.zeros((steps + 1, x.shape[1]), dtype=float)
    H[steps] = _min_put_payoff(x[steps], strike)
    disc = np.exp(-float(r) * float(dt))
    for t in range(steps - 1, 0, -1):
        regressand = disc * H[t + 1]
        Phi = min_put_polynomial_basis(x[t], strike, degree, sort_state)
        continuation = Phi @ th[t]
        immediate = _min_put_payoff(x[t], strike)
        exercise = (immediate > 0.0) & (immediate >= continuation)
        H[t] = np.where(exercise, immediate, regressand)
    H[0] = disc * H[1]
    return H

import numpy as np


def single_projection_alpha_martingale(paths: "np.ndarray", H_D: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    H = np.asarray(H_D, dtype=float)
    steps = x.shape[0] - 1
    n_paths = x.shape[1]
    disc = np.exp(-float(r) * float(dt))
    bank = np.exp(float(r) * float(dt) * np.arange(steps + 1, dtype=float))
    M = np.zeros((steps + 1, n_paths), dtype=float)
    for t in range(steps):
        regressand = disc * H[t + 1]
        Phi_next = min_put_polynomial_basis(x[t + 1], strike, degree, sort_state)
        gamma = _ols(Phi_next, regressand)
        value_next = Phi_next @ gamma
        if t == 0:
            continuation = np.full(n_paths, np.mean(regressand), dtype=float)
        else:
            Phi_now = min_put_polynomial_basis(x[t], strike, degree, sort_state)
            alpha = _ols(Phi_now, regressand)
            continuation = Phi_now @ alpha
        M[t + 1] = M[t] + (value_next - continuation) / bank[t]
    return M

import numpy as np


def double_projection_beta_martingale(paths: "np.ndarray", H_D: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    x = np.asarray(paths, dtype=float)
    H = np.asarray(H_D, dtype=float)
    steps = x.shape[0] - 1
    n_paths = x.shape[1]
    disc = np.exp(-float(r) * float(dt))
    bank = np.exp(float(r) * float(dt) * np.arange(steps + 1, dtype=float))
    M = np.zeros((steps + 1, n_paths), dtype=float)
    for t in range(steps):
        regressand = disc * H[t + 1]
        Phi_next = min_put_polynomial_basis(x[t + 1], strike, degree, False)
        gamma = _ols(Phi_next, regressand)
        value_next = Phi_next @ gamma
        if t == 0:
            continuation = np.full(n_paths, np.mean(value_next), dtype=float)
        else:
            Phi_now = min_put_polynomial_basis(x[t], strike, degree, False)
            beta = _ols(Phi_now, value_next)
            continuation = Phi_now @ beta
        M[t + 1] = M[t] + (value_next - continuation) / bank[t]
    return M

import numpy as np


def _dual_upper_price(paths: "np.ndarray", martingale: "np.ndarray", strike: float, r: float, dt: float) -> float:
    x = np.asarray(paths, dtype=float)
    M = np.asarray(martingale, dtype=float)
    steps = x.shape[0] - 1
    bank = np.exp(float(r) * float(dt) * np.arange(steps + 1, dtype=float))
    discounted_immediate = np.stack([_min_put_payoff(x[t], strike) / bank[t] for t in range(steps + 1)])
    return float(np.mean(np.max(discounted_immediate[1:] - M[1:], axis=0)))


def unsorted_primal_dual_bounds(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    theta = fit_min_put_primal_policy(train_paths, strike, r, dt, degree, False)
    H = evaluate_min_put_payoff_process(eval_paths, theta, strike, r, dt, degree, False)
    M_alpha = single_projection_alpha_martingale(eval_paths, H, strike, r, dt, degree, False)
    M_beta = double_projection_beta_martingale(eval_paths, H, strike, r, dt, degree)
    lower = float(np.mean(H[0]))
    upper_alpha = _dual_upper_price(eval_paths, M_alpha, strike, r, dt)
    upper_beta = _dual_upper_price(eval_paths, M_beta, strike, r, dt)
    return np.array([lower, upper_alpha, upper_beta], dtype=float)

import numpy as np


def sorting_aware_bound_vector(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    unsorted = unsorted_primal_dual_bounds(train_paths, eval_paths, strike, r, dt, degree)
    theta_sorted = fit_min_put_primal_policy(train_paths, strike, r, dt, degree, True)
    H_sorted = evaluate_min_put_payoff_process(eval_paths, theta_sorted, strike, r, dt, degree, True)
    M_alpha_sorted = single_projection_alpha_martingale(eval_paths, H_sorted, strike, r, dt, degree, True)
    lower_sorted = float(np.mean(H_sorted[0]))
    upper_alpha_sorted = _dual_upper_price(eval_paths, M_alpha_sorted, strike, r, dt)
    return np.array([unsorted[0], unsorted[1], unsorted[2], lower_sorted, upper_alpha_sorted], dtype=float)

import numpy as np


def cumulative_sorting_basis_path(s0: "np.ndarray", strike: float, r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, train_seed: int, eval_seed: int, degrees: tuple) -> float:
    train = simulate_correlated_antithetic_gbm_paths(s0, r, q, sigma, rho, maturity, steps, n_paths, train_seed)
    eval_paths = simulate_correlated_antithetic_gbm_paths(s0, r, q, sigma, rho, maturity, steps, n_paths, eval_seed)
    dt = float(maturity) / int(steps)
    rows = []
    for degree in degrees:
        rows.append(sorting_aware_bound_vector(train, eval_paths, strike, r, dt, int(degree)))
    bounds = np.asarray(rows, dtype=float)
    lower_u = bounds[:, 0]
    lower_s = bounds[:, 3]
    gaps = np.column_stack((
        (bounds[:, 1] - lower_u) / lower_u,
        (bounds[:, 2] - lower_u) / lower_u,
        (bounds[:, 4] - lower_s) / lower_s,
        (lower_s - lower_u) / lower_u,
    ))
    return float(np.sum(np.linalg.norm(np.diff(gaps, axis=0), axis=1)))
SCICODE_GOLD_EOF
