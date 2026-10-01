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


def smooth_put_payoff(s: "np.ndarray", strike: float, eps: float) -> "np.ndarray":
    s = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("s must be finite.")
    strike, eps = float(strike), float(eps)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    x = strike - s
    poly = (35.0 / 256.0 * eps + 0.5 * x + 35.0 / (64.0 * eps) * x ** 2 - 35.0 / (128.0 * eps ** 3) * x ** 4
            + 7.0 / (64.0 * eps ** 5) * x ** 6 - 5.0 / (256.0 * eps ** 7) * x ** 8)
    return np.where(x <= -eps, 0.0, np.where(x >= eps, x, poly))

import numpy as np


def exact_difference_weights(n_intervals: int, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or n_intervals < 2:
        raise ValueError("n_intervals must be an integer of at least 2.")
    rate, volatility = float(rate), float(volatility)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be finite and positive.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive.")
    alpha = rate / (0.5 * volatility ** 2)
    m = np.arange(1, int(n_intervals), dtype=float)
    a1 = (m + 1.0) / m ** alpha - m / (m + 1.0) ** alpha
    a2 = (m + 2.0) / m ** alpha - m / (m + 2.0) ** alpha
    a3 = (m + 2.0) / (m + 1.0) ** alpha - (m + 1.0) / (m + 2.0) ** alpha
    return a1, a2, a3

import numpy as np


def nsfd_coefficients(n_intervals: int, time_step: float, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    time_step = float(time_step)
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive.")
    a1, a2, a3 = exact_difference_weights(n_intervals, rate, volatility)
    psi1 = float(np.expm1(float(rate) * time_step))
    q = a2 - a1 - a3
    upper = a1 / q
    centre = -a2 / q - 1.0 / psi1
    lower = (a2 - a1) / q - 1.0
    return lower, centre, upper, psi1

import numpy as np
from scipy.special import ndtr


def _bs_put(s, strike, rate, volatility, tau):
    price = np.full(s.shape, strike * np.exp(-rate * tau))
    pos = s > 0.0
    root = volatility * np.sqrt(tau)
    d1 = (np.log(s[pos] / strike) + (rate + 0.5 * volatility ** 2) * tau) / root
    price[pos] = strike * np.exp(-rate * tau) * ndtr(-(d1 - root)) - s[pos] * ndtr(-d1)
    return price


def smoothed_put_exact(s: "np.ndarray", strike: float, rate: float, volatility: float, tau: float, eps: float) -> "np.ndarray":
    s = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("s must be finite and non-negative.")
    strike, rate, volatility, tau, eps = float(strike), float(rate), float(volatility), float(tau), float(eps)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative.")
    if not np.isfinite(eps) or eps <= 0.0 or eps >= strike:
        raise ValueError("eps must be finite with 0 < eps < strike.")
    if tau == 0.0:
        return smooth_put_payoff(s, strike, eps)
    price = _bs_put(s, strike, rate, volatility, tau)
    pos = s > 0.0
    # Composite Gauss-Legendre in x = K - S on [-eps, 0] and [0, eps]; the kink of the integrand is a panel edge,
    # and the panels are narrow compared with the lognormal spread at the band.
    root = volatility * np.sqrt(tau)
    per_half = int(min(4096, max(8, np.ceil(4.0 * eps / ((strike - eps) * root)))))
    edges = np.linspace(-eps, eps, 2 * per_half + 1)
    xg, wg = np.polynomial.legendre.leggauss(16)
    half = 0.5 * np.diff(edges)
    x = (half[:, None] * xg[None, :] + 0.5 * (edges[:-1] + edges[1:])[:, None]).ravel()
    w = (half[:, None] * wg[None, :]).ravel()
    even = (35.0 / 256.0 * eps + 35.0 / (64.0 * eps) * x ** 2 - 35.0 / (128.0 * eps ** 3) * x ** 4
            + 7.0 / (64.0 * eps ** 5) * x ** 6 - 5.0 / (256.0 * eps ** 7) * x ** 8)
    bump = even - 0.5 * np.abs(x)
    big_s = strike - x
    sp = s[pos].reshape(-1, 1)
    z = (np.log(big_s[None, :] / sp) - (rate - 0.5 * volatility ** 2) * tau) / root
    density = np.exp(-0.5 * z ** 2) / (np.sqrt(2.0 * np.pi) * root * big_s[None, :])
    price[pos] += np.exp(-rate * tau) * (density * (bump * w)[None, :]).sum(axis=1)
    return price

import numpy as np
from scipy.linalg import solve_banded


def nsfd_time_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, left_value: float, right_value: float) -> "np.ndarray":
    v_prev = np.asarray(v_prev, dtype=float)
    if v_prev.ndim != 1 or v_prev.size < 3 or not np.all(np.isfinite(v_prev)):
        raise ValueError("v_prev must be a finite 1-D array with at least 3 entries.")
    n = v_prev.size - 2
    lower, centre, upper = (np.asarray(a, dtype=float) for a in (lower, centre, upper))
    if lower.shape != (n,) or centre.shape != (n,) or upper.shape != (n,):
        raise ValueError("coefficient arrays must have shape (M-1,).")
    psi1, left_value, right_value = float(psi1), float(left_value), float(right_value)
    if not np.isfinite(psi1) or psi1 <= 0.0:
        raise ValueError("psi1 must be finite and positive.")
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        raise ValueError("boundary values must be finite.")
    rhs = -v_prev[1:-1] / psi1
    rhs[0] -= lower[0] * left_value
    rhs[-1] -= upper[-1] * right_value
    bands = np.zeros((3, n))
    bands[0, 1:] = upper[:-1]
    bands[1] = centre
    bands[2, :-1] = lower[1:]
    interior = solve_banded((1, 1), bands, rhs)
    return np.concatenate(([left_value], interior, [right_value]))

import numpy as np


def solve_nsfd_put(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int, n_steps: int) -> "np.ndarray":
    strike, maturity, s_max, eps = float(strike), float(maturity), float(s_max), float(eps)
    if not all(np.isfinite([strike, maturity, s_max, eps])):
        raise ValueError("strike, maturity, s_max and eps must be finite.")
    if strike <= 0.0 or maturity <= 0.0 or eps <= 0.0 or eps >= strike or strike + eps >= s_max:
        raise ValueError("need strike, maturity > 0, 0 < eps < strike and strike + eps < s_max.")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    dt = maturity / int(n_steps)
    lower, centre, upper, psi1 = nsfd_coefficients(n_intervals, dt, rate, volatility)
    s = np.linspace(0.0, s_max, int(n_intervals) + 1)
    grid = np.empty((int(n_steps) + 1, s.size))
    grid[0] = smooth_put_payoff(s, strike, eps)
    for k in range(1, int(n_steps) + 1):
        left = strike * np.exp(-float(rate) * k * dt)
        grid[k] = nsfd_time_step(grid[k - 1], lower, centre, upper, psi1, left, 0.0)
    return grid

import numpy as np


def max_nodal_error(grid: "np.ndarray", strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float) -> float:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a finite 2-D array with at least two rows and columns.")
    maturity, s_max = float(maturity), float(s_max)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(s_max) or s_max <= 0.0:
        raise ValueError("s_max must be finite and positive.")
    n_steps, n_intervals = grid.shape[0] - 1, grid.shape[1] - 1
    s = np.linspace(0.0, s_max, n_intervals + 1)
    error = 0.0
    for k in range(n_steps + 1):
        exact = smoothed_put_exact(s, strike, rate, volatility, k * maturity / n_steps, eps)
        error = max(error, float(np.max(np.abs(grid[k] - exact))))
    return error

import numpy as np
from scipy.linalg import solve_banded


def nsfd_american_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, obstacle: "np.ndarray", left_value: float, right_value: float) -> "np.ndarray":
    v_prev = np.asarray(v_prev, dtype=float)
    if v_prev.ndim != 1 or v_prev.size < 3 or not np.all(np.isfinite(v_prev)):
        raise ValueError("v_prev must be a finite 1-D array with at least 3 entries.")
    n = v_prev.size - 2
    lower, centre, upper = (np.asarray(a, dtype=float) for a in (lower, centre, upper))
    if lower.shape != (n,) or centre.shape != (n,) or upper.shape != (n,):
        raise ValueError("coefficient arrays must have shape (M-1,).")
    obstacle = np.asarray(obstacle, dtype=float)
    if obstacle.shape != v_prev.shape or not np.all(np.isfinite(obstacle)):
        raise ValueError("obstacle must be a finite array of shape (M+1,).")
    psi1, left_value, right_value = float(psi1), float(left_value), float(right_value)
    if not np.isfinite(psi1) or psi1 <= 0.0:
        raise ValueError("psi1 must be finite and positive.")
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        raise ValueError("boundary values must be finite.")
    b = v_prev[1:-1] / psi1
    b[0] += lower[0] * left_value
    b[-1] += upper[-1] * right_value
    g = obstacle[1:-1]
    a_lo, a_ce, a_up = -lower, -centre, -upper
    exercise = np.zeros(n, dtype=bool)
    # Policy iteration on min(A v - b, v - g) = 0; for an M-matrix it terminates in at most n + 1 solves.
    for _ in range(n + 2):
        bands = np.zeros((3, n))
        bands[0, 1:] = np.where(exercise[:-1], 0.0, a_up[:-1])
        bands[1] = np.where(exercise, 1.0, a_ce)
        bands[2, :-1] = np.where(exercise[1:], 0.0, a_lo[1:])
        v = solve_banded((1, 1), bands, np.where(exercise, g, b))
        residual = a_ce * v - b
        residual[1:] += a_lo[1:] * v[:-1]
        residual[:-1] += a_up[:-1] * v[1:]
        new_exercise = residual > v - g
        if np.array_equal(new_exercise, exercise):
            break
        exercise = new_exercise
    else:
        raise RuntimeError("policy iteration did not terminate.")
    return np.concatenate(([left_value], v, [right_value]))

import numpy as np


def solve_nsfd_american_put(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int, n_steps: int) -> "np.ndarray":
    strike, maturity, s_max, eps = float(strike), float(maturity), float(s_max), float(eps)
    if not all(np.isfinite([strike, maturity, s_max, eps])):
        raise ValueError("strike, maturity, s_max and eps must be finite.")
    if strike <= 0.0 or maturity <= 0.0 or eps <= 0.0 or eps >= strike or strike + eps >= s_max:
        raise ValueError("need strike, maturity > 0, 0 < eps < strike and strike + eps < s_max.")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    dt = maturity / int(n_steps)
    lower, centre, upper, psi1 = nsfd_coefficients(n_intervals, dt, rate, volatility)
    s = np.linspace(0.0, s_max, int(n_intervals) + 1)
    payoff = smooth_put_payoff(s, strike, eps)
    grid = np.empty((int(n_steps) + 1, s.size))
    grid[0] = payoff
    for k in range(1, int(n_steps) + 1):
        grid[k] = nsfd_american_step(grid[k - 1], lower, centre, upper, psi1, payoff, strike, 0.0)
    return grid

import numpy as np


def nsfd_american_put_study(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int) -> "np.ndarray":
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or n_intervals < 2:
        raise ValueError("n_intervals must be an integer of at least 2.")
    m = int(n_intervals)
    european_grid = solve_nsfd_put(strike, rate, volatility, maturity, s_max, eps, m, m)
    error = max_nodal_error(european_grid, strike, rate, volatility, maturity, s_max, eps)
    american_grid = solve_nsfd_american_put(strike, rate, volatility, maturity, s_max, eps, m, m)
    s = np.linspace(0.0, float(s_max), m + 1)
    european = float(np.interp(float(strike), s, european_grid[-1]))
    american = float(np.interp(float(strike), s, american_grid[-1]))
    return np.array([error, european, american, american - european])
SCICODE_GOLD_EOF
