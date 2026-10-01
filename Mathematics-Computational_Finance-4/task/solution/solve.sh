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


def simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_steps: int,
    n_paths: int,
    n_assets: int,
    seed: int,
) -> "np.ndarray":
    for name, value in (("n_steps", n_steps), ("n_paths", n_paths), ("n_assets", n_assets)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    spot, maturity = float(spot), float(maturity)
    rate, dividend_yield, volatility = float(rate), float(dividend_yield), float(volatility)
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive.")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not (np.isfinite(rate) and np.isfinite(dividend_yield)):
        raise ValueError("rate and dividend_yield must be finite.")
    if not np.isfinite(volatility) or volatility < 0.0:
        raise ValueError("volatility must be finite and non-negative.")

    dt = maturity / int(n_steps)
    z = np.random.default_rng(int(seed)).standard_normal((int(n_paths), int(n_steps), int(n_assets)))
    increments = (rate - dividend_yield - 0.5 * volatility * volatility) * dt + volatility * np.sqrt(dt) * z
    log_paths = np.concatenate((np.zeros((int(n_paths), 1, int(n_assets))), np.cumsum(increments, axis=1)), axis=1)
    return spot * np.exp(log_paths)

import numpy as np
from scipy import special


def _bivariate_normal_cdf(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho, via Owen's T function."""
    h, k = np.broadcast_arrays(np.atleast_1d(np.asarray(h, dtype=float)), np.atleast_1d(np.asarray(k, dtype=float)))
    s = np.sqrt(1.0 - rho * rho)
    out = np.empty(h.shape)
    both_zero = (h == 0.0) & (k == 0.0)
    out[both_zero] = 0.25 + np.arcsin(rho) / (2.0 * np.pi)
    m = ~both_zero
    hh, kk = h[m], k[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        ah = np.where(hh != 0.0, (kk - rho * hh) / (hh * s), np.copysign(np.inf, kk))
        ak = np.where(kk != 0.0, (hh - rho * kk) / (kk * s), np.copysign(np.inf, hh))
    beta = np.where((hh * kk < 0.0) | ((hh * kk == 0.0) & (hh + kk < 0.0)), 0.5, 0.0)
    out[m] = (0.5 * special.ndtr(hh) + 0.5 * special.ndtr(kk)
              - special.owens_t(hh, ah) - special.owens_t(kk, ak) - beta)
    return out


def price_two_asset_max_call(
    s1: "np.ndarray",
    s2: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
) -> "np.ndarray":
    a1 = np.asarray(s1, dtype=float)
    a2 = np.asarray(s2, dtype=float)
    try:
        a1, a2 = np.broadcast_arrays(a1, a2)
    except ValueError as exc:
        raise ValueError("s1 and s2 must broadcast together.") from exc
    if not (np.all(np.isfinite(a1)) and np.all(np.isfinite(a2))) or np.any(a1 <= 0.0) or np.any(a2 <= 0.0):
        raise ValueError("Asset prices must be finite and positive.")
    strike, tau, rate = float(strike), float(tau), float(rate)
    q1, q2, sigma1, sigma2, rho = float(q1), float(q2), float(sigma1), float(sigma2), float(rho)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative.")
    if not (np.isfinite(rate) and np.isfinite(q1) and np.isfinite(q2)):
        raise ValueError("rate and dividend yields must be finite.")
    if not (np.isfinite(sigma1) and np.isfinite(sigma2)) or sigma1 <= 0.0 or sigma2 <= 0.0:
        raise ValueError("volatilities must be finite and positive.")
    if not np.isfinite(rho) or rho <= -1.0 or rho >= 1.0:
        raise ValueError("rho must lie in (-1, 1).")

    if tau == 0.0:
        return np.maximum(np.maximum(a1, a2) - strike, 0.0)

    b1, b2 = rate - q1, rate - q2
    sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2.0 * rho * sigma1 * sigma2)
    root = np.sqrt(tau)
    d = (np.log(a1 / a2) + (b1 - b2 + 0.5 * sigma * sigma) * tau) / (sigma * root)
    y1 = (np.log(a1 / strike) + (b1 + 0.5 * sigma1 * sigma1) * tau) / (sigma1 * root)
    y2 = (np.log(a2 / strike) + (b2 + 0.5 * sigma2 * sigma2) * tau) / (sigma2 * root)
    rho1 = (sigma1 - rho * sigma2) / sigma
    rho2 = (sigma2 - rho * sigma1) / sigma
    shape = a1.shape
    price = (a1.ravel() * np.exp((b1 - rate) * tau) * _bivariate_normal_cdf(y1.ravel(), d.ravel(), rho1)
             + a2.ravel() * np.exp((b2 - rate) * tau) * _bivariate_normal_cdf(y2.ravel(), -d.ravel() + sigma * root, rho2)
             - strike * np.exp(-rate * tau)
             * (1.0 - _bivariate_normal_cdf(-y1.ravel() + sigma1 * root, -y2.ravel() + sigma2 * root, rho)))
    return price.reshape(shape)

import numpy as np


def build_max_call_payoffs_and_basis(
    paths: "np.ndarray",
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> "tuple[np.ndarray, np.ndarray]":
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 2 or x.shape[2] < 2:
        raise ValueError("paths must have shape (n_paths, n_times, n_assets) with n_times >= 2 and n_assets >= 2.")
    if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("paths must be finite and positive.")
    strike, maturity = float(strike), float(maturity)
    rate, dividend_yield, volatility = float(rate), float(dividend_yield), float(volatility)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not (np.isfinite(rate) and np.isfinite(dividend_yield)):
        raise ValueError("rate and dividend_yield must be finite.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive.")

    n_paths, n_times, _ = x.shape
    n_steps = n_times - 1
    ranked = np.sort(x, axis=2)
    top, second = ranked[:, :, -1], ranked[:, :, -2]
    payoffs = np.maximum(top - strike, 0.0)
    x1, x2 = top / strike, second / strike
    euro = np.empty((n_paths, n_times))
    for j in range(n_times):
        tau = maturity - j * maturity / n_steps if j < n_steps else 0.0
        euro[:, j] = price_two_asset_max_call(
            top[:, j], second[:, j], strike, max(tau, 0.0), rate, dividend_yield, dividend_yield,
            volatility, volatility, 0.0) / strike
    features = [np.ones_like(x1), x1, x1 ** 2, x1 ** 3, x1 ** 4, x2, x2 ** 2, x2 ** 3, x2 ** 4,
                x1 * x2, x1 ** 2 * x2, x1 * x2 ** 2, x1 ** 2 * x2 ** 2, euro, euro ** 2, euro ** 3]
    return payoffs, np.stack(features, axis=2)

import numpy as np


def fit_backward_primal(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(payoffs, dtype=float)
    b = np.asarray(basis, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 2:
        raise ValueError("payoffs must have shape (n_paths, n_times) with n_times >= 2.")
    if not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("payoffs must be finite and non-negative.")
    if b.ndim != 3 or b.shape[:2] != h.shape or b.shape[2] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("basis must be a finite array of shape (n_paths, n_times, p) with p >= 1.")
    maturity, rate = float(maturity), float(rate)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")

    n_paths, n_times, p = b.shape
    n_steps = n_times - 1
    one_step = np.exp(-rate * maturity / n_steps)
    coefficients = np.zeros((n_times, p))
    policy = np.zeros((n_paths, n_times), dtype=bool)
    values = np.zeros((n_paths, n_times))
    values[:, n_steps] = h[:, n_steps]
    policy[:, n_steps] = h[:, n_steps] > 0.0
    tail = h[:, n_steps].copy()
    for j in range(n_steps - 1, 0, -1):
        target = one_step * tail
        beta = np.linalg.lstsq(b[:, j, :], target, rcond=None)[0]
        coefficients[j] = beta
        continuation = b[:, j, :] @ beta
        exercise = (h[:, j] > 0.0) & (h[:, j] >= continuation)
        tail = np.where(exercise, h[:, j], target)
        values[:, j] = tail
        policy[:, j] = exercise
    values[:, 0] = one_step * values[:, 1]
    return coefficients, policy, values

import numpy as np


def apply_frozen_primal_policy(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    h = np.asarray(payoffs, dtype=float)
    b = np.asarray(basis, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 2:
        raise ValueError("payoffs must have shape (n_paths, n_times) with n_times >= 2.")
    if not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("payoffs must be finite and non-negative.")
    if b.ndim != 3 or b.shape[:2] != h.shape or b.shape[2] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("basis must be a finite array of shape (n_paths, n_times, p) with p >= 1.")
    if c.shape != (h.shape[1], b.shape[2]) or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be a finite array of shape (n_times, p).")
    maturity, rate = float(maturity), float(rate)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")

    n_paths, n_times = h.shape
    n_steps = n_times - 1
    dt = maturity / n_steps
    stops = np.full(n_paths, n_steps, dtype=np.int64)
    alive = np.ones(n_paths, dtype=bool)
    for j in range(1, n_steps):
        continuation = b[:, j, :] @ c[j]
        exercise = alive & (h[:, j] > 0.0) & (h[:, j] >= continuation)
        stops[exercise] = j
        alive[exercise] = False
    cashflows = h[np.arange(n_paths), stops] * np.exp(-rate * stops * dt)
    return stops, cashflows, float(np.mean(cashflows))

import numpy as np


def build_backward_discounted_process(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray]":
    h = np.asarray(payoffs, dtype=float)
    b = np.asarray(basis, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 2:
        raise ValueError("payoffs must have shape (n_paths, n_times) with n_times >= 2.")
    if not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("payoffs must be finite and non-negative.")
    if b.ndim != 3 or b.shape[:2] != h.shape or b.shape[2] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("basis must be a finite array of shape (n_paths, n_times, p) with p >= 1.")
    if c.shape != (h.shape[1], b.shape[2]) or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be a finite array of shape (n_times, p).")
    maturity, rate = float(maturity), float(rate)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")

    n_paths, n_times = h.shape
    n_steps = n_times - 1
    dt = maturity / n_steps
    one_step = np.exp(-rate * dt)
    values = np.empty((n_paths, n_times))
    values[:, n_steps] = h[:, n_steps]
    for j in range(n_steps - 1, 0, -1):
        continuation = b[:, j, :] @ c[j]
        exercise = (h[:, j] > 0.0) & (h[:, j] >= continuation)
        values[:, j] = np.where(exercise, h[:, j], one_step * values[:, j + 1])
    values[:, 0] = one_step * values[:, 1]
    discounted = values * np.exp(-rate * dt * np.arange(n_times))[None, :]
    return values, discounted

import numpy as np
from scipy import integrate, special


def _bvn_owen(h, k, rho):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho, via Owen's T function."""
    if h == 0.0 and k == 0.0:
        return 0.25 + np.arcsin(rho) / (2.0 * np.pi)
    s = np.sqrt(1.0 - rho * rho)
    ah = (k - rho * h) / (h * s) if h != 0.0 else np.copysign(np.inf, k)
    ak = (h - rho * k) / (k * s) if k != 0.0 else np.copysign(np.inf, h)
    beta = 0.5 if (h * k < 0.0 or (h * k == 0.0 and h + k < 0.0)) else 0.0
    return float(0.5 * special.ndtr(h) + 0.5 * special.ndtr(k)
                 - special.owens_t(h, ah) - special.owens_t(k, ak) - beta)


def _tvn(a, r):
    """P(X1 <= a1, X2 <= a2, X3 <= a3) for a standard trivariate normal with correlation matrix r.

    Conditions on X1 and integrates the exact bivariate distribution function of (X2, X3) given X1.
    """
    r12, r13, r23 = r[0, 1], r[0, 2], r[1, 2]
    s12, s13 = np.sqrt(1.0 - r12 * r12), np.sqrt(1.0 - r13 * r13)
    rc = (r23 - r12 * r13) / (s12 * s13)

    def _integrand(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi) * _bvn_owen((a[1] - r12 * x) / s12, (a[2] - r13 * x) / s13, rc)

    lower = min(-40.0, a[0] - 1.0)
    value, _ = integrate.quad(_integrand, lower, a[0], epsabs=1e-15, epsrel=1e-13, limit=500)
    return value


def price_three_asset_max_call(
    spots: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    dividend_yields: "np.ndarray",
    volatilities: "np.ndarray",
    correlation: "np.ndarray",
) -> "np.ndarray":
    s = np.asarray(spots, dtype=float)
    q = np.asarray(dividend_yields, dtype=float)
    v = np.asarray(volatilities, dtype=float)
    c = np.asarray(correlation, dtype=float)
    if s.ndim < 1 or s.shape[-1] != 3 or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("spots must be a finite, strictly positive array whose last axis has length 3.")
    strike, tau, rate = float(strike), float(tau), float(rate)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("dividend_yields must be a finite array of shape (3,).")
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("volatilities must be a finite, strictly positive array of shape (3,).")
    if (c.shape != (3, 3) or not np.all(np.isfinite(c)) or not np.allclose(c, c.T, rtol=0.0, atol=1e-12)
            or not np.allclose(np.diag(c), 1.0, rtol=0.0, atol=1e-12) or np.min(np.linalg.eigvalsh(c)) <= 1e-12):
        raise ValueError("correlation must be a finite, symmetric, positive definite (3, 3) matrix with unit diagonal.")

    flat = s.reshape(-1, 3)
    if tau == 0.0:
        return np.maximum(flat.max(axis=1) - strike, 0.0).reshape(s.shape[:-1])

    root = np.sqrt(tau)
    pair = np.sqrt(np.maximum(v[:, None] ** 2 + v[None, :] ** 2 - 2.0 * c * np.outer(v, v), 0.0))
    prices = np.empty(flat.shape[0])
    for n, x in enumerate(flat):
        total = 0.0
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            upper = np.array([
                (np.log(x[i] / strike) + (rate - q[i] + 0.5 * v[i] ** 2) * tau) / (v[i] * root),
                (np.log(x[i] / x[j]) + (q[j] - q[i] + 0.5 * pair[i, j] ** 2) * tau) / (pair[i, j] * root),
                (np.log(x[i] / x[k]) + (q[k] - q[i] + 0.5 * pair[i, k] ** 2) * tau) / (pair[i, k] * root),
            ])
            corr = np.eye(3)
            corr[0, 1] = corr[1, 0] = (v[i] - c[i, j] * v[j]) / pair[i, j]
            corr[0, 2] = corr[2, 0] = (v[i] - c[i, k] * v[k]) / pair[i, k]
            corr[1, 2] = corr[2, 1] = (v[i] ** 2 - c[i, j] * v[i] * v[j] - c[i, k] * v[i] * v[k]
                                       + c[j, k] * v[j] * v[k]) / (pair[i, j] * pair[i, k])
            total += x[i] * np.exp(-q[i] * tau) * _tvn(upper, corr)
        below = -(np.log(x / strike) + (rate - q - 0.5 * v ** 2) * tau) / (v * root)
        total -= strike * np.exp(-rate * tau) * (1.0 - _tvn(below, c))
        prices[n] = total
    return prices.reshape(s.shape[:-1])

import numpy as np


def _alpha_martingale(basis, backward_values, maturity, rate):
    """Single-projection alpha martingale in time-zero money, shape (n_paths, n_steps + 1)."""
    b = np.asarray(basis, dtype=float)
    v = np.asarray(backward_values, dtype=float)
    n_paths, n_times, _ = b.shape
    n_steps = n_times - 1
    dt = float(maturity) / n_steps
    increments = np.empty((n_paths, n_steps))
    for j in range(n_steps):
        y = np.exp(-float(rate) * (j + 1) * dt) * v[:, j + 1]
        later = b[:, j + 1, :] @ np.linalg.lstsq(b[:, j + 1, :], y, rcond=None)[0]
        if j == 0:
            earlier = np.mean(y)
        else:
            earlier = b[:, j, :] @ np.linalg.lstsq(b[:, j, :], y, rcond=None)[0]
        increments[:, j] = later - earlier
    return np.concatenate((np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)), axis=1)


def compute_dual_upper_bound(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_assets: int,
    n_steps: int,
    n_training_paths: int,
    training_seed: int,
    n_pricing_paths: int,
    pricing_seed: int,
) -> "tuple[np.ndarray, np.ndarray, float, float, float]":
    for name, value in (("spot", spot), ("strike", strike), ("maturity", maturity), ("volatility", volatility)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (("rate", rate), ("dividend_yield", dividend_yield)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if isinstance(n_assets, bool) or not isinstance(n_assets, (int, np.integer)) or n_assets not in (2, 3):
        raise ValueError("n_assets must be the integer 2 or 3.")
    for name, value in (("n_steps", n_steps), ("n_training_paths", n_training_paths), ("n_pricing_paths", n_pricing_paths)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("training_seed", training_seed), ("pricing_seed", pricing_seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    training_paths = simulate_gbm_paths(
        spot, maturity, rate, dividend_yield, volatility, n_steps, n_training_paths, n_assets, training_seed)
    training_payoffs, training_basis = build_max_call_payoffs_and_basis(
        training_paths, strike, maturity, rate, dividend_yield, volatility)
    coefficients, _, _ = fit_backward_primal(training_payoffs, training_basis, maturity, rate)

    pricing_paths = simulate_gbm_paths(
        spot, maturity, rate, dividend_yield, volatility, n_steps, n_pricing_paths, n_assets, pricing_seed)
    payoffs, basis = build_max_call_payoffs_and_basis(
        pricing_paths, strike, maturity, rate, dividend_yield, volatility)
    _, _, primal_lower_bound = apply_frozen_primal_policy(payoffs, basis, coefficients, maturity, rate)
    backward_values, _ = build_backward_discounted_process(payoffs, basis, coefficients, maturity, rate)
    martingale = _alpha_martingale(basis, backward_values, maturity, rate)

    discount = np.exp(-float(rate) * float(maturity) / int(n_steps) * np.arange(int(n_steps) + 1))
    discounted_payoffs = payoffs * discount[None, :]
    dual_path_values = np.max(discounted_payoffs[:, 1:] - martingale[:, 1:], axis=1)
    dual_upper_bound = float(np.mean(dual_path_values))
    if int(n_assets) == 2:
        european_price = float(price_two_asset_max_call(
            spot, spot, strike, maturity, rate, dividend_yield, dividend_yield, volatility, volatility, 0.0))
    else:
        european_price = float(price_three_asset_max_call(
            np.full(3, float(spot)), strike, maturity, rate, np.full(3, float(dividend_yield)),
            np.full(3, float(volatility)), np.eye(3)))
    return (discounted_payoffs, dual_path_values, dual_upper_bound,
            float(dual_upper_bound - primal_lower_bound), european_price)
SCICODE_GOLD_EOF
