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
 
def polynomial_test_functions(grid: "np.ndarray", left_endpoints: "np.ndarray", support_length: float, power: int) -> "np.ndarray":
    x = np.asarray(grid, dtype=float)
    x1 = np.asarray(left_endpoints, dtype=float)
    if x.ndim != 1 or x.size == 0 or x1.ndim != 1 or x1.size == 0:
        raise ValueError("grid and left_endpoints must be nonempty 1-D arrays")
    if not (np.isfinite(support_length) and support_length > 0):
        raise ValueError("support_length must be positive")
    if isinstance(power, bool) or int(power) != power or power < 2:
        raise ValueError("power must be an integer >= 2")
    p = int(power)
    ell = float(support_length)
    u = (x[None, :] - x1[:, None]) / ell
    inside = (u > 0.0) & (u < 1.0)
    q = np.where(inside, 4.0 * u * (1.0 - u), 0.0)
    phi = np.where(inside, q ** p, 0.0)
    dphi = np.where(inside, p * q ** (p - 1) * 4.0 * (1.0 - 2.0 * u) / ell, 0.0)
    return np.stack([phi, dphi])

import numpy as np
 
def _log_survival(a: "np.ndarray", wf: "np.ndarray", cf: "np.ndarray") -> "np.ndarray":
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    for w, c in zip(wf, cf):
        out += w * (a if c == 0.0 else np.expm1(c * a) / c)
    return out
 
def _birth_profile(a: "np.ndarray", wb: "np.ndarray", mb: "np.ndarray", s: float) -> "np.ndarray":
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    for w, mu in zip(wb, mb):
        out += w * np.exp(-(a - mu) ** 2 / (2.0 * s * s))
    return out
 
def _initial_profile(u: "np.ndarray", support: float) -> "np.ndarray":
    u = np.asarray(u, dtype=float)
    return np.where((u >= 0.0) & (u <= support), 1.0 - np.cos(2.0 * np.pi * u / support), 0.0)
 
def _renewal_trapezoid(k: float, n_nodes: int, support: float, wf, cf, wb, mb, s) -> "np.ndarray":
    tau = np.arange(n_nodes + 1) * k
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 0.5 * support * (x + 1.0)
    uw = 0.5 * support * gw
    arg = u[None, :] + tau[:, None]
    forcing = (_birth_profile(arg, wb, mb, s) * np.exp(_log_survival(arg, wf, cf) - _log_survival(u, wf, cf)[None, :]) * _initial_profile(u, support)[None, :]) @ uw
    kernel = _birth_profile(tau, wb, mb, s) * np.exp(_log_survival(tau, wf, cf))
    flux = np.empty(n_nodes + 1)
    flux[0] = forcing[0]
    denom = 1.0 - 0.5 * k * kernel[0]
    for n in range(1, n_nodes + 1):
        memory = np.dot(kernel[1:n], flux[n - 1:0:-1]) + 0.5 * kernel[n] * flux[0]
        flux[n] = (forcing[n] + k * memory) / denom
    return flux
 
def simulate_age_structured(initial_support: float, age_step: float, n_ages: int, n_steps: int, source_coefficients: "np.ndarray", source_rates: "np.ndarray", birth_coefficients: "np.ndarray", birth_means: "np.ndarray", birth_width: float) -> "np.ndarray":
    wf = np.asarray(source_coefficients, dtype=float).ravel()
    cf = np.asarray(source_rates, dtype=float).ravel()
    wb = np.asarray(birth_coefficients, dtype=float).ravel()
    mb = np.asarray(birth_means, dtype=float).ravel()
    if wf.size != cf.size or wb.size != mb.size:
        raise ValueError("coefficient and parameter arrays must have matching lengths")
    if not (age_step > 0 and birth_width > 0 and initial_support > 0):
        raise ValueError("age_step, birth_width and initial_support must be positive")
    if int(n_ages) != n_ages or n_ages < 1 or int(n_steps) != n_steps or n_steps < 0:
        raise ValueError("n_ages must be >= 1 and n_steps >= 0")
    h, L0 = float(age_step), float(initial_support)
    n_ages, n_steps = int(n_ages), int(n_steps)
    if L0 + n_steps * h > n_ages * h * (1.0 + 1e-12):
        raise ValueError("individuals would leave the age domain")
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    out = np.empty_like(T)
    old = A > T
    out[old] = _initial_profile(A[old] - T[old], L0) * np.exp(_log_survival(A[old], wf, cf) - _log_survival(A[old] - T[old], wf, cf))
    if (~old).any():
        # birth flux on a grid of step h/4 containing all t - a = (i - j - 1/2) h, with
        # Richardson extrapolation of the trapezoidal Volterra scheme (error ~ k^6)
        base = h / 4.0
        levels = []
        for level in range(3):
            k = base / 2 ** level
            n_nodes = int(round(n_steps * h / k))
            levels.append(_renewal_trapezoid(k, n_nodes, L0, wf, cf, wb, mb, birth_width)[::2 ** level])
        power = 2
        while len(levels) > 1:
            levels = [(2 ** power * levels[i + 1] - levels[i]) / (2 ** power - 1) for i in range(len(levels) - 1)]
            power += 2
        flux = levels[0]
        idx = np.rint((T[~old] - A[~old]) / base).astype(int)
        out[~old] = flux[idx] * np.exp(_log_survival(A[~old], wf, cf))
    return out

import numpy as np
from scipy.optimize import brentq
 
def _lognormal_sigma(noise_to_signal_ratio: float) -> float:
    r = float(noise_to_signal_ratio)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("noise_to_signal_ratio must be a finite nonnegative number")
    # E[(e^z - 1)^2] = e^{2 s^2} - 2 e^{s^2/2} + 1 = x^4 - 2x + 1 with x = e^{s^2/2} >= 1
    if r == 0.0:
        return 0.0
    upper = 2.0
    while upper ** 4 - 2.0 * upper + 1.0 < r:
        upper *= 2.0
    x = brentq(lambda y: y ** 4 - 2.0 * y + 1.0 - r, 1.0, upper, xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    return float(np.sqrt(2.0 * np.log(x)))
 
def add_lognormal_noise(clean_density: "np.ndarray", noise_to_signal_ratio: float, seed: int) -> "np.ndarray":
    sigma = _lognormal_sigma(noise_to_signal_ratio)
    clean = np.asarray(clean_density, dtype=float)
    z = np.random.default_rng(seed).standard_normal(clean.shape)
    return clean * np.exp(sigma * z)

import numpy as np
 
def _trapezoid_weights(n_points: int, step: float) -> "np.ndarray":
    w = np.full(n_points, float(step))
    w[0] = w[-1] = 0.5 * float(step)
    return w
 
def assemble_weak_system(noisy_density: "np.ndarray", time_step: float, age_step: float, sigma: float, aging_speed: float, source_rates: "np.ndarray", birth_means: "np.ndarray", birth_width: float, n_time_tests: int, n_age_tests: int, support_ratio_time: float, support_ratio_age: float, power: int) -> "np.ndarray":
    n = np.asarray(noisy_density, dtype=float)
    if n.ndim != 2 or n.shape[0] < 2 or n.shape[1] < 2:
        raise ValueError("noisy_density must be 2-D with at least 2 rows and 2 columns")
    if not (time_step > 0 and age_step > 0 and birth_width > 0):
        raise ValueError("steps and birth_width must be positive")
    if not sigma >= 0:
        raise ValueError("sigma must be nonnegative")
    if int(n_time_tests) != n_time_tests or int(n_age_tests) != n_age_tests or n_time_tests < 2 or n_age_tests < 2:
        raise ValueError("need at least two test functions in each direction")
    if not (0 < support_ratio_time <= 1 and 0 < support_ratio_age <= 1):
        raise ValueError("support ratios must lie in (0, 1]")
    rates = np.asarray(source_rates, dtype=float).ravel()
    means = np.asarray(birth_means, dtype=float).ravel()
    t = np.arange(n.shape[0]) * float(time_step)
    a = (np.arange(n.shape[1]) + 0.5) * float(age_step)
    a_end = n.shape[1] * float(age_step)
    ell_t = support_ratio_time * t[-1]
    ell_a = support_ratio_age * a_end
    t1 = np.arange(int(n_time_tests)) * (t[-1] - ell_t) / (int(n_time_tests) - 1)
    a1 = np.arange(int(n_age_tests)) * (a_end - ell_a) / (int(n_age_tests) - 1)
    tf = polynomial_test_functions(t, t1, ell_t, power)
    af = polynomial_test_functions(a, a1, ell_a, power)
    phi, dphi = tf[0], tf[1]
    psi, dpsi = af[0], af[1]
    wt = _trapezoid_weights(t.size, time_step)
    wa = np.full(a.size, float(age_step))
 
    def _inner(time_part, age_part, field):
        return (time_part @ (field * wt[:, None] * wa[None, :]) @ age_part.T).ravel()
 
    source_fields = [np.exp(c * a)[None, :] * n for c in rates]
    birth_profiles = [np.exp(-(a - mu) ** 2 / (2.0 * birth_width ** 2)) for mu in means]
    b_pde = -_inner(dphi, psi, n) - float(aging_speed) * _inner(phi, dpsi, n)
    G_pde = np.zeros((b_pde.size, rates.size + means.size))
    for m, field in enumerate(source_fields):
        G_pde[:, m] = _inner(phi, psi, field)
    debias = np.exp(0.5 * float(sigma) ** 2)
    nt = n / debias
    total = nt @ wa
    b_ode = -(dphi * wt[None, :]) @ total
    G_ode = np.zeros((phi.shape[0], rates.size + means.size))
    for m, c in enumerate(rates):
        G_ode[:, m] = (phi * wt[None, :]) @ (nt @ (wa * np.exp(c * a)))
    for m, prof in enumerate(birth_profiles):
        G_ode[:, rates.size + m] = (phi * wt[None, :]) @ (nt @ (wa * prof))
    G = np.vstack([G_pde, G_ode])
    b = np.concatenate([b_pde, b_ode])
    return np.column_stack([G, b])

import numpy as np
 
def _restricted_lstsq(G: "np.ndarray", b: "np.ndarray", support: "np.ndarray") -> "np.ndarray":
    w = np.zeros(G.shape[1])
    if support.any():
        w[support] = np.linalg.lstsq(G[:, support], b, rcond=None)[0]
    return w
 
def mstls_sparse_regression(G: "np.ndarray", b: "np.ndarray", lambdas: "np.ndarray") -> "np.ndarray":
    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    lams = np.asarray(lambdas, dtype=float).ravel()
    if G.ndim != 2 or G.shape[0] < 1 or G.shape[1] < 1 or b.size != G.shape[0]:
        raise ValueError("G must be (R, J) and b must have length R")
    if lams.size == 0 or np.any(~(lams > 0)):
        raise ValueError("lambdas must be a nonempty array of positive values")
    col = np.linalg.norm(G, axis=0)
    if np.any(col == 0):
        raise ValueError("G has a zero column")
    J = G.shape[1]
    w_ls = np.linalg.lstsq(G, b, rcond=None)[0]
    fit_ls = np.linalg.norm(G @ w_ls)
    if fit_ls == 0:
        raise ValueError("G w_LS vanishes; loss undefined")
    ratio = np.linalg.norm(b) / col
    best = None
    for lam in lams:
        lower = lam * np.maximum(1.0, ratio)
        upper = np.minimum(1.0, ratio) / lam
        w = w_ls.copy()
        prev = np.ones(J, dtype=bool)
        while True:
            cur = (np.abs(w) >= lower) & (np.abs(w) <= upper)
            if np.array_equal(cur, prev):
                break
            w = _restricted_lstsq(G, b, cur)
            prev = cur
        loss = np.linalg.norm(G @ (w - w_ls)) / fit_ls + np.count_nonzero(prev) / J
        if best is None or loss < best[0] or (loss == best[0] and lam < best[1]):
            best = (loss, float(lam), w)
    return np.append(best[2], best[1])

import numpy as np
 
def boundary_bagging_regression(G: "np.ndarray", b: "np.ndarray", n_pde_rows: int, n_source_terms: int, lambdas: "np.ndarray") -> "np.ndarray":
    G = np.asarray(G, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    if G.ndim != 2 or b.size != G.shape[0]:
        raise ValueError("G must be (R, J) and b must have length R")
    R, J = G.shape
    p, mf = int(n_pde_rows), int(n_source_terms)
    if not (0 < p < R and 0 < mf < J):
        raise ValueError("block sizes inconsistent with G")
    w = mstls_sparse_regression(G, b, lambdas)[:-1]
    w_f, w_beta = w[:mf], w[mf:]
    xi_f, xi_beta, b_ode = G[p:, :mf], G[p:, mf:], b[p:]
    w_hat = mstls_sparse_regression(xi_beta, b_ode - xi_f @ w_f, lambdas)[:-1]
    s_joint, s_hat = w_beta != 0, w_hat != 0
    if not np.array_equal(s_joint, s_hat):
        keep = s_joint & s_hat
        if not keep.any():
            keep = s_joint | s_hat
        cols = np.concatenate([np.ones(mf, dtype=bool), keep])
        w_sub = mstls_sparse_regression(G[:, cols], b, lambdas)[:-1]
        out = np.zeros(J)
        out[cols] = w_sub
        return out
    cand_joint = np.concatenate([w_f, w_beta])
    cand_hat = np.concatenate([w_f, w_hat])
    nb = np.linalg.norm(b)
    if np.linalg.norm(b - G @ cand_hat) / nb < np.linalg.norm(b - G @ cand_joint) / nb:
        return cand_hat
    return cand_joint

import numpy as np
 
def prediction_error(learned_weights: "np.ndarray", initial_support: float, age_step: float, n_train_steps: int, source_rates: "np.ndarray", birth_means: "np.ndarray", birth_width: float, true_density: "np.ndarray") -> float:
    w = np.asarray(learned_weights, dtype=float).ravel()
    rates = np.asarray(source_rates, dtype=float).ravel()
    means = np.asarray(birth_means, dtype=float).ravel()
    truth = np.asarray(true_density, dtype=float)
    if w.size != rates.size + means.size or truth.ndim != 2:
        raise ValueError("inconsistent shapes")
    n_rows, n_ages = truth.shape
    k = int(n_train_steps)
    if k != n_train_steps or not (0 <= k < n_rows - 1):
        raise ValueError("n_train_steps out of range")
    pred = simulate_age_structured(initial_support, age_step, n_ages, n_rows - 1, w[:rates.size], rates, w[rates.size:], means, birth_width)
    wt = _trapezoid_weights(n_rows - k, age_step)
    weight = wt[:, None] * float(age_step)
    denom = np.sum(weight * truth[k:] ** 2)
    if not denom > 0:
        raise ValueError("true density vanishes on the testing window")
    return float(np.sqrt(np.sum(weight * (pred[k:] - truth[k:]) ** 2) / denom))

import numpy as np
 
def wsindy_prediction_pipeline(noise_to_signal_ratio: float, seed: int, age_step: float, n_time_tests: int, n_age_tests: int) -> float:
    h = float(age_step)
    if not h > 0:
        raise ValueError("age_step must be positive")
    n_age, n_train = 25.0 / h, 5.0 / h
    if abs(n_age - round(n_age)) > 1e-9 or abs(n_train - round(n_train)) > 1e-9:
        raise ValueError("age_step must divide 25 and 5")
    n_age, n_train = int(round(n_age)), int(round(n_train))
    n_total = 2 * n_train
    rates = np.array([0.08, 0.40, 0.72, 1.04, 1.36])
    means = np.array([5.0, 10.0, 15.0])
    width = 5.0
    truth = simulate_age_structured(15.0, h, n_age, n_total, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([10.0]), width)
    sigma = _lognormal_sigma(noise_to_signal_ratio)
    noisy = add_lognormal_noise(truth[:n_train + 1], noise_to_signal_ratio, seed)
    system = assemble_weak_system(noisy, h, h, sigma, 1.0, rates, means, width, n_time_tests, n_age_tests, 0.5, 0.5, 14)
    G, b = system[:, :-1], system[:, -1]
    lambdas = 10.0 ** (-4.0 + 4.0 * np.arange(50) / 49.0)
    w = boundary_bagging_regression(G, b, int(n_time_tests) * int(n_age_tests), rates.size, lambdas)
    return prediction_error(w, 15.0, h, n_train, rates, means, width, truth)
SCICODE_GOLD_EOF
