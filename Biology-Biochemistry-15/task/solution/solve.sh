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


def _check_kin(kin):
    k = np.asarray(kin, dtype=np.float64)
    if k.shape != (2, 2, 4) or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("kin must be a (2, 2, 4) array of finite positive constants")
    return k


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def reduced_transition_rates(kin: "np.ndarray", conc: "np.ndarray", ppi: float,
                                     kp_const: float) -> "np.ndarray":
    """Eqs. (6)-(12): transition rates of the reduced two-state scheme.

    Returns rates[m, n, nn, :] = (w_EF[m,n], w_FE[m,n], w_pol[m,n], w_depol[m,n,nn]) where nn is the
    template unit that FOLLOWS the pair m:n. Only the depolymerization rate depends on nn, through the
    Michaelis-Menten denominator Q_nn of the next quasi-equilibrated binding step (eq. 9); w_EF carries
    the competitive denominator Q_n of its own template unit (eq. 6); k_depol = k_pol / K_P (eq. 12).
    """
    k = _check_kin(kin)
    c = np.asarray(conc, dtype=np.float64).ravel()
    if c.shape != (2,) or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conc must be two finite positive concentrations (dATP, dTTP)")
    ppi = _check_positive(ppi, "ppi")
    kp_const = _check_positive(kp_const, "kp_const")
    K = k[:, :, 0]
    # Q_n = 1 + sum_m [mP] / K_mn: competitive binding of both nucleotides at template unit n
    Q = 1.0 + np.array([np.sum(c / K[:, n]) for n in range(2)])
    rates = np.zeros((2, 2, 2, 4), dtype=np.float64)
    for m in range(2):
        for n in range(2):
            w_ef = k[m, n, 1] * c[m] / (K[m, n] * Q[n])
            w_fe = k[m, n, 2]
            w_pol = k[m, n, 3]
            for nn in range(2):
                w_dep = (k[m, n, 3] / kp_const) * ppi / Q[nn]
                rates[m, n, nn] = (w_ef, w_fe, w_pol, w_dep)
    return rates

import numpy as np


def backward_map_parameters(rates: "np.ndarray") -> "np.ndarray":
    """Eqs. (30)-(31): alpha[m,n] and beta[m,n,nn] of the scalar iterated function system.

    Returns params[m, n, nn, :] = (alpha, beta) with alpha = w_pol w_EF / (w_pol + w_FE) (effective
    rate of a completed incorporation) and beta = w_depol w_FE / (w_pol + w_FE) (effective rate of a
    completed removal); alpha does not depend on nn.
    """
    r = np.asarray(rates, dtype=np.float64)
    if r.shape != (2, 2, 2, 4) or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("rates must be a (2, 2, 2, 4) array of finite nonnegative rates")
    w_ef, w_fe, w_pol, w_dep = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    if np.any(w_pol + w_fe <= 0.0):
        raise ValueError("w_pol + w_FE must be positive for every pair")
    params = np.zeros((2, 2, 2, 2), dtype=np.float64)
    params[..., 0] = w_pol * w_ef / (w_pol + w_fe)
    params[..., 1] = w_dep * w_fe / (w_pol + w_fe)
    return params

import numpy as np


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def backward_iteration(params: "np.ndarray", template: "np.ndarray") -> "np.ndarray":
    """Eq. (29) run backward around the periodic template until the fixed cycle converges.

    x[l] for l = 0..L with x[0] == x[L] (periodic closure, eq. 39 regime); site l (1-based) has
    template unit template[l-1] and next unit template[l % L]. The map is
    x_{l-1} = x_l * sum_m alpha[m, n_l] / (x_l + beta[m, n_l, n_{l+1}]). Started from x_L = 1 and
    iterated loop after loop; converged when the whole cycle moves by less than 1e-13 relatively
    (at most 100000 loops).
    """
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (2, 2, 2, 2) or not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("params must be a (2, 2, 2, 2) array of finite nonnegative values")
    t = _check_template(template)
    max_loops, rtol = 100000, 1e-13          # fixed convergence settings of the method
    L = t.size
    alpha, beta = p[..., 0], p[..., 1]
    x = np.zeros(L + 1, dtype=np.float64)
    x[L] = 1.0
    for _ in range(int(max_loops)):
        prev = x.copy()
        for l in range(L, 0, -1):
            n, nn = t[l - 1], t[l % L]
            xl = x[l]
            x[l - 1] = xl * np.sum(alpha[:, n, nn] / (xl + beta[:, n, nn]))
        x[L] = x[0]
        if np.all(np.abs(x - prev) <= rtol * np.maximum(np.abs(x), 1e-300)):
            break
    return x

import numpy as np


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def site_transfer_factors(x: "np.ndarray", params: "np.ndarray", rates: "np.ndarray",
                                  template: "np.ndarray") -> "np.ndarray":
    """Eq. (33): the entries Y^E_{m,l} and Y^F_{m,l} of the reduced 2x2 transfer matrices.

    Returns Y[l-1, m, s] for site l = 1..L, copy unit m and structural state s (0 = E, 1 = F):
    Y^E = alpha / (x_l + beta), Y^F = (alpha / w_pol) (x_l + w_depol) / (x_l + beta), with beta and
    w_depol taken at the doublet (n_l, n_{l+1}). Column sums R^E_l = sum_m Y^E and R^F_l = sum_m Y^F
    are the forward-iteration factors of eqs. (35)-(36).
    """
    t = _check_template(template)
    L = t.size
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.shape != (L + 1,) or not np.all(np.isfinite(xx)) or np.any(xx < 0.0):
        raise ValueError("x must have L + 1 finite nonnegative entries")
    p = np.asarray(params, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if p.shape != (2, 2, 2, 2) or r.shape != (2, 2, 2, 4):
        raise ValueError("params must be (2, 2, 2, 2) and rates (2, 2, 2, 4)")
    Y = np.zeros((L, 2, 2), dtype=np.float64)
    for l in range(1, L + 1):
        n, nn = t[l - 1], t[l % L]
        for m in range(2):
            alpha, beta = p[m, n, nn]
            w_pol, w_dep = r[m, n, nn, 2], r[m, n, nn, 3]
            den = xx[l] + beta
            if den <= 0.0:
                raise ValueError("x_l + beta must be positive")
            Y[l - 1, m, 0] = alpha / den
            Y[l - 1, m, 1] = (alpha / w_pol) * (xx[l] + w_dep) / den
    return Y

import numpy as np


def mean_growth_velocity(x: "np.ndarray", Y: "np.ndarray") -> float:
    """Eqs. (40)-(41): v = 1 / < tau_l >, tau_l = (1 / x_l) (1 + R^F_l / R^E_l) averaged over the period."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    YY = np.asarray(Y, dtype=np.float64)
    L = YY.shape[0] if YY.ndim == 3 else 0
    if YY.shape != (L, 2, 2) or L < 1 or xx.shape != (L + 1,):
        raise ValueError("Y must be (L, 2, 2) and x must have L + 1 entries")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(YY)) or np.any(xx[1:] <= 0.0):
        raise ValueError("x and Y must be finite with positive x_l")
    RE = YY[:, :, 0].sum(axis=1)
    RF = YY[:, :, 1].sum(axis=1)
    if np.any(RE <= 0.0):
        raise ValueError("R^E_l must be positive")
    tau = (1.0 / xx[1:]) * (1.0 + RF / RE)
    return float(1.0 / np.mean(tau))

import numpy as np


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def error_probability(Y: "np.ndarray", template: "np.ndarray") -> float:
    """Eqs. (43)-(44): eta = < Y^E_{n_l, l} / R^E_l >, the mean fraction of incorrect pairs (m = n)."""
    t = _check_template(template)
    YY = np.asarray(Y, dtype=np.float64)
    if YY.shape != (t.size, 2, 2) or not np.all(np.isfinite(YY)) or np.any(YY < 0.0):
        raise ValueError("Y must be a finite nonnegative (L, 2, 2) array matching the template")
    RE = YY[:, :, 0].sum(axis=1)
    if np.any(RE <= 0.0):
        raise ValueError("R^E_l must be positive")
    mu_wrong = YY[np.arange(t.size), t, 0] / RE
    return float(np.mean(mu_wrong))

import numpy as np


def _check_kin(kin):
    k = np.asarray(kin, dtype=np.float64)
    if k.shape != (2, 2, 4) or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("kin must be a (2, 2, 4) array of finite positive constants")
    return k


def _check_conc(conc):
    c = np.asarray(conc, dtype=np.float64).ravel()
    if c.shape != (2,) or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conc must be two finite positive concentrations (dATP, dTTP)")
    return c


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _mean_log_backward_forward_ratio(kin, conc, ppi, kp_const, template, scale):
    """Appendix C eqs. (112)-(113) evaluated in the stationary limit x_l = 0 (the onset itself):
    < ln(b_l / a_l) > over the period at both concentrations multiplied by `scale`."""
    rates = reduced_transition_rates(kin, np.asarray(conc, dtype=np.float64) * scale, ppi, kp_const)
    params = backward_map_parameters(rates)
    t = np.asarray(template, dtype=np.int64)
    L = t.size
    Y = site_transfer_factors(np.zeros(L + 1), params, rates, t)
    RE = Y[:, :, 0].sum(axis=1)
    RF = Y[:, :, 1].sum(axis=1)
    logs = np.zeros(L)
    for l in range(1, L + 1):
        n = t[l - 1]
        prev = l - 2 if l >= 2 else L - 1          # site l-1; periodic closure: site 0 == site L
        a = np.sum(rates[:, n, 0, 0]) / (1.0 + RF[prev] / RE[prev])
        b = np.sum(Y[l - 1, :, 1] * rates[:, n, 0, 1]) / (RE[l - 1] + RF[l - 1])
        logs[l - 1] = np.log(b / a)
    return float(np.mean(logs))


def onset_of_growth_scale(kin: "np.ndarray", conc: "np.ndarray", ppi: float, kp_const: float,
                                  template: "np.ndarray") -> float:
    """Onset of steady growth (Appendix C, gamma -> 0): the factor s applied to both nucleotide
    concentrations at which < ln(b_l / a_l) > = 0 over the periodic template. The sign change is
    bracketed by scanning s over decades from s = 1 (downward if the given concentrations grow,
    upward if they stall) and located by bisection on log10(s) to 1e-12."""
    k = _check_kin(kin)
    c = _check_conc(conc)
    ppi = _check_positive(ppi, "ppi")
    kp_const = _check_positive(kp_const, "kp_const")
    t = _check_template(template)
    tol = 1e-12                               # fixed bisection tolerance on log10(s) of the method
    f = lambda ls: _mean_log_backward_forward_ratio(k, c, ppi, kp_const, t, 10.0 ** ls)
    f1 = f(0.0)
    step = -1.0 if f1 < 0.0 else 1.0                # growing at s = 1: onset lies below; else above
    lo = 0.0
    for _ in range(30):
        hi = lo + step
        if f(hi) * f1 < 0.0:
            break
        lo = hi
    else:
        raise ValueError("no onset of growth found within 30 decades of the given concentrations")
    lo, hi = (min(lo, hi), max(lo, hi))
    f_lo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) * f_lo > 0.0:
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return float(10.0 ** (0.5 * (lo + hi)))

import numpy as np


def _check_kin(kin):
    k = np.asarray(kin, dtype=np.float64)
    if k.shape != (2, 2, 4) or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("kin must be a (2, 2, 4) array of finite positive constants")
    return k


def _check_conc(conc):
    c = np.asarray(conc, dtype=np.float64).ravel()
    if c.shape != (2,) or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conc must be two finite positive concentrations (dATP, dTTP)")
    return c


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def steady_growth_velocity(kin: "np.ndarray", conc: "np.ndarray", ppi: float, kp_const: float,
                                   template: "np.ndarray") -> float:
    """End to end: reduced rates -> IFS parameters -> backward iteration -> transfer factors -> v.
    Returns the mean growth velocity in nucleotides per second; 0.0 if the configuration lies at or
    below the onset of growth (onset scale factor >= 1), where the exact solution has x_l -> 0."""
    k = _check_kin(kin)
    c = _check_conc(conc)
    ppi = _check_positive(ppi, "ppi")
    kp_const = _check_positive(kp_const, "kp_const")
    t = _check_template(template)
    if onset_of_growth_scale(k, c, ppi, kp_const, t) >= 1.0:
        return 0.0
    rates = reduced_transition_rates(k, c, ppi, kp_const)
    params = backward_map_parameters(rates)
    x = backward_iteration(params, t)
    Y = site_transfer_factors(x, params, rates, t)
    eta = error_probability(Y, t)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("error probability outside [0, 1]: inconsistent transfer factors")
    return mean_growth_velocity(x, Y)
SCICODE_GOLD_EOF
