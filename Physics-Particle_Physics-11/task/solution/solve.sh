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


def profile_background_estimate(n: float, m: float, tau: float, s: float) -> float:
    n = float(n)
    m = float(m)
    tau = float(tau)
    s = float(s)
    if not (math.isfinite(n) and n >= 0.0):
        raise ValueError("n must be a finite count of at least zero")
    if not (math.isfinite(m) and m >= 0.0):
        raise ValueError("m must be a finite count of at least zero")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError("s must be a finite signal strength of at least zero")

    linear = n + m - (1.0 + tau) * s
    # The discriminant is a square plus a non-negative term, so the root is real.
    root = math.sqrt(linear * linear + 4.0 * (1.0 + tau) * s * m)
    if linear >= 0.0:
        return float((linear + root) / (2.0 * (1.0 + tau)))
    # For a negative linear term the sum would cancel, so use the conjugate form,
    # which is the same root written as 2 s m / (root - linear).
    return float(2.0 * s * m / (root - linear))

import math

import numpy as np


def signed_likelihood_root(n: "np.ndarray", m: "np.ndarray", tau: float) -> "np.ndarray":
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")
    for name, counts in (("n", n), ("m", m)):
        if not np.all(np.isfinite(counts)):
            raise ValueError(f"{name} must be finite")
        if np.any(counts < 0.0):
            raise ValueError(f"{name} must be at least zero")

    n, m = np.broadcast_arrays(n, m)
    # Each term is x ln(ratio) with the continuous limit 0 ln(0) = 0. The ratios
    # are written as 1 + delta and evaluated with log1p, because at large counts
    # the rounding of a ratio near one would be amplified by the count in front.
    term_n = np.zeros(n.shape, dtype=float)
    term_m = np.zeros(m.shape, dtype=float)
    on = n > 0.0
    off = m > 0.0
    term_n[on] = n[on] * np.log1p((m[on] - tau * n[on]) / ((1.0 + tau) * n[on]))
    term_m[off] = m[off] * np.log1p((tau * n[off] - m[off]) / ((1.0 + tau) * m[off]))
    statistic = np.maximum(-2.0 * (term_n + term_m), 0.0)
    return np.sign(n - m / tau) * np.sqrt(statistic)

import math


def auxiliary_statistic(n: float, m: float, tau: float, s: float) -> float:
    # The profiling step validates every argument and raises for invalid data.
    b = profile_background_estimate(n, m, tau, s)
    n = float(n)
    m = float(m)
    tau = float(tau)
    s = float(s)
    # Sample-space boundaries take the continuous limit of the expression. An empty
    # signal region gives zero. An empty control region gives zero when the profiled
    # background stays finite (s = 0 or n > (1 + tau) s); when it vanishes linearly
    # with m the limit is sqrt(n) ln(n / s), and at the crossover sqrt(n / 2) ln(n / s).
    if n == 0.0:
        return 0.0
    if m == 0.0:
        excess = n - (1.0 + tau) * s
        if s == 0.0 or excess > 0.0:
            return 0.0
        if excess < 0.0:
            return float(math.sqrt(n) * math.log(n / s))
        return float(math.sqrt(n / 2.0) * math.log(n / s))
    mean_on = s + b
    scale = math.sqrt(n * m) / math.sqrt(n / mean_on ** 2 + m / b ** 2)
    bracket = math.log(n / mean_on) / b - math.log(m / (tau * b)) / mean_on
    return float(scale * bracket)

import math


def corrected_discovery_significance(r: float, u: float) -> float:
    r = float(r)
    u = float(u)
    if not (math.isfinite(r) and math.isfinite(u)):
        raise ValueError("r and u must be finite")
    if r == 0.0 or u == 0.0:
        # The logarithmic adjustment is undefined, so r* = r.
        r_star = r
    else:
        if (r > 0.0) != (u > 0.0):
            raise ValueError("r and u must share their sign")
        r_star = r + math.log(u / r) / r
    return float(max(0.0, r_star))

import math


def _asimov_log_series(coeff):
    # coeff[0]=1. If A'=A L', solve order by order for L=log(A).
    result=[0.0]*len(coeff)
    for n in range(1,len(coeff)):
        result[n]=coeff[n]-sum(k*result[k]*coeff[n-k] for k in range(1,n))/n
    return result

def _asimov_poly(coeff, x):
    result=0.0
    for c in reversed(coeff):
        result=result*x+c
    return result

def _asimov_small_signal(s, b, sigma_b, degree=14):
    tau=b/(sigma_b*sigma_b)
    t=s/b
    a=1/(1+tau)
    scale=math.sqrt(b*tau/(1+tau))
    # r^2 = b*tau/(1+tau) * t^2 * F(t).
    # F_j=2 (-1)^j (1+a+...+a^j)/((j+1)(j+2)).
    f=[2*(-1)**j*sum(a**k for k in range(j+1))/((j+1)*(j+2)) for j in range(degree+1)]
    lf=_asimov_log_series(f)
    ll=_asimov_log_series([(-1)**j/(j+1) for j in range(degree+1)])
    # log(u/r)=C(t); evaluate C(t)/t directly, without forming u/r.
    c=[0.0]+[.5*(-1)**(j+1)*(1-a**j)/j+ll[j]-.5*lf[j] for j in range(1,degree+1)]
    c[1]=(tau-1)/(6*(1+tau))
    froot=math.sqrt(_asimov_poly(f,t))
    r=t*scale*froot
    rstar=r+_asimov_poly(c[1:],t)/(scale*froot)
    return r,rstar


def first_order_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    s = float(s)
    b = float(b)
    sigma_b = float(sigma_b)
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError("s must be a finite signal strength of at least zero")
    if not (math.isfinite(b) and b > 0.0):
        raise ValueError("b must be a finite background above zero")
    if not (math.isfinite(sigma_b) and sigma_b > 0.0):
        raise ValueError("sigma_b must be a finite standard deviation above zero")

    # Evaluate through the scale factor tau = b / sigma_b^2. The second term is
    # tau b ln(1 + s / ((1 + tau) b)), whose logarithm of a number close to one
    # is taken with log1p so that the result stays accurate as sigma_b -> 0.
    if s / b <= 0.01:
        return float(_asimov_small_signal(s, b, sigma_b)[0])
    tau = b / (sigma_b * sigma_b)
    term_on = (s + b) * math.log((s + (1.0 + tau) * b) / ((1.0 + tau) * (s + b)))
    term_off = tau * b * math.log1p(s / ((1.0 + tau) * b))
    return float(math.sqrt(max(-2.0 * (term_on + term_off), 0.0)))

import math


def corrected_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    # The first-order step validates the configuration and gives the signed root at
    # the Asimov point, which is non-negative for s >= 0.
    r_asimov = first_order_asimov_significance(s, b, sigma_b)
    s = float(s)
    b = float(b)
    sigma_b = float(sigma_b)
    tau = b / (sigma_b * sigma_b)
    if 0.0 < s / b <= 0.01:
        return float(max(0.0, _asimov_small_signal(s, b, sigma_b)[1]))
    if s == 0.0:
        # Both the root and the auxiliary statistic vanish. Expanding r_A and u_A to
        # third order in s gives ln(u_A / r_A) / r_A -> (tau - 1) / (6 sqrt(tau b (1 + tau))),
        # the limit used by the reference implementation at the hypothesis point.
        limit = (tau - 1.0) / (6.0 * math.sqrt(tau * b * (1.0 + tau)))
        return float(max(0.0, limit))
    u_asimov = auxiliary_statistic(s + b, tau * b, tau, 0.0)
    return corrected_discovery_significance(r_asimov, u_asimov)

import math

import numpy as np
from scipy.stats import norm, poisson


def _poisson_grid(mean: float, observed: int) -> "np.ndarray":
    """Support 0..K with K at least the observed count and omitting less than 1e-12 of mass."""
    upper = int(math.ceil(poisson.ppf(1.0 - 1e-12, mean))) if mean > 0.0 else 0
    return np.arange(max(upper, int(observed)) + 1)


def profile_construction_significance(n: int, m: int, tau: float) -> float:
    for name, count in (("n", n), ("m", m)):
        value = float(count)
        if not (math.isfinite(value) and value >= 0.0 and value == math.floor(value)):
            raise ValueError(f"{name} must be an integer count of at least zero")
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")
    n = int(n)
    m = int(m)

    r_observed = float(signed_likelihood_root(np.array(float(n)), np.array(float(m)), tau))
    if r_observed <= 0.0:
        return 0.0

    b_null = profile_background_estimate(n, m, tau, 0.0)
    n_grid = _poisson_grid(b_null, n)
    m_grid = _poisson_grid(tau * b_null, m)
    probability = poisson.pmf(n_grid, b_null)[:, None] * poisson.pmf(m_grid, tau * b_null)[None, :]
    roots = signed_likelihood_root(n_grid[:, None], m_grid[None, :], tau)
    # The observed pair belongs to its own tail whatever the round-off of its root.
    in_tail = roots >= r_observed - 1e-12 * max(1.0, r_observed)
    p_value = float(probability[in_tail].sum())
    return float(max(0.0, norm.isf(p_value)))

import math

import numpy as np
from scipy.stats import poisson


def median_discovery_significance(s: float, b: float, tau: float) -> float:
    s = float(s)
    b = float(b)
    tau = float(tau)
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError("s must be a finite signal strength of at least zero")
    if not (math.isfinite(b) and b > 0.0):
        raise ValueError("b must be a finite background above zero")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")

    n_grid = _poisson_grid(s + b, 0)
    m_grid = _poisson_grid(tau * b, 0)
    probability = (poisson.pmf(n_grid, s + b)[:, None] * poisson.pmf(m_grid, tau * b)[None, :]).ravel()
    significance = np.array([
        profile_construction_significance(int(n), int(m), tau)
        for n in n_grid for m in m_grid
    ])
    # Accumulate probability in increasing order of Z and stop at the first outcome
    # whose cumulative probability reaches one half.
    order = np.argsort(significance, kind="stable")
    cumulative = np.cumsum(probability[order])
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return float(significance[order][min(index, len(order) - 1)])

def orchestrate_asimov_residual(s: float, b: float, sigma_b: float) -> tuple:
    # The first-order step validates the configuration.
    z_first_order = first_order_asimov_significance(s, b, sigma_b)
    z_corrected = corrected_asimov_significance(s, b, sigma_b)
    tau = float(b) / (float(sigma_b) * float(sigma_b))
    z_median = median_discovery_significance(s, b, tau)
    return (float(z_corrected - z_median), float(z_corrected), float(z_median), float(z_first_order))
SCICODE_GOLD_EOF
