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
from scipy.special import erf, erfc

_SQRT_HALF_PI = np.sqrt(np.pi / 2.0)
_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _cb_tail_integral(a, b, alpha, n):
    r"""Integral of $A\,(B-t)^{-n}$ over $[a,b]$ with $b\leq-\alpha$, stable for all $n>0$."""
    A = (n / alpha) ** n * np.exp(-0.5 * alpha * alpha)
    B = n / alpha - alpha
    U = np.log(B - a)          # U >= V
    V = np.log(B - b)
    c = 1.0 - n
    d = U - V
    if c == 0.0:
        g = d
    else:
        g = np.expm1(c * d) / c
    return A * np.exp(c * V) * g


def _cb_core_integral(a, b):
    r"""Integral of $\exp(-t^2/2)$ over $[a,b]$ with $a\geq-\alpha$, via $\operatorname{erfc}$ (no cancellation)."""
    from scipy.special import erfc
    return _SQRT_HALF_PI * (erfc(a * _INV_SQRT2) - erfc(b * _INV_SQRT2))


def crystal_ball_bin_fractions(edges: np.ndarray, mu: float, sigma: float,
                                       alpha: float, n: float) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges must be a finite 1D array with at least 2 entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    for name, v in (("mu", mu), ("sigma", sigma), ("alpha", alpha), ("n", n)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite")
    if sigma <= 0 or alpha <= 0 or n <= 0:
        raise ValueError("sigma, alpha and n must be > 0")
    alpha = float(alpha)
    n = float(n)
    t = (edges - float(mu)) / float(sigma)
    a, b = t[:-1], t[1:]
    tc = -alpha
    out = np.zeros(a.size)
    tail = a < tc
    if np.any(tail):
        out[tail] += _cb_tail_integral(a[tail], np.minimum(b[tail], tc), alpha, n)
    core = b > tc
    if np.any(core):
        out[core] += _cb_core_integral(np.maximum(a[core], tc), b[core])
    total = out.sum()
    if not (total > 0.0) or not np.isfinite(total):
        raise ValueError("window integral underflows or is not finite")
    return (out / total).astype(float)

import numpy as np

_CMS_F_NARROW = 0.4       # weight of the narrower Crystal Ball
_CMS_WIDTH_RATIO = 1.55   # sigma2 / sigma1
_CMS_ALPHA1, _CMS_N1 = 2.0, 1.0
_CMS_ALPHA2, _CMS_N2 = 2.0, 2.0


def _double_cb_bin_fractions(edges, mu, sigma1):
    r"""$f\,\mathrm{CB}_1+(1-f)\,\mathrm{CB}_2$, each component window-normalized, common mean."""
    frac1 = crystal_ball_bin_fractions(edges, mu, sigma1, _CMS_ALPHA1, _CMS_N1)
    frac2 = crystal_ball_bin_fractions(edges, mu, _CMS_WIDTH_RATIO * sigma1,
                                               _CMS_ALPHA2, _CMS_N2)
    return _CMS_F_NARROW * frac1 + (1.0 - _CMS_F_NARROW) * frac2


def upsilon_signal_expected_counts(edges: np.ndarray, yields: np.ndarray, mu1: float,
                                           sigma1: float, masses: np.ndarray) -> np.ndarray:
    yields = np.asarray(yields, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if yields.shape != (3,) or masses.shape != (3,):
        raise ValueError("yields and masses must have shape (3,)")
    if not (np.all(np.isfinite(yields)) and np.all(np.isfinite(masses))):
        raise ValueError("yields and masses must be finite")
    if np.any(yields < 0):
        raise ValueError("yields must be non-negative")
    if np.any(masses <= 0) or not np.all(np.diff(masses) > 0):
        raise ValueError("masses must be positive and strictly increasing")
    rows = []
    for s in range(3):
        mu_s = mu1 + (masses[s] - masses[0])        # world-average mass differences (paper, Sec. 4)
        sig_s = sigma1 * masses[s] / masses[0]      # widths scale with mass
        rows.append(yields[s] * _double_cb_bin_fractions(edges, mu_s, sig_s))
    return np.vstack(rows).astype(float)

import numpy as np

def quadratic_background_bin_fractions(edges: np.ndarray, b1: float, b2: float) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges must be a finite 1D array with at least 2 entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    if not (np.isfinite(b1) and np.isfinite(b2)):
        raise ValueError("b1 and b2 must be finite")
    mc = 0.5 * (edges[0] + edges[-1])
    h = 0.5 * (edges[-1] - edges[0])
    x = (edges - mc) / h
    P = x + b1 * x ** 2 / 2.0 + b2 * x ** 3 / 3.0
    bins = np.diff(P)
    if np.any(bins <= 0):
        raise ValueError("background integral must be positive in every bin")
    return (bins / (P[-1] - P[0])).astype(float)

import numpy as np
from scipy.special import erf, erfc

def _gauss_bin_integrals(edges, m, s):
    r"""Probability of $\mathcal{N}(m,s^2)$ in each bin, without cancellation in either tail."""
    from scipy.special import erf, erfc
    z = (edges - m) / (np.sqrt(2.0) * s)
    za, zb = z[:-1], z[1:]
    out = np.empty(za.size)
    up = za >= 0.0                 # bin entirely above the mean
    lo = zb <= 0.0                 # bin entirely below the mean
    mid = ~(up | lo)
    out[up] = 0.5 * (erfc(za[up]) - erfc(zb[up]))
    out[lo] = 0.5 * (erfc(-zb[lo]) - erfc(-za[lo]))
    out[mid] = 0.5 * (erf(zb[mid]) - erf(za[mid]))
    return out


def generate_pseudo_data(edges: np.ndarray, yields: np.ndarray, means: np.ndarray,
                                 widths: np.ndarray, n_bkg: float, exp_slope: float,
                                 seed: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges must be a finite 1D array with at least 2 entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    yields = np.asarray(yields, dtype=float)
    means = np.asarray(means, dtype=float)
    widths = np.asarray(widths, dtype=float)
    for arr in (yields, means, widths):
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError("yields, means and widths must be finite with shape (3,)")
    if np.any(yields < 0) or np.any(widths <= 0):
        raise ValueError("yields must be >= 0 and widths > 0")
    if not np.isfinite(n_bkg) or n_bkg < 0:
        raise ValueError("n_bkg must be >= 0")
    if not np.isfinite(exp_slope) or exp_slope <= 0:
        raise ValueError("exp_slope must be > 0")
    nu = np.zeros(edges.size - 1)
    for s in range(3):
        nu += yields[s] * _gauss_bin_integrals(edges, means[s], widths[s])
    e = np.exp(-exp_slope * edges)
    nu += n_bkg * (e[:-1] - e[1:]) / (e[0] - e[-1])
    rng = np.random.default_rng(int(seed))
    return rng.poisson(nu).astype(np.int64)

import numpy as np
from scipy.optimize import minimize


def _extended_binned_nll(params, counts, edges, masses):
    r"""$\sum_i(\nu_i-n_i\ln\nu_i)$ of the CMS model; raises ValueError where undefined."""
    params = np.asarray(params, dtype=float)
    if params.shape != (8,) or not np.all(np.isfinite(params)):
        raise ValueError("params must be finite with shape (8,)")
    N1, N2, N3, NB, mu1, s1, b1, b2 = params
    if NB < 0:
        raise ValueError("N_bkg must be non-negative")
    sig = upsilon_signal_expected_counts(edges, np.array([N1, N2, N3]), mu1, s1, masses)
    nu = sig.sum(axis=0) + NB * quadratic_background_bin_fractions(edges, b1, b2)
    if np.any(nu <= 0):
        raise ValueError("expected counts must be positive in every bin")
    return float(np.sum(nu - counts * np.log(nu)))


def fit_upsilon_mass_spectrum(counts: np.ndarray, edges: np.ndarray, masses: np.ndarray,
                                      start: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    start = np.asarray(start, dtype=float)
    if counts.ndim != 1 or counts.size != edges.size - 1:
        raise ValueError("counts must be 1D with len(edges) - 1 entries")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be finite and non-negative")
    if start.shape != (8,) or not np.all(np.isfinite(start)):
        raise ValueError("start must be finite with shape (8,)")
    _extended_binned_nll(start, counts, edges, masses)  # raises if undefined

    def objective(p):
        try:
            return _extended_binned_nll(p, counts, edges, masses)
        except ValueError:
            return 1e30

    opts = {"maxiter": 20000, "maxfev": 20000, "xatol": 1e-10, "fatol": 1e-12}
    x, fbest = start.copy(), objective(start)
    for _ in range(5):
        res = minimize(objective, x, method="Nelder-Mead", options=opts)
        improved = fbest - res.fun
        x, fbest = res.x, res.fun
        if improved < 1e-9:
            break
    return np.asarray(x, dtype=float)

import numpy as np

def cross_section_times_bf(n_signal: float, lumi_fb_inv: float, pt_min: float, pt_max: float,
                                   abs_y_min: float, abs_y_max: float, eff_mu1: float, eff_mu2: float,
                                   rho_pair: float, acceptance: float) -> float:
    vals = [n_signal, lumi_fb_inv, pt_min, pt_max, abs_y_min, abs_y_max,
            eff_mu1, eff_mu2, rho_pair, acceptance]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("all inputs must be finite")
    if n_signal < 0 or lumi_fb_inv <= 0:
        raise ValueError("n_signal must be >= 0 and luminosity > 0")
    if pt_min < 0 or pt_max <= pt_min or abs_y_min < 0 or abs_y_max <= abs_y_min:
        raise ValueError("invalid pT or |y| interval")
    for v in (eff_mu1, eff_mu2, rho_pair, acceptance):
        if not (0.0 < v <= 1.0):
            raise ValueError("efficiencies and acceptance must lie in (0, 1]")
    lumi_pb = 1000.0 * lumi_fb_inv
    delta_y = 2.0 * (abs_y_max - abs_y_min)
    delta_pt = pt_max - pt_min
    eps = eff_mu1 * eff_mu2 * rho_pair
    return float(n_signal / (lumi_pb * delta_y * delta_pt * eps * acceptance))

import numpy as np

def polarization_scale_factor(lambda_theta: float, k_plus: float, k_minus: float) -> float:
    if not all(np.isfinite(v) for v in (lambda_theta, k_plus, k_minus)):
        raise ValueError("inputs must be finite")
    if lambda_theta < -1.0 or lambda_theta > 1.0:
        raise ValueError("lambda_theta must lie in [-1, 1]")
    if k_plus <= 0 or k_minus <= 0:
        raise ValueError("k_plus and k_minus must be > 0")
    # Normalized decay distribution W = 3(1 + lam c^2) / [2(3 + lam)], c = cos(theta).
    # Accepted fraction: A_lam / A_0 = 3 (1 + lam r) / (3 + lam), with r = <c^2> over the
    # accepted unpolarized events.  Hence k(lam) = A_0 / A_lam = (3 + lam) / [3 (1 + lam r)],
    # which gives k(0) = 1 identically and k(+1)/k(-1) = 2 (1 - r) / (1 + r).
    q = k_plus / k_minus
    if q > 2.0:
        raise ValueError("k_plus / k_minus must not exceed 2")
    r = (2.0 - q) / (2.0 + q)          # acceptance-weighted <cos^2 theta>, in [0, 1)
    return float((3.0 + lambda_theta) / (3.0 * (1.0 + lambda_theta * r)))

import numpy as np

_TABLE1_K_PLUS_70_100_Y06 = 1.12    # CMS Table 1, 70-100 GeV, |y| < 0.6, lambda = +1
_TABLE1_K_MINUS_70_100_Y06 = 0.83   # CMS Table 1, 70-100 GeV, |y| < 0.6, lambda = -1


def upsilon_polarized_cross_section(lambda_theta: float = -0.5, seed: int = 13600,
                                            state_index: int = 2) -> float:
    if state_index not in (0, 1, 2):
        raise ValueError("state_index must be 0, 1 or 2")
    k = polarization_scale_factor(lambda_theta, _TABLE1_K_PLUS_70_100_Y06,
                                          _TABLE1_K_MINUS_70_100_Y06)
    edges = np.linspace(8.5, 11.5, 76)
    masses = np.array([9.46040, 10.0234, 10.3551])
    counts = generate_pseudo_data(edges, np.array([2800.0, 1300.0, 900.0]),
                                          masses - 0.012, np.array([0.072, 0.077, 0.080]),
                                          7000.0, 0.4, seed)
    start = np.array([2500.0, 1200.0, 800.0, 7000.0, 9.45, 0.06, 0.0, 0.0])
    p = fit_upsilon_mass_spectrum(counts, edges, masses, start)
    xsec0 = cross_section_times_bf(p[state_index], 37.4, 70.0, 100.0, 0.0, 0.6,
                                           0.92, 0.90, 0.95, 0.38)
    return float(k * xsec0)
SCICODE_GOLD_EOF
