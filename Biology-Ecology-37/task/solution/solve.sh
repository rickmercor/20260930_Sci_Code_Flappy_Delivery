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


def _bcf_probability(name: str, value: float) -> complex:
    """A breeding probability strictly inside (0, 1); complex values with that real part pass (complex-step use)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, complex, np.number)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value) or not 0.0 < float(np.real(value)) < 1.0:
        raise ValueError(f"{name} must be a finite number strictly between 0 and 1")
    return value


def breeding_cycle_fecundity(psi2: float, psi3: float, entry_age: int, ages: "np.ndarray") -> "np.ndarray":
    p2 = _bcf_probability("psi2", psi2)
    p3 = _bcf_probability("psi3", psi3)
    if isinstance(entry_age, bool) or not isinstance(entry_age, (int, np.integer)) or not 1 <= entry_age <= 20:
        raise ValueError("entry_age must be an integer from 1 to 20")
    a = np.asarray(ages)
    if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 200:
        raise ValueError("ages must be a non-empty 1-D array of integers from 0 to 200")
    # states (pregnant, calf, resting); column-stochastic: next = T @ current
    zero = 0.0 * p2
    T = np.array([[zero, p2, p3], [zero + 1.0, zero, zero], [zero, 1.0 - p2, 1.0 - p3]])
    beta2 = p3 / (2.0 * p3 + (1.0 - p2))        # calf-state share of the stationary distribution (1 - p2 exact)
    steps = int(a.max()) - entry_age
    calf = [zero] * (max(steps, 0) + 1)
    state = np.array([zero, zero, zero + 1.0])   # resting at entry
    for t in range(max(steps, 0) + 1):
        calf[t] = state[1]
        state = T @ state
    fec = [(calf[int(x) - entry_age] if x >= entry_age else zero) / beta2 for x in a]
    return np.array([beta2] + fec)

import numpy as np


def _sa_year(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1900 <= value <= 2200:
        raise ValueError(f"{name} must be an integer from 1900 to 2200")
    return int(value)


def stage_abundances(theta: "np.ndarray", ref_year: int, years: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    re_th = np.real(th)
    if not (0.0 <= re_th[0] <= 30.0 and -0.5 <= re_th[1] <= 0.5 and np.all(np.abs(re_th[2:]) <= 15.0)):
        raise ValueError("theta must lie in the stated domain")
    y0 = _sa_year("ref_year", ref_year)
    y = np.asarray(years)
    if y.ndim != 1 or y.size == 0 or not np.issubdtype(y.dtype, np.integer) or y.min() < 1900 or y.max() > 2200:
        raise ValueError("years must be a non-empty 1-D array of integers from 1900 to 2200")
    log_n, r = th[0], th[1]
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    lam = np.exp(r)
    lam_minus_phi_a = np.expm1(r) + 1.0 / (1.0 + np.exp(th[2]))   # exp(r) - phi_A without cancellation near phi_A = 1
    if not float(np.real(lam_minus_phi_a)) > 0.0:
        raise ValueError("exp(r) must exceed adult survival")
    adult = np.exp(log_n + r * (y - y0))
    # juveniles aged 1..5 per adult: sum_a lam^-a phi_J^(a-1) over the adult sum lam^-6 phi_J^5 / (1 - phi_A/lam)
    juv_sum = sum(lam ** (-a) * phi_j ** (a - 1) for a in range(1, 6))
    ratio = juv_sum * lam_minus_phi_a * lam ** 5 / phi_j ** 5
    return np.array([adult, adult * ratio])

import numpy as np


def _sc_number(name: str, value: float, low: float, high: float, open_ends: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    v = float(value)
    if not np.isfinite(v) or v < low or v > high or (open_ends and (v <= low or v >= high)):
        raise ValueError(f"{name} must be a finite number in the stated range")
    return v


def sample_composition(sample_sizes: "np.ndarray", growth_rate: float, adult_survival: float,
                               juvenile_survival: float, max_female_age: int, max_male_age: int) -> "np.ndarray":
    n = np.asarray(sample_sizes, dtype=float)
    if n.ndim != 1 or n.size == 0 or not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("sample_sizes must be a non-empty 1-D array of finite non-negative numbers")
    r = _sc_number("growth_rate", growth_rate, -0.5, 0.5, False)
    phi_a = _sc_number("adult_survival", adult_survival, 0.0, 1.0, True)
    phi_j = _sc_number("juvenile_survival", juvenile_survival, 0.0, 1.0, True)
    lam = np.exp(r)
    if phi_a >= lam:
        raise ValueError("adult_survival must be below exp(growth_rate)")
    for name, v, lo, hi in (("max_female_age", max_female_age, 6, 100), ("max_male_age", max_male_age, 0, 5)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    ages = np.arange(1, max_female_age + 1)
    rel = np.where(ages < 6, lam ** (-ages) * phi_j ** (ages - 1),
                   lam ** (-ages) * phi_j ** 5 * phi_a ** (ages - 6))
    rel_m = np.where(ages <= max_male_age, rel, 0.0)
    share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())
    return n[:, None, None] * share[None, :, :]

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr


def _aes_log_mass(upper: "np.ndarray", lower: "np.ndarray") -> "np.ndarray":
    """log(Phi(upper) - Phi(lower)) for lower < upper, accurate in both tails."""
    flip = lower > 0.0                                     # both in the upper tail: use Phi(-lower) - Phi(-upper)
    lo = np.where(flip, -upper, lower)
    hi = np.where(flip, -lower, upper)
    log_hi = log_ndtr(hi)
    return log_hi + np.log1p(-np.exp(log_ndtr(lo) - log_hi))


def _aes_loglik(sigma: float, a: "np.ndarray", e: "np.ndarray", max_age: int) -> float:
    reported = _aes_log_mass((e + 0.5 - a) / sigma, (e - 0.5 - a) / sigma)
    in_range = _aes_log_mass((max_age + 0.5 - a) / sigma, (0.5 - a) / sigma)
    return float(np.sum(reported - in_range))


def _aes_score(sigma: float, a: "np.ndarray", e: "np.ndarray", max_age: int) -> float:
    """Derivative of the log-likelihood with respect to sigma."""
    def _dlog(upper: "np.ndarray", lower: "np.ndarray") -> "np.ndarray":
        u, v = upper / sigma, lower / sigma
        log_mass = _aes_log_mass(u, v)
        log_pdf = lambda x: -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)
        return -(u * np.exp(log_pdf(u) - log_mass) - v * np.exp(log_pdf(v) - log_mass)) / sigma
    return float(np.sum(_dlog(e + 0.5 - a, e - 0.5 - a) - _dlog(max_age + 0.5 - a, 0.5 - a)))


def age_error_sd(true_ages: "np.ndarray", estimated_ages: "np.ndarray", max_age: int) -> float:
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 2 <= max_age <= 100:
        raise ValueError("max_age must be an integer from 2 to 100")
    a = np.asarray(true_ages)
    e = np.asarray(estimated_ages)
    for v in (a, e):
        if v.ndim != 1 or v.size == 0 or not np.issubdtype(v.dtype, np.integer) or v.min() < 1 or v.max() > max_age:
            raise ValueError("ages must be non-empty 1-D integer arrays with values from 1 to max_age")
    if a.shape != e.shape:
        raise ValueError("true_ages and estimated_ages must have the same length")
    if np.all(a == e):
        raise ValueError("at least one estimated age must differ from its true age")
    a = a.astype(float)
    e = e.astype(float)
    grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 2001))
    values = np.array([_aes_loglik(s, a, e, max_age) for s in grid])
    k = int(np.argmax(values))
    if k == 0 or k == grid.size - 1:
        raise ValueError("the likelihood is largest at sigma = 0.05 or sigma = 50")
    for width in range(1, grid.size):                      # the score changes sign around the grid's best point
        lo, hi = grid[max(k - width, 0)], grid[min(k + width, grid.size - 1)]
        s_lo, s_hi = _aes_score(lo, a, e, max_age), _aes_score(hi, a, e, max_age)
        if s_lo >= 0.0 >= s_hi:
            break
    else:
        raise ValueError("the likelihood has no interior maximum for sigma from 0.05 to 50")
    if s_lo == 0.0:
        root = lo
    elif s_hi == 0.0:
        root = hi
    else:                                                  # the root of the score, not a comparison of likelihood values
        root = brentq(_aes_score, lo, hi, args=(a, e, max_age), xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    return float(root)

import numpy as np
from scipy.special import log_ndtr


def _ean_log_mass(upper: "np.ndarray", lower: "np.ndarray") -> "np.ndarray":
    """log(Phi(upper) - Phi(lower)) for lower < upper, accurate in both tails."""
    flip = lower > 0.0
    lo = np.where(flip, -upper, lower)
    hi = np.where(flip, -lower, upper)
    log_hi = log_ndtr(hi)
    return log_hi + np.log1p(-np.exp(log_ndtr(lo) - log_hi))


def _ean_error_matrix(error_sd: float, max_age: int) -> "np.ndarray":
    """P[a - 1, e - 1]: probability of estimated age e for true age a."""
    a = np.arange(1, max_age + 1, dtype=float)[:, None]
    e = np.arange(1, max_age + 1, dtype=float)[None, :]
    log_p = _ean_log_mass((e + 0.5 - a) / error_sd, (e - 0.5 - a) / error_sd) \
        - _ean_log_mass((max_age + 0.5 - a) / error_sd, (0.5 - a) / error_sd)
    return np.exp(log_p)


def estimated_age_numbers(samples: "np.ndarray", error_sd: float) -> "np.ndarray":
    m = np.asarray(samples, dtype=float)
    if m.ndim != 3 or m.shape[0] < 1 or m.shape[1] != 2 or not 2 <= m.shape[2] <= 100 \
            or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("samples must be a finite non-negative array of shape (Y, 2, A) with 2 <= A <= 100")
    if isinstance(error_sd, bool) or not isinstance(error_sd, (int, float, np.integer, np.floating)) \
            or not np.isfinite(error_sd) or not 0.05 <= float(error_sd) <= 50.0:
        raise ValueError("error_sd must be a finite number from 0.05 to 50")
    P = _ean_error_matrix(float(error_sd), m.shape[2])
    return m[:, :, :, None] * P[None, None, :, :]

import numpy as np


def _mop_inputs(sample_years: "np.ndarray", first_year: int, numbers: "np.ndarray") -> tuple:
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer) or ys.min() < 1900 or ys.max() > 2200 \
            or np.any(np.diff(ys) <= 0):
        raise ValueError("sample_years must be a non-empty strictly increasing 1-D array of integers from 1900 to 2200")
    ys = ys.astype(np.int64)
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year < ys.max():
        raise ValueError("first_year must be an integer from 1900 to 2200 below max(sample_years)")
    J = np.asarray(numbers, dtype=float)
    if J.ndim != 4 or J.shape[0] != ys.size or J.shape[1] != 2 or J.shape[2] != J.shape[3] or not 6 <= J.shape[2] <= 100 \
            or not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("numbers must be a finite non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100")
    if J.shape[2] + int(ys.max()) - int(ys.min()) > 200:
        raise ValueError("A + max(sample_years) - min(sample_years) must not exceed 200")
    return ys, int(first_year), J


def _mop_true_age_weights(J: "np.ndarray") -> "np.ndarray":
    """W[i, s, a - 1, e - 1]: probability of true age a given sampling year i, sex s and estimated age e."""
    total = J.sum(axis=2, keepdims=True)
    return np.where(total > 0, J / np.where(total > 0, total, 1.0), 0.0)


def mother_offspring_probabilities(theta: "np.ndarray", ref_year: int, entry_age: int, sample_years: "np.ndarray",
                                           first_year: int, numbers: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    ys, first, J = _mop_inputs(sample_years, first_year, numbers)
    n_a = J.shape[2]
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    psi2 = 1.0 / (1.0 + np.exp(-th[4]))
    psi3 = 1.0 / (1.0 + np.exp(-th[5]))
    births = np.arange(first, int(ys.max()))                      # modelled birth years a sampled animal can have
    adult = stage_abundances(th, ref_year, births)[0]      # validates theta and ref_year
    ages = np.arange(1, n_a + 1)
    yi = ys[:, None, None, None]
    ai = ages[None, :, None, None]
    b = ys[None, None, :, None] - ages[None, None, None, :]         # true birth year of the other animal
    age_at_birth = ai + (b - yi)
    oldest = int(age_at_birth.max())
    fec = breeding_cycle_fecundity(psi2, psi3, entry_age, np.arange(0, max(oldest, 0) + 1))[1:]
    before = yi < b
    dt = np.where(before, b - yi, 0)                               # survival only from sampling up to the birth year
    juv_years = np.clip(np.minimum(6 - ai, dt), 0, None)           # years spent at ages 1 to 5
    surv = phi_j ** juv_years * phi_a ** (dt - juv_years)
    modelled = (b >= first) & (age_at_birth >= 1)
    idx_b = np.clip(b - first, 0, births.size - 1)
    true_prob = np.where(modelled, surv * fec[np.clip(age_at_birth, 0, oldest)] / adult[idx_b], 0.0 * surv)
    W = _mop_true_age_weights(J)
    half = np.einsum("iae,iajc->iejc", W[:, 0], true_prob, optimize=True)          # mother's true age averaged out
    return np.einsum("iejc,jscf->iejsf", half, W, optimize=True)                   # the other animal's true age averaged out

import numpy as np


def _hsp_calf_return(psi2: complex, psi3: complex, max_gap: int) -> "np.ndarray":
    """Probability of being in the calf state g years after being in it, g = 0..max_gap."""
    zero = 0.0 * psi2
    T = np.array([[zero, psi2, psi3], [zero + 1.0, zero, zero], [zero, 1.0 - psi2, 1.0 - psi3]])
    state = np.array([zero, zero + 1.0, zero])
    out = []
    for _ in range(max_gap + 1):
        out.append(state[1])
        state = T @ state
    return np.array(out)


def _hsp_inputs(sample_years: "np.ndarray", first_year: int, numbers: "np.ndarray") -> tuple:
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer) or ys.min() < 1900 or ys.max() > 2200 \
            or np.any(np.diff(ys) <= 0):
        raise ValueError("sample_years must be a non-empty strictly increasing 1-D array of integers from 1900 to 2200")
    ys = ys.astype(np.int64)
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year < ys.max():
        raise ValueError("first_year must be an integer from 1900 to 2200 below max(sample_years)")
    J = np.asarray(numbers, dtype=float)
    if J.ndim != 4 or J.shape[0] != ys.size or J.shape[1] != 2 or J.shape[2] != J.shape[3] or not 6 <= J.shape[2] <= 100 \
            or not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("numbers must be a finite non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100")
    return ys, int(first_year), J


def half_sibling_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray", first_year: int,
                                       numbers: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    ys, first, J = _hsp_inputs(sample_years, first_year, numbers)
    n_a = J.shape[2]
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    psi2 = 1.0 / (1.0 + np.exp(-th[4]))
    psi3 = 1.0 / (1.0 + np.exp(-th[5]))
    births = np.arange(first, int(ys.max()))                      # modelled birth years a sampled animal can have
    adult = stage_abundances(th, ref_year, births)[0]      # validates theta and ref_year
    beta2 = breeding_cycle_fecundity(psi2, psi3, 1, np.array([0]))[0]
    ret = _hsp_calf_return(psi2, psi3, births.size)
    b = (ys[:, None] - np.arange(1, n_a + 1)[None, :])             # (Y, A): true birth year of each (year, true age)
    b1 = b[:, :, None, None]
    b2 = b[None, None, :, :]
    early, late = np.minimum(b1, b2), np.maximum(b1, b2)
    gap = late - early
    modelled = (early >= first) & (gap > 0)
    g = np.where(modelled, gap, 0)
    idx_late = np.clip(late - first, 0, births.size - 1)
    true_prob = np.where(modelled, phi_a ** g * ret[g] / (adult[idx_late] * beta2), 0.0 * phi_a)
    total = J.sum(axis=2, keepdims=True)
    W = np.where(total > 0, J / np.where(total > 0, total, 1.0), 0.0)     # P(true age | year, sex, estimated age)
    half = np.einsum("isae,iajc->isejc", W, true_prob, optimize=True)
    return np.einsum("isejc,jtcf->isejtf", half, W, optimize=True)

import numpy as np


def self_recapture_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray",
                                         numbers: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer) or ys.min() < 1900 or ys.max() > 2200 \
            or np.any(np.diff(ys) <= 0):
        raise ValueError("sample_years must be a non-empty strictly increasing 1-D array of integers from 1900 to 2200")
    ys = ys.astype(np.int64)
    J = np.asarray(numbers, dtype=float)
    if J.ndim != 4 or J.shape[0] != ys.size or J.shape[1] != 2 or J.shape[2] != J.shape[3] or not 6 <= J.shape[2] <= 100 \
            or not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("numbers must be a finite non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100")
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    stage_n = stage_abundances(th, ref_year, ys)                  # (2, Y): adult, juvenile; validates theta
    ages = np.arange(1, J.shape[2] + 1)
    y1 = ys[:, None, None, None]
    a1 = ages[None, :, None, None]
    y2 = ys[None, None, :, None]
    d = np.arange(2)[None, None, None, :]
    dt = np.clip(y2 - y1, 0, None)
    juv_years = np.clip(np.minimum(6 - a1, dt), 0, None)
    surv = phi_j ** juv_years * phi_a ** (dt - juv_years)
    stage_later = (a1 + dt >= 6).astype(np.int64)
    denom = np.where(d == 1, stage_n[0][None, None, :, None], stage_n[1][None, None, :, None])
    true_prob = np.where((y2 > y1) & (stage_later == d), surv / denom, 0.0 * surv)     # (Y, a, Y, d)
    total = J[:, 0].sum(axis=1, keepdims=True)
    W = np.where(total > 0, J[:, 0] / np.where(total > 0, total, 1.0), 0.0)           # (Y, a, e) for females
    return np.einsum("iae,iamd->iemd", W, true_prob, optimize=True)

import numpy as np


def comparison_counts(numbers: "np.ndarray", sample_years: "np.ndarray", first_year: int, entry_age: int,
                              kind: str) -> "np.ndarray":
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer) or ys.min() < 1900 or ys.max() > 2200 \
            or np.any(np.diff(ys) <= 0):
        raise ValueError("sample_years must be a non-empty strictly increasing 1-D array of integers from 1900 to 2200")
    ys = ys.astype(np.int64)
    J = np.asarray(numbers, dtype=float)
    if J.ndim != 4 or J.shape[0] != ys.size or J.shape[1] != 2 or J.shape[2] != J.shape[3] or not 6 <= J.shape[2] <= 100 \
            or not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("numbers must be a finite non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100")
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year <= 2200:
        raise ValueError("first_year must be an integer from 1900 to 2200")
    if isinstance(entry_age, bool) or not isinstance(entry_age, (int, np.integer)) or not 1 <= entry_age <= 20:
        raise ValueError("entry_age must be an integer from 1 to 20")
    if kind not in ("MOP", "HSP", "SP"):
        raise ValueError('kind must be "MOP", "HSP" or "SP"')
    n_y, n_a = ys.size, J.shape[2]
    first_birth_age = entry_age + 2
    est = J.sum(axis=2)                                             # (Y, 2, A): expected numbers by estimated age
    e = np.arange(1, n_a + 1)
    birth_est = ys[:, None] - e[None, :]                            # (Y, A): estimated birth years
    if kind == "MOP":
        usable = est * (birth_est >= first_year)[:, None, :]        # potential offspring, either sex
        same_year_young = np.eye(n_y, dtype=bool)[:, None, :, None, None] & (e < first_birth_age)[None, None, None, None, :]
        pairs = est[:, 0, :][:, :, None, None, None] * usable[None, None, :, :, :]
        return np.where(same_year_young, 0.0, pairs)
    if kind == "HSP":
        b1 = birth_est[:, None, :, None, None, None]
        b2 = birth_est[None, None, None, :, None, :]
        gap = b2 - b1
        admitted = (b1 >= first_year) & (b2 >= first_year) & (gap > 0) & (gap < 2 * first_birth_age + 2)
        return np.where(admitted, est[:, :, :, None, None, None] * est[None, None, None, :, :, :], 0.0)
    true_f = J[:, 0].sum(axis=2)                                    # (Y, A) females by true age
    stage_tot = np.stack([true_f[:, :5].sum(axis=1), true_f[:, 5:].sum(axis=1)], axis=1)      # (Y, 2)
    later = ys[None, :] > ys[:, None]
    return est[:, 0, :][:, :, None, None] * stage_tot[None, None, :, :] * later[:, None, :, None]

import numpy as np


def _pfi_complex_step(fun, theta: "np.ndarray") -> tuple:
    """Value and exact first derivatives of an array-valued function of theta (complex step, h = 1e-30)."""
    base = np.real(fun(theta.astype(float)))
    grads = []
    for k in range(theta.size):
        th = theta.astype(complex)
        th[k] += 1e-30j
        grads.append(np.imag(fun(th)) / 1e-30)
    return base, np.stack(grads, axis=-1)


def pseudo_fisher_information(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                                      sample_years: "np.ndarray", first_year: int, use_kin: bool) -> "np.ndarray":
    th = np.asarray(theta, dtype=float)
    if th.shape != (6,) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    if not isinstance(use_kin, (bool, np.bool_)):
        raise ValueError("use_kin must be a bool")
    ys = np.asarray(sample_years)
    if ys.ndim != 1 or ys.size == 0 or not np.issubdtype(ys.dtype, np.integer):
        raise ValueError("sample_years must be a non-empty 1-D integer array")
    if isinstance(first_year, bool) or not isinstance(first_year, (int, np.integer)) or not 1900 <= first_year < ys.max():
        raise ValueError("first_year must be an integer from 1900 to 2200 below max(sample_years)")
    J = np.asarray(numbers, dtype=float)
    parts = [("SP", lambda t: self_recapture_probabilities(t, ref_year, ys, J))]
    if use_kin:
        parts += [("MOP", lambda t: mother_offspring_probabilities(t, ref_year, entry_age, ys, first_year, J)),
                  ("HSP", lambda t: half_sibling_probabilities(t, ref_year, ys, first_year, J))]
    info = np.zeros((6, 6))
    for kind, fun in parts:
        n = comparison_counts(J, ys, first_year, entry_age, kind)
        p, dp = _pfi_complex_step(fun, th)
        use = (n > 0) & (p > 0)
        d = dp[use] * (np.sqrt(n[use]) / np.sqrt(p[use]))[:, None]      # no overflow when n and p are both tiny
        info += d.T @ d
    return 0.5 * (info + info.T)

import numpy as np


def _dp_inverse(info: "np.ndarray") -> "np.ndarray":
    """Inverse of a positive definite information matrix, or ValueError."""
    try:
        np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        raise ValueError("the information matrix is not positive definite") from None
    return np.linalg.inv(info)


def design_precision(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                             sample_years: "np.ndarray", first_year: int, target_year: int) -> "np.ndarray":
    if isinstance(target_year, bool) or not isinstance(target_year, (int, np.integer)) or not 1900 <= target_year <= 2200:
        raise ValueError("target_year must be an integer from 1900 to 2200")
    th = np.asarray(theta, dtype=float)
    info_all = pseudo_fisher_information(th, ref_year, entry_age, numbers, sample_years, first_year, True)
    info_sp = pseudo_fisher_information(th, ref_year, entry_age, numbers, sample_years, first_year, False)
    cov = _dp_inverse(info_all)
    cov_sp = _dp_inverse(info_sp[:4, :4])
    grad_n = np.array([1.0, float(target_year - ref_year), 0.0, 0.0, 0.0, 0.0])   # of ln N(target_year)
    cv_all = np.sqrt(grad_n @ cov @ grad_n)
    cv_sp = np.sqrt(grad_n[:4] @ cov_sp @ grad_n[:4])
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    se_phi_a = phi_a * (1.0 - phi_a) * np.sqrt(cov[2, 2])
    grad_b = np.zeros(6)
    for k in (4, 5):
        t = th.astype(complex)
        t[k] += 1e-30j
        grad_b[k] = np.imag(breeding_cycle_fecundity(1.0 / (1.0 + np.exp(-t[4])), 1.0 / (1.0 + np.exp(-t[5])),
                                                             entry_age, np.array([0]))[0]) / 1e-30
    se_beta = np.sqrt(grad_b @ cov @ grad_b)
    ys = np.asarray(sample_years)
    J = np.asarray(numbers, dtype=float)
    expected = [
        np.sum(comparison_counts(J, ys, first_year, entry_age, "MOP")
               * mother_offspring_probabilities(th, ref_year, entry_age, ys, first_year, J)),
        np.sum(comparison_counts(J, ys, first_year, entry_age, "HSP")
               * half_sibling_probabilities(th, ref_year, ys, first_year, J)),
        np.sum(comparison_counts(J, ys, first_year, entry_age, "SP")
               * self_recapture_probabilities(th, ref_year, ys, J)),
    ]
    return np.array([cv_all, cv_sp, se_phi_a, se_beta] + expected, dtype=float)

import numpy as np


def smallest_design_expected_kin_pairs(theta: "np.ndarray", ref_year: int, entry_age: int, first_year: int,
                                               past_years: "np.ndarray", past_size: float, new_years: "np.ndarray",
                                               size_step: int, max_size: int, max_female_age: int, max_male_age: int,
                                               target_year: int, cv_target: float, reference_true_ages: "np.ndarray",
                                               reference_estimated_ages: "np.ndarray") -> float:
    th = np.asarray(theta, dtype=float)
    if th.shape != (6,) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    past = np.asarray(past_years)
    new = np.asarray(new_years)
    for name, v in (("past_years", past), ("new_years", new)):
        if v.ndim != 1 or v.size == 0 or not np.issubdtype(v.dtype, np.integer) or np.any(np.diff(v) <= 0):
            raise ValueError(f"{name} must be a non-empty strictly increasing 1-D integer array")
    if past.max() >= new.min():
        raise ValueError("every past year must be earlier than every new year")
    if isinstance(past_size, bool) or not isinstance(past_size, (int, float, np.integer, np.floating)) \
            or not np.isfinite(past_size) or not 0 <= past_size <= 1e6:
        raise ValueError("past_size must be a finite number from 0 to 1e6")
    for name, v, lo, hi in (("size_step", size_step, 1, 10000), ("max_size", max_size, 1, 100000)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if max_size < size_step or max_size % size_step:
        raise ValueError("max_size must be a multiple of size_step")
    if isinstance(cv_target, bool) or not isinstance(cv_target, (int, float, np.integer, np.floating)) \
            or not np.isfinite(cv_target) or not 0 < cv_target < 1:
        raise ValueError("cv_target must be a finite number strictly between 0 and 1")
    error_sd = age_error_sd(reference_true_ages, reference_estimated_ages, max_female_age)
    years = np.concatenate([past, new]).astype(np.int64)
    r = float(th[1])
    phi_a = 1.0 / (1.0 + np.exp(-th[2]))
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    for n in range(size_step, max_size + 1, size_step):
        sizes = np.array([float(past_size)] * past.size + [float(n)] * new.size)
        samples = sample_composition(sizes, r, phi_a, phi_j, max_female_age, max_male_age)
        numbers = estimated_age_numbers(samples, error_sd)
        precision = design_precision(th, ref_year, entry_age, numbers, years, first_year, target_year)
        if precision[0] <= cv_target:
            return float(precision[4] + precision[5] + precision[6])
    raise ValueError("no annual sample size up to max_size reaches cv_target")
SCICODE_GOLD_EOF
