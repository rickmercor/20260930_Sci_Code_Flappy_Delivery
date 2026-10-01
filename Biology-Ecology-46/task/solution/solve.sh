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
from scipy.optimize import brentq
from scipy.special import digamma


def newborn_calibration(scores: "np.ndarray") -> "np.ndarray":
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("scores must be a one-dimensional array of at least two positive, finite values")
    if np.all(x == x[0]):
        raise ValueError("the scores must not all be equal")
    mean = float(np.mean(x))
    gap = float(np.log(mean) - np.mean(np.log(x)))         # log of arithmetic over geometric mean, > 0
    if not gap > 0.0:
        raise ValueError("the scores are too nearly equal for a finite maximum-likelihood dispersion")

    def _shape_equation(log_shape: float) -> float:
        shape = np.exp(log_shape)
        return float(log_shape - digamma(shape) - gap)     # decreasing in the shape; its root is the MLE

    lo, hi = -30.0, 60.0
    log_shape = brentq(_shape_equation, lo, hi, xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    return np.array([mean, float(np.exp(-log_shape))])

import numpy as np


def sampled_age_distribution(mortality: float, juvenile_mortality: float, maturity_age: int, growth: float, min_age: int, max_age: int) -> "np.ndarray":
    if not (np.isfinite(mortality) and mortality > 0.0):
        raise ValueError("mortality must be positive and finite")
    if not (np.isfinite(juvenile_mortality) and juvenile_mortality > 0.0):
        raise ValueError("juvenile_mortality must be positive and finite")
    if not np.isfinite(growth):
        raise ValueError("growth must be finite")
    if not mortality + growth > 0.0:
        raise ValueError("mortality + growth must be positive")
    for v in (min_age, max_age, maturity_age):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("min_age, max_age and maturity_age must be integers")
    if not 0 <= maturity_age <= max_age:
        raise ValueError("need 0 <= maturity_age <= max_age")
    if not (1 <= max_age <= 400 and 0 <= min_age <= max_age):
        raise ValueError("need 0 <= min_age <= max_age and 1 <= max_age <= 400")
    ages = np.arange(max_age + 1, dtype=float)
    lived = (juvenile_mortality * np.minimum(ages, float(maturity_age))      # the juvenile rate, then the adult one
             + mortality * np.maximum(ages - float(maturity_age), 0.0))
    weights = np.where(ages >= min_age, np.exp(-lived - growth * ages), 0.0)  # survival times the birth cohort's size
    return weights / weights.sum()

import numpy as np
from scipy.special import gammainc, gammaincc


def score_bin_probabilities(lower: float, upper: float, alpha: float, beta: float, dispersion: float, max_age: int) -> "np.ndarray":
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
        raise ValueError("the bin edges must be finite with 0 <= lower < upper")
    for v in (alpha, beta, dispersion):
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError("alpha, beta and dispersion must be positive and finite")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 0 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 0 to 400")
    shape = 1.0 / dispersion
    scale = dispersion * (alpha + beta * np.arange(max_age + 1, dtype=float))
    x_lo, x_hi = lower / scale, upper / scale
    lower_side = gammainc(shape, x_hi) - gammainc(shape, x_lo)
    upper_side = gammaincc(shape, x_lo) - gammaincc(shape, x_hi)   # the same mass, without cancellation above the mode
    return np.where(x_lo > shape, upper_side, lower_side)

import numpy as np


def maternal_half_sibling_matrix(year_1: int, year_2: int, mortality: float, reference_females: float, growth: float, reference_year: int, within_cohort_factor: float, breeding_period: int, max_age: int) -> "np.ndarray":
    for v in (year_1, year_2, reference_year):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("years must be integers")
    if not (np.isfinite(mortality) and mortality > 0.0):
        raise ValueError("mortality must be positive and finite")
    if not (np.isfinite(reference_females) and reference_females > 0.0):
        raise ValueError("reference_females must be positive and finite")
    if not np.isfinite(growth):
        raise ValueError("growth must be finite")
    if not (np.isfinite(within_cohort_factor) and within_cohort_factor >= 0.0):
        raise ValueError("within_cohort_factor must be finite and non-negative")
    if isinstance(breeding_period, bool) or not isinstance(breeding_period, (int, np.integer)) or breeding_period < 1:
        raise ValueError("breeding_period must be an integer of at least 1")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 0 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 0 to 400")
    ages = np.arange(max_age + 1, dtype=float)
    birth_1 = (float(year_1) - ages)[:, None]
    birth_2 = (float(year_2) - ages)[None, :]
    gap = np.abs(birth_2 - birth_1)
    younger = np.maximum(birth_1, birth_2)                 # the mother is drawn among the females breeding in the later year
    breeders = reference_females * np.exp(growth * (younger - float(reference_year))) / float(breeding_period)
    probabilities = np.exp(-mortality * gap) / breeders    # the earlier mother survives the gap, then is drawn
    probabilities = np.where(gap == 0.0, within_cohort_factor * probabilities, probabilities)
    on_cycle = np.remainder(gap, float(breeding_period)) == 0.0     # a female breeds only in her own years
    return np.where(on_cycle, probabilities, 0.0)

import numpy as np


def _cpp_age_given_class(cls: "np.ndarray", calibration: "np.ndarray", demography: "np.ndarray", max_age: int) -> "np.ndarray":
    """Conditional distribution of age given the class's survey gear and score bin."""
    year, lower, upper, min_age = cls
    prior = sampled_age_distribution(float(demography[0]), float(demography[4]), int(demography[5]),
                                             float(demography[2]), int(min_age), max_age)
    bins = score_bin_probabilities(float(lower), float(upper), float(calibration[0]), float(calibration[1]),
                                           float(calibration[2]), max_age)
    weights = prior * bins
    total = weights.sum()
    if not total > 0.0:
        raise ValueError("the class's bin has probability zero at every retained age")
    return weights / total


def class_pair_mhsp_probability(class_1: "np.ndarray", class_2: "np.ndarray", calibration: "np.ndarray", demography: "np.ndarray", reference_year: int, breeding_period: int, max_age: int) -> float:
    c1 = np.asarray(class_1, dtype=float)
    c2 = np.asarray(class_2, dtype=float)
    cal = np.asarray(calibration, dtype=float)
    dem = np.asarray(demography, dtype=float)
    if c1.shape != (4,) or c2.shape != (4,) or cal.shape != (3,) or dem.shape != (6,):
        raise ValueError("class_1 and class_2 need shape (4,), calibration (3,) and demography (6,)")
    if isinstance(reference_year, bool) or not isinstance(reference_year, (int, np.integer)):
        raise ValueError("reference_year must be an integer")
    if isinstance(breeding_period, bool) or not isinstance(breeding_period, (int, np.integer)) or breeding_period < 1:
        raise ValueError("breeding_period must be an integer of at least 1")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 1 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 1 to 400")
    for c in (c1, c2):
        if not np.all(np.isfinite(c)) or c[0] != np.round(c[0]) or c[3] != np.round(c[3]):
            raise ValueError("a class needs a finite whole year and minimum age")
        if not (0.0 <= c[1] < c[2]) or not (0 <= c[3] <= max_age):
            raise ValueError("a class needs 0 <= lower < upper and 0 <= min_age <= max_age")
    if not (np.all(np.isfinite(cal)) and np.all(cal > 0.0)):
        raise ValueError("alpha, beta and dispersion must be positive and finite")
    if not (np.all(np.isfinite(dem)) and dem[0] > 0.0 and dem[1] > 0.0 and dem[0] + dem[2] > 0.0
            and dem[3] >= 0.0 and dem[4] > 0.0):
        raise ValueError("demography must be finite with mortality > 0, reference_females > 0, "
                         "mortality + growth > 0, within_cohort_factor >= 0 and juvenile_mortality > 0")
    if dem[5] != np.round(dem[5]) or not 0 <= dem[5] <= max_age:
        raise ValueError("maturity_age must be a whole number from 0 to max_age")
    ages_1 = _cpp_age_given_class(c1, cal, dem, max_age)
    ages_2 = _cpp_age_given_class(c2, cal, dem, max_age)
    kin = maternal_half_sibling_matrix(int(c1[0]), int(c2[0]), float(dem[0]), float(dem[1]), float(dem[2]),
                                               int(reference_year), float(dem[3]), int(breeding_period), max_age)
    return float(ages_1 @ kin @ ages_2)

import numpy as np
from scipy.optimize import brentq, minimize, root


def _cf_comparisons(pair: "np.ndarray", size: "np.ndarray") -> float:
    """Number of comparisons of a class pair: within one class, or across two."""
    if np.array_equal(pair[0], pair[1]):
        n = float(size[0])
        return n * (n - 1.0) / 2.0
    return float(size[0]) * float(size[1])


def _cf_probabilities(theta: "np.ndarray", classes: "np.ndarray", newborn_estimates: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, max_age: int) -> "np.ndarray":
    """The model probability of every class pair at one parameter vector.

    Algebraically this is the fifth step for each pair; it is written out here so that each distinct class and each
    distinct pair of survey years is built once rather than once per pair, which the search calls for thousands of
    times. The two agree to rounding, which `trap_harness/h7_equivalents_and_edges.py` checks.
    """
    beta, mortality, females, cohort = (float(v) for v in theta)
    alpha, dispersion = float(newborn_estimates[0]), float(newborn_estimates[1])
    ages, kernels = {}, {}
    out = np.empty(classes.shape[0])
    for k in range(classes.shape[0]):
        keys = []
        for side in (0, 1):
            cls = classes[k, side]
            key = (float(cls[0]), float(cls[1]), float(cls[2]), float(cls[3]))
            if key not in ages:
                prior = sampled_age_distribution(mortality, float(juvenile_mortality), int(maturity_age),
                                                         float(growth), int(cls[3]), int(max_age))
                bins = score_bin_probabilities(float(cls[1]), float(cls[2]), alpha, beta, dispersion,
                                                       int(max_age))
                weights = prior * bins
                total = weights.sum()
                if not total > 0.0:
                    raise ValueError("a class's bin has probability zero at every retained age")
                ages[key] = weights / total
            keys.append(key)
        years = (int(classes[k, 0, 0]), int(classes[k, 1, 0]))
        if years not in kernels:
            kernels[years] = maternal_half_sibling_matrix(years[0], years[1], mortality, females,
                                                                  float(growth), int(reference_year), cohort,
                                                                  int(breeding_period), int(max_age))
        out[k] = float(ages[keys[0]] @ kernels[years] @ ages[keys[1]])
    return out


def _cf_negative_log_likelihood(theta: "np.ndarray", classes: "np.ndarray", trials: "np.ndarray", counts: "np.ndarray", newborn_estimates: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, max_age: int) -> tuple:
    """Minus the maximised log likelihood at one (beta, mortality, within-cohort factor), with the abundance profiled."""
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0):
        return 1e18, float("nan")
    try:
        shape_rates = _cf_probabilities(np.array([theta[0], theta[1], 1.0, theta[2]]), classes, newborn_estimates,
                                        growth, juvenile_mortality, maturity_age, reference_year,
                                        breeding_period, max_age)
    except ValueError:
        return 1e18, float("nan")
    if not np.all(np.isfinite(shape_rates)) or np.any(shape_rates <= 0.0):
        return 1e18, float("nan")
    females = _cf_profile_abundance(shape_rates, trials, counts)
    p = shape_rates / females
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        return 1e18, float("nan")
    return float(-np.sum(counts * np.log(p) + (trials - counts) * np.log1p(-p))), females


def _cf_profile_abundance(shape_rates: "np.ndarray", trials: "np.ndarray", counts: "np.ndarray") -> float:
    """The abundance that maximises the likelihood at fixed slope, mortality and within-cohort factor.

    Every class-pair probability is inversely proportional to the abundance, so with q the probability at
    reference_females = 1 the likelihood depends on lambda = 1 / reference_females alone, and its derivative
    sum(x / lambda - (n - x) q / (1 - q lambda)) falls strictly as lambda grows: one root, found by bisection.
    """
    top = 1.0 / float(np.max(shape_rates))                       # above this a probability would reach one

    def _slope(lam):
        p = shape_rates * lam
        return float(np.sum(counts / lam - (trials - counts) * shape_rates / (1.0 - p)))

    low, high = top * 1e-12, top * (1.0 - 1e-12)
    if _slope(low) < 0.0:
        return 1.0 / low
    if _slope(high) > 0.0:
        return 1.0 / high
    return 1.0 / brentq(_slope, low, high, xtol=1e-16, rtol=8.9e-16, maxiter=200)


def _cf_fit_from(start: "np.ndarray", floor: float, criterion) -> tuple:
    """Maximise over slope, mortality and within-cohort factor, the abundance profiled out at every step.

    The mortality is bounded below, so the maximum can sit on that bound; a search parameterised by
    floor + exp(u) only creeps towards it. Both the interior search and the search along the bound are run.
    """
    def _packed(u):
        return criterion(np.array([np.exp(u[0]), floor + np.exp(u[1]), np.exp(u[2])]))

    def _on_bound(v):
        return criterion(np.array([np.exp(v[0]), floor, np.exp(v[1])]))

    u0 = np.array([np.log(start[0]), np.log(max(start[1] - floor, 1e-8)), np.log(start[2])])
    best = minimize(_packed, u0, method="Nelder-Mead",
                    options=dict(xatol=1e-10, fatol=1e-12, maxiter=8000, maxfev=8000))
    best = minimize(_packed, best.x, method="Nelder-Mead",
                    options=dict(xatol=1e-12, fatol=1e-13, maxiter=8000, maxfev=8000))
    value = float(best.fun)
    theta = np.array([np.exp(best.x[0]), floor + np.exp(best.x[1]), np.exp(best.x[2])])
    edge = minimize(_on_bound, np.array([np.log(start[0]), np.log(start[2])]), method="Nelder-Mead",
                    options=dict(xatol=1e-12, fatol=1e-13, maxiter=8000, maxfev=8000))
    edge = minimize(_on_bound, edge.x, method="Nelder-Mead",
                    options=dict(xatol=1e-13, fatol=1e-14, maxiter=8000, maxfev=8000))
    if float(edge.fun) < value - 1e-12:
        value = float(edge.fun)
        theta = np.array([np.exp(edge.x[0]), floor, np.exp(edge.x[1])])
    return value, theta


def contrast_fit(classes: "np.ndarray", sizes: "np.ndarray", counts: "np.ndarray", newborn_estimates: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, max_age: int) -> "np.ndarray":
    cls = np.asarray(classes, dtype=float)
    siz = np.asarray(sizes, dtype=float)
    cnt = np.asarray(counts, dtype=float)
    nb = np.asarray(newborn_estimates, dtype=float)
    if cls.ndim != 3 or cls.shape[0] < 1 or cls.shape[1:] != (2, 4):
        raise ValueError("classes needs shape (K, 2, 4) with K >= 1")
    if siz.shape != (cls.shape[0], 2) or cnt.shape != (cls.shape[0],) or nb.shape != (2,):
        raise ValueError("sizes needs shape (K, 2), counts shape (K,) and newborn_estimates shape (2,)")
    if isinstance(reference_year, bool) or not isinstance(reference_year, (int, np.integer)):
        raise ValueError("reference_year must be an integer")
    if isinstance(breeding_period, bool) or not isinstance(breeding_period, (int, np.integer)) or breeding_period < 1:
        raise ValueError("breeding_period must be an integer of at least 1")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 1 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 1 to 400")
    if not (np.all(np.isfinite(nb)) and np.all(nb > 0.0)):
        raise ValueError("alpha and dispersion must be positive and finite")
    if not np.isfinite(growth):
        raise ValueError("growth must be finite")
    if not (np.isfinite(juvenile_mortality) and juvenile_mortality > 0.0):
        raise ValueError("juvenile_mortality must be positive and finite")
    if isinstance(maturity_age, bool) or not isinstance(maturity_age, (int, np.integer)) \
            or not 0 <= maturity_age <= max_age:
        raise ValueError("maturity_age must be an integer from 0 to max_age")
    if not (np.isfinite(min_mortality) and min_mortality > 0.0):
        raise ValueError("min_mortality must be positive and finite")
    for k in range(cls.shape[0]):
        for c in (cls[k, 0], cls[k, 1]):
            if not np.all(np.isfinite(c)) or c[0] != np.round(c[0]) or c[3] != np.round(c[3]):
                raise ValueError("a class needs a finite whole year and minimum age")
            if not (0.0 <= c[1] < c[2]) or not (0 <= c[3] <= max_age):
                raise ValueError("a class needs 0 <= lower < upper and 0 <= min_age <= max_age")
    if not (np.all(np.isfinite(siz)) and np.all(siz == np.round(siz)) and np.all(siz >= 1.0)):
        raise ValueError("the class sizes must be whole numbers of at least 1")
    if not (np.all(np.isfinite(cnt)) and np.all(cnt == np.round(cnt))):
        raise ValueError("the counts must be whole numbers")
    trials = np.array([_cf_comparisons(cls[k], siz[k]) for k in range(cls.shape[0])])
    if not np.all(trials >= 1.0):
        raise ValueError("every class pair needs at least one comparison")
    if not (np.all(cnt > 0.0) and np.all(cnt < trials)):
        raise ValueError("every count must be strictly between 0 and the pair's number of comparisons")

    def _criterion(theta):
        return _cf_negative_log_likelihood(theta, cls, trials, cnt, nb, float(growth), float(juvenile_mortality),
                                           int(maturity_age), int(reference_year), int(breeding_period),
                                           int(max_age))[0]

    floor = float(min_mortality)
    grid = [np.array([b, m, w])                              # a coarse sweep of the three free parameters
            for b in (0.35, 0.7, 1.2, 2.0)
            for m in (floor + 1e-4, floor + 0.03, 0.14, 0.3, 0.6)
            for w in (0.35, 1.2, 3.0)]
    scored = sorted(((_criterion(g), g) for g in grid), key=lambda r: r[0])
    kept, best = [], None
    for value, start in scored:                              # refine the best starts, spread apart in log space
        if len(kept) == 6:
            break
        key = np.log(np.array([start[0], start[1] - floor + 1e-8, start[2]]))
        if value < 1e17 and all(np.max(np.abs(key - seen)) > 0.6 for seen in kept):
            kept.append(key)
            got, theta = _cf_fit_from(start, floor, _criterion)
            if np.isfinite(got) and got < 1e17 and (best is None or got < best[0] - 1e-9):
                best = (got, theta)
    if best is None:
        raise ValueError("no admissible parameters give every class pair a probability strictly between 0 and 1")
    beta, mortality, cohort = best[1]
    females = _cf_negative_log_likelihood(best[1], cls, trials, cnt, nb, float(growth), float(juvenile_mortality),
                                          int(maturity_age), int(reference_year), int(breeding_period),
                                          int(max_age))[1]
    return np.array([beta, mortality, females, cohort])

import numpy as np


def kin_rate_per_million(newborn_scores: "np.ndarray", contrast_classes: "np.ndarray", contrast_sizes: "np.ndarray", contrast_counts: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, target_classes: "np.ndarray", max_age: int) -> float:
    targets = np.asarray(target_classes, dtype=float)
    if targets.shape != (2, 4):
        raise ValueError("target_classes must have shape (2, 4)")
    alpha, dispersion = newborn_calibration(newborn_scores)
    beta, mortality, females, factor = contrast_fit(contrast_classes, contrast_sizes, contrast_counts,
                                                            np.array([alpha, dispersion]), growth, juvenile_mortality,
                                                            maturity_age, reference_year, breeding_period,
                                                            min_mortality, max_age)
    probability = class_pair_mhsp_probability(targets[0], targets[1], np.array([alpha, beta, dispersion]),
                                                      np.array([mortality, females, growth, factor,
                                                                juvenile_mortality, maturity_age]),
                                                      reference_year, breeding_period, max_age)
    return 1.0e6 * probability
SCICODE_GOLD_EOF
