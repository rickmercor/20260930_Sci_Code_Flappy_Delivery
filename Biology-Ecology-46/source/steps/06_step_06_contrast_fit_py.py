"""
Calibration slope, mortality, abundance and within-cohort factor most likely to have produced the observed half-sibling counts.

A close-kin survey reports, for each class pair it summarises, how many comparisons were made and how many of

them turned out to be maternal half-sibling pairs. Every comparison is one Bernoulli trial whose success

probability the model supplies, so a set of class pairs is a set of binomial counts and the calibration slope

and the demography can be read off the counts by maximum likelihood. With more class pairs than parameters no

choice of parameters reproduces every observed rate exactly, and the estimate is then the point that makes the

counts as likely as possible rather than the solution of a system of equations: how the misfit of one class

pair is traded against another is decided by the likelihood, not by the modeller.

Returns
-------
np.ndarray, shape (4,): beta, mortality, reference_females and within_cohort_factor, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contrast_fit(classes: "np.ndarray", sizes: "np.ndarray", counts: "np.ndarray", newborn_estimates: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, max_age: int) -> "np.ndarray":
    '''Calibration slope, mortality, abundance and within-cohort factor most likely to have produced the observed half-sibling counts.

    classes[k] holds the two classes of class pair k, each given as
    [year, lower, upper, min_age]; sizes[k] holds the numbers of animals in
    those two classes; and counts[k] is the number of maternal half-sibling
    pairs found among the pair's comparisons. When the two classes of a pair
    are identical (all four entries equal), the comparisons are the unordered
    pairs of distinct animals within that class; otherwise they are all pairs
    with one animal from each class. Each comparison is an independent trial
    whose probability of being a maternal half-sibling pair is the one the
    fifth step defines, at alpha and dispersion fixed to newborn_estimates
    and at the candidate beta, mortality, reference_females and
    within_cohort_factor, with the same growth, juvenile_mortality,
    maturity_age, reference_year, breeding_period and max_age throughout. The
    mortality the fit returns is the rate of mature animals: the young die at
    juvenile_mortality until they reach maturity_age. Return the
    (beta, mortality, reference_females, within_cohort_factor) that maximise
    the probability of the observed counts over beta > 0,
    mortality >= min_mortality, reference_females > 0 and
    within_cohort_factor > 0, which is unique, each entry to a relative
    accuracy of 1e-6. Parameters under which some class pair has no
    admissible probability are not candidates and do not stop the search.

    Parameters
    ----------
    classes : np.ndarray
        Shape (K, 2, 4) with K >= 1: the two classes of each class pair.
    sizes : np.ndarray
        Shape (K, 2): the number of animals in each class of each pair, whole
        numbers of at least 2 within a pair of identical classes and at least
        1 otherwise.
    counts : np.ndarray
        Shape (K,): the half-sibling pairs found, whole numbers with
        0 < counts[k] < the number of comparisons of pair k.
    newborn_estimates : np.ndarray
        Shape (2,): alpha and dispersion, each positive and finite.
    growth : float
        Annual growth rate of births and of adult females, finite.
    juvenile_mortality : float
        Annual mortality rate before maturity, positive and finite.
    maturity_age : int
        Age at which an animal matures, a whole number from 0 to max_age.
    reference_year : int
        The year in which the adult females number reference_females.
    breeding_period : int
        The number of years between one breeding and the next, at least 1.
    min_mortality : float
        The lowest annual mortality allowed, positive and finite.
    max_age : int
        The oldest age carried, a whole number from 1 to 400.

    Returns
    -------
    np.ndarray
        Shape (4,): beta, mortality, reference_females and
        within_cohort_factor, in that order.

    Raises
    ------
    ValueError
        If classes does not have shape (K, 2, 4) with K >= 1, sizes shape
        (K, 2), counts shape (K,) or newborn_estimates shape (2,); if a class
        year or min_age is not a whole number, a bin is not
        0 <= lower < upper, or min_age is not in 0 <= min_age <= max_age; if
        a size or count is not a whole number, a size is too small, or a
        count is not strictly between 0 and the pair's number of
        comparisons; if an entry of newborn_estimates is not positive and
        finite; if growth is not finite, juvenile_mortality is not positive
        and finite, maturity_age is not a whole number from 0 to max_age, or
        min_mortality is not positive and finite; if reference_year is not an integer, breeding_period is not
        an integer of at least 1, or max_age is not an integer from 1 to
        400; or if no admissible parameters give every class pair a
        probability strictly between 0 and 1.
    '''
    return x  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
                prior = _oracle_sampled_age_distribution(mortality, float(juvenile_mortality), int(maturity_age),
                                                         float(growth), int(cls[3]), int(max_age))
                bins = _oracle_score_bin_probabilities(float(cls[1]), float(cls[2]), alpha, beta, dispersion,
                                                       int(max_age))
                weights = prior * bins
                total = weights.sum()
                if not total > 0.0:
                    raise ValueError("a class's bin has probability zero at every retained age")
                ages[key] = weights / total
            keys.append(key)
        years = (int(classes[k, 0, 0]), int(classes[k, 1, 0]))
        if years not in kernels:
            kernels[years] = _oracle_maternal_half_sibling_matrix(years[0], years[1], mortality, females,
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


def _oracle_contrast_fit(classes: "np.ndarray", sizes: "np.ndarray", counts: "np.ndarray", newborn_estimates: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, max_age: int) -> "np.ndarray":
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the study's eight class pairs, a three-year breeding cycle ---
        {
            "setup": "import numpy as np\n"
                     "A16 = [2016.0, 3.0, 5.0, 0.0]\nA20 = [2020.0, 3.0, 5.0, 0.0]\n"
                     "A24 = [2024.0, 3.0, 5.0, 4.0]\nB16 = [2016.0, 9.0, 11.0, 0.0]\n"
                     "B20 = [2020.0, 9.0, 11.0, 0.0]\n"
                     "cls = np.array([[A16, A16], [A16, A20], [A16, A24], [A16, B16],\n"
                     "                [A20, A20], [A20, A24], [B16, B16], [B16, B20]])\n"
                     "siz = np.array([[520, 520], [520, 480], [520, 300], [520, 150],\n"
                     "                [480, 480], [480, 300], [150, 150], [150, 140]])\n"
                     "cnt = np.array([108, 71, 21, 14, 84, 39, 12, 9])\n"
                     "nb = np.array([1.4883333333333333, 0.009032872781593575])\n",
            "call": "contrast_fit(cls, siz, cnt, nb, 0.05, 0.3, 8, 2024, 3, 0.05, 120)",
            "gold_call": "_oracle_contrast_fit(cls, siz, cnt, nb, 0.05, 0.3, 8, 2024, 3, 0.05, 120)",
            "tol": 1e-6,
        },
        # --- boundary: six class pairs over two surveys, a two-year cycle and a growing population ---
        {
            "setup": "import numpy as np\n"
                     "A12 = [2012.0, 3.0, 5.0, 0.0]\n"
                     "A16 = [2016.0, 3.0, 5.0, 0.0]\n"
                     "B12 = [2012.0, 9.0, 11.0, 0.0]\n"
                     "B16 = [2016.0, 9.0, 11.0, 2.0]\n"
                     "cls = np.array([[A12, A12], [A12, A16], [A12, B12], [A12, B16], [A16, A16], [B12, B12]])\n"
                     "siz = np.array([[450, 450], [450, 380], [450, 160], [450, 120], [380, 380], [160, 160]])\n"
                     "cnt = np.array([63, 49, 11, 19, 43, 9])\n"
                     "nb = np.array([1.5075, 0.0025970907262187976])\n",
            "call": "contrast_fit(cls, siz, cnt, nb, 0.03, 0.42, 5, 2016, 2, 0.05, 100)",
            "gold_call": "_oracle_contrast_fit(cls, siz, cnt, nb, 0.03, 0.42, 5, 2016, 2, 0.05, 100)",
            "tol": 1e-6,
        },
        # --- edge: seven class pairs, a four-year cycle, a shrinking population and a sharp score ---
        {
            "setup": "import numpy as np\n"
                     "A18 = [2018.0, 3.0, 5.0, 0.0]\n"
                     "A22 = [2022.0, 3.0, 5.0, 3.0]\n"
                     "B18 = [2018.0, 7.0, 9.0, 0.0]\n"
                     "C18 = [2018.0, 11.0, 13.0, 0.0]\n"
                     "cls = np.array([[A18, A18], [A18, A22], [A18, B18], [A18, C18], [A22, A22], [B18, B18],\n"
                     "                [B18, A22]])\n"
                     "siz = np.array([[520, 520], [520, 340], [520, 300], [520, 220], [340, 340], [300, 300],\n"
                     "                [300, 340]])\n"
                     "cnt = np.array([217, 48, 44, 9, 152, 62, 19])\n"
                     "nb = np.array([1.35125, 0.0014762600088645752])\n",
            "call": "contrast_fit(cls, siz, cnt, nb, -0.015, 0.28, 9, 2022, 4, 0.05, 110)",
            "gold_call": "_oracle_contrast_fit(cls, siz, cnt, nb, -0.015, 0.28, 9, 2022, 4, 0.05, 110)",
            "tol": 1e-6,
        },
    ]
