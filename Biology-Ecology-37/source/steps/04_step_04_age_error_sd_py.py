"""
Maximum-likelihood standard deviation of the error of epigenetic age estimates, from reference animals of known age.

Close-kin comparisons are classified by the ages of the sampled animals, and in a wild

marine mammal those ages come from an epigenetic clock. A clock is fitted to animals of

known age, and its error is summarised by one standard deviation that is

then treated as known in every kinship calculation. Clock estimates are reported as

whole years within the plausible range of ages, so each reference animal contributes

the probability of its reported whole-year estimate, not a density value, and animals

near either end of the range have their estimates confined to it.

Returns
-------
float, the maximum-likelihood standard deviation of the clock's age error
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def age_error_sd(true_ages: "np.ndarray", estimated_ages: "np.ndarray", max_age: int) -> float:
    '''Maximum-likelihood standard deviation of the error of epigenetic age estimates, from reference animals of known age.

    An animal of true age a receives the estimated age obtained by adding to
    a a normal error with mean 0 and standard deviation sigma, rounding the
    result to the nearest whole year, and conditioning on the rounded value
    lying between 1 and max_age inclusive. The reference animals'
    estimates are independent given their true ages. Return the sigma
    between 0.05 and 50 that maximises the likelihood of the estimated ages
    given the true ages, to a relative accuracy of 1e-10.

    Parameters
    ----------
    true_ages : np.ndarray
        Shape (K,), K >= 1; true ages of the reference animals, integers
        from 1 to max_age.
    estimated_ages : np.ndarray
        Shape (K,); their estimated ages, integers from 1 to max_age, with at
        least one different from its true age.
    max_age : int
        Oldest possible estimated age, an integer from 2 to 100.

    Returns
    -------
    sigma : float
        The maximum-likelihood standard deviation, as a native Python float.

    Raises
    ------
    ValueError
        If true_ages and estimated_ages are not one-dimensional integer
        arrays of the same non-zero length with values from 1 to max_age, if
        every estimated age equals its true age, if max_age is not an integer
        from 2 to 100, or if the likelihood over sigma from 0.05 to 50 is
        largest at sigma = 0.05 or sigma = 50.
    '''
    return sigma  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_age_error_sd(true_ages: "np.ndarray", estimated_ages: "np.ndarray", max_age: int) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: thirty known-age females, young animals near the lower end of the range ---
        {
            "setup": "import numpy as np\n"
                     "true = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 34, 36])\n"
                     "est = np.array([2, 2, 4, 1, 3, 4, 2, 4, 5, 6, 4, 3, 3, 5, 4, 9, 6, 8, 9, 14, 15, 17, 19, 18, 20, 28, 28, 31, 32, 36])\n",
            "call": "age_error_sd(true, est, 37)",
            "gold_call": "_oracle_age_error_sd(true, est, 37)",
        },
        # --- boundary: a narrow age range, every animal within a year of an end of it ---
        {
            "setup": "import numpy as np\ntrue = np.array([1, 1, 2, 2, 9, 9, 10, 10])\nest = np.array([1, 2, 1, 3, 10, 8, 10, 9])\n",
            "call": "age_error_sd(true, est, 10)",
            "gold_call": "_oracle_age_error_sd(true, est, 10)",
        },
        # --- edge: a precise clock, a single one-year error among many exact estimates ---
        {
            "setup": "import numpy as np\ntrue = np.arange(3, 23)\nest = true.copy()\nest[7] = 11\n",
            "call": "age_error_sd(true, est, 40)",
            "gold_call": "_oracle_age_error_sd(true, est, 40)",
        },
    ]
