"""
Expected numbers of sampled animals by sex, true age and estimated age in each sampling year.

Once ages come from a clock, a design has to plan with the estimated ages the genotyped

samples will carry, not the true ones. The expected numbers of samples by true age and

the clock's error distribution together give the expected numbers by true and

estimated age jointly. Summed over true ages they are the planned sample sizes by

estimated age; read the other way, they say how likely each true age is for an animal

that carries a given estimate, which is how aging error enters kinship probabilities.

Returns
-------
np.ndarray of shape (Y, 2, A, A), the expected number of sampled animals by year, sex, true age and estimated age
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimated_age_numbers(samples: "np.ndarray", error_sd: float) -> "np.ndarray":
    '''Expected numbers of sampled animals by sex, true age and estimated age in each sampling year.

    samples[i, s, a - 1] is the expected number of animals of sex s (0
    female, 1 male) and true age a sampled in year i, as in the third step,
    for ages 1 to A. Each animal's estimated age follows the error model of
    the fourth step with standard deviation error_sd and max_age = A:
    the true age plus a normal error, rounded to the nearest whole year and
    conditioned on lying between 1 and A.

    Parameters
    ----------
    samples : np.ndarray
        Shape (Y, 2, A), Y >= 1, 2 <= A <= 100; finite, non-negative
        expected numbers.
    error_sd : float
        Standard deviation of the error, a finite number from 0.05 to 50.

    Returns
    -------
    numbers : np.ndarray
        Shape (Y, 2, A, A). Entry [i, s, a - 1, e - 1] is the expected number
        of animals of sex s sampled in year i with true age a and estimated
        age e.

    Raises
    ------
    ValueError
        If samples is not a finite, non-negative array of shape (Y, 2, A)
        with Y >= 1 and 2 <= A <= 100, or if error_sd is not a finite number
        from 0.05 to 50.
    '''
    return numbers  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_estimated_age_numbers(samples: "np.ndarray", error_sd: float) -> "np.ndarray":
    m = np.asarray(samples, dtype=float)
    if m.ndim != 3 or m.shape[0] < 1 or m.shape[1] != 2 or not 2 <= m.shape[2] <= 100 \
            or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("samples must be a finite non-negative array of shape (Y, 2, A) with 2 <= A <= 100")
    if isinstance(error_sd, bool) or not isinstance(error_sd, (int, float, np.integer, np.floating)) \
            or not np.isfinite(error_sd) or not 0.05 <= float(error_sd) <= 50.0:
        raise ValueError("error_sd must be a finite number from 0.05 to 50")
    P = _ean_error_matrix(float(error_sd), m.shape[2])
    return m[:, :, :, None] * P[None, None, :, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    comp = ("ages = np.arange(1, 38)\n"
            "lam = np.exp(-0.012)\n"
            "rel = np.where(ages < 6, lam ** (-ages) * 0.85 ** (ages - 1), lam ** (-ages) * 0.85 ** 5 * 0.955 ** (ages - 6))\n"
            "rel_m = np.where(ages <= 5, rel, 0.0)\n"
            "share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())\n")
    return [
        # --- normal: two sampling years of the shipped composition, the fitted clock error ---
        {
            "setup": "import numpy as np\n" + comp + "samples = np.array([800.0, 1000.0])[:, None, None] * share[None, :, :]\n",
            "call": "estimated_age_numbers(samples, 1.6625615194893861)",
            "gold_call": "_oracle_estimated_age_numbers(samples, 1.6625615194893861)",
        },
        # --- boundary: two ages only, so every estimate is confined to the two ends ---
        {
            "setup": "import numpy as np\nsamples = np.array([[[3.0, 1.0], [2.0, 0.0]]])\n",
            "call": "estimated_age_numbers(samples, 0.8)",
            "gold_call": "_oracle_estimated_age_numbers(samples, 0.8)",
        },
        # --- edge: a very precise clock, where the off-diagonal probabilities sit far in the normal tail ---
        {
            "setup": "import numpy as np\nsamples = np.linspace(1.0, 60.0, 3 * 2 * 20).reshape(3, 2, 20)\n",
            "call": "estimated_age_numbers(samples, 0.12)",
            "gold_call": "_oracle_estimated_age_numbers(samples, 0.12)",
        },
    ]
