"""
Probability that a female sampled in one year is the same animal as a female sampled in a later year, by the first sample's estimated age and the second sample's stage.

Genotypes identify an animal, so two samples taken in different years can be the same

female: individual mark-recapture expressed as one more kinship type. In a

stage-structured model adult ages need not be tracked, so the later sample is

described only by its stage, which field workers can tell reliably, while the first

sample's clock estimate leaves its true age uncertain. Whether the two can be the same

animal then depends on whether the first would have reached the later stage in the

intervening years, whether she survived them, and how many females of that stage

there were to choose from.

Returns
-------
np.ndarray of shape (Y, A, Y, 2), the probability that a female of a given year and estimated age is the same animal as a later sample of a given stage
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_recapture_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray",
                                 numbers: "np.ndarray") -> "np.ndarray":
    '''Probability that a female sampled in one year is the same animal as a female sampled in a later year, by the first sample's estimated age and the second sample's stage.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step. numbers[i, s, a - 1, e - 1] is as returned by the
    fifth step for animals sampled in sample_years (ages 1 to A). Entry
    [i, e - 1, m, d] of the result is the probability that a female sampled
    without harm in sample_years[i] with estimated age e is the same animal
    as a female sampled in sample_years[m] in stage d (d = 0 juvenile, ages 1
    to 5; d = 1 adult, 6 or older), whose stage is known without error.

    Given its sampling year and estimated age, the first female's true age a
    has probability proportional to numbers[i, 0, a - 1, e - 1], and the
    entry is the average over a of the probability for that true age; it is
    zero when the class has no expected animals and whenever
    sample_years[m] <= sample_years[i]. For true age a, the first female must
    survive the years between the two samples (the juvenile rate for each
    year spent at ages 1 to 5, the adult rate after) and be in stage d at
    the second sample, and is then equally likely to be any female of that
    stage in that year (stage abundances from the second step).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers.

    Returns
    -------
    probabilities : np.ndarray
        Shape (Y, A, Y, 2), as described above; a probability too small for
        double precision may be returned as zero.

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain of the second
        step, if ref_year is not an integer from 1900 to 2200, if sample_years
        is not a non-empty, strictly increasing one-dimensional array of
        integers from 1900 to 2200, or if numbers is not a finite,
        non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100.
    '''
    return probabilities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_self_recapture_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray",
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
    stage_n = _oracle_stage_abundances(th, ref_year, ys)                  # (2, Y): adult, juvenile; validates theta
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    theta = ("theta = np.array([np.log(48000.0), -0.012, np.log(0.955 / 0.045), np.log(0.85 / 0.15), "
             "np.log(0.16 / 0.84), np.log(0.38 / 0.62)])\n")
    numbers = ("ages = np.arange(1, 38)\n"
               "lam = np.exp(-0.012)\n"
               "rel = np.where(ages < 6, lam ** (-ages) * 0.85 ** (ages - 1), lam ** (-ages) * 0.85 ** 5 * 0.955 ** (ages - 6))\n"
               "rel_m = np.where(ages <= 5, rel, 0.0)\n"
               "share = np.array([rel, rel_m]) / (rel.sum() + rel_m.sum())\n"
               "a = np.arange(1, 38)[:, None]\ne = np.arange(1, 38)[None, :]\n"
               "from scipy.stats import norm\n"
               "P = norm.cdf((e + 0.5 - a) / SD) - norm.cdf((e - 0.5 - a) / SD)\nP = P / P.sum(axis=1, keepdims=True)\n"
               "samples = np.array(SIZES)[:, None, None] * share[None, :, :]\n"
               "numbers = samples[:, :, :, None] * P[None, None, :, :]\n")
    return [
        # --- normal: the shipped two-period design with the fitted clock ---
        {
            "setup": "import numpy as np\n" + theta + numbers.replace("SD", "1.6625615194893861").replace(
                "SIZES", "[800.0] * 5 + [1000.0] * 4") + "years = np.array([2014, 2015, 2016, 2017, 2018, 2024, 2025, 2026, 2027])\n",
            "call": "self_recapture_probabilities(theta, 2016, years, numbers)",
            "gold_call": "_oracle_self_recapture_probabilities(theta, 2016, years, numbers)",
        },
        # --- boundary: consecutive years and a precise clock, so estimated juveniles cross into adulthood ---
        {
            "setup": "import numpy as np\n" + theta + numbers.replace("SD", "0.4").replace("SIZES", "[300.0, 300.0]")
                     + "years = np.array([2020, 2021])\n",
            "call": "self_recapture_probabilities(theta, 2016, years, numbers)",
            "gold_call": "_oracle_self_recapture_probabilities(theta, 2016, years, numbers)",
        },
        # --- edge: a small, fast-growing population and an imprecise clock over a ten-year gap ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(3000.0), 0.05, np.log(0.93 / 0.07), np.log(0.7 / 0.3), "
                     "np.log(0.25 / 0.75), np.log(0.5 / 0.5)])\n"
                     "a = np.arange(1, 16)[:, None]\ne = np.arange(1, 16)[None, :]\n"
                     "from scipy.stats import norm\n"
                     "P = norm.cdf((e + 0.5 - a) / 3.0) - norm.cdf((e - 0.5 - a) / 3.0)\nP = P / P.sum(axis=1, keepdims=True)\n"
                     "samples = np.linspace(5.0, 40.0, 3 * 2 * 15).reshape(3, 2, 15)\n"
                     "numbers = samples[:, :, :, None] * P[None, None, :, :]\nyears = np.array([2003, 2008, 2013])\n",
            "call": "self_recapture_probabilities(theta, 2012, years, numbers)",
            "gold_call": "_oracle_self_recapture_probabilities(theta, 2012, years, numbers)",
        },
    ]
